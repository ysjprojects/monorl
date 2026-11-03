"""Weight-sync timing: how long ``WeightSync.sync(trainer, engines)`` takes per strategy.

Two sources of weights:

* a real runtime (``monorl.distributed.build_runtime``: the configured trainer + engines + strategy);
* :class:`SyntheticWeightSource` / :class:`SyntheticEngine` with ``--fake-params N`` random
  tensors, so the sync path can be sized without loading a model.

Strategies come from the ``weight_sync`` registry namespace; the ones that cannot be built or run on
this machine (``cuda_ipc`` without CUDA, ``broadcast`` outside a multi-process group) are reported
as skipped with the reason instead of being faked. torch is imported lazily inside the functions
that need it so ``python -m monorl.bench --help`` stays torch-free.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monorl.bench.stats import HostInfo, LatencyStats, RssTracker, fmt_seconds, rate
from monorl.core.errors import MonorlError
from monorl.core.registry import registry

if TYPE_CHECKING:
    import torch

    from monorl.config.schema import RunConfig
    from monorl.core.interfaces import RolloutEngine, WeightSync

SYNTHETIC_CHUNK = 1 << 20
"""Elements per synthetic tensor (4 MiB in fp32): large enough to amortise per-tensor overhead
like a real transformer block, small enough to keep many tensors in the stream."""


# -- synthetic trainer / engine --------------------------------------------------------------


class SyntheticWeightSource:
    """The part of :class:`~monorl.core.interfaces.Trainer` weight sync touches: ``step``,
    ``export_weights`` and ``save_pretrained`` (HF layout: ``config.json`` + ``model.safetensors``)."""

    def __init__(self, num_params: int, *, dtype: str = "float32", device: str = "cpu", chunk: int = SYNTHETIC_CHUNK) -> None:
        import torch

        if num_params < 1:
            raise ValueError("num_params must be >= 1")
        self.step = 0
        self.dtype = getattr(torch, dtype)
        self.device = torch.device(device)
        self.tensors: dict[str, torch.Tensor] = {}
        remaining = int(num_params)
        i = 0
        gen = torch.Generator().manual_seed(0)
        while remaining > 0:
            n = min(chunk, remaining)
            self.tensors[f"synthetic.layers.{i}.weight"] = torch.randn(n, generator=gen).to(self.dtype).to(self.device)
            remaining -= n
            i += 1

    @property
    def num_params(self) -> int:
        return sum(t.numel() for t in self.tensors.values())

    @property
    def num_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        yield from self.tensors.items()

    def save_pretrained(self, path: Path) -> None:
        from safetensors.torch import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"model_type": "monorl-synthetic", "num_params": self.num_params}))
        save_file({k: v.contiguous().cpu() for k, v in self.tensors.items()}, str(path / "model.safetensors"))


class SyntheticEngine:
    """The part of :class:`~monorl.core.interfaces.RolloutEngine` weight sync touches. Holds a
    preallocated copy of every parameter and ``copy_``s incoming weights into it, like a real
    engine updating its parameter storage."""

    def __init__(self, source: SyntheticWeightSource) -> None:
        import torch

        self.params: dict[str, torch.Tensor] = {k: torch.empty_like(v) for k, v in source.tensors.items()}
        self.updates = 0
        self.bytes_received = 0

    def update_weights(self, weights: Any) -> None:
        for name, tensor in weights:
            dst = self.params[name]
            dst.copy_(tensor)
            self.bytes_received += dst.numel() * dst.element_size()
        self.updates += 1

    def load_weights_from_path(self, path: Any) -> None:
        from safetensors.torch import load_file

        self.update_weights(load_file(str(Path(path) / "model.safetensors")).items())

    def update_weights_from_ipc(self, handles: Sequence[Any]) -> None:
        self.update_weights((h.name, h.open()) for h in handles)


# -- results ---------------------------------------------------------------------------------


@dataclass(slots=True)
class SyncResult:
    strategy: str
    repeats: int
    bytes: int
    seconds: LatencyStats
    gb_per_s: float
    metrics: dict[str, float]
    skipped: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SyncBenchmark:
    source: str
    """``runtime`` (configured trainer/engines) or ``synthetic``."""
    num_params: int | None
    engines: int
    results: list[SyncResult]
    fastest: str | None
    recommendations: list[str]
    host: HostInfo
    peak_rss_bytes: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "sync",
            "source": self.source,
            "num_params": self.num_params,
            "engines": self.engines,
            "results": [r.to_dict() for r in self.results],
            "fastest": self.fastest,
            "recommendations": list(self.recommendations),
            "host": self.host.to_dict(),
            "peak_rss_bytes": self.peak_rss_bytes,
            "extra": dict(self.extra),
        }


# -- strategy resolution ---------------------------------------------------------------------


@dataclass(slots=True)
class StrategyCandidate:
    name: str
    sync: WeightSync | None
    reason: str | None = None


def resolve_strategies(config: RunConfig, names: Sequence[str] | None = None, *, sync_dir: str | None = None) -> list[StrategyCandidate]:
    """Build each named (default: every registered) ``weight_sync`` strategy; the ones whose
    factory refuses this machine are returned with ``sync=None`` and the refusal as ``reason``."""
    wanted = list(names) if names else registry.names("weight_sync")
    out: list[StrategyCandidate] = []
    for name in wanted:
        try:
            factory = registry.get("weight_sync", name)
            kw: dict[str, Any] = {}
            if name == "checkpoint":
                kw["sync_dir"] = sync_dir or tempfile.mkdtemp(prefix="monorl-bench-sync-")
            sync = factory(config, **kw)
        except MonorlError as e:
            out.append(StrategyCandidate(name, None, str(e)))
            continue
        except (RuntimeError, ValueError, TypeError) as e:
            out.append(StrategyCandidate(name, None, f"{type(e).__name__}: {e}"))
            continue
        out.append(StrategyCandidate(name, sync))
    return out


# -- measurement -----------------------------------------------------------------------------


def measure_strategy(
    candidate: StrategyCandidate, trainer: Any, engines: Sequence[RolloutEngine], *, repeats: int = 3, rss: RssTracker | None = None
) -> SyncResult:
    if candidate.sync is None:
        return SyncResult(candidate.name, 0, 0, LatencyStats.from_samples([]), 0.0, {}, skipped=True, reason=candidate.reason)
    rss = rss or RssTracker()
    times: list[float] = []
    metrics: dict[str, float] = {}
    total_bytes = 0
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        try:
            metrics = {k: float(v) for k, v in candidate.sync.sync(trainer, engines).items()}
        except MonorlError as e:
            return SyncResult(candidate.name, len(times), 0, LatencyStats.from_samples(times), 0.0, {}, skipped=True, reason=str(e))
        except (RuntimeError, OSError) as e:
            return SyncResult(
                candidate.name, len(times), 0, LatencyStats.from_samples(times), 0.0, {}, skipped=True, reason=f"{type(e).__name__}: {e}"
            )
        times.append(time.perf_counter() - t0)
        rss.sample()
        total_bytes = int(metrics.get("weight_sync/bytes", total_bytes))
    stats = LatencyStats.from_samples(times)
    return SyncResult(
        strategy=candidate.name,
        repeats=len(times),
        bytes=total_bytes,
        seconds=stats,
        gb_per_s=rate(total_bytes / 1e9, stats.p50_s),
        metrics=metrics,
    )


def recommend(results: Sequence[SyncResult], *, engines: int) -> tuple[str | None, list[str]]:
    ran = [r for r in results if not r.skipped and r.repeats > 0]
    if not ran:
        return None, ["no strategy ran on this machine: " + "; ".join(f"{r.strategy}: {r.reason}" for r in results)]
    fastest = min(ran, key=lambda r: r.seconds.p50_s)
    notes = [
        f"fastest: cluster.weight_sync={fastest.strategy} at p50 {fmt_seconds(fastest.seconds.p50_s)} "
        f"({fastest.gb_per_s:.2f} GB/s, {fastest.bytes / 1e9:.3f} GB to {engines} engine(s))"
    ]
    for r in ran:
        if r is fastest or fastest.seconds.p50_s <= 0:
            continue
        notes.append(f"{r.strategy}: {r.seconds.p50_s / fastest.seconds.p50_s:.1f}x slower (p50 {fmt_seconds(r.seconds.p50_s)})")
    for r in results:
        if r.skipped:
            notes.append(f"{r.strategy}: skipped — {r.reason}")
    return fastest.strategy, notes


def run_sync_benchmark(
    trainer: Any,
    engines: Sequence[RolloutEngine],
    candidates: Sequence[StrategyCandidate],
    *,
    repeats: int = 3,
    source: str = "runtime",
    num_params: int | None = None,
    extra: dict[str, Any] | None = None,
) -> SyncBenchmark:
    if not candidates:
        raise ValueError("no weight-sync strategies to benchmark")
    rss = RssTracker()
    results = [measure_strategy(c, trainer, engines, repeats=repeats, rss=rss) for c in candidates]
    fastest, notes = recommend(results, engines=len(engines))
    return SyncBenchmark(
        source=source,
        num_params=num_params,
        engines=len(engines),
        results=results,
        fastest=fastest,
        recommendations=notes,
        host=HostInfo.detect(),
        peak_rss_bytes=rss.peak_bytes,
        extra=dict(extra or {}),
    )


def synthetic_config() -> RunConfig:
    """A valid :class:`RunConfig` for strategy factories when no recipe is given."""
    from monorl.config.schema import RunConfig

    return RunConfig.from_dict(
        {
            "model": {"name_or_path": "monorl/synthetic"},
            "data": {"uri": "synthetic.jsonl"},
            "rewards": [{"type": "exact_match"}],
        }
    )


__all__ = [
    "SYNTHETIC_CHUNK",
    "StrategyCandidate",
    "SyncBenchmark",
    "SyncResult",
    "SyntheticEngine",
    "SyntheticWeightSource",
    "measure_strategy",
    "recommend",
    "resolve_strategies",
    "run_sync_benchmark",
    "synthetic_config",
]
