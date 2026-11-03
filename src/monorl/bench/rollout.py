"""Rollout-engine throughput: ``engine.generate`` at several batch sizes.

Pure measurement; the engine, prompts and sampling params are built by :mod:`monorl.bench.cli`
(engine from the registry, prompts from ``monorl.data.open_source``) or by tests (fakes).
No torch import here: completions are counted through ``Completion.token_ids``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from monorl.bench.stats import HostInfo, LatencyStats, RssTracker, fmt_seconds, rate

if TYPE_CHECKING:
    from monorl.core.interfaces import RolloutEngine
    from monorl.core.types import PromptInput, SamplingParams

DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 4, 16, 64)


@dataclass(slots=True)
class RolloutBatchResult:
    """Throughput of ``generate`` at one batch size (prompts per call)."""

    batch_size: int
    calls: int
    prompts: int
    requests: int
    """Sequences produced = prompts * params.n."""
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    tokens_per_s: float
    """Completion tokens per second (the number that bounds RL step time)."""
    total_tokens_per_s: float
    """(prompt + completion) tokens per second; prompt tokens counted when the engine reports them."""
    requests_per_s: float
    latency: LatencyStats
    """Per-``generate`` call wall time."""
    request_latency: LatencyStats
    """Call latency divided by the sequences in that call (amortised per-sequence cost)."""
    mean_completion_tokens: float
    peak_rss_bytes: int
    truncated_by_budget: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RolloutBenchmark:
    backend: str
    model: str
    sampling: dict[str, Any]
    results: list[RolloutBatchResult]
    best_batch_size: int | None
    recommendations: list[str]
    host: HostInfo
    baseline_rss_bytes: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "rollout",
            "backend": self.backend,
            "model": self.model,
            "sampling": dict(self.sampling),
            "results": [r.to_dict() for r in self.results],
            "best_batch_size": self.best_batch_size,
            "recommendations": list(self.recommendations),
            "host": self.host.to_dict(),
            "baseline_rss_bytes": self.baseline_rss_bytes,
            "extra": dict(self.extra),
        }


def _prompt_tokens(prompt: PromptInput, completions: Sequence[Any]) -> int:
    for c in completions:
        ids = getattr(c, "prompt_token_ids", None)
        if ids is not None:
            return len(ids)
    if prompt and isinstance(prompt[0], int):
        return len(prompt)
    return 0


def chunk_prompts(prompts: Sequence[PromptInput], batch_size: int) -> list[Sequence[PromptInput]]:
    """Full batches of ``batch_size``; when there are fewer prompts than that, one smaller batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if len(prompts) <= batch_size:
        return [prompts] if prompts else []
    full = len(prompts) // batch_size
    return [prompts[i * batch_size : (i + 1) * batch_size] for i in range(full)]


def measure_batch_size(
    engine: RolloutEngine,
    prompts: Sequence[PromptInput],
    params: SamplingParams,
    batch_size: int,
    *,
    seconds: float | None = None,
    rss: RssTracker | None = None,
) -> RolloutBatchResult:
    """Run ``generate`` over ``prompts`` in batches of ``batch_size`` (at least one call, then stop
    once ``seconds`` of wall time have been spent) and summarise."""
    rss = rss or RssTracker()
    latencies: list[float] = []
    per_request: list[float] = []
    n_prompts = n_requests = prompt_tokens = completion_tokens = 0
    truncated = False
    t_start = time.perf_counter()
    for batch in chunk_prompts(prompts, batch_size):
        if seconds is not None and latencies and (time.perf_counter() - t_start) >= seconds:
            truncated = True
            break
        t0 = time.perf_counter()
        outputs = engine.generate(batch, params)
        dt = time.perf_counter() - t0
        rss.sample()
        if len(outputs) != len(batch):
            raise RuntimeError(f"engine returned {len(outputs)} groups for {len(batch)} prompts")
        latencies.append(dt)
        n_prompts += len(batch)
        batch_requests = 0
        for prompt, group in zip(batch, outputs, strict=True):
            prompt_tokens += _prompt_tokens(prompt, group)
            batch_requests += len(group)
            for c in group:
                completion_tokens += len(c.token_ids)
        n_requests += batch_requests
        per_request.append(dt / max(1, batch_requests))
    total = sum(latencies)
    return RolloutBatchResult(
        batch_size=batch_size,
        calls=len(latencies),
        prompts=n_prompts,
        requests=n_requests,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        seconds=total,
        tokens_per_s=rate(completion_tokens, total),
        total_tokens_per_s=rate(prompt_tokens + completion_tokens, total),
        requests_per_s=rate(n_requests, total),
        latency=LatencyStats.from_samples(latencies),
        request_latency=LatencyStats.from_samples(per_request),
        mean_completion_tokens=completion_tokens / n_requests if n_requests else 0.0,
        peak_rss_bytes=rss.peak_bytes,
        truncated_by_budget=truncated,
    )


def recommend(results: Sequence[RolloutBatchResult], *, group_size: int, baseline_rss: int) -> tuple[int | None, list[str]]:
    """Pick the batch size with the best completion-token throughput and phrase config advice."""
    valid = [r for r in results if r.calls > 0]
    if not valid:
        return None, ["no batches ran; increase --prompts or --seconds"]
    best = max(valid, key=lambda r: r.tokens_per_s)
    notes = [
        f"best completion throughput {best.tokens_per_s:,.0f} tok/s at batch size {best.batch_size} "
        f"(p50 call latency {fmt_seconds(best.latency.p50_s)})",
        f"set rollout.extra.batch_max_size={best.batch_size} so the dynamic batcher coalesces parallel episodes into "
        f"calls of that size; keep rollout.extra.max_parallel_episodes >= {best.batch_size}",
    ]
    prompts_per_step = max(1, best.batch_size // max(1, group_size))
    notes.append(
        f"with rollout.group_size={group_size} a step of trainer.prompts_per_step={prompts_per_step} "
        f"fills one {best.batch_size}-sequence call; larger steps run several calls back to back"
    )
    largest = max(valid, key=lambda r: r.batch_size)
    if largest is not best and largest.tokens_per_s < 0.9 * best.tokens_per_s:
        notes.append(
            f"throughput drops past batch size {best.batch_size} "
            f"({largest.tokens_per_s:,.0f} tok/s at {largest.batch_size}): the engine is saturated, "
            "add data_parallel_size replicas instead of larger batches"
        )
    peak = max(r.peak_rss_bytes for r in valid)
    growth = peak - baseline_rss
    if growth > 1 << 30:
        notes.append(f"peak RSS grew by {growth / (1 << 30):.1f} GiB over the baseline; check that the host has headroom")
    return best.batch_size, notes


def run_rollout_benchmark(
    engine: RolloutEngine,
    prompts: Sequence[PromptInput],
    params: SamplingParams,
    *,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    seconds: float | None = None,
    warmup: bool = True,
    backend: str = "",
    model: str = "",
    group_size: int | None = None,
    extra: dict[str, Any] | None = None,
) -> RolloutBenchmark:
    if not prompts:
        raise ValueError("rollout benchmark needs at least one prompt")
    sizes = sorted({int(b) for b in batch_sizes if int(b) >= 1})
    if not sizes:
        raise ValueError("batch_sizes must contain at least one positive size")
    rss = RssTracker()
    baseline = rss.start_bytes
    if warmup:
        engine.generate(prompts[:1], params)
        rss.sample()
    results = [measure_batch_size(engine, prompts, params, b, seconds=seconds, rss=rss) for b in sizes]
    best, notes = recommend(results, group_size=group_size or params.n, baseline_rss=baseline)
    return RolloutBenchmark(
        backend=backend,
        model=model,
        sampling=asdict(params),
        results=results,
        best_batch_size=best,
        recommendations=notes,
        host=HostInfo.detect(),
        baseline_rss_bytes=baseline,
        extra=dict(extra or {}),
    )


__all__ = [
    "DEFAULT_BATCH_SIZES",
    "RolloutBatchResult",
    "RolloutBenchmark",
    "chunk_prompts",
    "measure_batch_size",
    "recommend",
    "run_rollout_benchmark",
]
