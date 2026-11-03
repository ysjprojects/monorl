"""Torch-free measurement primitives shared by every benchmark: percentiles, latency summaries,
peak-RSS tracking and the host description attached to every report."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import psutil


def percentile(samples: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy ``'linear'`` method). ``q`` in ``[0, 100]``."""
    if not samples:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


@dataclass(slots=True)
class LatencyStats:
    """Summary of a set of durations in seconds."""

    count: int
    mean_s: float
    p50_s: float
    p95_s: float
    min_s: float
    max_s: float

    @classmethod
    def from_samples(cls, samples: Iterable[float]) -> LatencyStats:
        values = [float(s) for s in samples]
        if not values:
            return cls(count=0, mean_s=0.0, p50_s=0.0, p95_s=0.0, min_s=0.0, max_s=0.0)
        return cls(
            count=len(values),
            mean_s=sum(values) / len(values),
            p50_s=percentile(values, 50),
            p95_s=percentile(values, 95),
            min_s=min(values),
            max_s=max(values),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RssTracker:
    """Tracks the peak resident set size of this process across ``sample()`` calls."""

    def __init__(self, pid: int | None = None) -> None:
        self._proc = psutil.Process(pid or os.getpid())
        self.start_bytes = self.current()
        self.peak_bytes = self.start_bytes

    def current(self) -> int:
        return int(self._proc.memory_info().rss)

    def sample(self) -> int:
        rss = self.current()
        if rss > self.peak_bytes:
            self.peak_bytes = rss
        return rss


@dataclass(slots=True)
class HostInfo:
    """Where the benchmark ran. ``torch`` fields are filled only when torch is already imported."""

    platform: str
    python: str
    cpu_count: int
    memory_bytes: int
    torch: str | None = None
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def detect(cls) -> HostInfo:
        info = cls(
            platform=f"{platform.system()} {platform.machine()}",
            python=platform.python_version(),
            cpu_count=os.cpu_count() or 1,
            memory_bytes=int(psutil.virtual_memory().total),
        )
        torch = sys.modules.get("torch")
        if torch is not None:
            info.torch = str(torch.__version__)
            if torch.cuda.is_available():
                info.device = f"cuda x{torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                info.device = "mps"
            else:
                info.device = "cpu"
        return info

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rate(count: float, seconds: float) -> float:
    return float(count) / seconds if seconds > 0 else 0.0


def fmt_bytes(n: float) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def fmt_params(n: float) -> str:
    value = float(n)
    if value >= 1e9:
        return f"{value / 1e9:.2f} B"
    if value >= 1e6:
        return f"{value / 1e6:.1f} M"
    return f"{value / 1e3:.0f} K"


def fmt_seconds(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1.0:
        return f"{s * 1e3:.1f} ms"
    if s < 90.0:
        return f"{s:.2f} s"
    return f"{s / 60.0:.1f} min"


__all__ = [
    "HostInfo",
    "LatencyStats",
    "RssTracker",
    "fmt_bytes",
    "fmt_params",
    "fmt_seconds",
    "percentile",
    "rate",
]
