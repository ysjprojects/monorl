from __future__ import annotations

import sys

import pytest

from monorl.bench.stats import HostInfo, LatencyStats, RssTracker, fmt_bytes, fmt_seconds, percentile, rate


def test_percentile_linear_interpolation() -> None:
    samples = list(range(1, 101))
    assert percentile(samples, 50) == pytest.approx(50.5)
    assert percentile(samples, 95) == pytest.approx(95.05)
    assert percentile(samples, 0) == 1
    assert percentile(samples, 100) == 100
    assert percentile([3.0], 95) == 3.0
    assert percentile([5.0, 1.0], 50) == pytest.approx(3.0)


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_latency_stats_from_samples_and_empty() -> None:
    s = LatencyStats.from_samples([0.1, 0.3, 0.2, 0.4])
    assert s.count == 4
    assert s.mean_s == pytest.approx(0.25)
    assert s.p50_s == pytest.approx(0.25)
    assert s.p95_s == pytest.approx(0.385)
    assert (s.min_s, s.max_s) == (0.1, 0.4)
    assert s.to_dict()["p50_s"] == pytest.approx(0.25)
    empty = LatencyStats.from_samples([])
    assert empty.count == 0 and empty.p95_s == 0.0


def test_rss_tracker_monotonic_peak() -> None:
    t = RssTracker()
    assert t.start_bytes > 0
    big = bytearray(32 << 20)
    rss = t.sample()
    assert t.peak_bytes >= rss >= 0
    del big
    t.sample()
    assert t.peak_bytes >= rss


def test_rate_and_formatting() -> None:
    assert rate(10, 2) == 5.0
    assert rate(10, 0) == 0.0
    assert fmt_bytes(512) == "512 B"
    assert fmt_bytes(1536) == "1.5 KiB"
    assert fmt_bytes(3 << 30) == "3.0 GiB"
    assert fmt_seconds(0.0005) == "500 µs"
    assert fmt_seconds(0.25) == "250.0 ms"
    assert fmt_seconds(2.5) == "2.50 s"
    assert fmt_seconds(600) == "10.0 min"


def test_host_info_without_torch_fields() -> None:
    info = HostInfo.detect()
    assert info.cpu_count >= 1 and info.memory_bytes > 0
    if "torch" not in sys.modules:
        assert info.torch is None and info.device is None
    d = info.to_dict()
    assert set(d) >= {"platform", "python", "cpu_count", "memory_bytes", "torch", "device"}
