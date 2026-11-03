from __future__ import annotations

import json
from pathlib import Path

import pytest

from monorl.bench.rollout import (
    RolloutBenchmark,
    chunk_prompts,
    measure_batch_size,
    recommend,
    run_rollout_benchmark,
)
from monorl.bench.stats import RssTracker
from monorl.core.types import SamplingParams

from .conftest import FakeEngine


def test_chunk_prompts_full_batches_only(chat_prompts) -> None:
    assert [len(c) for c in chunk_prompts(chat_prompts, 4)] == [4, 4]
    assert [len(c) for c in chunk_prompts(chat_prompts, 10)] == [10]
    assert [len(c) for c in chunk_prompts(chat_prompts, 64)] == [10]
    assert chunk_prompts([], 4) == []
    with pytest.raises(ValueError):
        chunk_prompts(chat_prompts, 0)


def test_measure_batch_size_counts_tokens_and_requests(chat_prompts) -> None:
    engine = FakeEngine(tokens_per_completion=5)
    params = SamplingParams(n=2, max_new_tokens=8)
    r = measure_batch_size(engine, chat_prompts, params, 4, rss=RssTracker())
    assert engine.calls == [4, 4]
    assert r.calls == 2 and r.prompts == 8 and r.requests == 16
    assert r.completion_tokens == 16 * 5
    # fake prompt ids are len(messages) + 3 = 4 per prompt
    assert r.prompt_tokens == 8 * 4
    assert r.mean_completion_tokens == 5.0
    assert r.seconds > 0
    assert r.tokens_per_s == pytest.approx(r.completion_tokens / r.seconds)
    assert r.total_tokens_per_s == pytest.approx((r.prompt_tokens + r.completion_tokens) / r.seconds)
    assert r.requests_per_s == pytest.approx(16 / r.seconds)
    assert r.latency.count == 2 and r.request_latency.count == 2
    assert r.request_latency.mean_s == pytest.approx(r.latency.mean_s / 8)
    assert r.peak_rss_bytes > 0
    assert not r.truncated_by_budget


def test_measure_batch_size_tokens_per_s_matches_known_delay(chat_prompts) -> None:
    engine = FakeEngine(tokens_per_completion=10, seconds_per_call=0.02)
    r = measure_batch_size(engine, chat_prompts, SamplingParams(n=1), 5)
    assert r.calls == 2 and r.completion_tokens == 100
    assert r.seconds >= 0.04
    assert 100 / 0.2 < r.tokens_per_s <= 100 / 0.04 + 1e-9
    assert r.latency.p50_s >= 0.02 and r.latency.p95_s >= r.latency.p50_s


def test_measure_batch_size_respects_time_budget(chat_prompts) -> None:
    engine = FakeEngine(seconds_per_call=0.03)
    r = measure_batch_size(engine, chat_prompts, SamplingParams(n=1), 1, seconds=0.0)
    assert r.calls == 1 and r.truncated_by_budget
    assert engine.calls == [1]


def test_measure_batch_size_rejects_wrong_group_count(chat_prompts) -> None:
    class Broken(FakeEngine):
        def generate(self, prompts, params):
            return super().generate(prompts, params)[:-1]

    with pytest.raises(RuntimeError, match="returned 1 groups for 2 prompts"):
        measure_batch_size(Broken(), chat_prompts[:2], SamplingParams(), 2)


def test_run_rollout_benchmark_picks_best_and_serialises(chat_prompts, tmp_path: Path) -> None:
    # per-prompt cost makes larger batches no faster; a per-call cost makes them faster.
    engine = FakeEngine(tokens_per_completion=4, seconds_per_call=0.01)
    params = SamplingParams(n=2, max_new_tokens=4)
    bench = run_rollout_benchmark(
        engine, chat_prompts, params, batch_sizes=[1, 5, 5, 10], warmup=True, backend="fake", model="m", group_size=2
    )
    assert isinstance(bench, RolloutBenchmark)
    assert [r.batch_size for r in bench.results] == [1, 5, 10]
    assert engine.calls[0] == 1  # warmup call
    assert bench.best_batch_size in (5, 10)
    best = max(bench.results, key=lambda r: r.tokens_per_s)
    assert bench.best_batch_size == best.batch_size
    assert any("batch_max_size" in n for n in bench.recommendations)
    assert any("prompts_per_step" in n for n in bench.recommendations)
    d = bench.to_dict()
    assert d["kind"] == "rollout" and d["sampling"]["n"] == 2 and len(d["results"]) == 3
    assert d["results"][0]["latency"]["p50_s"] > 0
    json.loads(json.dumps(d))

    from monorl.bench.report import write_json

    out = write_json(bench, tmp_path / "r.json")
    assert json.loads(out.read_text())["best_batch_size"] == bench.best_batch_size


def test_recommend_flags_saturation_and_empty() -> None:
    from monorl.bench.rollout import RolloutBatchResult
    from monorl.bench.stats import LatencyStats

    def res(bs: int, tps: float) -> RolloutBatchResult:
        lat = LatencyStats.from_samples([1.0])
        return RolloutBatchResult(bs, 1, bs, bs, 0, int(tps), 1.0, tps, tps, bs, lat, lat, 1.0, 1 << 20)

    best, notes = recommend([res(4, 100.0), res(16, 300.0), res(64, 200.0)], group_size=4, baseline_rss=0)
    assert best == 16
    assert any("saturated" in n for n in notes)
    assert any("prompts_per_step=4" in n for n in notes)
    assert recommend([], group_size=4, baseline_rss=0) == (None, ["no batches ran; increase --prompts or --seconds"])


def test_run_rollout_benchmark_validates_inputs(chat_prompts) -> None:
    with pytest.raises(ValueError):
        run_rollout_benchmark(FakeEngine(), [], SamplingParams())
    with pytest.raises(ValueError):
        run_rollout_benchmark(FakeEngine(), chat_prompts, SamplingParams(), batch_sizes=[0])
