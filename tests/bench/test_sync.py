from __future__ import annotations

import json
from pathlib import Path

import pytest

from monorl.bench.sync import (
    StrategyCandidate,
    SyntheticEngine,
    SyntheticWeightSource,
    measure_strategy,
    recommend,
    resolve_strategies,
    run_sync_benchmark,
    synthetic_config,
)
from monorl.core.errors import WeightSyncError

torch = pytest.importorskip("torch")


def test_synthetic_source_and_engine_roundtrip(tmp_path: Path) -> None:
    src = SyntheticWeightSource(2_500_000, chunk=1_000_000)
    assert src.num_params == 2_500_000 and len(src.tensors) == 3
    assert src.num_bytes == 2_500_000 * 4
    eng = SyntheticEngine(src)
    eng.update_weights(src.export_weights())
    assert eng.updates == 1 and eng.bytes_received == src.num_bytes
    assert all(torch.equal(eng.params[k], v) for k, v in src.tensors.items())
    src.save_pretrained(tmp_path / "w")
    assert (tmp_path / "w" / "model.safetensors").exists() and (tmp_path / "w" / "config.json").exists()
    eng2 = SyntheticEngine(src)
    eng2.load_weights_from_path(tmp_path / "w")
    assert eng2.bytes_received == src.num_bytes
    with pytest.raises(ValueError):
        SyntheticWeightSource(0)


def test_resolve_strategies_reports_unusable_ones(tmp_path: Path) -> None:
    cfg = synthetic_config()
    candidates = {c.name: c for c in resolve_strategies(cfg, sync_dir=str(tmp_path / "sync"))}
    assert candidates["inprocess"].sync is not None
    assert candidates["checkpoint"].sync is not None
    assert candidates["checkpoint"].sync.dir == tmp_path / "sync"
    assert candidates["broadcast"].sync is None and candidates["broadcast"].reason
    if not torch.cuda.is_available():
        assert candidates["cuda_ipc"].sync is None and "CUDA" in candidates["cuda_ipc"].reason
    only = resolve_strategies(cfg, ["inprocess"])
    assert [c.name for c in only] == ["inprocess"]


def test_measure_strategy_times_inprocess_and_checkpoint(tmp_path: Path) -> None:
    cfg = synthetic_config()
    src = SyntheticWeightSource(300_000, chunk=100_000)
    engines = [SyntheticEngine(src), SyntheticEngine(src)]
    for cand in resolve_strategies(cfg, ["inprocess", "checkpoint"], sync_dir=str(tmp_path)):
        r = measure_strategy(cand, src, engines, repeats=2)
        assert not r.skipped, r.reason
        assert r.repeats == 2 and r.seconds.count == 2 and r.seconds.p50_s > 0
        # inprocess counts the stream per engine; checkpoint counts the safetensors file once
        assert r.bytes >= src.num_bytes
        assert r.gb_per_s == pytest.approx(r.bytes / 1e9 / r.seconds.p50_s)
        assert "weight_sync/seconds" in r.metrics
    assert all(e.updates == 4 for e in engines)


def test_measure_strategy_skips_failures() -> None:
    class Failing:
        name = "failing"

        def sync(self, trainer, engines):
            raise WeightSyncError("nope")

    src = SyntheticWeightSource(10)
    r = measure_strategy(StrategyCandidate("failing", Failing()), src, [SyntheticEngine(src)])
    assert r.skipped and r.reason == "nope" and r.repeats == 0
    r2 = measure_strategy(StrategyCandidate("absent", None, "not built"), src, [])
    assert r2.skipped and r2.reason == "not built"


def test_run_sync_benchmark_recommends_fastest(tmp_path: Path) -> None:
    cfg = synthetic_config()
    src = SyntheticWeightSource(200_000, chunk=50_000)
    engines = [SyntheticEngine(src)]
    cands = resolve_strategies(cfg, sync_dir=str(tmp_path))
    bench = run_sync_benchmark(src, engines, cands, repeats=2, source="synthetic", num_params=src.num_params)
    names = {r.strategy: r for r in bench.results}
    assert not names["inprocess"].skipped and not names["checkpoint"].skipped
    assert names["broadcast"].skipped
    assert bench.fastest in ("inprocess", "checkpoint")
    assert any(n.startswith("fastest: cluster.weight_sync=") for n in bench.recommendations)
    assert any("skipped" in n for n in bench.recommendations)
    d = bench.to_dict()
    assert d["kind"] == "sync" and d["num_params"] == 200_000 and d["engines"] == 1
    json.loads(json.dumps(d))


def test_recommend_handles_all_skipped() -> None:
    from monorl.bench.stats import LatencyStats
    from monorl.bench.sync import SyncResult

    r = SyncResult("x", 0, 0, LatencyStats.from_samples([]), 0.0, {}, skipped=True, reason="no")
    fastest, notes = recommend([r], engines=1)
    assert fastest is None and "x: no" in notes[0]
