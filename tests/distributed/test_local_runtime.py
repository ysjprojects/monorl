from __future__ import annotations

from pathlib import Path

import distributed_fakes
import torch

from monorl.core.registry import registry
from monorl.distributed import CheckpointWeightSync, InProcessWeightSync, Runtime, build_runtime


def _ensure_fakes() -> None:
    if not registry.contains("rollout_engine", "fake"):
        distributed_fakes.register(registry)


def test_local_runtime_uses_planner_and_inprocess_sync(make_config):
    _ensure_fakes()
    rt = build_runtime(make_config(rollout={"data_parallel_size": 2}))
    assert isinstance(rt, Runtime)
    assert len(rt.engines) == 2 and [e.replica for e in rt.engines] == [0, 1]
    assert isinstance(rt.weight_sync, InProcessWeightSync)
    assert rt.placement["mode"] == "local" and rt.placement["placement"] == "colocated"
    assert rt.placement["weight_sync"] == "inprocess" and rt.placement["engine_replicas"] == 2
    assert rt.placement["rollout_backend"] == "fake" and rt.placement["trainer_backend"] == "fake"
    rt.weight_sync.sync(rt.trainer, rt.engines)
    for e in rt.engines:
        assert torch.equal(e.weights["model.embed.weight"], rt.trainer.weights["model.embed.weight"])
    closed = []
    rt.closers.append(lambda: closed.append(True))
    rt.close()
    assert all(e.shut_down for e in rt.engines) and closed == [True]


def test_local_runtime_explicit_checkpoint_sync(make_config, tmp_path: Path):
    _ensure_fakes()
    cfg = make_config(cluster={"weight_sync": "checkpoint", "extra": {"sync_dir": str(tmp_path / "sync")}})
    rt = build_runtime(cfg)
    assert isinstance(rt.weight_sync, CheckpointWeightSync)
    assert rt.weight_sync.dir == tmp_path / "sync"
    assert rt.placement["weight_sync"] == "checkpoint"
    rt.weight_sync.sync(rt.trainer, rt.engines)
    assert torch.equal(rt.engines[0].weights["model.embed.weight"], rt.trainer.weights["model.embed.weight"])
    rt.close()
