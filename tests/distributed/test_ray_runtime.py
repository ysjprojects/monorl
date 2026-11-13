from __future__ import annotations

from pathlib import Path

import pytest
import torch
from distributed_fakes import make_weights, weight_sum

from monorl.core.batch import TrainBatch
from monorl.core.interfaces import RolloutEngine, Trainer
from monorl.core.types import SamplingParams, SpeculationStats
from monorl.distributed import build_runtime
from monorl.distributed.ray_runtime import RayBroadcastWeightSync, RayObjectStoreWeightSync

ray = pytest.importorskip("ray")

pytestmark = pytest.mark.timeout(180)


@pytest.fixture(scope="module")
def ray_session():
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def _batch(n: int = 4, t: int = 3) -> TrainBatch:
    ids = torch.arange(n * t).view(n, t)
    return TrainBatch(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        response_mask=torch.ones_like(ids),
        advantages=torch.ones(n, t),
        rewards=torch.ones(n),
        group_ids=[f"g{i // 2}" for i in range(n)],
    )


def _tag(engine: RolloutEngine) -> str:
    out = engine.generate([[{"role": "user", "content": "hi"}]], SamplingParams(n=1))
    return out[0][0].text.split(" ")[0]


def _trainer_tag(trainer: Trainer) -> str:
    return f"w={weight_sum(dict(trainer.export_weights())):.6f}"


def test_ray_runtime_object_store_sync(ray_session, make_config, tmp_path: Path):
    cfg = make_config(
        cluster={"mode": "ray", "extra": {"providers": ["distributed_fakes"], "sync_chunk_bytes": 64}}
    )
    rt = build_runtime(cfg)
    try:
        assert rt.placement["mode"] == "ray" and rt.placement["placement"] == "colocated"
        assert rt.placement["placement_group"]["strategy"] == "PACK"
        assert rt.placement["weight_sync"] == "object_store"
        assert isinstance(rt.weight_sync, RayObjectStoreWeightSync)
        assert len(rt.engines) == 1
        engine, trainer = rt.engines[0], rt.trainer
        assert isinstance(engine, RolloutEngine) and isinstance(trainer, Trainer)

        prompts = [[{"role": "user", "content": "hi"}], [1, 2, 3]]
        out = engine.generate(prompts, SamplingParams(n=2, max_new_tokens=4))
        assert len(out) == 2 and all(len(c) == 2 for c in out)
        assert out[1][1].text == f"w={weight_sum(make_weights(0)):.6f} p=1 n=1"
        assert out[0][0].logprobs == [0.0, 0.0]
        assert engine.tokenizer == {"name": "fake-tokenizer"}
        # the optional drafting counters cross the actor boundary and restart with every read
        assert engine.speculation_stats() == SpeculationStats(steps=4, drafted=4, accepted=4, emitted=8)
        assert engine.speculation_stats() == SpeculationStats()

        assert trainer.step == 0
        metrics = trainer.train_step(_batch())
        assert trainer.step == 1
        assert metrics["loss/policy"] == pytest.approx(12.0) and metrics["batch/size"] == 4.0
        assert trainer.compute_logprobs(_batch(), reference=True).shape == (4, 3)
        torch.testing.assert_close(trainer.compute_values(_batch()), _batch().input_ids / 10)

        assert _tag(engine) != _trainer_tag(trainer)
        sync_metrics = rt.weight_sync.sync(trainer, rt.engines)
        assert sync_metrics["weight_sync/bytes"] == sum(t.numel() * 4 for t in make_weights(1).values())
        assert sync_metrics["weight_sync/chunks"] == 2  # two tensors, 64-byte chunks
        assert _tag(engine) == _trainer_tag(trainer)

        trainer.save_checkpoint(tmp_path / "ckpt")
        assert (tmp_path / "ckpt" / "ckpt.pt").exists()
        trainer.save_pretrained(tmp_path / "export")
        engine.load_weights_from_path(tmp_path / "export")
        engine.sleep()
        engine.wake_up()
        actors = list(engine.actors) + list(trainer.actors)
    finally:
        rt.close()

    for actor in actors:
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.node_ip.remote(), timeout=30)


def test_ray_runtime_broadcast_sync_cpu(ray_session, make_config):
    cfg = make_config(
        cluster={
            "mode": "ray",
            "placement": "disaggregated",
            "weight_sync": "broadcast",
            "extra": {"providers": ["distributed_fakes"]},
        }
    )
    rt = build_runtime(cfg)
    try:
        assert rt.placement["placement"] == "disaggregated"
        assert rt.placement["placement_group"]["strategy"] == "SPREAD"
        assert isinstance(rt.weight_sync, RayBroadcastWeightSync)
        engine, trainer = rt.engines[0], rt.trainer
        assert _tag(engine) != _trainer_tag(trainer)
        metrics = rt.weight_sync.sync(trainer, rt.engines)
        assert rt.weight_sync.spec is not None and rt.weight_sync.spec.backend == "gloo"
        assert rt.weight_sync.spec.world_size == 2
        assert metrics["weight_sync/buckets"] == 1.0
        assert _tag(engine) == _trainer_tag(trainer)
        trainer.train_step(_batch())
        rt.weight_sync.sync(trainer, rt.engines)  # group is reused across steps
        assert _tag(engine) == _trainer_tag(trainer)
    finally:
        rt.close()
