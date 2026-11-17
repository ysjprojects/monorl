from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from distributed_fakes import FakeEngine, FakeTrainer, make_weights, weight_sum

from monorl.core.batch import TrainBatch
from monorl.core.interfaces import RolloutEngine, Trainer
from monorl.core.types import Completion, PromptInput, SamplingParams, SpeculationStats
from monorl.distributed import build_runtime
from monorl.distributed.ray_runtime import RayBroadcastWeightSync, RayObjectStoreWeightSync
from monorl.mtp import WIRE_PREFIX

ray = pytest.importorskip("ray")

# One local Ray for the module: under pytest-xdist its tests stay on the worker that started it.
pytestmark = [pytest.mark.timeout(180), pytest.mark.xdist_group("ray")]


@pytest.fixture(scope="module")
def ray_session():
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


# -- a provider for the actors (``cluster.extra.providers = ["test_ray_runtime"]``) ---------------

MTP_NAME = WIRE_PREFIX + "mtp.head.weight"


def mixed_state() -> dict[str, torch.Tensor]:
    """Every kind of tensor a state holds, in the forms a trainer may hand out: dtypes with and
    without a numpy counterpart, views that are not contiguous, a parameter, no dimension, no
    element, and values that only survive a byte-exact transfer."""
    g = torch.Generator().manual_seed(7)
    return {
        "f32.weight": torch.nn.Parameter(torch.randn(8, 4, generator=g)),
        "f32.transposed": torch.randn(6, 5, generator=g).t(),
        "f32.strided": torch.randn(16, generator=g)[1::3],
        "f32.special": torch.tensor([float("nan"), -0.0, float("inf"), 1e-45]),
        "bf16.weight": torch.randn(5, 3, generator=g).to(torch.bfloat16),
        "f16.weight": torch.randn(3, generator=g).to(torch.float16),
        "i64.ids": torch.arange(-3, 4, dtype=torch.int64) * (2**40 + 1),
        "bool.mask": torch.tensor([True, False, True]),
        "scalar": torch.tensor(2.5),
        "empty": torch.empty(0, 3),
        # larger than Ray inlines into the owner's memory, like real weights: read from the store
        "f32.large": torch.randn(300, 200, generator=g),
        MTP_NAME: torch.randn(250, 200, generator=g),
    }


def fingerprint(state: dict[str, torch.Tensor]) -> dict[str, list[Any]]:
    """Name → dtype, shape, the values' bytes (their digest when there are many) and whether
    autograd tracks the tensor."""
    out = {}
    for name, tensor in state.items():
        t = tensor.detach()
        widened = t.to(torch.float32) if t.dtype == torch.bfloat16 else t  # exact; numpy has no bf16
        raw = widened.contiguous().numpy().tobytes()
        shown = raw.hex() if len(raw) <= 64 else hashlib.sha256(raw).hexdigest()
        out[name] = [str(t.dtype), list(t.shape), shown, tensor.requires_grad]
    return out


class StateTrainer(FakeTrainer):
    def __init__(self, **kw: Any) -> None:
        super().__init__(1, **kw)
        self.weights = mixed_state()

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        yield from self.weights.items()


class StateEngine(FakeEngine):
    """Holds copies of what it was given, and treats the MTP layers' tensors as its own the way a
    drafter does that adopts them as parameters: it writes to them."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(0, **kw)
        self.held: dict[str, torch.Tensor] = {}

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        self.held = {}
        for name, tensor in weights:
            self.held[name] = tensor.clone()
            if name == MTP_NAME:
                tensor.zero_()

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        return [[Completion(text=json.dumps(fingerprint(self.held)), token_ids=[])] for _ in prompts]


def register(registry: Any) -> None:
    registry.register("rollout_engine", "state", lambda config, **kw: StateEngine(**kw))
    registry.register("trainer", "state", lambda config, **kw: StateTrainer(**kw))
    registry.register("rollout_engine", "probed_hf", probed_hf_engine)


def _held(engine: RolloutEngine) -> dict[str, list[Any]]:
    return json.loads(engine.generate([[1]], SamplingParams(n=1))[0][0].text)


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


def test_object_store_sync_delivers_every_tensor_exactly(ray_session, make_config):
    """Dtype, shape and bytes of every tensor of a mixed state reach the engine through the object
    store, detached; and what one reader does with its tensors does not reach the next reader."""
    cfg = make_config(
        rollout={"backend": "state"},
        trainer={"backend": "state"},
        cluster={"mode": "ray", "extra": {"providers": ["test_ray_runtime"], "sync_chunk_bytes": 64}},
    )
    state = mixed_state()
    sent = {name: [*described[:3], False] for name, described in fingerprint(state).items()}
    rt = build_runtime(cfg)
    try:
        engine, trainer = rt.engines[0], rt.trainer
        assert _held(engine) == {}
        metrics = rt.weight_sync.sync(trainer, rt.engines)
        assert _held(engine) == sent
        assert list(_held(engine)) == list(state)  # in the order the trainer exports them
        assert metrics["weight_sync/bytes"] == sum(t.numel() * t.element_size() for t in state.values())
        # 64-byte chunks: five tensors that exceed one alone, and the seven small ones in three
        assert metrics["weight_sync/chunks"] == 7

        # The same objects again, after the engine wrote to the MTP tensors it was given.
        refs, _ = ray.get(trainer.actors[0].export_weight_chunks.remote(64))
        for _ in range(2):
            engine.pull_weights_async(refs).result()
            assert _held(engine) == sent

        # The driver's view of the trainer's weights: the same tensors, the caller's to change.
        exported = dict(trainer.export_weights())
        assert fingerprint(exported) == sent
        exported["i64.ids"].add_(1)
        exported["f32.large"].zero_()
        assert fingerprint(dict(trainer.export_weights())) == sent
    finally:
        rt.close()


@pytest.mark.parametrize("strategy", ["inprocess", "broadcast"])
def test_ray_sync_sends_what_changed_and_everything_when_an_actor_may_not_hold_it(
    ray_session, make_config, tmp_path: Path, strategy
):
    """The trainer trains one of its two tensors. An engine actor is sent that one only, as long
    as it holds the rest: not at first, not after a sync or another update of it failed, not after
    its engine was built anew."""
    fail_flag = tmp_path / "fail"
    cfg = make_config(
        rollout={"backend": "flaky", "extra": {"fail_flag": str(fail_flag)}},
        trainer={"backend": "adapter"},
        cluster={
            "mode": "ray",
            "placement": "disaggregated",
            "weight_sync": strategy,
            "extra": {"providers": ["distributed_fakes"]},
        },
    )
    rt = build_runtime(cfg)
    try:
        engine, trainer = rt.engines[0], rt.trainer

        def synced() -> float:
            sent = rt.weight_sync.sync(trainer, rt.engines)["weight_sync/bytes"]
            assert _tag(engine) == _trainer_tag(trainer)
            return sent

        everything = synced()
        trainer.train_step(_batch())
        trained = synced()
        assert trained < everything
        if strategy == "inprocess":  # the object store counts the tensors' bytes and nothing else
            assert (everything, trained) == (4 * (8 * 4 + 4 * 4), 4 * 4 * 4)

        trainer.train_step(_batch())
        fail_flag.touch()
        with pytest.raises(ray.exceptions.RayTaskError, match="broke down"):
            rt.weight_sync.sync(trainer, rt.engines)
        fail_flag.unlink()
        assert synced() == everything
        trainer.train_step(_batch())
        assert synced() == trained

        fail_flag.touch()
        with pytest.raises(ray.exceptions.RayTaskError, match="broke down"):
            engine.update_weights(make_weights(3).items())  # weights that are no push of the trainer's
        fail_flag.unlink()
        assert synced() == everything

        ray.get(engine.actors[0].build.remote({}))  # a new engine in the actor, with weights of its own
        assert _tag(engine) != _trainer_tag(trainer)
        assert synced() == everything
    finally:
        rt.close()


def probed_hf_engine(config: Any, *, replica: int = 0, **_: Any) -> RolloutEngine:
    """The ``hf`` engine with other weights than the checkpoint's and a ``generate`` that tells
    what its model holds, which is all a test sees of an engine inside an actor."""
    from monorl.rollout.hf_engine import HFRolloutEngine

    class Probed(HFRolloutEngine):
        def generate(self, prompts: Any, params: Any) -> list[list[Completion]]:
            held = fingerprint(dict(self.model.state_dict()))
            return [[Completion(text=json.dumps(held), token_ids=[])] for _ in prompts]

    engine = Probed(config, replica=replica)
    with torch.no_grad():
        for parameter in engine.model.parameters():
            parameter.add_(1.0)
    return engine


def test_object_store_sync_of_lora_training_into_hf_engine(ray_session, make_config, tiny_model_dir: Path):
    """The reference trainer and the ``hf`` engine in actors: after every sync the engine holds
    exactly what the trainer exports, though only the first sync carries the frozen tensors."""
    from transformers import AutoTokenizer

    from tests.trainer.conftest import make_batch as make_train_batch

    cfg = make_config(
        model={"lora": {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.0}},
        rollout={"backend": "probed_hf", "device": "cpu"},
        trainer={
            "backend": "reference",
            "device": "cpu",
            "precision": "fp32",
            "micro_batch_size": 2,
            "learning_rate": 1e-2,
            "max_steps": 10,
            "max_seq_len": 64,
        },
        cluster={"mode": "ray", "extra": {"providers": ["test_ray_runtime"]}},
    )
    batch = make_train_batch(AutoTokenizer.from_pretrained(tiny_model_dir))
    rt = build_runtime(cfg)
    try:
        engine, trainer = rt.engines[0], rt.trainer

        def synced() -> float:
            sent = rt.weight_sync.sync(trainer, rt.engines)["weight_sync/bytes"]
            exported = fingerprint(dict(trainer.export_weights()))
            assert _held(engine) == exported
            return sent

        assert _held(engine) != fingerprint(dict(trainer.export_weights()))
        everything = synced()
        for _ in range(3):
            trainer.train_step(batch)
            assert 0 < synced() < everything / 10
    finally:
        rt.close()


def test_ray_actors_load_the_providers_of_the_run_config(ray_session, make_config):
    """``providers`` at the top of the config reach the actors, where engine and trainer are built."""
    cfg = make_config(
        providers=["top_level_provider"],
        rollout={"backend": "provided"},
        trainer={"backend": "provided"},
        cluster={"mode": "ray"},
    )
    rt = build_runtime(cfg)
    try:
        assert rt.placement["rollout_backend"] == "provided" and rt.placement["trainer_backend"] == "provided"
        before = _tag(rt.engines[0])
        rt.trainer.train_step(_batch())
        rt.weight_sync.sync(rt.trainer, rt.engines)
        assert _tag(rt.engines[0]) == _trainer_tag(rt.trainer) != before
    finally:
        rt.close()


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
