from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from distributed_fakes import (
    CollectorEngine,
    FakeEngine,
    FakeTrainer,
    broadcast_worker,
    ipc_consumer,
    make_broadcast_weights,
)

from monorl.core.errors import WeightSyncError
from monorl.core.interfaces import WeightSync
from monorl.core.registry import registry
from monorl.distributed import (
    BroadcastGroupSpec,
    BroadcastWeightSync,
    CheckpointWeightSync,
    CudaIpcWeightSync,
    InProcessWeightSync,
)

HAS_CUDA = torch.cuda.is_available()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# -- checkpoint -------------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path: Path):
    trainer, engines = FakeTrainer(seed=5), [FakeEngine(seed=0), FakeEngine(seed=0)]
    sync = CheckpointWeightSync(tmp_path / "sync")
    assert isinstance(sync, WeightSync) and sync.name == "checkpoint"
    for e in engines:
        assert not torch.equal(e.weights["model.embed.weight"], trainer.weights["model.embed.weight"])

    metrics = sync.sync(trainer, engines)

    for e in engines:
        for name, t in trainer.weights.items():
            assert torch.equal(e.weights[name], t)
        assert e.updates == 1
    assert metrics["weight_sync/bytes"] > 0 and metrics["weight_sync/seconds"] >= 0
    assert (sync.weights_dir / "weights.pt").exists()
    assert not list(tmp_path.joinpath("sync").glob(".weights-*"))

    trainer.train_step(_batch())
    sync.sync(trainer, engines)
    assert torch.equal(engines[0].weights["model.embed.weight"], trainer.weights["model.embed.weight"])


def _batch():
    from monorl.core.batch import TrainBatch

    ids = torch.zeros(2, 3, dtype=torch.long)
    return TrainBatch(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        response_mask=torch.ones_like(ids),
        advantages=torch.ones(2, 3),
        rewards=torch.ones(2),
        group_ids=["g", "g"],
    )


# -- broadcast --------------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_broadcast_between_two_spawned_cpu_processes(tmp_path: Path):
    seed = 11
    spec = BroadcastGroupSpec(
        master_addr="127.0.0.1", master_port=_free_port(), world_size=2, backend="gloo", timeout_s=60, bucket_bytes=256
    )
    out = tmp_path / "received.pt"
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=broadcast_worker, args=(rank, spec, seed, str(out))) for rank in (0, 1)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=90)
    assert [p.exitcode for p in procs] == [0, 0]

    received = torch.load(out)
    expected = make_broadcast_weights(seed)
    assert list(received) == list(expected)
    for name, t in expected.items():
        assert received[name].dtype == t.dtype and received[name].shape == t.shape
        assert torch.equal(received[name], t), name
    sender = ast.literal_eval((tmp_path / "received.pt.rank0.json").read_text())
    receiver = ast.literal_eval((tmp_path / "received.pt.rank1.json").read_text())
    assert sender["weight_sync/buckets"] == receiver["weight_sync/buckets"] >= 4  # 256-byte buckets, mixed dtypes
    payload = sum(t.numel() * t.element_size() for t in expected.values())
    assert receiver["weight_sync/bytes"] == payload


def test_broadcast_validation():
    spec = BroadcastGroupSpec(master_addr="127.0.0.1", master_port=1, world_size=2)
    with pytest.raises(WeightSyncError, match="rank"):
        BroadcastWeightSync(spec, rank=2)
    with pytest.raises(WeightSyncError, match="rank 0"):
        BroadcastWeightSync(spec, rank=0).receive(CollectorEngine())
    with pytest.raises(WeightSyncError, match="only rank 0"):
        BroadcastWeightSync(spec, rank=1).send([])
    with pytest.raises(WeightSyncError, match="cannot receive broadcasts"):
        BroadcastWeightSync(spec, rank=0).sync(FakeTrainer(), [FakeEngine()])
    with pytest.raises(WeightSyncError, match="at least one engine"):
        BroadcastWeightSync(spec, rank=0).sync(FakeTrainer(), [])
    if not HAS_CUDA:
        with pytest.raises(WeightSyncError, match="nccl"):
            BroadcastWeightSync(BroadcastGroupSpec("127.0.0.1", 1, 2, backend="nccl"), rank=0)


# -- cuda ipc ---------------------------------------------------------------------------------


@pytest.mark.skipif(HAS_CUDA, reason="CUDA present")
def test_cuda_ipc_rejected_without_cuda():
    with pytest.raises(WeightSyncError, match="needs CUDA"):
        CudaIpcWeightSync()


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")
@pytest.mark.timeout(120)
def test_cuda_ipc_handles_open_in_another_process():
    trainer = FakeTrainer(seed=3)
    trainer.weights = {k: v.cuda() for k, v in trainer.weights.items()}
    sync = CudaIpcWeightSync()
    handles, keepalive = sync.export_handles(trainer.export_weights())
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=ipc_consumer, args=(handles, q))
    p.start()
    sums = q.get(timeout=60)
    p.join(timeout=30)
    assert p.exitcode == 0
    for name, t in trainer.weights.items():
        assert sums[name] == pytest.approx(float(t.double().sum().item()))
    del keepalive

    class _IpcEngine(FakeEngine):
        def update_weights_from_ipc(self, hs):
            self.opened = [h.name for h in hs]

    local, remote = FakeEngine(seed=0), _IpcEngine(seed=0)
    metrics = sync.sync(trainer, [local, remote])
    assert metrics["weight_sync/tensors"] == len(trainer.weights)
    assert remote.opened == list(trainer.weights)
    assert torch.equal(local.weights["model.embed.weight"].cpu(), trainer.weights["model.embed.weight"].cpu())


# -- registry ---------------------------------------------------------------------------------


def test_registry_factories(make_config, tmp_path: Path):
    cfg = make_config()
    assert registry.names("weight_sync") == ["broadcast", "checkpoint", "cuda_ipc", "inprocess"]
    assert isinstance(registry.get("weight_sync", "inprocess")(cfg), InProcessWeightSync)
    ck = registry.get("weight_sync", "checkpoint")(cfg, sync_dir=str(tmp_path / "ck"))
    assert isinstance(ck, CheckpointWeightSync) and ck.dir == tmp_path / "ck"
    cfg_extra = make_config(cluster={"extra": {"sync_dir": str(tmp_path / "shared")}})
    assert registry.get("weight_sync", "checkpoint")(cfg_extra).dir == tmp_path / "shared"
    with pytest.raises(WeightSyncError, match="BroadcastGroupSpec"):
        registry.get("weight_sync", "broadcast")(cfg)
    spec = BroadcastGroupSpec("127.0.0.1", 1, 2)
    assert isinstance(registry.get("weight_sync", "broadcast")(cfg, spec=spec, rank=1), BroadcastWeightSync)
    if not HAS_CUDA:
        with pytest.raises(WeightSyncError):
            registry.get("weight_sync", "cuda_ipc")(cfg)


def test_inprocess_sync_metrics():
    trainer, engine = FakeTrainer(seed=9), FakeEngine(seed=0)
    metrics = InProcessWeightSync().sync(trainer, [engine])
    assert torch.equal(engine.weights["model.embed.weight"], trainer.weights["model.embed.weight"])
    assert metrics["weight_sync/bytes"] == sum(t.numel() * t.element_size() for t in trainer.weights.values())
