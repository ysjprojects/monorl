"""Two-rank data-parallel :class:`ReferenceTrainer` (gloo, spawned CPU processes) must match a
single-process step over the whole batch: the Ray runtime shards each ``TrainBatch`` contiguously
across trainer ranks, so a rank's gradient is only a partial sum until it is all-reduced."""

from __future__ import annotations

import math
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch
from monorl.trainer.reference import ReferenceTrainer
from tests.trainer.conftest import make_batch, make_config, reinforce_loss

WORLD_SIZE = 2
SEED = 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _config(model_dir: Path, lora: bool, loss_agg: str) -> RunConfig:
    # micro_batch_size=1 so every rank's micro-batch partition matches the single-process one.
    config = make_config(model_dir, lora=lora, micro_batch_size=1)
    config.algorithm.loss_agg = loss_agg
    return config


def _shard(batch: TrainBatch, rank: int, world_size: int) -> TrainBatch:
    """Same contiguous split as ``RayTrainer._shards`` (3 sequences → ``[0:2]``, ``[2:3]``)."""
    per = math.ceil(len(batch) / world_size)
    return batch.slice(rank * per, min(len(batch), (rank + 1) * per))


def _trainable(trainer: ReferenceTrainer) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}


def _worker(rank: int, port: int, model_dir: str, lora: bool, loss_agg: str, batch_path: str, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.manual_seed(SEED)
    trainer = ReferenceTrainer(_config(Path(model_dir), lora, loss_agg), loss_fn=reinforce_loss, rank=rank, world_size=WORLD_SIZE)
    assert trainer.rank == rank and trainer.world_size == WORLD_SIZE
    batch: TrainBatch = torch.load(batch_path, weights_only=False)
    metrics = trainer.train_step(_shard(batch, rank, WORLD_SIZE))
    logprobs = trainer.compute_logprobs(batch)
    out = Path(out_dir)
    trainer.save_checkpoint(out / "ckpt")  # rank 0 writes, every rank waits at the barrier
    assert (out / "ckpt" / "trainer_state.pt").exists()
    trainer.save_pretrained(out / "export")
    assert (out / "export" / "config.json").exists()
    exported = dict(trainer.export_weights())
    torch.save(
        {"params": _trainable(trainer), "metrics": metrics, "logprobs": logprobs, "export_keys": sorted(exported)},
        out / f"rank{rank}.pt",
    )
    trainer.shutdown()


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    ("lora", "loss_agg"),
    [(False, "token"), (True, "token"), (False, "sequence")],
    ids=["full-token", "lora-token", "full-sequence"],
)
def test_two_rank_step_matches_single_process(tiny_model_dir: Path, tokenizer, tmp_path: Path, lora: bool, loss_agg: str) -> None:
    # 3 sequences with 6/5/6 response tokens (incl. EOS): the shards are uneven in both sequences
    # and tokens, so only global loss normalisation reproduces the full-batch gradient.
    batch = make_batch(tokenizer, batch_size=3)
    assert [int(m.sum()) for m in batch.response_mask] == [6, 5, 6]
    batch_path = tmp_path / "batch.pt"
    torch.save(batch, batch_path)

    mp.spawn(
        _worker,
        args=(_free_port(), str(tiny_model_dir), lora, loss_agg, str(batch_path), str(tmp_path)),
        nprocs=WORLD_SIZE,
        join=True,
    )
    ranks = [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(WORLD_SIZE)]

    torch.manual_seed(SEED)
    single = ReferenceTrainer(_config(tiny_model_dir, lora, loss_agg), loss_fn=reinforce_loss)
    assert single.world_size == 1 and single._dist is None
    expected_metrics = single.train_step(batch)
    expected = _trainable(single)
    assert expected and (all("lora_" in n for n in expected) if lora else True)

    for r, data in enumerate(ranks):
        assert list(data["params"]) == list(expected), f"rank {r} trainable parameter set"
        for name, tensor in expected.items():
            torch.testing.assert_close(data["params"][name], tensor, atol=1e-6, rtol=0, msg=f"rank {r} {name}")
        for key in ("loss/policy", "loss/total", "loss/entropy", "grad_norm", "lr", "tokens/response_mean"):
            assert data["metrics"][key] == pytest.approx(expected_metrics[key], abs=1e-5), f"rank {r} {key}"
        torch.testing.assert_close(data["logprobs"], single.compute_logprobs(batch), atol=1e-5, rtol=0)
        assert data["export_keys"] == sorted(dict(single.export_weights()))
    torch.testing.assert_close(ranks[0]["params"], ranks[1]["params"], atol=0, rtol=0)
    assert (tmp_path / "ckpt" / "trainer_state.pt").exists()


def test_env_rank_resolution_and_validation(tiny_model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    trainer = ReferenceTrainer(make_config(tiny_model_dir), loss_fn=reinforce_loss)
    assert (trainer.rank, trainer.world_size) == (0, 1) and trainer._dist is None
    trainer.shutdown()  # no group to leave
    with pytest.raises(ValueError, match="rank"):
        ReferenceTrainer(make_config(tiny_model_dir), loss_fn=reinforce_loss, rank=2, world_size=2)
