"""The value head of ``algorithm.name: ppo`` in the reference trainer: it learns the returns a
batch carries, composes with micro-batching, trainer ranks and checkpoints, and never leaves the
trainer with the exported policy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from monorl.algorithms import explained_variance
from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch
from monorl.core.errors import ConfigError
from monorl.trainer.reference import VALUE_LR_SCALE, ReferenceTrainer
from tests.trainer.conftest import make_batch, make_config
from tests.trainer.test_distributed_trainer import WORLD_SIZE, _free_port, _shard

LORA = pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
RETURNS = [1.0, -1.0, 0.5, 2.0]


def _config(tiny_model_dir: Path, *, lora: bool = False, algorithm: dict | None = None, **trainer: Any):
    cfg = make_config(tiny_model_dir, lora=lora, **trainer).to_dict()
    cfg["algorithm"].update({"name": "ppo", **(algorithm or {})})
    return RunConfig.from_dict(cfg)


def _critic_batch(tokenizer: Any, *, returns: list[float] = RETURNS) -> TrainBatch:
    """``make_batch`` plus one value position per row, the last prompt token, with a return that
    differs from row to row, and mixed-sign advantages."""
    batch = make_batch(tokenizer, batch_size=len(returns))
    rows = torch.arange(len(batch))
    state = batch.response_mask.argmax(dim=1) - 1
    batch.value_mask = torch.zeros_like(batch.response_mask)
    batch.value_mask[rows, state] = 1
    batch.returns = torch.zeros(batch.input_ids.shape)
    batch.returns[rows, state] = torch.tensor(returns)
    batch.values = torch.zeros(batch.input_ids.shape)
    generator = torch.Generator().manual_seed(1)
    batch.advantages = torch.randn(batch.advantages.shape, generator=generator) * batch.response_mask
    return batch


def _refresh_values(trainer: ReferenceTrainer, batch: TrainBatch) -> float:
    """What the loop does per rollout batch: the no-grad value pass. Returns the explained variance."""
    values = trainer.compute_values(batch)
    batch.values = values * batch.value_mask
    return explained_variance(values, batch.returns, batch.value_mask)


@LORA
def test_value_head_learns_the_returns(tiny_model_dir: Path, tokenizer, lora: bool) -> None:
    trainer = ReferenceTrainer(_config(tiny_model_dir, lora=lora, learning_rate=1e-3))
    batch = _critic_batch(tokenizer)
    batch.advantages = torch.zeros_like(batch.advantages)
    assert not trainer.compute_values(batch).any()  # zero-initialised: the first advantages are returns
    losses, explained = [], []
    for _ in range(40):
        explained.append(_refresh_values(trainer, batch))
        metrics = trainer.train_step(batch)
        losses.append(metrics["loss/value"])
        assert metrics["value/clipfrac"] == 0.0  # fresh values every step: nothing to clip against
    explained.append(_refresh_values(trainer, batch))
    # 0.5 * mean((0 - R)^2) at the start
    assert losses[0] == pytest.approx(0.5 * sum(r * r for r in RETURNS) / len(RETURNS))
    assert losses[-1] < 0.25 * losses[0]
    assert explained[0] == pytest.approx(0.0, abs=1e-6) and explained[-1] > 0.75
    assert metrics["loss/total"] == pytest.approx(
        metrics["loss/policy"] + 0.5 * metrics["loss/value"], abs=1e-5
    )
    assert metrics["lr"] == pytest.approx(1e-3)  # the policy's rate; the head trains VALUE_LR_SCALE faster
    assert [g["lr"] for g in trainer.optimizer.param_groups] == pytest.approx([1e-3, 1e-3 * VALUE_LR_SCALE])


def test_values_are_read_at_the_state_and_do_not_see_the_action(tiny_model_dir: Path, tokenizer) -> None:
    trainer = ReferenceTrainer(_config(tiny_model_dir))
    batch = _critic_batch(tokenizer)
    for _ in range(3):
        trainer.train_step(batch)
    values = trainer.compute_values(batch)
    assert values.shape == batch.input_ids.shape and values.dtype == torch.float32
    assert torch.all(values[batch.attention_mask == 0] == 0)
    # other response tokens and padding to the right leave the value of the last prompt token alone
    changed = batch.slice(0, len(batch))
    changed.input_ids = batch.input_ids.clone()
    changed.input_ids[batch.response_mask == 1] = 4321
    states = batch.value_mask.bool()
    torch.testing.assert_close(trainer.compute_values(changed)[states], values[states], atol=1e-5, rtol=0)


@pytest.mark.parametrize("loss_agg", ["token", "sequence", "rollout"])
def test_micro_batched_update_equals_the_single_pass_update(
    tiny_model_dir: Path, tokenizer, loss_agg: str
) -> None:
    """Rows hold different numbers of value positions, so a micro-batch's value loss has to be
    weighed by its share of the batch's positions."""
    batch = _critic_batch(tokenizer, returns=[1.0, -1.0, 0.5, 2.0, -0.5, 3.0])
    batch.value_mask[0, 2] = 1  # a second turn's state in rows 0 and 3
    batch.value_mask[3, 1] = 1
    batch.returns[0, 2], batch.returns[3, 1] = -2.0, 0.25
    batch.values = torch.full(batch.input_ids.shape, 0.9) * batch.value_mask  # far enough to clip some
    if loss_agg == "rollout":
        mask = batch.response_mask.float()
        per_rollout = mask.sum(dim=1).reshape(3, 2).sum(dim=1).repeat_interleave(2)
        batch.loss_weights = mask / per_rollout[:, None]

    def step(micro_batch_size: int):
        torch.manual_seed(0)
        config = _config(tiny_model_dir, algorithm={"loss_agg": loss_agg}, micro_batch_size=micro_batch_size)
        trainer = ReferenceTrainer(config)
        trainer.train_step(batch)  # moves the head off zero so that the second step has a trunk gradient
        metrics = trainer.train_step(batch)
        return metrics, trainer.compute_values(batch), trainer.compute_logprobs(batch)

    split, split_values, split_logprobs = step(1)
    whole, whole_values, whole_logprobs = step(6)
    for key in ("loss/value", "value/clipfrac", "loss/policy", "loss/total", "grad_norm"):
        assert split[key] == pytest.approx(whole[key], rel=1e-4, abs=1e-6), key
    assert 0.0 < whole["value/clipfrac"] < 1.0
    torch.testing.assert_close(split_values, whole_values, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(split_logprobs, whole_logprobs, atol=1e-5, rtol=1e-5)


def test_value_loss_is_clipped_against_the_values_of_the_batch(tiny_model_dir: Path, tokenizer) -> None:
    batch = _critic_batch(tokenizer)
    batch.values = torch.full(batch.input_ids.shape, 5.0) * batch.value_mask  # the head predicts 0

    def loss(value_clip: float | None) -> dict[str, float]:
        return ReferenceTrainer(_config(tiny_model_dir, algorithm={"value_clip": value_clip})).train_step(
            batch
        )

    plain = 0.5 * sum(r * r for r in RETURNS) / len(RETURNS)
    unclipped = loss(None)
    assert unclipped["loss/value"] == pytest.approx(plain) and unclipped["value/clipfrac"] == 0.0
    # the prediction clipped to 5 - 0.5 is further from every return than the prediction itself
    clipped = loss(0.5)
    assert clipped["loss/value"] == pytest.approx(0.5 * sum((4.5 - r) ** 2 for r in RETURNS) / len(RETURNS))
    assert clipped["value/clipfrac"] == 1.0


@LORA
def test_checkpoint_restores_the_value_head_and_the_export_is_the_policy_alone(
    tiny_model_dir: Path, tokenizer, tmp_path: Path, lora: bool
) -> None:
    from transformers import AutoModelForCausalLM

    config = _config(tiny_model_dir, lora=lora, learning_rate=1e-3)
    trainer = ReferenceTrainer(config)
    batch = _critic_batch(tokenizer)
    for _ in range(3):
        _refresh_values(trainer, batch)
        trainer.train_step(batch)
    values = trainer.compute_values(batch)
    assert values[batch.value_mask.bool()].abs().min() > 0
    trainer.save_checkpoint(tmp_path / "ckpt")

    restored = ReferenceTrainer(config)
    assert not restored.compute_values(batch).any()
    restored.load_checkpoint(tmp_path / "ckpt")
    torch.testing.assert_close(restored.compute_values(batch), values, atol=1e-6, rtol=0)
    # the optimizer state of the head came back too: both take the same next step
    for t in (trainer, restored):
        _refresh_values(t, batch)
        t.train_step(batch)
    torch.testing.assert_close(
        restored.compute_values(batch), trainer.compute_values(batch), atol=1e-5, rtol=0
    )

    # a trainer without a critic exports the same parameter names; the saved model is a plain causal LM
    plain = make_config(tiny_model_dir, lora=lora)
    assert sorted(dict(trainer.export_weights())) == sorted(dict(ReferenceTrainer(plain).export_weights()))
    trainer.save_pretrained(tmp_path / "final")
    exported = AutoModelForCausalLM.from_pretrained(tmp_path / "final")
    assert not [name for name in exported.state_dict() if "value" in name]
    logits = exported(input_ids=batch.input_ids[:1], attention_mask=batch.attention_mask[:1]).logits
    expected = trainer.compute_logprobs(batch)[0]
    actual = torch.log_softmax(logits[0, :-1], dim=-1).gather(-1, batch.input_ids[0, 1:, None])[:, 0]
    torch.testing.assert_close(actual * batch.response_mask[0, 1:], expected[1:], atol=1e-4, rtol=0)

    # a checkpoint of a run without a critic cannot resume a ppo run
    ReferenceTrainer(plain).save_checkpoint(tmp_path / "plain")
    with pytest.raises(RuntimeError, match="no value head"):
        ReferenceTrainer(config).load_checkpoint(tmp_path / "plain")


def test_batches_without_value_targets_and_trainers_without_a_critic(tiny_model_dir: Path, tokenizer) -> None:
    batch = _critic_batch(tokenizer)
    trainer = ReferenceTrainer(_config(tiny_model_dir))
    plain_batch = make_batch(tokenizer)
    metrics = trainer.train_step(plain_batch)  # no returns: a policy step only
    assert "loss/value" not in metrics and not trainer.compute_values(batch).any()

    no_critic = ReferenceTrainer(make_config(tiny_model_dir))
    assert no_critic.value_head is None and len(no_critic.optimizer.param_groups) == 1
    assert "loss/value" not in no_critic.train_step(batch)  # value fields are ignored without a head
    with pytest.raises(ConfigError, match="no value head"):
        no_critic.compute_values(batch)


def _rank_worker(rank: int, port: int, model_dir: str, batch_path: str, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.manual_seed(0)
    trainer = ReferenceTrainer(_config(Path(model_dir), micro_batch_size=1), rank=rank, world_size=WORLD_SIZE)
    batch: TrainBatch = torch.load(batch_path, weights_only=False)
    shard = _shard(batch, rank, WORLD_SIZE)
    trainer.train_step(shard)
    metrics = trainer.train_step(shard)
    trainer.save_checkpoint(Path(out_dir) / "ckpt")
    torch.save(
        {
            "metrics": metrics,
            "values": trainer.compute_values(batch),
            "head": {k: v.detach().clone() for k, v in trainer.value_head.state_dict().items()},
        },
        Path(out_dir) / f"rank{rank}.pt",
    )
    trainer.shutdown()


@pytest.mark.timeout(300)
def test_two_rank_step_matches_single_process(tiny_model_dir: Path, tokenizer, tmp_path: Path) -> None:
    """Rows 0-1 go to rank 0 and row 2 to rank 1; row 0 has two value positions, so the ranks hold
    3 and 1 of them and only the global count reproduces the full-batch value gradient."""
    batch = _critic_batch(tokenizer, returns=[1.0, -1.0, 0.5])
    batch.value_mask[0, 2] = 1
    batch.returns[0, 2] = -2.0
    batch_path = tmp_path / "batch.pt"
    torch.save(batch, batch_path)
    mp.spawn(
        _rank_worker,
        args=(_free_port(), str(tiny_model_dir), str(batch_path), str(tmp_path)),
        nprocs=WORLD_SIZE,
        join=True,
    )
    ranks = [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(WORLD_SIZE)]

    torch.manual_seed(0)
    single = ReferenceTrainer(_config(tiny_model_dir, micro_batch_size=1))
    single.train_step(batch)
    expected = single.train_step(batch)
    for r, data in enumerate(ranks):
        for key in ("loss/value", "value/clipfrac", "loss/policy", "loss/total", "grad_norm"):
            assert data["metrics"][key] == pytest.approx(expected[key], abs=1e-5), f"rank {r} {key}"
        torch.testing.assert_close(data["values"], single.compute_values(batch), atol=1e-5, rtol=0)
        for name, tensor in single.value_head.state_dict().items():
            torch.testing.assert_close(data["head"][name], tensor, atol=1e-6, rtol=0, msg=f"rank {r} {name}")
    restored = ReferenceTrainer(_config(tiny_model_dir, micro_batch_size=1))
    restored.load_checkpoint(tmp_path / "ckpt")
    torch.testing.assert_close(restored.compute_values(batch), ranks[0]["values"], atol=1e-6, rtol=0)
