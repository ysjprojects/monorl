from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.types import SamplingParams, Trajectory, Turn
from monorl.orchestrator.batching import build_train_batch
from monorl.rollout.hf_engine import HFRolloutEngine
from monorl.trainer.reference import ReferenceTrainer
from tests.trainer.conftest import make_batch, make_config

# The tiny random model is near-uniform over its 49k tokens at temperature 1; at 0.08 its 16 most
# likely tokens carry about 70% of the mass, which gives score centering something to subtract.
PEAKED = 0.08


def _config(
    tiny_model_dir: Path, *, micro_batch_size: int = 2, temperature: float = 1.0, **algorithm: Any
) -> RunConfig:
    cfg = make_config(tiny_model_dir, micro_batch_size=micro_batch_size).to_dict()
    cfg["rollout"]["temperature"] = temperature
    cfg["algorithm"].update(algorithm)
    return RunConfig.from_dict(cfg)


def _step(tiny_model_dir: Path, batch: Any, micro_batch_size: int, **config: Any):
    torch.manual_seed(0)
    trainer = ReferenceTrainer(_config(tiny_model_dir, micro_batch_size=micro_batch_size, **config))
    metrics = trainer.train_step(batch)
    return metrics, trainer.compute_logprobs(batch)


def _stale_batch(tiny_model_dir: Path, tokenizer: Any, *, loss_agg: str):
    """A batch whose proximal log-probs are the trainer's and whose behaviour log-probs are off by
    up to a factor e^0.5 per token, with mixed-sign advantages."""
    batch = make_batch(tokenizer, batch_size=6)
    generator = torch.Generator().manual_seed(1)
    mask = batch.response_mask.float()
    batch.advantages = torch.randn(batch.advantages.shape, generator=generator) * mask
    proximal = ReferenceTrainer(make_config(tiny_model_dir)).compute_logprobs(batch)
    batch.old_logprobs = proximal
    batch.behavior_logprobs = proximal + (torch.rand(proximal.shape, generator=generator) - 0.5) * mask
    if loss_agg == "rollout":  # rows 0-1, 2-3, 4-5 are the sequences of three rollouts
        per_rollout = mask.sum(dim=1).reshape(3, 2).sum(dim=1).repeat_interleave(2)
        batch.loss_weights = mask / per_rollout[:, None]
    return batch


@pytest.mark.parametrize("loss_agg", ["token", "sequence", "rollout"])
def test_micro_batching_matches_single_pass_with_importance_corrections(
    tiny_model_dir: Path, tokenizer, loss_agg: str
) -> None:
    batch = _stale_batch(tiny_model_dir, tokenizer, loss_agg=loss_agg)
    off_policy = {
        "anchor": "trainer",
        "importance_weights": "sequence",
        "importance_cap": 1.3,
        "ratio_mask": "token",
        "ratio_mask_low": 0.8,
        "ratio_mask_high": 1.4,
    }
    (m2, lp2), (m6, lp6) = (
        _step(tiny_model_dir, batch, mbs, loss_agg=loss_agg, off_policy=off_policy) for mbs in (2, 6)
    )
    assert m2["loss/policy"] == pytest.approx(m6["loss/policy"], abs=1e-5)
    assert m2["grad_norm"] == pytest.approx(m6["grad_norm"], rel=1e-4)
    torch.testing.assert_close(lp2, lp6, atol=1e-5, rtol=0)
    # the corrections are live: the same batch without them takes a different step
    plain, _ = _step(tiny_model_dir, batch, 6, loss_agg=loss_agg)
    assert abs(plain["grad_norm"] - m6["grad_norm"]) > 1e-3 * plain["grad_norm"]


def _rollouts(tiny_model_dir: Path, top_k: int):
    engine = HFRolloutEngine(_config(tiny_model_dir, temperature=PEAKED))
    chats = [[{"role": "user", "content": f"Count to {i + 2}."}] for i in range(3)]
    params = SamplingParams(n=2, max_new_tokens=6, temperature=PEAKED, seed=11, top_logprobs=top_k)
    groups = engine.generate(chats, params)
    engine.shutdown()
    return [
        Trajectory(
            sample=None,
            group_id=f"g{i}",
            messages=[*chat, {"role": "assistant", "content": c.text}],
            turns=[Turn(prompt_token_ids=c.prompt_token_ids, completion=c)],
        )
        for i, (chat, group) in enumerate(zip(chats, groups, strict=True))
        for c in group
    ]


def _centering_batch(tiny_model_dir: Path, tokenizer: Any, *, shift: float):
    trajectories = _rollouts(tiny_model_dir, top_k=16)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
    batch, _ = build_train_batch(
        trajectories, advantages, pad_token_id=tokenizer.pad_token_id, max_seq_len=64, loss_agg="token"
    )
    if shift:  # a sampler that disagrees with the trainer on the head probabilities
        generator = torch.Generator().manual_seed(2)
        noise = (torch.rand(batch.behavior_top_logprobs.shape, generator=generator) - 0.5) * shift
        batch.behavior_top_logprobs = batch.behavior_top_logprobs + noise
        batch.old_logprobs = batch.old_logprobs + noise[..., 0] * batch.response_mask
    return batch


def test_score_centering_is_inactive_when_the_engine_sampled_from_the_trainer_policy(
    tiny_model_dir: Path, tokenizer
) -> None:
    # engine top-k ids and log-probs line up with the trainer's distribution at every response
    # position, so every centering coefficient q_v - rho p_v vanishes
    batch = _centering_batch(tiny_model_dir, tokenizer, shift=0.0)
    assert batch.behavior_top_ids.shape == (*batch.input_ids.shape, 16)
    centered, _ = _step(tiny_model_dir, batch, 6, temperature=PEAKED, off_policy={"score_centering": True})
    assert centered["offpolicy/sc_coef_l1"] == pytest.approx(0.0, abs=5e-3)
    assert centered["offpolicy/sc_tail_ratio"] == pytest.approx(1.0, abs=5e-2)
    batch.behavior_top_ids = batch.behavior_top_logprobs = None
    plain, _ = _step(tiny_model_dir, batch, 6, temperature=PEAKED)
    assert centered["grad_norm"] == pytest.approx(plain["grad_norm"], rel=1e-2)


def test_micro_batching_matches_single_pass_with_score_centering(tiny_model_dir: Path, tokenizer) -> None:
    batch = _centering_batch(tiny_model_dir, tokenizer, shift=1.0)
    config = {"temperature": PEAKED, "off_policy": {"score_centering": True}}
    (m2, lp2), (m6, lp6) = (_step(tiny_model_dir, batch, mbs, **config) for mbs in (2, 6))
    assert m6["offpolicy/sc_coef_l1"] > 0.05
    assert m2["loss/policy"] == pytest.approx(m6["loss/policy"], abs=1e-5)
    assert m2["offpolicy/sc_coef_l1"] == pytest.approx(m6["offpolicy/sc_coef_l1"], rel=1e-4)
    assert m2["grad_norm"] == pytest.approx(m6["grad_norm"], rel=1e-4)
    torch.testing.assert_close(lp2, lp6, atol=1e-5, rtol=0)
    batch.behavior_top_ids = batch.behavior_top_logprobs = None
    plain, _ = _step(tiny_model_dir, batch, 6, temperature=PEAKED)
    assert abs(plain["grad_norm"] - m6["grad_norm"]) > 1e-3 * plain["grad_norm"]
