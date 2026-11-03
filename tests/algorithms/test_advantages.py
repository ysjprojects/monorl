from __future__ import annotations

import pytest
import torch

from monorl.algorithms import (
    apply_overlong_penalty,
    compute_advantages,
    discount_turn_rewards,
    dynamic_sampling_mask,
)
from monorl.config.schema import AlgorithmConfig

REWARDS = torch.tensor([1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 5.0])
GROUPS = ["a", "a", "a", "b", "b", "b", "c"]


def _per_group(values: torch.Tensor, groups: list[str]) -> dict[str, torch.Tensor]:
    return {g: values[[i for i, gi in enumerate(groups) if gi == g]] for g in dict.fromkeys(groups)}


def test_group_normalised_advantages_are_zero_mean_unit_std_per_group():
    adv = compute_advantages(REWARDS, GROUPS, algorithm=AlgorithmConfig(name="grpo"))
    for g, vals in _per_group(adv, GROUPS).items():
        assert abs(vals.mean().item()) < 1e-5, g
        if vals.numel() > 1:
            assert abs(vals.std().item() - 1.0) < 1e-3, g
    # singleton group → zero advantage, no NaN
    assert adv[6].item() == 0.0
    assert torch.isfinite(adv).all()


def test_group_norm_without_std_keeps_mean_subtraction_only():
    cfg = AlgorithmConfig(name="grpo", advantage_std_norm=False)
    adv = compute_advantages(REWARDS, GROUPS, algorithm=cfg)
    expected = torch.tensor([2 / 3, -1 / 3, -1 / 3, -1.0, 0.0, 1.0, 0.0])
    torch.testing.assert_close(adv, expected)


def test_batch_and_none_normalisation():
    cfg = AlgorithmConfig(name="dapo", advantage_norm="batch")
    adv = compute_advantages(REWARDS, GROUPS, algorithm=cfg)
    assert abs(adv.mean().item()) < 1e-5
    assert abs(adv.std().item() - 1.0) < 1e-3
    raw = compute_advantages(REWARDS, GROUPS, algorithm=AlgorithmConfig(name="grpo", advantage_norm="none"))
    torch.testing.assert_close(raw, REWARDS)


def test_rloo_leave_one_out_baseline_hand_example():
    rewards = torch.tensor([1.0, 2.0, 6.0, 4.0])
    groups = ["g", "g", "g", "solo"]
    adv = compute_advantages(rewards, groups, algorithm=AlgorithmConfig(name="rloo"))
    # baselines: (2+6)/2=4, (1+6)/2=3.5, (1+2)/2=1.5; singleton baseline 0
    torch.testing.assert_close(adv, torch.tensor([-3.0, -1.5, 4.5, 4.0]))


def test_reinforce_and_ppo_return_raw_rewards():
    for name in ("reinforce", "ppo"):
        adv = compute_advantages(REWARDS, GROUPS, algorithm=AlgorithmConfig(name=name))
        torch.testing.assert_close(adv, REWARDS)
        assert adv.data_ptr() != REWARDS.data_ptr()


def test_compute_advantages_validates_inputs_and_handles_empty():
    cfg = AlgorithmConfig(name="grpo")
    with pytest.raises(ValueError):
        compute_advantages(REWARDS, GROUPS[:-1], algorithm=cfg)
    with pytest.raises(ValueError):
        compute_advantages(REWARDS[None], GROUPS, algorithm=cfg)
    assert compute_advantages(torch.zeros(0), [], algorithm=cfg).shape == (0,)


def test_dynamic_sampling_mask_drops_zero_variance_groups():
    rewards = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.5])
    groups = ["same", "same", "mixed", "mixed", "zero", "zero", "solo"]
    keep = dynamic_sampling_mask(rewards, groups)
    assert keep.tolist() == [False, False, True, True, False, False, False]
    assert keep.dtype == torch.bool


def test_apply_overlong_penalty():
    rewards = torch.tensor([1.0, 1.0, 0.0])
    out = apply_overlong_penalty(rewards, ["stop", "length", "length"], 0.5)
    torch.testing.assert_close(out, torch.tensor([1.0, 0.5, -0.5]))
    torch.testing.assert_close(rewards, torch.tensor([1.0, 1.0, 0.0]))  # input untouched
    with pytest.raises(ValueError):
        apply_overlong_penalty(rewards, ["stop"], 0.5)


def test_discount_turn_rewards():
    assert discount_turn_rewards([1.0, 1.0, 1.0], 0.5) == pytest.approx(1.75)
    assert discount_turn_rewards([0.0, 0.0, 2.0], 1.0) == pytest.approx(2.0)
    assert discount_turn_rewards([], 0.9) == 0.0
