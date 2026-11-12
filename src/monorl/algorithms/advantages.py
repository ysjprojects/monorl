"""Sequence-level advantage estimators and reward shaping (pure torch, no model code).

``compute_advantages`` dispatches on ``AlgorithmConfig.name``:

* ``grpo``/``dapo``: rewards normalised per ``advantage_norm`` (``group`` = per prompt group,
  ``batch`` = across the batch, ``none`` = raw). ``advantage_std_norm=False`` keeps the mean
  subtraction but skips the division (Dr. GRPO).
* ``rloo``: leave-one-out baseline — each reward minus the mean of the *other* members of its
  group (0 baseline for singleton groups).
* ``reinforce``: raw rewards.
* ``ppo``: raw rewards; the advantages come from :func:`monorl.algorithms.ppo.gae` over the
  value head's estimates.

This is the estimator of ``algorithm.credit_assignment: episode``, where every token of an episode
carries its episode's advantage. :mod:`monorl.algorithms.credit` applies the same estimators per
turn (``credit_assignment: turn``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from monorl.config.schema import AlgorithmConfig
from monorl.core.types import FinishReason

EPS = 1e-6


def _group_index(group_ids: Sequence[str], device: torch.device) -> tuple[torch.Tensor, int]:
    """Map group id strings to dense integer indices (first-seen order)."""
    lookup: dict[str, int] = {}
    idx = [lookup.setdefault(g, len(lookup)) for g in group_ids]
    return torch.tensor(idx, dtype=torch.long, device=device), len(lookup)


def _group_stats(
    values: torch.Tensor, group_ids: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-element group ``(count, mean, unbiased std)`` broadcast back to ``[B]``."""
    idx, n_groups = _group_index(group_ids, values.device)
    ones = torch.ones_like(values)
    count = torch.zeros(n_groups, dtype=values.dtype, device=values.device).scatter_add_(0, idx, ones)
    total = torch.zeros_like(count).scatter_add_(0, idx, values)
    mean = total / count
    sq = torch.zeros_like(count).scatter_add_(0, idx, (values - mean[idx]).square())
    # Unbiased std (matches torch.std / TRL / verl); singleton groups get std 0.
    std = torch.where(count > 1, (sq / (count - 1).clamp(min=1)).sqrt(), torch.zeros_like(sq))
    return count[idx], mean[idx], std[idx]


def compute_advantages(
    rewards: torch.Tensor, group_ids: Sequence[str], *, algorithm: AlgorithmConfig
) -> torch.Tensor:
    """Sequence-level advantages ``[B]`` from scalar rewards ``[B]`` (see module docstring)."""
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be [B], got shape {tuple(rewards.shape)}")
    if len(group_ids) != rewards.shape[0]:
        raise ValueError(f"group_ids has {len(group_ids)} entries for {rewards.shape[0]} rewards")
    rewards = rewards.detach().to(torch.float32)
    if rewards.numel() == 0:
        return rewards.clone()

    name = algorithm.name
    if name in ("reinforce", "ppo"):
        return rewards.clone()
    if name == "rloo":
        count, mean, _ = _group_stats(rewards, group_ids)
        baseline = torch.where(
            count > 1, (mean * count - rewards) / (count - 1).clamp(min=1), torch.zeros_like(rewards)
        )
        return rewards - baseline

    # grpo / dapo
    norm = algorithm.advantage_norm
    if norm == "none":
        return rewards.clone()
    if norm == "group":
        _, mean, std = _group_stats(rewards, group_ids)
    elif norm == "batch":
        mean = rewards.mean().expand_as(rewards)
        std = (rewards.std(unbiased=True) if rewards.numel() > 1 else rewards.new_zeros(())).expand_as(
            rewards
        )
    else:
        raise ValueError(f"unknown advantage_norm {norm!r}")
    adv = rewards - mean
    if algorithm.advantage_std_norm:
        adv = adv / (std + EPS)
    return adv


def dynamic_sampling_mask(rewards: torch.Tensor, group_ids: Sequence[str]) -> torch.Tensor:
    """DAPO dynamic sampling: boolean ``[B]`` keeping only members of groups whose rewards are
    not all identical (zero-variance groups carry no learning signal)."""
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be [B], got shape {tuple(rewards.shape)}")
    if rewards.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=rewards.device)
    rewards = rewards.detach().to(torch.float32)
    idx, n_groups = _group_index(group_ids, rewards.device)
    gmax = torch.full((n_groups,), float("-inf"), device=rewards.device).scatter_reduce(
        0, idx, rewards, "amax"
    )
    gmin = torch.full((n_groups,), float("inf"), device=rewards.device).scatter_reduce(
        0, idx, rewards, "amin"
    )
    return (gmax - gmin)[idx] > 0


def apply_overlong_penalty(
    rewards: torch.Tensor, finish_reasons: Sequence[FinishReason], penalty: float
) -> torch.Tensor:
    """Subtract ``penalty`` from the reward of every completion that was cut off by the token
    limit (``finish_reason == "length"``). Returns a new tensor."""
    if len(finish_reasons) != rewards.shape[0]:
        raise ValueError(f"{len(finish_reasons)} finish_reasons for {rewards.shape[0]} rewards")
    if penalty == 0.0:
        return rewards.clone()
    truncated = torch.tensor(
        [fr == "length" for fr in finish_reasons], dtype=rewards.dtype, device=rewards.device
    )
    return rewards - penalty * truncated


def discount_turn_rewards(turn_rewards: Sequence[float], gamma: float) -> float:
    """Discounted return of a multi-turn episode: ``sum_t gamma**t * r_t`` (turn 0 undiscounted)."""
    total = 0.0
    weight = 1.0
    for r in turn_rewards:
        total += weight * float(r)
        weight *= gamma
    return total
