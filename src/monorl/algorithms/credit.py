"""Turn-level credit assignment for multi-turn episodes (pure torch, no model code).

Per-turn quantities are ``[N, K]`` tables: one row per trajectory, right-padded to the largest
number of turns ``K`` in the batch, next to a boolean ``mask`` of the turns that exist.

* :func:`turn_table` pads per-trajectory lists into such a table.
* :func:`turn_returns`: the return from every turn on, ``G_k = sum_{j >= k} gamma^(j-k) r_j``.
  Rewards that preceded a turn are not part of its return.
* :func:`turn_advantages`: the estimator of ``AlgorithmConfig.name`` applied per turn index. Turn
  ``k`` of a trajectory is compared with turn ``k`` of the other members of its prompt group; a
  member with fewer turns stands in with its **last** turn, i.e. an episode that has ended keeps
  its final return-to-go. Every comparison therefore has the whole group in it and the baseline of
  a turn depends on that turn's own action exactly as the episode-level estimator's does (not at
  all for ``rloo``, through the group mean and std for ``grpo``/``dapo``). With one turn per
  episode, or with rewards that arrive at the final turn only and ``gamma = 1``, the result equals
  :func:`~monorl.algorithms.advantages.compute_advantages` on the episode rewards.
* :func:`turn_metrics`: how far the advantages of an episode's turns are apart.

The learned baseline of ``ppo`` uses the same tables with
:func:`monorl.algorithms.ppo.gae` (a turn is one step). docs/CREDIT_ASSIGNMENT.md has the formulas.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from monorl.algorithms.advantages import compute_advantages
from monorl.algorithms.ppo import gae
from monorl.config.schema import AlgorithmConfig


def turn_table(per_turn: Sequence[Sequence[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(table, mask)``: ``per_turn[i][k]`` as a zero-padded float ``[N, K]`` tensor and the
    boolean ``[N, K]`` mask of the entries that exist."""
    width = max((len(row) for row in per_turn), default=0)
    table = torch.zeros((len(per_turn), width), dtype=torch.float32)
    mask = torch.zeros((len(per_turn), width), dtype=torch.bool)
    for i, row in enumerate(per_turn):
        table[i, : len(row)] = torch.tensor([float(v) for v in row], dtype=torch.float32)
        mask[i, : len(row)] = True
    return table, mask


def turn_returns(rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted return from every turn on, ``[N, K]``: ``G_k = r_k + gamma * G_{k+1}``, zero
    outside ``mask``. This is GAE without a critic (``V = 0``, ``lam = 1``)."""
    return gae(rewards, torch.zeros_like(rewards, dtype=torch.float32), mask, gamma, 1.0)[0]


def turn_advantages(
    returns: torch.Tensor, mask: torch.Tensor, group_ids: Sequence[str], *, algorithm: AlgorithmConfig
) -> torch.Tensor:
    """Per-turn advantages ``[N, K]`` from per-turn returns ``[N, K]`` (module docstring): for
    every turn index the episode-level estimator over the group's aligned returns, where a
    trajectory without that turn is represented by the return of its last turn. Zero outside
    ``mask``."""
    if returns.ndim != 2 or returns.shape != mask.shape:
        raise ValueError(
            f"returns and mask must share shape [N, K], got {tuple(returns.shape)} and {tuple(mask.shape)}"
        )
    if len(group_ids) != returns.shape[0]:
        raise ValueError(f"group_ids has {len(group_ids)} entries for {returns.shape[0]} trajectories")
    mask = mask.to(torch.bool)
    returns = returns.detach().to(torch.float32)
    if returns.numel() == 0:
        return returns.clone()
    if not bool(mask[:, 0].all()):
        raise ValueError("every trajectory needs at least one turn")
    last = returns.gather(1, (mask.sum(dim=1) - 1)[:, None])
    aligned = torch.where(mask, returns, last)
    columns = [
        compute_advantages(aligned[:, k], group_ids, algorithm=algorithm) for k in range(returns.shape[1])
    ]
    return torch.stack(columns, dim=1) * mask


def turn_metrics(advantages: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Diagnostics of per-turn advantages ``[N, K]``: ``advantage/turn_spread`` (largest minus
    smallest advantage among the turns of an episode, mean over episodes; 0 when every turn of an
    episode is credited alike), ``advantage/turn_spread_max`` (its maximum), and the mean
    magnitude of the advantage of first and of last turns (``advantage/first_turn_abs``,
    ``advantage/last_turn_abs``)."""
    mask = mask.to(torch.bool)
    if advantages.numel() == 0:
        return {}
    advantages = advantages.detach().to(torch.float32)
    high = advantages.masked_fill(~mask, float("-inf")).amax(dim=1)
    low = advantages.masked_fill(~mask, float("inf")).amin(dim=1)
    spread = high - low
    last = advantages.gather(1, (mask.sum(dim=1) - 1).clamp(min=0)[:, None])[:, 0]
    return {
        "advantage/turn_spread": float(spread.mean().item()),
        "advantage/turn_spread_max": float(spread.max().item()),
        "advantage/first_turn_abs": float(advantages[:, 0].abs().mean().item()),
        "advantage/last_turn_abs": float(last.abs().mean().item()),
    }
