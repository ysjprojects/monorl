"""Reward and episode statistics of scored trajectories, shared by the RL loop, loop evaluation
and the pipeline sampling stages. Torch-free.

* :func:`reward_metrics` — ``reward/mean``, ``reward/std`` and ``reward/<name>`` per breakdown key.
* :func:`episode_metrics` — unprefixed agent statistics (success rate, pass@k, turns, context
  size, tool use, ...).
* :func:`step_metrics` — what a training step reports: the reward metrics plus the episode
  metrics as ``reward/success_rate``, ``reward/pass_at_k`` and ``episode/<key>``.
* :func:`eval_metrics` — what an evaluation reports: everything under ``eval/``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from monorl.core.reporting import num_context_tokens
from monorl.core.types import Trajectory

_REWARD_SCOPED = ("success_rate", "pass_at_k")


def reward_metrics(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    rewards = [float(t.reward) for t in trajectories]
    n = len(rewards)
    if not n:
        return {"reward/mean": 0.0, "reward/std": 0.0}
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    metrics = {"reward/mean": mean, "reward/std": math.sqrt(var)}
    totals: dict[str, float] = {}
    for t in trajectories:
        for k, v in t.reward_breakdown.items():
            totals[k] = totals.get(k, 0.0) + float(v)
    for k, total in totals.items():
        # Keys present on a subset of trajectories (env return, overlong penalty) average over all
        # trajectories, counting absent entries as 0.
        metrics[f"reward/{k}"] = total / n
    return metrics


def episode_metrics(trajectories: Sequence[Trajectory], *, success_threshold: float) -> dict[str, float]:
    """Agent statistics over scored episodes (empty input → ``{}``).

    ``success_rate`` counts episodes with ``reward >= success_threshold``; ``pass_at_k`` is the
    fraction of ``group_id`` groups with at least one success and is reported only when some group
    has two or more members. ``tool_calls_mean`` / ``invalid_calls_mean`` are per-episode sums of
    what the environment put in ``Turn.info`` and appear only when some turn reports them;
    ``threads_mean`` only when some episode has more than one conversation.
    """
    n = len(trajectories)
    if not n:
        return {}
    solved = [float(t.reward) >= success_threshold for t in trajectories]
    metrics = {"success_rate": sum(solved) / n}

    groups: dict[str, bool] = {}
    sizes: dict[str, int] = {}
    for t, ok in zip(trajectories, solved, strict=True):
        groups[t.group_id] = groups.get(t.group_id, False) or ok
        sizes[t.group_id] = sizes.get(t.group_id, 0) + 1
    if max(sizes.values()) > 1:
        metrics["pass_at_k"] = sum(groups.values()) / len(groups)

    turns = [len(t.turns) for t in trajectories]
    metrics["turns_mean"] = sum(turns) / n
    metrics["turns_max"] = float(max(turns))
    metrics["done_frac"] = sum(1 for t in trajectories if t.done) / n
    metrics["truncated_frac"] = (
        sum(1 for t in trajectories if t.turns and t.turns[-1].completion.finish_reason == "length") / n
    )
    context = [num_context_tokens(t) for t in trajectories]
    metrics["context_tokens_mean"] = sum(context) / n
    metrics["context_tokens_max"] = float(max(context))

    for key in ("tool_calls", "invalid_calls"):
        if any(key in turn.info for t in trajectories for turn in t.turns):
            total = sum(_count(turn.info.get(key)) for t in trajectories for turn in t.turns)
            metrics[f"{key}_mean"] = total / n
    if any(t.extra_threads for t in trajectories):
        metrics["threads_mean"] = sum(len(t.threads) for t in trajectories) / n
    # stepped episodes that ran out of context, and agents that were refused a call for it
    exhausted = sum(
        1 for t in trajectories if t.info.get("context_exhausted") or t.info.get("context_exceeded_calls")
    )
    if exhausted:
        metrics["context_exhausted_frac"] = exhausted / n
    return metrics


def step_metrics(trajectories: Sequence[Trajectory], *, success_threshold: float) -> dict[str, float]:
    metrics = reward_metrics(trajectories)
    for k, v in episode_metrics(trajectories, success_threshold=success_threshold).items():
        metrics[f"reward/{k}" if k in _REWARD_SCOPED else f"episode/{k}"] = v
    return metrics


def eval_metrics(trajectories: Sequence[Trajectory], *, success_threshold: float) -> dict[str, float]:
    """``eval/reward_mean``, ``eval/reward_std``, ``eval/reward/<name>``, ``eval/<episode metric>``
    and ``eval/num_samples`` (episodes scored)."""
    rewards = reward_metrics(trajectories)
    metrics = {
        "eval/reward_mean": rewards.pop("reward/mean"),
        "eval/reward_std": rewards.pop("reward/std"),
    }
    metrics.update({f"eval/{k}": v for k, v in rewards.items()})
    episodes = episode_metrics(trajectories, success_threshold=success_threshold)
    metrics.update({f"eval/{k}": v for k, v in episodes.items()})
    metrics["eval/num_samples"] = float(len(trajectories))
    return metrics


def _count(value: Any) -> float:
    """A reported count is either a number or the collection of the things counted."""
    if isinstance(value, bool) or value is None:
        return float(bool(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(len(value))
    except TypeError:
        return 0.0


__all__ = ["episode_metrics", "eval_metrics", "reward_metrics", "step_metrics"]
