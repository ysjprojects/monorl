"""PPO-specific pieces: generalised advantage estimation, the clipped value loss and diagnostics
of the value baseline.

The reference trainer's critic is a value head read at the state of every credited step (the last
prompt token before a turn's completion); the orchestrator runs :func:`gae` over ``[N, K]`` tables
of per-turn rewards and values (:mod:`monorl.algorithms.credit`), one step per turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class ValueLossOutput:
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over right-padded ``[B, T]`` sequences of steps (the
    tokens of a response, or the turns of an episode).

    ``rewards[b, t]`` is the reward received *after* step ``t`` (for tokens typically zero
    everywhere except the final response token), ``values[b, t]`` the critic's estimate of the
    state step ``t`` was taken in, and ``mask`` the steps that exist. Positions outside the mask
    have zero advantage/return and are treated as terminal (``V_{t+1} = 0``) by their predecessor.

    Returns ``(advantages, returns)``, both ``[B, T]`` and detached, with ``returns = adv + values``.
    """
    if rewards.shape != values.shape or rewards.shape != mask.shape:
        raise ValueError("rewards, values and mask must share shape [B, T]")
    rewards = rewards.detach().to(torch.float32)
    values = values.detach().to(torch.float32)
    maskf = mask.to(torch.float32)
    _, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(rewards.shape[0], dtype=rewards.dtype, device=rewards.device)
    for t in range(T - 1, -1, -1):
        next_value = values[:, t + 1] * maskf[:, t + 1] if t + 1 < T else torch.zeros_like(last_adv)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = (delta + gamma * lam * last_adv) * maskf[:, t]
        advantages[:, t] = last_adv
    returns = (advantages + values) * maskf
    return advantages, returns


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor | None,
    clip: float,
    mask: torch.Tensor | None = None,
) -> ValueLossOutput:
    """PPO clipped value loss ``0.5 * max((v - R)^2, (clip(v, v_old ± clip) - R)^2)``.

    ``old_values=None`` or ``clip <= 0`` disables clipping. ``mask`` (``[B, T]``) restricts the
    mean to response positions; without it every position counts.
    """
    if mask is None:
        mask = torch.ones_like(values)
    maskf = mask.to(values.dtype)
    returns = returns.detach()
    err = (values - returns).square()
    if old_values is not None and clip > 0.0:
        old_values = old_values.detach()
        v_clipped = old_values + (values - old_values).clamp(-clip, clip)
        err_clipped = (v_clipped - returns).square()
        per_token = torch.maximum(err, err_clipped)
        clipfrac = ((err_clipped > err).to(values.dtype) * maskf).sum() / maskf.sum().clamp(min=1.0)
    else:
        per_token = err
        clipfrac = values.new_zeros(())
    loss = 0.5 * (per_token * maskf).sum() / maskf.sum().clamp(min=1.0)
    return ValueLossOutput(
        loss=loss, metrics={"loss/value": float(loss.item()), "value/clipfrac": float(clipfrac.item())}
    )


def explained_variance(
    values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """``1 - Var(returns - values) / Var(returns)`` over ``mask`` (everything without it): 1 for a
    perfect value estimate, 0 for one that does no better than a constant, negative for one that
    does worse. ``nan`` when the returns do not vary (fewer than two entries included)."""
    values = values.detach().to(torch.float32)
    returns = returns.detach().to(torch.float32)
    if mask is not None:
        keep = mask.to(torch.bool)
        values, returns = values[keep], returns[keep]
    if returns.numel() < 2:
        return float("nan")
    variance = returns.var(unbiased=False)
    if float(variance.item()) <= 0.0:
        return float("nan")
    return float((1.0 - (returns - values).var(unbiased=False) / variance).item())


def value_metrics(
    values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, float]:
    """``value/mean``, ``value/return_mean`` and, when it is defined, ``value/explained_variance``
    of value estimates against their targets over ``mask``."""
    keep = torch.ones_like(values, dtype=torch.bool) if mask is None else mask.to(torch.bool)
    count = int(keep.sum().item())
    if count == 0:
        return {}
    metrics = {
        "value/mean": float(values.detach()[keep].to(torch.float32).mean().item()),
        "value/return_mean": float(returns.detach()[keep].to(torch.float32).mean().item()),
    }
    explained = explained_variance(values, returns, keep)
    if explained == explained:  # not nan
        metrics["value/explained_variance"] = explained
    return metrics
