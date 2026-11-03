"""Token-level policy-gradient loss shared by every trainer (pure torch).

``policy_loss`` implements the PPO clipped surrogate with the knobs of
:class:`~monorl.config.schema.AlgorithmConfig`:

* symmetric clipping (``clip_eps``) or DAPO clip-higher (``clip_eps_high``);
* ``loss_agg="token"`` (mean over all response tokens in the micro-batch) or ``"sequence"``
  (per-sequence token mean, then mean over sequences);
* KL penalty to a reference policy (``kl_coef`` × :func:`~monorl.algorithms.kl.kl_penalty`);
* entropy bonus (``entropy_coef``).

With ``old_logprobs=None`` the ratio is identically 1 but keeps its gradient through
``logprobs`` (on-policy REINFORCE / single-epoch GRPO), so clipping never triggers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from monorl.algorithms.kl import kl_penalty
from monorl.config.schema import AlgorithmConfig

METRIC_KEYS: tuple[str, ...] = (
    "loss/policy",
    "loss/kl",
    "loss/entropy",
    "policy/clipfrac",
    "policy/approx_kl",
    "policy/ratio_mean",
)


@dataclass(slots=True)
class PolicyLossOutput:
    loss: torch.Tensor
    """Scalar loss to backpropagate (policy + kl_coef*kl - entropy_coef*entropy)."""
    metrics: dict[str, float] = field(default_factory=dict)


def masked_aggregate(values: torch.Tensor, mask: torch.Tensor, loss_agg: str) -> torch.Tensor:
    """Reduce ``[B, T]`` values over ``mask`` to a scalar; 0 when the mask is empty."""
    mask = mask.to(values.dtype)
    if loss_agg == "token":
        return (values * mask).sum() / mask.sum().clamp(min=1.0)
    if loss_agg == "sequence":
        per_seq_tokens = mask.sum(dim=1)
        per_seq = (values * mask).sum(dim=1) / per_seq_tokens.clamp(min=1.0)
        has_tokens = (per_seq_tokens > 0).to(values.dtype)
        return (per_seq * has_tokens).sum() / has_tokens.sum().clamp(min=1.0)
    raise ValueError(f"unknown loss_agg {loss_agg!r}; expected 'token' or 'sequence'")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(values.dtype)
    return float(((values * mask).sum() / mask.sum().clamp(min=1.0)).item())


def policy_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor | None,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    algorithm: AlgorithmConfig,
    ref_logprobs: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
) -> PolicyLossOutput:
    """Clipped policy-gradient loss over ``[B, T]`` token logprobs (see module docstring).

    ``advantages`` is per-token ``[B, T]`` (a per-sequence ``[B]`` tensor is broadcast).
    Everything is masked by ``response_mask``; an empty mask yields a zero loss that still
    participates in autograd.
    """
    if logprobs.ndim != 2:
        raise ValueError(f"logprobs must be [B, T], got shape {tuple(logprobs.shape)}")
    if advantages.ndim == 1:
        advantages = advantages[:, None].expand_as(logprobs)
    mask = response_mask.to(logprobs.dtype)
    advantages = advantages.detach().to(logprobs.dtype)

    if old_logprobs is None:
        log_ratio = logprobs - logprobs.detach()
    else:
        log_ratio = logprobs - old_logprobs.detach()
    log_ratio = log_ratio * mask  # keep exp() finite on padded positions
    ratio = torch.exp(log_ratio)

    clip_low = 1.0 - algorithm.clip_eps
    clip_high = 1.0 + (algorithm.clip_eps_high if algorithm.clip_eps_high is not None else algorithm.clip_eps)
    unclipped = ratio * advantages
    clipped = ratio.clamp(clip_low, clip_high) * advantages
    pg_tokens = -torch.minimum(unclipped, clipped)
    pg_loss = masked_aggregate(pg_tokens, mask, algorithm.loss_agg)
    total = pg_loss

    metrics: dict[str, float] = {"loss/policy": float(pg_loss.item())}

    if ref_logprobs is not None:
        kl_tokens = kl_penalty(logprobs, ref_logprobs.detach(), algorithm.kl_estimator)
        kl_loss = masked_aggregate(kl_tokens, mask, algorithm.loss_agg)
        metrics["loss/kl"] = float(kl_loss.item())
        if algorithm.kl_coef > 0.0:
            total = total + algorithm.kl_coef * kl_loss
    else:
        metrics["loss/kl"] = 0.0

    if entropy is not None:
        ent_loss = masked_aggregate(entropy, mask, algorithm.loss_agg)
        metrics["loss/entropy"] = float(ent_loss.item())
        if algorithm.entropy_coef > 0.0:
            total = total - algorithm.entropy_coef * ent_loss
    else:
        metrics["loss/entropy"] = 0.0

    with torch.no_grad():
        clipped_active = (clipped < unclipped).to(logprobs.dtype)
        metrics["policy/clipfrac"] = _masked_mean(clipped_active, mask)
        # Schulman k3 estimate of KL(old || new) from the importance ratio.
        metrics["policy/approx_kl"] = _masked_mean((ratio - 1.0) - log_ratio, mask)
        metrics["policy/ratio_mean"] = _masked_mean(ratio, mask)

    return PolicyLossOutput(loss=total, metrics=metrics)
