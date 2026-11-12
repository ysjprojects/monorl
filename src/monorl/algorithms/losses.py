"""Token-level policy-gradient loss shared by every trainer (pure torch).

``policy_loss`` implements the PPO clipped surrogate with the knobs of
:class:`~monorl.config.schema.AlgorithmConfig`:

* symmetric clipping (``clip_eps``) or DAPO clip-higher (``clip_eps_high``);
* ``loss_agg="token"`` (mean over all response tokens in the micro-batch), ``"sequence"``
  (per-sequence token mean, then mean over sequences) or ``"rollout"`` (per-rollout token mean,
  then mean over rollouts, through the batch's ``loss_weights``);
* KL penalty to a reference policy (``kl_coef`` × :func:`~monorl.algorithms.kl.kl_penalty`);
* entropy bonus (``entropy_coef``);
* off-policy corrections of the policy term (``off_policy``, :mod:`monorl.algorithms.offpolicy`):
  importance weights and a ratio mask between the proximal policy (``old_logprobs``) and the
  behaviour policy (``behavior_logprobs``), or score centering over the sampler's top-k tokens.

With ``old_logprobs=None`` the ratio is identically 1 but keeps its gradient through
``logprobs`` (on-policy REINFORCE / single-epoch GRPO), so clipping never triggers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from monorl.algorithms.kl import kl_penalty
from monorl.algorithms.offpolicy import importance_correction, score_centering
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


def masked_aggregate(
    values: torch.Tensor, mask: torch.Tensor, loss_agg: str, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Reduce ``[B, T]`` values over ``mask`` to a scalar; 0 when the mask is empty.

    ``weights`` (``[B, T]``) turn the reduction into a weighted token mean whatever ``loss_agg``
    says; batch builders provide them for ``"rollout"``, where a rollout can span several
    sequences. Without weights every sequence is its own rollout, so ``"rollout"`` reduces like
    ``"sequence"``.
    """
    mask = mask.to(values.dtype)
    if weights is not None:
        weights = weights.to(values.dtype) * mask
        return (values * weights).sum() / weights.sum().clamp(min=1e-8)
    if loss_agg == "token":
        return (values * mask).sum() / mask.sum().clamp(min=1.0)
    if loss_agg in ("sequence", "rollout"):
        per_seq_tokens = mask.sum(dim=1)
        per_seq = (values * mask).sum(dim=1) / per_seq_tokens.clamp(min=1.0)
        has_tokens = (per_seq_tokens > 0).to(values.dtype)
        return (per_seq * has_tokens).sum() / has_tokens.sum().clamp(min=1.0)
    raise ValueError(f"unknown loss_agg {loss_agg!r}; expected 'token', 'sequence' or 'rollout'")


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
    loss_weights: torch.Tensor | None = None,
    behavior_logprobs: torch.Tensor | None = None,
    top_logprobs: torch.Tensor | None = None,
    behavior_top_logprobs: torch.Tensor | None = None,
) -> PolicyLossOutput:
    """Clipped policy-gradient loss over ``[B, T]`` token logprobs (see module docstring).

    ``advantages`` is per-token ``[B, T]`` (a per-sequence ``[B]`` tensor is broadcast).
    Everything is masked by ``response_mask``; an empty mask yields a zero loss that still
    participates in autograd. ``loss_weights`` (``TrainBatch.loss_weights``) weigh the tokens of
    every loss term's mean (:func:`masked_aggregate`).

    ``behavior_logprobs`` (``TrainBatch.behavior_logprobs``) are the sampling log-probs when
    ``old_logprobs`` holds a separate proximal policy: the policy term of every token is then
    multiplied by the ``algorithm.off_policy`` importance weight and ratio mask, which leave the
    denominators of the mean untouched. ``behavior_top_logprobs`` (``[B, T, K]``, the sampler's
    top-k next-token log-probs, ``-inf`` padded) together with ``top_logprobs`` (this policy's
    log-probs of the same tokens, with gradient) add the score-centering term; that needs
    ``old_logprobs`` to be the sampling log-probs, so it excludes ``behavior_logprobs``.
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
    if behavior_logprobs is not None and old_logprobs is not None and algorithm.off_policy.corrects:
        correction = importance_correction(old_logprobs, behavior_logprobs, mask, algorithm.off_policy)
        pg_tokens = pg_tokens * correction.weights.to(pg_tokens.dtype)
    centering = None
    if top_logprobs is not None and behavior_top_logprobs is not None:
        if behavior_logprobs is not None:
            raise ValueError(
                "score centering weighs scores by the ratio to the behaviour policy; it cannot be combined "
                "with a separate proximal policy (behavior_logprobs)"
            )

        def surrogate_weight(r: torch.Tensor) -> torch.Tensor:
            # d/dtheta of -min(r A, clip(r) A) is -A f(r) dlog p with f(r) = r where the clip is inactive.
            adv = advantages[..., None]
            return r * ~(((adv > 0) & (r > clip_high)) | ((adv < 0) & (r < clip_low)))

        centering = score_centering(
            top_logprobs, behavior_top_logprobs, None if old_logprobs is None else surrogate_weight
        )
        pg_tokens = pg_tokens + advantages * centering.term.to(pg_tokens.dtype)
    pg_loss = masked_aggregate(pg_tokens, mask, algorithm.loss_agg, loss_weights)
    total = pg_loss

    metrics: dict[str, float] = {"loss/policy": float(pg_loss.item())}

    if ref_logprobs is not None:
        kl_tokens = kl_penalty(logprobs, ref_logprobs.detach(), algorithm.kl_estimator)
        kl_loss = masked_aggregate(kl_tokens, mask, algorithm.loss_agg, loss_weights)
        metrics["loss/kl"] = float(kl_loss.item())
        if algorithm.kl_coef > 0.0:
            total = total + algorithm.kl_coef * kl_loss
    else:
        metrics["loss/kl"] = 0.0

    if entropy is not None:
        ent_loss = masked_aggregate(entropy, mask, algorithm.loss_agg, loss_weights)
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
        if centering is not None:
            metrics["offpolicy/sc_coef_l1"] = _masked_mean(centering.coef_l1.to(mask.dtype), mask)
            metrics["offpolicy/sc_tail_ratio"] = _masked_mean(centering.tail_ratio.to(mask.dtype), mask)

    return PolicyLossOutput(loss=total, metrics=metrics)
