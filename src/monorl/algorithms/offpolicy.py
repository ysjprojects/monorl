"""Off-policy corrections for the policy loss (pure torch): docs/OFF_POLICY.md.

Three policies are involved in an update. The *behaviour* policy ``mu`` sampled the tokens (the
rollout engine's log-probs), the *proximal* policy ``pi_prox`` anchors the clipped ratio
(``TrainBatch.old_logprobs``) and ``pi_theta`` is being optimised. With ``rho = pi_prox / mu``:

* :func:`importance_correction` — the per-token coefficient ``w * keep`` of decoupled PPO
  ``-w * min(r A, clip(r) A)``, ``r = pi_theta / pi_prox`` (Hilton et al., arXiv:2110.00641; AReaL,
  arXiv:2505.24298, Eq. 5). ``w`` is the importance ratio truncated at ``importance_cap`` per token
  (``min(rho_t, cap)``) or per sequence (``min(prod_t rho_t, cap)``), as verl's rollout correction
  computes it (``verl/trainer/ppo/rollout_corr_helper.py``, ``rollout_is``). ``keep`` zeroes the
  tokens or sequences whose ratio leaves ``[ratio_mask_low, ratio_mask_high]``; like IcePop
  (arXiv:2510.18855) and AReaL's rejection mask it leaves the denominator of the loss mean alone,
  so a rejected token contributes zero instead of shrinking the batch.
* :func:`offpolicy_metrics` — per-batch diagnostics of those coefficients and of the gap between
  ``mu`` and ``pi_prox``.
* :func:`score_centering` — the centering term of Marek & Ryabinin (arXiv:2609.20807, Eq. 14) for
  an estimator whose sampled-token score is weighted by ``f(pi_theta / mu)``.

Every coefficient is detached: importance weights change the measure, they are not optimised.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from monorl.config.schema import OffPolicyConfig

SAFETY_BOUND = 20.0
"""Log-ratios are clamped to ``[-20, 20]`` before ``exp`` (verl's ``SAFETY_BOUND``)."""
TAIL_MASS_FLOOR = 1e-6
"""Floor of the head-complement probability mass in score centering (arXiv:2609.20807, A.3)."""

METRIC_KEYS: tuple[str, ...] = (
    "offpolicy/weight_mean",
    "offpolicy/weight_max",
    "offpolicy/truncated_frac",
    "offpolicy/masked_frac",
    "offpolicy/ess",
    "offpolicy/kl",
    "offpolicy/kl_k3",
)


@dataclass(slots=True)
class ImportanceCorrection:
    weights: torch.Tensor
    """``[B, T]`` coefficient of the per-token policy loss: truncated importance weight times the
    ratio mask, zero outside the response mask."""
    truncated: torch.Tensor
    """``[B, T]`` bool: response tokens whose importance weight was cut to the cap."""
    masked: torch.Tensor
    """``[B, T]`` bool: response tokens dropped by the ratio mask."""


def _ratio(log_ratio: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_ratio.clamp(-SAFETY_BOUND, SAFETY_BOUND))


def importance_correction(
    proximal_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    off_policy: OffPolicyConfig,
) -> ImportanceCorrection:
    """Coefficients of decoupled PPO for ``[B, T]`` log-probs of the sampled tokens.

    A sequence is a row: its ratio is the product (``"sequence"``) or geometric mean
    (``"sequence_mean"``) of the ratios of its response tokens. Equal proximal and behaviour
    log-probs give weights of exactly 1 and mask nothing.
    """
    mask = response_mask.bool()
    log_ratio = (proximal_logprobs.detach().float() - behavior_logprobs.detach().float()) * mask
    token_ratio = _ratio(log_ratio)
    seq_log_ratio = log_ratio.sum(dim=-1, keepdim=True)

    if off_policy.importance_weights == "token":
        weights = token_ratio
    elif off_policy.importance_weights == "sequence":
        weights = _ratio(seq_log_ratio).expand_as(log_ratio)
    else:
        weights = torch.ones_like(log_ratio)
    truncated = torch.zeros_like(mask)
    if off_policy.importance_weights != "none" and off_policy.importance_cap is not None:
        truncated = (weights > off_policy.importance_cap) & mask
        weights = weights.clamp(max=off_policy.importance_cap)

    masked = torch.zeros_like(mask)
    if off_policy.ratio_mask != "none":
        if off_policy.ratio_mask == "token":
            stat = token_ratio
        elif off_policy.ratio_mask == "sequence":
            stat = _ratio(seq_log_ratio).expand_as(log_ratio)
        else:
            tokens = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            stat = _ratio(seq_log_ratio / tokens).expand_as(log_ratio)
        masked = ((stat < off_policy.ratio_mask_low) | (stat > off_policy.ratio_mask_high)) & mask
    return ImportanceCorrection(weights=weights * (mask & ~masked), truncated=truncated, masked=masked)


def offpolicy_metrics(
    proximal_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    off_policy: OffPolicyConfig,
) -> dict[str, float]:
    """Diagnostics of one batch (:data:`METRIC_KEYS`), all over its response tokens.

    ``weight_mean``/``weight_max``: the applied coefficients (masked tokens count as 0);
    ``truncated_frac``/``masked_frac``: share of tokens cut to the cap / dropped by the ratio mask;
    ``ess``: effective sample size ``(sum w)^2 / (n sum w^2)`` as a fraction of the tokens (1 when
    all coefficients are equal); ``kl``: ``mean(log mu - log pi_prox)``, the direct estimate of
    ``KL(mu || pi_prox)`` from tokens sampled by ``mu``; ``kl_k3``: its non-negative k3 estimate
    ``mean(rho - 1 - log rho)``.
    """
    correction = importance_correction(proximal_logprobs, behavior_logprobs, response_mask, off_policy)
    mask = response_mask.bool()
    n = max(int(mask.sum().item()), 1)
    weights = correction.weights
    total = float(weights.sum().item())
    squares = float(weights.square().sum().item())
    log_ratio = (proximal_logprobs.detach().float() - behavior_logprobs.detach().float()) * mask
    k3 = (_ratio(log_ratio) - 1.0 - log_ratio.clamp(-SAFETY_BOUND, SAFETY_BOUND)) * mask
    return {
        "offpolicy/weight_mean": total / n,
        "offpolicy/weight_max": float(weights.max().item()) if weights.numel() else 0.0,
        "offpolicy/truncated_frac": float(correction.truncated.sum().item()) / n,
        "offpolicy/masked_frac": float(correction.masked.sum().item()) / n,
        "offpolicy/ess": total * total / (n * squares) if squares > 0.0 else 0.0,
        "offpolicy/kl": -float(log_ratio.sum().item()) / n,
        "offpolicy/kl_k3": float(k3.sum().item()) / n,
    }


@dataclass(slots=True)
class ScoreCentering:
    term: torch.Tensor
    """``[B, T]`` ``sum_v sg[c_v] log p_v`` over the head; add ``advantage * term`` to the
    per-token loss ``-advantage * sg[f(r_y)] * log p_y``."""
    coef_l1: torch.Tensor
    """``[B, T]`` ``sum_v |c_v|``: how far the centering moves the score (0 on-policy)."""
    tail_ratio: torch.Tensor
    """``[B, T]`` sampler-to-trainer ratio of the probability mass outside the head."""


def score_centering(
    top_logprobs: torch.Tensor,
    behavior_top_logprobs: torch.Tensor,
    weight_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> ScoreCentering:
    """Expected weighted score under the sampler, as a loss term (arXiv:2609.20807, Eq. 14).

    ``behavior_top_logprobs`` (``[B, T, K]``) are the sampler's log-probs ``log q_v`` of its ``K``
    most likely next tokens ``H`` at every position (``-inf`` pads positions with fewer entries)
    and ``top_logprobs`` the trainer's log-probs ``log p_v`` of the same tokens, with gradient.
    The sampler's tail is modelled by the trainer's, rescaled by the tail-mass ratio
    ``rho = (1 - sum_H q_v) / (1 - sum_H p_v)`` (Eq. 9, both masses floored at
    :data:`TAIL_MASS_FLOOR`), which leaves the head-only coefficients

        c_v = q_v f(p_v / q_v) - alpha p_v,     alpha = rho f(1 / rho)             (Eq. 12)

    ``weight_fn`` is ``f``, the weight the estimator gives a score as a function of the
    trainer-to-sampler ratio; it receives ``[B, T, K]`` ratios (and ``[B, T, 1]`` for the tail) and
    ``None`` means ``f = 1`` (plain policy gradient, Eq. 8).
    """
    valid = torch.isfinite(behavior_top_logprobs)
    q_logp = behavior_top_logprobs.float().masked_fill(~valid, 0.0)
    p_logp = top_logprobs.float().masked_fill(~valid, 0.0)
    q = torch.exp(q_logp) * valid
    p = torch.exp(p_logp.detach()) * valid
    q_tail = (1.0 - q.sum(dim=-1, keepdim=True)).clamp(min=TAIL_MASS_FLOOR)
    p_tail = (1.0 - p.sum(dim=-1, keepdim=True)).clamp(min=TAIL_MASS_FLOOR)
    rho = q_tail / p_tail
    if weight_fn is None:
        head_weights, tail_weight = torch.ones_like(q), torch.ones_like(rho)
    else:
        head_weights = weight_fn(_ratio(p_logp.detach() - q_logp))
        tail_weight = weight_fn(1.0 / rho)
    coef = (q * head_weights - rho * tail_weight * p).detach()
    return ScoreCentering(
        term=(coef * p_logp).sum(dim=-1),
        coef_l1=coef.abs().sum(dim=-1),
        tail_ratio=rho.squeeze(-1),
    )


def head_mass(behavior_top_logprobs: torch.Tensor, response_mask: torch.Tensor) -> float:
    """Mean probability mass the sampler's recorded top-k covers, over the response tokens."""
    mask = response_mask.bool()
    covered = behavior_top_logprobs.float().exp().sum(dim=-1) * mask
    return float(covered.sum().item()) / max(int(mask.sum().item()), 1)
