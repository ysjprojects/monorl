"""Supervised fine-tuning objective: masked next-token negative log-likelihood (pure torch).

``sft_loss`` consumes the same ``[B, T]`` token logprobs the trainers already compute for the
policy-gradient losses, so SFT runs through the unchanged ``train_step`` micro-batching: the
trainer selects :func:`sft_policy_loss` (same call signature as
:func:`~monorl.algorithms.losses.policy_loss`) via :func:`select_loss_fn`.

Label smoothing follows the HF ``LabelSmoother`` definition::

    loss_t = (1 - eps) * (-log p_t[y_t]) + eps * (-(1/V) * sum_v log p_t[v])

The smoothing term only needs the vocabulary-mean logprob per position, a ``[B, T]`` tensor
(:func:`mean_logprob_all_tokens`), which the trainer derives from the logits it already holds
instead of keeping the full ``[B, T, V]`` log-softmax alive for the loss; passing ``logits``
directly is supported for callers that have them.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from monorl.algorithms.losses import PolicyLossOutput, masked_aggregate, policy_loss
from monorl.config.schema import AlgorithmConfig

SFT_METRIC_KEYS: tuple[str, ...] = ("loss/sft", "sft/ppl", "sft/tokens")

_PPL_MAX_NLL = 80.0  # exp(80) is the largest perplexity worth reporting; beyond that it overflows


def _vocab_mean_logprobs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled = logits[:, :-1].float() / temperature
    mean = scaled.mean(dim=-1) - torch.logsumexp(scaled, dim=-1)
    return torch.nn.functional.pad(mean, (1, 0))


def mean_logprob_all_tokens(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """``[B, T]`` vocabulary-mean log-probability of the predictive distribution at every position
    (position 0 is 0), aligned like the trainer's token logprobs (prediction for ``input_ids[:, t]``
    comes from ``logits[:, t-1]``)."""
    return _vocab_mean_logprobs(logits, temperature)


def sft_loss(
    logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    loss_agg: str = "token",
    mean_logprob_all_tokens: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> PolicyLossOutput:
    """Masked NLL over ``[B, T]`` token logprobs.

    ``loss_agg="token"`` averages over every masked token in the batch; ``"sequence"`` averages per
    sequence first and then over sequences (see :func:`~monorl.algorithms.losses.masked_aggregate`).
    ``label_smoothing > 0`` needs ``mean_logprob_all_tokens`` (``[B, T]``, cheapest) or ``logits``
    (``[B, T, V]``, reduced here with ``temperature``). Metrics: ``loss/sft`` (the value being
    optimised, smoothing included), ``sft/ppl`` (``exp`` of the unsmoothed token-mean NLL) and
    ``sft/tokens`` (masked token count).
    """
    if logprobs.ndim != 2:
        raise ValueError(f"logprobs must be [B, T], got shape {tuple(logprobs.shape)}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    mask = response_mask.to(logprobs.dtype)
    nll = -logprobs
    if label_smoothing > 0.0:
        if mean_logprob_all_tokens is None:
            if logits is None:
                raise ValueError("label_smoothing > 0 needs mean_logprob_all_tokens or logits")
            mean_logprob_all_tokens = _vocab_mean_logprobs(logits, temperature)
        if mean_logprob_all_tokens.shape != logprobs.shape:
            raise ValueError(
                f"mean_logprob_all_tokens must match logprobs {tuple(logprobs.shape)}, "
                f"got {tuple(mean_logprob_all_tokens.shape)}"
            )
        nll = (1.0 - label_smoothing) * nll + label_smoothing * (-mean_logprob_all_tokens)
    loss = masked_aggregate(nll, mask, loss_agg)

    with torch.no_grad():
        tokens = float(mask.sum().item())
        token_nll = float((-logprobs.detach() * mask).sum().item() / max(tokens, 1.0))
    metrics = {
        "loss/sft": float(loss.detach().item()),
        "sft/ppl": math.exp(min(token_nll, _PPL_MAX_NLL)),
        "sft/tokens": tokens,
    }
    return PolicyLossOutput(loss=loss, metrics=metrics)


def sft_policy_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor | None,
    advantages: torch.Tensor | None,
    response_mask: torch.Tensor,
    *,
    algorithm: AlgorithmConfig,
    ref_logprobs: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    mean_logprob_all_tokens: torch.Tensor | None = None,
) -> PolicyLossOutput:
    """Trainer-facing adapter: the policy-loss call signature with ``old_logprobs``/``advantages``/
    ``ref_logprobs``/``entropy`` ignored, dispatching to :func:`sft_loss` with the ``sft_*`` knobs
    of ``algorithm``."""
    return sft_loss(
        logprobs,
        response_mask,
        label_smoothing=algorithm.sft_label_smoothing,
        loss_agg=algorithm.loss_agg,
        mean_logprob_all_tokens=mean_logprob_all_tokens,
    )


def select_loss_fn(algorithm: AlgorithmConfig) -> Callable[..., PolicyLossOutput]:
    """The trainer loss for ``algorithm``: :func:`sft_policy_loss` for ``sft``, else
    :func:`~monorl.algorithms.losses.policy_loss`."""
    return sft_policy_loss if algorithm.name == "sft" else policy_loss
