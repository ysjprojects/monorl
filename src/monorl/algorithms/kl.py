"""Per-token KL estimators between the policy and a reference (Schulman, "Approximating KL").

All three estimate ``KL(pi || ref) = E_pi[log pi - log ref]`` from samples drawn under ``pi``:

* ``k1``: ``log pi - log ref`` — unbiased, high variance, may be negative per token.
* ``k2``: ``0.5 * (log pi - log ref)^2`` — biased, low variance, non-negative.
* ``k3``: ``exp(log ref - log pi) - (log ref - log pi) - 1`` — unbiased, low variance,
  non-negative (the GRPO default).
"""

from __future__ import annotations

from typing import Literal

import torch

KLEstimator = Literal["k1", "k2", "k3"]


def kl_penalty(logprobs: torch.Tensor, ref_logprobs: torch.Tensor, estimator: KLEstimator) -> torch.Tensor:
    """Elementwise KL estimate ``[B, T]``; gradients flow through ``logprobs`` only."""
    log_ratio = logprobs - ref_logprobs
    if estimator == "k1":
        return log_ratio
    if estimator == "k2":
        return 0.5 * log_ratio.square()
    if estimator == "k3":
        neg = -log_ratio
        return torch.exp(neg) - neg - 1.0
    raise ValueError(f"unknown KL estimator {estimator!r}; expected k1, k2 or k3")
