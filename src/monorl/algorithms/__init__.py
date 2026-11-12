"""Pure-torch objectives: advantage estimators, policy/value losses, KL estimators, SFT NLL.

No model code lives here; trainers call :func:`policy_loss` (or :func:`sft_loss` through
:func:`select_loss_fn`) on logprobs they computed and the orchestrator calls
:func:`compute_advantages` on scalar rewards.
"""

from monorl.algorithms.advantages import (
    apply_overlong_penalty,
    compute_advantages,
    discount_turn_rewards,
    dynamic_sampling_mask,
)
from monorl.algorithms.kl import kl_penalty
from monorl.algorithms.losses import METRIC_KEYS, PolicyLossOutput, masked_aggregate, policy_loss
from monorl.algorithms.ppo import ValueLossOutput, gae, value_loss
from monorl.algorithms.sft import (
    SFT_METRIC_KEYS,
    mean_logprob_all_tokens,
    select_loss_fn,
    sft_loss,
    sft_policy_loss,
)

__all__ = [
    "METRIC_KEYS",
    "SFT_METRIC_KEYS",
    "PolicyLossOutput",
    "ValueLossOutput",
    "apply_overlong_penalty",
    "compute_advantages",
    "discount_turn_rewards",
    "dynamic_sampling_mask",
    "gae",
    "kl_penalty",
    "masked_aggregate",
    "mean_logprob_all_tokens",
    "policy_loss",
    "select_loss_fn",
    "sft_loss",
    "sft_policy_loss",
    "value_loss",
]
