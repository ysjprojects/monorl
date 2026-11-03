"""Pure-torch RL objectives: advantage estimators, policy/value losses, KL estimators.

No model code lives here; trainers call :func:`policy_loss` on logprobs they computed and the
orchestrator calls :func:`compute_advantages` on scalar rewards.
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

__all__ = [
    "METRIC_KEYS",
    "PolicyLossOutput",
    "ValueLossOutput",
    "apply_overlong_penalty",
    "compute_advantages",
    "discount_turn_rewards",
    "dynamic_sampling_mask",
    "gae",
    "kl_penalty",
    "masked_aggregate",
    "policy_loss",
    "value_loss",
]
