"""Reward functions (registry namespace ``reward``) and the composite builder.

Factories take ``(params: dict, **kw)``. :func:`build_reward` turns the ``rewards`` list of a
:class:`~monorl.config.schema.RunConfig` into one weighted :class:`CompositeReward`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from monorl.config.schema import RewardSpec, RunConfig
from monorl.core.interfaces import RewardFunction
from monorl.core.registry import Registry, registry
from monorl.rewards.code_exec import make_code_exec
from monorl.rewards.composite import CompositeReward, member_keys
from monorl.rewards.exact_match import make_exact_match
from monorl.rewards.format import make_format
from monorl.rewards.length import make_length
from monorl.rewards.llm_judge import make_llm_judge
from monorl.rewards.math_verify_reward import make_math_verify
from monorl.rewards.reward_model import make_reward_model

__all__ = ["CompositeReward", "REWARD_NAMES", "build_reward", "make_composite", "register"]


def _coerce_specs(raw: Sequence[RewardSpec | dict[str, Any]]) -> list[RewardSpec]:
    return [s if isinstance(s, RewardSpec) else RewardSpec.model_validate(s) for s in raw]


def build_reward(
    specs: Sequence[RewardSpec | dict[str, Any]], *, config: RunConfig | None = None
) -> RewardFunction:
    """Weighted composite of ``specs``; ``RewardResult.breakdown[spec.params["name"] or spec.type]``
    holds each member's raw score. Factories receive ``config=config`` when given."""
    specs = _coerce_specs(specs)
    kw: dict[str, Any] = {} if config is None else {"config": config}
    members = []
    for key, spec in zip(member_keys(specs), specs, strict=True):
        factory = registry.get("reward", spec.type)
        members.append((key, float(spec.weight), factory(dict(spec.params), **kw)))
    return CompositeReward(members)


def make_composite(params: dict[str, Any], **kw: Any) -> CompositeReward:
    specs = params.get("rewards")
    if not specs:
        raise ValueError("composite reward needs params.rewards (list of {type, weight, params})")
    built = build_reward(specs, config=kw.get("config"))
    built.name = str(params.get("name") or "composite")
    return built


REWARD_NAMES: dict[str, Any] = {
    "exact_match": make_exact_match,
    "format": make_format,
    "math_verify": make_math_verify,
    "code_exec": make_code_exec,
    "length": make_length,
    "llm_judge": make_llm_judge,
    "reward_model": make_reward_model,
    "composite": make_composite,
}


def register(reg: Registry) -> None:
    for name, factory in REWARD_NAMES.items():
        reg.register("reward", name, factory)
