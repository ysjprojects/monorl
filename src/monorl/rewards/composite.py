"""``composite``: weighted sum of member rewards with a per-member breakdown.

``RewardResult.score = Σ weight_i * raw_i``; ``breakdown[key_i] = raw_i`` where ``key_i`` is
``params["name"]`` of the member spec, else its ``type`` (duplicates get ``#2``, ``#3``, …).
Member ``info`` dicts are nested under ``info[key_i]``.

As a registered reward (``type: composite``), ``params["rewards"]`` holds the member specs
(``[{type, weight, params}, ...]``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from monorl.config.schema import RewardSpec
from monorl.core.interfaces import RewardFunction
from monorl.core.types import RewardResult, Trajectory


def member_keys(specs: Sequence[RewardSpec]) -> list[str]:
    keys: list[str] = []
    for spec in specs:
        base = str(spec.params.get("name") or spec.type)
        key, n = base, 1
        while key in keys:
            n += 1
            key = f"{base}#{n}"
        keys.append(key)
    return keys


class CompositeReward:
    def __init__(
        self, members: Sequence[tuple[str, float, RewardFunction]], *, name: str = "composite"
    ) -> None:
        if not members:
            raise ValueError("composite reward needs at least one member")
        self.name = name
        self.members = list(members)

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        per_member = [(key, weight, fn(trajectories)) for key, weight, fn in self.members]
        out: list[RewardResult] = []
        for i in range(len(trajectories)):
            total = 0.0
            breakdown: dict[str, float] = {}
            info: dict[str, Any] = {}
            for key, weight, results in per_member:
                if len(results) != len(trajectories):
                    raise ValueError(
                        f"reward {key!r} returned {len(results)} results for {len(trajectories)} trajectories"
                    )
                r = results[i]
                breakdown[key] = float(r.score)
                total += weight * float(r.score)
                if r.info:
                    info[key] = r.info
            out.append(RewardResult(score=total, breakdown=breakdown, info=info))
        return out
