"""``length``: keep responses inside a target length band.

Reward is ``base`` (default 1.0) when the length is within ``[min_tokens, max_tokens]`` and
drops by ``penalty_per_token`` for every unit outside the band, floored at ``min_reward``
(default 0.0). Use ``unit: chars`` or ``unit: words`` to measure characters/words instead of
policy tokens (``Trajectory.num_response_tokens``). ``scope`` selects last turn or all turns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from monorl.core.types import RewardResult, Trajectory
from monorl.rewards.base import check_trajectories, response_text, reward_name


class LengthReward:
    def __init__(self, params: dict[str, Any]) -> None:
        self.name = reward_name(params, "length")
        self.min_len = int(params.get("min_tokens", 0))
        max_len = params.get("max_tokens")
        self.max_len = int(max_len) if max_len is not None else None
        if self.max_len is not None and self.max_len < self.min_len:
            raise ValueError("length reward: max_tokens must be >= min_tokens")
        self.penalty = float(params.get("penalty_per_token", 0.01))
        self.base = float(params.get("base", 1.0))
        self.min_reward = float(params.get("min_reward", 0.0))
        self.unit = str(params.get("unit", "tokens"))
        if self.unit not in ("tokens", "chars", "words"):
            raise ValueError(f"length reward: unknown unit {self.unit!r}")
        self.scope = str(params.get("scope", "last"))

    def measure(self, trajectory: Trajectory) -> int:
        if self.unit == "tokens":
            if self.scope == "last":
                return len(trajectory.turns[-1].completion.token_ids) if trajectory.turns else 0
            return trajectory.num_response_tokens
        text = response_text(trajectory, self.scope)
        return len(text) if self.unit == "chars" else len(text.split())

    def score_length(self, length: int) -> float:
        excess = 0
        if length < self.min_len:
            excess = self.min_len - length
        elif self.max_len is not None and length > self.max_len:
            excess = length - self.max_len
        return max(self.min_reward, self.base - self.penalty * excess)

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        check_trajectories(trajectories)
        out = []
        for t in trajectories:
            length = self.measure(t)
            score = self.score_length(length)
            out.append(RewardResult(score=score, breakdown={self.name: score}, info={"length": length}))
        return out


def make_length(params: dict[str, Any], **_: Any) -> LengthReward:
    return LengthReward(params)
