"""Inline reward + 2-turn environment used by the TRL backend tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from monorl.core.types import Completion, Message, RewardResult, Sample, StepResult, Trajectory

REWARD_NAME = "trl_test_vowels"
ENV_NAME = "trl_test_two_turn"


class VowelReward:
    """Counts vowels in the response — varies between completions of one group."""

    name = REWARD_NAME

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        out = []
        for t in trajectories:
            score = float(sum(c in "aeiou" for c in t.response_text.lower()))
            out.append(
                RewardResult(
                    score=score, breakdown={REWARD_NAME: score}, info={"chars": len(t.response_text)}
                )
            )
        return out


class TwoTurnEnv:
    """Asks a follow-up after the first answer; per-turn env rewards 0.5 then 1.0."""

    max_turns = 2
    resets: list[str] = []

    def __init__(self, **_: Any) -> None:
        self.turn = 0

    def reset(self, sample: Sample) -> list[Message]:
        TwoTurnEnv.resets.append(sample.id)
        self.turn = 0
        return list(sample.messages)

    def step(self, action: Completion) -> StepResult:
        self.turn += 1
        if self.turn == 1:
            return StepResult(
                messages=[{"role": "user", "content": "Are you sure? Answer again."}], reward=0.5, done=False
            )
        return StepResult(messages=[], reward=1.0, done=True)

    def close(self) -> None:
        pass
