"""Single-turn environment: the prompt is the whole episode; rewards come from reward functions."""

from __future__ import annotations

from typing import Any

from monorl.core.types import Completion, Message, Sample, StepResult


class SingleTurnEnv:
    max_turns = 1

    def __init__(self, *, system_prompt: str | None = None, **_: Any) -> None:
        self.system_prompt = system_prompt
        self._sample: Sample | None = None

    def reset(self, sample: Sample) -> list[Message]:
        self._sample = sample
        messages = list(sample.messages)
        if self.system_prompt and not (messages and messages[0].get("role") == "system"):
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def step(self, action: Completion) -> StepResult:
        return StepResult(messages=[], reward=0.0, done=True, info={})

    def close(self) -> None:
        self._sample = None
