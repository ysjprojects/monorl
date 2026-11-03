from __future__ import annotations

from typing import Any

import pytest

from monorl.core.types import Completion, Sample, Trajectory, Turn


def make_trajectory(
    response: str,
    reference: Any = None,
    *,
    prompt: str = "Solve it.",
    finish_reason: str = "stop",
    token_ids: list[int] | None = None,
    earlier_turns: list[str] | None = None,
    group_id: str = "g0",
) -> Trajectory:
    sample = Sample(id="s0", messages=[{"role": "user", "content": prompt}], reference=reference)
    turns = []
    for text in [*(earlier_turns or []), response]:
        ids = token_ids if token_ids is not None else list(range(len(text.split())))
        turns.append(Turn(prompt_token_ids=[1, 2, 3], completion=Completion(text=text, token_ids=ids, finish_reason=finish_reason)))
    return Trajectory(sample=sample, group_id=group_id, messages=list(sample.messages), turns=turns)


@pytest.fixture
def traj():
    return make_trajectory
