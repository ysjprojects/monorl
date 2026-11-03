from __future__ import annotations

import pytest

from monorl.core.types import Sample
from monorl.orchestrator.sampler import PromptSampler


def _samples(n: int) -> list[Sample]:
    return [Sample(id=str(i), messages=[{"role": "user", "content": str(i)}]) for i in range(n)]


def test_first_epoch_keeps_source_order_and_wraps() -> None:
    s = PromptSampler(_samples(5), prompts_per_step=2, seed=1)
    assert [x.id for x in s.batch(1)] == ["0", "1"]
    assert [x.id for x in s.batch(2)] == ["2", "3"]
    third = [x.id for x in s.batch(3)]
    assert third[0] == "4" and third[1] in {"0", "1", "2", "3"}  # second slot comes from epoch 1
    assert s.epoch_of_step(3) == 0 and s.epoch_of_step(4) == 1


def test_batches_are_a_pure_function_of_step_and_seed() -> None:
    a = PromptSampler(_samples(7), prompts_per_step=3, seed=9)
    b = PromptSampler(_samples(7), prompts_per_step=3, seed=9)
    c = PromptSampler(_samples(7), prompts_per_step=3, seed=10)
    for step in (1, 5, 9, 40):
        assert [x.id for x in a.batch(step)] == [x.id for x in b.batch(step)]
    ids_a = [[x.id for x in a.batch(step)] for step in range(4, 12)]
    ids_c = [[x.id for x in c.batch(step)] for step in range(4, 12)]
    assert ids_a != ids_c  # later epochs are seed-dependent permutations
    # every epoch is a permutation: 7 samples over steps covering one epoch
    seen = [x.id for step in range(1, 8) for x in a.batch(step)][7:14]
    assert sorted(seen) == [str(i) for i in range(7)]


def test_no_shuffle_keeps_order_across_epochs() -> None:
    s = PromptSampler(_samples(3), prompts_per_step=3, seed=0, shuffle=False)
    assert [x.id for x in s.batch(1)] == [x.id for x in s.batch(2)] == ["0", "1", "2"]


def test_resample_rounds_are_deterministic_and_independent() -> None:
    s = PromptSampler(_samples(10), prompts_per_step=4, seed=3)
    r1 = [x.id for x in s.batch(2, round=1)]
    assert r1 == [x.id for x in s.batch(2, round=1)]
    assert len(r1) == 4 and len(set(r1)) == 4
    assert r1 != [x.id for x in s.batch(2, round=2)]
    assert [x.id for x in s.batch(2)] == ["4", "5", "6", "7"]  # main stream unaffected


def test_rejects_empty_and_zero_step() -> None:
    with pytest.raises(ValueError):
        PromptSampler([], prompts_per_step=1, seed=0)
    with pytest.raises(ValueError):
        PromptSampler(_samples(2), prompts_per_step=1, seed=0).batch(0)
