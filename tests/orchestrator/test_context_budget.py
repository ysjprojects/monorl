"""The context budget of an episode: generation capped to the room left, episodes that end when
the next prompt does not fit, and the first prompt that does not fit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fakes
import pytest

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, DataError
from monorl.core.registry import registry
from monorl.core.reporting import iter_metrics
from monorl.core.types import Completion, Sample, SamplingParams
from monorl.orchestrator import run
from monorl.orchestrator.batching import prefix_chain_ok
from monorl.orchestrator.episodes import (
    EpisodeFailure,
    EpisodeSpec,
    chat_prompt_ids,
    run_episode,
    run_episodes,
)


class CappingEngine(fakes.FakeEngine):
    """Honours ``max_new_tokens`` like a real engine (the fake returns its whole answer) and
    remembers the limit of every call."""

    def __init__(self, config: RunConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.limits: list[int] = []

    def generate(self, prompts: Any, params: SamplingParams) -> list[list[Completion]]:
        self.limits.append(params.max_new_tokens)
        out = super().generate(prompts, params)
        for group in out:
            for c in group:
                if len(c.token_ids) > params.max_new_tokens:
                    c.token_ids = c.token_ids[: params.max_new_tokens]
                    c.logprobs = c.logprobs[: params.max_new_tokens]
                    c.text = self.tokenizer.decode(c.token_ids, skip_special_tokens=True)
                    c.finish_reason = "length"
        return out


def _sample(text: str = "abcdef") -> Sample:
    return Sample(id="s", messages=[{"role": "user", "content": text}], reference=text.upper())


def _episode(cfg: RunConfig, engine: Any, budget: int | None, sample: Sample | None = None) -> Any:
    return run_episode(
        EpisodeSpec(sample=sample or _sample(), group_id="g", engine=engine),
        env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1, max_new_tokens=16),
        max_context_tokens=budget,
    )


def _config(make_config: Any, max_turns: int = 3) -> RunConfig:
    return make_config(
        env={"type": "fake_multi", "max_turns": max_turns}, rollout={"extra": {"fake_mode": "always_right"}}
    )


def test_generation_is_capped_to_the_room_the_budget_leaves(make_config: Any) -> None:
    cfg = _config(make_config)
    engine = CappingEngine(cfg)
    prompt = len(chat_prompt_ids(engine.tokenizer, _sample().messages))
    trajectory = _episode(cfg, engine, budget=prompt + 3)
    assert engine.limits == [3]
    turn = trajectory.turns[0]
    assert len(turn.prompt_token_ids) + len(turn.completion.token_ids) == prompt + 3
    assert turn.completion.finish_reason == "length"
    assert trajectory.info["context_exhausted"] is True and not trajectory.done


def test_an_episode_ends_when_its_next_prompt_does_not_fit(make_config: Any) -> None:
    """The first turn fits with room to spare; the second prompt (first turn + answer + the
    environment's reply) would use up the budget, so the episode stops with its one turn."""
    cfg = _config(make_config)
    unlimited = _episode(cfg, CappingEngine(cfg), budget=None)
    assert len(unlimited.turns) == 3 and "context_exhausted" not in unlimited.info
    second_prompt = len(unlimited.turns[1].prompt_token_ids)

    engine = CappingEngine(cfg)
    trajectory = _episode(cfg, engine, budget=second_prompt)
    assert len(trajectory.turns) == 1 and engine.limits == [16]
    assert trajectory.turns[0].completion.token_ids == unlimited.turns[0].completion.token_ids
    assert trajectory.info["context_exhausted"] is True and not trajectory.done

    # one more token of budget lets the second turn in, capped to that one token
    engine = CappingEngine(cfg)
    trajectory = _episode(cfg, engine, budget=second_prompt + 1)
    assert len(trajectory.turns) == 2 and engine.limits == [16, 1]
    assert prefix_chain_ok(trajectory)


def test_a_first_prompt_over_the_budget_fails_the_episode(make_config: Any) -> None:
    cfg = _config(make_config)
    engine = CappingEngine(cfg)
    prompt = len(chat_prompt_ids(engine.tokenizer, _sample().messages))
    with pytest.raises(DataError, match="context budget"):
        _episode(cfg, engine, budget=prompt)
    assert engine.limits == []  # nothing was generated

    # ... which a run may tolerate like any other failed episode
    failures: list[EpisodeFailure] = []
    specs = [
        EpisodeSpec(sample=_sample("abcdefghijklmnopqrstuvwxyz"), group_id="long", engine=engine),
        EpisodeSpec(sample=_sample("abc"), group_id="short", engine=engine),
    ]
    trajectories = run_episodes(
        specs,
        env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1, max_new_tokens=16),
        max_parallel=1,
        max_context_tokens=prompt + 8,
        failures=failures,
    )
    assert [t.group_id for t in trajectories] == ["short"]
    assert [(f.spec.group_id, type(f.error)) for f in failures] == [("long", DataError)]


def test_the_budget_defaults_to_what_the_trainer_keeps(make_config: Any) -> None:
    """A three-turn episode needs 42 tokens. With ``trainer.max_seq_len: 40`` its last turn is cut
    to the room that is left, so no training sequence has to be cut off afterwards."""
    registry.register("rollout_engine", "capping", CappingEngine, overwrite=True)
    cfg = make_config(
        env={"type": "fake_multi", "max_turns": 3},
        rollout={"backend": "capping", "extra": {"fake_mode": "always_right"}},
        trainer={"max_steps": 1, "max_seq_len": 40},
    )
    assert cfg.context_budget == 40
    result = run(cfg)
    assert result.state == "succeeded", result.error
    row = next(row for _, row in iter_metrics(Path(result.run_dir)) if row["phase"] == "train")
    assert row["episode/context_exhausted_frac"] == 1.0 and row["episode/done_frac"] == 0.0
    assert row["tokens/truncated"] == 0.0
    batch = fakes.TRAINERS[-1].batches[0]
    assert batch.input_ids.shape[1] == 40


def test_budget_resolution_and_rules(make_config: Any) -> None:
    assert make_config(trainer={"max_seq_len": 64}).context_budget == 64
    assert make_config(trainer={"max_seq_len": 64}, rollout={"max_model_len": 48}).context_budget == 48
    assert make_config(trainer={"max_seq_len": 64}, rollout={"max_context_tokens": 32}).context_budget == 32
    with pytest.raises(ConfigError, match="max_context_tokens"):
        make_config(trainer={"max_seq_len": 64}, rollout={"max_context_tokens": 65})
    with pytest.raises(ConfigError, match="max_context_tokens"):
        make_config(trainer={"backend": "verl", "max_seq_len": 64}, rollout={"max_context_tokens": 32})
