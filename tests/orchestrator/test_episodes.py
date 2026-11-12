from __future__ import annotations

import threading
from typing import Any

import fakes
import pytest

from monorl.core.types import Sample, SamplingParams
from monorl.orchestrator.batching import prefix_chain_ok
from monorl.orchestrator.episodes import (
    EpisodeAborted,
    EpisodeSpec,
    chat_prompt_ids,
    run_episode,
    run_episodes,
)


def _sample(text: str = "abc", sid: str = "0") -> Sample:
    return Sample(id=sid, messages=[{"role": "user", "content": text}], reference=text.upper())


def _engine(make_config: Any, **extra: Any) -> fakes.FakeEngine:
    cfg = make_config(rollout={"extra": {**extra}})
    return fakes.FakeEngine(cfg)


def test_single_turn_episode_records_engine_prompt_ids(make_config: Any) -> None:
    cfg = make_config()
    engine = _engine(make_config, fake_mode="always_right")
    env_factory = lambda: fakes.FakeMultiTurnEnv(cfg)  # noqa: E731 - max_turns=1 by default
    traj = run_episode(
        EpisodeSpec(sample=_sample(), group_id="g", engine=engine),
        env_factory=env_factory,
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1),
    )
    assert len(traj.turns) == 1
    turn = traj.turns[0]
    assert turn.prompt_token_ids == chat_prompt_ids(engine.tokenizer, _sample().messages)
    assert turn.completion.text == "ABC"
    assert traj.messages[-1] == {"role": "assistant", "content": "ABC"}
    assert traj.done and traj.info["finish_reason"] == "stop"
    assert engine.calls == [{"num_prompts": 1, "n": 1, "temperature": 1.0}]


def test_multi_turn_prompts_are_token_exact_extensions(make_config: Any) -> None:
    cfg = make_config(env={"type": "fake_multi", "max_turns": 3, "params": {"turn_reward": 0.5}})
    engine = _engine(make_config, fake_mode="always_right")
    traj = run_episode(
        EpisodeSpec(sample=_sample(), group_id="g", engine=engine),
        env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1),
    )
    assert len(traj.turns) == 3
    assert prefix_chain_ok(traj)
    assert [t.reward for t in traj.turns] == [0.5, 0.5, 0.5]
    assert [t.info["turn"] for t in traj.turns] == [1, 2, 3]
    # turn 1 prompt == chat template over (user, assistant, user) conversation
    expected = chat_prompt_ids(engine.tokenizer, traj.messages[:3])
    assert traj.turns[1].prompt_token_ids == expected
    assert traj.turns[2].prompt_token_ids[: len(expected)] == expected
    # conversation keeps assistant + env messages in order
    assert [m["role"] for m in traj.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert traj.done


def test_episode_stops_when_env_is_done_early(make_config: Any) -> None:
    cfg = make_config(env={"type": "fake_multi", "max_turns": 5, "params": {"done_after": 2}})
    engine = _engine(make_config)
    traj = run_episode(
        EpisodeSpec(sample=_sample(), group_id="g", engine=engine),
        env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1),
    )
    assert len(traj.turns) == 2 and traj.done


def test_length_finish_ends_episode_not_done(make_config: Any) -> None:
    cfg = make_config(env={"type": "fake_multi", "max_turns": 4})
    engine = _engine(make_config, fake_finish="length")
    traj = run_episode(
        EpisodeSpec(sample=_sample(), group_id="g", engine=engine),
        env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1),
    )
    assert len(traj.turns) == 1 and not traj.done and traj.info["finish_reason"] == "length"


def test_tools_are_passed_to_chat_template_from_turn_zero(make_config: Any) -> None:
    tools = [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
    cfg = make_config(env={"type": "fake_multi", "max_turns": 2, "params": {"tools": tools}})
    engine = _engine(make_config)
    traj = run_episode(
        EpisodeSpec(sample=_sample(), group_id="g", engine=engine),
        env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1),
    )
    tok = engine.tokenizer
    for turn in traj.turns:
        assert turn.prompt_token_ids[0] == fakes._MARKERS["<|tools|>"]
        assert "calc" in tok.decode(turn.prompt_token_ids)
    # the engine received raw ids (already templated) on turn 0 because tools were present
    assert traj.turns[0].prompt_token_ids == chat_prompt_ids(tok, _sample().messages, tools)
    assert prefix_chain_ok(traj)


def test_run_episodes_parallel_preserves_order_and_closes_envs(make_config: Any) -> None:
    cfg = make_config(env={"type": "fake_multi", "max_turns": 2})
    engine = _engine(make_config, fake_mode="always_right", fake_latency=0.01)
    envs: list[fakes.FakeMultiTurnEnv] = []
    lock = threading.Lock()

    def factory() -> fakes.FakeMultiTurnEnv:
        env = fakes.FakeMultiTurnEnv(cfg)
        with lock:
            envs.append(env)
        return env

    specs = [
        EpisodeSpec(sample=_sample(f"p{i}", sid=str(i)), group_id=f"g{i // 2}", engine=engine)
        for i in range(8)
    ]
    trajs = run_episodes(
        specs, env_factory=factory, tokenizer=engine.tokenizer, params=SamplingParams(n=1), max_parallel=4
    )
    assert [t.sample.id for t in trajs] == [str(i) for i in range(8)]
    assert [t.group_id for t in trajs] == [s.group_id for s in specs]
    assert all(t.turns[0].completion.text == f"P{i}" for i, t in enumerate(trajs))
    assert len(envs) == 8 and all(e.closed for e in envs)


def test_episode_error_propagates_and_closes_env(make_config: Any) -> None:
    cfg = make_config()
    engine = _engine(make_config)

    class BoomEnv(fakes.FakeMultiTurnEnv):
        def step(self, action: Any) -> Any:
            raise RuntimeError("env exploded")

    envs: list[BoomEnv] = []

    def factory() -> BoomEnv:
        env = BoomEnv(cfg)
        envs.append(env)
        return env

    specs = [EpisodeSpec(sample=_sample(), group_id="g", engine=engine) for _ in range(3)]
    with pytest.raises(RuntimeError, match="env exploded"):
        run_episodes(
            specs, env_factory=factory, tokenizer=engine.tokenizer, params=SamplingParams(n=1), max_parallel=3
        )
    assert envs and all(e.closed for e in envs)


def test_abort_event_raises_episode_aborted(make_config: Any) -> None:
    cfg = make_config()
    engine = _engine(make_config)
    abort = threading.Event()
    abort.set()
    with pytest.raises(EpisodeAborted):
        run_episode(
            EpisodeSpec(sample=_sample(), group_id="g", engine=engine),
            env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
            tokenizer=engine.tokenizer,
            params=SamplingParams(n=1),
            abort=abort,
        )
