from __future__ import annotations

import threading
import time
from typing import Any

import fakes
import pytest

from monorl.core.types import Sample, SamplingParams, Trajectory, Turn
from monorl.orchestrator.batching import prefix_chain_ok
from monorl.orchestrator.episodes import (
    EpisodeAborted,
    EpisodeSpec,
    chat_prompt_ids,
    run_episode,
    run_episodes,
)
from monorl.rollout import BatchingEngine


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


def test_token_rendering_keeps_sampled_tokens_and_template_rendering_drops_them(make_config: Any) -> None:
    """The engine samples a token (9999) that its text does not re-tokenise to."""
    cfg = make_config(env={"type": "fake_multi", "max_turns": 2})

    def episode(turn_rendering: str) -> Any:
        engine = _engine(make_config, fake_mode="always_right", fake_mismatch=True)
        return run_episode(
            EpisodeSpec(sample=_sample(), group_id="g", engine=engine),
            env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
            tokenizer=engine.tokenizer,
            params=SamplingParams(n=1),
            turn_rendering=turn_rendering,
        )

    tokens = episode("tokens")
    assert 9999 in tokens.turns[0].completion.token_ids
    assert prefix_chain_ok(tokens) and 9999 in tokens.turns[1].prompt_token_ids

    template = episode("template")
    assert not prefix_chain_ok(template)
    assert template.turns[1].prompt_token_ids == chat_prompt_ids(fakes.FakeTokenizer(), template.messages[:3])
    # what the conversation says is the same either way
    assert tokens.messages == template.messages
    assert [t.message_index for t in tokens.turns] == [1, 3]


class _ScriptedDriver:
    """An ``EpisodeDriver``: runs the whole episode itself instead of being stepped."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False
        self.kwargs: dict[str, Any] = {}

    def run_episode(self, sample: Sample, **kwargs: Any) -> Trajectory:
        self.kwargs = kwargs
        if self.fail:
            raise RuntimeError("agent crashed")
        completion = kwargs["engine"].generate([list(sample.messages)], kwargs["params"])[0][0]
        return Trajectory(
            sample=sample,
            group_id=kwargs["group_id"],
            messages=[*sample.messages, {"role": "assistant", "content": completion.text}],
            turns=[Turn(prompt_token_ids=list(completion.prompt_token_ids), completion=completion)],
        )

    def close(self) -> None:
        self.closed = True


def test_episode_driver_runs_the_episode_itself(make_config: Any) -> None:
    engine = _engine(make_config, fake_mode="always_right")
    drivers: list[_ScriptedDriver] = []

    def factory() -> _ScriptedDriver:
        drivers.append(_ScriptedDriver())
        return drivers[-1]

    abort = threading.Event()
    specs = [
        EpisodeSpec(sample=_sample("abc", sid="0"), group_id="g0", engine=engine),
        EpisodeSpec(sample=_sample("xyz", sid="1"), group_id="g1", engine=engine),
    ]
    trajs = run_episodes(
        specs,
        env_factory=factory,
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1),
        max_parallel=1,
        abort=abort,
        turn_rendering="template",
    )
    assert [(t.group_id, t.last_response_text) for t in trajs] == [("g0", "ABC"), ("g1", "XYZ")]
    assert len(drivers) == 2 and all(d.closed for d in drivers)
    for driver in drivers:
        assert driver.kwargs["engine"] is engine and driver.kwargs["tokenizer"] is engine.tokenizer
        assert driver.kwargs["abort"] is abort and driver.kwargs["turn_rendering"] == "template"


def test_episode_driver_is_closed_when_it_fails(make_config: Any) -> None:
    engine = _engine(make_config)
    driver = _ScriptedDriver(fail=True)
    with pytest.raises(RuntimeError, match="agent crashed"):
        run_episode(
            EpisodeSpec(sample=_sample(), group_id="g", engine=engine),
            env_factory=lambda: driver,
            tokenizer=engine.tokenizer,
            params=SamplingParams(n=1),
        )
    assert driver.closed


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


# -- abort during generation -------------------------------------------------------------------


def _episode_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("monorl-episode")]


def _set_when(event: threading.Event, abort: threading.Event) -> None:
    def fire() -> None:
        if event.wait(10):
            abort.set()

    threading.Thread(target=fire, daemon=True).start()


@pytest.mark.parametrize(
    ("max_parallel", "batched"), [(1, False), (4, False), (4, True)], ids=["inline", "pool", "pool-batched"]
)
def test_abort_cancels_generation_in_flight(make_config: Any, max_parallel: int, batched: bool) -> None:
    cfg = make_config(env={"type": "fake_multi", "max_turns": 2})
    slow = fakes.SlowCancellableEngine(cfg)  # a generate call takes 60 s unless cancelled
    engine: Any = BatchingEngine(slow, max_batch_size=4, max_wait_ms=5) if batched else slow
    envs: list[fakes.FakeMultiTurnEnv] = []

    def factory() -> fakes.FakeMultiTurnEnv:
        env = fakes.FakeMultiTurnEnv(cfg)
        envs.append(env)
        return env

    abort = threading.Event()
    _set_when(slow.generating, abort)
    specs = [EpisodeSpec(sample=_sample(sid=str(i)), group_id=f"g{i}", engine=engine) for i in range(4)]
    t0 = time.monotonic()
    try:
        with pytest.raises(EpisodeAborted):
            run_episodes(
                specs,
                env_factory=factory,
                tokenizer=slow.tokenizer,
                params=SamplingParams(n=1),
                max_parallel=max_parallel,
                abort=abort,
            )
        assert time.monotonic() - t0 < 10.0
        assert _episode_threads() == []
        # inline: the specs after the aborted one never start; pool: up to one env per worker
        assert (len(envs) == 1 if max_parallel == 1 else 1 <= len(envs) <= 4) and all(e.closed for e in envs)
        assert slow.calls == [] and slow.cancel_calls >= 1  # nothing ran to completion
    finally:
        if batched:
            engine.shutdown()


def test_abort_cancels_a_generate_submitted_after_the_first_cancel(make_config: Any) -> None:
    cfg = make_config(env={"type": "fake_multi", "max_turns": 2})
    engine = fakes.SlowCancellableEngine(cfg)
    engine.generate_seconds = 0.0  # turn 0 answers at once
    abort = threading.Event()

    class AbortBetweenCheckAndGenerate(fakes.FakeTokenizer):
        """Turn 1 is templated after the episode's abort check and before its ``generate``: set
        the abort there and hold the episode until the first ``cancel()`` has found nothing."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
            if len(messages) > 1 and not abort.is_set():
                engine.generate_seconds = 60.0
                abort.set()
                deadline = time.monotonic() + 10
                while engine.cancel_calls == 0 and time.monotonic() < deadline:
                    time.sleep(0.002)
            return super().apply_chat_template(messages, **kwargs)

    env = fakes.FakeMultiTurnEnv(cfg)
    t0 = time.monotonic()
    with pytest.raises(EpisodeAborted):
        run_episodes(
            [EpisodeSpec(sample=_sample(), group_id="g", engine=engine)],
            env_factory=lambda: env,
            tokenizer=AbortBetweenCheckAndGenerate(),
            params=SamplingParams(n=1),
            max_parallel=1,
            abort=abort,
        )
    assert time.monotonic() - t0 < 10.0
    assert len(engine.calls) == 1 and engine.cancel_calls >= 2  # turn 0 completed, turn 1 was cut
    assert env.closed and _episode_threads() == []


def test_abort_waits_for_env_step_in_flight_and_skips_engines_without_cancel(make_config: Any) -> None:
    cfg = make_config(env={"type": "fake_multi", "max_turns": 2})
    fast = fakes.FakeEngine(make_config(rollout={"extra": {"fake_mode": "always_right"}}))  # no cancel()
    slow = fakes.SlowCancellableEngine(cfg)
    in_step = threading.Event()
    release_step = threading.Event()

    class GatedEnv(fakes.FakeMultiTurnEnv):
        """The episode of sample "0" blocks in ``step`` until released."""

        def __init__(self) -> None:
            super().__init__(cfg)
            self.steps_finished = 0
            self.gated = False

        def reset(self, sample: Sample) -> Any:
            self.gated = sample.id == "0"
            return super().reset(sample)

        def step(self, action: Any) -> Any:
            if self.gated:
                in_step.set()
                assert release_step.wait(10)
            result = super().step(action)
            self.steps_finished += 1
            return result

    envs: list[GatedEnv] = []
    lock = threading.Lock()

    def factory() -> GatedEnv:
        env = GatedEnv()
        with lock:
            envs.append(env)
        return env

    abort = threading.Event()

    def fire() -> None:
        if in_step.wait(10) and slow.generating.wait(10):
            abort.set()
            time.sleep(0.3)  # well past the cancel of the other episode's generation
            release_step.set()

    threading.Thread(target=fire, daemon=True).start()
    specs = [
        EpisodeSpec(sample=_sample(sid="0"), group_id="g0", engine=fast),
        EpisodeSpec(sample=_sample(sid="1"), group_id="g1", engine=slow),
    ]
    t0 = time.monotonic()
    with pytest.raises(EpisodeAborted):
        run_episodes(
            specs,
            env_factory=factory,
            tokenizer=fast.tokenizer,
            params=SamplingParams(n=1),
            max_parallel=2,
            abort=abort,
        )
    assert time.monotonic() - t0 < 10.0
    assert release_step.is_set()  # did not return while the env step was still running
    gated = next(e for e in envs if e.gated)
    assert gated.steps_finished == 1  # the step in flight completed; turn 2 never started
    assert len(fast.calls) == 1 and slow.calls == []
    assert len(envs) == 2 and all(e.closed for e in envs)
    assert _episode_threads() == []
