from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from monorl.adapters.openenv import is_available, register
from monorl.adapters.openenv.env import (
    ClientBackend,
    InProcessBackend,
    OpenEnvEnv,
    backend_from_params,
    make_openenv_env,
    observation_to_text,
    parse_json_action,
)
from monorl.core.errors import EnvironmentError_
from monorl.core.interfaces import Environment
from monorl.core.registry import Registry
from monorl.core.types import Completion, Sample


def _c(text: str) -> Completion:
    return Completion(text=text, token_ids=[])


def _sample(i: str = "s", **metadata) -> Sample:
    return Sample(id=i, messages=[{"role": "user", "content": "Solve the task."}], metadata=metadata)


# -- helpers ----------------------------------------------------------------------------------


def test_observation_to_text():
    assert observation_to_text({"output": "hi"}) == "hi"
    assert observation_to_text({"a": "x", "b": 1}) == '{"a": "x", "b": 1}'
    assert (
        observation_to_text({"hint": "higher", "remaining": 2}, "Hint: {hint} ({remaining} left, {missing})")
        == "Hint: higher (2 left, )"
    )


def test_parse_json_action():
    assert parse_json_action('{"guess": 3}') == {"guess": 3}
    assert parse_json_action('I think:\n```json\n{"guess": 4}\n```') == {"guess": 4}
    assert parse_json_action("guess 5") is None
    assert parse_json_action("[1,2]") is None


def test_action_class_inference():
    from sample_envs import GuessAction, GuessEnv, ReverseAction, ReverseEnv

    assert InProcessBackend.resolve_action_class(ReverseEnv) is ReverseAction
    assert InProcessBackend.resolve_action_class(GuessEnv, "sample_envs.GuessAction") is GuessAction
    with pytest.raises(ValueError):
        InProcessBackend.resolve_action_class(ReverseEnv, "sample_envs.ReverseObservation")


def test_backend_from_params_requires_target():
    with pytest.raises(ValueError):
        backend_from_params({})


def test_provider_registers_and_is_available():
    assert is_available()
    reg = Registry()
    register(reg)
    assert callable(reg.get("env", "openenv"))


# -- in-process environment -------------------------------------------------------------------


def test_in_process_text_episode():
    env = make_openenv_env(
        {"env_class": "sample_envs.ReverseEnv", "env_kwargs": {"word": "abc"}, "system_prompt": "sys"},
        max_turns=3,
    )
    assert isinstance(env, Environment) and isinstance(env, OpenEnvEnv)
    assert env.tools == [] and env.max_turns == 3
    msgs = env.reset(_sample())
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Solve the task."},
        {"role": "user", "content": "Reverse the word: abc"},
    ]
    assert env.backend.env.state.episode_id == "s"
    r = env.step(_c("wrong"))
    assert r.done is False and r.reward == 0.0
    assert r.messages == [{"role": "user", "content": "wrong (attempt 1)"}]
    assert r.info["action"] == {"text": "wrong"} and r.info["observation"] == {"output": "wrong (attempt 1)"}
    r = env.step(_c("cba"))
    assert r.done is True and r.reward == 1.0 and r.messages == []
    env.close()
    assert env.backend.env.closed


def test_in_process_reset_kwargs_from_params_and_sample():
    env = make_openenv_env(
        {"env_class": "sample_envs.ReverseEnv", "reset_kwargs": {"word": "xy"}}, max_turns=3
    )
    assert env.reset(_sample())[-1]["content"] == "Reverse the word: xy"
    assert env.reset(_sample(reset_kwargs={"word": "pq"}))[-1]["content"] == "Reverse the word: pq"


def test_in_process_truncates_at_max_turns():
    env = make_openenv_env(
        {"env_class": "sample_envs.ReverseEnv", "env_kwargs": {"word": "abc", "attempts": 10}}, max_turns=2
    )
    env.reset(_sample())
    assert env.step(_c("no")).done is False
    r = env.step(_c("no"))
    assert r.done is True and r.messages == []


def test_in_process_json_actions_and_invalid_penalty():
    env = make_openenv_env(
        {
            "env_class": "sample_envs.GuessEnv",
            "env_kwargs": {"target": 7},
            "action_mode": "json",
            "action_field": "ignored",
            "observation_template": "Hint: {hint} ({remaining} left)",
            "invalid_action_penalty": 0.5,
        },
        max_turns=5,
    )
    assert env.reset(_sample())[-1]["content"] == "Hint: guess a number between 1 and 10 (5 left)"
    bad = env.step(_c("seven?"))
    assert bad.reward == -0.5 and bad.done is False and bad.info["invalid_action"] is True
    assert bad.messages[0]["role"] == "user" and "Invalid action" in bad.messages[0]["content"]
    assert env.backend.env.state.step_count == 0  # env untouched
    r = env.step(_c('```json\n{"guess": 3}\n```'))
    assert r.messages == [{"role": "user", "content": "Hint: higher (4 left)"}] and r.reward == pytest.approx(
        -0.1
    )
    r = env.step(_c('{"guess": 7}'))
    assert r.done and r.reward == 1.0


def test_in_process_rejects_wrong_action_shape():
    env = make_openenv_env({"env_class": "sample_envs.GuessEnv", "action_mode": "json"}, max_turns=3)
    env.reset(_sample())
    with pytest.raises(EnvironmentError_, match="rejected"):
        env.step(_c('{"guess": "not-an-int"}'))


def test_bad_action_mode():
    with pytest.raises(ValueError):
        make_openenv_env({"env_class": "sample_envs.GuessEnv", "action_mode": "yaml"}, max_turns=1)


# -- real openenv http server ------------------------------------------------------------------


def test_http_client_episode(reverse_server_url):
    env = make_openenv_env(
        {"url": reverse_server_url, "reset_kwargs": {"word": "hello"}, "timeout_s": 20}, max_turns=3
    )
    assert isinstance(env.backend, ClientBackend)
    msgs = env.reset(_sample("remote"))
    assert msgs[-1] == {"role": "user", "content": "Reverse the word: hello"}
    r = env.step(_c("nope"))
    assert r.done is False and r.messages[0]["content"].startswith("wrong")
    r = env.step(_c("olleh"))
    assert r.done is True and r.reward == 1.0
    env.close()
    # a closed session can be reused: reset reconnects
    assert env.reset(_sample("again"))[-1]["content"] == "Reverse the word: hello"
    env.close()


def test_http_concurrent_sessions(reverse_server_url):
    def run(i: int) -> float:
        env = make_openenv_env(
            {"url": reverse_server_url, "reset_kwargs": {"word": f"w{i}"}, "timeout_s": 20}, max_turns=2
        )
        try:
            assert env.reset(_sample(f"c{i}"))[-1]["content"] == f"Reverse the word: w{i}"
            return env.step(_c(f"{i}w")).reward
        finally:
            env.close()

    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(run, range(4))) == [1.0] * 4


def test_http_unreachable_server():
    env = make_openenv_env({"url": "http://127.0.0.1:9", "timeout_s": 1}, max_turns=1)
    with pytest.raises(EnvironmentError_, match="connect"):
        env.reset(_sample())
    env.close()
