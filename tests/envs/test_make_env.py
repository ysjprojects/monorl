from __future__ import annotations

import sys
from pathlib import Path

import pytest

from monorl.config.schema import RunConfig
from monorl.core.interfaces import Environment
from monorl.core.registry import registry
from monorl.core.types import Completion, Sample
from monorl.envs import RemoteEnv, SingleTurnEnv, ToolCallingEnv, make_env
from monorl.envs.remote import shutdown_pools
from monorl.envs.server import MEPServer, start_http_server
from monorl.envs.tool_calling import make_tool_calling_env

OPENENV_TESTS = Path(__file__).resolve().parents[1] / "adapters" / "openenv"


def _config(run_config_dict: dict, env: dict) -> RunConfig:
    return RunConfig.from_dict({**run_config_dict, "env": env})


def test_registered_names_include_builtins():
    names = registry.names("env")
    assert {"single_turn", "tool_calling", "remote"} <= set(names)


def test_make_single_turn(run_config_dict):
    cfg = _config(run_config_dict, {"type": "single_turn"})
    cfg = RunConfig.from_dict({**cfg.to_dict(), "data": {**cfg.data.model_dump(), "system_prompt": "sys"}})
    env = make_env(cfg)
    assert isinstance(env, SingleTurnEnv) and env.max_turns == 1
    msgs = env.reset(Sample(id="s", messages=[{"role": "user", "content": "hi"}]))
    assert msgs[0] == {"role": "system", "content": "sys"}


def test_make_tool_calling_respects_explicit_and_param_max_turns(run_config_dict):
    env = make_env(_config(run_config_dict, {"type": "tool_calling", "params": {"tools": ["calculator"]}}))
    assert isinstance(env, ToolCallingEnv) and env.max_turns == 4
    env = make_env(_config(run_config_dict, {"type": "tool_calling", "params": {"tools": ["calculator"], "max_turns": 6}}))
    assert env.max_turns == 6
    env = make_env(_config(run_config_dict, {"type": "tool_calling", "max_turns": 2, "params": {"tools": ["calculator"], "max_turns": 6}}))
    assert env.max_turns == 2
    assert make_env(_config(run_config_dict, {"type": "tool_calling"})) is not env  # fresh instance per call


def test_make_remote_http_and_stdio(run_config_dict):
    handle = start_http_server(MEPServer(lambda: make_tool_calling_env({"tools": ["calculator"]}, max_turns=5)), port=0)
    try:
        env = make_env(_config(run_config_dict, {"type": "remote", "params": {"transport": "http", "url": handle.url}}))
        assert isinstance(env, RemoteEnv) and env.max_turns == 5  # from spec
        env = make_env(_config(run_config_dict, {"type": "remote", "max_turns": 2, "params": {"transport": "http", "url": handle.url}}))
        assert env.max_turns == 2  # explicit override wins
        script = Path(__file__).with_name("mep_tool_server.py")
        env = make_env(
            _config(run_config_dict, {"type": "remote", "params": {"transport": "stdio", "command": [sys.executable, str(script)], "timeout_s": 30}})
        )
        assert isinstance(env, Environment) and env.spec.name == "crashable_tools" and env.max_turns == 3
        env.reset(Sample(id="x", messages=[{"role": "user", "content": "q"}]))
        assert env.step(Completion(text="done", token_ids=[])).done
    finally:
        handle.stop()
        shutdown_pools()


def test_make_openenv(run_config_dict, monkeypatch):
    pytest.importorskip("openenv.core")
    monkeypatch.syspath_prepend(str(OPENENV_TESTS))
    cfg = _config(
        run_config_dict,
        {"type": "openenv", "max_turns": 2, "params": {"env_class": "sample_envs.ReverseEnv", "env_kwargs": {"word": "ab"}}},
    )
    env = make_env(cfg)
    assert type(env).__name__ == "OpenEnvEnv" and env.max_turns == 2
    msgs = env.reset(Sample(id="s", messages=[]))
    assert msgs[-1]["content"] == "Reverse the word: ab"
    assert env.step(Completion(text="ba", token_ids=[])).reward == 1.0
    env.close()


def test_hermes_name_resolves_through_adapter_provider():
    pytest.importorskip("monorl.adapters.hermes")
    assert callable(registry.get("env", "hermes"))
