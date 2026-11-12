"""``env: hermes`` against the real hermes-agent 0.19 tool runtime; the test plays the policy."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from monorl.adapters.hermes import PATCHES, is_available, register
from monorl.adapters.hermes.env import (
    HermesEnv,
    make_hermes_env,
    resolve_tool_names,
    sandbox_overrides,
)
from monorl.config.schema import RunConfig
from monorl.core.registry import registry
from monorl.core.types import Completion, Sample

pytestmark = pytest.mark.timeout(180)


def call(name: str, **arguments: Any) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def completion(text: str) -> Completion:
    return Completion(text=text, token_ids=[])


def sample(task: str, sid: str = "s1") -> Sample:
    return Sample(id=sid, messages=[{"role": "user", "content": task}])


# -- provider / config ------------------------------------------------------------------------


def test_registry_resolves_and_builds_from_run_config(
    run_config_dict: dict[str, Any], workspace: Path
) -> None:
    assert is_available()
    assert callable(registry.get("env", "hermes"))
    cfg = RunConfig.model_validate(
        {
            **run_config_dict,
            "env": {
                "type": "hermes",
                "max_turns": 3,
                "params": {"tools": ["read_file"], "workspace_dir": str(workspace)},
            },
        }
    )
    env = registry.get("env", "hermes")(cfg)
    try:
        assert isinstance(env, HermesEnv)
        assert env.max_turns == 3
        assert env.tool_names == ["read_file"]
        assert env.model_name == cfg.model.name_or_path
    finally:
        env.close()


def test_register_is_idempotent_and_patchset_registered() -> None:
    from monorl.patching import get_patchsets, status

    register(registry)  # same factory → no RegistryError
    assert PATCHES.framework == "hermes" and PATCHES.distribution == "hermes-agent"
    assert PATCHES in get_patchsets("hermes")
    assert PATCHES.installed_version() is not None
    assert status("hermes") == []


def test_unknown_param_and_unknown_tool_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown params"):
        make_hermes_env({"workspace": "/tmp"}, max_turns=1)
    with pytest.raises(ValueError, match="unknown Hermes tool/toolset names"):
        resolve_tool_names(["read_file", "no_such_tool"])


def test_resolve_tool_names_mixes_tools_and_toolsets() -> None:
    assert resolve_tool_names(["terminal", "read_file", "write_file"]) == [
        "read_file",
        "terminal",
        "write_file",
    ]
    assert set(resolve_tool_names(["file"])) >= {"read_file", "write_file", "patch", "search_files"}


def test_sandbox_overrides() -> None:
    assert sandbox_overrides(None) == ("local", {})
    assert sandbox_overrides({"backend": "docker", "image": "python:3.12"}) == (
        "docker",
        {"docker_image": "python:3.12", "env_type": "docker"},
    )
    with pytest.raises(ValueError, match="backend must be one of"):
        sandbox_overrides({"backend": "firecracker"})
    with pytest.raises(ValueError, match="not supported by the 'local' backend"):
        sandbox_overrides({"image": "python:3.12"})
    with pytest.raises(ValueError, match="unknown keys"):
        sandbox_overrides({"cpu": 2})


# -- episodes ---------------------------------------------------------------------------------


def test_episode_with_hermes_file_tools(workspace: Path, skills_dir: Path) -> None:
    env = HermesEnv(
        tools=["read_file", "write_file"],
        skills=["arith"],
        skills_dir=str(skills_dir),
        workspace_dir=str(workspace),
        max_turns=4,
        system_prompt="Answer with the file content when done.",
    )
    try:
        assert [t["function"]["name"] for t in env.tools] == ["read_file", "write_file"]
        messages = env.reset(sample("Write 'hi' to notes.txt, read it back and report it."))
        assert messages[0]["role"] == "system"
        system = messages[0]["content"]
        assert "Hermes Agent" in system  # Hermes' own identity / prompt builder
        assert "Pipe expressions into `bc`" in system  # skill body injected
        assert "Answer with the file content when done." in system
        assert f"Working directory for this task: {workspace}" in system
        assert messages[1] == {
            "role": "user",
            "content": "Write 'hi' to notes.txt, read it back and report it.",
        }

        step1 = env.step(completion(call("write_file", path="notes.txt", content="hi")))
        assert not step1.done
        assert len(step1.messages) == 1
        tool_msg = step1.messages[0]
        assert tool_msg["role"] == "tool" and tool_msg["name"] == "write_file"
        assert tool_msg["tool_call_id"] == step1.info["tool_calls"][0]["id"]
        assert set(tool_msg) <= {"role", "name", "tool_call_id", "content"}
        assert "error" not in json.loads(tool_msg["content"])
        assert (workspace / "notes.txt").read_text() == "hi"

        step2 = env.step(completion("Let me check." + call("read_file", path="notes.txt")))
        assert not step2.done
        assert json.loads(step2.messages[0]["content"])["content"].startswith("1|hi")

        step3 = env.step(completion("The file contains: hi"))
        assert step3.done and step3.messages == [] and step3.info["final_answer"] is True
        assert step3.info["truncated"] is False
        assert [h["turn"] for h in env.history] == [1, 2, 3]
    finally:
        env.close()


def test_custom_tool_registered_through_hermes_registry(workspace: Path) -> None:
    env = HermesEnv(
        tools=["monorl_calc"], tool_modules=["sample_tools"], workspace_dir=str(workspace), max_turns=3
    )
    try:
        assert env.tool_names == ["monorl_calc"]
        assert env.tools[0]["function"]["parameters"]["required"] == ["expression"]
        env.reset(sample("What is 6*7?"))
        step = env.step(completion(call("monorl_calc", expression="6*7")))
        assert json.loads(step.messages[0]["content"]) == {"result": 42}
        assert env.step(completion("42")).done
    finally:
        env.close()


def test_terminal_tool_runs_in_episode_workspace_and_is_truncated(workspace: Path) -> None:
    env = HermesEnv(tools=["terminal"], workspace_dir=str(workspace), max_turns=3, max_tool_output_chars=300)
    try:
        env.reset(sample("count"))
        step = env.step(completion(call("terminal", command="pwd && seq 1 2000")))
        content = step.messages[0]["content"]
        assert str(workspace) in content
        assert len(content) < 600 and "Truncated" in content
    finally:
        env.close()


def test_parallel_tool_calls_in_one_turn(workspace: Path) -> None:
    (workspace / "a.txt").write_text("A")
    (workspace / "b.txt").write_text("B")
    env = HermesEnv(tools=["read_file"], workspace_dir=str(workspace), max_turns=2)
    try:
        env.reset(sample("read both"))
        step = env.step(completion(call("read_file", path="a.txt") + call("read_file", path="b.txt")))
        assert [m["name"] for m in step.messages] == ["read_file", "read_file"]
        bodies = [json.loads(m["content"])["content"] for m in step.messages]
        assert bodies[0].startswith("1|A") and bodies[1].startswith("1|B")
        assert [m["tool_call_id"] for m in step.messages] == [c["id"] for c in step.info["tool_calls"]]
    finally:
        env.close()


def test_unknown_and_malformed_calls_follow_hermes_loop(workspace: Path) -> None:
    env = HermesEnv(
        tools=["read_file", "write_file"], workspace_dir=str(workspace), max_turns=4, invalid_call_penalty=0.5
    )
    try:
        env.reset(sample("x"))
        # Hermes repairs near-miss names (agent_runtime_helpers.repair_tool_call) ...
        (workspace / "a.txt").write_text("A")
        step = env.step(completion(call("Read_File", path="a.txt")))
        assert step.reward == 0.0 and step.messages[0]["name"] == "read_file"
        # ... and errors out hallucinated ones with its own message.
        step = env.step(completion(call("launch_rockets", target="moon")))
        assert step.reward == -0.5 and step.info["invalid_calls"] == 1
        assert (
            "Tool 'launch_rockets' does not exist. Available tools: read_file, write_file"
            in step.messages[0]["content"]
        )
        assert step.messages[0]["role"] == "tool"
        # malformed JSON inside <tool_call> is reported, not executed
        step = env.step(completion("<tool_call>{not json}</tool_call>"))
        assert step.reward == -0.5 and step.info["tool_calls"] == []
        assert step.messages[0]["content"].startswith("error: malformed <tool_call> JSON")
        assert not step.done
    finally:
        env.close()


def test_max_turns_cutoff(workspace: Path) -> None:
    env = HermesEnv(tools=["read_file"], workspace_dir=str(workspace), max_turns=2)
    try:
        (workspace / "a.txt").write_text("A")
        env.reset(sample("loop"))
        first = env.step(completion(call("read_file", path="a.txt")))
        assert not first.done and first.messages
        second = env.step(completion(call("read_file", path="a.txt")))
        assert second.done and second.messages == []
        assert second.info == {**second.info, "turn": 2, "final_answer": False, "truncated": True}
        with pytest.raises(RuntimeError, match="before reset"):
            env.step(completion("again"))
    finally:
        env.close()


def test_compact_system_prompt_lists_tool_schemas(workspace: Path) -> None:
    env = HermesEnv(
        tools=["read_file"], workspace_dir=str(workspace), max_turns=1, hermes_system_prompt=False
    )
    try:
        system = env.reset(sample("x"))[0]["content"]
        assert system.startswith("You are Hermes Agent, a tool-using assistant.")
        assert '"name": "read_file"' in system and "<tool_call>" in system
    finally:
        env.close()


# -- lifecycle --------------------------------------------------------------------------------


def _thread_names() -> set[str]:
    return {t.name for t in threading.enumerate()}


def _run_episode(hermes_home: Path) -> Path:
    env = HermesEnv(tools=["terminal", "write_file"], hermes_home=str(hermes_home), max_turns=3)
    env.reset(sample("hello", sid="ep"))
    episode_dir = Path(env.cwd)
    assert episode_dir.is_dir() and episode_dir.name.startswith("ep-")
    env.step(completion(call("write_file", path="out.txt", content="x")))
    assert (episode_dir / "out.txt").read_text() == "x"
    step = env.step(completion(call("terminal", command="cat out.txt")))
    assert json.loads(step.messages[0]["content"])["output"] == "x"
    task_id = env._task_id
    env.close()
    env.close()  # idempotent

    from tools.terminal_tool import _active_environments, resolve_task_overrides

    assert not episode_dir.exists()  # per-episode workspace removed
    assert resolve_task_overrides(task_id) == {}  # Hermes task overrides cleared
    assert task_id not in _active_environments  # no sandbox keyed to the episode survives
    with pytest.raises(RuntimeError, match="closed"):
        env.reset(sample("again"))
    return episode_dir


def test_close_cleans_up_and_leaks_no_threads(tmp_path: Path) -> None:
    hermes_home = tmp_path / "hermes-home"
    # Warm-up: Hermes starts its process-wide daemon singletons (terminal reaper, process
    # monitor) on first tool use; they are not per-episode resources.
    _run_episode(hermes_home)
    baseline = _thread_names()
    _run_episode(hermes_home)
    deadline = time.monotonic() + 10
    while _thread_names() - baseline and time.monotonic() < deadline:
        time.sleep(0.05)
    leaked = _thread_names() - baseline
    assert leaked == set(), f"threads leaked by a HermesEnv episode: {leaked}"
    assert all(t.daemon for t in threading.enumerate() if t is not threading.main_thread())
    assert (hermes_home / "skills").exists() or hermes_home.exists()  # state went to params.hermes_home


def test_reset_reuses_agent_across_episodes(workspace: Path) -> None:
    env = HermesEnv(tools=["write_file"], max_turns=2)
    try:
        env.reset(sample("one", sid="a"))
        first_dir = Path(env.cwd)
        env.step(completion(call("write_file", path="f.txt", content="1")))
        env.reset(sample("two", sid="b"))
        second_dir = Path(env.cwd)
        assert first_dir != second_dir and not first_dir.exists() and second_dir.is_dir()
        assert env.turn == 0 and env.history == []
        step = env.step(completion("done"))
        assert step.done
    finally:
        env.close()
