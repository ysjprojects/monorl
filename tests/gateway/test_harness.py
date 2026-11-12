"""``env.type: harness`` end to end: script agents run as separate processes against the gateway,
through ``run_episodes`` and through a training run."""

from __future__ import annotations

import json
import shlex
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fakes
import pytest

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, EnvironmentError_
from monorl.core.reporting import iter_metrics, iter_trajectories
from monorl.core.types import Sample, SamplingParams, Trajectory
from monorl.envs import make_env
from monorl.gateway import get_gateway
from monorl.gateway.agents.calc import SYSTEM_PROMPT
from monorl.orchestrator import run
from monorl.orchestrator.batching import continuation_parents
from monorl.orchestrator.episodes import EpisodeAborted, EpisodeSpec, run_episodes
from tests.gateway.client import wait_dead, wait_for_file, write_agent

# Two chained calls, a side conversation of its own, a third chained call; posts a reward.
MAIN_AGENT = """
messages = list(task["messages"])
first = chat(messages)
messages += [first, {"role": "user", "content": "again"}]
second = chat(messages)
chat([{"role": "user", "content": "side question"}])
messages += [second, {"role": "user", "content": task["messages"][-1]["content"]}]
final = chat(messages)
post(os.environ["MONORL_REWARD_URL"], {"reward": 0.25})
report(os.path.join(sys.argv[1], task["id"] + ".json"), {
    "task": task,
    "task_file": os.environ["MONORL_TASK_FILE"],
    "task_file_arg": sys.argv[2],
    "cwd": os.getcwd(),
    "workdir": os.environ["MONORL_WORKDIR"],
    "notes": open("notes.txt").read(),
    "greeting": os.environ["GREETING"],
})
print(final["content"])
"""

# Scores 0.5 when the agent printed the reference.
VERIFIER = """
import json, sys
reference = json.load(open(sys.argv[1]))["reference"]
answer = open(sys.argv[2]).read().strip()
print("comparing", repr(answer), "with", repr(reference))
print(0.5 if answer == reference else 0.0)
"""

# One call, then tool work that never ends (with a child process of its own).
BUSY_AGENT = """
chat(task["messages"])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
report(sys.argv[1], {"pid": os.getpid(), "child": child.pid})
time.sleep(600)
"""


def _sample(text: str = "abc", sample_id: str = "s0", **metadata: Any) -> Sample:
    return Sample(
        id=sample_id, messages=[{"role": "user", "content": text}], reference=text.upper(), metadata=metadata
    )


def _config(
    make_config: Callable[..., RunConfig],
    command: Any,
    *,
    max_turns: int | None = None,
    engine: dict[str, Any] | None = None,
    **params: Any,
) -> RunConfig:
    """A run whose environment is ``command``; the fake engine answers with the last user message
    in upper case unless ``engine`` (``rollout.extra``) says otherwise."""
    env: dict[str, Any] = {"type": "harness", "params": {"command": command, **params}}
    if max_turns is not None:
        env["max_turns"] = max_turns
    return make_config(rollout={"extra": engine or {"fake_mode": "always_right"}}, env=env)


def _episodes(
    config: RunConfig, engine: Any, samples: list[Sample], *, abort: threading.Event | None = None
) -> list[Trajectory]:
    specs = [EpisodeSpec(sample=s, group_id=f"g{i}", engine=engine) for i, s in enumerate(samples)]
    return run_episodes(
        specs,
        env_factory=lambda: make_env(config),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1, max_new_tokens=32),
        max_parallel=4,
        abort=abort,
    )


def test_agent_processes_become_trajectories(make_config: Callable[..., RunConfig], tmp_path: Path) -> None:
    agent = write_agent(tmp_path, MAIN_AGENT)
    verifier = tmp_path / "verify.py"
    verifier.write_text(VERIFIER)
    reports, template, root = tmp_path / "reports", tmp_path / "template", tmp_path / "episodes"
    reports.mkdir()
    template.mkdir()
    (template / "notes.txt").write_text("from the template")
    config = _config(
        make_config,
        ["{python}", str(agent), str(reports), "{task_file}"],
        max_turns=8,
        verifier=shlex.join(["{python}", str(verifier), "{reference_file}", "{stdout_file}"]),
        env={"GREETING": "hello {model}"},
        workdir={"root": str(root), "template": str(template)},
        timeout=120,
    )
    engine = fakes.FakeEngine(config)
    answer = "workings #### ABC"
    samples = [
        _sample(
            "abc", "s0", level=3, answer_raw=answer, response=answer, row={"answer": answer, "topic": "x"}
        ),
        _sample("def", "s1"),
    ]

    first, second = _episodes(config, engine, samples)

    assert first.sample is samples[0] and (first.group_id, second.group_id) == ("g0", "g1")
    assert [(t.thread, t.message_index) for t in first.turns] == [(0, 1), (0, 3), (1, 1), (0, 5)]
    assert [m["role"] for m in first.messages] == ["user", "assistant"] * 3
    assert [m["content"] for m in first.messages] == ["abc", "ABC", "again", "AGAIN", "abc", "ABC"]
    assert first.extra_threads == [
        [{"role": "user", "content": "side question"}, {"role": "assistant", "content": "SIDE QUESTION"}]
    ]
    assert [t.info["bridged"] for t in first.turns] == [False, True, False, True]
    assert continuation_parents(first) == [None, 0, None, 1]  # the chained calls are token-exact
    assert [t.completion.text for t in second.turns] == ["DEF", "AGAIN", "SIDE QUESTION", "DEF"]
    for trajectory in (first, second):
        # verifier (0.5: the agent printed the reference) + posted (0.25), on the last turn
        assert [t.reward for t in trajectory.turns] == [0.0, 0.0, 0.0, 0.75]
        info = trajectory.info
        assert trajectory.done and (info["exit_code"], info["timed_out"], info["num_turns"]) == (0, False, 4)
        assert info["posted_reward"] == 0.25
        assert (info["verifier"]["reward"], info["verifier"]["exit_code"]) == (0.5, 0)
    assert first.info["stdout_tail"] == "ABC\n" and first.info["stderr_tail"] == ""
    assert first.info["verifier"]["stdout_tail"] == "comparing 'ABC' with 'ABC'\n0.5\n"

    seen = json.loads((reports / "s0.json").read_text())
    # The task file has no reference, and none of the data layer's copies of the answer.
    assert seen["task"] == {
        "id": "s0",
        "messages": [{"role": "user", "content": "abc"}],
        "metadata": {"level": 3, "row": {"topic": "x"}},
    }
    assert seen["task_file_arg"] == seen["task_file"]
    assert seen["cwd"] == seen["workdir"] and seen["notes"] == "from the template"
    assert seen["greeting"] == f"hello {config.model.name_or_path}"
    # Nothing of the episodes is left behind.
    assert list(root.iterdir()) == [] and not Path(seen["task_file"]).exists()
    assert get_gateway().num_sessions == 0


def test_max_turns_caps_the_policy_calls(make_config: Callable[..., RunConfig], tmp_path: Path) -> None:
    agent = write_agent(
        tmp_path,
        """
messages = list(task["messages"])
for i in range(5):
    try:
        reply = chat(messages)
    except urllib.error.HTTPError as e:
        print(e.code, json.load(e)["error"]["code"])
        break
    messages += [reply, {"role": "user", "content": f"more{i}"}]
""",
    )
    config = _config(make_config, ["{python}", str(agent)], max_turns=2)
    engine = fakes.FakeEngine(config)
    (trajectory,) = _episodes(config, engine, [_sample()])
    assert [t.completion.text for t in trajectory.turns] == ["ABC", "MORE0"]
    assert trajectory.info["stdout_tail"] == "400 max_turns_exceeded\n"
    assert not trajectory.done and len(engine.calls) == 2


def test_timeout_kills_the_process_group_and_keeps_the_episode(
    make_config: Callable[..., RunConfig], tmp_path: Path
) -> None:
    pids = tmp_path / "pids.json"
    config = _config(make_config, ["{python}", str(write_agent(tmp_path, BUSY_AGENT)), str(pids)], timeout=4)
    (trajectory,) = _episodes(config, fakes.FakeEngine(config), [_sample()])
    assert [t.completion.text for t in trajectory.turns] == ["ABC"]
    assert trajectory.info["timed_out"] and trajectory.info["exit_code"] == -9 and not trajectory.done
    running = json.loads(pids.read_text())
    assert wait_dead(running["pid"]) and wait_dead(running["child"])


def test_abort_kills_the_agent_and_raises(make_config: Callable[..., RunConfig], tmp_path: Path) -> None:
    pids = tmp_path / "pids.json"
    config = _config(make_config, ["{python}", str(write_agent(tmp_path, BUSY_AGENT)), str(pids)])
    abort = threading.Event()
    running: dict[str, int] = {}

    def abort_during_tool_work() -> None:
        running.update(wait_for_file(pids))
        abort.set()

    threading.Thread(target=abort_during_tool_work, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(EpisodeAborted):
        _episodes(config, fakes.FakeEngine(config), [_sample()], abort=abort)
    assert time.monotonic() - started < 60  # not the agent's 600 s
    assert wait_dead(running["pid"]) and wait_dead(running["child"])
    assert get_gateway().num_sessions == 0


def test_cancelled_generation_aborts_the_episode(
    make_config: Callable[..., RunConfig], tmp_path: Path
) -> None:
    agent = write_agent(
        tmp_path,
        """
report(sys.argv[1], {"pid": os.getpid()})
chat(task["messages"])
time.sleep(600)
""",
    )
    pids = tmp_path / "pids.json"
    config = _config(make_config, ["{python}", str(agent), str(pids)])
    engine = fakes.SlowCancellableEngine(config)  # a generate call takes 60 s unless cancelled

    def cancel_generation() -> None:
        if engine.generating.wait(60):
            engine.cancel()

    threading.Thread(target=cancel_generation, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(EpisodeAborted):
        _episodes(config, engine, [_sample()])
    assert time.monotonic() - started < 50
    assert engine.cancel_calls == 1 and wait_dead(json.loads(pids.read_text())["pid"])


def test_agent_without_policy_calls_is_an_environment_error(
    make_config: Callable[..., RunConfig], tmp_path: Path
) -> None:
    agent = write_agent(tmp_path, 'sys.exit("boom: cannot find my config")\n')
    config = _config(make_config, ["{python}", str(agent)])
    with pytest.raises(
        EnvironmentError_, match=r"no policy call \(exit code 1\).*boom: cannot find my config"
    ):
        _episodes(config, fakes.FakeEngine(config), [_sample()])
    assert get_gateway().num_sessions == 0


def test_engine_failure_is_the_episode_error(make_config: Callable[..., RunConfig], tmp_path: Path) -> None:
    class BrokenEngine(fakes.FakeEngine):
        def generate(self, prompts: Any, params: SamplingParams) -> Any:
            raise RuntimeError("CUDA out of memory")

    config = _config(make_config, ["{python}", str(write_agent(tmp_path, 'chat(task["messages"])\n'))])
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _episodes(config, BrokenEngine(config), [_sample()])


def test_verifier_exit_status_is_the_reward(make_config: Callable[..., RunConfig], tmp_path: Path) -> None:
    agent = write_agent(tmp_path, 'open("answer.txt", "w").write(chat(task["messages"])["content"])\n')
    # runs in the agent's working directory; passes for the episode that answered "ABC"
    check = "import sys; sys.exit(open('answer.txt').read() != 'ABC')"
    config = _config(make_config, ["{python}", str(agent)], verifier=["{python}", "-c", check])
    passed, failed = _episodes(config, fakes.FakeEngine(config), [_sample("abc"), _sample("def", "s1")])
    assert (passed.turns[-1].reward, passed.info["verifier"]["exit_code"]) == (1.0, 0)
    assert (failed.turns[-1].reward, failed.info["verifier"]["exit_code"]) == (0.0, 1)


def test_unknown_params_are_rejected(make_config: Callable[..., RunConfig]) -> None:
    with pytest.raises(ConfigError, match="time_out"):
        make_env(_config(make_config, ["agent"], time_out=5))


@pytest.mark.parametrize(("flags", "examples"), [("", 2), (" --no-examples", 0)])
def test_calc_agent_runs_its_tool_loop(
    make_config: Callable[..., RunConfig], flags: str, examples: int
) -> None:
    script = [
        '<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>',
        "It is 42.",
    ]
    config = _config(
        make_config,
        "{python} -m monorl.gateway.agents.calc" + flags,
        engine={"fake_mode": "script", "fake_script": script},
    )
    (trajectory,) = _episodes(config, fakes.FakeEngine(config), [_sample("What is 6*7?")])
    first, second = trajectory.turns
    (call,) = first.info["tool_calls"]
    assert (call["name"], call["arguments"], call["result"]) == ("calculator", {"expression": "6*7"}, "42")
    assert (second.info["parent"], second.info["bridged"]) == (0, True)
    assert trajectory.info["stdout_tail"] == "It is 42.\n" and trajectory.info["exit_code"] == 0
    # system prompt, the worked examples (question, call, result, answer), then the task's own loop
    messages = trajectory.messages
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "tool", "assistant"] * (examples + 1)
    assert messages[-4]["content"] == "What is 6*7?" and first.message_index == len(messages) - 3
    if examples:
        worked = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "17 * 3"}}\n</tool_call>'
        assert messages[2] == {
            "role": "assistant",
            "content": worked,
            "tool_calls": [
                {"id": "call_example_1", "name": "calculator", "arguments": {"expression": "17 * 3"}}
            ],
        }
        assert [m["content"] for m in messages[3:5]] == ["51", "51"]


def test_training_run_on_a_harness_env(make_config: Callable[..., RunConfig], tmp_path: Path) -> None:
    agent = write_agent(
        tmp_path,
        """
messages = list(task["messages"])
messages += [chat(messages), {"role": "user", "content": task["messages"][-1]["content"]}]
print(chat(messages)["content"])
""",
    )
    config = _config(make_config, ["{python}", str(agent)], timeout=120)

    result = run(config)

    assert result.state == "succeeded", result.error
    assert result.final_step == 2
    run_dir = Path(result.run_dir)
    rows = [row for _, row in iter_metrics(run_dir) if row["phase"] == "train"]
    assert [row["reward/mean"] for row in rows] == [1.0, 1.0]  # the last call answers the prompt
    assert [row["tokens/prefix_mismatch"] for row in rows] == [0.0, 0.0]
    trajectories = [row for _, row in iter_trajectories(run_dir)]
    assert len(trajectories) == 2 * config.sequences_per_step()
    assert all(t["num_turns"] == 2 and t["info"]["exit_code"] == 0 for t in trajectories)
    (trainer,) = fakes.TRAINERS
    assert len(trainer.batches) == 2
    for batch in trainer.batches:
        # One sequence per episode, with both completions ("ABC" + end marker, twice) as loss positions.
        assert batch.response_mask.sum(dim=-1).tolist() == [8] * config.sequences_per_step()
        assert batch.old_logprobs is not None


@pytest.mark.slow
def test_calc_agent_trains_with_the_hf_engine_and_reference_trainer(run_config_dict: dict[str, Any]) -> None:
    """The path of ``configs/recipes/grpo_harness_calc_cpu.yaml`` on the tiny random Llama; no fakes."""
    config = RunConfig.from_dict(
        {
            **run_config_dict,
            "rollout": {**run_config_dict["rollout"], "stop": ["</tool_call>"]},
            # room for the agent's system prompt and worked examples in front of the question
            "trainer": {**run_config_dict["trainer"], "max_steps": 1, "max_seq_len": 480},
            "env": {
                "type": "harness",
                "max_turns": 2,
                "params": {
                    "command": ["{python}", "-m", "monorl.gateway.agents.calc", "--max-steps", "2"],
                    "timeout": 300,
                },
            },
        }
    )
    result = run(config)

    assert result.state == "succeeded", result.error
    run_dir = Path(result.run_dir)
    (row,) = [row for _, row in iter_metrics(run_dir) if row["phase"] == "train"]
    assert row["tokens/truncated"] == 0.0 and row["tokens/prompt_mean"] > 100
    assert 0 < row["tokens/response_mean"] <= 2 * config.rollout.max_new_tokens
    trajectories = [row for _, row in iter_trajectories(run_dir)]
    assert len(trajectories) == config.sequences_per_step()
    assert all(1 <= t["num_turns"] <= 2 and t["info"]["exit_code"] == 0 for t in trajectories)
