"""``sandbox.backend: docker`` against a real Docker daemon. Skipped when no daemon answers or
the test image is not local::

    .venv/bin/python -m pytest tests/envs/test_sandbox_docker.py -q

The image (``MONORL_SANDBOX_TEST_IMAGE``, default ``monorl:cpu``: ``docker build -f
docker/Dockerfile.cpu -t monorl:cpu .``) needs ``python3`` and ``bash``; nothing is pulled. The
example-task test also needs ``python:3.12-slim-bookworm`` locally and is skipped without it.
Everything the tests start or build is removed again: containers by the episode, images built
from task Dockerfiles (``monorl-sandbox:<hash>``) by the ``built_images`` fixture.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from monorl.config.schema import DataConfig, RunConfig
from monorl.core.errors import EpisodeAborted
from monorl.core.types import Sample, SamplingParams, Trajectory
from monorl.data import open_source
from monorl.envs import make_env, sandbox
from monorl.gateway import get_gateway
from monorl.orchestrator.episodes import EpisodeSpec, run_episodes
from tests.envs.harbor_tasks import write_task
from tests.gateway.client import wait_for_file, write_agent

IMAGE = os.environ.get("MONORL_SANDBOX_TEST_IMAGE", "monorl:cpu")
EXAMPLE_IMAGE = "python:3.12-slim-bookworm"
EXAMPLES = Path(__file__).parents[2] / "examples" / "sandbox"


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120)


def _has_image(image: str) -> bool:
    return _docker("image", "inspect", "--format", "{{.Id}}", image).returncode == 0


@pytest.fixture(scope="module", autouse=True)
def _daemon() -> None:
    if shutil.which("docker") is None:
        pytest.skip("no docker CLI")
    try:
        present = _has_image(IMAGE)
    except subprocess.TimeoutExpired:
        pytest.skip("the Docker daemon does not answer")
    if not present:
        pytest.skip(f"no Docker daemon answers, or it has no image {IMAGE!r} (MONORL_SANDBOX_TEST_IMAGE)")


@pytest.fixture
def built_images() -> Iterator[None]:
    """Remove the images the test built from task Dockerfiles."""

    def tags() -> set[str]:
        return set(_docker("images", "--format", "{{.Repository}}:{{.Tag}}", "monorl-sandbox").stdout.split())

    before = tags()
    yield
    for tag in sorted(tags() - before):
        _docker("rmi", tag)
        for key in [key for key in sandbox._images if key[1] == tag]:
            del sandbox._images[key]  # the sandbox module remembers which images exist


def _own_containers() -> list[str]:
    """Containers of this process that still exist (running or not)."""
    return _docker("ps", "-aq", "--filter", f"label={sandbox.OWNER_LABEL}={sandbox._owner()}").stdout.split()


def _sample(text: str = "abc", sample_id: str = "s0") -> Sample:
    return Sample(id=sample_id, messages=[{"role": "user", "content": text}], reference=text.upper())


def _config(make_config: Callable[..., RunConfig], command: Any, **params: Any) -> RunConfig:
    engine = params.pop("engine", {"fake_mode": "always_right"})
    box = {"backend": "docker", **params.pop("sandbox", {})}
    env = {"type": "harness", "params": {"command": command, "sandbox": box, **params}}
    return make_config(rollout={"extra": engine}, env=env)


def _episodes(
    config: RunConfig, samples: list[Sample], *, abort: threading.Event | None = None
) -> list[Trajectory]:
    import fakes

    engine = fakes.FakeEngine(config)
    specs = [EpisodeSpec(sample=s, group_id=f"g{i}", engine=engine) for i, s in enumerate(samples)]
    return run_episodes(
        specs,
        env_factory=lambda: make_env(config),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1, max_new_tokens=32),
        max_parallel=4,
        abort=abort,
    )


@pytest.fixture
def shared(tmp_path: Path) -> dict[str, Any]:
    """Directories mounted into the containers: ``/agent`` (scripts, read-only) and ``/reports``
    (what the agents tell the test)."""
    agent, reports = tmp_path / "agent", tmp_path / "reports"
    agent.mkdir()
    reports.mkdir()
    reports.chmod(0o777)
    return {"agent": agent, "reports": reports, "mounts": [f"{agent}:/agent:ro", f"{reports}:/reports"]}


def test_episode_in_a_real_container(
    make_config: Callable[..., RunConfig], shared: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOST_SECRET", "do not leak")
    write_agent(
        shared["agent"],
        """
reply = chat(task["messages"])
open("answer.txt", "w").write(reply["content"])
report("/reports/seen.json", {
    "task": task, "cwd": os.getcwd(), "env": dict(os.environ), "argv": sys.argv, "uid": os.getuid(),
    "host_python": os.path.exists(sys.argv[2]),
})
print(reply["content"])
""",
    )
    check = (
        "import json, sys; reference = json.load(open(sys.argv[1]))['reference']; "
        "print(float(open(sys.argv[2]).read().strip() == reference == open('answer.txt').read()))"
    )
    config = _config(
        make_config,
        ["{python}", "/agent/agent.py", "{task_file}", sys.executable],
        verifier=["{python}", "-c", check, "{reference_file}", "{stdout_file}"],
        sandbox={"image": IMAGE, "mounts": shared["mounts"], "cpus": 1, "memory": "512m", "pids_limit": 128},
    )

    (trajectory,) = _episodes(config, [_sample()])

    # the agent reached the gateway from inside the container, and the verifier its files
    assert [t.completion.text for t in trajectory.turns] == ["ABC"]
    assert trajectory.turns[-1].reward == 1.0 and trajectory.info["verifier"]["stderr_tail"] == ""
    info = trajectory.info["sandbox"]
    assert (info["backend"], info["image"], info["cpus"], info["memory"]) == ("docker", IMAGE, 1.0, "512m")
    assert info["cpu_seconds"] > 0 and info["memory_peak_bytes"] > 1 << 20 and info["oom_killed"] is False
    seen = json.loads((shared["reports"] / "seen.json").read_text())
    assert seen["cwd"] == "/monorl/work" == seen["env"]["MONORL_WORKDIR"]
    assert seen["argv"][1] == "/monorl/task.json" == seen["env"]["MONORL_TASK_FILE"]
    assert seen["task"]["messages"] == [{"role": "user", "content": "abc"}]
    assert "HOST_SECRET" not in seen["env"] and not seen["host_python"]
    assert _own_containers() == [] and get_gateway().num_sessions == 0


def test_timeout_and_abort_remove_the_container(
    make_config: Callable[..., RunConfig], shared: dict[str, Any]
) -> None:
    write_agent(
        shared["agent"],
        """
chat(task["messages"])
subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
report("/reports/" + task["id"] + ".json", {"pid": os.getpid()})
time.sleep(600)
""",
    )
    box = {"image": IMAGE, "mounts": shared["mounts"]}
    # 1.0 when a process of the agent script is still running in the container
    alive = (
        "import os; name = b'agent' + b'.py'; pids = [p for p in os.listdir('/proc') if p.isdigit()]; "
        "print(float(any(name in open('/proc/' + p + '/cmdline', 'rb').read() for p in pids)))"
    )

    # timeout: the episode counts, and by the time the verifier looks nothing of the agent runs
    config = _config(
        make_config,
        ["{python}", "/agent/agent.py"],
        timeout=5,
        verifier=["{python}", "-c", alive],
        sandbox=box,
    )
    started = time.monotonic()
    (trajectory,) = _episodes(config, [_sample()])
    assert time.monotonic() - started < 60
    assert trajectory.info["timed_out"] and not trajectory.done
    assert trajectory.info["verifier"]["reward"] == 0.0 and trajectory.info["verifier"]["exit_code"] == 0
    assert _own_containers() == []

    # abort: the episode raises and the container is gone
    abort = threading.Event()

    def abort_during_tool_work() -> None:
        wait_for_file(shared["reports"] / "s1.json", timeout=60)
        abort.set()

    threading.Thread(target=abort_during_tool_work, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(EpisodeAborted):
        _episodes(
            _config(make_config, ["{python}", "/agent/agent.py"], sandbox=box),
            [_sample("def", "s1")],
            abort=abort,
        )
    assert time.monotonic() - started < 60
    assert _own_containers() == [] and get_gateway().num_sessions == 0


def test_memory_limit_is_enforced_and_reported(
    make_config: Callable[..., RunConfig], shared: dict[str, Any]
) -> None:
    write_agent(
        shared["agent"],
        """
chat(task["messages"])
hog = bytearray(1 << 30)
for i in range(0, len(hog), 4096):
    hog[i] = 1
print("survived")
""",
    )
    config = _config(
        make_config,
        ["{python}", "/agent/agent.py"],
        sandbox={"image": IMAGE, "mounts": shared["mounts"], "memory": "96m"},
    )
    (trajectory,) = _episodes(config, [_sample()])
    assert trajectory.info["exit_code"] == 137 and "survived" not in trajectory.info["stdout_tail"]
    assert trajectory.info["sandbox"]["oom_killed"] is True
    assert _own_containers() == []


# Answers, then tries to grade itself.
TASK_AGENT = """
found = {path: os.path.exists(path) for path in ("/tests", "/solution", "/entrypoint-ran")}
reply = chat(task["messages"])
open("answer.txt", "w").write(reply["content"])
os.makedirs("/tests")
open("/tests/planted.txt", "w").write("x")
open("/logs/verifier/reward.txt", "w").write("1")
report("/reports/" + task["id"] + ".json", {
    "task": task, "cwd": os.getcwd(), "found": found, "files": sorted(os.listdir(".")),
    "flavour": os.environ.get("FLAVOUR"),
})
"""

# The image has an entrypoint of its own (Harbor keeps it) and a file in its WORKDIR.
TASK_DOCKERFILE = f"""FROM {IMAGE}
RUN printf '#!/bin/sh\\necho started > /entrypoint-ran\\nexec "$@"\\n' > /entrypoint.sh && chmod +x /entrypoint.sh
WORKDIR /task
COPY seed.txt /task/seed.txt
ENTRYPOINT ["/entrypoint.sh"]
"""

TASK_TOML = """
schema_version = "1.4"
[agent]
timeout_sec = 120
[verifier]
timeout_sec = 60
env = {{ EXPECTED = "{expected}" }}
[environment]
cpus = 1
memory_mb = 512
env = {{ FLAVOUR = "${{HOST_FLAVOUR:-plain}}" }}
[environment.healthcheck]
command = "test -f /entrypoint-ran"
interval_sec = 0.2
start_interval_sec = 0.2
"""

# Harbor's contract with real container paths.
TASK_TEST_SH = """#!/bin/bash
if [ -e /tests/planted.txt ] || [ -e /logs/verifier/reward.txt ]; then exit 3; fi
if [ "$(cat /task/answer.txt)" = "$EXPECTED" ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
"""


def test_task_directories_in_real_containers(
    make_config: Callable[..., RunConfig], shared: dict[str, Any], tmp_path: Path, built_images: None
) -> None:
    for name, expected in (("right", "ABC"), ("wrong", "something else")):
        write_task(
            tmp_path / "tasks" / name,
            TASK_TOML.format(expected=expected),
            test_sh=TASK_TEST_SH,
            dockerfile=TASK_DOCKERFILE,
        )
    write_agent(shared["agent"], TASK_AGENT)
    samples = list(open_source(DataConfig(uri=str(tmp_path / "tasks"), shuffle=False)))
    config = _config(make_config, ["{python}", "/agent/agent.py"], sandbox={"mounts": shared["mounts"]})

    right, wrong = _episodes(config, samples)

    # the task's tests decide; what the agent planted under /tests and /logs/verifier is gone
    assert right.info["verifier"]["rewards"] == {"reward": 1.0} and right.turns[-1].reward == 1.0
    assert wrong.info["verifier"]["rewards"] == {"reward": 0.0} and wrong.turns[-1].reward == 0.0
    # both tasks have the same environment: one image, built from the Dockerfile and kept
    image = right.info["sandbox"]["image"]
    assert (
        image == wrong.info["sandbox"]["image"] and image.startswith("monorl-sandbox:") and _has_image(image)
    )
    assert (right.info["sandbox"]["cpus"], right.info["sandbox"]["memory"]) == (1.0, "512m")
    for name in ("right", "wrong"):
        seen = json.loads((shared["reports"] / f"{name}.json").read_text())
        # the image's WORKDIR and entrypoint are in effect; tests and solution were not there
        assert seen["cwd"] == "/task" and seen["files"] == ["answer.txt", "seed.txt"]
        assert seen["found"] == {"/tests": False, "/solution": False, "/entrypoint-ran": True}
        assert seen["flavour"] == "plain" and seen["task"]["metadata"]["row"] == {"id": name}
    assert _own_containers() == [] and get_gateway().num_sessions == 0


def test_example_tasks_with_the_example_agent(
    make_config: Callable[..., RunConfig], built_images: None
) -> None:
    """The shipped tasks, solved (or not) through ``shell_agent.py`` by a scripted policy."""
    if not _has_image(EXAMPLE_IMAGE):
        pytest.skip(f"the example tasks need the image {EXAMPLE_IMAGE} (docker pull {EXAMPLE_IMAGE})")
    samples = {s.id: s for s in open_source(DataConfig(uri=str(EXAMPLES / "tasks"), shuffle=False))}
    commands = {
        ("hello-file", "echo 'Hello, world!' > /app/hello.txt"): 1.0,
        ("hello-file", "echo 'Hello' > /app/hello.txt"): 0.0,
        ("count-lines", "wc -l < words.txt > answer.txt"): 1.0,
        ("sort-numbers", "sort -n numbers.txt > sorted.txt"): 1.0,
        ("sort-numbers", "cp numbers.txt sorted.txt"): 0.5,  # the right numbers, unsorted
    }
    for (task, command), reward in commands.items():
        call = json.dumps({"name": "shell", "arguments": {"command": command}})
        config = _config(
            make_config,
            ["{python}", "/opt/monorl-agents/shell_agent.py", "--max-steps", "3"],
            engine={"fake_mode": "script", "fake_script": [f"<tool_call>{call}</tool_call>", "DONE"]},
            sandbox={"mounts": [f"{EXAMPLES / 'agents'}:/opt/monorl-agents:ro"]},
        )
        (trajectory,) = _episodes(config, [samples[task]])
        assert trajectory.turns[-1].reward == reward, (task, command, trajectory.info)
        assert trajectory.turns[0].info["tool_calls"][0]["result"] == "exit status 0"
        assert trajectory.info["stdout_tail"] == "DONE\n" and trajectory.info["task"] == f"monorl/{task}"
    assert _own_containers() == []
