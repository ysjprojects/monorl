"""``env.type: harness`` with ``sandbox.backend: docker``: the episode lifecycle, driven through a
stand-in ``docker`` executable (``fake_docker_cli.py``) that runs container commands on this
machine. ``test_sandbox_docker.py`` repeats the essentials against a real daemon."""

from __future__ import annotations

import json
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, EnvironmentError_, EpisodeAborted
from monorl.core.types import Sample, SamplingParams, Trajectory
from monorl.envs import make_env, sandbox
from monorl.envs.sandbox import Build, build_tag, daemon_in_vm, gateway_address
from monorl.gateway import Gateway, get_gateway
from monorl.orchestrator.episodes import EpisodeSpec, run_episodes
from tests.envs.conftest import FakeDocker
from tests.envs.harbor_tasks import TEST_SH, write_task
from tests.gateway.client import wait_dead, wait_for_file, write_agent

# One call; leaves the answer in the working directory and tells the test what it saw.
AGENT = """
reply = chat(task["messages"])
open("answer.txt", "w").write(reply["content"])
report(sys.argv[1], {
    "task": task, "argv": sys.argv, "cwd": os.getcwd(), "env": dict(os.environ),
    "files": sorted(os.listdir(".")),
})
print(reply["content"])
"""

# Scores 1.0 when the agent printed the reference and its answer file is in the working directory.
VERIFIER = """
import json, sys
reference = json.load(open(sys.argv[1]))["reference"]
print(float(open(sys.argv[2]).read().strip() == reference == open("answer.txt").read()))
"""

# One call, then work that never ends, with a child process of its own.
BUSY_AGENT = """
chat(task["messages"])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
report(sys.argv[1], {"pid": os.getpid(), "child": child.pid})
time.sleep(600)
"""

# Scores 1.0 when the process whose pid is in the file named by argv[1] under "child" is alive.
ALIVE_VERIFIER = """
import json, os, sys
try:
    os.kill(json.load(open(sys.argv[1]))["child"], 0)
    print(1.0)
except ProcessLookupError:
    print(0.0)
"""


def _sample(text: str = "abc", sample_id: str = "s0") -> Sample:
    return Sample(id=sample_id, messages=[{"role": "user", "content": text}], reference=text.upper())


def _config(make_config: Callable[..., RunConfig], command: Any, **params: Any) -> RunConfig:
    """A harness run in ``box`` containers of the stand-in daemon; the fake engine answers with
    the last user message in upper case. Agents run on this machine, so they are told to reach
    the gateway on loopback unless the test is about that."""
    engine = params.pop("engine", {"fake_mode": "always_right"})
    box = {"backend": "docker", "image": "box", "python": sys.executable, **params.pop("sandbox", {})}
    params.setdefault("gateway", {"host": "127.0.0.1"})
    env = {"type": "harness", "params": {"command": command, "sandbox": box, **params}}
    return make_config(rollout={"extra": engine}, env=env)


def _episodes(
    config: RunConfig, samples: list[Sample], *, abort: threading.Event | None = None, engine: Any = None
) -> list[Trajectory]:
    import fakes

    engine = engine or fakes.FakeEngine(config)
    specs = [EpisodeSpec(sample=s, group_id=f"g{i}", engine=engine) for i, s in enumerate(samples)]
    return run_episodes(
        specs,
        env_factory=lambda: make_env(config),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1, max_new_tokens=32),
        max_parallel=4,
        abort=abort,
    )


def _lifecycle(docker: FakeDocker) -> list[str]:
    return [call["command"] for call in docker.calls("run", "exec", "restart", "cp", "rm")]


def test_agent_and_verifier_run_in_one_container(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOST_SECRET", "do not leak")
    monkeypatch.setenv("FAKE_DOCKER_CGROUP", "1500000,1048576,0")
    agent, report, root = write_agent(tmp_path, AGENT), tmp_path / "report.json", tmp_path / "episodes"
    verifier = tmp_path / "verify.py"
    verifier.write_text(VERIFIER)
    config = _config(
        make_config,
        ["{python}", str(agent), str(report), "{task_file}", "{workdir}"],
        verifier=["{python}", str(verifier), "{reference_file}", "{stdout_file}"],
        env={"GREETING": "hello {model}"},
        workdir={"root": str(root)},
        sandbox={"cpus": 1.5, "memory": "256m", "user": "1000"},
    )

    (trajectory,) = _episodes(config, [_sample()])

    assert [t.completion.text for t in trajectory.turns] == ["ABC"]
    assert trajectory.turns[-1].reward == 1.0 and trajectory.info["verifier"]["exit_code"] == 0
    assert trajectory.done and trajectory.info["exit_code"] == 0 and trajectory.info["stdout_tail"] == "ABC\n"
    container = trajectory.info["sandbox"].pop("container")
    assert trajectory.info["sandbox"] == {
        "backend": "docker",
        "image": "box",
        "cpus": 1.5,
        "memory": "256m",
        "cpu_seconds": 1.5,
        "memory_peak_bytes": 1048576,
        "oom_killed": False,
    }
    # Inside the container the paths the agent is given are the container's ...
    (run,) = docker.calls("run")
    assert run["argv"][run["argv"].index("--cpus") + 1] == "1.5"
    assert run["argv"][run["argv"].index("--memory") + 1] == "256m"
    started = docker.calls("exec")[0]
    assert started["argv"][-5:] == [
        sys.executable,
        str(agent),
        str(report),
        "/monorl/task.json",
        "/monorl/work",
    ]
    assert (started["env"]["MONORL_TASK_FILE"], started["env"]["MONORL_WORKDIR"]) == (
        "/monorl/task.json",
        "/monorl/work",
    )
    # ... and they lead to the episode's files: the task, a working directory of its own.
    seen = json.loads(report.read_text())
    assert seen["task"]["messages"] == [{"role": "user", "content": "abc"}]
    assert seen["cwd"].endswith("/work") and seen["files"] == ["answer.txt"]
    assert seen["env"]["FAKE_DOCKER_USER"] == "1000"
    assert seen["env"]["GREETING"] == f"hello {config.model.name_or_path}"
    # Nothing of monorl's own environment is in the container.
    assert "HOST_SECRET" not in seen["env"] and "HOME" not in seen["env"]
    # The session id is the credential: it is in no command line of the host.
    session = seen["env"]["OPENAI_API_KEY"]
    assert session == seen["env"]["MONORL_SESSION"] and len(session) >= 32
    assert not any(session in arg for call in docker.calls() for arg in call["argv"])
    # agent, resource usage, restart (nothing of the agent runs any more), verifier, removal
    assert _lifecycle(docker) == ["run", "exec", "exec", "restart", "exec", "rm"]
    assert docker.calls("rm")[0]["argv"][-1] == container and docker.containers() == []
    assert list(root.iterdir()) == [] and get_gateway().num_sessions == 0


def test_docker_engine_opens_the_episode_directory_before_removing_the_container(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docker Engine on Linux, monorl not root: what the container wrote into the episode
    directory belongs to the container's users, so it is opened up from inside before the
    container goes, or the directory could not be deleted."""
    monkeypatch.setattr(sandbox, "PLATFORM", "linux")
    monkeypatch.setattr("os.getuid", lambda: 1000)
    agent, root = write_agent(tmp_path, 'print(chat(task["messages"])["content"])\n'), tmp_path / "episodes"
    config = _config(make_config, ["{python}", str(agent)], workdir={"root": str(root)})

    (trajectory,) = _episodes(config, [_sample()])

    assert trajectory.info["stdout_tail"] == "ABC\n"
    *_, opened, removed = docker.calls("exec", "rm")
    assert opened["command"] == "exec" and opened["argv"][-4:] == ["chmod", "-R", "a+rwX", "/monorl"]
    assert opened["argv"][opened["argv"].index("-u") + 1] == "0"
    assert removed["command"] == "rm" and docker.containers() == [] and list(root.iterdir()) == []


def test_container_agents_are_told_the_name_containers_reach_the_host_under(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docker Engine on Linux: ``host.docker.internal`` is mapped to the bridge gateway for the
    container; ``env.params.gateway`` picks the bind address and port."""
    monkeypatch.setattr(sandbox, "PLATFORM", "linux")
    monkeypatch.setenv("FAKE_DOCKER_OS", "Ubuntu 24.04.1 LTS")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # what the container's resolver would do with the name
    body = 'os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_BASE_URL"].replace("host.docker.internal", "127.0.0.1")\n'
    agent = write_agent(tmp_path, body + 'print(chat(task["messages"])["content"])\n')
    config = _config(make_config, ["{python}", str(agent)], gateway={"bind": "127.0.0.1", "port": port})

    (trajectory,) = _episodes(config, [_sample()])

    assert trajectory.info["stdout_tail"] == "ABC\n"
    (run,) = docker.calls("run")
    assert run["argv"][run["argv"].index("--add-host") + 1] == "host.docker.internal:host-gateway"
    env = docker.calls("exec")[0]["env"]
    assert env["OPENAI_BASE_URL"] == f"http://host.docker.internal:{port}/s/{env['MONORL_SESSION']}/v1"
    assert env["MONORL_REWARD_URL"] == f"http://host.docker.internal:{port}/s/{env['MONORL_SESSION']}/reward"
    assert "host.docker.internal" in env["NO_PROXY"].split(",")


@pytest.mark.parametrize(
    ("operating_system", "platform", "network", "expected"),
    [
        ("Docker Desktop", "darwin", None, ("127.0.0.1", "host.docker.internal")),
        ("Docker Desktop", "linux", None, ("127.0.0.1", "host.docker.internal")),
        ("Ubuntu 24.04.1 LTS", "darwin", None, ("127.0.0.1", "host.docker.internal")),  # colima, a remote VM
        ("Ubuntu 24.04.1 LTS", "linux", None, ("0.0.0.0", "host.docker.internal")),
        ("Ubuntu 24.04.1 LTS", "linux", "host", ("127.0.0.1", "127.0.0.1")),
    ],
)
def test_gateway_address_follows_the_platform(
    operating_system: str, platform: str, network: str | None, expected: tuple[str, str]
) -> None:
    in_vm = daemon_in_vm(operating_system, platform)
    assert gateway_address(in_vm=in_vm, network=network) == expected


def test_anthropic_variables_are_set_when_the_gateway_serves_that_api(
    make_config: Callable[..., RunConfig], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Gateway,
        "anthropic_base_url",
        lambda self, session, host=None: self.session_url(session, host),
        raising=False,
    )
    agent, report = write_agent(tmp_path, AGENT), tmp_path / "report.json"
    env = {
        "type": "harness",
        "params": {"command": ["{python}", str(agent), str(report), "{anthropic_base_url}"]},
    }
    config = make_config(rollout={"extra": {"fake_mode": "always_right"}}, env=env)
    _episodes(config, [_sample()])
    seen = json.loads(report.read_text())
    assert (
        seen["env"]["ANTHROPIC_BASE_URL"]
        == seen["env"]["OPENAI_BASE_URL"].removesuffix("/v1")
        == seen["argv"][2]
    )
    assert seen["env"]["ANTHROPIC_API_KEY"] == seen["env"]["MONORL_SESSION"]


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The options every ``Gateway.open_session`` call of the test was made with."""
    seen: list[dict[str, Any]] = []
    open_session = Gateway.open_session

    def recording(self: Gateway, **options: Any) -> Any:
        seen.append(options)
        return open_session(self, **options)

    monkeypatch.setattr(Gateway, "open_session", recording)
    return seen


def _local_config(make_config: Callable[..., RunConfig], tmp_path: Path, **params: Any) -> RunConfig:
    command = [sys.executable, str(write_agent(tmp_path, 'chat(task["messages"])\n'))]
    env = {"type": "harness", "params": {"command": command, **params}}
    return make_config(rollout={"extra": {"fake_mode": "always_right"}}, env=env)


@pytest.mark.parametrize(
    ("sandboxed", "configured", "fetch_images"),
    [(False, None, True), (True, None, False), (True, True, True), (False, False, False)],
)
def test_sandboxed_agents_cannot_make_the_host_fetch_image_urls_unless_allowed(
    make_config: Callable[..., RunConfig],
    docker: FakeDocker,
    tmp_path: Path,
    opened: list[dict[str, Any]],
    sandboxed: bool,
    configured: bool | None,
    fetch_images: bool,
) -> None:
    if sandboxed:
        agent = write_agent(tmp_path, 'chat(task["messages"])\n')
        config = _config(make_config, ["{python}", str(agent)], fetch_images=configured)
    else:
        config = _local_config(make_config, tmp_path, fetch_images=configured)
    _episodes(config, [_sample()])
    assert [options["fetch_images"] for options in opened] == [fetch_images]


def test_the_policy_processor_is_passed_for_engines_that_do_not_bring_theirs(
    make_config: Callable[..., RunConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opened: list[dict[str, Any]],
) -> None:
    import fakes

    from monorl.envs import harness
    from monorl.rollout import media

    processor, loads = object(), []
    monkeypatch.setattr(harness, "_processors", {})
    monkeypatch.setattr(
        media, "load_processor", lambda config: loads.append(config.model.name_or_path) or processor
    )
    config = _local_config(make_config, tmp_path)

    # an engine without a `processor` (vLLM, SGLang): the policy's, loaded once per process
    _episodes(config, [_sample(), _sample("def", "s1")])
    assert [options["processor"] for options in opened] == [processor, processor]
    assert loads == [config.model.name_or_path]

    # an engine that says what its processor is (here: none, a text-only policy) is left to the gateway
    class TextOnlyEngine(fakes.FakeEngine):
        processor = None

    _episodes(config, [_sample()], engine=TextOnlyEngine(config))
    assert "processor" not in opened[-1] and len(loads) == 1

    # a text-only policy has no processor to pass
    monkeypatch.setattr(harness, "_processors", {})
    monkeypatch.setattr(media, "load_processor", lambda config: None)
    _episodes(config, [_sample()])
    assert "processor" not in opened[-1]


def test_timeout_stops_the_agent_before_the_verifier_runs(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path
) -> None:
    pids, verifier = tmp_path / "pids.json", tmp_path / "alive.py"
    verifier.write_text(ALIVE_VERIFIER)
    config = _config(
        make_config,
        ["{python}", str(write_agent(tmp_path, BUSY_AGENT)), str(pids)],
        timeout=3,
        verifier=["{python}", str(verifier), str(pids)],
    )
    (trajectory,) = _episodes(config, [_sample()])
    assert [t.completion.text for t in trajectory.turns] == ["ABC"]
    assert trajectory.info["timed_out"] and not trajectory.done
    # the verifier found nothing of the agent alive: the container was restarted first
    assert trajectory.info["verifier"]["reward"] == 0.0 and trajectory.turns[-1].reward == 0.0
    assert _lifecycle(docker) == ["run", "exec", "exec", "restart", "exec", "rm"]
    running = json.loads(pids.read_text())
    assert wait_dead(running["pid"]) and wait_dead(running["child"])
    assert docker.containers() == [] and get_gateway().num_sessions == 0


@pytest.mark.parametrize("keep_processes", [False, True])
def test_processes_the_agent_left_running_are_stopped_unless_kept(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path, keep_processes: bool
) -> None:
    agent = write_agent(
        tmp_path,
        """
chat(task["messages"])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
report(sys.argv[1], {"child": child.pid})
""",
    )
    pids, verifier = tmp_path / "pids.json", tmp_path / "alive.py"
    verifier.write_text(ALIVE_VERIFIER)
    config = _config(
        make_config,
        ["{python}", str(agent), str(pids)],
        verifier=["{python}", str(verifier), str(pids)],
        sandbox={"keep_processes": keep_processes},
    )
    (trajectory,) = _episodes(config, [_sample()])
    # the verifier saw the agent's service exactly when the sandbox keeps processes
    assert trajectory.info["verifier"]["reward"] == float(keep_processes)
    assert ("restart" in _lifecycle(docker)) is not keep_processes
    # either way nothing outlives the episode
    assert wait_dead(json.loads(pids.read_text())["child"]) and docker.containers() == []


def test_abort_removes_the_container_and_raises(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path
) -> None:
    pids, root = tmp_path / "pids.json", tmp_path / "episodes"
    config = _config(
        make_config,
        ["{python}", str(write_agent(tmp_path, BUSY_AGENT)), str(pids)],
        workdir={"root": str(root)},
    )
    abort = threading.Event()
    running: dict[str, int] = {}

    def abort_during_tool_work() -> None:
        running.update(wait_for_file(pids))
        abort.set()

    threading.Thread(target=abort_during_tool_work, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(EpisodeAborted):
        _episodes(config, [_sample()], abort=abort)
    assert time.monotonic() - started < 60  # not the agent's 600 s
    assert wait_dead(running["pid"]) and wait_dead(running["child"])
    assert _lifecycle(docker)[-1] == "rm" and docker.containers() == []
    assert list(root.iterdir()) == [] and get_gateway().num_sessions == 0


@pytest.mark.parametrize(
    ("failing", "command", "error"),
    [
        (
            "run",
            ["{python}", "-c", "pass"],
            r"cannot start a sandbox container from 'box'.*run failed \(injected\)",
        ),
        (
            "",
            ["{python}", "-c", "import sys; sys.exit('boom: no config')"],
            r"no policy call \(exit code 1\).*boom",
        ),
        ("", ["no-such-agent"], r"no policy call \(exit code 127\).*unable to start container process"),
        ("restart", None, r"cannot restart the container.*restart failed \(injected\)"),
        ("cp", None, r"cannot copy .*template to /srv.*cp failed \(injected\)"),
    ],
)
def test_failures_are_environment_errors_and_leave_nothing_behind(
    make_config: Callable[..., RunConfig],
    docker: FakeDocker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing: str,
    command: list[str] | None,
    error: str,
) -> None:
    monkeypatch.setenv("FAKE_DOCKER_FAIL", failing)
    root, template = tmp_path / "episodes", tmp_path / "template"
    template.mkdir()
    params: dict[str, Any] = {"workdir": {"root": str(root)}}
    if command is None:  # a healthy agent: the failure is the sandbox's
        command = ["{python}", str(write_agent(tmp_path, AGENT)), str(tmp_path / "report.json")]
        params["verifier"] = ["{python}", "-c", "pass"]
    if failing == "cp":
        params.update(workdir={"root": str(root), "template": str(template)}, sandbox={"workdir": "/srv"})
    with pytest.raises(EnvironmentError_, match=error):
        _episodes(_config(make_config, command, **params), [_sample()])
    assert docker.containers() == [] and docker.calls()[-1]["command"] == "rm"
    assert list(root.iterdir()) == [] and get_gateway().num_sessions == 0


def test_missing_cli_and_daemon_are_reported_at_once(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(make_config, ["{python}", "-c", "pass"])
    monkeypatch.setenv("FAKE_DOCKER_DAEMON", "down")
    started = time.monotonic()
    with pytest.raises(EnvironmentError_, match="Docker daemon does not answer.*Cannot connect"):
        _episodes(config, [_sample()])
    assert time.monotonic() - started < 20 and get_gateway().num_sessions == 0

    empty = tmp_path / "no-docker-here"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(ConfigError, match="no `docker` executable on PATH"):
        make_env(config)


def test_images_are_pulled_or_built_once_and_the_image_workdir_can_be_used(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path
) -> None:
    agent = write_agent(
        tmp_path, AGENT.replace("sys.argv[1]", 'os.path.join(sys.argv[1], task["id"] + ".json")')
    )
    reports, template, context = tmp_path / "reports", tmp_path / "template", tmp_path / "context"
    for directory in (reports, template, context):
        directory.mkdir()
    (template / "notes.txt").write_text("from the template")
    (context / "Dockerfile").write_text("FROM box\nWORKDIR /srv/app\nCOPY data.txt .\n")
    (context / "data.txt").write_text("baked in")
    samples = [_sample("abc", "s0"), _sample("def", "s1")]

    # an image that is not local yet is pulled, once for both episodes
    pulled = _config(
        make_config, ["{python}", str(agent), str(reports)], sandbox={"image": "registry/tool:1"}
    )
    _episodes(pulled, samples)
    assert [call["argv"][-1] for call in docker.calls("pull")] == ["registry/tool:1"]

    # a build context is built once; the image's WORKDIR is the working directory, the template is copied in
    built = _config(
        make_config,
        ["{python}", str(agent), str(reports)],
        workdir={"template": str(template)},
        sandbox={"image": None, "build": {"context": str(context)}, "workdir": "image"},
    )
    first, second = _episodes(built, samples)
    assert len(docker.calls("build")) == 1
    assert first.info["sandbox"]["image"] == second.info["sandbox"]["image"] == build_tag(Build(str(context)))
    for sample in samples:
        seen = json.loads((reports / f"{sample.id}.json").read_text())
        assert seen["cwd"].endswith("/srv/app") and seen["env"]["MONORL_WORKDIR"].endswith("/srv/app")
        assert seen["files"] == ["answer.txt", "data.txt", "notes.txt"]
    told = [call["env"]["MONORL_WORKDIR"] for call in docker.calls("exec") if "MONORL_WORKDIR" in call["env"]]
    assert told[-2:] == ["/srv/app", "/srv/app"]

    # another context is another image
    (context / "data.txt").write_text("changed")
    assert build_tag(Build(str(context))) != first.info["sandbox"]["image"]


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"backend": "podman"}, "backend must be one of"),
        ({"backend": "docker", "imgae": "box"}, r"unknown env.params.sandbox options \['imgae'\]"),
        ({"image": "box"}, "need `backend: docker`"),
        ({"backend": "docker", "image": "box", "build": "."}, "`image` or `build`, not both"),
        ({"backend": "docker", "image": "box", "network": "none"}, "without a way to the policy"),
        ({"backend": "docker", "image": "box", "mounts": ["/no/such/dir:/data"]}, "does not exist"),
        ({"backend": "docker", "image": "box", "workdir": "relative/dir"}, "absolute container path"),
        ({"backend": "docker", "image": "box", "cpus": 0}, "positive number"),
    ],
)
def test_sandbox_options_are_validated(
    make_config: Callable[..., RunConfig], docker: FakeDocker, options: dict[str, Any], error: str
) -> None:
    env = {"type": "harness", "params": {"command": ["agent"], "sandbox": options}}
    with pytest.raises(ConfigError, match=error):
        make_env(make_config(env=env))


def test_containers_are_removed_when_monorl_is_killed(docker: FakeDocker, tmp_path: Path) -> None:
    """SIGKILL leaves monorl no chance to clean up: its watchdog process removes the container."""
    script = tmp_path / "owner.py"
    script.write_text(
        """
import sys, time
from pathlib import Path
from monorl.envs.sandbox import SandboxConfig, make_sandbox

box = make_sandbox(SandboxConfig.from_params({"backend": "docker", "image": "box"}))
episode = Path(sys.argv[1])
(episode / "work").mkdir(parents=True)
box.start(episode)
print(box.name, flush=True)
time.sleep(600)
"""
    )
    owner = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path / "episode")], stdout=subprocess.PIPE, text=True
    )
    try:
        assert owner.stdout is not None
        name = owner.stdout.readline().strip()
        assert docker.containers() == [name]
        owner.send_signal(signal.SIGKILL)
        owner.wait(timeout=10)
        deadline = time.monotonic() + 30
        while docker.containers() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert docker.containers() == []
    finally:
        owner.kill()
        owner.wait()


# -- task directories ------------------------------------------------------------------------

# Answers, then tries to grade itself: a reward file and a tests directory of its own.
TASK_AGENT = """
root = os.environ["FAKE_ROOT"]
tests_at_start = os.path.exists(root + "/tests")
reply = chat(task["messages"])
open("answer.txt", "w").write(reply["content"])
os.makedirs(root + "/tests")
open(root + "/tests/planted.txt", "w").write("x")
open(root + "/logs/verifier/reward.txt", "w").write("1")
report(os.path.join(sys.argv[1], task["id"] + ".json"), {
    "task": task, "cwd": os.getcwd(), "env": dict(os.environ), "tests_at_start": tests_at_start,
    "files": sorted(os.listdir(".")),
})
"""


def _task_samples(tasks: Path) -> list[Sample]:
    from monorl.config.schema import DataConfig
    from monorl.data import open_source

    return list(open_source(DataConfig(uri=str(tasks), shuffle=False)))


def test_task_directories_bring_their_environment_verifier_and_limits(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOST_FLAVOUR", "spicy")
    tasks, reports = tmp_path / "tasks", tmp_path / "reports"
    reports.mkdir()
    write_task(
        tasks / "prebuilt",
        """
schema_version = "1.4"
[task]
name = "acme/prebuilt"
[metadata]
solution_explanation = "say it in upper case"
[agent]
user = "agent"
[verifier]
env = { EXPECTED = "ABC" }
[environment]
docker_image = "box"
workdir = "/app"
cpus = 2
memory_mb = 256
[environment.env]
FLAVOUR = "${HOST_FLAVOUR}"
PLAIN = "${NOT_SET_ANYWHERE:-plain}"
""",
        instruction="<!-- harbor-canary GUID 1234 -->\n\nabc\n",
    )
    write_task(
        tasks / "built",
        '[verifier]\nuser = "grader"\nenv = { EXPECTED = "something else" }\n',
        dockerfile="FROM box\nWORKDIR /srv\nCOPY seed.txt .\n",
    )
    samples = _task_samples(tasks)
    assert [(s.id, s.messages, s.reference) for s in samples] == [
        ("built", [{"role": "user", "content": "abc"}], None),
        ("prebuilt", [{"role": "user", "content": "abc"}], None),
    ]
    config = _config(make_config, ["{python}", str(write_agent(tmp_path, TASK_AGENT)), str(reports)])

    built, prebuilt, again = _episodes(config, [*samples, samples[0]])

    # The task's own verifier decides, not what the agent wrote to the reward file.
    assert (prebuilt.turns[-1].reward, built.turns[-1].reward, again.turns[-1].reward) == (1.0, 0.0, 0.0)
    assert prebuilt.info["verifier"] == {
        "reward": 1.0,
        "rewards": {"reward": 1.0},
        "exit_code": 0,
        "timed_out": False,
        "stdout_tail": "verifying as ''\n",
        "stderr_tail": "",
    }
    assert built.info["verifier"]["stdout_tail"] == "verifying as 'grader'\n"
    assert (prebuilt.info["task"], built.info["task"]) == ("acme/prebuilt", "built")
    # Each task ran in its own image, with its limits; the Dockerfile was built once for two episodes.
    assert prebuilt.info["sandbox"]["image"] == "box"
    assert (prebuilt.info["sandbox"]["cpus"], prebuilt.info["sandbox"]["memory"]) == (2.0, "256m")
    assert built.info["sandbox"]["image"] == again.info["sandbox"]["image"] != "box"
    assert len(docker.calls("build")) == 1
    seen = json.loads((reports / "prebuilt.json").read_text())
    assert seen["cwd"].endswith("/app") and seen["env"]["FAKE_DOCKER_USER"] == "agent"
    assert (seen["env"]["FLAVOUR"], seen["env"]["PLAIN"]) == ("spicy", "plain")
    seen_built = json.loads((reports / "built.json").read_text())
    assert seen_built["cwd"].endswith("/srv") and seen_built["files"] == ["answer.txt", "seed.txt"]
    # The agent is told the instruction and nothing else: not where the task lives, not its
    # author's notes; and no tests are in its container while it runs.
    for report in (seen, seen_built):
        assert report["task"]["messages"] == [{"role": "user", "content": "abc"}]
        assert report["task"]["metadata"] == {"row": {"id": report["task"]["id"]}}
        assert not report["tests_at_start"]
        assert str(tasks) not in json.dumps(report)
    assert not any("solution" in arg for call in docker.calls() for arg in call["argv"])
    assert docker.containers() == [] and get_gateway().num_sessions == 0


def test_task_limits_replace_the_configured_timeouts(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path
) -> None:
    slow_test = TEST_SH + "sleep 60\n"
    write_task(
        tmp_path / "tasks" / "slow",
        '[agent]\ntimeout_sec = 2\n[verifier]\ntimeout_sec = 1\n[environment]\ndocker_image = "box"\n',
        test_sh=slow_test,
    )
    pids = tmp_path / "pids.json"
    config = _config(
        make_config,
        ["{python}", str(write_agent(tmp_path, BUSY_AGENT)), str(pids)],
        timeout=600,
        verifier_timeout=600,
    )
    started = time.monotonic()
    (trajectory,) = _episodes(config, _task_samples(tmp_path / "tasks"))
    assert time.monotonic() - started < 60
    assert trajectory.info["timed_out"] and not trajectory.done
    verdict = trajectory.info["verifier"]
    assert (verdict["timed_out"], verdict["reward"]) == (True, 0.0) and "rewards" not in verdict
    assert wait_dead(json.loads(pids.read_text())["pid"]) and docker.containers() == []


def test_reward_json_and_broken_verifiers(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path
) -> None:
    environment = '[environment]\ndocker_image = "box"\n'
    json_reward = (
        '#!/bin/bash\necho \'{"correct": 1, "tidy": 0.5}\' > "$FAKE_ROOT/logs/verifier/reward.json"\n'
    )
    labelled = '#!/bin/bash\necho \'{"reward": 0.25, "tidy": 1}\' > "$FAKE_ROOT/logs/verifier/reward.json"\n'
    write_task(tmp_path / "tasks" / "a-metrics", environment, test_sh=json_reward)
    write_task(tmp_path / "tasks" / "b-labelled", environment, test_sh=labelled)
    write_task(tmp_path / "tasks" / "c-silent", environment, test_sh="#!/bin/bash\nexit 0\n")
    write_task(tmp_path / "tasks" / "d-garbage", environment, test_sh=json_reward.replace("0.5", '"high"'))
    agent = write_agent(tmp_path, 'chat(task["messages"])\n')
    metrics, labelled_reward, silent, garbage = _episodes(
        _config(make_config, ["{python}", str(agent)]), _task_samples(tmp_path / "tasks")
    )
    assert metrics.info["verifier"]["rewards"] == {"correct": 1.0, "tidy": 0.5}
    assert metrics.turns[-1].reward == 0.75  # no "reward" among the metrics: their mean
    assert labelled_reward.turns[-1].reward == 0.25
    assert (
        silent.turns[-1].reward == 0.0
        and "neither reward.json nor reward.txt" in silent.info["verifier"]["error"]
    )
    assert garbage.turns[-1].reward == 0.0 and "finite numbers" in garbage.info["verifier"]["error"]


def test_healthcheck_gates_the_agent(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path
) -> None:
    toml = """
[environment]
docker_image = "box"
workdir = "/app"
[environment.healthcheck]
command = '{command}'
interval_sec = 0.05
retries = {retries}
"""
    # fails once, then passes: the environment needed a moment
    warming_up = 'test -f "$FAKE_ROOT/app/warm" || { touch "$FAKE_ROOT/app/warm"; exit 1; }'
    write_task(tmp_path / "ok" / "warm", toml.format(command=warming_up, retries=3))
    write_task(tmp_path / "bad" / "cold", toml.format(command="echo still cold >&2; exit 1", retries=2))
    agent, report = write_agent(tmp_path, AGENT), tmp_path / "report.json"
    config = _config(make_config, ["{python}", str(agent), str(report)], verifier=None)

    _episodes(config, _task_samples(tmp_path / "ok"))
    assert "warm" in json.loads(report.read_text())["files"]

    with pytest.raises(EnvironmentError_, match="did not become healthy.*2 times in a row.*still cold"):
        _episodes(config, _task_samples(tmp_path / "bad"))
    assert docker.containers() == [] and get_gateway().num_sessions == 0


def test_tasks_need_a_container(make_config: Callable[..., RunConfig], tmp_path: Path) -> None:
    write_task(tmp_path / "tasks" / "t", '[environment]\ndocker_image = "box"\n')
    env = {"type": "harness", "params": {"command": [sys.executable, "-c", "pass"]}}
    config = make_config(rollout={"extra": {"fake_mode": "always_right"}}, env=env)
    with pytest.raises(ConfigError, match="sandbox.backend to 'docker'"):
        _episodes(config, _task_samples(tmp_path / "tasks"))
    assert get_gateway().num_sessions == 0


def test_example_shell_agent_solves_a_task(
    make_config: Callable[..., RunConfig], docker: FakeDocker, tmp_path: Path
) -> None:
    agent = Path(__file__).parents[2] / "examples" / "sandbox" / "agents" / "shell_agent.py"
    test_sh = """#!/bin/bash
if [ "$(cat "${FAKE_ROOT:-}/app/hello.txt")" = "Hello, world!" ]; then echo 1; else echo 0; fi > "${FAKE_ROOT:-}/logs/verifier/reward.txt"
"""
    write_task(
        tmp_path / "tasks" / "hello",
        '[environment]\ndocker_image = "box"\nworkdir = "/app"\n',
        instruction="Create the file hello.txt. It must contain exactly one line: Hello, world!",
        test_sh=test_sh,
    )
    script = [
        '<tool_call>{"name": "shell", "arguments": {"command": "echo \'Hello, world!\' > hello.txt && ls"}}</tool_call>',
        "DONE",
    ]
    config = _config(
        make_config,
        ["{python}", str(agent), "--max-steps", "3"],
        engine={"fake_mode": "script", "fake_script": script},
    )
    (trajectory,) = _episodes(config, _task_samples(tmp_path / "tasks"))
    first, second = trajectory.turns
    (call,) = first.info["tool_calls"]
    assert (call["name"], call["result"]) == ("shell", "exit status 0\nhello.txt")
    assert second.completion.text == "DONE" and second.reward == 1.0
    assert trajectory.info["stdout_tail"] == "DONE\n" and trajectory.done
