"""``env: harness`` — an external agent program is the environment.

Every episode starts the configured command as its own process. The agent reads its task from a
file and calls the policy through the capture gateway (:mod:`monorl.gateway`), an
OpenAI-compatible endpoint served from the rollout engine of the run; the gateway records each
call with its exact tokens, and when the agent exits the calls are the episode's turns.
:class:`HarnessEnv` is an :class:`~monorl.core.interfaces.EpisodeDriver`: the episode runner
hands it the whole episode instead of stepping it.

``env.params``:

* ``command`` (required) — the agent, as an argv list or a string (split like a shell would, run
  without one). Placeholders, also usable in ``env`` values and ``verifier``: ``{task_file}``,
  ``{workdir}``, ``{base_url}``, ``{api_key}``, ``{model}``, ``{python}`` (this interpreter; in a
  container ``sandbox.python``), ``{session}``, ``{reward_url}``, ``{sample_id}``, and
  ``{anthropic_base_url}`` when the gateway serves the Anthropic API. Other braces are left alone.
* ``timeout`` — seconds the agent may run (default 600). Then it is killed; the episode still
  counts when the agent called the policy at least once.
* ``env`` — extra environment variables for the agent and the verifier.
* ``verifier`` — optional command run in the working directory after the agent ended; its
  result is the episode's reward: the last line of its stdout when that is a number, otherwise
  1.0 for exit status 0 and 0.0 for any other. Additional placeholders: ``{reference_file}``
  (the full sample including the reference, written only now), ``{stdout_file}``,
  ``{stderr_file}`` (the agent's captured output) and ``{exit_code}``.
  ``verifier_timeout`` — seconds (default 60); a verifier that runs longer scores 0.
* ``workdir`` — where the fresh per-episode working directory comes from: ``root`` (directory
  the episode directories are created under; default: the system temp dir) and ``template`` (a
  directory copied into every working directory).
* ``model`` — the model name the agent is told (default: ``model.name_or_path``).
* ``sandbox`` — where the agent and the verifier run (:mod:`monorl.envs.sandbox`,
  ``docs/SANDBOX.md``): ``{backend: local}`` (default; processes on this host) or
  ``{backend: docker, image: ...}`` (a container per episode). In a container every path a
  placeholder or variable names is the container's.
* ``gateway`` — ``bind`` (address the capture gateway listens on), ``port`` (default: any free
  one) and ``host`` (the name the agent reaches this machine under). The defaults follow the
  sandbox: ``127.0.0.1`` for local processes, and for containers what the platform needs
  (:func:`monorl.envs.sandbox.gateway_address`).
* ``fetch_images`` — whether the gateway downloads images an agent sends as ``http(s)`` URLs.
  Default: ``true`` for local processes, ``false`` in a container (a sandboxed agent must not be
  able to make the trainer host issue requests to addresses of its choosing; such images are
  refused, ``data:`` URLs still work). An explicit value wins.

``env.max_turns``, when set, caps the policy calls of an episode; further calls are refused with
an OpenAI-style error.

The agent's process gets ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` (so an unmodified OpenAI SDK
client reaches the policy), ``ANTHROPIC_BASE_URL`` and ``ANTHROPIC_API_KEY`` when the gateway
serves the Anthropic API, and ``MONORL_BASE_URL``, ``MONORL_SESSION``, ``MONORL_TASK_FILE``,
``MONORL_WORKDIR``, ``MONORL_MODEL``, ``MONORL_REWARD_URL``. The task file holds the sample's
``id``, ``messages`` and ``metadata`` — never the reference, nor the copies of the answer the data
layer keeps in the metadata.

Tasks. A sample may bring its own environment, verifier and limits: a mapping under
``metadata["row"]["task"]``, which a task-directory dataset (:mod:`monorl.data.tasks`, the Harbor
task format) fills in. They replace the ``env.params`` values for that sample::

    {"name": str,
     "sandbox": {...},          # env.params.sandbox options laid over the configured ones
     "env": {name: value},      # the container's environment; "${VAR}" / "${VAR:-default}" are read from this host
     "timeout": seconds,        # the agent's; absent: env.params.timeout
     "user": str,               # the agent's user
     "healthcheck": {command, interval_sec, timeout_sec, start_period_sec, start_interval_sec, retries},
     "verifier": {"tests": directory, "timeout": seconds, "user": str, "env": {name: value}}}

A task verifier is the Harbor one: after the agent has ended, ``tests`` is copied to ``/tests``
and ``/tests/test.sh`` runs in the working directory; the reward is what it wrote to
``/logs/verifier/reward.json`` or ``reward.txt``. The agent is never told the task mapping, and
nothing of ``tests`` is in its container before it has ended. Tasks need ``sandbox.backend:
docker``.

Rewards: the verifier's result and rewards the agent posted to ``MONORL_REWARD_URL`` are added
to the last turn's ``reward``; reward functions score the trajectory as for any environment.
``Trajectory.info`` carries ``exit_code``, ``timed_out``, ``stdout_tail``, ``stderr_tail``,
``posted_reward`` / ``verifier`` when there were any, ``sandbox`` for a container (image,
container name, limits, CPU seconds, peak memory, ``oom_killed``) and ``task`` (its name).

Torch-free; the gateway's HTTP stack is imported with the first episode.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monorl.core.errors import ConfigError, EnvironmentError_, EpisodeAborted, GenerationFailed
from monorl.core.types import Sample, SamplingParams, Trajectory
from monorl.envs.protocol import encode_sample
from monorl.envs.sandbox import (
    POLL_S,
    DockerSandbox,
    LocalSandbox,
    SandboxConfig,
    docker_cli,
    kill_group,
    make_sandbox,
)

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.interfaces import RolloutEngine
    from monorl.gateway import Gateway, Session

DEFAULT_TIMEOUT_S = 600.0
DEFAULT_VERIFIER_TIMEOUT_S = 60.0
TASK_VERIFIER_TIMEOUT_S = 600.0
"""Seconds a task's ``tests/test.sh`` may run when the task does not say (Harbor's default)."""
TAIL_CHARS = 2000
"""How much of the end of the agent's (and verifier's) output goes to ``Trajectory.info``."""
TASK_TESTS_DIR = "/tests"
TASK_LOGS_DIR = "/logs"
"""Where a task's tests are copied to and where its verifier leaves the reward (Harbor's paths)."""

_PARAMS = (
    "command",
    "timeout",
    "env",
    "verifier",
    "verifier_timeout",
    "workdir",
    "model",
    "sandbox",
    "gateway",
    "fetch_images",
)
_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")
_HOST_VARIABLE_RE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
_ANSWER_METADATA = ("answer_raw", "response", "assistant_turn")
"""Sample metadata in which the data layer keeps the answer column."""


def task_spec(sample: Sample) -> Mapping[str, Any] | None:
    """The task mapping ``sample`` brings (module docstring), if it brings one."""
    row = sample.metadata.get("row")
    task = row.get("task") if isinstance(row, Mapping) else None
    if isinstance(task, Mapping) and ("sandbox" in task or "verifier" in task):
        return task
    return None


def task_payload(sample: Sample) -> dict[str, Any]:
    """What the agent is told about ``sample``: id, messages and metadata. The reference is left
    out, and so are the data layer's copies of the answer — its metadata entries and any column
    of ``metadata["row"]`` that holds the same value — and the sample's task mapping (where its
    tests live, what its author noted)."""
    hidden = [sample.reference, *(sample.metadata.get(key) for key in _ANSWER_METADATA)]
    hidden = [value for value in hidden if value is not None]
    metadata = {key: value for key, value in sample.metadata.items() if key not in _ANSWER_METADATA}
    row = metadata.get("row")
    if isinstance(row, Mapping):
        task = task_spec(sample)
        metadata["row"] = {
            key: value
            for key, value in row.items()
            if not (key == "task" and task is not None)
            and not any(value == answer or str(value) == str(answer) for answer in hidden)
        }
    return {"id": sample.id, "messages": list(sample.messages), "metadata": metadata}


class HarnessEnv:
    """One agent process per episode, served by the capture gateway (module docstring)."""

    def __init__(
        self,
        *,
        command: str | list[str],
        timeout: float = DEFAULT_TIMEOUT_S,
        env: Mapping[str, Any] | None = None,
        verifier: str | list[str] | None = None,
        verifier_timeout: float = DEFAULT_VERIFIER_TIMEOUT_S,
        workdir: Mapping[str, Any] | None = None,
        max_turns: int | None = None,
        model: str = "monorl",
        sandbox: Mapping[str, Any] | None = None,
        gateway: Mapping[str, Any] | None = None,
        fetch_images: bool | None = None,
        processor: Callable[[], Any] | None = None,
    ) -> None:
        """``processor`` loads the policy's ``AutoProcessor`` (``None`` for a text-only policy); it
        is called for engines that do not bring theirs. The other arguments are ``env.params``."""
        self.command = _argv(command, "command")
        self.verifier = None if verifier is None else _argv(verifier, "verifier")
        self.timeout = _seconds(timeout, "timeout")
        self.verifier_timeout = _seconds(verifier_timeout, "verifier_timeout")
        if not isinstance(env or {}, Mapping) or not isinstance(workdir or {}, Mapping):
            raise ConfigError("env.params.env and env.params.workdir must be mappings")
        self.env = {str(key): str(value) for key, value in (env or {}).items()}
        workdir = dict(workdir or {})
        if set(workdir) - {"root", "template"}:
            raise ConfigError(f"env.params.workdir accepts 'root' and 'template', got {sorted(workdir)}")
        self.workdir_root = _directory(workdir.get("root"))
        self.workdir_template = _directory(workdir.get("template"))
        if self.workdir_template is not None and not self.workdir_template.is_dir():
            raise ConfigError(
                f"env.params.workdir.template {str(self.workdir_template)!r} is not a directory"
            )
        self.max_turns = None if max_turns is None else int(max_turns)
        self.model = str(model)
        self.sandbox = SandboxConfig.from_params(sandbox)
        if self.sandbox.backend == "docker":
            docker_cli()  # no CLI: fail when the environment is built, not in the first episode
        if not isinstance(gateway or {}, Mapping) or set(gateway or {}) - {"bind", "port", "host"}:
            raise ConfigError("env.params.gateway accepts 'bind', 'port' and 'host'")
        gateway = {key: value for key, value in (gateway or {}).items() if value is not None}
        self.gateway_bind = str(gateway["bind"]) if "bind" in gateway else None
        self.gateway_host = str(gateway["host"]) if "host" in gateway else None
        try:
            self.gateway_port = int(gateway.get("port", 0))
        except (TypeError, ValueError) as e:
            raise ConfigError(
                f"env.params.gateway.port must be a port number, got {gateway['port']!r}"
            ) from e
        if fetch_images is not None and not isinstance(fetch_images, bool):
            raise ConfigError(f"env.params.fetch_images must be true or false, got {fetch_images!r}")
        self.fetch_images = fetch_images
        self._load_processor = processor
        self._gateway: Gateway | None = None
        self._session: Session | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._sandbox: LocalSandbox | DockerSandbox | None = None
        self._dir: Path | None = None

    # -- EpisodeDriver ----------------------------------------------------------------------
    def run_episode(
        self,
        sample: Sample,
        *,
        group_id: str,
        engine: RolloutEngine,
        tokenizer: Any,
        params: SamplingParams,
        abort: threading.Event | None = None,
        turn_rendering: str = "tokens",
        max_context_tokens: int | None = None,
    ) -> Trajectory:
        from monorl.gateway import build_trajectory, get_gateway

        task = task_spec(sample)
        config = self.sandbox if task is None else self.sandbox.override(task.get("sandbox") or {})
        if task is not None and config.backend != "docker":
            raise ConfigError(
                f"sample {sample.id!r} is a task with an environment of its own: it runs in a "
                "container, set env.params.sandbox.backend to 'docker'"
            )
        task_verifier = task.get("verifier") if task is not None else None
        if not isinstance(task_verifier, Mapping) or not task_verifier.get("tests"):
            task_verifier = None
        agent_user = _optional(task.get("user")) if task is not None else None
        timeout = self.timeout
        if task is not None and task.get("timeout") is not None:
            timeout = _seconds(task["timeout"], "timeout")

        self._sandbox = sandbox = make_sandbox(config)
        bind, host = sandbox.gateway_address()
        host = self.gateway_host or host
        self._gateway = gateway = get_gateway(host=self.gateway_bind or bind, port=self.gateway_port)
        options: dict[str, Any] = {}
        if self._load_processor is not None and not _brings_processor(engine):
            processor = self._load_processor()  # vLLM / SGLang engines keep theirs to themselves
            if processor is not None:
                options["processor"] = processor
        self._session = session = gateway.open_session(
            engine=engine,
            tokenizer=tokenizer,
            params=params,
            turn_rendering=turn_rendering,
            max_context_tokens=max_context_tokens,
            model=self.model,
            max_calls=self.max_turns,
            # an agent in a sandbox does not get to make this host fetch URLs
            fetch_images=config.backend == "local" if self.fetch_images is None else self.fetch_images,
            **options,
        )
        users = (config.user, agent_user, task_verifier.get("user") if task_verifier else None)
        # a container user other than the image's default has to get into the episode directory
        shared = isinstance(sandbox, DockerSandbox) and any(user is not None for user in users)
        episode_dir = self._make_dir(shared=shared)
        mounted_work = config.workdir is None  # the working directory is the episode's, on this host
        if mounted_work:
            self._make_workdir(episode_dir / "work", shared=shared)
        logs = None
        if task_verifier is not None:
            logs = episode_dir / "logs"
            for name in ("agent", "verifier", "artifacts"):
                (logs / name).mkdir(parents=True)
                (logs / name).chmod(0o777)  # whoever the task runs as writes here
        task_file = episode_dir / "task.json"
        task_file.write_text(
            json.dumps(task_payload(sample), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        sandbox.start(
            episode_dir, logs=logs, env=_host_env(task.get("env") if task is not None else None), abort=abort
        )
        if isinstance(sandbox, DockerSandbox) and not mounted_work and self.workdir_template is not None:
            sandbox.copy_in(self.workdir_template, sandbox.workdir)
        stdout, stderr = episode_dir / "agent.stdout", episode_dir / "agent.stderr"
        values = {
            "task_file": sandbox.path(task_file),
            "workdir": sandbox.workdir,
            "base_url": gateway.base_url(session, host),
            "api_key": session.id,
            "session": session.id,
            "model": self.model,
            "python": sandbox.python,
            "reward_url": gateway.reward_url(session, host),
            "sample_id": sample.id,
        }
        anthropic_base_url = getattr(gateway, "anthropic_base_url", None)
        if anthropic_base_url is not None:  # the gateway serves the Anthropic API as well
            values["anthropic_base_url"] = anthropic_base_url(session, host)
        env = self._environ(values, sandbox.environ(host))
        if task is not None and task.get("healthcheck"):
            self._healthcheck(task["healthcheck"], group_id=group_id, abort=abort)

        exit_code, timed_out = self._run(
            self.command,
            values,
            env=env,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            what="agent",
            group_id=group_id,
            abort=abort,
            session=session,
            user=agent_user,
        )
        gateway.close_session(session)  # what is recorded now is the episode
        verifying = bool(session.calls) and (task_verifier is not None or self.verifier is not None)
        # Nothing of the agent runs while the verifier's inputs (tests, reference) are around.
        sandbox.settle(restart=verifying and (timed_out or not config.keep_processes))
        sandbox_info = sandbox.info()
        if not session.calls:
            ended = f"timed out after {timeout:g}s" if timed_out else f"exit code {exit_code}"
            if sandbox_info.get("oom_killed"):
                ended += ", out of memory"
            raise EnvironmentError_(
                f"episode {group_id}: the harness agent made no policy call ({ended}); "
                f"stderr: {_tail(stderr) or '(empty)'}"
            )
        trajectory = build_trajectory(session.calls, sample, group_id)
        trajectory.done = not timed_out and not session.refused
        if session.context_exceeded:
            trajectory.info["context_exceeded_calls"] = session.context_exceeded
        trajectory.info.update(
            exit_code=exit_code, timed_out=timed_out, stdout_tail=_tail(stdout), stderr_tail=_tail(stderr)
        )
        if sandbox_info:
            trajectory.info["sandbox"] = sandbox_info
        if task is not None and task.get("name") is not None:
            trajectory.info["task"] = str(task["name"])
        reward = 0.0
        if session.reward is not None:
            trajectory.info["posted_reward"] = session.reward
            reward += session.reward
        verdict = None
        if task_verifier is not None:
            assert isinstance(sandbox, DockerSandbox) and logs is not None
            verdict = self._verify_task(task_verifier, sandbox, logs, group_id=group_id, abort=abort)
        elif self.verifier is not None:
            reference_file = episode_dir / "reference.json"
            reference_file.write_text(
                json.dumps(encode_sample(sample), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            values.update(
                reference_file=sandbox.path(reference_file),
                stdout_file=sandbox.path(stdout),
                stderr_file=sandbox.path(stderr),
                exit_code=str(exit_code),
            )
            verdict = self._verify(values, env, group_id=group_id, abort=abort)
        if verdict is not None:
            trajectory.info["verifier"] = verdict
            reward += verdict["reward"]
        trajectory.turns[-1].reward += reward
        return trajectory

    def close(self) -> None:
        """Close the session, kill and reap whatever still runs, remove the container (if the
        episode had one) and the episode directory."""
        session, self._session = self._session, None
        if session is not None and self._gateway is not None:
            self._gateway.close_session(session)
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            kill_group(proc)
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:
            sandbox.close()
        episode_dir, self._dir = self._dir, None
        if episode_dir is not None:
            shutil.rmtree(episode_dir, ignore_errors=True)

    # -- internals --------------------------------------------------------------------------
    def _make_dir(self, *, shared: bool) -> Path:
        """A fresh directory for the episode: ``task.json``, captured output and ``work/``.
        ``shared``: users other than the owner get in."""
        if self.workdir_root is not None:
            self.workdir_root.mkdir(parents=True, exist_ok=True)
        self._dir = Path(tempfile.mkdtemp(prefix="monorl-harness-", dir=self.workdir_root)).resolve()
        if shared:
            self._dir.chmod(0o755)
        return self._dir

    def _make_workdir(self, workdir: Path, *, shared: bool) -> None:
        if self.workdir_template is not None:
            shutil.copytree(self.workdir_template, workdir, symlinks=True)
        else:
            workdir.mkdir()
        if shared:
            workdir.chmod(0o777)

    def _environ(self, values: Mapping[str, str], base: dict[str, str]) -> dict[str, str]:
        """The agent's environment: ``base`` (the sandbox's), the gateway and task variables,
        ``env.params.env``."""
        env = dict(base)
        env.update(
            OPENAI_BASE_URL=values["base_url"],
            OPENAI_API_KEY=values["api_key"],
            MONORL_BASE_URL=values["base_url"],
            MONORL_SESSION=values["session"],
            MONORL_TASK_FILE=values["task_file"],
            MONORL_WORKDIR=values["workdir"],
            MONORL_MODEL=values["model"],
            MONORL_REWARD_URL=values["reward_url"],
        )
        if "anthropic_base_url" in values:
            env.update(ANTHROPIC_BASE_URL=values["anthropic_base_url"], ANTHROPIC_API_KEY=values["api_key"])
        env.update({key: _fill(value, values) for key, value in self.env.items()})
        return env

    def _healthcheck(self, check: Mapping[str, Any], *, group_id: str, abort: threading.Event | None) -> None:
        """Wait until a task's environment is ready, with Docker's ``HEALTHCHECK`` semantics (as
        Harbor runs it): ``command`` is repeated until it exits 0; failures during the start
        period do not count, ``retries`` consecutive ones after it fail the episode."""
        assert self._dir is not None
        command = ["bash", "-c", str(check["command"])]
        retries = int(check.get("retries", 3))
        start_period_ends = time.monotonic() + float(check.get("start_period_sec", 0.0))
        failures = 0
        while True:
            starting = time.monotonic() < start_period_ends
            exit_code, timed_out = self._run(
                command,
                {},
                env={},
                stdout=self._dir / "healthcheck.stdout",
                stderr=self._dir / "healthcheck.stderr",
                timeout=_seconds(check.get("timeout_sec", 30.0), "healthcheck timeout"),
                what="healthcheck",
                group_id=group_id,
                abort=abort,
            )
            if exit_code == 0 and not timed_out:
                return
            if not starting:
                failures += 1
                if failures >= retries:
                    raise EnvironmentError_(
                        f"episode {group_id}: the task environment did not become healthy "
                        f"({check['command']!r} failed {failures} times in a row); "
                        f"stderr: {_tail(self._dir / 'healthcheck.stderr') or '(empty)'}"
                    )
            pause = float(check.get("start_interval_sec" if starting else "interval_sec", 5.0))
            if abort is None:
                time.sleep(pause)
            elif abort.wait(pause):
                raise EpisodeAborted(f"episode {group_id} aborted while its environment was starting")

    def _verify(
        self,
        values: Mapping[str, str],
        env: Mapping[str, str],
        *,
        group_id: str,
        abort: threading.Event | None,
    ) -> dict[str, Any]:
        """Run the verifier; ``{reward, exit_code, timed_out, stdout_tail, stderr_tail}``."""
        assert self.verifier is not None and self._dir is not None
        stdout, stderr = self._dir / "verifier.stdout", self._dir / "verifier.stderr"
        verifier_env = {
            **env,
            "MONORL_REFERENCE_FILE": values["reference_file"],
            "MONORL_AGENT_STDOUT": values["stdout_file"],
            "MONORL_AGENT_STDERR": values["stderr_file"],
            "MONORL_AGENT_EXIT_CODE": values["exit_code"],
        }
        exit_code, timed_out = self._run(
            self.verifier,
            values,
            env=verifier_env,
            stdout=stdout,
            stderr=stderr,
            timeout=self.verifier_timeout,
            what="verifier",
            group_id=group_id,
            abort=abort,
        )
        output = stdout.read_text(encoding="utf-8", errors="replace")
        return {
            "reward": 0.0 if timed_out else _verifier_reward(output, exit_code),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        }

    def _verify_task(
        self,
        verifier: Mapping[str, Any],
        sandbox: DockerSandbox,
        logs: Path,
        *,
        group_id: str,
        abort: threading.Event | None,
    ) -> dict[str, Any]:
        """Run a task's verifier the way Harbor does: whatever the agent left under ``/tests``
        and any reward file it wrote are removed, the task's ``tests`` directory is copied to
        ``/tests``, ``/tests/test.sh`` runs in the working directory and the reward is read from
        ``/logs/verifier``. ``{reward, exit_code, timed_out, stdout_tail, stderr_tail}`` plus
        ``rewards`` (every metric of ``reward.json``) and ``error`` (no usable reward file)."""
        assert self._dir is not None
        script = f"{TASK_TESTS_DIR}/test.sh"
        reward_files = [f"{TASK_LOGS_DIR}/verifier/reward.json", f"{TASK_LOGS_DIR}/verifier/reward.txt"]
        sandbox.root(["rm", "-rf", TASK_TESTS_DIR, *reward_files], "clear the verifier's paths")
        sandbox.copy_in(Path(str(verifier["tests"])), TASK_TESTS_DIR)
        sandbox.root(["chmod", "+x", script], "make the test script executable")
        stdout, stderr = self._dir / "verifier.stdout", self._dir / "verifier.stderr"
        timeout = verifier.get("timeout")
        exit_code, timed_out = self._run(
            [script],
            {},
            env=_host_env(verifier.get("env")),
            stdout=stdout,
            stderr=stderr,
            timeout=TASK_VERIFIER_TIMEOUT_S if timeout is None else _seconds(timeout, "verifier timeout"),
            what="verifier",
            group_id=group_id,
            abort=abort,
            user=_optional(verifier.get("user")),
        )
        verdict: dict[str, Any] = {
            "reward": 0.0,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        }
        if not timed_out:
            try:
                rewards = _task_rewards(logs / "verifier")
            except (OSError, ValueError) as e:
                verdict["error"] = str(e)
            else:
                verdict["rewards"] = rewards
                values = list(rewards.values())
                verdict["reward"] = float(
                    rewards["reward"] if "reward" in rewards else sum(values) / len(values)
                )
        return verdict

    def _run(
        self,
        command: list[str],
        values: Mapping[str, str],
        *,
        env: Mapping[str, str],
        stdout: Path,
        stderr: Path,
        timeout: float,
        what: str,
        group_id: str,
        abort: threading.Event | None,
        session: Session | None = None,
        user: str | None = None,
    ) -> tuple[int, bool]:
        """Run ``command`` in the sandbox's working directory until it exits or ``timeout``
        passes; ``(exit code, timed out)``. Its process group on this host is always killed and
        reaped before this returns (in a container that is the ``docker exec`` client; the
        container's processes end with ``settle`` or ``close``). Raises ``EpisodeAborted`` once
        ``abort`` is set or a generation of ``session`` was cancelled, and the engine's error when
        a generation of ``session`` failed."""
        assert self._sandbox is not None
        argv = [_fill(arg, values) for arg in command]
        try:
            with open(stdout, "wb") as out, open(stderr, "wb") as err:
                proc = self._sandbox.popen(argv, env=env, stdout=out, stderr=err, user=user)
        except OSError as e:
            raise EnvironmentError_(
                f"episode {group_id}: cannot start the harness {what} {argv!r}: {e}"
            ) from e
        self._proc = proc
        deadline = time.monotonic() + timeout
        timed_out = False

        def stopped() -> bool:
            return (abort is not None and abort.is_set()) or (
                session is not None and (session.cancelled.is_set() or session.failure is not None)
            )

        try:
            while proc.poll() is None and not stopped():
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                try:
                    proc.wait(timeout=POLL_S)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            kill_group(proc)  # the process, and whatever it left running in its group
            self._proc = None
        if session is not None and session.failure is not None:
            raise GenerationFailed(
                f"episode {group_id}: the rollout engine failed: {session.failure}"
            ) from session.failure
        if stopped():
            raise EpisodeAborted(f"episode {group_id} aborted while its harness {what} was running")
        return proc.returncode, timed_out


def make_harness_env(
    params: Mapping[str, Any],
    *,
    max_turns: int | None = None,
    model: str = "monorl",
    processor: Callable[[], Any] | None = None,
) -> HarnessEnv:
    """:class:`HarnessEnv` from ``env.params`` (module docstring); ``null`` params are unset.
    ``processor`` loads the policy's ``AutoProcessor`` (:func:`policy_processor`)."""
    given = {key: value for key, value in params.items() if value is not None}
    unknown = sorted(set(given) - set(_PARAMS))
    if unknown:
        raise ConfigError(f"unknown env.params for env.type 'harness': {unknown}; expected {list(_PARAMS)}")
    if not given.get("command"):
        raise ConfigError(
            "env.type 'harness' needs env.params.command (the agent's argv list or command string)"
        )
    return HarnessEnv(**{"model": model, **given}, max_turns=max_turns, processor=processor)


_processors: dict[tuple[Any, ...], Any] = {}
_processors_lock = threading.Lock()


def policy_processor(config: RunConfig) -> Any | None:
    """The ``AutoProcessor`` of the run's policy when it is a vision-language model, ``None`` for a
    text-only one; loaded once per process (:func:`monorl.rollout.media.load_processor`)."""
    model = config.model
    key = (model.name_or_path, model.tokenizer_name_or_path, model.revision, model.chat_template)
    with _processors_lock:
        if key not in _processors:
            from monorl.rollout.media import load_processor

            _processors[key] = load_processor(config)
        return _processors[key]


def _brings_processor(engine: Any) -> bool:
    """Whether ``engine``, or an engine it wraps, has a ``processor`` attribute: the gateway takes
    the processor (or its absence, for a text-only model) from there."""
    for _ in range(4):
        if engine is None:
            return False
        if hasattr(engine, "processor"):
            return True
        engine = getattr(engine, "engine", None)
    return False


def _argv(command: Any, name: str) -> list[str]:
    """``command`` as an argv list; a string is split like a shell would split it."""
    argv = shlex.split(command) if isinstance(command, str) else command
    if not isinstance(argv, (list, tuple)) or not argv:
        raise ConfigError(f"env.params.{name} must be a non-empty command string or argv list")
    return [str(arg) for arg in argv]


def _seconds(value: Any, name: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = math.nan
    if not 0 < seconds < math.inf:
        raise ConfigError(f"env.params.{name} must be a positive number of seconds, got {value!r}")
    return seconds


def _directory(path: Any) -> Path | None:
    return None if path is None else Path(str(path)).expanduser()


def _optional(value: Any) -> str | None:
    return None if value is None else str(value)


def _fill(text: str, values: Mapping[str, str]) -> str:
    """``text`` with the ``{placeholder}``s named in ``values`` replaced; other braces stay."""
    return _PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def _host_env(env: Mapping[str, Any] | None) -> dict[str, str]:
    """A task's environment variables with Harbor's templates resolved: a value that is exactly
    ``${VAR}`` or ``${VAR:-default}`` is read from this process's environment."""
    resolved: dict[str, str] = {}
    for key, value in (env or {}).items():
        template = _HOST_VARIABLE_RE.fullmatch(str(value))
        if template is None:
            resolved[str(key)] = str(value)
        elif template.group(1) in os.environ:
            resolved[str(key)] = os.environ[template.group(1)]
        elif template.group(2) is not None:
            resolved[str(key)] = template.group(2)
        else:
            raise ConfigError(
                f"the task's environment variable {key} is read from {template.group(1)}, "
                "which is not set in monorl's environment"
            )
    return resolved


def _verifier_reward(output: str, exit_code: int) -> float:
    """The last line of the verifier's stdout when it is a finite number, otherwise 1.0 for exit
    status 0 and 0.0 for any other."""
    lines = output.strip().splitlines()
    if lines:
        try:
            printed = float(lines[-1])
        except ValueError:
            pass
        else:
            if math.isfinite(printed):
                return printed
    return 1.0 if exit_code == 0 else 0.0


def _task_rewards(directory: Path) -> dict[str, float]:
    """The metrics a Harbor verifier left in ``directory``: ``reward.json`` (an object of
    numbers, or one number) if it is there, else the number in ``reward.txt`` as ``reward``.
    ``ValueError`` when there is no reward file or it does not hold finite numbers."""
    json_file, text_file = directory / "reward.json", directory / "reward.txt"
    if json_file.exists():
        rewards = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(rewards, dict):
            rewards = {"reward": rewards}
        name = json_file.name
    elif text_file.exists():
        rewards = {"reward": float(text_file.read_text(encoding="utf-8"))}
        name = text_file.name
    else:
        raise ValueError("the verifier wrote neither reward.json nor reward.txt to /logs/verifier")
    numbers = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in rewards.values()
    )
    if not rewards or not numbers:
        raise ValueError(f"{name} does not hold finite numbers: {rewards!r}")
    return {str(key): float(value) for key, value in rewards.items()}


def _tail(path: Path) -> str:
    """The last :data:`TAIL_CHARS` characters of a captured output file."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 4 * TAIL_CHARS))
        return f.read().decode("utf-8", errors="replace")[-TAIL_CHARS:]


__all__ = ["HarnessEnv", "make_harness_env", "policy_processor", "task_payload", "task_spec"]
