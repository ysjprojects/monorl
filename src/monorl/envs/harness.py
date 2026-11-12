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
  ``{workdir}``, ``{base_url}``, ``{api_key}``, ``{model}``, ``{python}`` (this interpreter),
  ``{session}``, ``{reward_url}``, ``{sample_id}``. Other braces are left alone.
* ``timeout`` — seconds the agent may run (default 600). Then its process group is killed; the
  episode still counts when the agent called the policy at least once.
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

``env.max_turns``, when set, caps the policy calls of an episode; further calls are refused with
an OpenAI-style error.

The agent's process gets ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` (so an unmodified OpenAI SDK
client reaches the policy) and ``MONORL_BASE_URL``, ``MONORL_SESSION``, ``MONORL_TASK_FILE``,
``MONORL_WORKDIR``, ``MONORL_MODEL``, ``MONORL_REWARD_URL``. The task file holds the sample's
``id``, ``messages`` and ``metadata`` — never the reference, nor the copies of the answer the data
layer keeps in the metadata.

Rewards: the verifier's result and rewards the agent posted to ``MONORL_REWARD_URL`` are added
to the last turn's ``reward``; reward functions score the trajectory as for any environment.
``Trajectory.info`` carries ``exit_code``, ``timed_out``, ``stdout_tail``, ``stderr_tail``, and
``posted_reward`` / ``verifier`` when there were any.

Torch-free; the gateway's HTTP stack is imported with the first episode.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monorl.core.errors import ConfigError, EnvironmentError_
from monorl.core.types import Sample, SamplingParams, Trajectory
from monorl.envs.protocol import encode_sample

if TYPE_CHECKING:
    from monorl.core.interfaces import RolloutEngine
    from monorl.gateway import Gateway, Session

DEFAULT_TIMEOUT_S = 600.0
DEFAULT_VERIFIER_TIMEOUT_S = 60.0
TAIL_CHARS = 2000
"""How much of the end of the agent's (and verifier's) output goes to ``Trajectory.info``."""
POLL_S = 0.05
"""How often a running process is checked against its deadline and the abort signals."""

_PARAMS = ("command", "timeout", "env", "verifier", "verifier_timeout", "workdir", "model")
_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")
_ANSWER_METADATA = ("answer_raw", "response", "assistant_turn")
"""Sample metadata in which the data layer keeps the answer column."""


def task_payload(sample: Sample) -> dict[str, Any]:
    """What the agent is told about ``sample``: id, messages and metadata. The reference is left
    out, and so are the data layer's copies of the answer — its metadata entries and any column
    of ``metadata["row"]`` that holds the same value."""
    hidden = [sample.reference, *(sample.metadata.get(key) for key in _ANSWER_METADATA)]
    hidden = [value for value in hidden if value is not None]
    metadata = {key: value for key, value in sample.metadata.items() if key not in _ANSWER_METADATA}
    row = metadata.get("row")
    if isinstance(row, Mapping):
        metadata["row"] = {
            key: value
            for key, value in row.items()
            if not any(value == answer or str(value) == str(answer) for answer in hidden)
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
    ) -> None:
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
        self._gateway: Gateway | None = None
        self._session: Session | None = None
        self._proc: subprocess.Popen[bytes] | None = None
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
    ) -> Trajectory:
        from monorl.gateway import build_trajectory, get_gateway

        self._gateway = gateway = get_gateway()
        self._session = session = gateway.open_session(
            engine=engine,
            tokenizer=tokenizer,
            params=params,
            turn_rendering=turn_rendering,
            model=self.model,
            max_calls=self.max_turns,
        )
        episode_dir = self._make_dir()
        workdir = episode_dir / "work"
        if self.workdir_template is not None:
            shutil.copytree(self.workdir_template, workdir, symlinks=True)
        else:
            workdir.mkdir()
        task_file = episode_dir / "task.json"
        task_file.write_text(
            json.dumps(task_payload(sample), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        stdout, stderr = episode_dir / "agent.stdout", episode_dir / "agent.stderr"
        values = {
            "task_file": str(task_file),
            "workdir": str(workdir),
            "base_url": gateway.base_url(session),
            "api_key": session.id,
            "session": session.id,
            "model": self.model,
            "python": sys.executable,
            "reward_url": gateway.reward_url(session),
            "sample_id": sample.id,
        }
        env = self._environ(values)

        exit_code, timed_out = self._run(
            self.command,
            values,
            env=env,
            cwd=workdir,
            stdout=stdout,
            stderr=stderr,
            timeout=self.timeout,
            what="agent",
            group_id=group_id,
            abort=abort,
            session=session,
        )
        gateway.close_session(session)  # what is recorded now is the episode
        if not session.calls:
            ended = f"timed out after {self.timeout:g}s" if timed_out else f"exit code {exit_code}"
            raise EnvironmentError_(
                f"episode {group_id}: the harness agent made no policy call ({ended}); "
                f"stderr: {_tail(stderr) or '(empty)'}"
            )
        trajectory = build_trajectory(session.calls, sample, group_id)
        trajectory.done = not timed_out and not session.refused
        trajectory.info.update(
            exit_code=exit_code, timed_out=timed_out, stdout_tail=_tail(stdout), stderr_tail=_tail(stderr)
        )
        reward = 0.0
        if session.reward is not None:
            trajectory.info["posted_reward"] = session.reward
            reward += session.reward
        if self.verifier is not None:
            (episode_dir / "reference.json").write_text(
                json.dumps(encode_sample(sample), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            values.update(
                reference_file=str(episode_dir / "reference.json"),
                stdout_file=str(stdout),
                stderr_file=str(stderr),
                exit_code=str(exit_code),
            )
            verdict = self._verify(values, env, cwd=workdir, group_id=group_id, abort=abort)
            trajectory.info["verifier"] = verdict
            reward += verdict["reward"]
        trajectory.turns[-1].reward += reward
        return trajectory

    def close(self) -> None:
        """Close the session, kill and reap whatever still runs, remove the episode directory."""
        session, self._session = self._session, None
        if session is not None and self._gateway is not None:
            self._gateway.close_session(session)
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            _kill_group(proc)
        episode_dir, self._dir = self._dir, None
        if episode_dir is not None:
            shutil.rmtree(episode_dir, ignore_errors=True)

    # -- internals --------------------------------------------------------------------------
    def _make_dir(self) -> Path:
        """A fresh directory for the episode: ``task.json``, captured output and ``work/``."""
        if self.workdir_root is not None:
            self.workdir_root.mkdir(parents=True, exist_ok=True)
        self._dir = Path(tempfile.mkdtemp(prefix="monorl-harness-", dir=self.workdir_root)).resolve()
        return self._dir

    def _environ(self, values: Mapping[str, str]) -> dict[str, str]:
        env = dict(os.environ)
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
        for key in ("NO_PROXY", "no_proxy"):  # the gateway is on localhost whatever proxy is configured
            env[key] = ",".join(filter(None, [env.get(key), "127.0.0.1,localhost"]))
        env.update({key: _fill(value, values) for key, value in self.env.items()})
        return env

    def _verify(
        self,
        values: Mapping[str, str],
        env: Mapping[str, str],
        *,
        cwd: Path,
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
            cwd=cwd,
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

    def _run(
        self,
        command: list[str],
        values: Mapping[str, str],
        *,
        env: Mapping[str, str],
        cwd: Path,
        stdout: Path,
        stderr: Path,
        timeout: float,
        what: str,
        group_id: str,
        abort: threading.Event | None,
        session: Session | None = None,
    ) -> tuple[int, bool]:
        """Run ``command`` in its own process group until it exits or ``timeout`` passes;
        ``(exit code, timed out)``. The group is always killed and reaped before this returns.
        Raises ``EpisodeAborted`` once ``abort`` is set or a generation of ``session`` was
        cancelled, and the engine's error when a generation of ``session`` failed."""
        argv = [_fill(arg, values) for arg in command]
        try:
            with open(stdout, "wb") as out, open(stderr, "wb") as err:
                proc = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=dict(env),
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
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
            _kill_group(proc)  # the process, and whatever it left running in its group
            self._proc = None
        if session is not None and session.failure is not None:
            raise session.failure
        if stopped():
            # imported here: the orchestrator package imports torch, environments stay torch-free
            from monorl.orchestrator.episodes import EpisodeAborted

            raise EpisodeAborted(f"episode {group_id} aborted while its harness {what} was running")
        return proc.returncode, timed_out


def make_harness_env(
    params: Mapping[str, Any], *, max_turns: int | None = None, model: str = "monorl"
) -> HarnessEnv:
    """:class:`HarnessEnv` from ``env.params`` (module docstring); ``null`` params are unset."""
    given = {key: value for key, value in params.items() if value is not None}
    unknown = sorted(set(given) - set(_PARAMS))
    if unknown:
        raise ConfigError(f"unknown env.params for env.type 'harness': {unknown}; expected {list(_PARAMS)}")
    if not given.get("command"):
        raise ConfigError(
            "env.type 'harness' needs env.params.command (the agent's argv list or command string)"
        )
    return HarnessEnv(**{"model": model, **given}, max_turns=max_turns)


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


def _fill(text: str, values: Mapping[str, str]) -> str:
    """``text`` with the ``{placeholder}``s named in ``values`` replaced; other braces stay."""
    return _PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), text)


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


def _tail(path: Path) -> str:
    """The last :data:`TAIL_CHARS` characters of a captured output file."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 4 * TAIL_CHARS))
        return f.read().decode("utf-8", errors="replace")[-TAIL_CHARS:]


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the process group ``proc`` leads (it was started in its own session) and reap
    ``proc``. Harmless when everything in the group has already exited."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


__all__ = ["HarnessEnv", "make_harness_env", "task_payload"]
