"""Where a harness agent and its verifier run: ``env.params.sandbox`` of ``env.type: harness``.

* ``backend: local`` (default) — processes on the trainer host, each in its own process group.
* ``backend: docker`` — one container per episode, driven through the Docker CLI. The container
  idles (``sh -c "sleep infinity"`` under ``--init``); the agent and the verifier are
  ``docker exec`` processes in it, so the verifier sees everything the agent left behind. The
  episode directory is bind-mounted at :data:`CONTAINER_DIR`, which is where the task file, the
  captured output and (by default) the working directory live inside the container.

Options of the docker backend (``docs/SANDBOX.md``): ``image`` or ``build`` (a context, built
once and cached under a content-addressed tag), ``build_timeout``, ``workdir``, ``mounts``,
``network``, ``cpus``, ``memory``, ``pids_limit``, ``gpus``, ``user``, ``platform``, ``python``,
``keep_entrypoint``, ``keep_processes``, ``args``, ``start_timeout``.

A container never outlives its episode: :meth:`DockerSandbox.close` removes it, and a watchdog
process per monorl process removes every container of that process once the process is gone,
however it ended (:func:`_watchdog`).

Torch-free, standard library only.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import math
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import IO, Any

from monorl.core.errors import ConfigError, EnvironmentError_, EpisodeAborted

log = logging.getLogger("monorl.envs.sandbox")

BACKENDS = ("local", "docker")
CONTAINER_DIR = "/monorl"
"""Where the episode directory (task file, captured output, ``work/``) is mounted in a container."""
IMAGE_WORKDIR = "image"
"""``sandbox.workdir`` value that selects the image's own ``WORKDIR``."""
CONTAINER_HOST = "host.docker.internal"
"""The name a container reaches the trainer host under."""
LOOPBACK = "127.0.0.1"
OWNER_LABEL = "monorl.harness.owner"
"""Container label naming the monorl process that started it (``<pid>-<token>``)."""
POLL_S = 0.05
"""How often a running process is checked against its deadline and the abort signals."""
CLI_TIMEOUT_S = 60.0
"""Seconds a Docker CLI management call (exec of a helper, restart, cp, rm, inspect) may take."""
DAEMON_TIMEOUT_S = 20.0
"""Seconds ``docker info`` may take before the daemon counts as not answering."""
PLATFORM = sys.platform

_KEEP_ALIVE = ["sh", "-c", "sleep infinity"]
_CGROUP_FILES = ["/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/memory.events", "/sys/fs/cgroup/memory.peak"]
_CLI_VARIABLES = ("HOME", "PATH", "TMPDIR", "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR")
"""Variables the Docker CLI reads itself; with ``DOCKER_*`` and ``*_PROXY`` they are handed to a
container by value, never through the CLI's own environment."""


# -- options ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Build:
    """An image built from a directory: ``docker build [-f dockerfile] [--build-arg ...] context``."""

    context: str
    dockerfile: str | None = None
    args: tuple[tuple[str, str], ...] = ()
    target: str | None = None


@dataclass(frozen=True)
class SandboxConfig:
    """Validated ``env.params.sandbox`` (module docstring, ``docs/SANDBOX.md``)."""

    backend: str = "local"
    image: str | None = None
    build: Build | None = None
    build_timeout: float = 600.0
    workdir: str | None = None
    mounts: tuple[str, ...] = ()
    network: str | None = None
    cpus: float | None = None
    memory: str | None = None
    pids_limit: int | None = None
    gpus: str | None = None
    user: str | None = None
    platform: str | None = None
    python: str = "python3"
    keep_entrypoint: bool = False
    keep_processes: bool = False
    args: tuple[str, ...] = ()
    start_timeout: float = 120.0

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None) -> SandboxConfig:
        """``env.params.sandbox`` as a config; ``null`` values are unset. Raises ``ConfigError``."""
        if params is not None and not isinstance(params, Mapping):
            raise ConfigError("env.params.sandbox must be a mapping, e.g. {backend: docker, image: ...}")
        config = cls().override(params or {})
        if config.backend == "local":
            given = sorted(
                key for key, value in (params or {}).items() if value is not None and key != "backend"
            )
            if given:
                raise ConfigError(
                    f"env.params.sandbox options {given} need `backend: docker` (the backend is 'local')"
                )
        return config

    def override(self, options: Mapping[str, Any]) -> SandboxConfig:
        """This config with ``options`` (raw ``env.params.sandbox`` entries, e.g. the ones a task
        brings) replacing its own. ``image`` and ``build`` replace each other."""
        unknown = sorted(set(options) - set(_PARSERS))
        if unknown:
            raise ConfigError(f"unknown env.params.sandbox options {unknown}; expected {sorted(_PARSERS)}")
        parsed = {key: _PARSERS[key](value) for key, value in options.items() if value is not None}
        if "image" in parsed and "build" not in parsed:
            parsed["build"] = None
        elif "build" in parsed and "image" not in parsed:
            parsed["image"] = None
        config = replace(self, **parsed)
        if config.image is not None and config.build is not None:
            raise ConfigError("env.params.sandbox takes `image` or `build`, not both")
        return config


def _text(name: str) -> Any:
    def parse(value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)) or not str(value).strip():
            raise ConfigError(f"env.params.sandbox.{name} must be a non-empty string, got {value!r}")
        return str(value).strip()

    return parse


def _flag(name: str) -> Any:
    def parse(value: Any) -> bool:
        if not isinstance(value, bool):
            raise ConfigError(f"env.params.sandbox.{name} must be true or false, got {value!r}")
        return value

    return parse


def _positive(name: str, kind: type = float) -> Any:
    def parse(value: Any) -> Any:
        try:
            number = math.nan if isinstance(value, bool) else float(value)
        except (TypeError, ValueError):
            number = math.nan
        if not 0 < number < math.inf or (kind is int and number != int(number)):
            what = "a positive integer" if kind is int else "a positive number"
            raise ConfigError(f"env.params.sandbox.{name} must be {what}, got {value!r}")
        return kind(number)

    return parse


def _backend(value: Any) -> str:
    backend = str(value).lower()
    if backend not in BACKENDS:
        raise ConfigError(f"env.params.sandbox.backend must be one of {list(BACKENDS)}, got {value!r}")
    return backend


def _existing_dir(value: Any, name: str) -> str:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise ConfigError(f"env.params.sandbox.{name} {str(value)!r} is not a directory")
    return str(path)


def _build(value: Any) -> Build:
    spec = {"context": value} if isinstance(value, (str, os.PathLike)) else value
    if not isinstance(spec, Mapping) or not spec.get("context"):
        raise ConfigError(
            "env.params.sandbox.build must be a build context directory or "
            "{context, dockerfile, args, target}"
        )
    unknown = sorted(set(spec) - {"context", "dockerfile", "args", "target"})
    if unknown:
        raise ConfigError(f"unknown env.params.sandbox.build options {unknown}")
    context = _existing_dir(spec["context"], "build.context")
    dockerfile = spec.get("dockerfile")
    if dockerfile is not None:
        dockerfile = str((Path(context) / str(dockerfile)).resolve())
        if not Path(dockerfile).is_file():
            raise ConfigError(f"env.params.sandbox.build.dockerfile {dockerfile!r} is not a file")
    args = spec.get("args") or {}
    if not isinstance(args, Mapping):
        raise ConfigError("env.params.sandbox.build.args must be a mapping of build arguments")
    target = spec.get("target")
    return Build(
        context=context,
        dockerfile=dockerfile,
        args=tuple(sorted((str(key), str(item)) for key, item in args.items())),
        target=None if target is None else str(target),
    )


def _workdir(value: Any) -> str:
    workdir = str(value)
    if workdir != IMAGE_WORKDIR and not PurePosixPath(workdir).is_absolute():
        raise ConfigError(
            f"env.params.sandbox.workdir must be an absolute container path or {IMAGE_WORKDIR!r}, "
            f"got {value!r}"
        )
    return workdir


def _mounts(value: Any) -> tuple[str, ...]:
    """``host:container[:options]`` bind mounts with the host side made absolute."""
    entries = [value] if isinstance(value, str) else value
    if not isinstance(entries, (list, tuple)):
        raise ConfigError("env.params.sandbox.mounts must be a list of 'host:container[:ro]' strings")
    mounts = []
    for entry in entries:
        host, _, rest = str(entry).partition(":")
        target = rest.split(":")[0]
        if not host or not PurePosixPath(target).is_absolute():
            raise ConfigError(
                f"env.params.sandbox.mounts entry {entry!r} must look like 'host:container[:ro]' "
                "with an absolute container path"
            )
        source = Path(host).expanduser().resolve()
        if not source.exists():  # the daemon would create it, as a root-owned directory
            raise ConfigError(f"env.params.sandbox.mounts: {host!r} does not exist")
        mounts.append(f"{source}:{rest}")
    return tuple(mounts)


def _network(value: Any) -> str:
    network = str(value)
    if network == "none":
        raise ConfigError(
            "env.params.sandbox.network 'none' leaves the agent without a way to the policy "
            "(the capture gateway runs on the trainer host)"
        )
    return network


def _args(value: Any) -> tuple[str, ...]:
    args = shlex.split(value) if isinstance(value, str) else value
    if not isinstance(args, (list, tuple)):
        raise ConfigError("env.params.sandbox.args must be a list of `docker run` arguments")
    return tuple(str(arg) for arg in args)


_PARSERS: dict[str, Any] = {
    "backend": _backend,
    "image": _text("image"),
    "build": _build,
    "build_timeout": _positive("build_timeout"),
    "workdir": _workdir,
    "mounts": _mounts,
    "network": _network,
    "cpus": _positive("cpus"),
    "memory": _text("memory"),
    "pids_limit": _positive("pids_limit", int),
    "gpus": _text("gpus"),
    "user": _text("user"),
    "platform": _text("platform"),
    "python": _text("python"),
    "keep_entrypoint": _flag("keep_entrypoint"),
    "keep_processes": _flag("keep_processes"),
    "args": _args,
    "start_timeout": _positive("start_timeout"),
}
assert set(_PARSERS) == {f.name for f in fields(SandboxConfig)}


# -- processes -------------------------------------------------------------------------------


def kill_group(proc: subprocess.Popen[Any]) -> None:
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


def _no_proxy(*hosts: str | None) -> str:
    return ",".join(dict.fromkeys(host for host in (*hosts, LOOPBACK, "localhost") if host))


# -- local -----------------------------------------------------------------------------------


class LocalSandbox:
    """Processes on the trainer host, in the episode's working directory."""

    backend = "local"
    python = sys.executable

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self.workdir = ""

    def gateway_address(self) -> tuple[str, str | None]:
        """``(bind address, advertised host)`` of the capture gateway for this sandbox."""
        return LOOPBACK, None

    def start(
        self,
        episode_dir: Path,
        *,
        logs: Path | None = None,
        env: Mapping[str, str] | None = None,
        abort: threading.Event | None = None,
    ) -> None:
        self.workdir = str(episode_dir / "work")

    def path(self, host_path: Path) -> str:
        """``host_path`` (inside the episode directory) as the sandbox's processes see it."""
        return str(host_path)

    def environ(self, gateway_host: str | None = None) -> dict[str, str]:
        """What every process starts from: monorl's own environment; the gateway is on localhost
        whatever proxy is configured."""
        env = dict(os.environ)
        for key in ("NO_PROXY", "no_proxy"):
            env[key] = _no_proxy(env.get(key), gateway_host)
        return env

    def popen(
        self,
        argv: list[str],
        *,
        env: Mapping[str, str],
        stdout: IO[bytes],
        stderr: IO[bytes],
        user: str | None = None,
    ) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            argv,
            cwd=self.workdir,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )

    def settle(self, *, restart: bool) -> None:
        """The agent has ended; its process group was killed with it."""

    def info(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


# -- docker ----------------------------------------------------------------------------------


def docker_cli() -> str:
    """Path of the Docker CLI; ``ConfigError`` when there is none."""
    path = shutil.which("docker")
    if path is None:
        raise ConfigError(
            "env.params.sandbox.backend 'docker' needs the Docker CLI: no `docker` executable on PATH"
        )
    return path


def _docker(
    docker: str, args: list[str], *, timeout: float = CLI_TIMEOUT_S, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a Docker CLI command to completion; ``EnvironmentError_`` when it cannot run or hangs."""
    try:
        return subprocess.run(
            [docker, *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=None if env is None else dict(env),
        )
    except subprocess.TimeoutExpired as e:
        raise EnvironmentError_(
            f"`docker {args[0]}` did not finish within {timeout:g}s; is the Docker daemon responding?"
        ) from e
    except OSError as e:
        raise EnvironmentError_(f"cannot run the Docker CLI {docker!r}: {e}") from e


def _said(result: subprocess.CompletedProcess[str]) -> str:
    """What a failed CLI call reported, shortened to its end."""
    return (result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}")[-1000:]


_lock = threading.Lock()
_daemons: dict[tuple[str, str | None, str | None], str] = {}
_tags: dict[tuple[Any, ...], str] = {}
_images: dict[tuple[str, str], str] = {}
_image_locks: dict[tuple[str, str], threading.Lock] = {}
_watchdogs: dict[tuple[int, str], subprocess.Popen[bytes]] = {}
_OWNER_TOKEN = secrets.token_hex(4)


def daemon_os(docker: str) -> str:
    """``OperatingSystem`` of the daemon behind ``docker`` (``"Docker Desktop"``, ``"Ubuntu
    24.04"``); asked once per process. ``EnvironmentError_`` when no daemon answers."""
    key = (docker, os.environ.get("DOCKER_HOST"), os.environ.get("DOCKER_CONTEXT"))
    with _lock:
        known = _daemons.get(key)
    if known is not None:
        return known
    result = _docker(docker, ["info", "--format", "{{.OperatingSystem}}"], timeout=DAEMON_TIMEOUT_S)
    if result.returncode != 0:
        raise EnvironmentError_(
            f"the Docker daemon does not answer ({_said(result)}); start Docker Desktop or the docker "
            "service, or use env.params.sandbox.backend: local"
        )
    with _lock:
        _daemons[key] = result.stdout.strip()
    return _daemons[key]


def daemon_in_vm(operating_system: str, platform: str | None = None) -> bool:
    """Whether containers run in a VM next to the trainer (Docker Desktop, any daemon reached
    from macOS or Windows) rather than on the trainer's own kernel (Docker Engine on Linux)."""
    return (platform or PLATFORM) != "linux" or operating_system == "Docker Desktop"


def gateway_address(*, in_vm: bool, network: str | None) -> tuple[str, str]:
    """``(bind address, advertised host)`` of the capture gateway for containers.

    * VM-based daemons forward ``host.docker.internal`` to the host's loopback: the gateway stays
      on ``127.0.0.1``.
    * Docker Engine on Linux: containers reach the host through the bridge gateway, which
      ``--add-host host.docker.internal:host-gateway`` names; the gateway has to listen there, so
      it binds every interface. With ``network: host`` the container shares the host's loopback.
    """
    if in_vm:
        return LOOPBACK, CONTAINER_HOST
    if network == "host":
        return LOOPBACK, LOOPBACK
    return "0.0.0.0", CONTAINER_HOST


def _owner() -> str:
    return f"{os.getpid()}-{_OWNER_TOKEN}"


_WATCHDOG = """
import subprocess, sys
docker, label = sys.argv[1:3]
sys.stdin.read()  # returns once the owner is gone, however it ended
listed = subprocess.run([docker, "ps", "-aq", "--filter", "label=" + label], capture_output=True, text=True)
if listed.stdout.split():
    subprocess.run([docker, "rm", "-f", "-v", *listed.stdout.split()], capture_output=True)
"""


def _watchdog(docker: str) -> None:
    """Make sure this process has its container watchdog: a detached helper holding the read end
    of a pipe whose write end only this process has. When the process is gone — exit, ``SIGTERM``,
    ``SIGKILL`` — the helper sees end-of-file and removes every container labelled with this
    process's owner id."""
    key = (os.getpid(), docker)
    with _lock:
        running = _watchdogs.get(key)
        if running is not None and running.poll() is None:
            return
        _watchdogs[key] = subprocess.Popen(
            [sys.executable, "-c", _WATCHDOG, docker, f"{OWNER_LABEL}={_owner()}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def _forget_watchdogs() -> None:
    """In a forked child: drop the parent's watchdog pipes, the child gets its own watchdog."""
    for proc in _watchdogs.values():
        if proc.stdin is not None:
            proc.stdin.close()
    _watchdogs.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_watchdogs)


@contextlib.contextmanager
def _image_lock(key: tuple[str, str], abort: threading.Event | None) -> Iterator[None]:
    """One pull or build per image at a time; waiting episodes still honour ``abort``."""
    with _lock:
        lock = _image_locks.setdefault(key, threading.Lock())
    while not lock.acquire(timeout=POLL_S):
        if abort is not None and abort.is_set():
            raise EpisodeAborted("aborted while waiting for a sandbox image")
    try:
        yield
    finally:
        lock.release()


def build_tag(build: Build, platform: str | None = None) -> str:
    """Content-addressed tag of the image ``build`` produces: the same context, Dockerfile and
    arguments give the same tag, so the image is built once and found again by later runs."""
    digest = hashlib.sha256()
    context = Path(build.context)
    described = (build.dockerfile, build.args, build.target, platform)
    digest.update(repr(described).encode())
    for path in sorted(p for p in context.rglob("*") if p.is_file() or p.is_symlink()):
        digest.update(path.relative_to(context).as_posix().encode())
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(os.readlink(path).encode())
        else:
            digest.update(b"x" if os.access(path, os.X_OK) else b"-")
            with open(path, "rb") as f:
                while chunk := f.read(1 << 20):
                    digest.update(chunk)
        digest.update(b"\0")
    return f"monorl-sandbox:{digest.hexdigest()[:20]}"


def _image_workdir(docker: str, image: str) -> str | None:
    """``WORKDIR`` of a local image (``/`` when it sets none); ``None`` when the image is absent."""
    result = _docker(docker, ["image", "inspect", "--format", "{{.Config.WorkingDir}}", image])
    return (result.stdout.strip() or "/") if result.returncode == 0 else None


def _long_command(
    docker: str, args: list[str], *, timeout: float, abort: threading.Event | None, what: str
) -> None:
    """A pull or build: runs up to ``timeout`` seconds and stops at once when ``abort`` is set."""
    with tempfile.TemporaryFile() as output:
        try:
            proc = subprocess.Popen(
                [docker, *args],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as e:
            raise EnvironmentError_(f"cannot run the Docker CLI {docker!r}: {e}") from e
        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                if abort is not None and abort.is_set():
                    raise EpisodeAborted(f"aborted while {what}")
                if time.monotonic() >= deadline:
                    raise EnvironmentError_(f"{what} did not finish within {timeout:g}s")
                time.sleep(POLL_S)
        finally:
            kill_group(proc)
        if proc.returncode != 0:
            output.seek(max(0, output.seek(0, os.SEEK_END) - 4000))
            tail = output.read().decode("utf-8", errors="replace").strip()[-2000:]
            raise EnvironmentError_(f"{what} failed (exit status {proc.returncode}): {tail}")


def ensure_image(
    docker: str, config: SandboxConfig, *, abort: threading.Event | None = None
) -> tuple[str, str]:
    """``(image reference, its WORKDIR)`` for ``config``: the configured image, pulled when it is
    not local, or the image of its build context, built when its tag does not exist yet. Once per
    image and process; parallel episodes wait for the one pull or build."""
    platform = [] if config.platform is None else ["--platform", config.platform]
    if config.build is not None:
        build = config.build
        with _lock:
            image = _tags.get((build, config.platform))
        if image is None:
            image = build_tag(build, config.platform)
            with _lock:
                _tags[(build, config.platform)] = image
        command = ["build", "-t", image, *platform]
        if build.dockerfile is not None:
            command += ["-f", build.dockerfile]
        for name, value in build.args:
            command += ["--build-arg", f"{name}={value}"]
        if build.target is not None:
            command += ["--target", build.target]
        command.append(build.context)
        what = f"building the sandbox image of {build.context}"
    elif config.image is not None:
        image = config.image
        command = ["pull", *platform, image]
        what = f"pulling the sandbox image {image}"
    else:
        raise ConfigError(
            "env.params.sandbox.backend 'docker' needs `image` or `build` "
            "(a task-directory dataset brings them with every task)"
        )
    key = (docker, image)
    with _image_lock(key, abort):
        workdir = _images.get(key)
        if workdir is None:
            workdir = _image_workdir(docker, image)
            if workdir is None:
                log.info("%s", what)
                _long_command(docker, command, timeout=config.build_timeout, abort=abort, what=what)
                workdir = _image_workdir(docker, image)
                if workdir is None:
                    raise EnvironmentError_(f"{what} left no image {image!r}")
            _images[key] = workdir
        return image, workdir


def _env_args(env: Mapping[str, str]) -> tuple[list[str], dict[str, str]]:
    """``-e`` arguments that hand ``env`` to a container, and the environment the CLI needs for
    them. Variables go by name (``-e NAME``, value taken from the CLI's environment), which keeps
    their values — the session id — out of the host's process list; the few names the CLI reads
    itself go by value."""
    args: list[str] = []
    cli_env = dict(os.environ)
    for key, value in env.items():
        by_value = key in _CLI_VARIABLES or key.startswith("DOCKER_") or key.upper().endswith("_PROXY")
        args += ["-e", f"{key}={value}" if by_value else key]
        if not by_value:
            cli_env[key] = value
    return args, cli_env


class DockerSandbox:
    """One container per episode (module docstring)."""

    backend = "docker"

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.docker = docker_cli()
        self.python = config.python
        self.name = f"monorl-harness-{secrets.token_hex(8)}"
        self.image: str | None = None
        self.workdir = ""
        self._mounted: list[tuple[Path, str]] = []
        self._created = False
        self._usage: dict[str, Any] = {}

    def _in_vm(self) -> bool:
        return daemon_in_vm(daemon_os(self.docker))

    def gateway_address(self) -> tuple[str, str | None]:
        return gateway_address(in_vm=self._in_vm(), network=self.config.network)

    def start(
        self,
        episode_dir: Path,
        *,
        logs: Path | None = None,
        env: Mapping[str, str] | None = None,
        abort: threading.Event | None = None,
    ) -> None:
        """Pull or build the image if needed and start the idle container: ``episode_dir`` at
        :data:`CONTAINER_DIR`, ``logs`` (if any) at ``/logs``, ``env`` as the container's
        environment."""
        config = self.config
        in_vm = self._in_vm()
        self.image, image_workdir = ensure_image(self.docker, config, abort=abort)
        if config.workdir is None:
            self.workdir = f"{CONTAINER_DIR}/work"
        else:
            self.workdir = image_workdir if config.workdir == IMAGE_WORKDIR else config.workdir
        self._mounted = [(episode_dir, CONTAINER_DIR)]
        if logs is not None:
            self._mounted.append((logs, "/logs"))
        _watchdog(self.docker)
        command = ["run", "-d", "--init", "--name", self.name, "--label", f"{OWNER_LABEL}={_owner()}"]
        if not config.keep_entrypoint:  # the image's service does not run, so neither does its health check
            command += ["--entrypoint", "", "--no-healthcheck"]
        for source, target in self._mounted:
            command += ["-v", f"{source}:{target}"]
        for mount in config.mounts:
            command += ["-v", mount]
        command += ["-w", self.workdir]
        if config.network is not None:
            command += ["--network", config.network]
        if not in_vm and config.network != "host":
            command += ["--add-host", f"{CONTAINER_HOST}:host-gateway"]
        for flag, value in (
            ("--cpus", config.cpus),
            ("--memory", config.memory),
            ("--pids-limit", config.pids_limit),
            ("--gpus", config.gpus),
            ("--platform", config.platform),
        ):
            if value is not None:
                command += [flag, f"{value:g}" if isinstance(value, float) else str(value)]
        env_args, cli_env = _env_args(env or {})
        command += [*env_args, *config.args, self.image, *_KEEP_ALIVE]
        self._created = True  # from here on a container of this name may exist
        result = _docker(self.docker, command, timeout=config.start_timeout, env=cli_env)
        if result.returncode != 0:
            raise EnvironmentError_(f"cannot start a sandbox container from {self.image!r}: {_said(result)}")

    def path(self, host_path: Path) -> str:
        for source, target in reversed(self._mounted):
            if host_path.is_relative_to(source):
                return str(PurePosixPath(target, *host_path.relative_to(source).parts))
        raise ValueError(f"{host_path} is not mounted in the sandbox")

    def environ(self, gateway_host: str | None = None) -> dict[str, str]:
        """Nothing of monorl's own environment enters a container; the gateway host is exempt
        from whatever proxy the image or the daemon configures."""
        no_proxy = _no_proxy(gateway_host)
        return {"NO_PROXY": no_proxy, "no_proxy": no_proxy}

    def popen(
        self,
        argv: list[str],
        *,
        env: Mapping[str, str],
        stdout: IO[bytes],
        stderr: IO[bytes],
        user: str | None = None,
    ) -> subprocess.Popen[bytes]:
        """``docker exec`` of ``argv`` in the working directory. Killing the returned CLI process
        does not stop ``argv`` inside the container: :meth:`settle` and :meth:`close` do."""
        env_args, cli_env = _env_args(env)
        command = [self.docker, "exec", "-w", self.workdir, *env_args]
        user = user or self.config.user
        if user is not None:
            command += ["-u", user]
        return subprocess.Popen(
            [*command, self.name, *argv],
            env=cli_env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )

    def root(self, argv: list[str], what: str) -> None:
        """Run a helper command as root in the container; ``EnvironmentError_`` when it fails."""
        result = _docker(self.docker, ["exec", "-u", "0", self.name, *argv])
        if result.returncode != 0:
            raise EnvironmentError_(f"sandbox {self.name}: cannot {what}: {_said(result)}")

    def copy_in(self, source: Path, target: str) -> None:
        """Copy the contents of the host directory ``source`` into the container directory
        ``target`` (created when missing)."""
        result = _docker(self.docker, ["cp", f"{source}{os.sep}.", f"{self.name}:{target}"])
        if result.returncode != 0:
            raise EnvironmentError_(f"sandbox {self.name}: cannot copy {source} to {target}: {_said(result)}")

    def settle(self, *, restart: bool) -> None:
        """The agent has ended: record what the container used, then — with ``restart`` — restart
        the container, which kills every process the agent left running and keeps the filesystem."""
        self._usage = self._read_usage()
        if restart:
            result = _docker(self.docker, ["restart", "-t", "0", self.name])
            if result.returncode != 0:
                raise EnvironmentError_(f"sandbox {self.name}: cannot restart the container: {_said(result)}")

    def _read_usage(self) -> dict[str, Any]:
        """CPU time, peak memory and OOM kills of the container so far, from its cgroup (v2);
        whatever cannot be read is left out."""
        try:
            result = _docker(self.docker, ["exec", self.name, "cat", *_CGROUP_FILES])
        except EnvironmentError_:
            return {}
        usage: dict[str, Any] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                if parts[0] == "usage_usec":
                    usage["cpu_seconds"] = int(parts[1]) / 1e6
                elif parts[0] == "oom_kill":
                    usage["oom_killed"] = int(parts[1]) > 0
            elif len(parts) == 1 and parts[0].isdigit():
                usage["memory_peak_bytes"] = int(parts[0])
        return usage

    def info(self) -> dict[str, Any]:
        """``Trajectory.info["sandbox"]``: where the episode ran, its limits and what it used."""
        info: dict[str, Any] = {"backend": self.backend, "image": self.image, "container": self.name}
        for key in ("cpus", "memory", "pids_limit"):
            if getattr(self.config, key) is not None:
                info[key] = getattr(self.config, key)
        return {**info, **self._usage}

    def close(self) -> None:
        """Remove the container (and its anonymous volumes). On Docker Engine, where files the
        container wrote into the episode directory belong to its users, they are first opened up
        so that this process can remove the directory."""
        if not self._created:
            return
        self._created = False
        try:
            if PLATFORM == "linux" and os.getuid() != 0:
                targets = [target for _, target in self._mounted]
                _docker(self.docker, ["exec", "-u", "0", self.name, "chmod", "-R", "a+rwX", *targets])
        except EnvironmentError_:
            pass
        try:
            result = _docker(self.docker, ["rm", "-f", "-v", self.name])
        except EnvironmentError_ as e:
            log.warning("sandbox container %s was not removed: %s", self.name, e)
            return
        if result.returncode != 0 and "No such container" not in result.stderr:
            log.warning("sandbox container %s was not removed: %s", self.name, _said(result))


def make_sandbox(config: SandboxConfig) -> LocalSandbox | DockerSandbox:
    return DockerSandbox(config) if config.backend == "docker" else LocalSandbox(config)


__all__ = [
    "BACKENDS",
    "CONTAINER_DIR",
    "CONTAINER_HOST",
    "IMAGE_WORKDIR",
    "Build",
    "DockerSandbox",
    "LocalSandbox",
    "SandboxConfig",
    "build_tag",
    "daemon_in_vm",
    "daemon_os",
    "docker_cli",
    "ensure_image",
    "gateway_address",
    "kill_group",
    "make_sandbox",
]
