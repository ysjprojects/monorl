"""Stand-in for the ``docker`` executable (``tests/envs/conftest.py`` puts it on ``PATH``).

It records every invocation in ``$FAKE_DOCKER_STATE/calls.jsonl`` and keeps "containers" as
directories: ``run`` creates ``containers/<name>/root`` with the bind mounts as symlinks, ``exec``
runs the command on this machine with container paths translated into that root, ``cp`` copies
into it, ``restart`` and ``rm`` kill what ``exec`` started. As with a real daemon, a process an
``exec`` started outlives the ``exec`` client: it runs in a session of its own and only
``restart`` / ``rm`` end it.

``exec`` builds the environment from scratch (``PATH``, the container's and the call's ``-e``
variables, ``FAKE_ROOT`` = the container's root, ``FAKE_DOCKER_USER`` = ``-u``), so scripts that
address the container filesystem write ``${FAKE_ROOT:-}/logs/...`` and work in a real container
as well.

Knobs (environment): ``FAKE_DOCKER_DAEMON=down``, ``FAKE_DOCKER_OS`` (``docker info``'s operating
system), ``FAKE_DOCKER_FAIL`` (comma-separated subcommands that fail), ``FAKE_DOCKER_CGROUP``
(``<cpu usec>,<peak bytes>,<oom kills>``: the container's cgroup files).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

STATE = Path(os.environ.get("FAKE_DOCKER_STATE", "/nonexistent"))
VALUE_FLAGS = {
    "--name", "--label", "-v", "-w", "-e", "-u", "-t", "-f", "--entrypoint", "--network", "--add-host",
    "--cpus", "--memory", "--pids-limit", "--gpus", "--platform", "--format", "--filter", "--build-arg",
    "--target",
}  # fmt: skip
CONTAINER_PATHS = ("/monorl", "/logs", "/tests", "/sys/fs/cgroup")
"""Always the container's, whether or not they exist in its root yet."""


def options(args: list[str]) -> tuple[list[tuple[str, str | None]], list[str]]:
    """Leading flags of a subcommand (with their values) and the positional rest."""
    found: list[tuple[str, str | None]] = []
    i = 0
    while i < len(args) and args[i].startswith("-"):
        if args[i].startswith("--") and "=" in args[i]:
            flag, value = args[i].split("=", 1)
            found.append((flag, value))
            i += 1
        elif args[i] in VALUE_FLAGS:
            found.append((args[i], args[i + 1]))
            i += 2
        else:
            found.append((args[i], None))
            i += 1
    return found, args[i:]


def values(found: list[tuple[str, str | None]], *flags: str) -> list[str]:
    return [value for flag, value in found if flag in flags and value is not None]


def variables(found: list[tuple[str, str | None]]) -> dict[str, str]:
    """``-e NAME=VALUE`` and ``-e NAME`` (value from this process's environment)."""
    env: dict[str, str] = {}
    for item in values(found, "-e"):
        name, given, value = item.partition("=")
        if given:
            env[name] = value
        elif name in os.environ:
            env[name] = os.environ[name]
    return env


def fail(message: str, status: int = 1) -> None:
    print(message, file=sys.stderr)
    sys.exit(status)


def read(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def container(name: str) -> Path:
    path = STATE / "containers" / name
    if not (path / "meta.json").exists():
        fail(f"Error response from daemon: No such container: {name}")
    return path


def inside(root: Path, text: str, *, always: bool = False) -> str:
    """``text`` translated into the container root when it is a container path."""
    if not text.startswith("/"):
        return text
    known = text.startswith(CONTAINER_PATHS) or os.path.lexists(root / text.split("/")[1])
    return f"{root}{text}" if always or known else text


def kill_processes(path: Path) -> None:
    for pid_file in (path / "pids").glob("*"):
        try:
            os.killpg(int(pid_file.name), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        pid_file.unlink(missing_ok=True)


def info(args: list[str]) -> None:
    if os.environ.get("FAKE_DOCKER_DAEMON") == "down":
        fail("Cannot connect to the Docker daemon at unix:///fake/docker.sock. Is the docker daemon running?")
    print(os.environ.get("FAKE_DOCKER_OS", "Docker Desktop"))


def image(args: list[str]) -> None:
    _, rest = options(args[1:])  # image inspect --format ... NAME
    images = read(STATE / "images.json", {})
    if rest[0] not in images:
        fail(f"Error response from daemon: No such image: {rest[0]}")
    print(images[rest[0]]["workdir"])


def add_image(name: str, workdir: str, context: str | None) -> None:
    images = read(STATE / "images.json", {})
    images[name] = {"workdir": workdir, "context": context}
    (STATE / "images.json").write_text(json.dumps(images))


def pull(args: list[str]) -> None:
    _, rest = options(args)
    add_image(rest[0], "", None)


def build(args: list[str]) -> None:
    found, rest = options(args)
    context = Path(rest[0])
    dockerfile = Path((values(found, "-f") or [str(context / "Dockerfile")])[0])
    workdir = ""
    for line in dockerfile.read_text().splitlines():
        if line.upper().startswith("WORKDIR "):
            workdir = line.split(None, 1)[1].strip()
    add_image(values(found, "-t")[0], workdir, str(context))


def run(args: list[str]) -> None:
    found, rest = options(args)
    name = values(found, "--name")[0]
    images = read(STATE / "images.json", {})
    if rest[0] not in images:
        fail(f"Unable to find image '{rest[0]}' locally", 125)
    path = STATE / "containers" / name
    root = path / "root"
    (path / "pids").mkdir(parents=True)
    root.mkdir()
    for mount in values(found, "-v"):
        source, target = mount.split(":")[:2]
        link = root / target.lstrip("/")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source)
    workdir = (values(found, "-w") or [images[rest[0]]["workdir"] or "/"])[0]
    (root / workdir.lstrip("/")).mkdir(parents=True, exist_ok=True)
    context = images[rest[0]]["context"]
    if context is not None:  # what the image's Dockerfile copied into its WORKDIR
        image_workdir = root / images[rest[0]]["workdir"].lstrip("/")
        shutil.copytree(
            context, image_workdir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("Dockerfile")
        )
    if os.environ.get("FAKE_DOCKER_CGROUP"):
        usec, peak, oom = os.environ["FAKE_DOCKER_CGROUP"].split(",")
        cgroup = root / "sys/fs/cgroup"
        cgroup.mkdir(parents=True)
        (cgroup / "cpu.stat").write_text(f"usage_usec {usec}\nuser_usec 1\nsystem_usec 1\n")
        (cgroup / "memory.events").write_text(f"low 0\nhigh 0\nmax 0\noom 0\noom_kill {oom}\n")
        (cgroup / "memory.peak").write_text(f"{peak}\n")
    meta = {
        "image": rest[0],
        "command": rest[1:],
        "workdir": workdir,
        "env": variables(found),
        "labels": values(found, "--label"),
    }
    (path / "meta.json").write_text(json.dumps(meta))
    print(f"fake-{name}")


def exec_(args: list[str]) -> None:
    found, rest = options(args)
    path = container(rest[0])
    root, meta = path / "root", read(path / "meta.json", {})
    user = (values(found, "-u") or [""])[0]
    as_root = user == "0"
    env = {**meta["env"], **variables(found)}
    env = {key: inside(root, value) for key, value in env.items()}
    env.update(PATH=os.environ["PATH"], FAKE_ROOT=str(root), FAKE_DOCKER_USER=user)
    argv = [inside(root, arg, always=as_root) for arg in rest[1:]]
    cwd = inside(root, (values(found, "-w") or [meta["workdir"]])[0], always=True)
    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    except OSError as e:
        fail(f"OCI runtime exec failed: exec failed: unable to start container process: {e}", 127)
    # kept after the command has ended: what it left running is still in its process group
    (path / "pids" / str(proc.pid)).touch()
    status = proc.wait()
    sys.exit(status if status >= 0 else 128 - status)


def restart(args: list[str]) -> None:
    _, rest = options(args)
    kill_processes(container(rest[0]))


def cp(args: list[str]) -> None:
    source, target = args
    name, _, destination = target.partition(":")
    root = container(name) / "root"
    shutil.copytree(source.removesuffix("/."), root / destination.lstrip("/"), dirs_exist_ok=True)


def rm(args: list[str]) -> None:
    _, names = options(args)
    for name in names:
        path = STATE / "containers" / name.removeprefix("fake-")
        if path.exists():
            kill_processes(path)
            shutil.rmtree(path)


def ps(args: list[str]) -> None:
    found, _ = options(args)
    labels = [item.removeprefix("label=") for item in values(found, "--filter")]
    for path in sorted((STATE / "containers").glob("*")):
        meta = read(path / "meta.json", {})
        if all(label in meta.get("labels", []) for label in labels):
            print(path.name)


COMMANDS = {
    "info": info,
    "image": image,
    "pull": pull,
    "build": build,
    "run": run,
    "exec": exec_,
    "restart": restart,
    "cp": cp,
    "rm": rm,
    "ps": ps,
}


def main() -> None:
    command, args = sys.argv[1], sys.argv[2:]
    if not STATE.is_dir():  # a watchdog outliving its test: nothing to do
        sys.exit(0)
    record: dict[str, Any] = {"command": command, "argv": sys.argv[1:]}
    if command in ("run", "exec"):
        record["env"] = variables(options(args)[0])
    with open(STATE / "calls.jsonl", "a") as log:
        log.write(json.dumps(record) + "\n")
    if command in os.environ.get("FAKE_DOCKER_FAIL", "").split(","):
        fail(f"Error response from daemon: {command} failed (injected)", 125)
    COMMANDS[command](args)


if __name__ == "__main__":
    main()
