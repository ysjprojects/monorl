"""Task directories as a dataset: the Harbor task format (Terminal-Bench 2 and the benchmarks
published for it).

A dataset URI that names a task directory, a directory of them, or a glob of either, is read as
one row per task::

    my-task/
      instruction.md           the agent's task                      -> the prompt
      task.toml                limits, environment, verifier config  -> the row's ``task`` mapping
      environment/Dockerfile   the task's image (or ``[environment].docker_image``)
      tests/test.sh            the verifier; writes /logs/verifier/reward.txt or reward.json
      solution/solve.sh        reference solution; never read

Specification: https://docs.harborframework.com/tasks/overview (``task.toml``:
``/tasks/configuration``, verifier: ``/tasks/verifier``, environment: ``/tasks/environment``),
implemented against ``harbor`` at commit ``feea734`` (``schema_version`` 1.4;
``src/harbor/models/task/config.py``, ``paths.py``, ``src/harbor/verifier/verifier.py``).

A row is ``{"id": <directory name>, "instruction": <instruction.md>, "task": {...}}``. The
``task`` mapping is what ``env.type: harness`` runs a sample with instead of its ``env.params``
(:mod:`monorl.envs.harness`): the image or build context, the working directory, CPU / memory /
GPU limits, the agent's timeout and user, the tests directory with the verifier's timeout, user
and environment. ``docs/SANDBOX.md`` has the field-by-field mapping and what is not supported;
a task that needs an unsupported feature is an error, or left out with
``data.extra.skip_unsupported: true``.
"""

from __future__ import annotations

import glob
import logging
import random
import re
import tomllib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from monorl.core.errors import DataError
from monorl.data import formats
from monorl.data.rows import RowRef, RowSource
from monorl.data.uri import ParsedURI

log = logging.getLogger("monorl.data.tasks")

TASK_FILE = "task.toml"
DEFAULT_BUILD_TIMEOUT_S = 600.0
DEFAULT_VERIFIER_TIMEOUT_S = 600.0
"""Harbor's defaults for ``[environment].build_timeout_sec`` and ``[verifier].timeout_sec``."""

_CANARY_LINE_RE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)
_SIZE_UNITS_MB = {"G": 1024.0, "M": 1.0, "K": 1 / 1024}
_HEALTHCHECK_DEFAULTS = {
    "interval_sec": 5.0,
    "timeout_sec": 30.0,
    "start_period_sec": 0.0,
    "start_interval_sec": 5.0,
    "retries": 3,
}


class UnsupportedTask(DataError):
    """A task directory monorl cannot run; the message says why."""


def find_task_dirs(path: str) -> list[Path]:
    """The task directories ``path`` names — a task directory, a directory whose children are
    task directories, or a glob matching either — sorted; empty when it names none."""
    matches = sorted(glob.glob(path)) if any(ch in path for ch in "*?[") else [path]
    found: list[Path] = []
    for match in map(Path, matches):
        if not match.is_dir():
            continue
        if (match / TASK_FILE).is_file():
            found.append(match)
        else:
            found.extend(
                child for child in sorted(match.iterdir()) if child.is_dir() and (child / TASK_FILE).is_file()
            )
    return found


def strip_canary(text: str) -> str:
    """``instruction.md`` without the leading canary comment lines benchmark authors add against
    training-data contamination (and the blank lines after them), as Harbor shows it to agents."""
    lines = text.split("\n")
    start = 0
    while start < len(lines) and _CANARY_LINE_RE.match(lines[start].strip()):
        start += 1
    while start < len(lines) and lines[start].strip() == "":
        start += 1
    return "\n".join(lines[start:])


def load_task(task_dir: Path) -> dict[str, Any]:
    """The row of one task directory (module docstring). Raises :class:`UnsupportedTask` naming
    everything about the task that cannot be honoured."""
    task_dir = task_dir.resolve()
    try:
        config = tomllib.loads((task_dir / TASK_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise UnsupportedTask(f"{TASK_FILE} cannot be read: {e}") from e
    environment, agent, verifier = (_table(config, name) for name in ("environment", "agent", "verifier"))
    environment_dir, tests_dir = task_dir / "environment", task_dir / "tests"
    instruction = task_dir / "instruction.md"

    problems: list[str] = []
    version = str(config.get("schema_version", config.get("version", "1.0")))
    if version.split(".")[0] != "1":
        problems.append(f"schema_version {version} (1.x is supported)")
    if config.get("steps") or (task_dir / "steps").is_dir():
        problems.append("multi-step task")
    elif not instruction.is_file():
        problems.append("no instruction.md")
    if (task_dir / "trajectory.json").is_file():
        problems.append("ships prior agent context (trajectory.json)")
    if str(environment.get("os", "linux")).lower() != "linux":
        problems.append(f"[environment].os = {environment['os']!r}")
    if verifier.get("environment_mode") == "separate" or "environment" in verifier:
        problems.append("separate verifier environment")
    elif not (tests_dir / "test.sh").is_file():
        problems.append("no tests/test.sh")
    network = environment.get("network_mode")
    if network is None and environment.get("allow_internet") is False:
        network = "no-network"
    for section, mode in (("environment", network), ("agent", agent.get("network_mode"))):
        # the agent has to reach the capture gateway, so its network cannot be closed or filtered
        if mode not in (None, "public"):
            problems.append(f"[{section}].network_mode = {mode!r}")
    if verifier.get("network_mode") not in (None, "public"):
        problems.append(f"[verifier].network_mode = {verifier['network_mode']!r}")

    sandbox: dict[str, Any] = {
        "workdir": environment.get("workdir") or "image",
        "keep_entrypoint": True,  # Harbor starts the image's entrypoint and idles as its command
        "build_timeout": float(environment.get("build_timeout_sec", DEFAULT_BUILD_TIMEOUT_S)),
    }
    if (environment_dir / "docker-compose.yaml").is_file():
        problems.append("multi-container environment (environment/docker-compose.yaml)")
    elif environment.get("docker_image"):
        sandbox["image"] = str(environment["docker_image"])
    elif (environment_dir / "Dockerfile").is_file():
        sandbox["build"] = {"context": str(environment_dir)}
    else:
        problems.append("no [environment].docker_image and no environment/Dockerfile")
    if environment.get("cpus") is not None:
        sandbox["cpus"] = float(environment["cpus"])
    try:
        memory_mb = _memory_mb(environment)
    except ValueError as e:
        problems.append(str(e))
    else:
        if memory_mb is not None:
            sandbox["memory"] = f"{memory_mb}m"
    if environment.get("gpus"):
        sandbox["gpus"] = str(environment["gpus"])
    if problems:
        raise UnsupportedTask("; ".join(problems))

    name = _table(config, "task").get("name") or task_dir.name
    task: dict[str, Any] = {
        "name": str(name),
        "dir": str(task_dir),
        "sandbox": sandbox,
        "env": {str(key): str(value) for key, value in _table(environment, "env").items()},
        "timeout": agent.get("timeout_sec"),
        "user": None if agent.get("user") is None else str(agent["user"]),
        "verifier": {
            "tests": str(tests_dir),
            "timeout": float(verifier.get("timeout_sec", DEFAULT_VERIFIER_TIMEOUT_S)),
            "user": None if verifier.get("user") is None else str(verifier["user"]),
            "env": {str(key): str(value) for key, value in _table(verifier, "env").items()},
        },
        "metadata": _table(config, "metadata"),
    }
    healthcheck = _table(environment, "healthcheck")
    if healthcheck:
        if not healthcheck.get("command"):
            raise UnsupportedTask("[environment.healthcheck] has no command")
        task["healthcheck"] = {**_HEALTHCHECK_DEFAULTS, **healthcheck}
    return {
        "id": task_dir.name,
        "instruction": strip_canary(instruction.read_text(encoding="utf-8")).strip(),
        "task": task,
    }


def _table(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def _memory_mb(environment: Mapping[str, Any]) -> int | None:
    """``[environment].memory_mb``, or the deprecated ``memory`` size string (``"2G"``, ``"512M"``)."""
    if environment.get("memory_mb") is not None:
        return int(environment["memory_mb"])
    memory = environment.get("memory")
    if memory is None:
        return None
    size = str(memory).strip().upper()
    if not size or size[-1] not in _SIZE_UNITS_MB:
        raise ValueError(f"[environment].memory = {memory!r} (expected a size like '2G' or '512M')")
    return int(float(size[:-1]) * _SIZE_UNITS_MB[size[-1]])


class TaskRows(RowSource):
    """One row per task directory; row ids are the directories."""

    streaming = False

    def __init__(self, parsed: ParsedURI, task_dirs: list[Path], *, skip_unsupported: bool = False) -> None:
        self.uri = parsed.raw
        self._rows: list[RowRef] = []
        unsupported: list[str] = []
        for task_dir in task_dirs:
            try:
                self._rows.append((str(task_dir), load_task(task_dir)))
            except UnsupportedTask as e:
                unsupported.append(f"{task_dir.name}: {e}")
        if unsupported and not skip_unsupported:
            raise DataError(
                f"{self.uri}: {len(unsupported)} of {len(task_dirs)} task directories cannot be run "
                f"({'; '.join(unsupported)}). Leave them out with data.extra.skip_unsupported: true."
            )
        if unsupported:
            log.warning("%s: leaving out %d unsupported tasks: %s", self.uri, len(unsupported), unsupported)
        if not self._rows:
            raise DataError(f"{self.uri}: none of the {len(task_dirs)} task directories can be run")

    def columns(self) -> dict[str, str]:
        return formats.infer_columns(self.head(formats.SNIFF_ROWS))

    def head(self, n: int) -> list[dict[str, Any]]:
        return [row for _, row in self._rows[:n]]

    def iter_rows(self, *, shuffle_seed: int | None = None) -> Iterator[RowRef]:
        rows = list(self._rows)
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(rows)
        return iter(rows)

    def known_len(self) -> int | None:
        return len(self._rows)


__all__ = ["TaskRows", "UnsupportedTask", "find_task_dirs", "load_task", "strip_canary"]
