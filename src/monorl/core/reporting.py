"""Run directory layout and the reporter every backend writes through.

::

    <run_dir>/
      config.yaml            resolved RunConfig
      status.json            {"state", "message", "step", "updated_at", "started_at", "pid", ...}
      metrics.jsonl          one JSON object per line: {"step", "time", "phase", <metric>: <float>...}
      trajectories.jsonl     sampled rollouts: {"step", "group_id", "prompt", "response", "reward", ...}
      run.log                human-readable log
      checkpoints/step-<n>/  resumable trainer state
      final/                 HF-format export of the final policy

The web server and CLI only ever read these files; they never import backend code.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from monorl.core.types import RunState, Trajectory, message_text

STATUS_FILE = "status.json"
METRICS_FILE = "metrics.jsonl"
TRAJECTORIES_FILE = "trajectories.jsonl"
LOG_FILE = "run.log"
CONFIG_FILE = "config.yaml"
CHECKPOINT_DIR = "checkpoints"
FINAL_DIR = "final"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


class RunReporter:
    """Thread-safe writer for the run directory. Cheap to construct; creates ``run_dir``."""

    def __init__(self, run_dir: str | os.PathLike[str], *, echo: bool | None = None) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if echo is None:
            # Processes whose stderr is already captured into run.log (server-spawned jobs) set
            # MONORL_REPORTER_ECHO=0 so each line is written once.
            echo = os.environ.get("MONORL_REPORTER_ECHO", "1") not in {"0", "false", "no"}
        self._echo = echo
        self._started_at = time.time()
        self._status: dict[str, Any] = {
            "state": "pending",
            "message": None,
            "step": 0,
            "started_at": self._started_at,
            "updated_at": self._started_at,
            "pid": os.getpid(),
        }

    # -- status --------------------------------------------------------------------------
    def set_status(
        self,
        state: RunState,
        message: str | None = None,
        *,
        step: int | None = None,
        **extra: Any,
    ) -> None:
        with self._lock:
            self._status.update({"state": state, "updated_at": time.time()})
            if message is not None:
                self._status["message"] = message
            if step is not None:
                self._status["step"] = step
            self._status.update(extra)
            _atomic_write_json(self.run_dir / STATUS_FILE, self._status)
        if message:
            self.log(f"[{state}] {message}")

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    # -- metrics -------------------------------------------------------------------------
    def log_metrics(self, step: int, metrics: dict[str, float], *, phase: str = "train") -> None:
        row: dict[str, Any] = {"step": int(step), "time": time.time(), "phase": phase}
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = v
        with self._lock:
            with open(self.run_dir / METRICS_FILE, "a") as f:
                f.write(json.dumps(row) + "\n")
            self._status["step"] = max(int(self._status.get("step", 0)), int(step))
            self._status["updated_at"] = row["time"]
            _atomic_write_json(self.run_dir / STATUS_FILE, self._status)

    def log_trajectories(self, step: int, trajectories: Sequence[Trajectory], *, max_items: int = 8) -> None:
        rows = []
        for t in list(trajectories)[:max_items]:
            rows.append(
                {
                    "step": int(step),
                    "sample_id": t.sample.id,
                    "group_id": t.group_id,
                    "prompt": _last_user_text(t.sample.messages),
                    "response": t.response_text,
                    "reward": float(t.reward),
                    "reward_breakdown": {k: float(v) for k, v in t.reward_breakdown.items()},
                    "num_turns": len(t.turns),
                    "num_response_tokens": t.num_response_tokens,
                    "reference": _jsonable(t.sample.reference),
                }
            )
        with self._lock, open(self.run_dir / TRAJECTORIES_FILE, "a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")

    # -- log -----------------------------------------------------------------------------
    def log(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        with self._lock, open(self.run_dir / LOG_FILE, "a") as f:
            f.write(line + "\n")
        if self._echo:
            print(line, file=sys.stderr, flush=True)

    # -- paths ---------------------------------------------------------------------------
    @property
    def checkpoint_root(self) -> Path:
        return self.run_dir / CHECKPOINT_DIR

    def checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_root / f"step-{step}"

    @property
    def final_dir(self) -> Path:
        return self.run_dir / FINAL_DIR


# -- readers used by the server/CLI -------------------------------------------------------


def read_status(run_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    p = Path(run_dir) / STATUS_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def iter_metrics(
    run_dir: str | os.PathLike[str], *, start_line: int = 0
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, row)`` starting at ``start_line`` (0-based) for incremental tailing."""
    p = Path(run_dir) / METRICS_FILE
    if not p.exists():
        return
    with open(p) as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_trajectories(
    run_dir: str | os.PathLike[str], *, start_line: int = 0
) -> Iterator[tuple[int, dict[str, Any]]]:
    p = Path(run_dir) / TRAJECTORIES_FILE
    if not p.exists():
        return
    with open(p) as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            line = line.strip()
            if line:
                try:
                    yield i, json.loads(line)
                except json.JSONDecodeError:
                    continue


def list_checkpoints(run_dir: str | os.PathLike[str]) -> list[str]:
    root = Path(run_dir) / CHECKPOINT_DIR
    if not root.exists():
        return []
    steps = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("step-"):
            try:
                steps.append((int(d.name[5:]), str(d)))
            except ValueError:
                continue
    return [p for _, p in sorted(steps)]


def _last_user_text(messages: Sequence[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return message_text(m)  # type: ignore[arg-type]
    return message_text(messages[-1]) if messages else ""  # type: ignore[arg-type]


def _jsonable(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return str(x)
