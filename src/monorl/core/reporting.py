"""Run directory layout and the reporter every backend writes through.

::

    <run_dir>/
      config.yaml            resolved RunConfig
      status.json            {"state", "message", "step", "updated_at", "started_at", "pid", ...}
      metrics.jsonl          one JSON object per line: {"step", "time", "phase", <metric>: <float>...}
      trajectories.jsonl     sampled rollouts, one JSON object per line (see :func:`trajectory_row`)
      run.log                human-readable log
      checkpoints/step-<n>/  resumable trainer state
      final/                 HF-format export of the final policy

The web server and CLI only ever read these files; they never import backend code.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from monorl.core.types import MEDIA_PART_TYPES, Message, RunState, Trajectory, message_text

STATUS_FILE = "status.json"
METRICS_FILE = "metrics.jsonl"
TRAJECTORIES_FILE = "trajectories.jsonl"
LOG_FILE = "run.log"
CONFIG_FILE = "config.yaml"
CHECKPOINT_DIR = "checkpoints"
FINAL_DIR = "final"

MAX_TEXT_CHARS = 20_000
"""Longest text kept per string of a trajectory row; the rest is cut with a visible marker."""
MAX_DATA_URL_BYTES = 64 * 1024
"""Largest inline ``data:`` media URL kept in a trajectory row."""
_MAX_MEDIA_REF_CHARS = 4096
_MAX_REPR_CHARS = 200
_MAX_JSON_DEPTH = 12


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

    def log_trajectories(
        self,
        step: int,
        trajectories: Sequence[Trajectory],
        *,
        max_items: int = 8,
        phase: str = "train",
        save_transcripts: bool = True,
        success_threshold: float = 1.0,
    ) -> None:
        """Append up to ``max_items`` rows (:func:`trajectory_row`) to ``trajectories.jsonl``."""
        rows = [
            trajectory_row(
                t,
                step=step,
                phase=phase,
                save_transcripts=save_transcripts,
                success_threshold=success_threshold,
            )
            for t in list(trajectories)[:max_items]
        ]
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


# -- trajectory rows ----------------------------------------------------------------------


def num_context_tokens(trajectory: Trajectory) -> int:
    """Largest context the policy reached in the episode: prompt + completion tokens of a turn."""
    return max((len(t.prompt_token_ids) + len(t.completion.token_ids) for t in trajectory.turns), default=0)


def trajectory_row(
    trajectory: Trajectory,
    *,
    step: int,
    phase: str = "train",
    save_transcripts: bool = True,
    success_threshold: float = 1.0,
) -> dict[str, Any]:
    """One ``trajectories.jsonl`` row. Always JSON-serialisable; never holds token ids or logprobs.

    ``phase`` is ``train`` | ``eval`` | ``rejection`` | ``sft``. With ``save_transcripts`` the row
    also carries ``threads`` (every conversation of the episode as chat messages) and ``turns``
    (``{index, thread, message_index, reward, finish_reason, prompt_tokens, response_tokens,
    info}`` per policy action, ``message_index`` pointing at the turn's assistant message in its
    thread). Text over :data:`MAX_TEXT_CHARS` is cut with a marker and inline media over
    :data:`MAX_DATA_URL_BYTES` is replaced by a note.
    """
    t = trajectory
    reward = float(t.reward)
    row: dict[str, Any] = {
        "step": int(step),
        "phase": phase,
        "sample_id": t.sample.id,
        "group_id": t.group_id,
        "prompt": _last_user_text(t.sample.messages),
        "response": t.response_text,
        "reward": reward,
        "reward_breakdown": {k: float(v) for k, v in t.reward_breakdown.items()},
        "success": reward >= success_threshold,
        "done": bool(t.done),
        "finish_reason": t.turns[-1].completion.finish_reason if t.turns else None,
        "num_turns": len(t.turns),
        "num_response_tokens": t.num_response_tokens,
        "num_context_tokens": num_context_tokens(t),
        "reference": _jsonable(t.sample.reference),
        "info": _json_safe(t.info),
    }
    if save_transcripts:
        threads = t.threads
        row["threads"] = [[_message_row(m) for m in thread] for thread in threads]
        indices = message_indices(t, threads)
        row["turns"] = [
            {
                "index": i,
                "thread": int(turn.thread),
                "message_index": indices[i],
                "reward": float(turn.reward),
                "finish_reason": turn.completion.finish_reason,
                "prompt_tokens": len(turn.prompt_token_ids),
                "response_tokens": len(turn.completion.token_ids),
                "info": _json_safe(turn.info),
            }
            for i, turn in enumerate(t.turns)
        ]
    return row


def message_indices(trajectory: Trajectory, threads: Sequence[Sequence[Message]]) -> list[int | None]:
    """Position of every turn's assistant message in its thread. Recorded indices win; otherwise
    the turns of a thread map, in order, onto that thread's last N assistant messages."""
    by_thread: dict[int, list[int]] = {}
    for i, turn in enumerate(trajectory.turns):
        by_thread.setdefault(int(turn.thread), []).append(i)
    out: list[int | None] = [turn.message_index for turn in trajectory.turns]
    for thread, turn_indices in by_thread.items():
        if not 0 <= thread < len(threads):
            continue
        assistants = [j for j, m in enumerate(threads[thread]) if m.get("role") == "assistant"]
        if len(assistants) < len(turn_indices):
            continue
        tail = assistants[len(assistants) - len(turn_indices) :]
        for i, j in zip(turn_indices, tail, strict=True):
            if out[i] is None:
                out[i] = j
    return out


def _message_row(message: Any) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        return {"role": "unknown", "content": _short_repr(message)}
    row: dict[str, Any] = {"role": str(message.get("role", "unknown"))}
    content = message.get("content")
    if content is None or isinstance(content, str):
        row["content"] = _cut_text(content) if content else content
    elif isinstance(content, (list, tuple)):
        row["content"] = [_part_row(p) for p in content]
    else:
        row["content"] = _short_repr(content)
    for key in ("name", "tool_call_id"):
        if message.get(key) is not None:
            row[key] = _cut_text(str(message[key]))
    if message.get("tool_calls"):
        row["tool_calls"] = _json_safe(message["tool_calls"])
    return row


def _part_row(part: Any) -> Any:
    if isinstance(part, str):
        return {"type": "text", "text": _cut_text(part)}
    if not isinstance(part, Mapping):
        return {"type": "text", "text": _short_repr(part)}
    kind = part.get("type")
    if kind not in MEDIA_PART_TYPES:
        return _json_safe(part)
    row = {str(k): _json_safe(v) for k, v in part.items() if k != kind}
    value = part.get(kind)
    note = _media_omission(value)
    row[kind] = value if note is None else None
    if note is not None:
        row["note"] = note
    return row


def _media_omission(value: Any) -> str | None:
    """Why a media value is left out of the row; ``None`` when it is a reference worth keeping."""
    if value is None:
        return "no media reference recorded"
    if not isinstance(value, str):
        return f"in-memory {type(value).__name__} omitted"
    if value.startswith("data:"):
        size = len(value.encode("utf-8", "replace"))
        if size > MAX_DATA_URL_BYTES:
            return f"data URL of {size} bytes omitted (limit {MAX_DATA_URL_BYTES})"
        return None
    if len(value) > _MAX_MEDIA_REF_CHARS or "\n" in value:
        return f"inline media data of {len(value)} characters omitted"
    return None


def _cut_text(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return f"{text[:MAX_TEXT_CHARS]}\n… [{len(text) - MAX_TEXT_CHARS} characters truncated]"


def _short_repr(x: Any) -> str:
    try:
        text = repr(x)
    except Exception:  # noqa: BLE001 - a broken __repr__ must not lose the row
        return f"<{type(x).__name__}>"
    if len(text) > _MAX_REPR_CHARS:
        return f"{text[:_MAX_REPR_CHARS]}… <{type(x).__name__}>"
    return text


def _json_safe(x: Any, depth: int = 0) -> Any:
    """``x`` as plain JSON data: long strings cut, anything else unknown as a short string."""
    if x is None or isinstance(x, (bool, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if isinstance(x, str):
        return _cut_text(x)
    if depth >= _MAX_JSON_DEPTH:
        return _short_repr(x)
    if isinstance(x, Mapping):
        return {str(k): _json_safe(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [_json_safe(v, depth + 1) for v in x]
    return _short_repr(x)
