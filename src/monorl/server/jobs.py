"""Job manager: spawns training runs and pipelines as subprocesses and reconciles their state
from the run directory (``status.json``) plus process liveness.

Jobs persist in ``$MONORL_HOME/jobs.json`` so the server can be restarted while runs continue.
A job record::

    {id, kind, name, run_dir, config, state, pid, created_at, started_at, finished_at, error,
     updated_at, step, message}

``kind`` is ``"run"`` (``config`` = RunConfig, command ``monorl train``) or ``"pipeline"``
(``config`` = PipelineConfig, command ``monorl pipeline``; stages live in ``<run_dir>/pipeline.json``).
``state`` is a :data:`monorl.core.types.RunState`; ``step``/``message``/``updated_at`` mirror the
trainer's ``status.json`` so list views do not have to open every run directory.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import psutil

from monorl.config.pipeline import PipelineConfig, is_pipeline_payload
from monorl.config.schema import RunConfig
from monorl.core.errors import MonorlError
from monorl.core.reporting import CONFIG_FILE, LOG_FILE, read_status
from monorl.pipeline.stages import PIPELINE_CONFIG_FILE, default_pipeline_name

JOBS_FILE = "jobs.json"
TERMINAL_STATES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})
DEFAULT_TRAIN_CMD: tuple[str, ...] = (sys.executable, "-m", "monorl.cli", "train")
DEFAULT_PIPELINE_CMD: tuple[str, ...] = (sys.executable, "-m", "monorl.cli", "pipeline")
JOB_KINDS: tuple[str, ...] = ("run", "pipeline")

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class JobNotFound(MonorlError):
    pass


class JobActive(MonorlError):
    """Raised when an operation needs a finished job but the process is still running."""


def default_home() -> Path:
    return Path(os.environ.get("MONORL_HOME") or "~/.monorl").expanduser()


def train_command() -> list[str]:
    """Command prefix used to launch a run; ``MONORL_TRAIN_CMD`` (shell-split) overrides it."""
    override = os.environ.get("MONORL_TRAIN_CMD")
    if override:
        return shlex.split(override)
    return list(DEFAULT_TRAIN_CMD)


def pipeline_command() -> list[str]:
    """Command prefix used to launch a pipeline; ``MONORL_PIPELINE_CMD`` (shell-split) overrides it."""
    override = os.environ.get("MONORL_PIPELINE_CMD")
    if override:
        return shlex.split(override)
    return list(DEFAULT_PIPELINE_CMD)


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def default_run_name(config: RunConfig) -> str:
    model = Path(config.model.name_or_path.rstrip("/")).name or "model"
    return _SAFE_NAME.sub("-", f"{model}-{config.algorithm.name}-{_timestamp()}")


def _process_alive(pid: int | None, started_at: float | None) -> bool:
    """True when ``pid`` is a live, non-zombie process that started no later than the job did
    (guards against pid reuse after a server restart)."""
    if not pid:
        return False
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if started_at is not None and proc.create_time() > started_at + 5.0:
            return False
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


class JobManager:
    def __init__(
        self,
        home: str | os.PathLike[str] | None = None,
        *,
        train_cmd: list[str] | None = None,
        pipeline_cmd: list[str] | None = None,
        grace_seconds: float | None = None,
    ) -> None:
        self.home = Path(home).expanduser() if home is not None else default_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / JOBS_FILE
        self._train_cmd = train_cmd
        self._pipeline_cmd = pipeline_cmd
        env_grace = os.environ.get("MONORL_CANCEL_GRACE")
        self.grace_seconds = grace_seconds if grace_seconds is not None else float(env_grace or 5.0)
        self._lock = threading.RLock()
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._load()

    # -- persistence ---------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return
        for job in data.get("jobs", []):
            if isinstance(job, dict) and "id" in job:
                job.setdefault("kind", "run")
                self._jobs[job["id"]] = job

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"jobs": list(self._jobs.values())}, indent=2, default=str))
        os.replace(tmp, self.path)

    # -- creation ------------------------------------------------------------------------
    def _unique_run_dir(self, root: Path, name: str) -> tuple[str, Path]:
        """``<root>/<name>``; a suffix is appended when the directory exists."""
        candidate, n = root / name, 1
        while candidate.exists():
            n += 1
            candidate = root / f"{name}-{n}"
        return (name if n == 1 else f"{name}-{n}"), candidate

    def _resolve_run_dir(self, config: RunConfig) -> tuple[str, Path]:
        root = Path(config.logging.run_root).expanduser()
        name = _SAFE_NAME.sub("-", config.run_name) if config.run_name else default_run_name(config)
        return self._unique_run_dir(root, name)

    def _resolve_pipeline_dir(self, config: PipelineConfig) -> tuple[str, Path]:
        root = Path(config.base.logging.run_root).expanduser()
        name = _SAFE_NAME.sub("-", config.name) if config.name else default_pipeline_name(config)
        return self._unique_run_dir(root, name)

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate, write the config into the run dir and spawn the worker process.

        ``payload`` is a RunConfig dict, ``{"pipeline": <PipelineConfig dict>}`` or a bare
        PipelineConfig dict (``base`` + ``stages``). Raises
        :class:`monorl.core.errors.ConfigError` on an invalid config.
        """
        if isinstance(payload, dict) and isinstance(payload.get("pipeline"), dict):
            return self._create_pipeline(payload["pipeline"])
        if is_pipeline_payload(payload):
            return self._create_pipeline(payload)
        config = RunConfig.from_dict(payload)
        with self._lock:
            name, run_dir = self._resolve_run_dir(config)
            config.run_name = name
            run_dir.mkdir(parents=True, exist_ok=False)
            config_path = run_dir / CONFIG_FILE
            config.to_yaml(config_path)
            cmd = [*(self._train_cmd or train_command()), "--config", str(config_path), "--run-dir", str(run_dir)]
            return self._spawn("run", name, run_dir, config.to_dict(), cmd)

    def _create_pipeline(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = PipelineConfig.from_dict(payload)
        with self._lock:
            name, run_dir = self._resolve_pipeline_dir(config)
            config = config.model_copy(update={"name": name})
            run_dir.mkdir(parents=True, exist_ok=False)
            config_path = run_dir / PIPELINE_CONFIG_FILE
            config.to_yaml(config_path)
            cmd = [*(self._pipeline_cmd or pipeline_command()), "--config", str(config_path), "--run-dir", str(run_dir)]
            return self._spawn("pipeline", name, run_dir, config.to_dict(), cmd)

    def _spawn(self, kind: str, name: str, run_dir: Path, config: dict[str, Any], cmd: list[str]) -> dict[str, Any]:
        now = time.time()
        job: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "kind": kind,
            "name": name,
            "run_dir": str(run_dir),
            "config": config,
            "state": "pending",
            "pid": None,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "updated_at": now,
            "step": 0,
            "message": None,
            "command": cmd,
        }
        try:
            log = open(run_dir / LOG_FILE, "ab")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(run_dir),
                    start_new_session=True,
                    # The reporter writes run.log itself; its stderr echo would be captured
                    # into the same file and duplicate every line.
                    env={**os.environ, "MONORL_REPORTER_ECHO": "0"},
                )
            finally:
                log.close()
        except OSError as e:
            worker = "trainer" if kind == "run" else "pipeline"
            job.update(state="failed", error=f"failed to spawn {worker}: {e}", finished_at=time.time())
        else:
            self._procs[job["id"]] = proc
            job.update(state="running", pid=proc.pid, started_at=time.time())
        self._jobs[job["id"]] = job
        self._save()
        return dict(job)

    # -- queries -------------------------------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh_all()
            return sorted((dict(j) for j in self._jobs.values()), key=lambda j: j["created_at"], reverse=True)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(f"unknown job {job_id!r}")
            self._refresh_one(job)
            return dict(job)

    def refresh(self, job_id: str | None = None) -> None:
        with self._lock:
            if job_id is None:
                self._refresh_all()
            else:
                self._refresh_one(self._require(job_id))

    def is_alive(self, job_id: str) -> bool:
        with self._lock:
            return self._alive(self._require(job_id))

    def _require(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(f"unknown job {job_id!r}")
        return job

    # -- reconciliation ------------------------------------------------------------------
    def _alive(self, job: dict[str, Any]) -> bool:
        proc = self._procs.get(job["id"])
        if proc is not None:
            return proc.poll() is None
        return _process_alive(job.get("pid"), job.get("started_at"))

    def _exit_code(self, job: dict[str, Any]) -> int | None:
        proc = self._procs.get(job["id"])
        return proc.returncode if proc is not None else None

    def _refresh_all(self) -> None:
        changed = False
        for job in self._jobs.values():
            changed |= self._refresh_one(job, save=False)
        if changed:
            self._save()

    def _refresh_one(self, job: dict[str, Any], *, save: bool = True) -> bool:
        if job["state"] in TERMINAL_STATES:
            return False
        before = (job["state"], job.get("step"), job.get("message"), job.get("updated_at"))
        status = read_status(job["run_dir"])
        if status:
            if status.get("state") in TERMINAL_STATES or status.get("state") == "running":
                job["state"] = status["state"]
            job["step"] = int(status.get("step", job.get("step") or 0))
            job["message"] = status.get("message")
            job["updated_at"] = float(status.get("updated_at") or job["updated_at"])
            if job["state"] == "failed" and status.get("message"):
                job["error"] = status["message"]
        if job["state"] in TERMINAL_STATES:
            job["finished_at"] = job.get("finished_at") or job["updated_at"]
        elif not self._alive(job):
            code = self._exit_code(job)
            now = time.time()
            if job.get("cancel_requested_at"):
                job["state"] = "cancelled"
                job["message"] = job.get("message") or "cancelled by user"
            else:
                job["state"] = "failed"
                job["error"] = (
                    f"trainer exited with code {code} without reporting a terminal status"
                    if code is not None
                    else "trainer process is gone without reporting a terminal status"
                )
                job["message"] = job["error"]
            job["finished_at"] = now
            job["updated_at"] = now
        after = (job["state"], job.get("step"), job.get("message"), job.get("updated_at"))
        changed = before != after
        if changed and save:
            self._save()
        if job["state"] in TERMINAL_STATES:
            self._procs.pop(job["id"], None)
        return changed

    # -- lifecycle -----------------------------------------------------------------------
    def _signal(self, job: dict[str, Any], sig: signal.Signals) -> None:
        pid = job.get("pid")
        if not pid:
            return
        try:
            os.killpg(pid, sig)  # the child runs in its own session: this reaches Ray/engine children
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def cancel(self, job_id: str) -> dict[str, Any]:
        """SIGTERM the run (so the trainer can write ``status: cancelled``) and SIGKILL it after
        ``grace_seconds`` if it is still alive. Returns immediately."""
        with self._lock:
            job = self._require(job_id)
            self._refresh_one(job)
            if job["state"] in TERMINAL_STATES:
                return dict(job)
            job["cancel_requested_at"] = time.time()
            job["message"] = "cancel requested"
            job["updated_at"] = job["cancel_requested_at"]
            self._signal(job, signal.SIGTERM)
            self._save()
            snapshot = dict(job)
        threading.Thread(target=self._escalate, args=(job_id,), daemon=True, name=f"cancel-{job_id}").start()
        return snapshot

    def _escalate(self, job_id: str) -> None:
        deadline = time.time() + self.grace_seconds
        while time.time() < deadline:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or not self._alive(job):
                    break
            time.sleep(min(0.05, max(self.grace_seconds, 0.0)))
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and self._alive(job):
                self._signal(job, signal.SIGKILL)
                proc = self._procs.get(job_id)
                if proc is not None:
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
            if job is not None:
                self._refresh_one(job)

    def delete(self, job_id: str, *, remove_files: bool = False) -> None:
        with self._lock:
            job = self._require(job_id)
            self._refresh_one(job)
            if job["state"] not in TERMINAL_STATES:
                raise JobActive(f"job {job_id!r} is {job['state']}; cancel it before deleting")
            del self._jobs[job_id]
            self._procs.pop(job_id, None)
            self._save()
        if remove_files:
            import shutil

            shutil.rmtree(job["run_dir"], ignore_errors=True)

    def shutdown(self) -> None:
        """Reap finished child handles; running jobs keep going (they persist in jobs.json)."""
        with self._lock:
            for proc in self._procs.values():
                proc.poll()
            self._procs.clear()
