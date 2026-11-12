"""``pipeline_backend: verl`` — compile the RunConfig, stage run-dir artefacts, launch verl.

Launch is a subprocess ``python -m monorl.adapters.verl.entry <overrides>`` (cwd = run dir) so the
monorl process never imports verl/ray. Its stdout+stderr stream into ``run.log``; metrics arrive
through the ``Tracking.log`` patch (``metrics.jsonl``); ``status.json`` is kept current here.
Cancellation (``cancel()``, SIGTERM/SIGINT in the main thread) forwards SIGTERM to the verl driver,
which is what ``ray job stop`` does, then SIGKILLs after :attr:`VerlBackend.grace_seconds`.

verl writes checkpoints to ``trainer.default_local_dir`` (= ``<run_dir>/checkpoints``) as
``global_step_<n>/`` plus ``latest_checkpointed_iteration.txt`` (``verl/trainer/ppo/v1/trainer_base.py``);
``step-<n>`` symlinks are added so :func:`monorl.core.reporting.list_checkpoints` sees them.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import yaml

from monorl.adapters.verl.compiler import compile_config, to_overrides
from monorl.adapters.verl.data import export_datasets, parquet_has_images
from monorl.adapters.verl.patches import RUN_DIR_ENV
from monorl.adapters.verl.reward import write_reward_module
from monorl.adapters.verl.tools import write_tool_config
from monorl.config.schema import RunConfig
from monorl.core.errors import BackendUnavailable
from monorl.core.reporting import CHECKPOINT_DIR, CONFIG_FILE, LOG_FILE, RunReporter, iter_metrics
from monorl.core.types import RunResult

ENTRY_MODULE = "monorl.adapters.verl.entry"
COMPILED_CONFIG_FILE = "verl_config.yaml"
OVERRIDES_FILE = "verl_overrides.txt"
LATEST_ITERATION_FILE = "latest_checkpointed_iteration.txt"
GLOBAL_STEP_PREFIX = "global_step_"
HF_EXPORT_SUBDIR = Path("actor") / "huggingface"


def is_available() -> bool:
    return importlib.util.find_spec("verl") is not None


def checkpoint_dirs(run_dir: str | Path) -> list[tuple[int, Path]]:
    """``(step, path)`` of verl ``global_step_<n>`` folders under ``<run_dir>/checkpoints``, ascending."""
    root = Path(run_dir) / CHECKPOINT_DIR
    if not root.is_dir():
        return []
    found = []
    for d in root.iterdir():
        if d.is_dir() and not d.is_symlink() and d.name.startswith(GLOBAL_STEP_PREFIX):
            try:
                found.append((int(d.name[len(GLOBAL_STEP_PREFIX) :]), d))
            except ValueError:
                continue
    return sorted(found)


def link_checkpoints(run_dir: str | Path) -> list[str]:
    """Expose verl checkpoints as ``step-<n>`` symlinks; returns the checkpoint paths."""
    out = []
    for step, path in checkpoint_dirs(run_dir):
        alias = path.parent / f"step-{step}"
        if not alias.exists() and not alias.is_symlink():
            try:
                alias.symlink_to(path.name, target_is_directory=True)
            except OSError:
                pass
        out.append(str(path))
    return out


def latest_step(run_dir: str | Path) -> int | None:
    marker = Path(run_dir) / CHECKPOINT_DIR / LATEST_ITERATION_FILE
    if marker.exists():
        try:
            return int(marker.read_text().strip())
        except ValueError:
            pass
    dirs = checkpoint_dirs(run_dir)
    return dirs[-1][0] if dirs else None


def final_model_path(run_dir: str | Path) -> str | None:
    dirs = checkpoint_dirs(run_dir)
    if not dirs:
        return None
    last = dirs[-1][1]
    for hf in (last / HF_EXPORT_SUBDIR, last / "huggingface"):  # PPO actor export / SFT ``hf_model`` export
        if hf.is_dir():
            return str(hf)
    return str(last)


class VerlBackend:
    name = "verl"
    grace_seconds = 30.0
    poll_seconds = 1.0

    def __init__(self, config: RunConfig | None = None, *, python: str | None = None) -> None:
        extra = dict(config.trainer.extra) if config is not None else {}
        self.python = python or extra.get("verl_python") or sys.executable
        self._custom_python = bool(python or extra.get("verl_python"))
        self._proc: subprocess.Popen[bytes] | None = None
        self._cancel = threading.Event()
        self._reporter: RunReporter | None = None

    # -- staging ------------------------------------------------------------------------
    def prepare(self, config: RunConfig, run_dir: str | Path) -> list[str]:
        """Write parquet data, reward module, tool config and the compiled tree; return overrides."""
        run_dir = Path(run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / CONFIG_FILE
        if not config_path.exists():
            config.to_yaml(config_path)
        paths = export_datasets(config, run_dir)
        multimodal = parquet_has_images(paths["train"])
        if not config.is_sft:
            write_reward_module(config, run_dir, config_path=config_path)
            if config.env.type == "tool_calling":
                write_tool_config(config, run_dir)
        tree = compile_config(config, run_dir, multimodal=multimodal)
        (run_dir / COMPILED_CONFIG_FILE).write_text(yaml.safe_dump(tree, sort_keys=False))
        overrides = to_overrides(tree)
        (run_dir / OVERRIDES_FILE).write_text("\n".join(overrides) + "\n")
        return overrides

    def build_command(self, overrides: list[str], *, sft: bool = False) -> list[str]:
        """``python -m monorl.adapters.verl.entry [--sft] <overrides>``."""
        return [self.python, "-m", ENTRY_MODULE, *(["--sft"] if sft else []), *overrides]

    def build_env(self, config: RunConfig, run_dir: str | Path) -> dict[str, str]:
        env = dict(os.environ)
        env[RUN_DIR_ENV] = str(Path(run_dir).expanduser().resolve())
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HYDRA_FULL_ERROR", "1")
        if config.cluster.ray_address:
            env["RAY_ADDRESS"] = config.cluster.ray_address
        return env

    # -- lifecycle ----------------------------------------------------------------------
    def cancel(self) -> None:
        """Request cancellation: SIGTERM to the verl driver (``ray job stop`` semantics)."""
        self._cancel.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                return
            if self._reporter is not None:
                self._reporter.log("cancel requested: sent SIGTERM to verl driver")

    def run(self, config: RunConfig, run_dir: Path, *, reporter: RunReporter) -> RunResult:
        run_dir = Path(run_dir).expanduser().resolve()
        if not self._custom_python and not is_available():
            raise BackendUnavailable("pipeline_backend 'verl' needs `pip install monorl[verl]` (verl>=0.9.1)")
        reporter.set_status("pending", "compiling verl config")
        overrides = self.prepare(config, run_dir)
        cmd = self.build_command(overrides, sft=config.is_sft)
        reporter.log("launching: " + " ".join(cmd[:3]) + f" (+{len(overrides)} overrides, see {OVERRIDES_FILE})")

        installed = self._install_signal_handlers()
        tail: deque[str] = deque(maxlen=20)
        try:
            with open(run_dir / LOG_FILE, "ab") as log_file:
                self._proc = subprocess.Popen(
                    cmd, cwd=run_dir, env=self.build_env(config, run_dir), stdout=log_file, stderr=subprocess.STDOUT
                )
                reporter.set_status("running", "verl started", step=0, child_pid=self._proc.pid)
                self._reporter = reporter
                self._wait(run_dir, reporter)
            returncode = self._proc.returncode
        finally:
            self._restore_signal_handlers(installed)
        tail.extend(self._tail_log(run_dir))

        checkpoints = link_checkpoints(run_dir)
        step = latest_step(run_dir)
        last_metrics = self._last_metrics(run_dir)
        if step is None:
            step = int(last_metrics.get("step", 0))
        if self._cancel.is_set():
            state, error = "cancelled", None
            reporter.set_status("cancelled", "verl run cancelled", step=step)
        elif returncode == 0:
            state, error = "succeeded", None
            reporter.set_status("succeeded", "verl run finished", step=step)
        else:
            state = "failed"
            error = f"verl exited with code {returncode}\n" + "\n".join(tail)
            reporter.set_status("failed", f"verl exited with code {returncode}", step=step)
        metrics = {k: float(v) for k, v in last_metrics.items() if isinstance(v, (int, float)) and k != "step"}
        return RunResult(
            state=state,
            final_step=step,
            run_dir=str(run_dir),
            checkpoints=checkpoints,
            final_model_path=final_model_path(run_dir) if state == "succeeded" else None,
            error=error,
            metrics=metrics,
        )

    # -- internals ----------------------------------------------------------------------
    def _wait(self, run_dir: Path, reporter: RunReporter) -> None:
        assert self._proc is not None
        proc = self._proc
        next_line = 0
        last_step = 0
        deadline: float | None = None
        while True:
            if proc.poll() is not None:
                break
            for i, row in iter_metrics(run_dir, start_line=next_line):
                next_line = i + 1
                last_step = max(last_step, int(row.get("step", 0)))
            status = reporter.status
            if last_step > int(status.get("step", 0)):
                reporter.set_status("running", step=last_step)
            link_checkpoints(run_dir)
            if self._cancel.is_set():
                if deadline is None:
                    deadline = time.monotonic() + self.grace_seconds
                elif time.monotonic() > deadline:
                    reporter.log("verl driver did not exit in time: sending SIGKILL")
                    proc.kill()
                    deadline = float("inf")
            time.sleep(self.poll_seconds)
        proc.wait()

    def _install_signal_handlers(self) -> dict[int, Any]:
        if threading.current_thread() is not threading.main_thread():
            return {}
        previous: dict[int, Any] = {}

        def handler(signum: int, _frame: Any) -> None:
            self.cancel()

        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, handler)
        return previous

    @staticmethod
    def _restore_signal_handlers(previous: dict[int, Any]) -> None:
        for sig, old in previous.items():
            signal.signal(sig, old)

    @staticmethod
    def _tail_log(run_dir: Path, n: int = 20) -> list[str]:
        path = run_dir / LOG_FILE
        if not path.exists():
            return []
        with open(path, "rb") as f:
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        return lines[-n:]

    @staticmethod
    def _last_metrics(run_dir: Path) -> dict[str, Any]:
        last: dict[str, Any] = {}
        for _, row in iter_metrics(run_dir):
            if row.get("phase", "train") == "train":
                last = row
        return last


def make_backend(config: RunConfig, **kwargs: Any) -> VerlBackend:
    return VerlBackend(config, python=kwargs.get("python"))
