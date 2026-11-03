"""``monorl.orchestrator.run``: the single entrypoint every launcher (CLI, server job, tests) calls.

Owns the run directory (:class:`~monorl.core.reporting.RunReporter`), signal handling and the
dispatch between pipeline backends (trl/verl/nemo_rl own their loop) and the component loop.
"""

from __future__ import annotations

import re
import signal
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from monorl.config.schema import RunConfig
from monorl.core.registry import registry
from monorl.core.reporting import CONFIG_FILE, RunReporter, list_checkpoints
from monorl.core.types import RunResult
from monorl.orchestrator.loop import ComponentLoop


def default_run_name(config: RunConfig) -> str:
    model = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(config.model.name_or_path.rstrip("/")).name).strip("-") or "model"
    return f"{model}-{config.algorithm.name}-{time.strftime('%Y%m%d-%H%M%S')}"


def resolve_run_dir(config: RunConfig) -> Path:
    """``<logging.run_root>/<run_name>`` (run_name defaults to ``<model>-<algorithm>-<timestamp>``)."""
    return Path(config.logging.run_root).expanduser() / (config.run_name or default_run_name(config))


def run(config: RunConfig, run_dir: Path | None = None, *, resume: bool = False) -> RunResult:
    if config.run_name is None:
        config = config.model_copy(update={"run_name": default_run_name(config)})
    target = Path(run_dir).expanduser() if run_dir is not None else resolve_run_dir(config)
    reporter = RunReporter(target)
    config.to_yaml(reporter.run_dir / CONFIG_FILE)
    cancel = threading.Event()
    backend = config.trainer.backend
    kind = "pipeline" if config.is_pipeline_backend else "component"
    with _signal_guard(cancel, reporter):
        reporter.set_status("running", f"starting {kind} backend {backend!r} (resume={resume})", step=0)
        try:
            if config.is_pipeline_backend:
                result = _run_pipeline(config, reporter)
            else:
                result = ComponentLoop(config, reporter, cancel=cancel, resume=resume).run()
        except KeyboardInterrupt:
            result = RunResult(
                state="cancelled",
                final_step=int(reporter.status.get("step", 0)),
                run_dir=str(reporter.run_dir),
                checkpoints=list_checkpoints(reporter.run_dir),
                error="interrupted",
            )
        except Exception as e:  # noqa: BLE001 - every failure must land in status.json
            tb = traceback.format_exc()
            reporter.log(tb)
            result = RunResult(
                state="failed",
                final_step=int(reporter.status.get("step", 0)),
                run_dir=str(reporter.run_dir),
                checkpoints=list_checkpoints(reporter.run_dir),
                error=f"{type(e).__name__}: {e}\n{tb}",
            )
            reporter.set_status("failed", f"{type(e).__name__}: {e}", step=result.final_step)
            return result
    message = {
        "succeeded": f"finished {result.final_step} steps",
        "cancelled": f"cancelled after step {result.final_step}",
        "failed": result.error or "failed",
    }.get(result.state, result.state)
    reporter.set_status(result.state, message, step=result.final_step)
    return result


def _run_pipeline(config: RunConfig, reporter: RunReporter) -> RunResult:
    backend = registry.get("pipeline_backend", config.trainer.backend)(config)
    result = backend.run(config, reporter.run_dir, reporter=reporter)
    if not result.checkpoints:
        result.checkpoints = list_checkpoints(reporter.run_dir)
    return result


@contextmanager
def _signal_guard(cancel: threading.Event, reporter: RunReporter) -> Iterator[None]:
    """SIGINT/SIGTERM set ``cancel`` (finish the step, checkpoint, status cancelled); a second
    signal raises ``KeyboardInterrupt`` for an immediate stop. Only the main thread may install
    handlers; elsewhere the run simply cannot be cancelled by signal."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        if cancel.is_set():
            raise KeyboardInterrupt(name)
        cancel.set()
        reporter.log(f"received {name}: finishing the current step, then stopping")

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, handler)
    try:
        yield
    finally:
        for sig, prev in previous.items():
            signal.signal(sig, prev)
