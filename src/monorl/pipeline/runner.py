"""``run_pipeline``: drives the stages of a :class:`~monorl.config.pipeline.PipelineConfig` in-process.

Train stages go through :func:`monorl.orchestrator.run` (own run dir, own ``status.json``);
rejection-sampling and eval stages through :mod:`monorl.pipeline.rejection`. The pipeline keeps
``pipeline.json`` (stage manifest) and a top-level ``status.json`` whose ``step`` is the number of
finished stages. SIGTERM/SIGINT cancel the active stage and the pipeline (the orchestrator installs
its own handlers for the duration of a train stage and restores ours afterwards).
"""

from __future__ import annotations

import gc
import signal
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monorl.config.pipeline import EvalStage, PipelineConfig, RejectionSampleStage, TrainStage
from monorl.core.errors import ConfigError, MonorlError
from monorl.core.reporting import RunReporter
from monorl.core.types import RunState
from monorl.pipeline.rejection import run_eval, run_rejection_sampling
from monorl.pipeline.stages import (
    PIPELINE_CONFIG_FILE,
    STAGES_DIR,
    Manifest,
    StageRecord,
    build_sampling_config,
    build_train_config,
    default_pipeline_name,
    known_data_configs,
    new_manifest,
    read_manifest,
    stage_final_model,
    validate_stage_configs,
)


@dataclass(slots=True)
class PipelineResult:
    state: RunState
    run_dir: str
    stages: list[dict[str, Any]] = field(default_factory=list)
    final_model_path: str | None = None
    error: str | None = None


class _Cancelled(Exception):
    """Internal: the active stage reported ``cancelled``."""


def resolve_pipeline_run_dir(config: PipelineConfig, run_dir: Path | None = None) -> tuple[str, Path]:
    """``(name, <base.logging.run_root>/<name>)``; an explicit ``run_dir`` wins, the name still
    defaults to ``<model>-pipeline-<timestamp>``."""
    name = config.name or default_pipeline_name(config)
    if run_dir is not None:
        return name, Path(run_dir).expanduser()
    return name, Path(config.base.logging.run_root).expanduser() / name


def run_pipeline(config: PipelineConfig, run_dir: Path | None = None, *, resume: bool = False) -> PipelineResult:
    name, target = resolve_pipeline_run_dir(config, run_dir)
    if config.name != name:
        config = config.model_copy(update={"name": name})
    reporter = RunReporter(target)
    config.to_yaml(reporter.run_dir / PIPELINE_CONFIG_FILE)
    stages_root = reporter.run_dir / STAGES_DIR

    try:
        manifest = _load_or_create_manifest(config, name, reporter.run_dir, resume=resume)
    except MonorlError as e:
        reporter.set_status("failed", str(e), step=0)
        return PipelineResult(state="failed", run_dir=str(reporter.run_dir), error=str(e))
    manifest.state = "running"
    manifest.error = None
    manifest.write()

    cancel = threading.Event()
    runner = _StageRunner(config, manifest, reporter, stages_root, cancel=cancel, resume=resume)
    with _signal_guard(cancel, reporter):
        reporter.set_status("running", f"pipeline {name}: {len(config.stages)} stages (resume={resume})", step=0)
        try:
            validate_stage_configs(config, stages_root)
        except (ConfigError, ValueError, TypeError) as e:
            return runner.finish("failed", f"invalid stage configuration: {e}")
        try:
            runner.run_all()
        except _Cancelled:
            return runner.finish("cancelled", None)
        except KeyboardInterrupt:
            runner.mark_active("cancelled", "interrupted")
            return runner.finish("cancelled", "interrupted")
        except Exception as e:  # noqa: BLE001 - every failure must land in status.json/pipeline.json
            tb = traceback.format_exc()
            reporter.log(tb)
            runner.mark_active("failed", f"{type(e).__name__}: {e}")
            return runner.finish("failed", f"{type(e).__name__}: {e}")
    if cancel.is_set():
        return runner.finish("cancelled", None)
    return runner.finish("succeeded", None)


# -- manifest handling ------------------------------------------------------------------------


def _load_or_create_manifest(config: PipelineConfig, name: str, run_dir: Path, *, resume: bool) -> Manifest:
    fresh = new_manifest(config, name, run_dir)
    if not resume:
        return fresh
    existing = read_manifest(run_dir)
    if existing is None:
        return fresh
    old = Manifest.from_dict(existing)
    if [(s.name, s.type) for s in old.stages] != [(s.name, s.type) for s in fresh.stages]:
        raise ConfigError(
            f"cannot resume {run_dir}: its pipeline.json stages {[s.name for s in old.stages]} do not match "
            f"the config stages {[s.name for s in fresh.stages]}"
        )
    for record, blank in zip(old.stages, fresh.stages, strict=True):
        record.run_dir = blank.run_dir
        if record.state != "succeeded":
            record.state = "pending"
            record.error = None
            record.finished_at = None
    old.final_model_path = None
    return old


class _StageRunner:
    def __init__(
        self,
        config: PipelineConfig,
        manifest: Manifest,
        reporter: RunReporter,
        stages_root: Path,
        *,
        cancel: threading.Event,
        resume: bool,
    ) -> None:
        self.config = config
        self.manifest = manifest
        self.reporter = reporter
        self.stages_root = stages_root
        self.cancel = cancel
        self.resume = resume
        self.previous_model: str | None = None
        self.previous_dataset: str | None = None
        self.active: StageRecord | None = None

    # -- bookkeeping --------------------------------------------------------------------------
    @property
    def finished(self) -> int:
        return sum(1 for s in self.manifest.stages if s.state == "succeeded")

    def _status(self, record: StageRecord, message: str) -> None:
        self.reporter.set_status("running", f"stage {record.index + 1}/{len(self.manifest.stages)} {message}", step=self.finished)

    def mark_active(self, state: str, error: str | None) -> None:
        record = self.active
        if record is None or record.state in ("succeeded", "failed", "cancelled"):
            return
        record.state = state
        record.error = error
        record.finished_at = time.time()
        self.manifest.write()

    def finish(self, state: RunState, error: str | None) -> PipelineResult:
        manifest = self.manifest
        manifest.state = state
        manifest.error = error
        manifest.final_model_path = self.previous_model if state == "succeeded" else None
        manifest.write()
        n = len(manifest.stages)
        message = {
            "succeeded": f"finished {n} stages",
            "cancelled": f"cancelled after {self.finished} of {n} stages",
            "failed": error or "failed",
        }[state]
        self.reporter.set_status(state, message, step=self.finished)
        return PipelineResult(
            state=state,
            run_dir=manifest.run_dir,
            stages=[s.to_dict() for s in manifest.stages],
            final_model_path=manifest.final_model_path,
            error=error,
        )

    # -- execution ----------------------------------------------------------------------------
    def run_all(self) -> None:
        for index, stage in enumerate(self.config.stages):
            record = self.manifest.stages[index]
            if self.cancel.is_set():
                raise _Cancelled
            if self.resume and record.state == "succeeded" and self._reuse(record):
                self.reporter.log(f"stage {index + 1} {record.name!r} already succeeded; reusing {record.run_dir}")
                continue
            self.active = record
            record.state = "running"
            record.error = None
            record.started_at = time.time()
            record.finished_at = None
            record.step = 0
            record.summary = {}
            self.manifest.write()
            self._status(record, f"{record.name} ({record.type})")
            self.reporter.log(f"stage {index + 1}/{len(self.config.stages)} {record.name!r} ({record.type}) -> {record.run_dir}")
            try:
                if isinstance(stage, TrainStage):
                    self._train(index, stage, record)
                elif isinstance(stage, RejectionSampleStage):
                    self._rejection(index, stage, record)
                elif isinstance(stage, EvalStage):
                    self._eval(index, stage, record)
            finally:
                gc.collect()
            self.manifest.write()
            if record.state == "cancelled":
                raise _Cancelled
            if record.state == "failed":
                raise RuntimeError(f"stage {record.name!r} failed: {record.error}")
            self.active = None
            self.reporter.set_status("running", f"stage {index + 1}/{len(self.config.stages)} {record.name} done", step=self.finished)

    def _reuse(self, record: StageRecord) -> bool:
        """Resume: adopt a finished stage's outputs if they still exist."""
        if record.type == "train":
            final = stage_final_model(record.run_dir)
            if not final.is_dir():
                return False
            self.previous_model = str(final)
            return True
        if record.type == "rejection_sample":
            dataset = record.summary.get("dataset")
            if not dataset or not Path(dataset).is_file():
                return False
            self.previous_dataset = str(dataset)
            return True
        return True

    def _train(self, index: int, stage: TrainStage, record: StageRecord) -> None:
        from monorl.orchestrator import run as run_training

        run_config = build_train_config(
            self.config,
            index,
            stages_root=self.stages_root,
            previous_model=self.previous_model,
            previous_dataset=self.previous_dataset,
        )
        stage_dir = Path(record.run_dir)
        resume_stage = self.resume and (stage_dir / "checkpoints").is_dir()
        result = run_training(run_config, stage_dir, resume=resume_stage)
        record.step = int(result.final_step)
        record.summary = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}
        record.state = result.state
        record.error = result.error if result.state == "failed" else None
        record.finished_at = time.time()
        if result.state == "succeeded":
            final = result.final_model_path or str(stage_final_model(stage_dir))
            self.previous_model = final
            record.summary["final_model_path"] = final

    def _rejection(self, index: int, stage: RejectionSampleStage, record: StageRecord) -> None:
        run_config = build_sampling_config(self.config, index, previous_model=self.previous_model)
        output = Path(stage.output).expanduser() if stage.output else None
        try:
            outcome = run_rejection_sampling(
                stage,
                run_config,
                run_dir=Path(record.run_dir),
                cancel=self.cancel,
                output=output,
                known_data=known_data_configs(self.config),
            )
        except KeyboardInterrupt:
            record.state, record.error, record.finished_at = "cancelled", "interrupted", time.time()
            return
        except Exception as e:  # noqa: BLE001 - recorded on the stage, re-raised by run_all
            record.state, record.error, record.finished_at = "failed", f"{type(e).__name__}: {e}", time.time()
            self.reporter.log(traceback.format_exc())
            return
        record.step = outcome.steps
        record.summary = outcome.summary()
        record.finished_at = time.time()
        if outcome.cancelled:
            record.state = "cancelled"
            return
        record.state = "succeeded"
        self.previous_dataset = outcome.dataset

    def _eval(self, index: int, stage: EvalStage, record: StageRecord) -> None:
        run_config = build_sampling_config(self.config, index, previous_model=self.previous_model)
        try:
            outcome = run_eval(stage, run_config, run_dir=Path(record.run_dir), cancel=self.cancel)
        except KeyboardInterrupt:
            record.state, record.error, record.finished_at = "cancelled", "interrupted", time.time()
            return
        except Exception as e:  # noqa: BLE001
            record.state, record.error, record.finished_at = "failed", f"{type(e).__name__}: {e}", time.time()
            self.reporter.log(traceback.format_exc())
            return
        record.step = outcome.steps
        record.summary = outcome.summary()
        record.finished_at = time.time()
        record.state = "cancelled" if outcome.cancelled else "succeeded"


# -- signals ----------------------------------------------------------------------------------


@contextmanager
def _signal_guard(cancel: threading.Event, reporter: RunReporter) -> Iterator[None]:
    """First SIGINT/SIGTERM sets ``cancel`` (the active sampling stage stops at the next chunk,
    the pipeline stops before the next stage); a second one raises ``KeyboardInterrupt``. Train
    stages temporarily replace these handlers with the orchestrator's own. Only the main thread
    may install handlers."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        if cancel.is_set():
            raise KeyboardInterrupt(name)
        cancel.set()
        reporter.log(f"received {name}: cancelling the pipeline after the current stage step")

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, handler)
    try:
        yield
    finally:
        for sig, prev in previous.items():
            signal.signal(sig, prev)


__all__ = ["PipelineResult", "resolve_pipeline_run_dir", "run_pipeline"]
