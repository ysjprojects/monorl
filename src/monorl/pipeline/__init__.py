"""Multi-stage pipelines (SFT → RL → rejection sampling → SFT → RL → eval).

Contract: :func:`run_pipeline`, :class:`PipelineResult`, :func:`read_manifest`; the run-dir layout
is documented in :mod:`monorl.pipeline.stages`. The runner (and torch) is imported lazily so the
server/CLI can read manifests without loading backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monorl.pipeline.stages import MANIFEST_FILE, PIPELINE_CONFIG_FILE, STAGES_DIR, read_manifest

if TYPE_CHECKING:
    from monorl.pipeline.runner import PipelineResult, resolve_pipeline_run_dir, run_pipeline

_LAZY = {"PipelineResult", "resolve_pipeline_run_dir", "run_pipeline"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from monorl.pipeline import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MANIFEST_FILE",
    "PIPELINE_CONFIG_FILE",
    "PipelineResult",
    "STAGES_DIR",
    "read_manifest",
    "resolve_pipeline_run_dir",
    "run_pipeline",
]
