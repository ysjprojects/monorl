"""Pipeline run-directory layout, the ``pipeline.json`` manifest and per-stage config assembly.

::

    <run_dir>/
      pipeline.yaml          resolved PipelineConfig
      status.json            same shape as a run; ``step`` = stages finished
      pipeline.json          manifest (see :class:`Manifest`)
      run.log                pipeline-level log
      stages/<02d>-<name>/   one standard run dir per stage (train) or
                             {dataset.jsonl, status.json, metrics.jsonl, trajectories.jsonl, run.log}
"""

from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from monorl.config.pipeline import PipelineConfig, RejectionSampleStage, TrainStage
from monorl.config.schema import DataConfig, RunConfig
from monorl.core.reporting import FINAL_DIR

PIPELINE_CONFIG_FILE = "pipeline.yaml"
MANIFEST_FILE = "pipeline.json"
STAGES_DIR = "stages"
DATASET_FILE = "dataset.jsonl"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    return _SAFE_NAME.sub("-", name).strip("-") or "stage"


def default_pipeline_name(config: PipelineConfig) -> str:
    model = safe_name(Path(config.base.model.name_or_path.rstrip("/")).name) or "model"
    return f"{model}-pipeline-{time.strftime('%Y%m%d-%H%M%S')}"


def stage_dir_name(index: int, name: str) -> str:
    return f"{index:02d}-{safe_name(name)}"


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``overrides`` wins, nested dicts are merged, lists/scalars replaced."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# -- manifest ---------------------------------------------------------------------------------


@dataclass
class StageRecord:
    index: int
    name: str
    type: str
    state: str = "pending"
    run_dir: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    step: int = 0
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageRecord:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Manifest:
    name: str
    run_dir: str
    stages: list[StageRecord]
    state: str = "pending"
    final_model_path: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_dir": self.run_dir,
            "state": self.state,
            "final_model_path": self.final_model_path,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stages": [s.to_dict() for s in self.stages],
        }

    def write(self) -> None:
        self.updated_at = time.time()
        path = Path(self.run_dir) / MANIFEST_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        os.replace(tmp, path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        return cls(
            name=str(data.get("name", "")),
            run_dir=str(data.get("run_dir", "")),
            stages=[StageRecord.from_dict(s) for s in data.get("stages", []) if isinstance(s, dict)],
            state=str(data.get("state", "pending")),
            final_model_path=data.get("final_model_path"),
            error=data.get("error"),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
        )


def read_manifest(run_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The ``pipeline.json`` of a pipeline run dir, or ``None`` when absent/unreadable."""
    path = Path(run_dir) / MANIFEST_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def new_manifest(config: PipelineConfig, name: str, run_dir: Path) -> Manifest:
    stages_root = run_dir / STAGES_DIR
    records = [
        StageRecord(index=i, name=s.name, type=s.type, run_dir=str(stages_root / stage_dir_name(i, s.name)))
        for i, s in enumerate(config.stages)
    ]
    return Manifest(name=name, run_dir=str(run_dir), stages=records)


# -- effective stage configs ------------------------------------------------------------------

PREVIOUS_MODEL = "previous"
PREVIOUS_DATASET = "previous"


def previous_dataset_config(data: DataConfig, dataset: str | os.PathLike[str]) -> DataConfig:
    """Data config for a train stage reading a rejection-sampling ``dataset.jsonl``: chat format,
    the trailing assistant turn becomes the SFT target (``metadata["assistant_turn"]``) and the
    row's ``reference`` column the verifier reference. Prompts were already rendered, so the
    template/field mapping of the base data config does not apply."""
    return data.model_copy(
        update={
            "uri": str(dataset),
            "split": "train",
            "hf_config_name": None,
            "format": "chat",
            "messages_field": "messages",
            "prompt_field": None,
            "answer_field": None,
            "response_field": None,
            "image_field": None,
            "prompt_template": None,
            "streaming": False,
        }
    )


def build_train_config(
    config: PipelineConfig,
    index: int,
    *,
    stages_root: Path,
    previous_model: str | None,
    previous_dataset: str | None,
) -> RunConfig:
    """Effective :class:`RunConfig` for train stage ``index``.

    ``overrides`` are deep-merged over the base; ``init_from`` picks the policy
    (``previous`` falls back to the merged model when no train stage ran yet); ``data_from:
    previous`` swaps the data section for the preceding rejection-sampling output; ``run_name``
    and ``logging.run_root`` place the run under ``<pipeline>/stages/``.
    """
    stage = config.stages[index]
    if not isinstance(stage, TrainStage):
        raise TypeError(f"stage {index} ({stage.name!r}) is {stage.type}, not train")
    merged = deep_merge(config.base.to_dict(), stage.overrides)
    model = merged.setdefault("model", {})
    if stage.init_from == "base":
        model["name_or_path"] = config.base.model.name_or_path
    elif stage.init_from == PREVIOUS_MODEL:
        if previous_model is not None:
            model["name_or_path"] = previous_model
    else:
        model["name_or_path"] = stage.init_from
    merged["run_name"] = stage_dir_name(index, stage.name)
    merged.setdefault("logging", {})["run_root"] = str(stages_root)
    run_config = RunConfig.from_dict(merged)
    if stage.data_from == PREVIOUS_DATASET:
        if previous_dataset is None:
            raise ValueError(f"stage {stage.name!r}: data_from='previous' but no rejection_sample output exists")
        run_config = run_config.model_copy(update={"data": previous_dataset_config(run_config.data, previous_dataset)})
    return run_config


def build_sampling_config(config: PipelineConfig, index: int, *, previous_model: str | None) -> RunConfig:
    """Effective :class:`RunConfig` for a rejection_sample / eval stage: base + ``rollout_overrides``
    (+ ``data`` / ``reward_overrides``) with the current policy as the model."""
    stage = config.stages[index]
    if isinstance(stage, TrainStage):
        raise TypeError(f"stage {index} ({stage.name!r}) is a train stage")
    merged = config.base.to_dict()
    if previous_model is not None:
        merged["model"]["name_or_path"] = previous_model
    merged["rollout"] = deep_merge(merged["rollout"], stage.rollout_overrides)
    if stage.data is not None:
        merged["data"] = stage.data.model_dump(mode="json")
    if isinstance(stage, RejectionSampleStage) and stage.reward_overrides is not None:
        merged["rewards"] = copy.deepcopy(stage.reward_overrides)
    merged["run_name"] = stage_dir_name(index, stage.name)
    return RunConfig.from_dict(merged)


def stage_final_model(stage_run_dir: str | os.PathLike[str]) -> Path:
    return Path(stage_run_dir) / FINAL_DIR


def known_data_configs(config: PipelineConfig) -> list[DataConfig]:
    """Data configs a pipeline already describes (base + each train stage's merged ``data``
    overrides). Rejection-sampling ``mix`` uris are resolved against these first so a mixed-in
    cold-start set keeps its field mapping (``hf_config_name``, ``response_field``, ...)."""
    out = [config.base.data]
    base = config.base.data.model_dump(mode="json")
    for stage in config.stages:
        if isinstance(stage, TrainStage) and isinstance(stage.overrides.get("data"), dict):
            out.append(DataConfig.model_validate(deep_merge(base, stage.overrides["data"])))
    return out


def validate_stage_configs(config: PipelineConfig, stages_root: Path) -> None:
    """Build every stage config once with placeholder hand-offs so override typos fail before any
    stage runs. Raises :class:`monorl.core.errors.ConfigError` / ``ValueError``."""
    for i, stage in enumerate(config.stages):
        if isinstance(stage, TrainStage):
            build_train_config(
                config, i, stages_root=stages_root, previous_model=None, previous_dataset="<previous>"
            )
        else:
            build_sampling_config(config, i, previous_model=None)


__all__ = [
    "DATASET_FILE",
    "MANIFEST_FILE",
    "PIPELINE_CONFIG_FILE",
    "STAGES_DIR",
    "Manifest",
    "StageRecord",
    "build_sampling_config",
    "build_train_config",
    "deep_merge",
    "default_pipeline_name",
    "known_data_configs",
    "new_manifest",
    "previous_dataset_config",
    "read_manifest",
    "safe_name",
    "stage_dir_name",
    "stage_final_model",
    "validate_stage_configs",
]
