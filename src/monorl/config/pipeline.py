"""Multi-stage training pipelines (DeepSeek-R1 style: cold-start SFT → RL → rejection sampling →
SFT → RL ...).

A :class:`PipelineConfig` holds one *base* :class:`RunConfig` and an ordered list of stages. Each
stage runs in its own run directory under ``<pipeline run dir>/stages/<index>-<name>/`` with the
standard run-dir layout, and the pipeline keeps a manifest (``pipeline.json``) so the CLI/server/UI
can show stage progress and resume skips finished stages.

Stage kinds:

* ``train`` — a full run (RL or SFT) whose config is the base merged with ``overrides``. The
  policy starts from the previous train stage's ``final/`` export (``init_from: previous``), the
  base model (``base``), or an explicit path. ``data_from: previous`` points ``data.uri`` at the
  dataset written by the preceding ``rejection_sample`` stage.
* ``rejection_sample`` — samples ``num_samples`` completions per prompt with the current policy,
  scores them with the configured rewards, keeps completions with reward >= ``min_reward``
  (at most ``max_per_prompt`` per prompt, optionally deduplicated / length-filtered), optionally
  mixes in extra SFT datasets, and writes a chat-format jsonl (prompt + chosen response) for the
  next SFT stage.
* ``eval`` — evaluation of the current policy on a dataset: greedy by default, or ``num_samples``
  sampled completions per prompt (adds ``eval/pass_at_k``); logs ``eval/reward_mean`` and the
  other ``eval/*`` metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from monorl.config.schema import DataConfig, RunConfig
from monorl.core.errors import ConfigError


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainStage(_Model):
    type: Literal["train"] = "train"
    name: str = Field(..., description="Stage name (used for the stage directory).")
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="RunConfig fields merged over the pipeline base (e.g. {algorithm: {name: sft}, trainer: {max_steps: 50}}).",
    )
    init_from: Literal["previous", "base"] | str = Field(
        "previous",
        description="Policy initialisation: 'previous' (last train stage's final export), 'base' (base model), or a path.",
    )
    data_from: Literal["previous", "config"] = Field(
        "config",
        description="'previous' uses the dataset written by the preceding rejection_sample stage as data.uri.",
    )


class RejectionSampleStage(_Model):
    type: Literal["rejection_sample"] = "rejection_sample"
    name: str
    data: DataConfig | None = Field(
        None, description="Prompts to sample from (defaults to the base data config)."
    )
    num_samples: int = Field(4, ge=1, description="Completions sampled per prompt.")
    min_reward: float = Field(1.0, description="Keep completions whose total reward is >= this value.")
    max_per_prompt: int = Field(
        1, ge=1, description="Keep at most this many accepted completions per prompt."
    )
    max_response_tokens: int | None = Field(
        None, ge=1, description="Drop accepted completions longer than this."
    )
    require_finish: bool = Field(True, description="Drop completions truncated by max_new_tokens.")
    dedupe: bool = Field(True, description="Drop duplicate responses for the same prompt.")
    mix: list[str] = Field(
        default_factory=list, description="Extra chat/prompt_answer dataset URIs appended to the output."
    )
    rollout_overrides: dict[str, Any] = Field(
        default_factory=dict, description="RolloutConfig overrides for sampling."
    )
    reward_overrides: list[dict[str, Any]] | None = Field(
        None, description="Replace the base rewards for scoring."
    )
    output: str | None = Field(None, description="Output jsonl path (default: <stage dir>/dataset.jsonl).")
    max_prompts: int | None = Field(None, ge=1, description="Sample from at most this many prompts.")


class EvalStage(_Model):
    type: Literal["eval"] = "eval"
    name: str
    data: DataConfig | None = Field(
        None, description="Evaluation dataset (defaults to base data.eval_uri or data)."
    )
    rollout_overrides: dict[str, Any] = Field(default_factory=dict)
    max_prompts: int | None = Field(None, ge=1)
    num_samples: int = Field(
        1,
        ge=1,
        description=(
            "Completions per prompt. 1 evaluates greedily; more samples that many at the rollout "
            "temperature and reports eval/pass_at_k."
        ),
    )


Stage = Annotated[TrainStage | RejectionSampleStage | EvalStage, Field(discriminator="type")]


class PipelineConfig(_Model):
    name: str | None = Field(
        None, description="Pipeline run name (defaults to '<model>-pipeline-<timestamp>')."
    )
    base: RunConfig = Field(..., description="Base run configuration every stage starts from.")
    stages: list[Stage] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _check(self) -> PipelineConfig:
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique, got {names}")
        for i, stage in enumerate(self.stages):
            if isinstance(stage, TrainStage) and stage.data_from == "previous":
                if i == 0 or not isinstance(self.stages[i - 1], RejectionSampleStage):
                    raise ValueError(
                        f"stage {stage.name!r}: data_from='previous' needs a rejection_sample stage right before it"
                    )
        return self

    # -- (de)serialisation ---------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        try:
            return cls.model_validate(data)
        except Exception as e:  # pydantic.ValidationError
            raise ConfigError(str(e)) from e

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        from monorl.config.loader import load_yaml

        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()


def is_pipeline_payload(data: dict[str, Any]) -> bool:
    """A yaml/json document is a pipeline when it has ``stages`` and ``base`` (vs. a RunConfig)."""
    return isinstance(data, dict) and "stages" in data and "base" in data
