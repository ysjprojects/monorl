"""Unified run configuration.

One :class:`RunConfig` describes a run end to end; adapters compile it into framework-native
configs (verl Hydra tree, NeMo-RL yaml, TRL ``GRPOConfig``...). Field ``description`` strings are
surfaced by the web UI (forms are generated from ``RunConfig.model_json_schema()``), so keep them
user-facing. Backend-specific knobs go in the ``extra`` dicts, never as top-level fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from monorl.core.errors import ConfigError


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class LoraConfig(_Model):
    enabled: bool = Field(False, description="Train LoRA adapters instead of full parameters.")
    r: int = Field(16, ge=1, description="LoRA rank.")
    alpha: int = Field(32, ge=1, description="LoRA alpha (scaling = alpha / r).")
    dropout: float = Field(0.05, ge=0.0, le=1.0)
    target_modules: list[str] | Literal["all-linear"] = Field(
        "all-linear", description="Module names to adapt, or 'all-linear'."
    )


class ModelConfig(_Model):
    name_or_path: str = Field(..., description="Hugging Face model id or local path.")
    revision: str | None = Field(None, description="Hub revision (branch, tag or commit).")
    tokenizer_name_or_path: str | None = Field(None, description="Override tokenizer (defaults to the model).")
    trust_remote_code: bool = False
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = Field(
        "auto", description="Parameter dtype; 'auto' picks bf16 on accelerators and fp32 on CPU."
    )
    attn_implementation: str | None = Field(None, description="e.g. 'flash_attention_2', 'sdpa', 'eager'.")
    chat_template: str | None = Field(None, description="Jinja chat template override.")
    lora: LoraConfig = Field(default_factory=LoraConfig)


class DataConfig(_Model):
    uri: str = Field(
        ...,
        description=(
            "Dataset location: 'hf://org/name' (Hub), 's3://bucket/prefix-or-file' (credentials from "
            "AWS_* env vars), or a local file/directory (.jsonl/.json/.parquet/.csv/.arrow)."
        ),
    )
    split: str = Field("train", description="Split name for Hub datasets (ignored for files).")
    hf_config_name: str | None = Field(None, description="Hub dataset config/subset name.")
    format: Literal["auto", "chat", "prompt_answer", "text"] = Field(
        "auto",
        description=(
            "'chat': a messages column; 'prompt_answer': prompt + reference answer columns; "
            "'text': a single prompt column; 'auto' sniffs columns."
        ),
    )
    messages_field: str | None = Field(None, description="Column holding chat messages (format=chat).")
    prompt_field: str | None = Field(None, description="Column holding the prompt/question.")
    answer_field: str | None = Field(None, description="Column holding the reference answer (optional).")
    image_field: str | list[str] | None = Field(
        None,
        description=(
            "Column(s) holding images (HF Image feature, file path/URI, bytes or base64) attached to the "
            "prompt as image content parts. Use '<image>' in prompt_template to position them."
        ),
    )
    response_field: str | None = Field(
        None,
        description="Column holding the target response for SFT (defaults to answer_field / the trailing assistant turn).",
    )
    system_prompt: str | None = Field(None, description="System prompt prepended to every sample.")
    prompt_template: str | None = Field(
        None, description="Python format string applied to the prompt column, e.g. '{prompt}\\nThink step by step.'"
    )
    max_prompt_tokens: int = Field(1024, ge=1, description="Samples with longer prompts are dropped.")
    limit: int | None = Field(None, ge=1, description="Use only the first N samples (after shuffling).")
    shuffle: bool = True
    streaming: bool = Field(False, description="Stream from Hub/S3 instead of materialising locally.")
    eval_uri: str | None = Field(None, description="Optional evaluation dataset (same format).")
    eval_split: str = "test"
    extra: dict[str, Any] = Field(default_factory=dict, description="Loader-specific options.")


class AlgorithmConfig(_Model):
    name: Literal["grpo", "dapo", "rloo", "reinforce", "ppo", "sft"] = Field(
        "grpo",
        description="Policy-gradient objective, or 'sft' for supervised fine-tuning on the dataset responses (no rollouts).",
    )
    sft_label_smoothing: float = Field(0.0, ge=0.0, lt=1.0, description="Label smoothing for sft.")
    sft_train_on_prompt: bool = Field(False, description="sft: include prompt tokens in the loss (default: response only).")
    kl_coef: float = Field(0.0, ge=0.0, description="Coefficient of the KL penalty to the reference policy (0 disables).")
    kl_estimator: Literal["k1", "k2", "k3"] = Field("k3", description="KL estimator (Schulman's k1/k2/k3).")
    clip_eps: float = Field(0.2, gt=0.0, description="PPO clipping epsilon (lower bound).")
    clip_eps_high: float | None = Field(None, gt=0.0, description="Asymmetric upper clip (DAPO clip-higher).")
    advantage_norm: Literal["group", "batch", "none"] = Field(
        "group", description="Normalise rewards within the prompt group (GRPO), across the batch, or not at all."
    )
    advantage_std_norm: bool = Field(True, description="Divide by std in group normalisation (False = Dr.GRPO style).")
    loss_agg: Literal["token", "sequence"] = Field(
        "token", description="Average the loss over all response tokens or per sequence first."
    )
    entropy_coef: float = Field(0.0, ge=0.0)
    num_epochs: int = Field(1, ge=1, description="Optimizer passes over each rollout batch.")
    dynamic_sampling: bool = Field(False, description="Drop groups whose rewards are all identical (DAPO).")
    overlong_penalty: float = Field(0.0, ge=0.0, description="Reward penalty for truncated (length) completions.")
    gamma: float = Field(1.0, ge=0.0, le=1.0, description="Discount for multi-turn per-turn rewards.")
    lam: float = Field(1.0, ge=0.0, le=1.0, description="GAE lambda (ppo only).")
    value_coef: float = Field(0.5, ge=0.0, description="Value loss weight (ppo only).")


class RolloutConfig(_Model):
    backend: str = Field("hf", description="Rollout engine: 'hf' (transformers), 'vllm', 'sglang'.")
    group_size: int = Field(4, ge=1, description="Completions sampled per prompt (G in GRPO).")
    max_new_tokens: int = Field(256, ge=1)
    temperature: float = Field(1.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = Field(-1, description="-1 disables top-k.")
    stop: list[str] = Field(default_factory=list)
    tensor_parallel_size: int = Field(1, ge=1)
    data_parallel_size: int = Field(1, ge=1, description="Number of independent engine replicas.")
    gpu_memory_utilization: float = Field(0.6, gt=0.0, le=1.0)
    max_model_len: int | None = None
    enable_sleep_mode: bool = Field(True, description="Offload engine memory while training (colocated).")
    async_steps_ahead: int = Field(
        0, ge=0, description="Generate up to N batches ahead of training (off-policy staleness bound). 0 = synchronous."
    )
    device: str | None = Field(None, description="Device for the 'hf' backend (cpu/mps/cuda); auto if unset.")
    extra: dict[str, Any] = Field(default_factory=dict, description="Engine-specific options passed through.")


class TrainerConfig(_Model):
    backend: str = Field(
        "reference",
        description="Trainer: 'reference' (torch), 'unsloth', or a pipeline backend: 'trl', 'verl', 'nemo_rl'.",
    )
    learning_rate: float = Field(1e-6, gt=0.0)
    prompts_per_step: int = Field(8, ge=1, description="Prompts per optimizer step (x group_size = sequences).")
    micro_batch_size: int = Field(4, ge=1, description="Sequences per forward/backward pass.")
    max_steps: int = Field(100, ge=1)
    warmup_steps: int = Field(0, ge=0)
    weight_decay: float = Field(0.0, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    precision: Literal["fp32", "bf16", "fp16"] = Field("bf16", description="Mixed-precision mode (fp32 forced on CPU).")
    gradient_checkpointing: bool = False
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    scheduler: Literal["constant", "linear", "cosine"] = "constant"
    save_every: int = Field(50, ge=1, description="Checkpoint interval in steps.")
    eval_every: int | None = Field(None, ge=1)
    max_seq_len: int = Field(2048, ge=16, description="Prompt + response truncation for training.")
    device: str | None = Field(None, description="cpu/mps/cuda; auto if unset.")
    extra: dict[str, Any] = Field(default_factory=dict, description="Backend-specific options (e.g. verl/nemo overrides).")


class RewardSpec(_Model):
    type: str = Field(..., description="Reward function name (see /api/system for the list).")
    weight: float = Field(1.0, description="Multiplier when combining several rewards.")
    params: dict[str, Any] = Field(default_factory=dict)


class EnvConfig(_Model):
    type: str = Field("single_turn", description="Environment: 'single_turn', 'tool_calling', 'openenv', 'hermes', ...")
    max_turns: int = Field(1, ge=1)
    params: dict[str, Any] = Field(default_factory=dict, description="Environment-specific parameters.")


class ClusterConfig(_Model):
    mode: Literal["local", "ray"] = Field("local", description="Run in-process or on a Ray cluster.")
    ray_address: str | None = Field(None, description="Ray address ('auto' to join an existing cluster).")
    num_nodes: int = Field(1, ge=1)
    gpus_per_node: int = Field(0, ge=0)
    placement: Literal["auto", "colocated", "disaggregated"] = Field(
        "auto",
        description="Share GPUs between rollout and training (colocated) or split them (disaggregated).",
    )
    rollout_gpus: int | None = Field(None, ge=0, description="GPUs for inference when disaggregated.")
    trainer_gpus: int | None = Field(None, ge=0, description="GPUs for training when disaggregated.")
    weight_sync: str = Field(
        "auto", description="Weight sync strategy: 'auto', 'inprocess', 'checkpoint', 'broadcast', 'cuda_ipc'."
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(_Model):
    run_root: str = Field("~/.monorl/runs", description="Directory that holds run directories.")
    log_every: int = Field(1, ge=1)
    save_trajectories: bool = Field(True, description="Write sampled rollouts to trajectories.jsonl.")
    trajectories_per_step: int = Field(8, ge=0)
    wandb_project: str | None = None
    tensorboard: bool = False


class RunConfig(_Model):
    run_name: str | None = Field(None, description="Defaults to '<model>-<algorithm>-<timestamp>'.")
    seed: int = 42
    model: ModelConfig
    data: DataConfig
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    rewards: list[RewardSpec] = Field(default_factory=list, description="Combined as a weighted sum.")
    env: EnvConfig = Field(default_factory=EnvConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        if self.algorithm.name == "sft":
            if self.env.type != "single_turn":
                raise ValueError("sft trains on dataset responses; env.type must be 'single_turn'")
            return self
        if self.algorithm.name == "ppo" and self.trainer.backend not in PIPELINE_BACKENDS:
            raise ValueError(
                "algorithm 'ppo' needs a critic (GAE/value loss), which only the verl pipeline backend provides; "
                "use grpo/dapo/rloo/reinforce with trainer.backend='reference'"
            )
        if self.algorithm.name in {"grpo", "dapo", "rloo"} and self.rollout.group_size < 2:
            raise ValueError(f"{self.algorithm.name} needs rollout.group_size >= 2")
        if self.algorithm.advantage_norm == "group" and self.rollout.group_size < 2:
            raise ValueError("advantage_norm='group' needs rollout.group_size >= 2")
        if self.cluster.placement == "disaggregated" and self.cluster.gpus_per_node > 0:
            r, t = self.cluster.rollout_gpus, self.cluster.trainer_gpus
            total = self.cluster.num_nodes * self.cluster.gpus_per_node
            if r is None or t is None:
                raise ValueError("disaggregated placement needs rollout_gpus and trainer_gpus")
            if r + t > total:
                raise ValueError(f"rollout_gpus + trainer_gpus = {r + t} exceeds {total} GPUs")
        if not self.rewards and self.env.type == "single_turn":
            raise ValueError("single_turn env needs at least one reward in `rewards`")
        return self

    @property
    def is_sft(self) -> bool:
        return self.algorithm.name == "sft"

    # -- (de)serialisation ---------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        try:
            return cls.model_validate(data)
        except Exception as e:  # pydantic.ValidationError
            raise ConfigError(str(e)) from e

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
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

    @property
    def is_pipeline_backend(self) -> bool:
        return self.trainer.backend in PIPELINE_BACKENDS

    def sequences_per_step(self) -> int:
        return self.trainer.prompts_per_step * self.rollout.group_size


PIPELINE_BACKENDS: frozenset[str] = frozenset({"trl", "verl", "nemo_rl"})
