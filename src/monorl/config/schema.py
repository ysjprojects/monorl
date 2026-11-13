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


class MtpConfig(_Model):
    """Multi-token prediction layers of the model (``docs/MTP.md``)."""

    enabled: bool = Field(
        False,
        description=(
            "Load the checkpoint's MTP (multi-token prediction) layers next to the policy and keep them in "
            "weight sync, checkpoints and the final export. Needs a model that ships MTP layers."
        ),
    )
    rollout: bool = Field(
        False,
        description=(
            "Draft tokens with the MTP layers during rollouts (speculative decoding). The policy verifies "
            "every draft, so the sampled tokens and their logprobs are the policy's; only speed changes."
        ),
    )
    num_speculative_tokens: int | None = Field(
        None,
        ge=1,
        description=(
            "Tokens drafted per decoding step. Default: one per MTP layer on the hf engine, 1 on vLLM, "
            "SGLang's own default."
        ),
    )
    train: bool = Field(
        False,
        description=(
            "Train the MTP layers with an auxiliary loss (predict the tokens after the next one) on the "
            "response tokens of every batch, in RL and SFT, so drafts keep matching a policy that changes."
        ),
    )
    loss_coef: float = Field(0.1, ge=0.0, description="Weight of the MTP loss in the total loss.")
    detach_trunk: bool = Field(
        True,
        description=(
            "true: the MTP loss updates the MTP layers only. false: it also flows into the policy's own "
            "parameters, as in pretraining."
        ),
    )

    @model_validator(mode="after")
    def _check(self) -> MtpConfig:
        if not self.enabled and (self.rollout or self.train):
            raise ValueError("model.mtp.rollout and model.mtp.train need model.mtp.enabled: true")
        return self


class ModelConfig(_Model):
    name_or_path: str = Field(..., description="Hugging Face model id or local path.")
    revision: str | None = Field(None, description="Hub revision (branch, tag or commit).")
    tokenizer_name_or_path: str | None = Field(
        None, description="Override tokenizer (defaults to the model)."
    )
    trust_remote_code: bool = False
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = Field(
        "auto", description="Parameter dtype; 'auto' picks bf16 on accelerators and fp32 on CPU."
    )
    attn_implementation: str | None = Field(None, description="e.g. 'flash_attention_2', 'sdpa', 'eager'.")
    chat_template: str | None = Field(None, description="Jinja chat template override.")
    lora: LoraConfig = Field(default_factory=LoraConfig)
    mtp: MtpConfig = Field(default_factory=MtpConfig)


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
    tools_field: str | None = Field(
        None,
        description=(
            "Column holding the tool schemas (a list of OpenAI-style function schemas, or its JSON string) a "
            "chat row's prompts are rendered with. Default: a 'tools' column of a chat dataset when it holds one."
        ),
    )
    system_prompt: str | None = Field(None, description="System prompt prepended to every sample.")
    prompt_template: str | None = Field(
        None,
        description="Python format string applied to the prompt column, e.g. '{prompt}\\nThink step by step.'",
    )
    max_prompt_tokens: int = Field(1024, ge=1, description="Samples with longer prompts are dropped.")
    limit: int | None = Field(None, ge=1, description="Use only the first N samples (after shuffling).")
    shuffle: bool = True
    streaming: bool = Field(False, description="Stream from Hub/S3 instead of materialising locally.")
    eval_uri: str | None = Field(None, description="Optional evaluation dataset (same format).")
    eval_split: str = "test"
    extra: dict[str, Any] = Field(default_factory=dict, description="Loader-specific options.")


class OffPolicyConfig(_Model):
    """Corrections for batches sampled by another policy than the one being updated (stale rollouts
    under ``rollout.async_steps_ahead``, engines whose weights or numerics lag): docs/OFF_POLICY.md."""

    anchor: Literal["auto", "behavior", "trainer"] = Field(
        "auto",
        description=(
            "Proximal policy the clipped ratio is anchored on. 'behavior': the policy that sampled the "
            "tokens (the engine's log-probs). 'trainer': the trainer's policy when the update on the batch "
            "starts (decoupled PPO; one extra forward pass per batch). 'auto': 'trainer' when "
            "importance_weights or ratio_mask is set, otherwise the backend's default."
        ),
    )
    importance_weights: Literal["none", "token", "sequence"] = Field(
        "none",
        description=(
            "Weigh the policy loss by the truncated importance ratio between the proximal and the behaviour "
            "policy, per token or per training sequence (product of its token ratios)."
        ),
    )
    importance_cap: float | None = Field(
        2.0, ge=1.0, description="Upper bound of the importance weights (null = untruncated)."
    )
    ratio_mask: Literal["none", "token", "sequence", "sequence_mean"] = Field(
        "none",
        description=(
            "Drop from the policy loss the tokens, or whole training sequences, whose proximal/behaviour "
            "importance ratio is outside [ratio_mask_low, ratio_mask_high]. 'sequence' tests the product "
            "of a sequence's token ratios, 'sequence_mean' their geometric mean."
        ),
    )
    ratio_mask_low: float = Field(
        0.5, gt=0.0, le=1.0, description="Lower bound of the accepted importance ratio."
    )
    ratio_mask_high: float = Field(5.0, ge=1.0, description="Upper bound of the accepted importance ratio.")
    score_centering: bool = Field(
        False,
        description=(
            "Subtract the expected score under the sampling policy from the policy gradient "
            "(arXiv:2609.20807). Needs the engine's top-k next-token log-probs and the 'behavior' anchor."
        ),
    )
    score_centering_top_k: int = Field(
        128, ge=1, description="Number of most likely next tokens the engine records per sampled token."
    )

    @model_validator(mode="after")
    def _check(self) -> OffPolicyConfig:
        if self.corrects and self.anchor == "behavior":
            raise ValueError(
                "off_policy.importance_weights / off_policy.ratio_mask compare the proximal with the behaviour "
                "policy, which are the same policy under off_policy.anchor='behavior'; use anchor 'trainer'"
            )
        if self.score_centering and self.decoupled:
            raise ValueError(
                "off_policy.score_centering weighs scores by the ratio of the current to the behaviour policy "
                "and needs off_policy.anchor='behavior' (or 'auto' without importance_weights / ratio_mask)"
            )
        return self

    @property
    def corrects(self) -> bool:
        """Importance weights or a ratio mask are applied to the policy loss."""
        return self.importance_weights != "none" or self.ratio_mask != "none"

    @property
    def decoupled(self) -> bool:
        """The proximal policy is the trainer's, separate from the behaviour policy."""
        return self.anchor == "trainer" or (self.anchor == "auto" and self.corrects)


class AlgorithmConfig(_Model):
    name: Literal["grpo", "dapo", "rloo", "reinforce", "ppo", "sft"] = Field(
        "grpo",
        description="Policy-gradient objective, or 'sft' for supervised fine-tuning on the dataset responses (no rollouts).",
    )
    sft_label_smoothing: float = Field(0.0, ge=0.0, lt=1.0, description="Label smoothing for sft.")
    sft_train_on_prompt: bool = Field(
        False, description="sft: include prompt tokens in the loss (default: response only)."
    )
    sft_turns: Literal["last", "all"] = Field(
        "last",
        description=(
            "sft: assistant turns of a conversation that carry the loss. 'last': only the final one, earlier "
            "turns are context. 'all': every assistant turn (multi-turn and tool-using conversations), except "
            "those a row marks with 'weight': 0."
        ),
    )
    kl_coef: float = Field(
        0.0, ge=0.0, description="Coefficient of the KL penalty to the reference policy (0 disables)."
    )
    kl_estimator: Literal["k1", "k2", "k3"] = Field("k3", description="KL estimator (Schulman's k1/k2/k3).")
    clip_eps: float = Field(0.2, gt=0.0, description="PPO clipping epsilon (lower bound).")
    clip_eps_high: float | None = Field(None, gt=0.0, description="Asymmetric upper clip (DAPO clip-higher).")
    advantage_norm: Literal["group", "batch", "none"] = Field(
        "group",
        description="Normalise rewards within the prompt group (GRPO), across the batch, or not at all.",
    )
    advantage_std_norm: bool = Field(
        True, description="Divide by std in group normalisation (False = Dr.GRPO style)."
    )
    loss_agg: Literal["token", "sequence", "rollout"] = Field(
        "token",
        description=(
            "Average the loss over all response tokens, per training sequence first, or per rollout "
            "first ('rollout': every episode counts equally however many sequences its turns form)."
        ),
    )
    entropy_coef: float = Field(0.0, ge=0.0)
    num_epochs: int = Field(1, ge=1, description="Optimizer passes over each rollout batch.")
    dynamic_sampling: bool = Field(False, description="Drop groups whose rewards are all identical (DAPO).")
    overlong_penalty: float = Field(
        0.0, ge=0.0, description="Reward penalty for truncated (length) completions."
    )
    credit_assignment: Literal["episode", "turn"] = Field(
        "episode",
        description=(
            "Which rewards a response token is credited with. 'episode': the reward of its whole episode, "
            "the same advantage for every token. 'turn': the return from its own turn on (later per-turn "
            "rewards discounted by gamma; the reward-function score arrives at the final turn), compared "
            "with the same turn of the other episodes of its prompt group. See docs/CREDIT_ASSIGNMENT.md."
        ),
    )
    gamma: float = Field(1.0, ge=0.0, le=1.0, description="Discount for multi-turn per-turn rewards.")
    lam: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="GAE lambda (ppo only): 1 = returns minus the value baseline, 0 = one-step TD error.",
    )
    value_coef: float = Field(0.5, ge=0.0, description="Value loss weight (ppo only).")
    value_clip: float | None = Field(
        0.5,
        gt=0.0,
        description=(
            "ppo only: the value loss also penalises a value estimate clipped to this distance from the "
            "estimate the advantages were computed with (null = plain squared error)."
        ),
    )
    off_policy: OffPolicyConfig = Field(
        default_factory=OffPolicyConfig, description="Off-policy corrections (off by default)."
    )


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
    max_context_tokens: int | None = Field(
        None,
        ge=2,
        description=(
            "Context budget of an episode in tokens: no prompt plus completion of a turn may exceed it. A "
            "turn's generation is capped to the room that is left, an episode without room for another "
            "turn ends, and an agent behind the gateway gets a context_length_exceeded error. Default: "
            "trainer.max_seq_len (and rollout.max_model_len when set), so that nothing an episode "
            "samples is cut off in training."
        ),
    )
    enable_sleep_mode: bool = Field(True, description="Offload engine memory while training (colocated).")
    async_steps_ahead: int = Field(
        0,
        ge=0,
        description="Generate up to N batches ahead of training (off-policy staleness bound). 0 = synchronous.",
    )
    turn_rendering: Literal["tokens", "template"] = Field(
        "tokens",
        description=(
            "How a multi-turn prompt is built. 'tokens': the sampled tokens of earlier turns stay in the "
            "context and only new messages are rendered. 'template': the chat template re-renders the "
            "whole conversation every turn."
        ),
    )
    device: str | None = Field(None, description="Device for the 'hf' backend (cpu/mps/cuda); auto if unset.")
    extra: dict[str, Any] = Field(default_factory=dict, description="Engine-specific options passed through.")


class TrainerConfig(_Model):
    backend: str = Field(
        "reference",
        description="Trainer: 'reference' (torch), 'unsloth', or a pipeline backend: 'trl', 'verl', 'nemo_rl'.",
    )
    learning_rate: float = Field(1e-6, gt=0.0)
    prompts_per_step: int = Field(
        8, ge=1, description="Prompts per optimizer step (x group_size = sequences)."
    )
    micro_batch_size: int = Field(4, ge=1, description="Sequences per forward/backward pass.")
    max_steps: int = Field(100, ge=1)
    warmup_steps: int = Field(0, ge=0)
    weight_decay: float = Field(0.0, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    precision: Literal["fp32", "bf16", "fp16"] = Field(
        "bf16", description="Mixed-precision mode (fp32 forced on CPU)."
    )
    gradient_checkpointing: bool = False
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    scheduler: Literal["constant", "linear", "cosine"] = "constant"
    save_every: int = Field(50, ge=1, description="Checkpoint interval in steps.")
    eval_every: int | None = Field(None, ge=1)
    max_seq_len: int = Field(2048, ge=16, description="Prompt + response truncation for training.")
    device: str | None = Field(None, description="cpu/mps/cuda; auto if unset.")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Backend-specific options (e.g. verl/nemo overrides)."
    )


class RewardSpec(_Model):
    type: str = Field(..., description="Reward function name (see /api/system for the list).")
    weight: float = Field(1.0, description="Multiplier when combining several rewards.")
    params: dict[str, Any] = Field(default_factory=dict)


class EnvConfig(_Model):
    type: str = Field(
        "single_turn", description="Environment: 'single_turn', 'tool_calling', 'openenv', 'hermes', ..."
    )
    max_turns: int = Field(1, ge=1)
    retries: int = Field(
        0,
        ge=0,
        description="Times an episode whose environment failed (a crashed agent, a failing step) is started again.",
    )
    max_failed_frac: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Share of a rollout batch's episodes that may fail for good; they are left out of the batch. "
            "Above it the run fails. 0 (default): any failed episode fails the run."
        ),
    )
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
        "auto",
        description="Weight sync strategy: 'auto', 'inprocess', 'checkpoint', 'broadcast', 'cuda_ipc'.",
    )
    extra: dict[str, Any] = Field(default_factory=dict)


class LoggingConfig(_Model):
    run_root: str = Field("~/.monorl/runs", description="Directory that holds run directories.")
    log_every: int = Field(1, ge=1)
    save_trajectories: bool = Field(True, description="Write sampled rollouts to trajectories.jsonl.")
    trajectories_per_step: int = Field(8, ge=0)
    save_transcripts: bool = Field(
        True,
        description="Include the full conversation and per-turn details of every saved rollout.",
    )
    success_threshold: float = Field(
        1.0,
        description="Total reward at or above which an episode counts as solved (success-rate and pass@k metrics).",
    )
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
        budget = self.rollout.max_context_tokens
        if budget is not None and budget > self.trainer.max_seq_len:
            raise ValueError(
                f"rollout.max_context_tokens={budget} exceeds trainer.max_seq_len={self.trainer.max_seq_len}: "
                "training would cut such episodes off; raise trainer.max_seq_len or lower the budget"
            )
        if self.trainer.backend in PIPELINE_BACKENDS:
            # TRL takes one result per requested rollout; verl and NeMo-RL run their own rollouts.
            if self.env.max_failed_frac > 0:
                raise ValueError(
                    f"env.max_failed_frac is not supported by trainer.backend={self.trainer.backend!r}: failed "
                    "episodes can only be left out of a batch by the native loop ('reference', 'unsloth')"
                )
            if self.trainer.backend != "trl" and (self.env.retries > 0 or budget is not None):
                raise ValueError(
                    f"env.retries and rollout.max_context_tokens are not honoured by trainer.backend="
                    f"{self.trainer.backend!r}, which runs its own rollouts; use its own length limits"
                )
        off, ro = self.algorithm.off_policy, self.rollout
        if off.score_centering and (
            ro.temperature <= 0.0
            or ro.top_k > 0
            or ro.top_p < 1.0
            or float(ro.extra.get("min_p") or 0.0) > 0.0
            or float(ro.extra.get("repetition_penalty") or 1.0) != 1.0
        ):
            raise ValueError(
                "off_policy.score_centering needs the full sampling distribution: rollout.temperature > 0 and "
                "no top_k / top_p / min_p / repetition_penalty truncation"
            )
        return self

    @model_validator(mode="after")
    def _check_mtp(self) -> RunConfig:
        mtp, backend = self.model.mtp, self.trainer.backend
        if not mtp.enabled:
            return self
        if backend == "unsloth":
            raise ValueError(
                "model.mtp is not supported with trainer.backend='unsloth': Unsloth loads the model itself and "
                "trains MTP heads with a trainer of its own; use trainer.backend='reference'"
            )
        if backend == "trl":
            raise ValueError(
                "model.mtp is not supported with trainer.backend='trl': TRL neither trains MTP layers nor "
                "drafts with them in its own generation; use trainer.backend='reference'"
            )
        if mtp.train and backend == "verl":
            raise ValueError(
                "model.mtp.train is not supported with trainer.backend='verl': verl trains MTP layers only "
                "with its Megatron engine, and monorl compiles the FSDP actor, which ignores the setting; "
                "model.mtp.rollout is compiled"
            )
        # verl and NeMo-RL run their own rollouts on vLLM / SGLang, whatever rollout.backend says
        served = self.rollout.backend in {"vllm", "sglang"} or backend in {"verl", "nemo_rl"}
        if mtp.rollout and served and float(self.rollout.extra.get("min_p") or 0.0) > 0.0:
            engine = self.rollout.backend if self.rollout.backend in {"vllm", "sglang"} else backend
            raise ValueError(
                f"model.mtp.rollout on {engine} cannot be combined with min_p: vLLM rejects it under "
                "speculative decoding and SGLang's verification does not apply it, so the sampled "
                "distribution would differ from the recorded logprobs"
            )
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

    @property
    def context_budget(self) -> int:
        """Tokens the context of an episode may hold: ``rollout.max_context_tokens``, else what
        both the trainer (``trainer.max_seq_len``) and the engine (``rollout.max_model_len``)
        accept."""
        ro = self.rollout
        if ro.max_context_tokens is not None:
            return ro.max_context_tokens
        return min(self.trainer.max_seq_len, ro.max_model_len or self.trainer.max_seq_len)

    def sequences_per_step(self) -> int:
        return self.trainer.prompts_per_step * self.rollout.group_size


PIPELINE_BACKENDS: frozenset[str] = frozenset({"trl", "verl", "nemo_rl"})
