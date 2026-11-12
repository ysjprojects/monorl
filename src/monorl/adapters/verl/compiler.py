"""Compile a :class:`~monorl.config.schema.RunConfig` into verl's Hydra config tree.

Targets verl **v0.9.1** (tag ``v0.9.1``, commit ``1876b06d0a3e4e71e06230be10af14492ca8a75b``). Every key
below exists in that tag's config tree unless prefixed with ``+`` (Hydra "add" override):

* ``verl/trainer/config/ppo_trainer.yaml``          algorithm.*, trainer.*
* ``verl/trainer/config/data/legacy_data.yaml``       data.*
* ``verl/trainer/config/actor/actor.yaml``            actor_rollout_ref.actor.* (+ dp_actor.yaml)
* ``verl/trainer/config/rollout/rollout.yaml``        actor_rollout_ref.rollout.*
* ``verl/trainer/config/model/hf_model.yaml``         actor_rollout_ref.model.*
* ``verl/trainer/config/reward/reward.yaml``          reward.custom_reward_function.*, reward.reward_manager.*
* ``verl/trainer/ppo/core_algos.py``                  ``AdvantageEstimator`` names
* ``verl/trainer/config/sft_trainer_engine.yaml``     data.*, trainer.*, checkpoint.* (SFT, with
  ``model@model: hf_model``, ``engine@engine: fsdp``, ``optim@optim: fsdp`` defaults) — v0.9.1
  replaced ``fsdp_sft_trainer`` with the engine-based ``verl/trainer/sft_trainer.py`` /
  ``sft_trainer_ray.py`` (``MultiTurnSFTDataset``: ``data.messages_key``, ``data.image_key``).

The result is a nested dict; :func:`to_overrides` flattens it into ``key=value`` strings for
``python -m verl.trainer.main_ppo`` (Hydra CLI grammar). ``trainer.extra`` is deep-merged last so
any verl knob can be set or overridden from the RunConfig. Multimodal runs set ``data.image_key``
and ``actor_rollout_ref.model.use_remove_padding`` as ``examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh`` does.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from monorl.adapters.verl.data import (
    IMAGES_KEY,
    MESSAGES_KEY,
    PROMPT_KEY,
    TOOL_AGENT,
    sft_train_parquet_path,
    sft_val_parquet_path,
    train_parquet_path,
    val_parquet_path,
)
from monorl.adapters.verl.reward import REWARD_FUNCTION_NAME, reward_module_path
from monorl.adapters.verl.tools import tool_config_path
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.reporting import CHECKPOINT_DIR

VERL_TAG = "v0.9.1"
VERL_COMMIT = "1876b06d0a3e4e71e06230be10af14492ca8a75b"
VERL_VERSIONS = ">=0.9,<0.10"

PROJECT_NAME = "monorl"
# ``trainer.extra`` keys consumed by monorl itself (never forwarded to verl).
RESERVED_EXTRA_KEYS = frozenset({"verl_python"})

# monorl AlgorithmConfig.kl_estimator -> actor.kl_loss_type (verl/trainer/ppo/core_algos.py kl_penalty)
KL_LOSS_TYPE = {"k1": "kl", "k2": "mse", "k3": "low_var_kl"}
# monorl loss_agg -> actor.loss_agg_mode (verl/trainer/ppo/core_algos.py agg_loss)
LOSS_AGG_MODE = {"token": "token-mean", "sequence": "seq-mean-token-mean"}
# monorl model.dtype -> rollout.dtype (rollout.yaml); "auto" keeps verl's bf16 default
ROLLOUT_DTYPE = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}
ROLLOUT_BACKENDS = ("vllm", "sglang")
DEFAULT_ROLLOUT_BACKEND = "vllm"
# Environments verl can run: its ``single_turn_agent`` loop and, for ``tool_calling``, its ``tool_agent``
# loop over monorl tools (:mod:`.tools`). verl never steps any other monorl environment.
ENV_TYPES = ("single_turn", "tool_calling")
# DAPO overlong buffer as a fraction of max_response_length (recipe/dapo uses 4096 of 20480).
OVERLONG_BUFFER_FRACTION = 0.2
DAPO_GEN_BATCH_MULTIPLIER = 2
DAPO_MAX_NUM_GEN_BATCHES = 10


def adv_estimator(config: RunConfig) -> str:
    """``algorithm.adv_estimator`` for the run (``AdvantageEstimator`` in core_algos.py)."""
    algo = config.algorithm
    if algo.name in ("grpo", "dapo"):
        return "grpo"
    if algo.name == "rloo":
        return "rloo"
    if algo.name == "ppo":
        return "gae"
    # reinforce: verl's REINFORCE++ normalises across the batch; the *_baseline variant subtracts
    # the per-prompt mean first, which is what monorl's group normalisation means.
    return "reinforce_plus_plus_baseline" if algo.advantage_norm == "group" else "reinforce_plus_plus"


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into ``base`` (in place); dict leaves merge, others overwrite."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def compile_config(
    config: RunConfig, run_dir: str | Path, *, multimodal: bool | None = None
) -> dict[str, Any]:
    """Build the verl override tree for ``config``; paths point into ``run_dir``.

    Does not touch the filesystem: the data/reward/tool artefacts referenced here are written by
    :mod:`monorl.adapters.verl.data`, :mod:`.reward` and :mod:`.tools` (``VerlBackend.prepare``).
    ``multimodal`` (default: ``data.image_field`` is set) adds the VLM keys; ``algorithm.name='sft'``
    dispatches to :func:`compile_sft_config`.
    """
    if config.is_sft:
        return compile_sft_config(config, run_dir, multimodal=multimodal)
    if multimodal is None:
        multimodal = config.data.image_field is not None
    run_dir = Path(run_dir).expanduser().resolve()
    algo, rollout, trainer, model, cluster, env, data, logging = (
        config.algorithm,
        config.rollout,
        config.trainer,
        config.model,
        config.cluster,
        config.env,
        config.data,
        config.logging,
    )
    if env.type not in ENV_TYPES:
        raise ConfigError(
            f"trainer.backend='verl' runs env.type in {list(ENV_TYPES)} only, got {env.type!r}: verl would "
            f"sample single-turn completions and never step that environment; use trainer.backend "
            f"'reference', 'trl' or 'nemo_rl'"
        )
    tool_calling = env.type == "tool_calling"
    n_gpus = max(1, cluster.gpus_per_node)
    use_kl_loss = algo.kl_coef > 0 and algo.name != "ppo"
    clip_high = algo.clip_eps_high if algo.clip_eps_high is not None else algo.clip_eps
    rollout_name = rollout.backend if rollout.backend in ROLLOUT_BACKENDS else DEFAULT_ROLLOUT_BACKEND
    loggers = ["console"]
    if logging.wandb_project:
        loggers.append("wandb")
    if logging.tensorboard:
        loggers.append("tensorboard")
    train_files = str(train_parquet_path(run_dir))
    val_files = str(val_parquet_path(run_dir)) if data.eval_uri else train_files

    tree: dict[str, Any] = {
        "data": {
            "train_files": train_files,
            "val_files": val_files,
            "prompt_key": PROMPT_KEY,
            "reward_fn_key": "data_source",
            "max_prompt_length": data.max_prompt_tokens,
            "max_response_length": rollout.max_new_tokens,
            "train_batch_size": trainer.prompts_per_step,
            "shuffle": data.shuffle,
            "seed": config.seed,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "return_raw_chat": True,
            "trust_remote_code": model.trust_remote_code,
        },
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "model": {
                "path": model.name_or_path,
                "tokenizer_path": model.tokenizer_name_or_path,
                "trust_remote_code": model.trust_remote_code,
                "custom_chat_template": model.chat_template,
                "enable_gradient_checkpointing": trainer.gradient_checkpointing,
                "lora_rank": model.lora.r if model.lora.enabled else 0,
                "lora_alpha": model.lora.alpha if model.lora.enabled else 16,
                "target_modules": model.lora.target_modules if model.lora.enabled else "all-linear",
            },
            "actor": {
                "optim": {
                    "lr": trainer.learning_rate,
                    "weight_decay": trainer.weight_decay,
                    "lr_warmup_steps": trainer.warmup_steps if trainer.warmup_steps > 0 else -1,
                    "total_training_steps": trainer.max_steps,
                },
                "grad_clip": trainer.max_grad_norm,
                "ppo_mini_batch_size": trainer.prompts_per_step,
                "ppo_micro_batch_size_per_gpu": trainer.micro_batch_size,
                "ppo_epochs": algo.num_epochs,
                "use_kl_loss": use_kl_loss,
                "kl_loss_coef": algo.kl_coef,
                "kl_loss_type": KL_LOSS_TYPE[algo.kl_estimator],
                "clip_ratio": algo.clip_eps,
                "clip_ratio_low": algo.clip_eps,
                "clip_ratio_high": clip_high,
                "loss_agg_mode": LOSS_AGG_MODE[algo.loss_agg],
                "entropy_coeff": algo.entropy_coef,
                "calculate_entropy": algo.entropy_coef > 0,
                "data_loader_seed": config.seed,
            },
            "rollout": {
                "name": rollout_name,
                "mode": "async",
                "n": rollout.group_size,
                "tensor_model_parallel_size": rollout.tensor_parallel_size,
                "data_parallel_size": rollout.data_parallel_size,
                "gpu_memory_utilization": rollout.gpu_memory_utilization,
                "temperature": rollout.temperature,
                "top_p": rollout.top_p,
                "top_k": rollout.top_k,
                "max_model_len": rollout.max_model_len,
                "free_cache_engine": rollout.enable_sleep_mode,
                "seed": config.seed,
                "calculate_log_probs": True,
                "multi_turn": {"enable": tool_calling},
                "agent": {"default_agent_loop": TOOL_AGENT if tool_calling else "single_turn_agent"},
            },
        },
        "algorithm": {
            "adv_estimator": adv_estimator(config),
            "gamma": algo.gamma,
            "lam": algo.lam,
            "norm_adv_by_std_in_grpo": algo.advantage_std_norm,
            "use_kl_in_reward": False,
            "filter_groups": {"enable": False},
        },
        "reward": {
            "custom_reward_function": {
                "path": str(reward_module_path(run_dir)),
                "name": REWARD_FUNCTION_NAME,
            },
        },
        "trainer": {
            "nnodes": cluster.num_nodes,
            "n_gpus_per_node": n_gpus,
            "total_training_steps": trainer.max_steps,
            "save_freq": trainer.save_every,
            "test_freq": trainer.eval_every if (trainer.eval_every and data.eval_uri) else -1,
            "val_before_train": False,
            "project_name": PROJECT_NAME,
            "experiment_name": config.run_name or run_dir.name,
            "logger": loggers,
            "default_local_dir": str(run_dir / CHECKPOINT_DIR),
            "resume_mode": "auto",
            "log_val_generations": logging.trajectories_per_step,
            "rollout_data_dir": str(run_dir / "rollouts") if logging.save_trajectories else None,
        },
    }

    if model.dtype in ROLLOUT_DTYPE:
        tree["actor_rollout_ref"]["rollout"]["dtype"] = ROLLOUT_DTYPE[model.dtype]

    if multimodal:
        # verl/utils/dataset/rl_dataset.py reads images from data.image_key; VLM actors need the
        # unpadded (rmpad) path (examples/grpo_trainer/run_qwen2_5_vl_7b_fsdp.sh).
        tree["data"]["image_key"] = IMAGES_KEY
        tree["actor_rollout_ref"]["model"]["use_remove_padding"] = True
        mm_limit = rollout.extra.get("limit_mm_per_prompt")
        if mm_limit is not None:
            limit = {"image": int(mm_limit)} if isinstance(mm_limit, int) else dict(mm_limit)
            tree["actor_rollout_ref"]["rollout"]["engine_kwargs"] = {
                rollout_name: {"+limit_mm_per_prompt": limit}
            }

    if tool_calling:
        tree["actor_rollout_ref"]["rollout"]["multi_turn"].update(
            {
                "tool_config_path": str(tool_config_path(run_dir)),
                "max_assistant_turns": env.max_turns,
                "max_user_turns": env.max_turns,
                "format": "hermes",
            }
        )

    if algo.name == "dapo":
        # DAPO = GRPO + clip-higher + dynamic sampling (algorithm.filter_groups) + overlong penalty
        # (reward manager "dapo", verl/experimental/reward_loop/reward_manager/dapo.py).
        tree["algorithm"]["filter_groups"] = {
            "enable": algo.dynamic_sampling,
            "metric": "reward",
            "max_num_gen_batches": DAPO_MAX_NUM_GEN_BATCHES,
        }
        if algo.dynamic_sampling:
            tree["data"]["gen_batch_size"] = trainer.prompts_per_step * DAPO_GEN_BATCH_MULTIPLIER
        if algo.overlong_penalty > 0:
            buffer_len = max(1, int(rollout.max_new_tokens * OVERLONG_BUFFER_FRACTION))
            tree["reward"]["reward_manager"] = {"name": "dapo"}
            tree["reward"]["+reward_kwargs"] = {
                "overlong_buffer_cfg": {
                    "enable": True,
                    "len": buffer_len,
                    "penalty_factor": algo.overlong_penalty,
                    "log": False,
                },
                "max_resp_len": rollout.max_new_tokens,
            }
    elif algo.dynamic_sampling:
        tree["algorithm"]["filter_groups"] = {
            "enable": True,
            "metric": "reward",
            "max_num_gen_batches": DAPO_MAX_NUM_GEN_BATCHES,
        }
        tree["data"]["gen_batch_size"] = trainer.prompts_per_step * DAPO_GEN_BATCH_MULTIPLIER

    if algo.name == "ppo":
        # GAE needs a critic (verl/trainer/ppo/utils.py need_critic) and the KL goes into the reward.
        tree["critic"] = {
            "model": {"path": model.name_or_path, "trust_remote_code": model.trust_remote_code},
            "optim": {"lr": trainer.learning_rate * 10},
            "ppo_micro_batch_size_per_gpu": trainer.micro_batch_size,
        }
        tree["algorithm"]["use_kl_in_reward"] = algo.kl_coef > 0
        tree["algorithm"]["kl_ctrl"] = {"type": "fixed", "kl_coef": algo.kl_coef}
        tree["algorithm"]["kl_penalty"] = KL_LOSS_TYPE[algo.kl_estimator]

    if cluster.placement == "disaggregated" and cluster.rollout_gpus and cluster.trainer_gpus:
        # Standalone rollout pool (rollout.yaml nnodes/n_gpus_per_node) with the v1 separate trainer.
        tree["actor_rollout_ref"]["hybrid_engine"] = False
        tree["actor_rollout_ref"]["rollout"]["nnodes"] = cluster.num_nodes
        tree["actor_rollout_ref"]["rollout"]["n_gpus_per_node"] = cluster.rollout_gpus
        tree["trainer"]["n_gpus_per_node"] = cluster.trainer_gpus
        tree["trainer"]["v1"] = {"trainer_mode": "separate_async"}

    extra = {k: v for k, v in trainer.extra.items() if k not in RESERVED_EXTRA_KEYS}
    return deep_merge(tree, extra)


# verl optim/fsdp.yaml lr_scheduler_type: "constant" or "cosine" (linear decay has no equivalent).
SFT_SCHEDULER = {"constant": "constant", "cosine": "cosine", "linear": "cosine"}
SFT_ENGINE_DTYPE = {"bf16": "bfloat16", "fp16": "float16"}


def compile_sft_config(
    config: RunConfig, run_dir: str | Path, *, multimodal: bool | None = None
) -> dict[str, Any]:
    """Override tree for ``verl.trainer.sft_trainer_ray`` (``sft_trainer_engine.yaml``).

    ``trainer.prompts_per_step`` sequences per optimizer step (``data.train_batch_size``),
    ``trainer.micro_batch_size`` → ``data.micro_batch_size_per_gpu``, ``trainer.max_seq_len`` →
    ``data.max_length``; ``trainer.total_epochs`` is set to ``max_steps`` so the epoch loop can
    never end before ``trainer.total_training_steps`` (the trainer returns at the last step).
    ``checkpoint.save_contents`` adds ``hf_model`` so every ``global_step_<n>/huggingface`` is a
    loadable export (the run's ``final/`` hand-off).
    """
    run_dir = Path(run_dir).expanduser().resolve()
    trainer, model, cluster, data, logging = (
        config.trainer,
        config.model,
        config.cluster,
        config.data,
        config.logging,
    )
    if multimodal is None:
        multimodal = data.image_field is not None
    loggers = ["console"]
    if logging.wandb_project:
        loggers.append("wandb")
    if logging.tensorboard:
        loggers.append("tensorboard")
    tree: dict[str, Any] = {
        "data": {
            "train_files": str(sft_train_parquet_path(run_dir)),
            "val_files": str(sft_val_parquet_path(run_dir)) if data.eval_uri else None,
            "messages_key": MESSAGES_KEY,
            "max_length": trainer.max_seq_len,
            "truncation": "right",
            "train_batch_size": trainer.prompts_per_step,
            "micro_batch_size_per_gpu": trainer.micro_batch_size,
            "use_dynamic_bsz": False,
            "shuffle": data.shuffle,
            "seed": config.seed,
            "trust_remote_code": model.trust_remote_code,
        },
        "model": {
            "path": model.name_or_path,
            "tokenizer_path": model.tokenizer_name_or_path,
            "trust_remote_code": model.trust_remote_code,
            "custom_chat_template": model.chat_template,
            "enable_gradient_checkpointing": trainer.gradient_checkpointing,
            "lora_rank": model.lora.r if model.lora.enabled else 0,
            "lora_alpha": model.lora.alpha if model.lora.enabled else 16,
            "target_modules": model.lora.target_modules if model.lora.enabled else "all-linear",
            "use_remove_padding": True,
        },
        "optim": {
            "lr": trainer.learning_rate,
            "weight_decay": trainer.weight_decay,
            "lr_warmup_steps": trainer.warmup_steps if trainer.warmup_steps > 0 else -1,
            "lr_scheduler_type": SFT_SCHEDULER[trainer.scheduler],
            "clip_grad": trainer.max_grad_norm,
            "total_training_steps": trainer.max_steps,
        },
        "engine": {"seed": config.seed},
        "checkpoint": {"save_contents": ["model", "optimizer", "extra", "hf_model"]},
        "trainer": {
            "nnodes": cluster.num_nodes,
            "n_gpus_per_node": max(1, cluster.gpus_per_node),
            "total_training_steps": trainer.max_steps,
            "total_epochs": trainer.max_steps,
            "save_freq": trainer.save_every,
            "test_freq": trainer.eval_every if (trainer.eval_every and data.eval_uri) else -1,
            "project_name": PROJECT_NAME,
            "experiment_name": config.run_name or run_dir.name,
            "logger": loggers,
            "default_local_dir": str(run_dir / CHECKPOINT_DIR),
            "resume_mode": "auto",
            "seed": config.seed,
        },
    }
    if trainer.precision in SFT_ENGINE_DTYPE:
        tree["engine"]["dtype"] = SFT_ENGINE_DTYPE[trainer.precision]
    if multimodal:
        tree["data"]["image_key"] = IMAGES_KEY
    extra = {k: v for k, v in trainer.extra.items() if k not in RESERVED_EXTRA_KEYS}
    return deep_merge(tree, extra)


# -- Hydra override rendering --------------------------------------------------------------

_PLAIN = re.compile(r"^[A-Za-z0-9_./:@%+-]+$")
_LITERAL = re.compile(r"^(true|false|null|True|False|None|[-+]?\d+(\.\d*)?([eE][-+]?\d+)?)$")


def format_value(value: Any) -> str:
    """Render one value in Hydra's CLI override grammar."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{format_value(v)}" for k, v in value.items()) + "}"
    text = str(value)
    if _PLAIN.match(text) and not _LITERAL.match(text):
        return text
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def flatten(tree: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """``{"a": {"b": 1, "+c": 2}}`` -> ``[("a.b", 1), ("+a.c", 2)]``; a ``+`` on any segment marks the
    whole key as a Hydra add-override."""
    out: list[tuple[str, Any]] = []
    for key, value in tree.items():
        add = key.startswith("+")
        name = key[1:] if add else key
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict) and value and not add:
            out.extend(flatten(value, path))
        elif isinstance(value, dict) and value:
            out.extend((f"+{k.lstrip('+')}", v) for k, v in flatten(value, path))
        else:
            out.append((f"+{path}" if add else path, value))
    return out


def to_overrides(tree: dict[str, Any]) -> list[str]:
    """Flatten the compiled tree into ``key=value`` Hydra CLI overrides."""
    return [f"{key}={format_value(value)}" for key, value in flatten(tree)]
