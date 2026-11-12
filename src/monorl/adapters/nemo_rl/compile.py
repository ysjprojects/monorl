"""Compile a :class:`~monorl.config.RunConfig` into the NeMo-RL GRPO (or SFT) yaml tree.

The tree mirrors ``examples/configs/grpo_math_1B.yaml`` of NVIDIA-NeMo/RL at tag **v0.7.0**
(commit ``81aa43dda4765b0429cf31dab44441e4e4383911``) and is validated upstream by
``nemo_rl.algorithms.grpo.MasterConfig`` (``policy``: ``nemo_rl/models/policy/__init__.py``
``PolicyConfig``; ``loss_fn``: ``nemo_rl/algorithms/loss/loss_functions.py`` ``ClippedPGLossConfig``;
``grpo``: ``GRPOConfig``; ``checkpointing``: ``nemo_rl/utils/checkpoint.py``; ``logger``:
``nemo_rl/utils/logger.py`` ``LoggerConfig``; ``cluster``: ``nemo_rl/distributed/virtual_cluster.py``;
generation: ``nemo_rl/models/generation/interfaces.py`` + ``vllm/config.py``).

``algorithm.name: sft`` compiles ``examples/configs/sft.yaml`` instead (``nemo_rl.algorithms.sft``
``MasterConfig``/``SFTConfig``: ``sft.{max_num_epochs,max_num_steps,val_period,...}``, the same
``policy``/``checkpointing``/``logger``/``cluster`` sections and a ``data`` section pointing the stock
``ResponseDataset`` + ``sft_processor`` at the exported chat jsonl) and is launched through
``examples/run_sft.py``.

Images: NeMo-RL v0.7.0 trains VLMs only through its own datasets/processors
(``examples/run_vlm_grpo.py`` / ``run_vlm_sft.py`` with ``vlm_hf_data_processor`` building
``message_log`` entries that carry processor tensors). The monorl bridge feeds tokenised text
``message_log`` entries (:mod:`monorl.adapters.nemo_rl.data`), so ``data.image_field`` is rejected
with a :class:`~monorl.core.errors.ConfigError`.

``compile_config`` is a pure function of the config and the run directory; ``trainer.extra["nemo_rl"]``
is deep-merged last so any upstream knob can be overridden.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

from monorl.adapters.nemo_rl.data import TASK_NAME, data_paths
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.reporting import CHECKPOINT_DIR, METRICS_FILE

#: NeMo-RL ``policy.precision`` / ``vllm_cfg.precision`` names.
PRECISION: dict[str, str] = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}

#: Monorl rollout backends and the NeMo-RL ``policy.generation.backend`` they map to.
GENERATION_BACKENDS: dict[str, str] = {"vllm": "vllm", "hf": "vllm"}

#: ``env.<name>`` key and ``data.default.env_name`` for the monorl environment bridge.
ENV_NAME = "monorl"

#: Dotted path of the Ray actor class (``monorl.adapters.nemo_rl.env_actor``) registered under ``ENV_NAME``.
ACTOR_FQN = "monorl.adapters.nemo_rl.env_actor.MonorlEnvironment"

#: Name of the yaml written next to the data export.
CONFIG_FILENAME = "nemo_rl.yaml"

SUPPORTED_ALGORITHMS: frozenset[str] = frozenset({"grpo", "dapo", "rloo", "reinforce", "sft"})

#: ``examples/run_sft.py`` data processor name (``nemo_rl/data/processors.py`` ``sft_processor``).
SFT_PROCESSOR = "sft_processor"
#: Built-in dataset class that keeps a ``messages`` column as-is (``response_datasets/response_dataset.py``).
SFT_DATASET = "ResponseDataset"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; ``override`` wins, nested dicts are merged key-wise."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def config_path(run_dir: str | Path) -> Path:
    return data_paths(run_dir).root / CONFIG_FILENAME


def _check_supported(config: RunConfig) -> None:
    algo = config.algorithm
    if algo.name not in SUPPORTED_ALGORITHMS:
        raise ConfigError(
            f"trainer.backend='nemo_rl' supports algorithm.name in {sorted(SUPPORTED_ALGORITHMS)}, "
            f"got {algo.name!r} (NeMo-RL GRPO has no critic; ppo needs trainer.backend='verl')"
        )
    if config.data.image_field is not None:
        raise ConfigError(
            "data.image_field is not supported with trainer.backend='nemo_rl': NeMo-RL v0.7.0 trains "
            "vision-language models only through its own datasets (examples/run_vlm_grpo.py, "
            "run_vlm_sft.py, nemo_rl/data/processors.py vlm_hf_data_processor), which the monorl text "
            "message_log bridge does not produce; use trainer.backend='trl', 'verl' or 'reference'"
        )
    if algo.name == "sft":
        return
    if algo.num_epochs != 1:
        raise ConfigError(
            "algorithm.num_epochs > 1 has no NeMo-RL equivalent (one optimizer pass per rollout batch)"
        )
    if algo.entropy_coef > 0:
        raise ConfigError(
            "algorithm.entropy_coef > 0 has no NeMo-RL equivalent (ClippedPGLossFn has no entropy bonus)"
        )
    if config.rollout.backend not in GENERATION_BACKENDS:
        raise ConfigError(
            f"trainer.backend='nemo_rl' supports rollout.backend in {sorted(GENERATION_BACKENDS)}, "
            f"got {config.rollout.backend!r}"
        )


def _optimizer(config: RunConfig) -> dict[str, Any]:
    tr = config.trainer
    if tr.optimizer == "sgd":
        return {
            "name": "torch.optim.SGD",
            "kwargs": {"lr": tr.learning_rate, "weight_decay": tr.weight_decay, "momentum": 0.9},
        }
    name = "torch.optim.AdamW" if tr.optimizer == "adamw" else "torch.optim.Adam"
    return {
        "name": name,
        "kwargs": {
            "lr": tr.learning_rate,
            "weight_decay": tr.weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    }


def _scheduler(config: RunConfig) -> dict[str, Any] | list[dict[str, Any]]:
    """``policy.scheduler``: a single ``{name, kwargs}`` or a ``SequentialLR`` list with warmup."""
    tr = config.trainer
    main_steps = max(tr.max_steps - tr.warmup_steps, 1)
    if tr.scheduler == "linear":
        main = {
            "name": "torch.optim.lr_scheduler.LinearLR",
            "kwargs": {"start_factor": 1.0, "end_factor": 0.0, "total_iters": main_steps},
        }
    elif tr.scheduler == "cosine":
        main = {
            "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "kwargs": {"T_max": main_steps, "eta_min": 0.0},
        }
    else:
        main = {
            "name": "torch.optim.lr_scheduler.ConstantLR",
            "kwargs": {"factor": 1.0, "total_iters": main_steps},
        }
    if tr.warmup_steps == 0:
        return main
    warmup = {
        "name": "torch.optim.lr_scheduler.LinearLR",
        "kwargs": {
            "start_factor": 1.0 / (tr.warmup_steps + 1),
            "end_factor": 1.0,
            "total_iters": tr.warmup_steps,
        },
    }
    return [warmup, main, {"milestones": [tr.warmup_steps]}]


def _lora_cfg(config: RunConfig, tensor_parallel_size: int) -> dict[str, Any]:
    lora = config.model.lora
    if not lora.enabled:
        return {"enabled": False}
    all_linear = lora.target_modules == "all-linear"
    return {
        "enabled": True,
        "target_modules": [] if all_linear else list(lora.target_modules),
        "exclude_modules": [],
        "match_all_linear": all_linear,
        "dim": lora.r,
        "alpha": lora.alpha,
        "dropout": lora.dropout,
        "dropout_position": "post",
        "lora_A_init": "xavier",
        "use_triton": tensor_parallel_size == 1,
    }


def _dtensor_cfg(config: RunConfig) -> dict[str, Any]:
    extra = dict(config.trainer.extra.get("dtensor") or {})
    tp = int(extra.get("tensor_parallel_size", 1))
    base = {
        "_v2": True,
        "enabled": True,
        "cpu_offload": False,
        "sequence_parallel": False,
        "activation_checkpointing": config.trainer.gradient_checkpointing,
        "tensor_parallel_size": tp,
        "context_parallel_size": 1,
        "custom_parallel_plan": None,
        "automodel_kwargs": {},
        "lora_cfg": _lora_cfg(config, tp),
    }
    return deep_merge(base, extra)


def _megatron_cfg(config: RunConfig, precision: str) -> dict[str, Any]:
    """``policy.megatron_cfg``: disabled unless ``trainer.extra["megatron"]`` enables it; then the
    v0.7.0 reference defaults filled from the trainer config and merged with the user's dict."""
    extra = dict(config.trainer.extra.get("megatron") or {})
    if not extra.get("enabled"):
        return {"enabled": False}
    tr = config.trainer
    lora = config.model.lora
    base = {
        "enabled": True,
        "force_reconvert_from_hf": False,
        "empty_unused_memory_level": 1,
        "activation_checkpointing": tr.gradient_checkpointing,
        "recompute_granularity": "full",
        "recompute_modules": None,
        "tensor_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "num_layers_in_first_pipeline_stage": None,
        "num_layers_in_last_pipeline_stage": None,
        "context_parallel_size": 1,
        "pipeline_dtype": precision,
        "sequence_parallel": False,
        "freeze_moe_router": True,
        "moe_router_dtype": "fp64",
        "moe_router_load_balancing_type": "none",
        "moe_router_bias_update_rate": 0.0,
        "moe_permute_fusion": True,
        "apply_rope_fusion": True,
        "bias_activation_fusion": True,
        "defer_fp32_logits": False,
        "moe_per_layer_logging": False,
        "moe_enable_deepep": False,
        "moe_token_dispatcher_type": "alltoall",
        "moe_shared_expert_overlap": False,
        "mtp_num_layers": 0,
        "mtp_loss_scaling_factor": 0.0,
        "mtp_use_repeated_layer": False,
        "mtp_detach_heads": False,
        "gradient_accumulation_fusion": False,
        "use_fused_weighted_squared_relu": False,
        "peft": (
            {
                "enabled": True,
                "target_modules": [] if lora.target_modules == "all-linear" else list(lora.target_modules),
                "exclude_modules": [],
                "dim": lora.r,
                "alpha": lora.alpha,
                "dropout": lora.dropout,
                "dropout_position": "post",
                "lora_A_init_method": "xavier",
                "lora_B_init_method": "zero",
                "a2a_experimental": False,
                "lora_dtype": None,
            }
            if lora.enabled
            else {"enabled": False}
        ),
        "optimizer": {
            "optimizer": "sgd" if tr.optimizer == "sgd" else "adam",
            "lr": tr.learning_rate,
            "min_lr": tr.learning_rate / 10,
            "weight_decay": tr.weight_decay,
            "bf16": precision == "bfloat16",
            "fp16": precision == "float16",
            "params_dtype": "float32",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "sgd_momentum": 0.9,
            "use_distributed_optimizer": True,
            "use_precision_aware_optimizer": True,
            "clip_grad": tr.max_grad_norm,
            "optimizer_cpu_offload": False,
            "optimizer_offload_fraction": 0.0,
        },
        "scheduler": {
            "start_weight_decay": tr.weight_decay,
            "end_weight_decay": tr.weight_decay,
            "weight_decay_incr_style": "constant",
            "lr_decay_style": {"constant": "constant", "linear": "linear", "cosine": "cosine"}[tr.scheduler],
            "lr_decay_iters": tr.max_steps,
            "lr_warmup_iters": tr.warmup_steps,
            "lr_warmup_init": tr.learning_rate / 10,
        },
        "distributed_data_parallel_config": {
            "grad_reduce_in_fp32": False,
            "overlap_grad_reduce": True,
            "overlap_param_gather": True,
            "use_custom_fsdp": False,
            "data_parallel_sharding_strategy": "optim_grads_params",
        },
        "fp8_cfg": {"enabled": False, "fp8": "e4m3", "fp8_recipe": "blockwise", "fp8_param": False},
        "env_vars": None,
    }
    return deep_merge(base, extra)


def build_env_config(config: RunConfig, run_dir: str | None = None) -> dict[str, Any]:
    """The ``env.monorl`` dict handed to the bridge actor constructor (yaml-safe)."""
    return {
        "run_config": config.to_dict(),
        "rewards": [spec.model_dump(mode="json") for spec in config.rewards],
        "run_dir": run_dir,
    }


def colocation(config: RunConfig) -> dict[str, Any]:
    """``policy.generation.colocated`` from ``cluster.placement`` (rules of ``grpo.setup`` v0.7.0:
    single node → ``resources.gpus_per_node`` is the inference share; multi-node → whole nodes)."""
    cl = config.cluster
    if cl.placement != "disaggregated":
        return {"enabled": True, "resources": {"gpus_per_node": None, "num_nodes": None}}
    rollout_gpus = cl.rollout_gpus or 0
    if cl.num_nodes == 1:
        resources = {"gpus_per_node": rollout_gpus, "num_nodes": None}
    else:
        per_node = max(cl.gpus_per_node, 1)
        resources = {
            "gpus_per_node": cl.gpus_per_node,
            "num_nodes": max(1, math.ceil(rollout_gpus / per_node)),
        }
    return {"enabled": False, "resources": resources}


def _generation(config: RunConfig, precision: str, async_rollouts: bool) -> dict[str, Any]:
    ro, tr = config.rollout, config.trainer
    max_model_len = ro.max_model_len or tr.max_seq_len
    return {
        "backend": GENERATION_BACKENDS[ro.backend],
        "max_new_tokens": ro.max_new_tokens,
        "temperature": ro.temperature,
        "top_p": ro.top_p,
        "top_k": None if ro.top_k is None or ro.top_k < 0 else ro.top_k,
        "stop_token_ids": None,
        "stop_strings": list(ro.stop) or None,
        "port_range_low": 11001,
        "port_range_high": 15000,
        "colocated": colocation(config),
        "vllm_cfg": {
            "async_engine": async_rollouts,
            "precision": precision,
            "kv_cache_dtype": "auto",
            "tensor_parallel_size": ro.tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": ro.gpu_memory_utilization,
            "max_model_len": max_model_len,
            "enforce_eager": False,
            "use_tqdm": False,
            "use_deep_gemm": False,
            "num_last_layers_in_bf16": 0,
            "num_first_layers_in_bf16": 0,
            "enable_vllm_metrics_logger": True,
            "vllm_metrics_logger_interval": 0.5,
        },
        "vllm_kwargs": dict(ro.extra.get("vllm_kwargs") or {}),
    }


def _policy(config: RunConfig, precision: str, *, sequences_per_step: int) -> dict[str, Any]:
    """``policy`` section shared by GRPO and SFT (``nemo_rl/models/policy/__init__.py`` ``PolicyConfig``)."""
    tr = config.trainer
    return {
        "model_name": config.model.name_or_path,
        "tokenizer": {
            "name": config.model.tokenizer_name_or_path or config.model.name_or_path,
            "chat_template_kwargs": None,
        },
        "hf_config_overrides": {},
        "train_global_batch_size": sequences_per_step,
        "train_micro_batch_size": tr.micro_batch_size,
        "max_total_sequence_length": tr.max_seq_len,
        "precision": precision,
        "offload_optimizer_for_logprob": False,
        "dtensor_cfg": _dtensor_cfg(config),
        "megatron_cfg": _megatron_cfg(config, precision),
        "draft": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "make_sequence_length_divisible_by": 1,
        "max_grad_norm": tr.max_grad_norm,
        "optimizer": _optimizer(config),
        "scheduler": _scheduler(config),
    }


def _checkpointing(config: RunConfig, run_dir: Path) -> dict[str, Any]:
    return {
        "enabled": True,
        "checkpoint_dir": str(run_dir / CHECKPOINT_DIR),
        "metric_name": None,
        "higher_is_better": True,
        "keep_top_k": None,
        "save_period": config.trainer.save_every,
        "checkpoint_must_save_by": None,
        "model_save_format": "safetensors",
        "save_consolidated": True,
        "save_optimizer": True,
        "is_peft": config.model.lora.enabled,
    }


def _logger(config: RunConfig, run_dir: Path, paths: Any) -> dict[str, Any]:
    lg = config.logging
    run_name = config.run_name or run_dir.name
    return {
        "log_dir": str(paths.root / "logs"),
        "num_val_samples_to_print": 0,
        "wandb_enabled": lg.wandb_project is not None,
        "tensorboard_enabled": lg.tensorboard,
        "mlflow_enabled": False,
        "swanlab_enabled": False,
        "monitor_gpus": False,
        "wandb": {"project": lg.wandb_project or "monorl", "name": run_name},
        "swanlab": {"project": "monorl", "name": run_name},
        "tensorboard": {},
        "mlflow": {
            "experiment_name": "monorl",
            "run_name": run_name,
            "tracking_uri": "http://localhost:5000",
        },
        "gpu_monitoring": {"collection_interval": 10, "flush_interval": 10},
        "monorl": {"enabled": True, "metrics_file": str(run_dir / METRICS_FILE), "run_dir": str(run_dir)},
    }


def _cluster(config: RunConfig) -> dict[str, Any]:
    cl = config.cluster
    return {
        "gpus_per_node": cl.gpus_per_node,
        "num_nodes": cl.num_nodes,
        "master_port_range_low": 25000,
        "master_port_range_high": 28000,
        "segment_size": None,
    }


def _override(config: RunConfig, compiled: dict[str, Any]) -> dict[str, Any]:
    override = config.trainer.extra.get("nemo_rl") or {}
    if not isinstance(override, dict):
        raise ConfigError("trainer.extra['nemo_rl'] must be a mapping of NeMo-RL config overrides")
    return deep_merge(compiled, override)


def compile_sft_config(config: RunConfig, run_dir: str | Path) -> dict[str, Any]:
    """The NeMo-RL SFT master config (``examples/configs/sft.yaml`` shape) for ``config``.

    ``trainer.prompts_per_step`` sequences per optimizer step; ``sft.max_num_steps = max_steps`` with
    ``max_num_epochs = max_steps`` so the step bound always governs (upstream trains
    ``min(max_num_epochs * len(dataloader), max_num_steps)`` steps); validation every
    ``trainer.eval_every`` steps on ``data.eval_uri`` (``val_loss`` → ``eval/loss``).
    """
    _check_supported(config)
    run_dir = Path(run_dir).expanduser().resolve()
    paths = data_paths(run_dir)
    tr, dt = config.trainer, config.data
    precision = PRECISION[tr.precision]
    eval_enabled = dt.eval_uri is not None and tr.eval_every is not None

    policy = _policy(config, precision, sequences_per_step=tr.prompts_per_step)
    if policy["megatron_cfg"]["enabled"]:
        policy["dtensor_cfg"] = {"enabled": False}
        policy["optimizer"] = None
        policy["scheduler"] = None

    sft: dict[str, Any] = {
        "max_num_epochs": tr.max_steps,
        "max_num_steps": tr.max_steps,
        "val_period": tr.eval_every if eval_enabled else 0,
        "val_batches": 8,
        "val_global_batch_size": tr.prompts_per_step,
        "val_micro_batch_size": tr.micro_batch_size,
        "val_at_start": False,
        "val_at_end": eval_enabled,
        "seed": config.seed,
        "only_unmask_final": False,
    }
    data: dict[str, Any] = {
        "max_input_seq_length": tr.max_seq_len,
        "add_bos": True,
        "add_eos": True,
        "add_generation_prompt": False,
        "shuffle": dt.shuffle,
        "num_workers": 1,
        "train": {
            "dataset_name": SFT_DATASET,
            "data_path": str(paths.sft_train_jsonl),
            "split_validation_size": 0,
            "seed": config.seed,
        },
        "validation": {"dataset_name": SFT_DATASET, "data_path": str(paths.sft_eval_jsonl)}
        if dt.eval_uri is not None
        else None,
        "default": {"prompt_file": None, "system_prompt_file": None, "processor": SFT_PROCESSOR},
    }
    checkpointing = _checkpointing(config, run_dir)
    if eval_enabled:
        checkpointing.update({"metric_name": "val:val_loss", "higher_is_better": False})
    compiled: dict[str, Any] = {
        "sft": sft,
        "checkpointing": checkpointing,
        "policy": policy,
        "data": data,
        "logger": _logger(config, run_dir, paths),
        "cluster": _cluster(config),
    }
    return _override(config, compiled)


def compile_config(config: RunConfig, run_dir: str | Path) -> dict[str, Any]:
    """The NeMo-RL master config (plain dict, yaml-serialisable) for ``config``.

    Paths inside point at the data export of :func:`monorl.adapters.nemo_rl.data.export_dataset`
    (``<run_dir>/nemo_rl/``), the checkpoint dir ``<run_dir>/checkpoints`` and the metrics file
    ``<run_dir>/metrics.jsonl`` written by the logger patch.
    """
    _check_supported(config)
    if config.is_sft:
        return compile_sft_config(config, run_dir)
    run_dir = Path(run_dir).expanduser().resolve()
    paths = data_paths(run_dir)
    algo, ro, tr, dt = config.algorithm, config.rollout, config.trainer, config.data
    precision = PRECISION[tr.precision]
    async_rollouts = ro.async_steps_ahead > 0
    group_normalise = algo.advantage_norm != "none" and algo.advantage_std_norm
    eval_enabled = dt.eval_uri is not None and tr.eval_every is not None

    policy = _policy(config, precision, sequences_per_step=config.sequences_per_step())
    policy["generation_batch_size"] = config.sequences_per_step()
    policy["logprob_batch_size"] = tr.micro_batch_size
    policy["logprob_chunk_size"] = None
    policy["generation"] = _generation(config, precision, async_rollouts)
    if policy["megatron_cfg"]["enabled"]:
        policy["dtensor_cfg"] = {"enabled": False}
        policy["optimizer"] = None
        policy["scheduler"] = None

    grpo: dict[str, Any] = {
        "num_prompts_per_step": tr.prompts_per_step,
        "num_generations_per_prompt": ro.group_size,
        "max_rollout_turns": config.env.max_turns,
        "max_num_epochs": 1,
        "max_num_steps": tr.max_steps,
        "normalize_rewards": group_normalise,
        "use_leave_one_out_baseline": algo.name == "rloo",
        "val_period": tr.eval_every if eval_enabled else 0,
        "val_at_start": False,
        "val_at_end": eval_enabled,
        "overlong_filtering": False,
        "advantage_clip_low": None,
        "advantage_clip_high": None,
        "max_val_samples": 256,
        "val_batch_size": 256,
        "seed": config.seed,
        "use_dynamic_sampling": algo.dynamic_sampling,
        "dynamic_sampling_max_gen_batches": 10,
        "batch_multiplier": 1,
        "reward_shaping": {
            "enabled": algo.overlong_penalty > 0,
            "overlong_buffer_length": 128,
            "overlong_buffer_penalty": algo.overlong_penalty,
            "max_response_length": ro.max_new_tokens,
            "stop_properly_penalty_coef": None,
        },
        "adv_estimator": {
            "name": "grpo",
            "normalize_rewards": group_normalise,
            "use_leave_one_out_baseline": algo.name == "rloo",
            "minus_baseline": algo.name != "reinforce",
        },
        "reward_scaling": {
            "enabled": False,
            "source_min": 0.0,
            "source_max": 1.0,
            "target_min": 0.0,
            "target_max": 1.0,
        },
        "seq_logprob_error_threshold": None,
        "invalid_tool_call_advantage": None,
        "malformed_thinking_advantage": None,
        "async_grpo": {
            "enabled": async_rollouts,
            "max_trajectory_age_steps": max(ro.async_steps_ahead, 1),
            "in_flight_weight_updates": False,
            "recompute_kv_cache_after_weight_updates": False,
        },
    }

    loss_fn: dict[str, Any] = {
        "reference_policy_kl_penalty": algo.kl_coef,
        "reference_policy_kl_type": algo.kl_estimator,
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "ratio_clip_min": algo.clip_eps,
        "ratio_clip_max": algo.clip_eps_high if algo.clip_eps_high is not None else algo.clip_eps,
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": async_rollouts,
        "truncated_importance_sampling_type": None,
        "truncated_importance_sampling_ratio": None,
        "truncated_importance_sampling_ratio_min": None,
        "sequence_level_importance_ratios": False,
        "token_level_loss": algo.loss_agg == "token",
        "force_on_policy_ratio": False,
        "use_kl_in_reward": False,
        "use_cispo": False,
        "disable_ppo_ratio": algo.name == "reinforce",
        "positive_example_nll_weight": 0.0,
    }

    dataset_class = f"{paths.module_name}.MonorlDataset"
    data: dict[str, Any] = {
        "max_input_seq_length": min(dt.max_prompt_tokens, tr.max_seq_len),
        "shuffle": dt.shuffle,
        "num_workers": 1,
        "use_multiple_dataloader": False,
        "train": {
            "dataset_name": dataset_class,
            "data_path": str(paths.train_jsonl),
            "split_validation_size": 0,
            "seed": config.seed,
        },
        "validation": (
            {"dataset_name": dataset_class, "data_path": str(paths.eval_jsonl)}
            if dt.eval_uri is not None
            else None
        ),
        "default": {
            "prompt_file": None,
            "system_prompt_file": None,
            "processor": TASK_NAME,
            "env_name": ENV_NAME,
        },
    }

    env: dict[str, Any] = {ENV_NAME: build_env_config(config, str(run_dir))}

    compiled: dict[str, Any] = {
        "grpo": grpo,
        "loss_fn": loss_fn,
        "checkpointing": _checkpointing(config, run_dir),
        "policy": policy,
        "data": data,
        "env": env,
        "logger": _logger(config, run_dir, paths),
        "cluster": _cluster(config),
        "data_plane": None,
        "on_policy_distillation": None,
    }
    return _override(config, compiled)


def write_config(compiled: dict[str, Any], run_dir: str | Path) -> Path:
    """Write ``compiled`` to ``<run_dir>/nemo_rl/nemo_rl.yaml`` and return the path."""
    path = config_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(compiled, sort_keys=False, default_flow_style=False))
    return path


__all__ = [
    "ACTOR_FQN",
    "CONFIG_FILENAME",
    "ENV_NAME",
    "GENERATION_BACKENDS",
    "PRECISION",
    "SUPPORTED_ALGORITHMS",
    "build_env_config",
    "colocation",
    "compile_config",
    "compile_sft_config",
    "config_path",
    "deep_merge",
    "write_config",
]
