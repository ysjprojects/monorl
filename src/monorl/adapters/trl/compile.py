"""Compile a :class:`~monorl.config.RunConfig` into ``trl.GRPOConfig`` keyword arguments.

Pure function of the config (``grpo_config_kwargs``) so the mapping is unit-testable without
instantiating ``TrainingArguments``; :func:`compile_grpo_config` builds the real object.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError

#: TRL loss formulation per (monorl algorithm, loss aggregation).
LOSS_TYPES: dict[tuple[str, str], str] = {
    ("grpo", "token"): "bnpo",
    ("grpo", "sequence"): "grpo",
    ("dapo", "token"): "dapo",
    ("dapo", "sequence"): "grpo",
}

SCHEDULERS = {"constant": "constant", "linear": "linear", "cosine": "cosine"}
OPTIMIZERS = {"adamw": "adamw_torch", "adam": "adamw_torch", "sgd": "sgd"}


def resolve_device(config: RunConfig) -> torch.device:
    """Device ``TrainingArguments`` will pick (explicit ``trainer.device`` wins)."""
    if config.trainer.device:
        return torch.device(config.trainer.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_batch(config: RunConfig) -> tuple[int, int]:
    """``(per_device_train_batch_size, gradient_accumulation_steps)`` whose product is
    ``prompts_per_step * group_size`` (TRL's generation batch), honouring ``micro_batch_size``
    as an upper bound on the per-device batch."""
    sequences = config.sequences_per_step()
    micro = min(config.trainer.micro_batch_size, sequences)
    while sequences % micro:
        micro -= 1
    return micro, sequences // micro


def grpo_config_kwargs(config: RunConfig, run_dir: Path, *, device: torch.device | None = None) -> dict[str, Any]:
    """Keyword arguments for ``GRPOConfig``; ``trainer.extra`` overrides any field last."""
    device = device or resolve_device(config)
    algo, tr, ro, lg = config.algorithm, config.trainer, config.rollout, config.logging

    loss_type = LOSS_TYPES.get((algo.name, algo.loss_agg))
    if loss_type is None:
        raise ConfigError(
            f"trainer.backend='trl' supports algorithm.name in {{'grpo', 'dapo'}}, got {algo.name!r}; "
            "use trainer.backend='reference' for rloo/reinforce (ppo needs a critic: trainer.backend='verl')"
        )
    if tr.optimizer == "adam" and tr.weight_decay > 0:
        raise ConfigError("trainer.optimizer='adam' with weight_decay > 0 has no TRL equivalent; use 'adamw'")

    if algo.advantage_norm == "group":
        scale_rewards = "group" if algo.advantage_std_norm else "none"
    else:
        scale_rewards = algo.advantage_norm

    per_device, grad_accum = split_batch(config)
    is_cuda = device.type == "cuda"
    kwargs: dict[str, Any] = {
        "output_dir": str(Path(run_dir) / "checkpoints"),
        "seed": config.seed,
        "learning_rate": tr.learning_rate,
        "max_steps": tr.max_steps,
        "num_generations": ro.group_size,
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": grad_accum,
        "per_device_eval_batch_size": ro.group_size,
        "max_completion_length": ro.max_new_tokens,
        "temperature": ro.temperature,
        "top_p": ro.top_p,
        "top_k": ro.top_k if ro.top_k > 0 else 0,
        "beta": algo.kl_coef,
        "epsilon": algo.clip_eps,
        "epsilon_high": algo.clip_eps_high,
        "loss_type": loss_type,
        "scale_rewards": scale_rewards,
        "entropy_coef": algo.entropy_coef,
        "num_iterations": algo.num_epochs,
        "warmup_steps": tr.warmup_steps,
        "weight_decay": tr.weight_decay,
        "max_grad_norm": tr.max_grad_norm,
        "optim": OPTIMIZERS[tr.optimizer],
        "lr_scheduler_type": SCHEDULERS[tr.scheduler],
        "bf16": tr.precision == "bf16" and is_cuda,
        "fp16": tr.precision == "fp16" and is_cuda,
        "use_cpu": device.type == "cpu",
        "gradient_checkpointing": tr.gradient_checkpointing,
        "save_strategy": "steps",
        "save_steps": tr.save_every,
        "logging_strategy": "steps",
        "logging_steps": lg.log_every,
        "report_to": "none",
        "log_completions": False,
        "disable_tqdm": True,
        "remove_unused_columns": False,
    }
    if config.data.eval_uri and tr.eval_every:
        kwargs.update({"eval_strategy": "steps", "eval_steps": tr.eval_every})
    if ro.backend == "vllm":
        extra = ro.extra
        server = bool(extra.get("server_base_url") or extra.get("server_host") or extra.get("mode") == "server")
        kwargs.update(
            {
                "use_vllm": True,
                "vllm_mode": "server" if server else "colocate",
                "vllm_gpu_memory_utilization": ro.gpu_memory_utilization,
                "vllm_tensor_parallel_size": ro.tensor_parallel_size,
                "vllm_enable_sleep_mode": ro.enable_sleep_mode,
                "vllm_max_model_length": ro.max_model_len,
            }
        )
        if server:
            kwargs["vllm_server_base_url"] = extra.get("server_base_url")
            kwargs["vllm_server_host"] = extra.get("server_host", "0.0.0.0")
            kwargs["vllm_server_port"] = int(extra.get("server_port", 8000))
    elif ro.backend != "hf":
        raise ConfigError(f"trainer.backend='trl' generates with rollout.backend 'hf' or 'vllm', got {ro.backend!r}")
    kwargs.update(tr.extra)
    return kwargs


def compile_grpo_config(config: RunConfig, run_dir: Path, *, device: torch.device | None = None) -> Any:
    """Instantiate ``trl.GRPOConfig``; unknown ``trainer.extra`` keys are a :class:`ConfigError`."""
    from trl import GRPOConfig

    kwargs = grpo_config_kwargs(config, run_dir, device=device)
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    unknown = sorted(k for k in config.trainer.extra if k not in fields)
    if unknown:
        raise ConfigError(f"trainer.extra keys are not GRPOConfig fields: {unknown}")
    return GRPOConfig(**kwargs)
