"""Compile a :class:`~monorl.config.RunConfig` into ``trl.GRPOConfig`` / ``trl.SFTConfig`` keyword arguments.

Pure functions of the config (``grpo_config_kwargs``, ``sft_config_kwargs``) so the mapping is
unit-testable without instantiating ``TrainingArguments``; :func:`compile_grpo_config` /
:func:`compile_sft_config` build the real objects.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError

#: TRL loss formulation per (monorl algorithm, loss aggregation). TRL trains one sequence per
#: rollout, so "rollout" aggregation is its per-sequence formulation.
LOSS_TYPES: dict[tuple[str, str], str] = {
    ("grpo", "token"): "bnpo",
    ("grpo", "sequence"): "grpo",
    ("grpo", "rollout"): "grpo",
    ("dapo", "token"): "dapo",
    ("dapo", "sequence"): "grpo",
    ("dapo", "rollout"): "grpo",
}

SCHEDULERS = {"constant": "constant", "linear": "linear", "cosine": "cosine"}
OPTIMIZERS = {"adamw": "adamw_torch", "adam": "adamw_torch", "sgd": "sgd"}


def importance_sampling_kwargs(config: RunConfig) -> dict[str, Any]:
    """``GRPOConfig`` ``vllm_importance_sampling_*`` fields for ``algorithm.off_policy``.

    TRL always clips against the training model's log-probs at generation time and, for vLLM
    rollouts only, multiplies the loss by ``exp(old - sampling log-probs)`` per token or per
    sequence, clamped (``*_truncate``) or zeroed outside ``[clip_min, clip_max]`` (``*_mask``).
    Without explicit options TRL's own defaults apply.
    """
    off = config.algorithm.off_policy
    if off.score_centering:
        raise ConfigError(
            "algorithm.off_policy.score_centering has no TRL equivalent (GRPOTrainer has importance-sampling "
            "corrections only); use trainer.backend='reference'"
        )
    if off.anchor == "behavior":
        raise ConfigError(
            "algorithm.off_policy.anchor='behavior' has no TRL equivalent: GRPOTrainer always clips against "
            "the training model's log-probs"
        )
    if off.anchor == "auto" and not off.corrects:
        return {}
    weights, mask, cap = off.importance_weights, off.ratio_mask, off.importance_cap
    if config.rollout.backend != "vllm":
        if off.corrects:
            raise ConfigError(
                "algorithm.off_policy.importance_weights / ratio_mask need rollout.backend='vllm' with "
                "trainer.backend='trl': GRPOTrainer corrects vLLM rollouts only (its own generation is the "
                "training policy)"
            )
        return {}
    if not off.corrects:
        return {"vllm_importance_sampling_correction": False}
    if weights != "none" and mask == "none" and cap is not None:
        mode, low, high = f"{weights}_truncate", None, cap
    elif mask == weights and (cap is None or cap >= off.ratio_mask_high):
        mode, low, high = f"{weights}_mask", off.ratio_mask_low, off.ratio_mask_high
    else:
        raise ConfigError(
            f"algorithm.off_policy importance_weights={weights!r} (cap {cap}) with ratio_mask={mask!r} has no "
            "TRL equivalent. Supported: capped weights alone ('token_truncate' / 'sequence_truncate'); a mask "
            "at the level of the weights with importance_cap null or >= ratio_mask_high ('token_mask' / "
            "'sequence_mask')"
        )
    return {
        "vllm_importance_sampling_correction": True,
        "vllm_importance_sampling_mode": mode,
        "vllm_importance_sampling_clip_min": low,
        "vllm_importance_sampling_clip_max": high,
    }


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
    ``prompts_per_step * group_size`` (TRL's generation batch) — or ``prompts_per_step`` sequences
    for SFT — honouring ``micro_batch_size`` as an upper bound on the per-device batch."""
    sequences = config.trainer.prompts_per_step if config.is_sft else config.sequences_per_step()
    micro = min(config.trainer.micro_batch_size, sequences)
    while sequences % micro:
        micro -= 1
    return micro, sequences // micro


def _training_args_kwargs(config: RunConfig, run_dir: Path, device: torch.device) -> dict[str, Any]:
    """``TrainingArguments`` fields shared by GRPO and SFT."""
    tr, lg = config.trainer, config.logging
    if tr.optimizer == "adam" and tr.weight_decay > 0:
        raise ConfigError("trainer.optimizer='adam' with weight_decay > 0 has no TRL equivalent; use 'adamw'")
    per_device, grad_accum = split_batch(config)
    is_cuda = device.type == "cuda"
    kwargs: dict[str, Any] = {
        "output_dir": str(Path(run_dir) / "checkpoints"),
        "seed": config.seed,
        "learning_rate": tr.learning_rate,
        "max_steps": tr.max_steps,
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": grad_accum,
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
        "disable_tqdm": True,
    }
    if config.data.eval_uri and tr.eval_every:
        kwargs.update({"eval_strategy": "steps", "eval_steps": tr.eval_every})
    return kwargs


def sft_config_kwargs(
    config: RunConfig, run_dir: Path, *, device: torch.device | None = None
) -> dict[str, Any]:
    """Keyword arguments for ``SFTConfig`` (TRL 1.17 ``trl/trainer/sft_config.py``).

    ``trainer.prompts_per_step`` sequences per optimizer step, ``max_length = trainer.max_seq_len``,
    ``completion_only_loss`` unless ``algorithm.sft_train_on_prompt``. ``label_smoothing_factor`` is
    left at TRL's 0 (deprecated in TRL 1.17): :func:`sft_label_smoothing` routes
    ``algorithm.sft_label_smoothing`` to the trainer subclass instead. ``trainer.extra`` overrides
    any field last.
    """
    device = device or resolve_device(config)
    algo, tr = config.algorithm, config.trainer
    if algo.name != "sft":
        raise ConfigError(f"sft_config_kwargs needs algorithm.name='sft', got {algo.name!r}")
    kwargs = _training_args_kwargs(config, run_dir, device)
    kwargs.update(
        {
            "per_device_eval_batch_size": kwargs["per_device_train_batch_size"],
            "max_length": tr.max_seq_len,
            "completion_only_loss": not algo.sft_train_on_prompt,
            "assistant_only_loss": False,
            "packing": False,
            "padding_free": False,
            "loss_type": "nll",
        }
    )
    kwargs.update(tr.extra)
    return kwargs


def sft_label_smoothing(config: RunConfig) -> float:
    """Label smoothing :class:`~monorl.adapters.trl.sft.LabelSmoothedSFTTrainer` has to apply; ``0.0``
    selects the stock ``trl.SFTTrainer``.

    ``algorithm.sft_label_smoothing`` unless ``trainer.extra`` overrides the loss: an explicit
    ``label_smoothing_factor`` hands smoothing to TRL's own (deprecated) ``LabelSmoother`` path, and
    TRL defines no smoothed form of a non-``nll`` ``loss_type`` (``dft``).
    """
    extra = config.trainer.extra
    if "label_smoothing_factor" in extra or extra.get("loss_type", "nll") != "nll":
        return 0.0
    return config.algorithm.sft_label_smoothing


def compile_sft_config(config: RunConfig, run_dir: Path, *, device: torch.device | None = None) -> Any:
    """Instantiate ``trl.SFTConfig``; unknown ``trainer.extra`` keys are a :class:`ConfigError`."""
    from trl import SFTConfig

    kwargs = sft_config_kwargs(config, run_dir, device=device)
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    unknown = sorted(k for k in config.trainer.extra if k not in fields)
    if unknown:
        raise ConfigError(f"trainer.extra keys are not SFTConfig fields: {unknown}")
    return SFTConfig(**kwargs)


def grpo_config_kwargs(
    config: RunConfig, run_dir: Path, *, device: torch.device | None = None
) -> dict[str, Any]:
    """Keyword arguments for ``GRPOConfig``; ``trainer.extra`` overrides any field last."""
    device = device or resolve_device(config)
    algo, tr, ro = config.algorithm, config.trainer, config.rollout

    loss_type = LOSS_TYPES.get((algo.name, algo.loss_agg))
    if loss_type is None:
        raise ConfigError(
            f"trainer.backend='trl' supports algorithm.name in {{'grpo', 'dapo', 'sft'}}, got {algo.name!r}; "
            "use trainer.backend='reference' for rloo/reinforce (ppo needs a critic: trainer.backend='verl')"
        )
    if algo.credit_assignment == "turn":
        raise ConfigError(
            "algorithm.credit_assignment='turn' is not supported with trainer.backend='trl': TRL's GRPOTrainer "
            "computes one advantage per completion from the rewards of its prompt group; use "
            "trainer.backend='reference'"
        )

    if algo.advantage_norm == "group":
        scale_rewards = "group" if algo.advantage_std_norm else "none"
    else:
        scale_rewards = algo.advantage_norm

    kwargs = _training_args_kwargs(config, run_dir, device)
    kwargs.update(
        {
            "num_generations": ro.group_size,
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
            "log_completions": False,
            "remove_unused_columns": False,
        }
    )
    if ro.backend == "vllm":
        extra = ro.extra
        server = bool(
            extra.get("server_base_url") or extra.get("server_host") or extra.get("mode") == "server"
        )
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
        raise ConfigError(
            f"trainer.backend='trl' generates with rollout.backend 'hf' or 'vllm', got {ro.backend!r}"
        )
    kwargs.update(importance_sampling_kwargs(config))
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
