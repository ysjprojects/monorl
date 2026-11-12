"""Versioned monkeypatches for NeMo-RL (``nemo_rl``, distribution ``nemo-rl``).

``nemo_rl.logger_metrics`` wraps ``nemo_rl.utils.logger.Logger.log_metrics`` (NVIDIA-NeMo/RL v0.7.0,
``81aa43dda4765b0429cf31dab44441e4e4383911``; signature
``log_metrics(self, metrics, step, prefix="", step_metric=None, step_finished=False)``) and appends a
contract-named row to ``<MONORL_RUN_DIR>/metrics.jsonl`` after every upstream call. ``grpo_train``
(``nemo_rl/algorithms/grpo.py``) logs with prefixes ``train`` (loss/reward/lr/kl_penalty/…, plus the
rollout metrics such as ``mean_gen_tokens_per_sample``), ``timing/train`` (``Timer`` labels),
``performance``, ``validation``/``timing/validation`` and ``timing/setup``.
"""

from __future__ import annotations

import functools
import inspect
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from monorl.core.reporting import METRICS_FILE
from monorl.patching import Patch, PatchSet

FRAMEWORK = "nemo_rl"
DISTRIBUTION = "nemo-rl"
VERSIONS = ">=0.3,<1"

#: Environment variable naming the monorl run directory (set by the launcher / entry wrapper).
RUN_DIR_ENV = "MONORL_RUN_DIR"
#: ``algorithm.name`` of the run (set by the launcher); ``sft`` renames ``train/loss`` → ``loss/sft``.
ALGORITHM_ENV = "MONORL_ALGORITHM"

#: ``nemo_rl/algorithms/sft.py`` ``train/<key>`` → monorl metric name.
SFT_TRAIN_METRICS: dict[str, str] = {
    "loss": "loss/sft",
    "global_valid_toks": "tokens/valid",
    "global_valid_seqs": "sequences/valid",
    "grad_norm": "grad_norm",
    "lr": "lr",
    "wd": "weight_decay",
}
#: ``validation/<key>`` → ``eval/*`` for SFT (``val_loss`` → ``eval/loss``).
SFT_VALIDATION_METRICS: dict[str, str] = {"val_loss": "eval/loss"}

#: ``train/<key>`` → monorl metric name (see ``docs/ARCHITECTURE.md`` metric names).
TRAIN_METRICS: dict[str, str] = {
    "reward": "reward/mean",
    "filtered_reward": "reward/filtered_mean",
    "loss": "loss/policy",
    "kl_penalty": "loss/kl",
    "entropy": "loss/entropy",
    "approx_entropy": "loss/entropy",
    "token_mult_prob_error": "policy/token_mult_prob_error",
    "probs_ratio": "policy/ratio_mean",
    "probs_ratio_clamped": "policy/ratio_clamped_mean",
    "sampling_importance_ratio": "policy/sampling_importance_ratio",
    "mean_response_length": "tokens/response_mean",
    "mean_gen_tokens_per_sample": "tokens/response_mean",
    "mean_prompt_length": "tokens/prompt_mean",
    "global_valid_toks": "tokens/valid",
    "global_valid_seqs": "sequences/valid",
    "num_valid_samples": "sequences/valid_samples",
    "grad_norm": "grad_norm",
    "lr": "lr",
    "wd": "weight_decay",
}

#: ``timing/train/<key>`` → ``time/*``.
TIMING_METRICS: dict[str, str] = {
    "total_step_time": "time/step",
    "generation": "time/rollout",
    "exposed_generation": "time/rollout",
    "policy_training": "time/train",
    "weight_sync": "time/sync",
    "prepare_for_generation/total": "time/sync",
    "prepare_for_generation/transfer_and_update_weights": "time/sync_transfer",
    "policy_and_reference_logprobs": "time/logprobs",
    "reward_calculation": "time/reward",
    "advantage_calculation": "time/advantage",
    "data_processing": "time/data",
    "checkpointing": "time/checkpoint",
}

#: ``validation/<key>`` → ``eval/*``.
VALIDATION_METRICS: dict[str, str] = {"accuracy": "eval/accuracy", "reward": "eval/reward_mean"}


def _as_float(value: Any) -> float | None:
    """Finite scalar → float; arrays/tensors only when they hold exactly one element."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    item = getattr(value, "item", None)
    if not callable(item):
        return None
    numel = getattr(value, "numel", None)
    if callable(numel):
        count = numel()
    else:
        size = getattr(value, "size", None)
        count = size if isinstance(size, int) else 1
    if count != 1:
        return None
    try:
        return _as_float(item())
    except (TypeError, ValueError, RuntimeError):
        return None


def phase_for(prefix: str) -> str:
    p = (prefix or "").strip("/")
    if p.startswith("validation") or p.startswith("timing/validation"):
        return "eval"
    if p == "timing/setup":
        return "setup"
    return "train"


def translate_metrics(metrics: dict[str, Any], prefix: str | None = "", *, algorithm: str | None = None) -> dict[str, float]:
    """Numeric ``metrics`` logged under NeMo-RL ``prefix`` → contract-named ``{name: float}``.

    Unknown keys are kept as ``nemo/<prefix>/<key>`` so nothing numeric is lost. ``algorithm``
    (default ``$MONORL_ALGORITHM``) selects the SFT names for ``train``/``validation`` payloads.
    """
    p = (prefix or "").strip("/")
    sft = (algorithm or os.environ.get(ALGORITHM_ENV)) == "sft"
    train_map = SFT_TRAIN_METRICS if sft else TRAIN_METRICS
    validation_map = SFT_VALIDATION_METRICS if sft else VALIDATION_METRICS
    out: dict[str, float] = {}
    for key, value in metrics.items():
        number = _as_float(value)
        if number is None:
            continue
        if p == "train":
            name = train_map.get(key, f"nemo/train/{key}")
        elif p == "timing/train":
            name = TIMING_METRICS.get(key, f"time/{key}")
        elif p == "validation":
            name = validation_map.get(key, f"eval/{key}")
        elif p == "timing/validation":
            name = f"time/eval/{key}"
        elif p == "timing/setup":
            name = f"time/setup/{key}"
        elif p == "performance":
            name = f"perf/{key}"
        elif p:
            name = f"nemo/{p}/{key}"
        else:
            name = f"nemo/{key}"
        out[name] = number
    return out


def metrics_path(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir) / METRICS_FILE


def append_metrics_row(run_dir: str | os.PathLike[str], step: int, metrics: dict[str, Any], prefix: str | None) -> dict[str, Any] | None:
    """Translate and append one row to ``<run_dir>/metrics.jsonl``; returns the row (or ``None`` if empty)."""
    translated = translate_metrics(metrics, prefix)
    if not translated:
        return None
    row: dict[str, Any] = {"step": int(step), "time": time.time(), "phase": phase_for(prefix or ""), **translated}
    path = metrics_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _bind_prefix(original: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any, Any]:
    """``(metrics, step, prefix)`` from the upstream call arguments."""
    try:
        bound = inspect.signature(original).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        return a.get("metrics"), a.get("step"), a.get("prefix", "")
    except (TypeError, ValueError):
        metrics = kwargs.get("metrics", args[1] if len(args) > 1 else {})
        step = kwargs.get("step", args[2] if len(args) > 2 else 0)
        prefix = kwargs.get("prefix", args[3] if len(args) > 3 else "")
        return metrics, step, prefix


def build_log_metrics(original: Any) -> Any:
    """``wrap`` builder: call upstream, then mirror the row into monorl's ``metrics.jsonl``."""

    @functools.wraps(original)
    def log_metrics(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        run_dir = os.environ.get(RUN_DIR_ENV)
        if run_dir:
            metrics, step, prefix = _bind_prefix(original, (self, *args), kwargs)
            if isinstance(metrics, dict):
                append_metrics_row(run_dir, int(step or 0), metrics, prefix)
        return result

    return log_metrics


def _verify_log_metrics(original: Any) -> bool:
    try:
        params = inspect.signature(original).parameters
    except (TypeError, ValueError):
        return False
    return {"metrics", "step", "prefix"} <= set(params)


PATCHES = PatchSet(
    framework=FRAMEWORK,
    distribution=DISTRIBUTION,
    patches=[
        Patch(
            id="nemo_rl.logger_metrics",
            target="nemo_rl.utils.logger.Logger.log_metrics",
            versions=VERSIONS,
            mode="wrap",
            build=build_log_metrics,
            verify=_verify_log_metrics,
            description="Mirror Logger.log_metrics rows into <MONORL_RUN_DIR>/metrics.jsonl with monorl metric names.",
        )
    ],
)

__all__ = [
    "ALGORITHM_ENV",
    "DISTRIBUTION",
    "FRAMEWORK",
    "PATCHES",
    "RUN_DIR_ENV",
    "TIMING_METRICS",
    "TRAIN_METRICS",
    "VALIDATION_METRICS",
    "VERSIONS",
    "append_metrics_row",
    "build_log_metrics",
    "phase_for",
    "translate_metrics",
]
