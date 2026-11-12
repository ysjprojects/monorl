"""Versioned monkeypatches for verl (``>=0.9,<0.10``; written against tag v0.9.1).

* ``verl.tracking.run_dir`` — wraps ``verl.utils.tracking.Tracking.__init__`` to remember where
  monorl's run directory is (``MONORL_RUN_DIR`` env, else the parent of
  ``trainer.default_local_dir`` from the config verl passes in).
* ``verl.tracking.metrics_jsonl`` — wraps ``verl.utils.tracking.Tracking.log(data, step, backend=None)``
  (``verl/utils/tracking.py``) to also append a contract-named row to ``<run_dir>/metrics.jsonl``
  (see :mod:`monorl.core.reporting`), so the monorl server/UI can follow a verl run without
  importing verl. The ``Tracking`` object lives inside verl's ``TaskRunner`` Ray actor
  (``verl/trainer/main_ppo.py``), which is why :mod:`monorl.adapters.verl.entry` re-applies the
  patchset inside the actor.
* ``verl.ray_runtime_env.monorl`` — wraps ``verl.trainer.main_ppo.get_ppo_ray_runtime_env``
  (``verl/trainer/constants_ppo.py``) so ``MONORL_*`` variables and ``PYTHONPATH`` reach every
  Ray worker, including on pre-existing clusters where workers do not inherit the driver env.
"""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path
from typing import Any

from monorl.core.reporting import METRICS_FILE
from monorl.patching import Patch, PatchSet

FRAMEWORK = "verl"
VERSIONS = ">=0.9,<0.10"
RUN_DIR_ENV = "MONORL_RUN_DIR"
FORWARDED_ENV_VARS = (RUN_DIR_ENV, "MONORL_DISABLE_PATCHES", "PYTHONPATH")

# verl metric name (verl/trainer/ppo/metric_utils.py, verl/workers/utils/losses.py,
# verl/trainer/ppo/v1/trainer_base.py timing_raw) -> monorl contract name (docs/ARCHITECTURE.md).
METRIC_MAP: dict[str, str] = {
    "critic/score/mean": "reward/mean",
    "critic/score/std": "reward/std",
    "actor/pg_loss": "loss/policy",
    "actor/kl_loss": "loss/kl",
    "actor/entropy": "loss/entropy",
    "actor/pg_clipfrac": "policy/clipfrac",
    "actor/ppo_kl": "policy/approx_kl",
    "response_length/mean": "tokens/response_mean",
    "prompt_length/mean": "tokens/prompt_mean",
    "actor/lr": "lr",
    "timing_s/gen": "time/rollout",
    "timing_s/update_actor": "time/train",
}
# verl/trainer/sft_trainer.py (and sft_trainer_ray.py) ``train/*`` + ``val/loss`` -> contract names.
SFT_METRIC_MAP: dict[str, str] = {
    "train/loss": "loss/sft",
    "train/lr": "lr",
    "train/grad_norm": "train/grad_norm",
    "train/global_tokens": "tokens/total",
    "val/loss": "eval/loss",
}


def map_metrics(data: dict[str, Any]) -> dict[str, float]:
    """Contract-named metrics derived from one verl ``Tracking.log`` payload (numeric values only).

    Mapped keys are added under their contract name; every numeric verl key is kept as-is too, and
    any other ``timing_s/<stage>`` becomes ``time/<stage>``.
    """
    out: dict[str, float] = {}
    for key, value in data.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        name = METRIC_MAP.get(key) or SFT_METRIC_MAP.get(key)
        if name is not None:
            out[name] = number
        elif key.startswith("timing_s/"):
            out["time/" + key[len("timing_s/") :]] = number
        out.setdefault(key, number)
    return out


def phase_for(data: dict[str, Any]) -> str:
    keys = [k for k in data if isinstance(k, str)]
    return "eval" if keys and all(k.startswith("val") for k in keys) else "train"


def append_metrics_row(run_dir: str | Path, step: int, data: dict[str, Any]) -> dict[str, Any]:
    """Append one ``metrics.jsonl`` row (same schema as ``RunReporter.log_metrics``) and return it."""
    row: dict[str, Any] = {"step": int(step), "time": time.time(), "phase": phase_for(data)}
    row.update(map_metrics(data))
    path = Path(run_dir) / METRICS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def resolve_run_dir(config: Any = None) -> str | None:
    env = os.environ.get(RUN_DIR_ENV)
    if env:
        return env
    try:
        local_dir = config["trainer"]["default_local_dir"] if config is not None else None
    except (KeyError, TypeError, IndexError):
        local_dir = None
    return str(Path(local_dir).parent) if local_dir else None


def _build_init(original: Any) -> Any:
    @functools.wraps(original)
    def __init__(self: Any, project_name: Any, experiment_name: Any, default_backend: Any = "console", config: Any = None) -> None:
        self._monorl_run_dir = resolve_run_dir(config)
        original(self, project_name, experiment_name, default_backend, config)

    return __init__


def _build_log(original: Any) -> Any:
    @functools.wraps(original)
    def log(self: Any, data: Any, step: Any, backend: Any = None) -> Any:
        result = original(self, data, step, backend)
        run_dir = getattr(self, "_monorl_run_dir", None) or resolve_run_dir()
        if run_dir and isinstance(data, dict):
            append_metrics_row(run_dir, int(step), data)
        return result

    return log


def _build_runtime_env(original: Any) -> Any:
    @functools.wraps(original)
    def get_ppo_ray_runtime_env(*args: Any, **kwargs: Any) -> dict[str, Any]:
        runtime_env = original(*args, **kwargs)
        env_vars = runtime_env.setdefault("env_vars", {})
        for key in FORWARDED_ENV_VARS:
            value = os.environ.get(key)
            if value:
                env_vars[key] = value
        return runtime_env

    return get_ppo_ray_runtime_env


def _has_log_signature(original: Any) -> bool:
    import inspect

    try:
        params = list(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        return False
    return params[:3] == ["self", "data", "step"]


PATCHES = PatchSet(
    framework=FRAMEWORK,
    patches=[
        Patch(
            id="verl.tracking.run_dir",
            target="verl.utils.tracking.Tracking.__init__",
            versions=VERSIONS,
            mode="wrap",
            build=_build_init,
            description="Remember monorl's run directory on the verl Tracking logger.",
        ),
        Patch(
            id="verl.tracking.metrics_jsonl",
            target="verl.utils.tracking.Tracking.log",
            versions=VERSIONS,
            mode="wrap",
            build=_build_log,
            verify=_has_log_signature,
            description="Mirror verl metrics into <run_dir>/metrics.jsonl under monorl's metric names.",
        ),
        Patch(
            id="verl.ray_runtime_env.monorl",
            target="verl.trainer.main_ppo.get_ppo_ray_runtime_env",
            versions=VERSIONS,
            mode="wrap",
            build=_build_runtime_env,
            description="Forward MONORL_RUN_DIR / PYTHONPATH to verl's Ray workers.",
        ),
    ],
)
