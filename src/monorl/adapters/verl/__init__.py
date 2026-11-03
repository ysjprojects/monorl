"""verl adapter provider: ``pipeline_backend: verl`` (verl v0.9.1, ``>=0.9,<0.10``).

Pieces (importing this package imports neither verl nor ray):

* :mod:`.compiler` — ``compile_config(RunConfig, run_dir) -> dict`` + ``to_overrides`` (Hydra CLI)
* :mod:`.data` — monorl samples -> verl RL parquet rows
* :mod:`.reward` — generated ``<run_dir>/monorl_reward.py`` (``custom_reward_function``)
* :mod:`.tools` / :mod:`.verl_tool` — tool-calling env -> verl multi-turn tool config + ``BaseTool``
* :mod:`.patches` — :data:`PATCHES` (``Tracking.log`` -> ``metrics.jsonl``, Ray runtime-env vars)
* :mod:`.entry` — ``python -m monorl.adapters.verl.entry`` (verl ``main_ppo`` with patches applied)
* :mod:`.backend` — :class:`VerlBackend` (stage run dir, launch, monitor, cancel, ``RunResult``)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monorl.adapters.verl.backend import VerlBackend, is_available, make_backend
from monorl.adapters.verl.compiler import VERL_COMMIT, VERL_TAG, VERL_VERSIONS, compile_config, to_overrides
from monorl.adapters.verl.patches import PATCHES
from monorl.patching import register_patchset

if TYPE_CHECKING:
    from monorl.core.registry import Registry

FRAMEWORK = "verl"

register_patchset(PATCHES)


def register(registry: Registry) -> None:
    registry.register("pipeline_backend", "verl", make_backend)


__all__ = [
    "FRAMEWORK",
    "PATCHES",
    "VERL_COMMIT",
    "VERL_TAG",
    "VERL_VERSIONS",
    "VerlBackend",
    "compile_config",
    "is_available",
    "register",
    "to_overrides",
]
