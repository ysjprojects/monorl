"""NeMo-RL adapter provider: ``pipeline_backend: nemo_rl``.

NeMo-RL (https://github.com/NVIDIA-NeMo/RL) is git-only; this adapter targets tag **v0.7.0**
(commit ``81aa43dda4765b0429cf31dab44441e4e4383911``) and never imports ``nemo_rl``/``ray`` on the
monorl side. The pieces:

* :mod:`.compile` — ``RunConfig`` → NeMo-RL master yaml (``examples/configs/grpo_math_1B.yaml`` shape).
* :mod:`.data` — monorl samples → jsonl + ``MonorlDataset``/``monorl_data_processor`` (``DatumSpec``).
* :mod:`.env_bridge` / :mod:`.env_actor` — ``EnvironmentInterface`` over monorl envs + rewards.
* :mod:`.patches` — ``PATCHES`` (``Logger.log_metrics`` → ``metrics.jsonl``).
* :mod:`.entry` — in-environment wrapper around ``examples/run_grpo.py``.
* :mod:`.backend` — the subprocess launcher implementing :class:`~monorl.core.interfaces.PipelineBackend`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monorl.adapters.nemo_rl.backend import NemoRLBackend, make_backend
from monorl.adapters.nemo_rl.compile import compile_config
from monorl.adapters.nemo_rl.patches import DISTRIBUTION, FRAMEWORK, PATCHES
from monorl.patching import register_patchset

if TYPE_CHECKING:
    from monorl.core.registry import Registry

#: Upstream tag/commit the adapter is written against.
TARGET_TAG = "v0.7.0"
TARGET_COMMIT = "81aa43dda4765b0429cf31dab44441e4e4383911"

register_patchset(PATCHES)


def is_available() -> bool:
    """True when ``nemo_rl`` is importable in *this* interpreter (the launcher only needs a checkout)."""
    return PATCHES.installed_version() is not None


def register(registry: Registry) -> None:
    registry.register("pipeline_backend", "nemo_rl", make_backend)


__all__ = [
    "DISTRIBUTION",
    "FRAMEWORK",
    "PATCHES",
    "TARGET_COMMIT",
    "TARGET_TAG",
    "NemoRLBackend",
    "compile_config",
    "is_available",
    "make_backend",
    "register",
]
