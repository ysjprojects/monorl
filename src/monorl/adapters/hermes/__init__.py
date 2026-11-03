"""Hermes Agent adapter provider: ``env: hermes`` (see :mod:`monorl.adapters.hermes.env`).

Importing this package does not import hermes-agent; the factory does, raising
:class:`~monorl.core.errors.BackendUnavailable` when the extra is missing. ``PATCHES`` is the
(empty) hermes :class:`~monorl.patching.PatchSet`, registered with the patch engine at import.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from monorl.adapters.hermes.env import HermesEnv, make_hermes_env
from monorl.adapters.hermes.patches import PATCHES
from monorl.patching import register_patchset

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.registry import Registry

FRAMEWORK = "hermes"

register_patchset(PATCHES)


def is_available() -> bool:
    return importlib.util.find_spec("run_agent") is not None


def _make(config: RunConfig, **kwargs: Any) -> HermesEnv:
    return make_hermes_env(
        config.env.params,
        max_turns=config.env.max_turns,
        system_prompt=config.data.system_prompt,
        model_name=config.model.name_or_path,
    )


def register(registry: Registry) -> None:
    registry.register("env", "hermes", _make)


__all__ = ["FRAMEWORK", "HermesEnv", "PATCHES", "is_available", "make_hermes_env", "register"]
