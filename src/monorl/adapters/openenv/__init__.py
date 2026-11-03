"""OpenEnv adapter provider: ``env: openenv`` (see :mod:`monorl.adapters.openenv.env`).

Importing this package does not import openenv-core; the factory does, raising
:class:`~monorl.core.errors.BackendUnavailable` when the extra is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monorl.adapters.openenv.env import OpenEnvEnv, make_openenv_env

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.registry import Registry

FRAMEWORK = "openenv"


def is_available() -> bool:
    try:
        import openenv.core  # noqa: F401
    except ImportError:
        return False
    return True


def _make(config: RunConfig, **kwargs: Any) -> OpenEnvEnv:
    return make_openenv_env(
        config.env.params,
        max_turns=config.env.max_turns,
        system_prompt=config.data.system_prompt,
        seed=kwargs.get("seed"),
    )


def register(registry: Registry) -> None:
    registry.register("env", "openenv", _make)


__all__ = ["FRAMEWORK", "OpenEnvEnv", "is_available", "make_openenv_env", "register"]
