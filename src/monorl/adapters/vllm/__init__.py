"""vLLM adapter: ``rollout_engine: vllm`` (colocate or server mode) + patches.

Importing this package never imports vLLM; the factory does, and raises
:class:`~monorl.core.errors.BackendUnavailable` with install instructions when it is missing.
``rollout.extra.mode`` selects ``"colocate"`` (default, in-process ``vllm.LLM``) or ``"server"``
(HTTP client for :mod:`monorl.adapters.vllm.server`).
"""

from __future__ import annotations

from typing import Any

from monorl.adapters.vllm.patches import PATCHES
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.patching import register_patchset

register_patchset(PATCHES)

MODES = ("colocate", "server")


def make_engine(config: RunConfig, replica: int = 0, **kwargs: Any) -> Any:
    mode = str(config.rollout.extra.get("mode", "colocate"))
    if mode == "server":
        from monorl.adapters.vllm.client import VllmServerClient

        return VllmServerClient(config, replica=replica, **kwargs)
    if mode == "colocate":
        from monorl.adapters.vllm.engine import VllmEngine

        return VllmEngine(config, replica=replica, **kwargs)
    raise ConfigError(f"rollout.extra.mode must be one of {MODES}, got {mode!r}")


def register(registry: Any) -> None:
    registry.register("rollout_engine", "vllm", make_engine)


__all__ = ["PATCHES", "make_engine", "register"]
