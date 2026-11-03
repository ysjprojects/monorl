"""SGLang adapter provider: ``rollout_engine: sglang`` plus the ``sglang`` :class:`PatchSet`.

Importing this package never imports sglang. The factory picks the colocated in-process
:class:`~monorl.adapters.sglang.engine.SGLangEngine` (default) or the HTTP
:class:`~monorl.adapters.sglang.client.SGLangServerClient` when
``rollout.extra.mode == "server"``; the colocated engine raises
:class:`~monorl.core.errors.BackendUnavailable` when sglang is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monorl.adapters.sglang.common import FRAMEWORK, SGLANG_VERSIONS
from monorl.adapters.sglang.patches import PATCHES
from monorl.core.errors import ConfigError

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.registry import Registry

__all__ = ["FRAMEWORK", "PATCHES", "SGLANG_VERSIONS", "is_available", "make_engine", "register"]

MODES: tuple[str, ...] = ("colocate", "server")


def is_available() -> bool:
    try:
        import sglang  # noqa: F401
    except ImportError:
        return False
    return True


def make_engine(config: RunConfig, replica: int = 0, **kwargs: Any) -> Any:
    """``rollout_engine: sglang`` factory. ``rollout.extra.mode``: ``colocate`` (default) or
    ``server``; extra keyword arguments are forwarded to the engine class."""
    mode = str(config.rollout.extra.get("mode", "colocate"))
    if mode == "server":
        from monorl.adapters.sglang.client import SGLangServerClient

        return SGLangServerClient(config, replica=replica, **kwargs)
    if mode != "colocate":
        raise ConfigError(f"rollout.extra.mode must be one of {MODES}, got {mode!r}")
    from monorl.adapters.sglang.engine import SGLangEngine

    return SGLangEngine(config, replica=replica, **kwargs)


def register(registry: Registry) -> None:
    registry.register("rollout_engine", "sglang", make_engine)
