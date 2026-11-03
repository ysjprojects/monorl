"""TRL adapter: ``pipeline_backend: trl`` plus the ``trl`` :class:`~monorl.patching.PatchSet`.

Importing this package registers :data:`PATCHES` with the patch engine; it does not import
``trl``/``torch`` models until a backend is instantiated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monorl.adapters.trl.patches import PATCHES

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.registry import Registry


def make_backend(config: RunConfig, **kwargs: Any) -> Any:
    from monorl.adapters.trl.backend import TRLPipelineBackend

    return TRLPipelineBackend(config, **kwargs)


def register(registry: Registry) -> None:
    registry.register("pipeline_backend", "trl", make_backend)


__all__ = ["PATCHES", "make_backend", "register"]
