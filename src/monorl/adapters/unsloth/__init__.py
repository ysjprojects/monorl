"""Unsloth adapter provider: ``trainer: unsloth`` (see :mod:`monorl.adapters.unsloth.trainer`).

Importing this package registers :data:`PATCHES` (:mod:`monorl.adapters.unsloth.patches`) and
nothing else: unsloth itself is imported by the factory, which raises
:class:`~monorl.core.errors.BackendUnavailable` when CUDA or the package is missing.

Import order: Unsloth patches ``trl``/``transformers``/``peft`` at import, so it should be the
first heavy import of the process when this backend is selected. :mod:`monorl.adapters.unsloth.preload`
does exactly that (``import monorl.adapters.unsloth.preload`` or ``python -m monorl.adapters.unsloth.preload
-- <cmd>``); the factory tolerates a late import (warning; ``trainer.extra.strict_import_order=true``
makes a pre-imported ``trl`` fatal) and :func:`~monorl.adapters.unsloth.patches.reconcile_trl_patches`
re-applies the TRL patchset on Unsloth's regenerated trainer classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monorl.adapters.unsloth.patches import PATCHES, reconcile_trl_patches

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.interfaces import Trainer
    from monorl.core.registry import Registry

FRAMEWORK = "unsloth"
PINNED_VERSION = "2026.10.11"


def is_available() -> bool:
    """unsloth importable *and* CUDA present (Unsloth's kernels are GPU-only)."""
    import importlib.util
    import sys

    if "unsloth" not in sys.modules and importlib.util.find_spec("unsloth") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def _make(config: RunConfig, **kwargs: Any) -> Trainer:
    from monorl.adapters.unsloth.trainer import make_unsloth_trainer

    trainer = make_unsloth_trainer(config, **kwargs)
    reconcile_trl_patches()
    return trainer


def register(registry: Registry) -> None:
    registry.register("trainer", "unsloth", _make)


__all__ = ["FRAMEWORK", "PATCHES", "PINNED_VERSION", "is_available", "reconcile_trl_patches", "register"]
