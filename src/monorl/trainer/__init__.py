"""Trainers: ``reference`` (plain torch/transformers, full or LoRA).

Importing this package does not import torch; :class:`ReferenceTrainer` is loaded on first use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.registry import Registry
    from monorl.trainer.reference import ReferenceTrainer

__all__ = ["ReferenceTrainer", "make_reference_trainer", "register"]


def __getattr__(name: str) -> Any:
    if name == "ReferenceTrainer":
        from monorl.trainer.reference import ReferenceTrainer

        return ReferenceTrainer
    raise AttributeError(name)


def make_reference_trainer(config: RunConfig, **kwargs: Any) -> ReferenceTrainer:
    """``loss_fn``, ``rank`` and ``world_size`` may be injected (the Ray runtime passes the latter
    two per :class:`~monorl.distributed.ray_runtime.TrainerActor`); everything else comes from ``config``."""
    from monorl.trainer.reference import ReferenceTrainer

    return ReferenceTrainer(
        config, loss_fn=kwargs.get("loss_fn"), rank=kwargs.get("rank"), world_size=kwargs.get("world_size")
    )


def register(registry: Registry) -> None:
    registry.register("trainer", "reference", make_reference_trainer)
