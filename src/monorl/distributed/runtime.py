"""Runtime assembly: turns a :class:`RunConfig` into live component instances.

The orchestrator only ever sees protocol objects; whether they live in-process or behind Ray
actors on other nodes is decided here. ``cluster.mode == "local"`` builds everything in-process
(this file); ``"ray"`` is provided by :mod:`monorl.distributed.ray_runtime`. Both consult
:func:`monorl.distributed.placement.plan_placement` and expose the resolved plan as
``Runtime.placement``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from monorl.core.registry import ensure_providers, registry

if TYPE_CHECKING:
    import torch

    from monorl.config.schema import RunConfig
    from monorl.core.interfaces import RolloutEngine, Trainer, WeightSync


@dataclass
class Runtime:
    engines: list[RolloutEngine]
    trainer: Trainer
    weight_sync: WeightSync
    placement: dict[str, Any] = field(default_factory=dict)
    """Resolved placement decisions (for logging/UI): mode, devices, gpus per role, sync strategy."""
    closers: list[Callable[[], None]] = field(default_factory=list)
    """Extra teardown steps run after engine shutdown (Ray actors, placement groups, sync groups)."""

    def close(self) -> None:
        for e in self.engines:
            try:
                e.shutdown()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        for closer in self.closers:
            try:
                closer()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass


class InProcessWeightSync:
    """Direct tensor hand-off: ``trainer.export_weights()`` -> ``engine.update_weights``."""

    name = "inprocess"

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        t0 = time.perf_counter()
        counted = [0]

        def stream() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in trainer.export_weights():
                counted[0] += tensor.numel() * tensor.element_size()
                yield name, tensor

        for engine in engines:
            engine.update_weights(stream())
        return {"weight_sync/bytes": float(counted[0]), "weight_sync/seconds": time.perf_counter() - t0}


def build_local_runtime(config: RunConfig) -> Runtime:
    from monorl.distributed.cluster import detect_cluster
    from monorl.distributed.placement import plan_placement

    plan = plan_placement(config, detect_cluster(use_ray=False))
    engine_factory = registry.get("rollout_engine", config.rollout.backend)
    trainer_factory = registry.get("trainer", config.trainer.backend)
    engines = [engine_factory(config, replica=i) for i in range(plan.engine_replicas)]
    trainer = trainer_factory(config)
    weight_sync: WeightSync
    if plan.weight_sync == "inprocess":
        weight_sync = InProcessWeightSync()
    else:
        weight_sync = registry.get("weight_sync", plan.weight_sync)(config)
    return Runtime(
        engines=engines,
        trainer=trainer,
        weight_sync=weight_sync,
        placement={
            **plan.to_dict(),
            "rollout_backend": config.rollout.backend,
            "trainer_backend": config.trainer.backend,
            "weight_sync": weight_sync.name,
        },
    )


def build_runtime(config: RunConfig) -> Runtime:
    ensure_providers(config)
    if config.cluster.mode == "local":
        return build_local_runtime(config)
    from monorl.distributed.ray_runtime import build_ray_runtime

    return build_ray_runtime(config)
