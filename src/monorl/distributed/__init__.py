"""Distributed execution: runtime assembly, placement planning, Ray launch, weight sync.

Public surface::

    build_runtime(config) -> Runtime            # local or ray, per config.cluster.mode
    plan_placement(config, cluster) -> PlacementPlan
    detect_cluster() -> ClusterInfo
    InProcessWeightSync | CheckpointWeightSync | BroadcastWeightSync | CudaIpcWeightSync

Weight-sync factories are registered under ``weight_sync: inprocess|checkpoint|broadcast|cuda_ipc``
(``factory(config, **kw)``). ``checkpoint`` takes ``sync_dir`` (kw or ``cluster.extra.sync_dir``,
else a fresh temp dir — set a shared path for multi-node runs); ``broadcast`` needs ``spec``
(:class:`BroadcastGroupSpec`) and ``rank`` kwargs supplied by the runtime that owns the group;
``cuda_ipc`` accepts ``device``. Importing this package stays torch-free; the strategy classes
are imported on first attribute access.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

from monorl.distributed.cluster import ClusterInfo, GpuInfo, detect_cluster
from monorl.distributed.placement import ModelEstimate, PlacementPlan, plan_placement
from monorl.distributed.runtime import InProcessWeightSync, Runtime, build_local_runtime, build_runtime

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.registry import Registry

_LAZY = {
    "BroadcastGroupSpec",
    "BroadcastWeightSync",
    "CheckpointWeightSync",
    "CudaIpcWeightSync",
    "IpcHandle",
    "open_ipc_handles",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from monorl.distributed import weight_sync

        return getattr(weight_sync, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def checkpoint_sync_dir(config: RunConfig, sync_dir: str | None = None) -> str:
    return sync_dir or config.cluster.extra.get("sync_dir") or tempfile.mkdtemp(prefix="monorl-weight-sync-")


def _make_checkpoint(config: RunConfig, *, sync_dir: str | None = None, **kw: Any) -> Any:
    from monorl.distributed.weight_sync import CheckpointWeightSync

    return CheckpointWeightSync(checkpoint_sync_dir(config, sync_dir))


def _make_broadcast(
    config: RunConfig, *, spec: Any = None, rank: int = 0, device: Any = None, **kw: Any
) -> Any:
    from monorl.core.errors import WeightSyncError
    from monorl.distributed.weight_sync import BroadcastWeightSync

    if spec is None:
        raise WeightSyncError(
            "weight_sync='broadcast' needs a BroadcastGroupSpec; it is built by the Ray runtime "
            "(cluster.mode='ray'), one process cannot hold both ranks"
        )
    return BroadcastWeightSync(spec, rank=rank, device=device)


def _make_cuda_ipc(config: RunConfig, *, device: Any = None, **kw: Any) -> Any:
    from monorl.distributed.weight_sync import CudaIpcWeightSync

    return CudaIpcWeightSync(device=device)


def register(registry: Registry) -> None:
    registry.register("weight_sync", "inprocess", lambda config, **kw: InProcessWeightSync())
    registry.register("weight_sync", "checkpoint", _make_checkpoint)
    registry.register("weight_sync", "broadcast", _make_broadcast)
    registry.register("weight_sync", "cuda_ipc", _make_cuda_ipc)


__all__ = [
    "BroadcastGroupSpec",
    "BroadcastWeightSync",
    "CheckpointWeightSync",
    "ClusterInfo",
    "CudaIpcWeightSync",
    "GpuInfo",
    "InProcessWeightSync",
    "IpcHandle",
    "ModelEstimate",
    "PlacementPlan",
    "Runtime",
    "build_local_runtime",
    "build_runtime",
    "checkpoint_sync_dir",
    "detect_cluster",
    "open_ipc_handles",
    "plan_placement",
    "register",
]
