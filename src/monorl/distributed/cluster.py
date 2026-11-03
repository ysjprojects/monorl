"""Hardware/cluster detection feeding the placement planner.

:func:`detect_cluster` collects what the placement planner needs: CPU count, CUDA devices with
their memory, MPS availability and — when a Ray driver is connected — the cluster-wide resource
totals and alive node count. The result is a plain dataclass so it can be logged, serialised into
``Runtime.placement`` and constructed by hand in tests.

GPU accounting rule: ``num_gpus`` is the Ray ``GPU`` resource total when Ray is connected **and**
the cluster is either multi-node or the driver itself sees CUDA; otherwise it is the local CUDA
device count. (Ray reports Apple Metal on macOS as a ``GPU`` resource, which no backend here can
train on, so a single-node cluster without CUDA defers to CUDA detection.)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_memory_bytes: int


@dataclass(frozen=True)
class ClusterInfo:
    cpu_count: int
    cuda_devices: tuple[GpuInfo, ...] = ()
    mps_available: bool = False
    ray_connected: bool = False
    ray_resources: dict[str, float] = field(default_factory=dict)
    ray_nodes: int = 0

    @property
    def has_cuda(self) -> bool:
        return bool(self.cuda_devices)

    @property
    def num_gpus(self) -> int:
        if self.ray_connected and (self.ray_nodes > 1 or self.has_cuda):
            return int(self.ray_resources.get("GPU", 0))
        return len(self.cuda_devices)

    @property
    def num_nodes(self) -> int:
        return max(1, self.ray_nodes) if self.ray_connected else 1

    @property
    def gpu_memory_bytes(self) -> int | None:
        """Smallest per-device memory among local CUDA devices (``None`` without CUDA)."""
        if not self.cuda_devices:
            return None
        return min(g.total_memory_bytes for g in self.cuda_devices)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cuda_devices"] = [asdict(g) for g in self.cuda_devices]
        d["num_gpus"] = self.num_gpus
        d["num_nodes"] = self.num_nodes
        return d


def _cuda_devices() -> tuple[GpuInfo, ...]:
    import torch

    if not torch.cuda.is_available():
        return ()
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devices.append(GpuInfo(index=i, name=props.name, total_memory_bytes=int(props.total_memory)))
    return tuple(devices)


def _mps_available() -> bool:
    import torch

    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _ray_state() -> tuple[bool, dict[str, float], int]:
    try:
        import ray
    except ImportError:
        return False, {}, 0
    if not ray.is_initialized():
        return False, {}, 0
    resources = {k: float(v) for k, v in ray.cluster_resources().items()}
    nodes = sum(1 for n in ray.nodes() if n.get("Alive"))
    return True, resources, nodes


def detect_cluster(*, use_ray: bool = True) -> ClusterInfo:
    """Inspect this process' hardware and, if ``use_ray`` and a Ray driver is connected, the cluster.

    Never initialises Ray itself; :func:`monorl.distributed.ray_runtime.build_ray_runtime` does that
    before calling this.
    """
    ray_connected, ray_resources, ray_nodes = _ray_state() if use_ray else (False, {}, 0)
    return ClusterInfo(
        cpu_count=os.cpu_count() or 1,
        cuda_devices=_cuda_devices(),
        mps_available=_mps_available(),
        ray_connected=ray_connected,
        ray_resources=ray_resources,
        ray_nodes=ray_nodes,
    )
