from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from monorl.config.schema import RunConfig  # noqa: E402
from monorl.distributed.cluster import ClusterInfo, GpuInfo  # noqa: E402

GIB = 2**30


def gpu_cluster(
    num_gpus: int,
    *,
    memory_gib: int = 80,
    ray_nodes: int = 0,
    ray_gpus: int | None = None,
) -> ClusterInfo:
    devices = tuple(GpuInfo(index=i, name="fake-gpu", total_memory_bytes=memory_gib * GIB) for i in range(num_gpus))
    connected = ray_nodes > 0
    return ClusterInfo(
        cpu_count=16,
        cuda_devices=devices,
        mps_available=False,
        ray_connected=connected,
        ray_resources={"GPU": float(ray_gpus if ray_gpus is not None else num_gpus), "CPU": 16.0} if connected else {},
        ray_nodes=ray_nodes,
    )


@pytest.fixture
def make_config(tiny_model_dir: Path, tmp_path: Path):
    def _make(**overrides: Any) -> RunConfig:
        payload: dict[str, Any] = {
            "run_name": "dist-test",
            "model": {"name_or_path": str(tiny_model_dir), "dtype": "float32"},
            "data": {"uri": str(tmp_path / "data.jsonl")},
            "rollout": {"backend": "fake", "group_size": 2, "max_new_tokens": 4},
            "trainer": {"backend": "fake", "prompts_per_step": 2, "max_steps": 1},
            "rewards": [{"type": "exact_match"}],
            "logging": {"run_root": str(tmp_path / "runs")},
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(payload.get(key), dict):
                payload[key] = {**payload[key], **value}
            else:
                payload[key] = value
        return RunConfig.from_dict(payload)

    return _make
