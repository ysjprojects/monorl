"""Throughput tooling: find out how to run RL optimally on this hardware.

``python -m monorl.bench`` (also ``monorl bench``) exposes:

* ``rollout`` — :func:`monorl.bench.rollout.run_rollout_benchmark`: engine tokens/s, requests/s,
  p50/p95 latency and peak RSS per batch size, with ``rollout.extra.batch_max_size`` advice;
* ``env`` — :func:`monorl.bench.env.run_env_benchmark`: MEP/1 episodes/s and step latency per
  parallelism level, to size ``parallel_episodes``;
* ``sync`` — :func:`monorl.bench.sync.run_sync_benchmark`: seconds per weight-sync strategy;
* ``plan`` — :func:`monorl.bench.plan.run_plan`: placement plan + step-time estimate.

Every result dataclass has ``to_dict()``; :func:`monorl.bench.report.write_json` serialises it.
Importing this package does not import torch.
"""

from __future__ import annotations

from monorl.bench.env import EnvBenchmark, EnvParallelResult, EnvTarget, run_env_benchmark
from monorl.bench.plan import PlanReport, StepEstimate, run_plan
from monorl.bench.rollout import RolloutBatchResult, RolloutBenchmark, run_rollout_benchmark
from monorl.bench.stats import HostInfo, LatencyStats, percentile
from monorl.bench.sync import SyncBenchmark, SyncResult, run_sync_benchmark

__all__ = [
    "EnvBenchmark",
    "EnvParallelResult",
    "EnvTarget",
    "HostInfo",
    "LatencyStats",
    "PlanReport",
    "RolloutBatchResult",
    "RolloutBenchmark",
    "StepEstimate",
    "SyncBenchmark",
    "SyncResult",
    "percentile",
    "run_env_benchmark",
    "run_plan",
    "run_rollout_benchmark",
    "run_sync_benchmark",
]
