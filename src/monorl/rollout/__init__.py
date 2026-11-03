"""Rollout engines: ``hf`` (transformers reference engine) and the :class:`BatchingEngine` proxy
that coalesces concurrent ``generate`` calls from parallel episodes into batched engine calls.

Importing this package does not import torch/transformers; ``HFRolloutEngine`` is loaded on first
use so the registry can enumerate providers from torch-free processes (server, CLI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monorl.rollout.batching import BatchingEngine, params_key

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.registry import Registry
    from monorl.rollout.hf_engine import HFRolloutEngine

__all__ = ["BatchingEngine", "HFRolloutEngine", "make_hf_engine", "params_key", "register"]


def __getattr__(name: str) -> Any:
    if name == "HFRolloutEngine":
        from monorl.rollout.hf_engine import HFRolloutEngine

        return HFRolloutEngine
    raise AttributeError(name)


def make_hf_engine(config: RunConfig, *, replica: int = 0, **_: Any) -> HFRolloutEngine:
    from monorl.rollout.hf_engine import HFRolloutEngine

    return HFRolloutEngine(config, replica=replica)


def register(registry: Registry) -> None:
    registry.register("rollout_engine", "hf", make_hf_engine)
