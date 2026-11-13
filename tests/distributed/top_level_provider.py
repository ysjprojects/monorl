"""A provider module named in ``RunConfig.providers`` (not in ``cluster.extra.providers``)."""

from __future__ import annotations

from typing import Any

from distributed_fakes import FakeEngine, FakeTrainer


def register(registry: Any) -> None:
    registry.register("rollout_engine", "provided", lambda config, **kw: FakeEngine(0, **kw))
    registry.register("trainer", "provided", lambda config, **kw: FakeTrainer(1, **kw))
