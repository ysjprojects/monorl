"""Orchestrator: ``run(config)`` drives data → parallel env episodes → rewards → advantages →
trainer → weight sync, or hands the whole run to a pipeline backend (trl/verl/nemo_rl).

Only :func:`run` and :func:`resolve_run_dir` are contract; the rest is implementation shared
with tests.
"""

from __future__ import annotations

from monorl.orchestrator.batching import build_train_batch, flatten_trajectory
from monorl.orchestrator.episodes import EpisodeSpec, chat_prompt_ids, run_episode, run_episodes
from monorl.orchestrator.loop import ComponentLoop
from monorl.orchestrator.run import default_run_name, resolve_run_dir, run
from monorl.orchestrator.sampler import PromptSampler

__all__ = [
    "ComponentLoop",
    "EpisodeSpec",
    "PromptSampler",
    "build_train_batch",
    "chat_prompt_ids",
    "default_run_name",
    "flatten_trajectory",
    "resolve_run_dir",
    "run",
    "run_episode",
    "run_episodes",
]
