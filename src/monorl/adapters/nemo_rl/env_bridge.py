"""Environment bridge: NeMo-RL ``EnvironmentInterface`` driven by monorl environments and rewards.

Targets ``nemo_rl/environments/interfaces.py`` of NVIDIA-NeMo/RL **v0.7.0**
(``81aa43dda4765b0429cf31dab44441e4e4383911``)::

    step(message_log_batch: list[list[{"role", "content"}]], metadata: list[dict]) -> EnvironmentReturn
    global_post_process_and_metrics(batch) -> (batch, metrics)

``nemo_rl/experience/rollouts.py::calculate_rewards`` calls ``step`` once per turn with the
role/content view of every active sample's message log and its ``extra_env_info`` (our
``DatumSpec`` metadata). The returned ``observations[i]`` is tokenised and appended to the log,
``rewards`` accumulate into ``total_reward`` and ``terminateds`` end episodes; ``metadata`` is handed
back on the next turn, which is how we track a live monorl ``Environment`` per episode.

:class:`MonorlEnvironmentBridge` is plain Python (unit-testable without Ray); the Ray actor
registered with NeMo-RL is :data:`monorl.adapters.nemo_rl.env_actor.MonorlEnvironment`.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch

from monorl.adapters.nemo_rl.compile import ACTOR_FQN, build_env_config
from monorl.config.schema import RunConfig
from monorl.core.interfaces import Environment, RewardFunction
from monorl.core.types import Completion, Message, Sample, Trajectory, Turn

try:  # the real NamedTuple when NeMo-RL is importable; identical field layout otherwise
    from nemo_rl.environments.interfaces import EnvironmentReturn
except ImportError:

    class EnvironmentReturn(NamedTuple):  # type: ignore[no-redef]
        observations: list[dict[str, str]]
        metadata: list[Any]
        next_stop_strings: list[list[str] | None] | list[None]
        rewards: torch.Tensor
        terminateds: torch.Tensor
        answers: list[str | None] | None


#: Metadata key carrying the live episode id between turns.
EPISODE_KEY = "monorl_episode"
#: Metadata key with the reward breakdown of a finished episode.
BREAKDOWN_KEY = "monorl_reward_breakdown"
#: Role of the terminal (empty) observation; NeMo-RL tokenises ``content`` and appends it.
TERMINAL_ROLE = "environment"


@dataclass(slots=True)
class _Episode:
    env: Environment
    sample: Sample
    turns: list[Turn] = field(default_factory=list)


def _latest_assistant(message_log: Sequence[dict[str, Any]]) -> str:
    for m in reversed(message_log):
        if m.get("role") == "assistant":
            return str(m.get("content", ""))
    return ""


def _role_content(message_log: Sequence[dict[str, Any]]) -> list[Message]:
    return [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in message_log]  # type: ignore[misc]


class MonorlEnvironmentBridge:
    """One monorl ``Environment`` per episode plus the configured reward functions.

    ``cfg`` is :func:`build_env_config`. Episodes are keyed by ``metadata[EPISODE_KEY]``; a batch item
    without one starts a fresh episode (``make_env`` + ``reset``). Each turn feeds the latest
    assistant message to ``Environment.step``; when the episode ends the monorl reward function
    scores the full trajectory and its score is added to the final turn's reward.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.config = RunConfig.from_dict(cfg["run_config"])
        specs = cfg.get("rewards")
        self.reward_fn: RewardFunction | None = None
        if specs:
            from monorl.rewards import build_reward

            self.reward_fn = build_reward(specs, config=self.config)
        self._episodes: dict[str, _Episode] = {}
        self._lock = threading.Lock()

    # -- episode bookkeeping ---------------------------------------------------------------
    def _episode_for(self, meta: dict[str, Any]) -> tuple[str, _Episode]:
        from monorl.envs import make_env

        episode_id = meta.get(EPISODE_KEY)
        with self._lock:
            if episode_id is not None and episode_id in self._episodes:
                return episode_id, self._episodes[episode_id]
        sample = Sample(
            id=str(meta.get("sample_id", "")),
            messages=list(meta.get("messages") or []),
            reference=meta.get("reference"),
            metadata=dict(meta.get("metadata") or {}),
        )
        env = make_env(self.config)
        env.reset(sample)
        episode = _Episode(env=env, sample=sample)
        episode_id = uuid.uuid4().hex
        with self._lock:
            self._episodes[episode_id] = episode
        return episode_id, episode

    def _finish(self, episode_id: str) -> None:
        with self._lock:
            episode = self._episodes.pop(episode_id, None)
        if episode is not None:
            episode.env.close()

    @property
    def num_live_episodes(self) -> int:
        with self._lock:
            return len(self._episodes)

    # -- EnvironmentInterface --------------------------------------------------------------
    def step(self, message_log_batch: list[list[dict[str, Any]]], metadata: list[dict[str, Any]]) -> EnvironmentReturn:
        observations: list[dict[str, str]] = []
        out_metadata: list[dict[str, Any]] = []
        rewards: list[float] = []
        terminateds: list[float] = []
        finished: list[tuple[int, Trajectory]] = []

        for i, (log, meta) in enumerate(zip(message_log_batch, metadata, strict=True)):
            meta = dict(meta or {})
            episode_id, episode = self._episode_for(meta)
            meta[EPISODE_KEY] = episode_id
            completion = Completion(text=_latest_assistant(log), token_ids=[], finish_reason="stop")
            result = episode.env.step(completion)
            episode.turns.append(Turn(prompt_token_ids=[], completion=completion, reward=float(result.reward), info=dict(result.info)))
            done = bool(result.done) or len(episode.turns) >= int(episode.env.max_turns)
            if done:
                trajectory = Trajectory(
                    sample=episode.sample,
                    group_id=episode.sample.id,
                    messages=_role_content(log) + list(result.messages),
                    turns=list(episode.turns),
                    reward=sum(t.reward for t in episode.turns),
                    done=True,
                    info={"episode": episode_id},
                )
                finished.append((i, trajectory))
                self._finish(episode_id)
                observations.append({"role": TERMINAL_ROLE, "content": ""})
            else:
                role = str(result.messages[0].get("role", "user")) if result.messages else "user"
                observations.append({"role": role, "content": "\n".join(str(m.get("content", "")) for m in result.messages)})
            out_metadata.append(meta)
            rewards.append(float(result.reward))
            terminateds.append(1.0 if done else 0.0)

        if finished and self.reward_fn is not None:
            results = self.reward_fn([t for _, t in finished])
            for (i, _), rr in zip(finished, results, strict=True):
                rewards[i] += float(rr.score)
                out_metadata[i][BREAKDOWN_KEY] = {k: float(v) for k, v in rr.breakdown.items()}

        return EnvironmentReturn(
            observations=observations,
            metadata=out_metadata,
            next_stop_strings=[None] * len(message_log_batch),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.float32),
            answers=None,
        )

    def global_post_process_and_metrics(self, batch: Any) -> tuple[Any, dict[str, float]]:
        """Batch-level metrics after all rollouts (``total_reward`` accumulated by NeMo-RL)."""
        rewards = None
        for key in ("total_reward", "rewards"):
            try:
                rewards = batch[key]
            except (KeyError, TypeError):
                continue
            break
        metrics: dict[str, float] = {"live_episodes": float(self.num_live_episodes)}
        if rewards is not None:
            t = torch.as_tensor(rewards, dtype=torch.float32)
            if t.ndim > 1:
                t = t[:, 0]
            if t.numel():
                metrics.update(
                    {
                        "reward_mean": float(t.mean()),
                        "reward_std": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
                        "reward_max": float(t.max()),
                        "reward_min": float(t.min()),
                        "num_samples": float(t.numel()),
                    }
                )
        return batch, metrics

    def shutdown(self) -> None:
        with self._lock:
            episodes = list(self._episodes.values())
            self._episodes.clear()
        for episode in episodes:
            episode.env.close()


__all__ = ["ACTOR_FQN", "BREAKDOWN_KEY", "EPISODE_KEY", "EnvironmentReturn", "MonorlEnvironmentBridge", "build_env_config"]
