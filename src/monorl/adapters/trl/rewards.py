"""Expose a monorl :class:`~monorl.core.interfaces.RewardFunction` as a TRL reward function.

TRL calls ``fn(prompts=, completions=, completion_ids=, **columns)``; monorl rewards score
:class:`~monorl.core.types.Trajectory` batches. The wrapper rebuilds trajectories — from the
``monorl_trajectory`` field when the multi-turn rollout produced them, otherwise from TRL's
single-turn prompt/completion pairs — and keeps the scored batch in :attr:`last_batch` so the
reporter can write ``trajectories.jsonl`` once TRL has computed advantages.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from monorl.adapters.trl.data import sample_from_row
from monorl.config.schema import AlgorithmConfig
from monorl.core.interfaces import RewardFunction
from monorl.core.types import Completion, RewardResult, Sample, Trajectory, Turn
from monorl.orchestrator.episodes import chat_prompt_ids

TRAJECTORY_FIELD = "monorl_trajectory"


def completion_text(completion: Any) -> str:
    """TRL completions are strings (standard) or ``[{"role": "assistant", "content": ...}]``."""
    if isinstance(completion, str):
        return completion
    parts = []
    for message in completion:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return "".join(parts)


class MonorlRewardFunc:
    """Callable TRL reward function; ``__name__`` decides TRL's ``rewards/<name>/*`` metric keys.

    Scores like the orchestrator: reward-function score (``0`` when ``config.rewards`` is empty and
    the environment owns the reward) + ``gamma``-discounted per-turn environment rewards, minus
    ``overlong_penalty`` for length-truncated episodes. ``RewardResult.breakdown`` members are
    logged through TRL's ``log_metric`` as ``reward/<member>``.
    """

    def __init__(
        self,
        reward: RewardFunction | None,
        *,
        tokenizer: Any,
        eos_token_ids: Sequence[int],
        algorithm: AlgorithmConfig | None = None,
        name: str = "monorl",
    ):
        self.reward = reward
        self.tokenizer = tokenizer
        self.eos_token_ids = {int(t) for t in eos_token_ids}
        self.algorithm = algorithm or AlgorithmConfig()
        self.__name__ = name
        self.__qualname__ = name
        self._lock = threading.Lock()
        self.last_batch: list[Trajectory] = []
        self.num_batches = 0

    def __call__(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        completion_ids: Sequence[Sequence[int]] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        trajectories = self.build_trajectories(prompts, completions, completion_ids, kwargs)
        if self.reward is not None:
            results = list(self.reward(trajectories))
            if len(results) != len(trajectories):
                raise ValueError(
                    f"reward {self.reward.name!r} returned {len(results)} results for {len(trajectories)} trajectories"
                )
        else:
            results = [RewardResult(score=0.0) for _ in trajectories]
        algo = self.algorithm
        sums: dict[str, float] = {}
        for traj, res in zip(trajectories, results, strict=True):
            breakdown = {k: float(v) for k, v in res.breakdown.items()}
            total = float(res.score)
            env_return = sum((algo.gamma**k) * float(turn.reward) for k, turn in enumerate(traj.turns))
            if any(turn.reward != 0.0 for turn in traj.turns):
                breakdown["env"] = env_return
            total += env_return
            if (
                algo.overlong_penalty > 0
                and traj.turns
                and traj.turns[-1].completion.finish_reason == "length"
            ):
                total -= algo.overlong_penalty
                breakdown["overlong_penalty"] = -algo.overlong_penalty
            traj.reward = total
            traj.reward_breakdown = breakdown
            if res.info:
                traj.info["reward_info"] = dict(res.info)
            for k, v in breakdown.items():
                sums[k] = sums.get(k, 0.0) + v
        log_metric = kwargs.get("log_metric")
        if callable(log_metric) and trajectories:
            for k, value in sums.items():
                log_metric(f"reward/{k}", value / len(trajectories))
        with self._lock:
            self.last_batch = trajectories
            self.num_batches += 1
        return [t.reward for t in trajectories]

    def build_trajectories(
        self,
        prompts: Sequence[Any],
        completions: Sequence[Any],
        completion_ids: Sequence[Sequence[int]] | None,
        kwargs: dict[str, Any],
    ) -> list[Trajectory]:
        ready = kwargs.get(TRAJECTORY_FIELD)
        if (
            isinstance(ready, list)
            and len(ready) == len(prompts)
            and all(isinstance(t, Trajectory) for t in ready)
        ):
            return list(ready)
        step = getattr(kwargs.get("trainer_state"), "global_step", 0)
        rows = _rows_from_columns(kwargs, len(prompts))
        out: list[Trajectory] = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions, strict=True)):
            sample = sample_from_row(rows[i]) if rows[i] else Sample(id=str(i), messages=list(prompt))
            ids = [int(t) for t in completion_ids[i]] if completion_ids is not None else []
            text = completion_text(completion)
            finished = bool(ids) and ids[-1] in self.eos_token_ids
            prompt_ids = (
                chat_prompt_ids(self.tokenizer, prompt)
                if _is_chat(prompt)
                else list(self.tokenizer(str(prompt), add_special_tokens=False)["input_ids"])
            )
            comp = Completion(
                text=text,
                token_ids=ids,
                finish_reason="stop" if finished or not ids else "length",
                prompt_token_ids=prompt_ids,
            )
            messages = list(prompt) if _is_chat(prompt) else [{"role": "user", "content": str(prompt)}]
            messages.append({"role": "assistant", "content": text})
            out.append(
                Trajectory(
                    sample=sample,
                    group_id=f"{step}-{sample.id}",
                    messages=messages,
                    turns=[Turn(prompt_token_ids=prompt_ids, completion=comp)],
                    done=True,
                    info={"finish_reason": comp.finish_reason, "num_turns": 1},
                )
            )
        return out


def _is_chat(prompt: Any) -> bool:
    return isinstance(prompt, list) and bool(prompt) and isinstance(prompt[0], dict) and "role" in prompt[0]


def _rows_from_columns(kwargs: dict[str, Any], n: int) -> list[dict[str, Any]]:
    """TRL hands dataset columns as parallel lists; regroup them per example."""
    columns = {
        k: v
        for k, v in kwargs.items()
        if isinstance(v, list) and len(v) == n and k not in ("completion_ids", TRAJECTORY_FIELD)
    }
    return [{k: v[i] for k, v in columns.items()} for i in range(n)]
