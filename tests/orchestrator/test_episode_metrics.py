"""Agent metrics: ``episode_metrics`` values and what the loop reports for training and eval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fakes
import pytest

from monorl.core.reporting import iter_metrics, iter_trajectories
from monorl.core.types import Completion, Message, Sample, Trajectory, Turn
from monorl.orchestrator import run
from monorl.orchestrator.metrics import episode_metrics, eval_metrics, step_metrics


def _trajectory(
    group: str,
    reward: float,
    turns: list[tuple[int, int, str, dict[str, Any]]],
    *,
    done: bool = True,
    extra_threads: list[list[Message]] | None = None,
) -> Trajectory:
    """``turns``: ``(prompt tokens, completion tokens, finish_reason, info)`` per turn."""
    return Trajectory(
        sample=Sample(id=group, messages=[{"role": "user", "content": "q"}]),
        group_id=group,
        messages=[{"role": "user", "content": "q"}],
        turns=[
            Turn(
                prompt_token_ids=[0] * p,
                completion=Completion(text="", token_ids=[0] * c, finish_reason=finish),  # type: ignore[arg-type]
                info=info,
            )
            for p, c, finish, info in turns
        ],
        reward=reward,
        reward_breakdown={"exact": reward},
        done=done,
        extra_threads=extra_threads or [],
    )


def test_episode_metrics_values() -> None:
    call = {"name": "add"}
    trajectories = [
        # group a: one solved, one truncated
        _trajectory(
            "a",
            1.0,
            [(10, 5, "tool_call", {"tool_calls": [call, call], "invalid_calls": 1}), (30, 10, "stop", {})],
        ),
        _trajectory("a", 0.0, [(10, 90, "length", {"tool_calls": [call], "invalid_calls": 2})], done=False),
        # group b: never solved
        _trajectory("b", 0.5, [(20, 4, "stop", {"tool_calls": []})]),
        _trajectory("b", 0.25, [(20, 6, "length", {}), (40, 20, "stop", {}), (70, 10, "stop", {})]),
    ]
    metrics = episode_metrics(trajectories, success_threshold=1.0)
    assert metrics == pytest.approx(
        {
            "success_rate": 0.25,
            "pass_at_k": 0.5,
            "turns_mean": 7 / 4,
            "turns_max": 3.0,
            "done_frac": 0.75,
            "truncated_frac": 0.25,  # only an episode whose *last* turn hit the length limit
            "context_tokens_mean": (40 + 100 + 24 + 80) / 4,
            "context_tokens_max": 100.0,
            "tool_calls_mean": 3 / 4,
            "invalid_calls_mean": 3 / 4,
        }
    )
    assert episode_metrics(trajectories, success_threshold=0.5)["success_rate"] == 0.5
    assert episode_metrics(trajectories, success_threshold=0.5)["pass_at_k"] == 1.0


def test_optional_episode_metrics_appear_only_when_reported() -> None:
    singles = [_trajectory("a", 1.0, [(3, 2, "stop", {})]), _trajectory("b", 0.0, [(3, 2, "stop", {})])]
    metrics = episode_metrics(singles, success_threshold=1.0)
    assert metrics["success_rate"] == 0.5
    assert not {"pass_at_k", "tool_calls_mean", "invalid_calls_mean", "threads_mean"} & set(metrics)
    assert episode_metrics([], success_threshold=1.0) == {}

    compacted = _trajectory(
        "c", 1.0, [(3, 2, "stop", {})], extra_threads=[[{"role": "user", "content": "old"}]]
    )
    assert episode_metrics([*singles, compacted], success_threshold=1.0)["threads_mean"] == pytest.approx(
        4 / 3
    )


def test_step_and_eval_metrics_prefixes() -> None:
    trajectories = [_trajectory("a", 1.0, [(3, 2, "stop", {})]), _trajectory("a", 0.0, [(3, 2, "stop", {})])]
    step = step_metrics(trajectories, success_threshold=1.0)
    assert step["reward/mean"] == 0.5 and step["reward/exact"] == 0.5
    assert step["reward/success_rate"] == 0.5 and step["reward/pass_at_k"] == 1.0
    assert step["episode/turns_mean"] == 1.0 and step["episode/context_tokens_max"] == 5.0
    assert "episode/success_rate" not in step and "reward/turns_mean" not in step

    evaluation = eval_metrics(trajectories, success_threshold=1.0)
    assert evaluation["eval/reward_mean"] == 0.5 and evaluation["eval/reward_std"] == 0.5
    assert evaluation["eval/reward/exact"] == 0.5 and evaluation["eval/num_samples"] == 2.0
    assert evaluation["eval/success_rate"] == 0.5 and evaluation["eval/pass_at_k"] == 1.0
    assert evaluation["eval/turns_mean"] == 1.0 and evaluation["eval/done_frac"] == 1.0


def test_loop_reports_agent_metrics_and_eval_rows(make_config: Any, tmp_path: Path) -> None:
    eval_file = fakes.write_jsonl(
        tmp_path / "eval.jsonl",
        [
            {"prompt": "xyz", "answer": "XYZ"},
            {"prompt": "uvw", "answer": "UVW"},
            {"prompt": "rst", "answer": "RST"},
        ],
    )
    cfg = make_config(
        data={"eval_uri": str(eval_file)},
        trainer={"max_steps": 1, "eval_every": 1},
        # one episode at a time: the fake engine answers right, wrong, right, wrong, ...
        rollout={"extra": {"max_parallel_episodes": 1}},
        logging={"trajectories_per_step": 2},
    )
    run_dir = Path(run(cfg).run_dir)
    metrics = {row["phase"]: row for _, row in iter_metrics(run_dir)}

    train = metrics["train"]  # 2 prompts x 2 samples: each group has one right and one wrong answer
    assert train["reward/success_rate"] == 0.5 and train["reward/pass_at_k"] == 1.0
    assert train["episode/turns_mean"] == 1.0 and train["episode/turns_max"] == 1.0
    assert train["episode/done_frac"] == 1.0 and train["episode/truncated_frac"] == 0.0
    assert train["episode/context_tokens_max"] >= train["episode/context_tokens_mean"] > 0

    evaluation = metrics["eval"]  # 3 greedy episodes: right, wrong, right
    assert evaluation["eval/num_samples"] == 3.0
    assert evaluation["eval/reward_mean"] == pytest.approx(2 / 3)
    assert evaluation["eval/success_rate"] == pytest.approx(2 / 3)
    assert evaluation["eval/turns_mean"] == 1.0 and evaluation["eval/done_frac"] == 1.0
    assert "eval/pass_at_k" not in evaluation  # one episode per prompt

    rows = [row for _, row in iter_trajectories(run_dir)]
    assert [r["phase"] for r in rows] == ["train", "train", "eval", "eval"]  # trajectories_per_step each
    assert [(r["response"], r["success"]) for r in rows[2:]] == [("XYZ", True), ("wrong", False)]
    for row in rows:
        (thread,) = row["threads"]
        (turn,) = row["turns"]
        assert thread[turn["message_index"]] == {"role": "assistant", "content": row["response"]}


def test_loop_rows_point_every_turn_at_its_assistant_message(make_config: Any) -> None:
    cfg = make_config(
        env={"type": "fake_multi", "max_turns": 3},
        rollout={"extra": {"fake_mode": "always_right"}},
        trainer={"max_steps": 1},
    )
    run_dir = Path(run(cfg).run_dir)
    rows = [row for _, row in iter_trajectories(run_dir)]
    assert rows
    for row in rows:
        (thread,) = row["threads"]
        assert [thread[t["message_index"]]["content"] for t in row["turns"]] == [
            row["prompt"].upper(),
            "AGAIN1",
            "AGAIN2",
        ]
        assert [t["info"]["turn"] for t in row["turns"]] == [1, 2, 3]
        assert (
            row["num_context_tokens"]
            == row["turns"][-1]["prompt_tokens"] + row["turns"][-1]["response_tokens"]
        )
    (train,) = [row for _, row in iter_metrics(run_dir) if row["phase"] == "train"]
    assert train["episode/turns_mean"] == 3.0


def test_loop_respects_transcript_and_threshold_settings(make_config: Any) -> None:
    cfg = make_config(
        trainer={"max_steps": 1},
        rollout={"extra": {"fake_mode": "script", "fake_script": ["nope"]}},
        logging={"save_transcripts": False, "success_threshold": 0.0},
    )
    run_dir = Path(run(cfg).run_dir)
    rows = [row for _, row in iter_trajectories(run_dir)]
    assert rows and all("threads" not in r and "turns" not in r for r in rows)
    assert all(r["reward"] == 0.0 and r["success"] is True for r in rows)
    (train,) = [row for _, row in iter_metrics(run_dir) if row["phase"] == "train"]
    assert train["reward/success_rate"] == 1.0 and train["reward/mean"] == 0.0
