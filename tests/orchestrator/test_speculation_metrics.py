"""``mtp/accept_rate`` and ``mtp/accept_length`` in the training loop: read from engines that
draft after every rollout step, kept apart from evaluations, absent when no engine drafts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fakes
import pytest

from monorl.core.registry import registry
from monorl.core.reporting import iter_metrics, iter_trajectories
from monorl.core.types import Completion, SamplingParams, SpeculationStats
from monorl.orchestrator import run


class DraftingEngine(fakes.FakeEngine):
    """Until its counters are first read every token costs the policy one pass; from then on
    every other token is an accepted draft."""

    def __init__(self, config: Any, replica: int = 0, **kwargs: Any) -> None:
        super().__init__(config, replica, **kwargs)
        self._stats = SpeculationStats()
        self._reads = 0

    def generate(self, prompts: Any, params: SamplingParams) -> list[list[Completion]]:
        out = super().generate(prompts, params)
        with self._lock:
            for completion in (c for group in out for c in group):
                self._stats.add(_cost(len(completion.token_ids), drafts_accepted=self._reads > 0))
        return out

    def speculation_stats(self) -> SpeculationStats:
        with self._lock:
            stats, self._stats = self._stats, SpeculationStats()
            self._reads += 1
        return stats


def _cost(tokens: int, *, drafts_accepted: bool) -> SpeculationStats:
    accepted = tokens // 2 if drafts_accepted else 0
    steps = tokens - accepted
    return SpeculationStats(steps=steps, drafted=steps, accepted=accepted, emitted=tokens)


def _expected(run_dir: Path, phase: str, step: int, *, drafts_accepted: bool) -> dict[str, float]:
    total = SpeculationStats()
    for _, row in iter_trajectories(run_dir):
        if row["phase"] == phase and row["step"] == step:
            for turn in row["turns"]:
                total.add(_cost(turn["response_tokens"], drafts_accepted=drafts_accepted))
    return total.metrics()


def test_drafting_statistics_are_reported_per_step_and_apart_from_evaluation(
    make_config: Any, tmp_path: Path
) -> None:
    registry.register("rollout_engine", "fake_drafting", DraftingEngine, overwrite=True)
    eval_file = fakes.write_jsonl(
        tmp_path / "eval.jsonl", [{"prompt": f"e{i}", "answer": f"E{i}"} for i in range(3)]
    )
    config = make_config(
        rollout={"backend": "fake_drafting", "extra": {"fake_mode": "always_right"}},
        data={"eval_uri": str(eval_file)},
        trainer={"max_steps": 2, "eval_every": 2},
        logging={"trajectories_per_step": 100},
    )
    result = run(config)
    assert result.state == "succeeded"
    run_dir = Path(result.run_dir)
    rows = {(row["phase"], row["step"]): row for _, row in iter_metrics(run_dir)}

    first, second = rows[("train", 1)], rows[("train", 2)]
    assert (first["mtp/accept_rate"], first["mtp/accept_length"]) == (0.0, 1.0)
    # the second step reports its own rollouts only: neither the first step's nor the evaluation's
    wanted = _expected(run_dir, "train", 2, drafts_accepted=True)
    assert wanted["mtp/accept_length"] > 1.3
    assert second["mtp/accept_rate"] == pytest.approx(wanted["mtp/accept_rate"])
    assert second["mtp/accept_length"] == pytest.approx(wanted["mtp/accept_length"])

    evaluation = rows[("eval", 2)]
    wanted = _expected(run_dir, "eval", 2, drafts_accepted=True)
    assert evaluation["eval/mtp/accept_length"] == pytest.approx(wanted["mtp/accept_length"])
    assert evaluation["eval/mtp/accept_rate"] == pytest.approx(wanted["mtp/accept_rate"])
    assert "mtp/accept_rate" not in evaluation


def test_engines_that_do_not_draft_report_nothing(make_config: Any) -> None:
    result = run(make_config(trainer={"max_steps": 1}))
    for _, row in iter_metrics(Path(result.run_dir)):
        assert not any(key.startswith(("mtp/", "eval/mtp/")) for key in row)
