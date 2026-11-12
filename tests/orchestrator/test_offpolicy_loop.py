from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fakes
import pytest
import torch

from monorl.core.registry import registry
from monorl.core.reporting import iter_metrics
from monorl.core.types import Completion, SamplingParams, TopLogprobs
from monorl.orchestrator import run


def _rows(run_dir: str) -> list[dict[str, Any]]:
    return [row for _, row in iter_metrics(Path(run_dir)) if row["phase"] == "train"]


def _offpolicy_keys(row: dict[str, Any]) -> set[str]:
    return {k for k in row if k.startswith("offpolicy/")}


class TopKEngine(fakes.FakeEngine):
    """Fake engine that answers ``SamplingParams.top_logprobs``: at every position the sampled
    token (logprob -0.1) followed by its successor ids."""

    def __init__(self, config: Any, replica: int = 0, **kwargs: Any) -> None:
        self.top_requests: list[int] = []
        super().__init__(config, replica, **kwargs)

    def generate(self, prompts: Sequence[Any], params: SamplingParams) -> list[list[Completion]]:
        out = super().generate(prompts, params)
        self.top_requests.append(params.top_logprobs)
        for group in out:
            for c in group:
                if params.top_logprobs > 0:
                    c.top_logprobs = TopLogprobs.from_positions(
                        [
                            [(token + i, -0.1 - i) for i in range(params.top_logprobs)]
                            for token in c.token_ids
                        ],
                        params.top_logprobs,
                    )
        return out


@pytest.fixture
def topk_engine(fake_components: Any) -> None:
    registry.register("rollout_engine", "fake_topk", TopKEngine, overwrite=True)


def test_default_config_keeps_the_engine_logprobs_as_the_only_policy(make_config: Any) -> None:
    result = run(make_config(trainer={"max_steps": 2}, rollout={"async_steps_ahead": 1}))
    trainer = fakes.TRAINERS[-1]
    assert trainer.logprob_calls == []
    for batch in trainer.batches:
        assert batch.behavior_logprobs is None and batch.behavior_top_ids is None
        assert batch.old_logprobs[batch.response_mask == 1].max().item() == pytest.approx(-0.1)
    assert all(not _offpolicy_keys(row) for row in _rows(result.run_dir))


def test_stale_batches_carry_both_policies_and_report_diagnostics(make_config: Any) -> None:
    cfg = make_config(
        trainer={"max_steps": 4, "save_every": 100, "extra": {"fake_sleep": 0.03}},
        rollout={"async_steps_ahead": 2, "extra": {"fake_latency": 0.001}},
        algorithm={
            "num_epochs": 2,
            "off_policy": {"importance_weights": "token", "importance_cap": 1.1, "ratio_mask": "token"},
        },
    )
    result = run(cfg)
    assert result.state == "succeeded"
    trainer = fakes.TRAINERS[-1]
    rows = _rows(result.run_dir)
    assert len(rows) == 4 and len(trainer.batches) == 8
    assert trainer.logprob_calls == [False] * 4  # the proximal anchor is computed once per batch
    assert max(row["policy/staleness"] for row in rows) >= 1.0
    for step, row in enumerate(rows):
        batch = trainer.batches[2 * step]
        assert trainer.batches[2 * step + 1] is batch  # both epochs train on the same anchor
        mask = batch.response_mask == 1
        # proximal: the fake trainer's -0.25; behaviour: the fake engine's -(0.1 + 0.01 j)
        assert torch.all(batch.old_logprobs[mask] == -0.25)
        behavior = batch.behavior_logprobs[mask]
        assert behavior.max().item() == pytest.approx(-0.1) and behavior.min().item() < -0.1
        assert torch.all(batch.behavior_logprobs[~mask] == 0)
        # rho = exp(-0.25 + 0.1 + 0.01 j) < 1: nothing is truncated, tokens with rho < 0.5 would be masked
        ratio = torch.exp(batch.old_logprobs[mask] - behavior)
        assert row["offpolicy/skipped"] == 0.0
        assert row["offpolicy/weight_mean"] == pytest.approx(ratio.mean().item(), rel=1e-5)
        assert row["offpolicy/weight_max"] == pytest.approx(ratio.max().item(), rel=1e-5)
        assert math.exp(-0.15) < row["offpolicy/weight_max"] < 1.0
        assert row["offpolicy/truncated_frac"] == 0.0 and row["offpolicy/masked_frac"] == 0.0
        assert row["offpolicy/ess"] == pytest.approx(
            (ratio.sum() ** 2 / (ratio.numel() * ratio.square().sum())).item(), rel=1e-5
        )
        assert row["offpolicy/kl"] == pytest.approx(
            (behavior - batch.old_logprobs[mask]).mean().item(), rel=1e-5
        )
        assert row["offpolicy/kl_k3"] == pytest.approx((ratio - 1 - ratio.log()).mean().item(), rel=1e-4)


def test_batches_without_engine_logprobs_train_uncorrected_and_say_so(make_config: Any) -> None:
    cfg = make_config(
        trainer={"max_steps": 1},
        rollout={"extra": {"fake_no_logprobs": True}},
        algorithm={"off_policy": {"importance_weights": "sequence"}},
    )
    result = run(cfg)
    assert result.state == "succeeded"
    batch = fakes.TRAINERS[-1].batches[0]
    assert batch.behavior_logprobs is None
    assert torch.all(batch.old_logprobs[batch.response_mask == 1] == -0.25)
    (row,) = _rows(result.run_dir)
    assert _offpolicy_keys(row) == {"offpolicy/skipped"} and row["offpolicy/skipped"] == 1.0


@pytest.mark.parametrize(
    "off_policy",
    [{"importance_weights": "token"}, {"ratio_mask": "sequence"}, {"anchor": "behavior"}],
)
def test_corrections_need_engine_logprobs_comparable_with_the_trainer(make_config: Any, off_policy) -> None:
    # the fake engine is not the hf engine: with top-k sampling its logprobs describe a truncated
    # distribution the trainer cannot reproduce
    result = run(make_config(rollout={"top_k": 20}, algorithm={"off_policy": off_policy}))
    assert result.state == "failed"
    assert "ConfigError" in result.error and "algorithm.off_policy" in result.error
    assert fakes.TRAINERS == []  # rejected before any component is built
    # the anchor on the trainer alone needs no behaviour logprobs: it runs and reports the skip
    result = run(
        make_config(
            rollout={"top_k": 20},
            trainer={"max_steps": 1},
            algorithm={"off_policy": {"anchor": "trainer"}},
            run_name="anchor-only",
        )
    )
    assert result.state == "succeeded"
    assert _rows(result.run_dir)[0]["offpolicy/skipped"] == 1.0


def test_score_centering_requests_top_k_and_aligns_it_with_every_turn(
    make_config: Any, topk_engine: None
) -> None:
    cfg = make_config(
        env={"type": "fake_multi", "max_turns": 2},
        rollout={
            "backend": "fake_topk",
            "turn_rendering": "template",  # with the mismatch token: one sequence per turn
            "extra": {"fake_mode": "always_right", "fake_mismatch": True},
        },
        trainer={"max_steps": 1},
        algorithm={"off_policy": {"score_centering": True, "score_centering_top_k": 3}},
    )
    result = run(cfg)
    assert result.state == "succeeded"
    assert set(fakes.ENGINES[-1].top_requests) == {3}
    batch = fakes.TRAINERS[-1].batches[0]
    assert len(batch) == 8 and batch.behavior_top_ids.shape == (*batch.input_ids.shape, 3)
    mask = batch.response_mask == 1
    # the head recorded for a position is the one of the token sampled there, in every row
    assert torch.equal(batch.behavior_top_ids[mask][:, 0], batch.input_ids[mask])
    assert torch.equal(batch.behavior_top_ids[mask][:, 2], batch.input_ids[mask] + 2)
    torch.testing.assert_close(
        batch.behavior_top_logprobs[mask], torch.tensor([-0.1, -1.1, -2.1]).expand(int(mask.sum()), 3)
    )
    assert torch.all(batch.behavior_top_logprobs[~mask] == float("-inf"))
    # score centering keeps the behaviour anchor: no second policy in the batch
    assert batch.behavior_logprobs is None and fakes.TRAINERS[-1].logprob_calls == []
    (row,) = _rows(result.run_dir)
    assert row["offpolicy/skipped"] == 0.0
    assert row["offpolicy/sc_head_mass"] == pytest.approx(sum(math.exp(-0.1 - i) for i in range(3)), rel=1e-5)


def test_score_centering_is_skipped_when_the_engine_returns_no_top_k(make_config: Any) -> None:
    cfg = make_config(trainer={"max_steps": 1}, algorithm={"off_policy": {"score_centering": True}})
    result = run(cfg)
    assert result.state == "succeeded"
    batch = fakes.TRAINERS[-1].batches[0]
    assert batch.behavior_top_ids is None and batch.behavior_top_logprobs is None
    (row,) = _rows(result.run_dir)
    assert _offpolicy_keys(row) == {"offpolicy/skipped"} and row["offpolicy/skipped"] == 1.0


def test_top_k_is_not_requested_without_score_centering(make_config: Any, topk_engine: None) -> None:
    run(make_config(trainer={"max_steps": 1}, rollout={"backend": "fake_topk"}))
    assert set(fakes.ENGINES[-1].top_requests) == {0}
    assert fakes.TRAINERS[-1].batches[0].behavior_top_ids is None
