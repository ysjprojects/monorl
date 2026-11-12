"""``algorithm.credit_assignment`` and the value baseline of ``ppo`` from trajectories to the batch
the trainer gets: per-token alignment of per-turn advantages, the positions of the value head, and
the loop on the fake components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fakes
import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.registry import registry
from monorl.core.reporting import iter_metrics
from monorl.core.types import Completion, RewardResult, Sample, StepResult, Trajectory, Turn
from monorl.orchestrator import run
from monorl.orchestrator.batching import build_train_batch, build_turn_batch

PAD = 0


def _turn(prompt: list[int], completion: list[int], reward: float = 0.0) -> Turn:
    return Turn(
        prompt_token_ids=prompt,
        completion=Completion(text="", token_ids=completion, logprobs=[-0.5] * len(completion)),
        reward=reward,
    )


def _traj(turns: list[Turn], group_id: str = "g") -> Trajectory:
    sample = Sample(id="s", messages=[{"role": "user", "content": "q"}], reference="a")
    return Trajectory(sample=sample, group_id=group_id, messages=[], turns=turns, reward=1.0)


def _two_row_trajectory() -> Trajectory:
    """Turn 1 continues turn 0 token for token; turn 2 starts a new context (a compacted history),
    so the episode is two training rows: [turn 0, turn 1] and [turn 2]."""
    return _traj(
        [
            _turn([1, 2], [3, 4]),
            _turn([1, 2, 3, 4, 5], [6]),
            _turn([9, 9, 9], [7, 8]),
        ]
    )


# -- batch building ---------------------------------------------------------------------------


def test_every_token_carries_the_advantage_of_its_own_turn_across_rows() -> None:
    trajectories = [_two_row_trajectory(), _traj([_turn([1, 2, 3], [4, 5])])]
    per_turn = torch.tensor([[10.0, 20.0, 30.0], [-1.0, 0.0, 0.0]])
    batch, stats = build_train_batch(
        trajectories, per_turn, pad_token_id=PAD, max_seq_len=64, loss_agg="rollout"
    )
    assert batch.input_ids.tolist() == [[1, 2, 3, 4, 5, 6], [9, 9, 9, 7, 8, PAD], [1, 2, 3, 4, 5, PAD]]
    assert batch.response_mask.tolist() == [[0, 0, 1, 1, 0, 1], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 0]]
    assert batch.advantages.tolist() == [
        [0, 0, 10.0, 10.0, 0, 20.0],
        [0, 0, 0, 30.0, 30.0, 0],
        [0, 0, 0, -1.0, -1.0, 0],
    ]
    assert stats["tokens/prefix_mismatch"] == 1 and stats["tokens/sequences"] == 3
    # the rollout aggregation still weighs the first episode's five tokens as one episode
    assert batch.loss_weights[0].tolist() == pytest.approx([0, 0, 0.2, 0.2, 0, 0.2])
    assert batch.loss_weights[1].tolist() == pytest.approx([0, 0, 0, 0.2, 0.2, 0])
    assert batch.values is None and batch.returns is None and batch.value_mask is None


def test_one_advantage_per_trajectory_is_broadcast_as_before() -> None:
    trajectories = [_two_row_trajectory(), _traj([_turn([1, 2, 3], [4, 5])])]
    batch, _ = build_train_batch(trajectories, torch.tensor([0.5, -2.0]), pad_token_id=PAD, max_seq_len=64)
    assert batch.advantages.tolist() == [
        [0, 0, 0.5, 0.5, 0, 0.5],
        [0, 0, 0, 0.5, 0.5, 0],
        [0, 0, 0, -2.0, -2.0, 0],
    ]
    with pytest.raises(ValueError, match="per-turn advantages"):
        build_train_batch(trajectories, torch.zeros(2, 2), pad_token_id=PAD, max_seq_len=64)


def test_layout_reads_and_places_values_at_the_last_prompt_token_of_every_turn() -> None:
    trajectories = [_two_row_trajectory(), _traj([_turn([1, 2, 3], [4, 5])])]
    batch, _, layout = build_turn_batch(trajectories, pad_token_id=PAD, max_seq_len=64)
    assert not batch.advantages.any()
    assert layout.owner.tolist() == [0, 0, 1]
    assert layout.turn_mask.tolist() == [[True, True, True], [True, False, False]]
    # a value for every position of the batch, as a trainer's compute_values returns it
    values = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    # states: turn 0 -> row 0 pos 1; turn 1 -> row 0 pos 4; turn 2 -> row 1 pos 2; other episode -> row 2 pos 2
    assert layout.state_values(values).tolist() == [[1.0, 4.0, 8.0], [14.0, 0.0, 0.0]]
    assert layout.state_mask().tolist() == [[0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
    targets = torch.tensor([[0.1, 0.2, 0.3], [0.4, 9.0, 9.0]])
    torch.testing.assert_close(
        layout.per_state(targets),
        torch.tensor([[0, 0.1, 0, 0, 0.2, 0], [0, 0, 0.3, 0, 0, 0], [0, 0, 0.4, 0, 0, 0]]),
    )
    # only the first turn of every episode (the episode as one step)
    first = torch.zeros_like(layout.turn_mask)
    first[:, 0] = True
    assert layout.state_mask(first).tolist() == [[0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]


def test_a_turn_cut_off_by_max_seq_len_has_no_tokens_and_no_state() -> None:
    trajectory = _traj([_turn([1, 2], [3, 4]), _turn([1, 2, 3, 4, 5, 6, 7], [8, 9])])
    batch, stats, layout = build_turn_batch([trajectory], pad_token_id=PAD, max_seq_len=5)
    assert stats["tokens/truncated"] == 1
    assert batch.response_mask.tolist() == [[0, 0, 1, 1, 0]]
    assert layout.state_mask().tolist() == [[0, 1, 0, 0, 0]]
    assert layout.state_values(torch.full((1, 5), 7.0)).tolist() == [[7.0, 0.0]]
    assert layout.per_token(torch.tensor([[1.0, 2.0]])).tolist() == [[0, 0, 1.0, 1.0, 0]]


# -- the loop on the fakes ----------------------------------------------------------------------


class ScriptedTurnsEnv:
    """The completion text decides the turn: ``go<r>`` earns ``r`` and continues, ``end<r>`` earns
    ``r`` and ends the episode, so episodes of one group differ in length and per-turn rewards."""

    def __init__(self, config: RunConfig, **_: Any) -> None:
        self.max_turns = config.env.max_turns

    def reset(self, sample: Sample) -> list[dict[str, Any]]:
        return list(sample.messages)

    def step(self, action: Completion) -> StepResult:
        done = action.text.startswith("end")
        reward = float(action.text.removeprefix("end").removeprefix("go"))
        return StepResult(
            messages=[] if done else [{"role": "user", "content": "n"}], reward=reward, done=done
        )

    def close(self) -> None:
        pass


class FinalAnswerReward:
    """``params["scores"]``: reward-function score by the text of the final answer."""

    name = "test_final"

    def __init__(self, params: dict[str, Any]) -> None:
        self.scores = dict(params.get("scores", {}))

    def __call__(self, trajectories: list[Trajectory]) -> list[RewardResult]:
        return [RewardResult(score=float(self.scores.get(t.last_response_text, 0.0))) for t in trajectories]


class NoCriticTrainer(fakes.FakeTrainer):
    compute_values = None


# Episodes in generation order (one at a time): group s1-p0 = [1, 0, 2] and [0, 3]; s1-p1 = [4] and [1, 1].
SCRIPT = ["go1", "go0", "end2", "go0", "end3", "end4", "go1", "end1"]


@pytest.fixture
def scripted(make_config: Any):
    registry.register("env", "scripted_turns", ScriptedTurnsEnv, overwrite=True)
    registry.register("reward", "test_final", lambda params, **kw: FinalAnswerReward(params), overwrite=True)
    registry.register("trainer", "fake_no_critic", NoCriticTrainer, overwrite=True)

    def _make(algorithm: dict[str, Any], **overrides: Any) -> RunConfig:
        sections: dict[str, Any] = {
            "env": {"type": "scripted_turns", "max_turns": 4},
            "rollout": {"extra": {"fake_mode": "script", "fake_script": SCRIPT, "max_parallel_episodes": 1}},
            "trainer": {"max_steps": 1, "max_seq_len": 128},
            "algorithm": {"advantage_norm": "none", **algorithm},
        }
        for key, value in overrides.items():
            sections[key] = {**sections[key], **value} if isinstance(sections.get(key), dict) else value
        return make_config(**sections)

    return _make


def _per_turn(batch: Any, tensor: torch.Tensor) -> list[list[float]]:
    """The value ``tensor`` holds on every completion span of every row (it must be constant on a
    span); the fake episodes are one row each."""
    out = []
    for mask, values in zip(batch.response_mask.tolist(), tensor.tolist(), strict=True):
        spans: list[list[float]] = []
        previous = 0
        for m, v in zip(mask, values, strict=True):
            if m and not previous:
                spans.append([])
            if m:
                spans[-1].append(v)
            previous = m
        assert all(len(set(span)) == 1 for span in spans), spans
        out.append([span[0] for span in spans])
    return out


def _states(batch: Any, tensor: torch.Tensor) -> list[list[float]]:
    """``tensor`` at the positions ``batch.value_mask`` marks, per row."""
    return [row[mask.bool()].tolist() for row, mask in zip(tensor, batch.value_mask, strict=True)]


def _metrics(result: Any) -> list[dict[str, Any]]:
    return [row for _, row in iter_metrics(Path(result.run_dir)) if row["phase"] == "train"]


def _approx(rows: list[list[float]]) -> list[Any]:
    return [pytest.approx(row) for row in rows]


def test_default_credit_gives_every_token_its_episode_advantage(scripted: Any) -> None:
    result = run(scripted({"name": "rloo"}))
    assert result.state == "succeeded"
    batch = fakes.TRAINERS[-1].batches[0]
    assert batch.rewards.tolist() == [3.0, 3.0, 4.0, 2.0]
    # the (fake, unnormalised) episode-level estimator is consulted once and its output broadcast
    assert [a.tolist() for a in fakes.ADVANTAGE_CALLS] == [[3.0, 3.0, 4.0, 2.0]]
    assert _per_turn(batch, batch.advantages) == [[3.0] * 3, [3.0] * 2, [4.0], [2.0] * 2]
    row = _metrics(result)[0]
    assert "advantage/turn_spread" not in row and "value/mean" not in row
    assert batch.value_mask is None and fakes.TRAINERS[-1].value_calls == []


def test_turn_credit_in_the_loop_with_unequal_turn_counts(scripted: Any) -> None:
    result = run(scripted({"name": "rloo", "credit_assignment": "turn", "loss_agg": "rollout"}))
    assert result.state == "succeeded"
    batch = fakes.TRAINERS[-1].batches[0]
    assert batch.rewards.tolist() == [3.0, 3.0, 4.0, 2.0]  # the episode reward is reported as before
    # returns from every turn on: [3, 2, 2], [3, 3], [4], [2, 1]; each minus the same turn of the
    # other member, which stands in with its last turn once it has ended
    assert _per_turn(batch, batch.advantages) == [[0.0, -1.0, -1.0], [0.0, 1.0], [2.0], [-2.0, -3.0]]
    assert batch.loss_weights is not None and batch.value_mask is None
    row = _metrics(result)[0]
    assert row["advantage/turn_spread"] == pytest.approx((1 + 1 + 0 + 1) / 4)
    assert row["advantage/turn_spread_max"] == 1.0
    assert row["advantage/first_turn_abs"] == 1.0 and row["advantage/last_turn_abs"] == pytest.approx(7 / 4)
    assert fakes.ADVANTAGE_CALLS == []  # the episode-level estimator was not consulted


def test_turn_credit_discounts_later_rewards_and_the_score_arrives_at_the_final_turn(scripted: Any) -> None:
    cfg = scripted(
        {"name": "reinforce", "credit_assignment": "turn", "gamma": 0.5},
        rewards=[{"type": "test_final", "params": {"scores": {"end3": 10.0}}}],
    )
    result = run(cfg)
    assert result.state == "succeeded"
    batch = fakes.TRAINERS[-1].batches[0]
    # episode rewards as before: score + sum_k 0.5^k r_k
    assert batch.rewards.tolist() == pytest.approx([1.5, 11.5, 4.0, 1.5])
    # [1, 0, 2] -> 1.5, 1, 2;  [0, 3 + 10] -> 6.5, 13;  [4];  [1, 1] -> 1.5, 1
    assert _per_turn(batch, batch.advantages) == _approx([[1.5, 1.0, 2.0], [6.5, 13.0], [4.0], [1.5, 1.0]])


def test_ppo_in_the_loop_runs_gae_over_turns_and_ships_values_and_returns(scripted: Any) -> None:
    cfg = scripted(
        {"name": "ppo", "credit_assignment": "turn", "lam": 0.5, "num_epochs": 2},
        rollout={"async_steps_ahead": 1},
        trainer={"max_steps": 2, "max_seq_len": 128},
    )
    result = run(cfg)
    assert result.state == "succeeded"
    trainer = fakes.TRAINERS[-1]
    assert (
        len(trainer.batches) == 4 and len(trainer.value_calls) == 2
    )  # one value pass per batch, not per epoch
    assert trainer.batches[0] is trainer.batches[1]
    batch = trainer.batches[0]
    # V = 0.25 everywhere, gamma = 1, lam = 0.5. [1, 0, 2]: d = 1, 0, 1.75 -> A = 1.4375, 0.875, 1.75;
    # [0, 3]: d = 0, 2.75 -> 1.375, 2.75;  [4]: 3.75;  [1, 1]: d = 1, 0.75 -> 1.375, 0.75
    assert _per_turn(batch, batch.advantages) == _approx(
        [[1.4375, 0.875, 1.75], [1.375, 2.75], [3.75], [1.375, 0.75]]
    )
    # one value position per turn: the last prompt token before each completion
    starts = batch.response_mask[:, 1:] * (1 - batch.response_mask[:, :-1])
    assert torch.equal(batch.value_mask[:, :-1], starts)
    assert _states(batch, batch.values) == _approx([[0.25] * 3, [0.25] * 2, [0.25], [0.25] * 2])
    assert _states(batch, batch.returns) == _approx([[1.6875, 1.125, 2.0], [1.625, 3.0], [4.0], [1.625, 1.0]])
    row = _metrics(result)[0]
    assert row["value/mean"] == pytest.approx(0.25)
    assert row["value/return_mean"] == pytest.approx((1.6875 + 1.125 + 2 + 1.625 + 3 + 4 + 1.625 + 1) / 8)
    assert row["value/explained_variance"] == pytest.approx(0.0, abs=1e-6)  # a constant explains nothing
    assert row["advantage/turn_spread_max"] == pytest.approx(1.375)
    assert "time/values" in row


def test_ppo_with_episode_credit_values_the_episode_at_its_first_state(scripted: Any) -> None:
    result = run(scripted({"name": "ppo", "lam": 0.5, "gamma": 0.5}))
    assert result.state == "succeeded"
    batch = fakes.TRAINERS[-1].batches[0]
    # episode rewards sum_k 0.5^k r_k = 1.5, 1.5, 4, 1.5; one step each: A = R - 0.25 on every token
    assert _per_turn(batch, batch.advantages) == _approx([[1.25] * 3, [1.25] * 2, [3.75], [1.25] * 2])
    assert batch.value_mask.sum(dim=1).tolist() == [1, 1, 1, 1]
    assert _states(batch, batch.returns) == _approx([[1.5], [1.5], [4.0], [1.5]])
    assert "advantage/turn_spread" not in _metrics(result)[0]


def test_ppo_needs_a_trainer_with_a_value_head(scripted: Any) -> None:
    result = run(scripted({"name": "ppo"}, trainer={"backend": "fake_no_critic", "max_steps": 1}))
    assert result.state == "failed"
    assert "needs a trainer with a value head" in (result.error or "")


# -- pipeline backends compute their own advantages ---------------------------------------------


def _backend_config(tmp_path: Path, backend: str, **algorithm: Any) -> RunConfig:
    return RunConfig.from_dict(
        {
            "model": {"name_or_path": "org/model"},
            "data": {"uri": str(tmp_path / "data.jsonl")},
            "rewards": [{"type": "exact_match"}],
            "rollout": {"backend": "vllm"},
            "trainer": {"backend": backend},
            "algorithm": algorithm,
        }
    )


def test_pipeline_backends_reject_turn_credit(tmp_path: Path) -> None:
    from monorl.adapters.nemo_rl.compile import compile_config as compile_nemo
    from monorl.adapters.trl.compile import grpo_config_kwargs
    from monorl.adapters.verl.compiler import compile_config as compile_verl

    turn = {"credit_assignment": "turn"}
    with pytest.raises(ConfigError, match="credit_assignment='turn'.*trainer.backend='verl'"):
        compile_verl(_backend_config(tmp_path, "verl", **turn), tmp_path)
    with pytest.raises(ConfigError, match="credit_assignment='turn'.*trainer.backend='nemo_rl'"):
        compile_nemo(_backend_config(tmp_path, "nemo_rl", **turn), tmp_path)
    with pytest.raises(ConfigError, match="credit_assignment='turn'.*trainer.backend='trl'"):
        grpo_config_kwargs(_backend_config(tmp_path, "trl", **turn), tmp_path, device=torch.device("cpu"))
    # the default is compiled as before; the value clip maps onto verl's critic, which cannot be unclipped
    default = compile_verl(_backend_config(tmp_path, "verl", name="ppo"), tmp_path)
    assert default["algorithm"]["adv_estimator"] == "gae" and "cliprange_value" not in default["critic"]
    clipped = compile_verl(_backend_config(tmp_path, "verl", name="ppo", value_clip=0.2), tmp_path)
    assert clipped["critic"]["cliprange_value"] == 0.2
    with pytest.raises(ConfigError, match="value_clip=null"):
        compile_verl(_backend_config(tmp_path, "verl", name="ppo", value_clip=None), tmp_path)
