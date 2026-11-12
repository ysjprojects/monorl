from __future__ import annotations

import pytest
import torch

from monorl.algorithms.losses import masked_aggregate
from monorl.core.types import Completion, Sample, Trajectory, Turn
from monorl.orchestrator.batching import (
    build_train_batch,
    flatten_trajectory,
    prefix_chain_ok,
    trajectory_rows,
)

PAD = 0


def _sample() -> Sample:
    return Sample(id="s", messages=[{"role": "user", "content": "q"}], reference="a")


def _traj(turns: list[Turn], reward: float = 1.0, group_id: str = "g") -> Trajectory:
    return Trajectory(sample=_sample(), group_id=group_id, messages=[], turns=turns, reward=reward)


def _turn(
    prompt: list[int], completion: list[int], logprobs: list[float] | None = None, finish: str = "stop"
) -> Turn:
    lp = [-0.5] * len(completion) if logprobs is None else logprobs
    return Turn(
        prompt_token_ids=prompt,
        completion=Completion(text="", token_ids=completion, logprobs=lp, finish_reason=finish),
    )


def test_single_turn_mask_and_padding() -> None:
    t1 = _traj([_turn([11, 12, 13], [21, 22], [-1.0, -2.0])], reward=1.0, group_id="g0")
    t2 = _traj([_turn([11, 12], [31], [-3.0])], reward=0.0, group_id="g0")
    batch, stats = build_train_batch([t1, t2], torch.tensor([1.0, -1.0]), pad_token_id=PAD, max_seq_len=64)

    assert batch.input_ids.tolist() == [[11, 12, 13, 21, 22], [11, 12, 31, PAD, PAD]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
    assert batch.response_mask.tolist() == [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0]]
    assert batch.old_logprobs is not None
    assert batch.old_logprobs.tolist() == [[0, 0, 0, -1.0, -2.0], [0, 0, -3.0, 0, 0]]
    assert batch.advantages.tolist() == [[0, 0, 0, 1.0, 1.0], [0, 0, -1.0, 0, 0]]
    assert batch.position_ids.tolist() == [[0, 1, 2, 3, 4], [0, 1, 2, 2, 2]]
    assert batch.rewards.tolist() == [1.0, 0.0]
    assert batch.group_ids == ["g0", "g0"]
    assert batch.ref_logprobs is None and batch.loss_weights is None
    assert stats["tokens/response_mean"] == 1.5
    assert stats["tokens/prompt_mean"] == 2.5
    assert stats["tokens/prefix_mismatch"] == 0
    assert stats["tokens/sequences"] == 2
    assert stats["tokens/truncated"] == 0
    assert len(batch) == 2 and batch.num_response_tokens == 3


def test_multi_turn_chain_masks_every_completion_span() -> None:
    # turn 0: prompt [1,2] completion [3,4]; turn 1 prompt = [1,2,3,4] + user tokens [5,6] + gen prompt [7]
    turns = [
        _turn([1, 2], [3, 4], [-0.1, -0.2]),
        _turn([1, 2, 3, 4, 5, 6, 7], [8, 9, 10], [-0.3, -0.4, -0.5]),
    ]
    traj = _traj(turns)
    assert prefix_chain_ok(traj)
    flat = flatten_trajectory(traj)
    assert flat.input_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert flat.response_mask == [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]
    assert flat.old_logprobs == [0, 0, -0.1, -0.2, 0, 0, 0, -0.3, -0.4, -0.5]
    assert flat.prompt_len == 2 and not flat.prefix_mismatch

    batch, stats = build_train_batch([traj], [2.0], pad_token_id=PAD, max_seq_len=64)
    assert len(batch) == 1
    assert batch.advantages[0].tolist() == [0, 0, 2.0, 2.0, 0, 0, 0, 2.0, 2.0, 2.0]
    assert stats["tokens/response_mean"] == 5.0


def test_a_broken_chain_trains_every_turn_in_its_own_sequence() -> None:
    """A turn whose prompt does not extend the previous turn token for token (the chat was
    re-rendered differently) must not cost the earlier turn its gradient."""
    turns = [
        _turn([1, 2], [3, 4], [-0.1, -0.2]),
        _turn([1, 2, 3, 99, 5, 6, 7], [8, 9], [-0.3, -0.4]),  # 99 != 4 -> chain broken
    ]
    traj = _traj(turns, reward=0.5, group_id="g7")
    assert not prefix_chain_ok(traj)
    other = _traj([_turn([1], [2])], reward=0.0, group_id="g8")
    batch, stats = build_train_batch([traj, other], [2.0, -1.0], pad_token_id=PAD, max_seq_len=64)

    assert batch.input_ids.tolist() == [
        [1, 2, 3, 4, PAD, PAD, PAD, PAD, PAD],
        [1, 2, 3, 99, 5, 6, 7, 8, 9],
        [1, 2, PAD, PAD, PAD, PAD, PAD, PAD, PAD],
    ]
    assert batch.response_mask.tolist() == [
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert batch.old_logprobs[0, 2:4].tolist() == pytest.approx([-0.1, -0.2])
    assert batch.old_logprobs[1, 7:9].tolist() == pytest.approx([-0.3, -0.4])
    # both sequences of the split trajectory carry its advantage, reward and group
    assert batch.advantages[:, :].sum(dim=1).tolist() == [4.0, 4.0, -1.0]
    assert batch.rewards.tolist() == [0.5, 0.5, 0.0]
    assert batch.group_ids == ["g7", "g7", "g8"]
    assert stats["tokens/prefix_mismatch"] == 1.0
    assert stats["tokens/sequences"] == 3.0


def test_single_sequence_view_of_a_broken_chain_is_its_final_turn() -> None:
    turns = [
        _turn([1, 2], [3, 4]),
        _turn([1, 2, 3, 99, 5, 6, 7], [8, 9]),
    ]
    flat = flatten_trajectory(_traj(turns))
    assert flat.input_ids == [1, 2, 3, 99, 5, 6, 7, 8, 9]
    assert flat.response_mask == [0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert flat.prompt_len == 7 and flat.prefix_mismatch


def test_sub_agent_context_is_a_sequence_of_its_own() -> None:
    """Main conversation, a sub-agent with its own context in between, main conversation again:
    the main turns stay one sequence although they are not adjacent."""
    turns = [
        _turn([1, 2], [3]),  # main
        _turn([50, 51], [52, 53]),  # sub-agent, unrelated context
        _turn([50, 51, 52, 53, 54], [55]),  # sub-agent continues
        _turn([1, 2, 3, 4, 5], [6, 7]),  # main continues turn 0
    ]
    rows = trajectory_rows(_traj(turns))
    assert [r.input_ids for r in rows] == [[50, 51, 52, 53, 54, 55], [1, 2, 3, 4, 5, 6, 7]]
    assert [r.response_mask for r in rows] == [[0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 1, 1]]
    assert all(r.prefix_mismatch for r in rows)


def test_a_turn_shared_by_two_branches_is_trained_once() -> None:
    """Two continuations of the same turn (a retry, a fork): the shared turn is a loss position in
    one sequence and plain context in the other."""
    turns = [
        _turn([1], [2, 3]),
        _turn([1, 2, 3, 4], [5]),  # branch A
        _turn([1, 2, 3, 9], [8, 7]),  # branch B continues turn 0 as well
    ]
    rows = trajectory_rows(_traj(turns))
    assert [r.input_ids for r in rows] == [[1, 2, 3, 4, 5], [1, 2, 3, 9, 8, 7]]
    assert [r.response_mask for r in rows] == [[0, 1, 1, 0, 1], [0, 0, 0, 0, 1, 1]]
    assert sum(sum(r.response_mask) for r in rows) == sum(len(t.completion.token_ids) for t in turns)


def test_compacted_context_starts_a_new_sequence() -> None:
    turns = [
        _turn([1, 2], [3]),
        _turn([1, 2, 3, 4], [5]),
        _turn([70, 71], [72]),  # history summarised: nothing of the old context remains
        _turn([70, 71, 72, 73], [74]),
    ]
    rows = trajectory_rows(_traj(turns))
    assert [r.input_ids for r in rows] == [[1, 2, 3, 4, 5], [70, 71, 72, 73, 74]]
    assert [r.response_mask for r in rows] == [[0, 0, 1, 0, 1], [0, 0, 1, 0, 1]]


def test_shorter_next_prompt_is_a_mismatch() -> None:
    traj = _traj([_turn([1, 2, 3], [4, 5]), _turn([1, 2], [6])])
    assert not prefix_chain_ok(traj)


def test_rollout_aggregation_weighs_every_trajectory_equally() -> None:
    """``loss_agg="rollout"``: a trajectory split into two sequences with 3+1 response tokens and
    one with 2 tokens each contribute their own token mean, then the two means are averaged."""
    split = _traj([_turn([1, 2], [3, 4, 5]), _turn([9], [8])])
    whole = _traj([_turn([1], [2, 3])])
    batch, _ = build_train_batch(
        [split, whole], [1.0, 1.0], pad_token_id=PAD, max_seq_len=64, loss_agg="rollout"
    )
    assert batch.loss_weights is not None
    assert batch.loss_weights.tolist() == [
        [0, 0, 0.25, 0.25, 0.25],
        [0, 0.25, 0, 0, 0],
        [0, 0.5, 0.5, 0, 0],
    ]
    values = torch.tensor([[0, 0, 1.0, 2.0, 3.0], [0, 6.0, 0, 0, 0], [0, 10.0, 20.0, 0, 0]])
    loss = masked_aggregate(values, batch.response_mask, "rollout", batch.loss_weights)
    assert loss.item() == pytest.approx(((1 + 2 + 3 + 6) / 4 + (10 + 20) / 2) / 2)
    # per-sequence aggregation would count the split trajectory twice
    per_sequence = masked_aggregate(values, batch.response_mask, "sequence")
    assert per_sequence.item() == pytest.approx(((1 + 2 + 3) / 3 + 6 + (10 + 20) / 2) / 3)
    # without weights every sequence is its own rollout
    assert masked_aggregate(values, batch.response_mask, "rollout").item() == pytest.approx(
        per_sequence.item()
    )

    token_batch, _ = build_train_batch([split, whole], [1.0, 1.0], pad_token_id=PAD, max_seq_len=64)
    assert token_batch.loss_weights is None


def test_rollout_weights_follow_truncation() -> None:
    traj = _traj([_turn([1, 2, 3], [4, 5, 6, 7])])
    batch, _ = build_train_batch([traj], [1.0], pad_token_id=PAD, max_seq_len=5, loss_agg="rollout")
    assert batch.loss_weights.tolist() == [[0, 0, 0, 0.5, 0.5]]


def test_truncation_to_max_seq_len() -> None:
    traj = _traj([_turn([1, 2, 3], [4, 5, 6, 7])])
    batch, stats = build_train_batch([traj], [1.0], pad_token_id=PAD, max_seq_len=5)
    assert batch.input_ids.tolist() == [[1, 2, 3, 4, 5]]
    assert batch.response_mask.tolist() == [[0, 0, 0, 1, 1]]
    assert batch.old_logprobs.shape == (1, 5)
    assert stats["tokens/truncated"] == 1.0


def test_missing_logprobs_yield_none_old_logprobs() -> None:
    ok = _traj([_turn([1], [2, 3])])
    missing = _traj(
        [Turn(prompt_token_ids=[1], completion=Completion(text="", token_ids=[2], logprobs=None))]
    )
    batch, _ = build_train_batch([ok, missing], [0.0, 0.0], pad_token_id=PAD, max_seq_len=8)
    assert batch.old_logprobs is None
    misaligned = _traj([_turn([1], [2, 3], [-1.0])])  # 1 logprob for 2 tokens
    assert flatten_trajectory(misaligned).old_logprobs is None
