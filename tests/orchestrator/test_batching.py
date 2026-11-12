from __future__ import annotations

import torch

from monorl.core.types import Completion, Sample, Trajectory, Turn
from monorl.orchestrator.batching import build_train_batch, flatten_trajectory, prefix_chain_ok

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
    assert batch.ref_logprobs is None
    assert stats["tokens/response_mean"] == 1.5
    assert stats["tokens/prompt_mean"] == 2.5
    assert stats["tokens/prefix_mismatch"] == 0
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
    assert batch.advantages[0].tolist() == [0, 0, 2.0, 2.0, 0, 0, 0, 2.0, 2.0, 2.0]
    assert stats["tokens/response_mean"] == 5.0


def test_prefix_mismatch_masks_only_final_completion() -> None:
    turns = [
        _turn([1, 2], [3, 4]),
        _turn([1, 2, 3, 99, 5, 6, 7], [8, 9]),  # 99 != 4 -> chain broken
    ]
    traj = _traj(turns)
    assert not prefix_chain_ok(traj)
    flat = flatten_trajectory(traj)
    assert flat.input_ids == [1, 2, 3, 99, 5, 6, 7, 8, 9]
    assert flat.response_mask == [0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert flat.prefix_mismatch
    _, stats = build_train_batch(
        [traj, _traj([_turn([1], [2])])], [1.0, 1.0], pad_token_id=PAD, max_seq_len=64
    )
    assert stats["tokens/prefix_mismatch"] == 1.0


def test_shorter_next_prompt_is_a_mismatch() -> None:
    traj = _traj([_turn([1, 2, 3], [4, 5]), _turn([1, 2], [6])])
    assert not prefix_chain_ok(traj)


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
