"""Multimodal trajectories through the orchestrator seams: episodes send chats to the engine on
every turn, and ``build_train_batch`` re-encodes them through the processor, verifies the ids
against what the engine conditioned on and fills ``TrainBatch.extra``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fakes
import pytest
import torch

from monorl.core.errors import MonorlError
from monorl.core.types import Completion, Sample, SamplingParams, Trajectory, Turn
from monorl.orchestrator.batching import build_train_batch, encode_trajectory_media, final_prompt_messages
from monorl.orchestrator.episodes import EpisodeSpec, run_episode
from monorl.rollout.media import vlm_prompt_ids
from tests.rollout.tiny_vlm import image_chat, load_vlm_processor, write_test_images


@pytest.fixture(scope="module")
def processor():
    return load_vlm_processor()


@pytest.fixture(scope="module")
def images(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return write_test_images(tmp_path_factory.mktemp("images"))


def _completion(tokenizer: Any, text: str, *, logprob: float = -0.5) -> Completion:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return Completion(text=text, token_ids=ids, logprobs=[logprob] * len(ids), finish_reason="stop")


def _trajectory(processor: Any, chat: list[dict], answers: list[str], *, group_id: str = "g") -> Trajectory:
    """Multi-turn trajectory whose prompts were produced by the processor recipe (as the HF engine
    does) and whose follow-up user turns come from the environment."""
    tok = processor.tokenizer
    messages = list(chat)
    turns = []
    for k, answer in enumerate(answers):
        prompt_ids = vlm_prompt_ids(processor, messages)
        completion = _completion(tok, answer)
        completion.prompt_token_ids = list(prompt_ids)
        turns.append(Turn(prompt_token_ids=prompt_ids, completion=completion, reward=0.0))
        messages.append({"role": "assistant", "content": answer})
        if k < len(answers) - 1:
            messages.append({"role": "user", "content": f"again{k}"})
    messages.append({"role": "user", "content": "env epilogue after the final step"})
    return Trajectory(
        sample=Sample(id="s", messages=chat), group_id=group_id, messages=messages, turns=turns, reward=1.0
    )


def test_final_prompt_messages_excludes_trailing_env_messages(
    processor: Any, images: dict[str, Path]
) -> None:
    traj = _trajectory(processor, image_chat(images["red"]), ["Red.", "Yes."])
    prompt = final_prompt_messages(traj)
    assert [m["role"] for m in prompt] == ["user", "assistant", "user"]
    assert prompt[0]["content"][0]["type"] == "image"
    assert vlm_prompt_ids(processor, prompt) == traj.turns[-1].prompt_token_ids


def test_build_train_batch_masks_exactly_the_completion_span(processor: Any, images: dict[str, Path]) -> None:
    tok = processor.tokenizer
    single = _trajectory(processor, image_chat(images["blue"]), ["Blue."], group_id="a")
    multi = _trajectory(processor, image_chat(images["red"]), ["Red.", "Yes."], group_id="b")
    batch, stats = build_train_batch(
        [single, multi], [1.0, -1.0], pad_token_id=tok.pad_token_id, max_seq_len=256, processor=processor
    )

    assert stats["tokens/prefix_mismatch"] == 0
    image_id = (
        int(processor.image_token_id)
        if hasattr(processor, "image_token_id")
        else tok.convert_tokens_to_ids("<|image_pad|>")
    )
    for b, traj in enumerate((single, multi)):
        last = traj.turns[-1]
        seq = list(last.prompt_token_ids) + list(last.completion.token_ids)
        n = len(seq)
        assert batch.input_ids[b, :n].tolist() == seq
        assert batch.attention_mask[b].sum() == n
        expected_mask = [0] * n
        for turn in traj.turns:
            start = len(turn.prompt_token_ids)
            expected_mask[start : start + len(turn.completion.token_ids)] = [1] * len(
                turn.completion.token_ids
            )
        assert batch.response_mask[b].tolist() == expected_mask + [0] * (batch.input_ids.shape[1] - n)
        # placeholder tokens are prompt positions, never loss positions
        assert torch.all(batch.response_mask[b][batch.input_ids[b] == image_id] == 0)
    assert batch.response_mask[1].sum() == sum(len(t.completion.token_ids) for t in multi.turns)
    assert torch.all(batch.old_logprobs[batch.response_mask == 1] == -0.5)
    assert torch.equal(batch.advantages[0], batch.response_mask[0].float())
    assert torch.equal(batch.advantages[1], -batch.response_mask[1].float())

    # media tensors: one dict per sequence, token-aligned type ids padded to the batch width
    assert batch.position_ids is None
    per_seq = batch.extra["per_sequence"]
    assert per_seq[0]["pixel_values"].shape[0] == 32 and per_seq[1]["pixel_values"].shape[0] == 16
    assert per_seq[0]["image_grid_thw"].tolist() == [[1, 4, 8]] and per_seq[1]["image_grid_thw"].tolist() == [
        [1, 4, 4]
    ]
    mm = batch.extra["mm_token_type_ids"]
    assert mm.shape == batch.input_ids.shape
    assert torch.equal(mm == 1, batch.input_ids == image_id)
    assert batch.slice(1, 2).extra["per_sequence"] == [per_seq[1]]


def test_build_train_batch_without_processor_is_text_only(processor: Any, images: dict[str, Path]) -> None:
    traj = _trajectory(processor, image_chat(images["red"]), ["Red."])
    batch, _ = build_train_batch(
        [traj], [1.0], pad_token_id=processor.tokenizer.pad_token_id, max_seq_len=256
    )
    assert batch.extra == {}
    assert batch.position_ids is not None
    text_traj = _trajectory(processor, [{"role": "user", "content": "no image"}], ["ok"])
    batch, _ = build_train_batch(
        [text_traj],
        [1.0],
        pad_token_id=processor.tokenizer.pad_token_id,
        max_seq_len=256,
        processor=processor,
    )
    assert batch.extra == {} and batch.position_ids is not None
    assert encode_trajectory_media(text_traj, processor) is None


def test_build_train_batch_rejects_misaligned_prompt_ids(processor: Any, images: dict[str, Path]) -> None:
    traj = _trajectory(processor, image_chat(images["red"]), ["Red."])
    traj.turns[-1].prompt_token_ids = traj.turns[-1].prompt_token_ids[
        1:
    ]  # engine conditioned on something else
    with pytest.raises(ValueError, match="differs from the engine"):
        build_train_batch(
            [traj], [1.0], pad_token_id=processor.tokenizer.pad_token_id, max_seq_len=256, processor=processor
        )


def test_truncation_clips_token_aligned_extra(processor: Any, images: dict[str, Path]) -> None:
    traj = _trajectory(processor, image_chat(images["red"]), ["Red."])
    prompt_len = len(traj.turns[0].prompt_token_ids)
    batch, stats = build_train_batch(
        [traj],
        [1.0],
        pad_token_id=processor.tokenizer.pad_token_id,
        max_seq_len=prompt_len - 2,
        processor=processor,
    )
    assert stats["tokens/truncated"] == 1
    assert batch.input_ids.shape[1] == prompt_len - 2
    assert batch.extra["mm_token_type_ids"].shape == batch.input_ids.shape


class _RecordingEngine:
    """Fake engine that remembers whether each prompt arrived as a chat or as token ids."""

    def __init__(self, inner: fakes.FakeEngine) -> None:
        self.inner = inner
        self.prompt_kinds: list[str] = []

    @property
    def tokenizer(self) -> Any:
        return self.inner.tokenizer

    def generate(self, prompts: Any, params: SamplingParams) -> list[list[Completion]]:
        self.prompt_kinds.extend("chat" if p and isinstance(p[0], dict) else "ids" for p in prompts)
        return self.inner.generate(prompts, params)


def _image_sample() -> Sample:
    return Sample(
        id="img",
        messages=[
            {
                "role": "user",
                "content": [{"type": "image", "image": "/tmp/x.png"}, {"type": "text", "text": "abc"}],
            }
        ],
        reference="ABC",
    )


def test_multimodal_episode_sends_chat_every_turn(make_config: Any) -> None:
    cfg = make_config(
        env={"type": "fake_multi", "max_turns": 3}, rollout={"extra": {"fake_mode": "always_right"}}
    )
    engine = _RecordingEngine(fakes.FakeEngine(cfg))
    traj = run_episode(
        EpisodeSpec(sample=_image_sample(), group_id="g", engine=engine),
        env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1),
    )
    assert len(traj.turns) == 3
    assert engine.prompt_kinds == ["chat", "chat", "chat"]
    assert all(t.prompt_token_ids == t.completion.prompt_token_ids for t in traj.turns)
    assert traj.messages[0]["content"][0] == {"type": "image", "image": "/tmp/x.png"}


def test_text_episode_sends_token_ids_on_every_turn(make_config: Any) -> None:
    cfg = make_config(
        env={"type": "fake_multi", "max_turns": 2}, rollout={"extra": {"fake_mode": "always_right"}}
    )
    engine = _RecordingEngine(fakes.FakeEngine(cfg))
    run_episode(
        EpisodeSpec(
            sample=Sample(id="t", messages=[{"role": "user", "content": "abc"}], reference="ABC"),
            group_id="g",
            engine=engine,
        ),
        env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1),
    )
    assert engine.prompt_kinds == ["ids", "ids"]


def test_multimodal_episode_rejects_tool_schemas(make_config: Any) -> None:
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    cfg = make_config(env={"type": "fake_multi", "max_turns": 2, "params": {"tools": tools}})
    engine = fakes.FakeEngine(cfg)
    with pytest.raises(MonorlError, match="tool schemas"):
        run_episode(
            EpisodeSpec(sample=_image_sample(), group_id="g", engine=engine),
            env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
            tokenizer=engine.tokenizer,
            params=SamplingParams(n=1),
        )


def test_multimodal_episode_runs_against_env_with_empty_tool_list(make_config: Any) -> None:
    """Remote and OpenEnv environments expose ``tools = []`` when they define none."""
    cfg = make_config(
        env={"type": "fake_multi", "max_turns": 2, "params": {"tools": []}},
        rollout={"extra": {"fake_mode": "always_right"}},
    )
    engine = _RecordingEngine(fakes.FakeEngine(cfg))
    traj = run_episode(
        EpisodeSpec(sample=_image_sample(), group_id="g", engine=engine),
        env_factory=lambda: fakes.FakeMultiTurnEnv(cfg),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1),
    )
    assert len(traj.turns) == 2
    assert engine.prompt_kinds == ["chat", "chat"]
