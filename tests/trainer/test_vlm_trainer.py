"""Reference trainer on a tiny Qwen2-VL: ``TrainBatch.extra`` media tensors flow through
``compute_logprobs``/``train_step`` (micro-batch slicing included) and match the engine."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from monorl.core.batch import TrainBatch
from monorl.core.types import Sample, SamplingParams, Trajectory, Turn
from monorl.orchestrator.batching import build_train_batch
from monorl.rollout.hf_engine import HFRolloutEngine
from monorl.trainer.reference import ReferenceTrainer
from tests.rollout.tiny_vlm import image_chat
from tests.trainer.conftest import make_config, reinforce_loss

TEXT_CHAT = [{"role": "user", "content": "Say hello."}]


@pytest.fixture(scope="module")
def vlm_engine(tiny_vlm_dir: Path):
    eng = HFRolloutEngine(make_config(tiny_vlm_dir))
    yield eng
    eng.shutdown()


@pytest.fixture(scope="module")
def vlm_batch(vlm_engine: HFRolloutEngine, test_images: dict[str, Path]) -> TrainBatch:
    """Two image prompts (different patch counts) + one text prompt, n=2 each, sampled by the
    engine; advantages +1 everywhere so the REINFORCE loss pushes sampled tokens up."""
    chats = [image_chat(test_images["blue"]), image_chat(test_images["red"], "Short answer."), TEXT_CHAT]
    out = vlm_engine.generate(chats, SamplingParams(n=2, max_new_tokens=6, temperature=1.0, seed=3))
    trajectories = []
    for i, (chat, group) in enumerate(zip(chats, out, strict=True)):
        for c in group:
            trajectories.append(
                Trajectory(
                    sample=Sample(id=f"s{i}", messages=chat),
                    group_id=f"g{i}",
                    messages=list(chat) + [{"role": "assistant", "content": c.text}],
                    turns=[Turn(prompt_token_ids=list(c.prompt_token_ids), completion=c)],
                    reward=1.0,
                )
            )
    batch, stats = build_train_batch(
        trajectories, [1.0] * len(trajectories), pad_token_id=vlm_engine.tokenizer.pad_token_id, max_seq_len=128, processor=vlm_engine.processor
    )
    assert stats["tokens/prefix_mismatch"] == 0
    return batch


def test_batch_carries_media_tensors(vlm_batch: TrainBatch, vlm_engine: HFRolloutEngine) -> None:
    assert vlm_batch.position_ids is None
    assert "per_sequence" in vlm_batch.extra and "mm_token_type_ids" in vlm_batch.extra
    per_seq = vlm_batch.extra["per_sequence"]
    assert len(per_seq) == len(vlm_batch) == 6
    assert per_seq[0]["pixel_values"].shape[0] == 32 and per_seq[2]["pixel_values"].shape[0] == 16
    assert per_seq[4] is None and per_seq[5] is None  # text prompts
    mm = vlm_batch.extra["mm_token_type_ids"]
    image_id = int(vlm_engine.model.config.image_token_id)
    assert mm.shape == vlm_batch.input_ids.shape
    assert torch.equal(mm == 1, vlm_batch.input_ids == image_id)
    # slice/to keep extras aligned
    sl = vlm_batch.slice(2, 5)
    assert len(sl.extra["per_sequence"]) == 3 and sl.extra["per_sequence"][0] is per_seq[2]
    assert sl.extra["mm_token_type_ids"].shape[0] == 3
    moved = vlm_batch.to("cpu")
    assert moved.extra["per_sequence"][0]["pixel_values"].device.type == "cpu"


@pytest.mark.parametrize("micro_batch_size", [1, 2, 6])
def test_compute_logprobs_matches_engine(tiny_vlm_dir: Path, vlm_batch: TrainBatch, micro_batch_size: int) -> None:
    trainer = ReferenceTrainer(make_config(tiny_vlm_dir, micro_batch_size=micro_batch_size, max_seq_len=128), loss_fn=reinforce_loss)
    assert type(trainer.hf_model).__name__ == "Qwen2VLForConditionalGeneration"
    lp = trainer.compute_logprobs(vlm_batch)
    assert lp.shape == vlm_batch.input_ids.shape
    mask = vlm_batch.response_mask.bool()
    assert torch.all(lp[~mask] == 0)
    torch.testing.assert_close(lp[mask], vlm_batch.old_logprobs[mask], atol=1e-4, rtol=0.0)


def test_lora_train_step_decreases_loss(tiny_vlm_dir: Path, vlm_batch: TrainBatch) -> None:
    trainer = ReferenceTrainer(make_config(tiny_vlm_dir, lora=True, micro_batch_size=2, max_seq_len=128), loss_fn=reinforce_loss)
    assert all("lora_" in n for n, p in trainer.model.named_parameters() if p.requires_grad)
    before = trainer.compute_logprobs(vlm_batch)
    losses = []
    for i in range(5):
        metrics = trainer.train_step(vlm_batch)
        losses.append(metrics["loss/policy"])
        assert trainer.step == i + 1
        assert metrics["grad_norm"] > 0
    assert losses[-1] < losses[0]
    after = trainer.compute_logprobs(vlm_batch)
    mask = vlm_batch.response_mask.bool()
    assert after[mask].mean() > before[mask].mean()
    # reference (adapters disabled) still equals the sampling policy
    ref = trainer.compute_logprobs(vlm_batch, reference=True)
    torch.testing.assert_close(ref[mask], vlm_batch.old_logprobs[mask], atol=1e-4, rtol=0.0)


def test_prepare_truncates_token_aligned_extra(tiny_vlm_dir: Path, vlm_batch: TrainBatch) -> None:
    width = vlm_batch.input_ids.shape[1]
    trainer = ReferenceTrainer(make_config(tiny_vlm_dir, max_seq_len=width - 3), loss_fn=reinforce_loss)
    prepared = trainer._prepare(vlm_batch)
    assert prepared.input_ids.shape[1] == width - 3
    assert prepared.extra["mm_token_type_ids"].shape[1] == width - 3
    assert len(prepared.extra["per_sequence"]) == len(vlm_batch)
