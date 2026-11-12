from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch


@dataclass
class SimpleLossOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


def reinforce_loss(logprobs, old_logprobs, advantages, response_mask, *, algorithm, ref_logprobs=None, entropy=None):
    """Token-mean REINFORCE: ``-mean(adv * logprob)`` over response positions."""
    mask = response_mask.float()
    loss = -(logprobs * advantages * mask).sum() / mask.sum().clamp(min=1)
    return SimpleLossOutput(loss=loss, metrics={"loss/policy": float(loss.detach().item())})


def make_config(tiny_model_dir: Path, *, lora: bool = False, **trainer: Any) -> RunConfig:
    trainer_cfg = {
        "backend": "reference",
        "device": "cpu",
        "precision": "fp32",
        "micro_batch_size": 2,
        "max_steps": 10,
        "learning_rate": 1e-2 if lora else 1e-3,
        "max_seq_len": 64,
    }
    trainer_cfg.update(trainer)
    return RunConfig.from_dict(
        {
            "seed": 0,
            "model": {
                "name_or_path": str(tiny_model_dir),
                "dtype": "float32",
                "lora": {"enabled": lora, "r": 4, "alpha": 8, "dropout": 0.0},
            },
            "data": {"uri": "unused.jsonl"},
            "rollout": {"backend": "hf", "group_size": 2, "max_new_tokens": 8, "device": "cpu"},
            "trainer": trainer_cfg,
            "rewards": [{"type": "exact_match"}],
        }
    )


def make_batch(tokenizer: Any, *, target: int = 1234, batch_size: int = 4, response_len: int = 5) -> TrainBatch:
    """Right-padded synthetic batch: chat prompts of different lengths, responses of ``target``
    repeated, advantage +1 on every response token."""
    from monorl.rollout.hf_engine import chat_prompt_ids

    prompts = [
        chat_prompt_ids(tokenizer, [{"role": "user", "content": f"Question number {i}: " + "pad " * (i % 3)}])
        for i in range(batch_size)
    ]
    responses = [[target] * (response_len - (i % 2)) + [tokenizer.eos_token_id] for i in range(batch_size)]
    width = max(len(p) + len(r) for p, r in zip(prompts, responses, strict=True))
    pad = tokenizer.pad_token_id
    input_ids = torch.full((batch_size, width), pad, dtype=torch.long)
    attention = torch.zeros((batch_size, width), dtype=torch.long)
    response_mask = torch.zeros((batch_size, width), dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompts, responses, strict=True)):
        seq = p + r
        input_ids[i, : len(seq)] = torch.tensor(seq)
        attention[i, : len(seq)] = 1
        response_mask[i, len(p) : len(seq)] = 1
    advantages = response_mask.float()
    return TrainBatch(
        input_ids=input_ids,
        attention_mask=attention,
        response_mask=response_mask,
        advantages=advantages,
        rewards=torch.ones(batch_size),
        group_ids=[f"g{i // 2}" for i in range(batch_size)],
    )


@pytest.fixture(scope="session")
def tokenizer(tiny_model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model_dir)


@pytest.fixture(scope="session")
def tiny_vlm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.rollout.tiny_vlm import build_tiny_vlm

    return build_tiny_vlm(tmp_path_factory.mktemp("tiny-qwen2vl"))


@pytest.fixture(scope="session")
def test_images(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from tests.rollout.tiny_vlm import write_test_images

    return write_test_images(tmp_path_factory.mktemp("images"))
