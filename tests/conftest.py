"""Shared fixtures. Workstreams add their own ``tests/<area>/conftest.py``; do not grow this file.

* ``tiny_model_dir`` — a randomly initialised ~1M-parameter Llama with the SmolLM tokenizer,
  saved to a session-scoped temp dir. Loadable by ``AutoModelForCausalLM`` / any engine,
  no network required (tokenizer comes from the HF cache; see ``TINY_TOKENIZER``).
* ``smol_model_id`` — a real 135M instruct model for ``@pytest.mark.slow`` end-to-end tests.
* ``run_config_dict`` — minimal valid :class:`RunConfig` payload for tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

TINY_TOKENIZER = "HuggingFaceTB/SmolLM-135M-Instruct"
SMOL_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MONORL_HOME", str(Path("/tmp") / "monorl-tests"))


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    out = tmp_path_factory.mktemp("tiny-llama")
    tok = AutoTokenizer.from_pretrained(TINY_TOKENIZER)
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def smol_model_id() -> str:
    return SMOL_MODEL_ID


@pytest.fixture
def run_config_dict(tiny_model_dir: Path, tmp_path: Path) -> dict[str, Any]:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(
        "\n".join(
            [
                '{"prompt": "What is 2+2?", "answer": "4"}',
                '{"prompt": "What is 3+5?", "answer": "8"}',
                '{"prompt": "What is 10-4?", "answer": "6"}',
                '{"prompt": "What is 7*3?", "answer": "21"}',
            ]
        )
        + "\n"
    )
    return {
        "run_name": "test-run",
        "model": {"name_or_path": str(tiny_model_dir), "dtype": "float32"},
        "data": {
            "uri": str(data_file),
            "prompt_field": "prompt",
            "answer_field": "answer",
            "max_prompt_tokens": 128,
        },
        "algorithm": {"name": "grpo"},
        "rollout": {"backend": "hf", "group_size": 2, "max_new_tokens": 8, "device": "cpu"},
        "trainer": {
            "backend": "reference",
            "prompts_per_step": 2,
            "micro_batch_size": 2,
            "max_steps": 2,
            "precision": "fp32",
            "device": "cpu",
            "save_every": 1,
            "max_seq_len": 256,
        },
        "rewards": [{"type": "exact_match", "weight": 1.0}],
        "logging": {"run_root": str(tmp_path / "runs")},
    }
