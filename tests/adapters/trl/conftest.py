"""Fixtures for the TRL pipeline backend tests: a 4-row jsonl, an inline reward, a 2-turn env."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from monorl.config.schema import RunConfig
from monorl.core.registry import registry

pytest.importorskip("trl")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from trl_support import ENV_NAME, REWARD_NAME, TwoTurnEnv, VowelReward  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _register_test_components() -> Iterator[None]:
    registry.register("reward", REWARD_NAME, lambda params, **kw: VowelReward(), overwrite=True)
    registry.register("env", ENV_NAME, lambda config, **kw: TwoTurnEnv(**config.env.params), overwrite=True)
    yield


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.jsonl"
    rows = [
        {"prompt": "What is 2+2?", "answer": "4"},
        {"prompt": "What is 3+5?", "answer": "8"},
        {"prompt": "What is 10-4?", "answer": "6"},
        {"prompt": "What is 7*3?", "answer": "21"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


@pytest.fixture
def trl_config(tiny_model_dir: Path, data_file: Path, tmp_path: Path) -> RunConfig:
    return RunConfig.from_dict(
        {
            "run_name": "trl-test",
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
                "backend": "trl",
                "prompts_per_step": 2,
                "micro_batch_size": 2,
                "max_steps": 2,
                "precision": "fp32",
                "device": "cpu",
                "save_every": 1,
                "max_seq_len": 256,
            },
            "rewards": [{"type": REWARD_NAME, "weight": 1.0}],
            "logging": {"run_root": str(tmp_path / "runs"), "trajectories_per_step": 4},
        }
    )
