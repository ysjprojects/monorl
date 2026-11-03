from __future__ import annotations

from pathlib import Path

import pytest

from monorl.config.schema import RunConfig


@pytest.fixture(scope="module")
def tiny_config(tiny_model_dir: Path) -> RunConfig:
    return RunConfig.from_dict(
        {
            "model": {"name_or_path": str(tiny_model_dir), "dtype": "float32"},
            "data": {"uri": "unused.jsonl"},
            "rollout": {"backend": "hf", "group_size": 2, "max_new_tokens": 8, "device": "cpu"},
            "trainer": {"backend": "reference", "device": "cpu", "precision": "fp32"},
            "rewards": [{"type": "exact_match"}],
        }
    )


@pytest.fixture(scope="module")
def engine(tiny_config: RunConfig):
    from monorl.rollout.hf_engine import HFRolloutEngine

    eng = HFRolloutEngine(tiny_config)
    yield eng
    eng.shutdown()
