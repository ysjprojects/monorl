from __future__ import annotations

from pathlib import Path

import pytest

from monorl.config.schema import RunConfig
from tests.rollout.tiny_vlm import build_tiny_vlm, write_test_images


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


@pytest.fixture(scope="session")
def tiny_vlm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Random tiny Qwen2-VL + real processor (see ``tests/rollout/tiny_vlm.py``)."""
    return build_tiny_vlm(tmp_path_factory.mktemp("tiny-qwen2vl"))


@pytest.fixture(scope="session")
def test_images(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return write_test_images(tmp_path_factory.mktemp("images"))


@pytest.fixture(scope="module")
def vlm_config(tiny_vlm_dir: Path) -> RunConfig:
    return RunConfig.from_dict(
        {
            "model": {"name_or_path": str(tiny_vlm_dir), "dtype": "float32"},
            "data": {"uri": "unused.jsonl"},
            "rollout": {"backend": "hf", "group_size": 2, "max_new_tokens": 8, "device": "cpu"},
            "trainer": {"backend": "reference", "device": "cpu", "precision": "fp32"},
            "rewards": [{"type": "exact_match"}],
        }
    )


@pytest.fixture(scope="module")
def vlm_engine(vlm_config: RunConfig):
    from monorl.rollout.hf_engine import HFRolloutEngine

    eng = HFRolloutEngine(vlm_config)
    yield eng
    eng.shutdown()
