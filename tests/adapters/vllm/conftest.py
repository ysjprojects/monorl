"""Fixtures: install the fake vLLM tree (``vllm_fakes``) in ``sys.modules`` for one test and
revert any monorl patches applied to it; ``no_vllm`` makes ``import vllm`` fail."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from monorl.config.schema import RunConfig

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from vllm_fakes import FakeLLM, FakePlatform, build_fake_vllm  # noqa: E402


@pytest.fixture
def fake_vllm(monkeypatch: pytest.MonkeyPatch):
    from monorl.patching import revert

    tree = build_fake_vllm()
    for name, m in tree.items():
        monkeypatch.setitem(sys.modules, name, m)
    FakeLLM.instances = []
    FakeLLM.finish_reasons = ["stop"]
    FakeLLM.extra_logprob_entries = 0
    FakePlatform.sleep_mode_available = True
    monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
    monkeypatch.delenv("MONORL_DISABLE_PATCHES", raising=False)
    try:
        yield tree
    finally:
        revert("vllm")


@pytest.fixture
def no_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import vllm`` fail even if a real vLLM is installed."""
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "vllm", None)


@pytest.fixture
def vllm_config(tiny_model_dir: Path, tmp_path: Path) -> RunConfig:
    data_file = tmp_path / "data.jsonl"
    data_file.write_text('{"prompt": "What is 2+2?", "answer": "4"}\n')
    return RunConfig.from_dict(
        {
            "seed": 11,
            "model": {"name_or_path": str(tiny_model_dir), "dtype": "bfloat16"},
            "data": {"uri": str(data_file)},
            "rollout": {
                "backend": "vllm",
                "group_size": 2,
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.5,
                "max_model_len": 1024,
                "enable_sleep_mode": True,
            },
            "rewards": [{"type": "exact_match"}],
            "logging": {"run_root": str(tmp_path / "runs")},
        }
    )
