"""Real-engine tests: run only where vLLM is installed and a CUDA/ROCm GPU is present."""

from __future__ import annotations

import pytest
import torch

vllm = pytest.importorskip("vllm")

from monorl.config.schema import RunConfig  # noqa: E402
from monorl.core.registry import registry  # noqa: E402
from monorl.core.types import SamplingParams  # noqa: E402
from monorl.patching import apply, status  # noqa: E402
from monorl.rollout.hf_engine import iter_hf_weights  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM engine needs a CUDA device"),
]


def test_patches_apply_to_installed_vllm() -> None:
    report = apply("vllm")
    assert report.failed == [], report
    assert {s.id for s in status("vllm") if s.applied} >= {"vllm.llm.sleep_guard", "vllm.outputs.sampled_logprobs"}


def test_colocated_engine_roundtrip(vllm_config: RunConfig, tiny_model_dir) -> None:
    vllm_config.rollout.tensor_parallel_size = 1
    vllm_config.rollout.gpu_memory_utilization = 0.3
    vllm_config.rollout.extra = {"enforce_eager": True, "max_num_seqs": 8}
    eng = registry.get("rollout_engine", "vllm")(vllm_config)
    try:
        chat = [{"role": "user", "content": "Say hi."}]
        params = SamplingParams(n=2, max_new_tokens=8, temperature=1.0, logprobs=True, seed=0)
        out = eng.generate([chat, [1, 2, 3, 4]], params)
        assert [len(g) for g in out] == [2, 2]
        for group in out:
            for c in group:
                assert 1 <= len(c.token_ids) <= 8
                assert c.logprobs is not None and len(c.logprobs) == len(c.token_ids)
                assert all(lp <= 0.0 for lp in c.logprobs)
                assert c.finish_reason in {"stop", "length"}
        assert out[1][0].prompt_token_ids == [1, 2, 3, 4]

        eng.update_weights(iter_hf_weights(tiny_model_dir))
        eng.load_weights_from_path(tiny_model_dir)
        eng.sleep()
        eng.wake_up()
        again = eng.generate([[1, 2, 3, 4]], SamplingParams(n=1, max_new_tokens=4, logprobs=False))
        assert len(again[0]) == 1 and again[0][0].logprobs is None
    finally:
        eng.shutdown()
