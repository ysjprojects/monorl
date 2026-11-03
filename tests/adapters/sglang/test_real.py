"""End-to-end against a real sglang install (needs CUDA). Skipped everywhere else."""

from __future__ import annotations

import pytest
import torch

sglang = pytest.importorskip("sglang")

from fake_sglang import make_config  # noqa: E402

from monorl.adapters.sglang.engine import SGLangEngine  # noqa: E402
from monorl.core.types import SamplingParams  # noqa: E402
from monorl.patching import apply, status  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def real_engine(smol_model_id, tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("sglang needs CUDA")
    data = tmp_path_factory.mktemp("data") / "d.jsonl"
    data.write_text('{"prompt": "What is 2+2?", "answer": "4"}\n')
    cfg = make_config(
        {
            "run_name": "sglang-real",
            "model": {"name_or_path": smol_model_id, "dtype": "bfloat16"},
            "data": {"uri": str(data)},
            "algorithm": {"name": "grpo"},
            "rollout": {"backend": "sglang", "group_size": 2, "max_new_tokens": 8},
            "trainer": {"backend": "reference", "max_steps": 1},
            "rewards": [{"type": "exact_match"}],
            "logging": {"run_root": str(tmp_path_factory.mktemp("runs"))},
        },
        gpu_memory_utilization=0.3,
        max_model_len=512,
        enable_sleep_mode=True,
        extra={"disable_cuda_graph": True},
    )
    engine = SGLangEngine(cfg)
    yield engine
    engine.shutdown()


def test_patches_apply_on_real_sglang():
    report = apply("sglang")
    assert report.ok, report
    assert all(s.applied for s in status("sglang") if s.applicable)


def test_real_generate_chat_and_tokens(real_engine):
    params = SamplingParams(n=2, max_new_tokens=8, temperature=1.0, seed=1)
    chat = [{"role": "user", "content": "Say hello."}]
    groups = real_engine.generate([chat, [1, 2, 3, 4]], params)
    assert [len(g) for g in groups] == [2, 2]
    for g in groups:
        for c in g:
            assert 0 < len(c.token_ids) <= 8
            assert c.logprobs is not None and len(c.logprobs) == len(c.token_ids)
            assert all(lp <= 0.0 for lp in c.logprobs)
            assert c.finish_reason in ("stop", "length")
    assert groups[1][0].prompt_token_ids == [1, 2, 3, 4]
    assert groups[0][0].token_ids != groups[0][1].token_ids  # distinct per-sample seeds


def test_real_update_weights_sleep_wake(real_engine):
    from transformers import AutoConfig

    name = "model.embed_tokens.weight"
    hf_cfg = AutoConfig.from_pretrained(real_engine.config.model.name_or_path)
    assert real_engine.engine.get_weights_by_name(name, truncate_size=1) is not None
    real_engine.sleep()
    real_engine.wake_up()
    w = torch.zeros(hf_cfg.vocab_size, hf_cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
    real_engine.update_weights([(name, w)])
    groups = real_engine.generate([[1, 2, 3]], SamplingParams(n=1, max_new_tokens=2))
    assert len(groups[0][0].token_ids) >= 1
