from __future__ import annotations

import math
import os

import pytest
import torch
from vllm_fakes import GPU_WORKER_ATTRS, CompletionOutput, FakeLLM, FakePlatform, Logprob

from monorl.adapters.vllm.common import WORKER_EXTENSION_CLS, chunk_weights, sampling_kwargs
from monorl.adapters.vllm.worker import MonorlWorkerExtension
from monorl.config.schema import RunConfig
from monorl.core.errors import BackendUnavailable, ConfigError
from monorl.core.interfaces import RolloutEngine
from monorl.core.registry import registry
from monorl.core.types import SamplingParams
from monorl.rollout.hf_engine import chat_prompt_ids


def _engine(config: RunConfig, **kw):
    return registry.get("rollout_engine", "vllm")(config, **kw)


# -- registry / availability --------------------------------------------------------------------


def test_registry_resolves_and_raises_backend_unavailable(no_vllm, vllm_config: RunConfig) -> None:
    factory = registry.get("rollout_engine", "vllm")
    with pytest.raises(BackendUnavailable) as ei:
        factory(vllm_config)
    msg = str(ei.value)
    assert "monorl[vllm]" in msg and "rollout.extra.mode='server'" in msg


def test_unknown_mode_rejected(vllm_config: RunConfig) -> None:
    vllm_config.rollout.extra = {"mode": "bogus"}
    with pytest.raises(ConfigError, match="rollout.extra.mode"):
        _engine(vllm_config)


# -- construction ---------------------------------------------------------------------------------


def test_llm_constructed_with_pinned_kwargs(fake_vllm, vllm_config: RunConfig) -> None:
    vllm_config.rollout.extra = {"max_num_seqs": 64, "engine_kwargs": {"enforce_eager": True}}
    eng = _engine(vllm_config, replica=3)
    assert isinstance(eng, RolloutEngine)
    llm = FakeLLM.instances[-1]
    assert llm.model == vllm_config.model.name_or_path
    kw = llm.kwargs
    assert kw["worker_extension_cls"] == WORKER_EXTENSION_CLS
    assert kw["tensor_parallel_size"] == 2
    assert kw["gpu_memory_utilization"] == 0.5
    assert kw["max_model_len"] == 1024
    assert kw["dtype"] == "bfloat16"
    assert kw["enable_sleep_mode"] is True
    assert kw["seed"] == 11 + 3
    assert kw["logprobs_mode"] == "processed_logprobs"
    assert kw["max_num_seqs"] == 64
    assert kw["enforce_eager"] is True  # engine_kwargs override
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert eng.tokenizer.chat_template  # HF tokenizer loaded for prompt formatting


def test_sleep_mode_disabled_when_platform_lacks_it(fake_vllm, vllm_config: RunConfig) -> None:
    FakePlatform.sleep_mode_available = False
    eng = _engine(vllm_config)
    llm = FakeLLM.instances[-1]
    assert llm.kwargs["enable_sleep_mode"] is False
    eng.sleep()
    eng.wake_up()
    assert "sleep" not in llm.names() and "wake_up" not in llm.names()


def test_out_of_process_engine_core_opt_out(fake_vllm, vllm_config: RunConfig) -> None:
    vllm_config.rollout.extra = {"in_process": False}
    _engine(vllm_config)
    assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in os.environ


# -- generate -------------------------------------------------------------------------------------


def test_sampling_param_mapping(fake_vllm, vllm_config: RunConfig) -> None:
    eng = _engine(vllm_config)
    params = SamplingParams(
        n=3,
        max_new_tokens=17,
        temperature=0.7,
        top_p=0.9,
        top_k=-1,
        min_p=0.05,
        repetition_penalty=1.1,
        stop=["</s>", "\n\n"],
        seed=5,
        logprobs=True,
    )
    eng.generate([[1, 2, 3]], params)
    sp = FakeLLM.instances[-1].calls[-1][1]["sampling_params"]
    assert (sp.n, sp.max_tokens, sp.temperature, sp.top_p) == (3, 17, 0.7, 0.9)
    assert sp.top_k == 0  # -1 (monorl "off") -> 0 (vLLM "off")
    assert sp.min_p == 0.05 and sp.repetition_penalty == 1.1
    assert sp.stop == ["</s>", "\n\n"] and sp.seed == 5
    assert sp.logprobs == 0  # sampled-token logprob only
    assert sampling_kwargs(SamplingParams(top_k=40, logprobs=False))["top_k"] == 40
    assert sampling_kwargs(SamplingParams(logprobs=False))["logprobs"] is None


def test_generate_token_and_chat_prompts(fake_vllm, vllm_config: RunConfig) -> None:
    eng = _engine(vllm_config)
    chat = [{"role": "user", "content": "What is 2+2?"}]
    out = eng.generate([[5, 6, 7], chat], SamplingParams(n=2, max_new_tokens=4))
    prompts = FakeLLM.instances[-1].calls[-1][1]["prompts"]
    expected_chat_ids = chat_prompt_ids(eng.tokenizer, chat)
    assert [p["prompt_token_ids"] for p in prompts] == [[5, 6, 7], expected_chat_ids]
    assert type(prompts[0]).__name__ == "TokensPrompt"
    assert len(out) == 2 and all(len(group) == 2 for group in out)
    assert out[0][0].prompt_token_ids == [5, 6, 7]
    assert out[1][1].prompt_token_ids == expected_chat_ids
    assert len(expected_chat_ids) > 3


def test_completion_parsing_logprobs_and_order(fake_vllm, vllm_config: RunConfig) -> None:
    FakeLLM.extra_logprob_entries = 2  # top-k entries present: must pick the sampled token's
    eng = _engine(vllm_config)
    out = eng.generate([[1], [2]], SamplingParams(n=2, logprobs=True))
    # order by CompletionOutput.index even though the fake returns them reversed
    assert [c.text for c in out[1]] == ["text-1-0", "text-1-1"]
    assert out[1][1].token_ids == [1, 1, 7]
    assert out[0][0].logprobs == pytest.approx([-0.1, -0.2, -0.3])
    assert all(isinstance(v, float) for v in out[0][0].logprobs)
    assert out[0][0].finish_reason == "stop"


def test_generate_without_logprobs(fake_vllm, vllm_config: RunConfig) -> None:
    eng = _engine(vllm_config)
    out = eng.generate([[1]], SamplingParams(n=1, logprobs=False))
    assert out[0][0].logprobs is None
    assert FakeLLM.instances[-1].calls[-1][1]["sampling_params"].logprobs is None


def test_finish_reason_mapping(fake_vllm, vllm_config: RunConfig) -> None:
    FakeLLM.finish_reasons = ["stop", "length", "abort", "repetition", None, "error"]
    eng = _engine(vllm_config)
    out = eng.generate([[1]], SamplingParams(n=6))
    assert [c.finish_reason for c in out[0]] == ["stop", "length", "error", "stop", "error", "error"]


def test_generate_empty(fake_vllm, vllm_config: RunConfig) -> None:
    eng = _engine(vllm_config)
    assert eng.generate([], SamplingParams()) == []
    assert "generate" not in FakeLLM.instances[-1].names()


# -- weights --------------------------------------------------------------------------------------


def test_update_weights_reaches_every_worker(fake_vllm, vllm_config: RunConfig) -> None:
    vllm_config.rollout.extra = {"chunk_bytes": 8 * 4}  # two fp32 2x2 tensors do not fit in one chunk
    eng = _engine(vllm_config)
    weights = [
        ("model.a", torch.full((2, 2), 1.0)),
        ("model.b", torch.full((2, 2), 2.0)),
        ("c", torch.tensor(3.0)),
    ]
    eng.update_weights(iter(weights))
    llm = FakeLLM.instances[-1]
    rpcs = [c for c in llm.calls if c[0] == "collective_rpc"]
    assert [c[1]["method"] for c in rpcs] == ["monorl_update_weights"] * 2
    assert [[n for n, _ in c[1]["args"][0]] for c in rpcs] == [["model.a", "model.b"], ["c"]]
    for worker in llm.workers:  # TP=2 -> both workers received every tensor through the extension
        loaded = worker.model_runner.model.loaded
        assert [n for n, _ in loaded] == ["model.a", "model.b", "c"]
        assert torch.equal(loaded[1][1], torch.full((2, 2), 2.0))
    assert llm.names()[-1] == "reset_prefix_cache"


def test_load_weights_from_path_uses_upstream_reload(fake_vllm, vllm_config: RunConfig, tmp_path) -> None:
    eng = _engine(vllm_config)
    eng.load_weights_from_path(tmp_path)
    llm = FakeLLM.instances[-1]
    rpc = [c for c in llm.calls if c[0] == "collective_rpc"][-1][1]
    assert rpc["method"] == "monorl_load_weights_from_path" and rpc["args"] == (str(tmp_path),)
    for worker in llm.workers:
        assert worker.reload_calls == [
            {"args": (), "weights_path": str(tmp_path), "is_checkpoint_format": True}
        ]
    assert llm.names()[-1] == "reset_prefix_cache"


def test_chunk_weights_detaches_and_bounds_size() -> None:
    w = [("x", torch.ones(4, requires_grad=True)), ("y", torch.ones(4)), ("z", torch.ones(1))]
    chunks = list(chunk_weights(w, max_bytes=20))
    assert [[n for n, _ in c] for c in chunks] == [["x"], ["y", "z"]]
    assert not chunks[0][0][1].requires_grad


def test_worker_extension_has_no_name_clash_with_gpu_worker() -> None:
    public = {a for a in dir(MonorlWorkerExtension) if not a.startswith("__")}
    assert public == {"monorl_update_weights", "monorl_load_weights_from_path", "monorl_device_info"}
    assert not public & GPU_WORKER_ATTRS  # WorkerWrapperBase.init_worker asserts this upstream


def test_worker_extension_device_info(fake_vllm, vllm_config: RunConfig) -> None:
    eng = _engine(vllm_config)
    infos = eng.llm.collective_rpc("monorl_device_info")
    assert infos == [{"device": "cpu", "dtype": "float32"}] * 2


# -- sleep / wake ---------------------------------------------------------------------------------


def test_sleep_wake_cycle(fake_vllm, vllm_config: RunConfig) -> None:
    vllm_config.rollout.extra = {"sleep_level": 2}
    eng = _engine(vllm_config)
    llm = FakeLLM.instances[-1]
    eng.sleep()
    eng.sleep()  # idempotent
    assert [c for c in llm.calls if c[0] == "sleep"] == [("sleep", {"level": 2, "mode": "abort"})]
    eng.update_weights([("w", torch.zeros(1))])  # only weights need to be mapped for loading
    wakes = [c[1]["tags"] for c in llm.calls if c[0] == "wake_up"]
    assert wakes == [["weights"]]
    eng.generate([[1]], SamplingParams())  # kv cache still released -> wake just that
    wakes = [c[1]["tags"] for c in llm.calls if c[0] == "wake_up"]
    assert wakes == [["weights"], ["kv_cache"]]
    eng.wake_up()  # everything mapped already: no extra call
    assert len([c for c in llm.calls if c[0] == "wake_up"]) == 2
    eng.sleep()
    eng.wake_up()
    assert [c[1]["tags"] for c in llm.calls if c[0] == "wake_up"][-1] == ["weights", "kv_cache"]


def test_shutdown_releases_engine(fake_vllm, vllm_config: RunConfig) -> None:
    eng = _engine(vllm_config)
    llm = FakeLLM.instances[-1]
    eng.shutdown()
    eng.shutdown()
    assert llm.llm_engine.engine_core.shutdown_calls == 1
    assert eng.llm is None


def test_sampled_logprob_missing_token_is_nan() -> None:
    from monorl.adapters.vllm.common import sampled_logprobs

    out = CompletionOutput(0, "", [1, 2], None, [{1: Logprob(-0.5)}, {9: Logprob(-0.1)}])
    lps = sampled_logprobs(out)
    assert lps[0] == -0.5 and math.isnan(lps[1])
