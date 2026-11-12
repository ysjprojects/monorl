"""Colocated ``SGLangEngine`` against the fake ``sglang.srt.entrypoints.engine.Engine``."""

from __future__ import annotations

import pytest
import torch
from fake_sglang import Engine as FakeEngine
from fake_sglang import MultiprocessingSerializer, make_config

from monorl.adapters.sglang.engine import SGLangEngine, import_sglang
from monorl.core.errors import BackendUnavailable, WeightSyncError
from monorl.core.interfaces import RolloutEngine
from monorl.core.types import SamplingParams


@pytest.fixture
def engine(fake_sglang, sglang_config):
    eng = SGLangEngine(sglang_config)
    yield eng
    eng.shutdown()


def fake_of(engine: SGLangEngine) -> FakeEngine:
    return engine.engine


# -- construction --------------------------------------------------------------------------------
def test_import_sglang_raises_backend_unavailable(no_sglang):
    with pytest.raises(BackendUnavailable, match="monorl\\[sglang\\]"):
        import_sglang()


def test_engine_constructs_with_server_args(fake_sglang, run_config_dict):
    cfg = make_config(
        run_config_dict,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        extra={"schedule_policy": "lpm", "enable_metrics": True, "random_seed": 3},
    )
    eng = SGLangEngine(cfg, replica=1)
    kw = fake_of(eng).kwargs
    assert kw["model_path"] == cfg.model.name_or_path
    assert kw["tp_size"] == 2 and kw["mem_fraction_static"] == 0.5 and kw["context_length"] == 2048
    assert kw["enable_memory_saver"] is True
    assert kw["schedule_policy"] == "lpm" and kw["enable_metrics"] is True and kw["random_seed"] == 3
    assert eng.replica == 1
    assert isinstance(eng, RolloutEngine)
    assert eng.tokenizer.chat_template


def test_engine_version_warning_outside_range(fake_sglang, sglang_config, caplog):
    fake_sglang["sglang"].__version__ = "0.4.0"
    with caplog.at_level("WARNING"):
        SGLangEngine(sglang_config)
    assert "outside the supported range" in caplog.text


# -- generate ------------------------------------------------------------------------------------
def test_generate_chat_prompt_uses_template_ids_and_logprobs(engine):
    messages = [{"role": "user", "content": "What is 2+2?"}]
    params = SamplingParams(n=2, max_new_tokens=3, temperature=0.8, top_p=0.9, stop=["\n"])
    groups = engine.generate([messages], params)
    assert len(groups) == 1 and len(groups[0]) == 2
    tok = engine.tokenizer
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    expected_ids = tok(text, add_special_tokens=False)["input_ids"]
    call = fake_of(engine).calls[-1]
    assert call.method == "generate"
    assert call.kwargs["prompt"] is None
    assert call.kwargs["input_ids"] == [expected_ids]
    assert call.kwargs["return_logprob"] is True
    assert call.kwargs["logprob_start_len"] == -1
    assert call.kwargs["sampling_params"] == [
        {
            "n": 2,
            "max_new_tokens": 3,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "skip_special_tokens": True,
            "stop": ["\n"],
        }
    ]
    for c in groups[0]:
        assert c.prompt_token_ids == expected_ids
        assert len(c.token_ids) == 3 and len(c.logprobs) == 3
        assert c.logprobs == [-0.5, -1.0, -1.5]
        assert c.finish_reason == "stop"
        assert c.text.startswith("text-0-")
    assert groups[0][0].token_ids != groups[0][1].token_ids


def test_generate_token_prompt_is_byte_exact(engine):
    raw = [100, 200, 300, 400]
    groups = engine.generate([raw, [7, 8]], SamplingParams(n=1, logprobs=False))
    call = fake_of(engine).calls[-1]
    assert call.kwargs["input_ids"] == [[100, 200, 300, 400], [7, 8]]
    assert call.kwargs["return_logprob"] is False
    assert [len(g) for g in groups] == [1, 1]
    assert groups[0][0].prompt_token_ids == raw and groups[0][0].prompt_token_ids is not raw
    assert groups[0][0].logprobs is None
    assert groups[1][0].text.startswith("text-1-")


def test_generate_mixed_prompts_preserve_order(engine):
    prompts = [[1, 2, 3], [{"role": "user", "content": "hi"}], [9]]
    groups = engine.generate(prompts, SamplingParams(n=3, logprobs=False))
    assert [len(g) for g in groups] == [3, 3, 3]
    assert [c.text.split("-")[1] for g in groups for c in g] == ["0"] * 3 + ["1"] * 3 + ["2"] * 3
    assert groups[0][0].prompt_token_ids == [1, 2, 3] and groups[2][0].prompt_token_ids == [9]


def test_generate_with_seed_fans_out_requests(engine):
    groups = engine.generate([[1, 2], [3]], SamplingParams(n=2, seed=42, logprobs=False))
    call = fake_of(engine).calls[-1]
    assert call.kwargs["input_ids"] == [[1, 2], [1, 2], [3], [3]]
    assert [sp["sampling_seed"] for sp in call.kwargs["sampling_params"]] == [42, 43, 42, 43]
    assert all(sp["n"] == 1 for sp in call.kwargs["sampling_params"])
    assert [len(g) for g in groups] == [2, 2]
    assert groups[0][0].token_ids != groups[0][1].token_ids


@pytest.mark.parametrize(
    "finish, expected", [("stop", "stop"), ("length", "length"), ("abort", "error"), (None, "error")]
)
def test_generate_finish_reasons(engine, finish, expected):
    fake_of(engine).script.finish = finish
    groups = engine.generate([[1]], SamplingParams(n=1))
    assert groups[0][0].finish_reason == expected


def test_generate_logprob_values_are_python_floats_even_from_tensors(engine):
    fake_of(engine).script.logprob_type = lambda v: torch.tensor(v)
    c = engine.generate([[1]], SamplingParams(n=1))[0][0]
    assert all(type(v) is float for v in c.logprobs)


def test_generate_falls_back_to_logprob_token_ids(engine):
    fake_of(engine).script.include_output_ids = False
    c = engine.generate([[1]], SamplingParams(n=1))[0][0]
    assert len(c.token_ids) == 3 and c.token_ids == [t for t in c.token_ids]


def test_generate_empty_and_single_dict_output(engine):
    assert engine.generate([], SamplingParams()) == []
    # the fake returns a bare dict for a single n=1 request given as list[int]; engine wraps it
    groups = engine.generate([[5]], SamplingParams(n=1))
    assert len(groups) == 1 and len(groups[0]) == 1


def test_generate_refuses_while_asleep(engine):
    engine.sleep()
    with pytest.raises(RuntimeError, match="asleep"):
        engine.generate([[1]], SamplingParams())


# -- weights -------------------------------------------------------------------------------------
def test_update_weights_serialises_named_tensors_per_rank_and_flushes(fake_sglang, run_config_dict):
    cfg = make_config(run_config_dict, tensor_parallel_size=2)
    eng = SGLangEngine(cfg)
    fake = fake_of(eng)
    w1 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    w2 = torch.ones(4, requires_grad=True)
    eng.update_weights([("model.a", w1), ("model.b", w2)])
    methods = fake.methods()
    assert methods == ["update_weights_from_tensor", "flush_cache"]
    call = fake.calls[0]
    assert call.kwargs["load_format"] is None and call.kwargs["flush_cache"] is False
    assert [n for n, _ in call.kwargs["named_tensors"]] == ["model.a", "model.b"]
    assert not call.kwargs["named_tensors"][1][1].requires_grad
    # one MultiprocessingSerializer payload per TP rank, deserialising to the same tensors
    payloads = call.kwargs["serialized_named_tensors"]
    assert len(payloads) == 2
    for payload in payloads:
        back = MultiprocessingSerializer.deserialize(payload)
        assert [n for n, _ in back] == ["model.a", "model.b"]
        assert torch.equal(back[0][1], w1) and torch.equal(back[1][1], w2.detach())


def test_update_weights_buckets_and_flushes_once(fake_sglang, run_config_dict):
    cfg = make_config(run_config_dict, extra={"bucket_bytes": 16})
    eng = SGLangEngine(cfg)
    fake = fake_of(eng)
    weights = [(f"w{i}", torch.zeros(4, dtype=torch.float32)) for i in range(5)]  # 16 bytes each
    eng.update_weights(iter(weights))
    assert fake.methods() == ["update_weights_from_tensor"] * 5 + ["flush_cache"]
    assert all(c.kwargs["flush_cache"] is False for c in fake.calls[:5])
    assert [c.kwargs["named_tensors"][0][0] for c in fake.calls[:5]] == ["w0", "w1", "w2", "w3", "w4"]


def test_update_weights_empty_does_not_flush(engine):
    engine.update_weights([])
    assert fake_of(engine).methods() == []


def test_update_weights_failure_raises(engine):
    fake_of(engine).script.update_result = (False, "shape mismatch")
    with pytest.raises(WeightSyncError, match="shape mismatch"):
        engine.update_weights([("w", torch.zeros(1))])


def test_flush_cache_failure_only_warns(engine, caplog):
    fake_of(engine).script.flush_success = False
    with caplog.at_level("WARNING"):
        engine.update_weights([("w", torch.zeros(1))])
    assert "flush_cache did not run" in caplog.text


def test_load_weights_from_path(engine, tmp_path):
    engine.load_weights_from_path(tmp_path)
    fake = fake_of(engine)
    assert fake.methods() == ["update_weights_from_disk", "flush_cache"]
    assert fake.calls[0].kwargs == {"model_path": str(tmp_path), "load_format": None}
    fake.script.disk_result = (False, "no such dir", 0)
    with pytest.raises(WeightSyncError, match="no such dir"):
        engine.load_weights_from_path(tmp_path)


# -- sleep / wake / shutdown ---------------------------------------------------------------------
@pytest.fixture
def cuda_available(monkeypatch):
    """The applied ``sglang.*_memory_guard`` patches skip the memory saver without CUDA."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_sleep_and_wake_call_memory_occupation(engine, cuda_available):
    engine.sleep()
    engine.sleep()  # idempotent
    engine.wake_up()
    engine.wake_up()
    fake = fake_of(engine)
    assert fake.methods() == ["release_memory_occupation", "resume_memory_occupation"]
    assert fake.calls[0].kwargs == {"tags": None} and fake.calls[1].kwargs == {"tags": None}


def test_sleep_tags_passthrough(fake_sglang, run_config_dict, cuda_available):
    eng = SGLangEngine(make_config(run_config_dict, extra={"sleep_tags": ["kv_cache"]}))
    eng.sleep()
    eng.wake_up()
    assert [c.kwargs["tags"] for c in fake_of(eng).calls] == [["kv_cache"], ["kv_cache"]]


def test_sleep_disabled_is_noop(fake_sglang, run_config_dict):
    eng = SGLangEngine(make_config(run_config_dict, enable_sleep_mode=False))
    assert fake_of(eng).kwargs["enable_memory_saver"] is False
    eng.sleep()
    eng.wake_up()
    assert fake_of(eng).methods() == []
    eng.generate([[1]], SamplingParams(n=1))  # not asleep


def test_sleep_guard_skips_memory_saver_without_cuda(engine, monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level("WARNING"):
        engine.sleep()
        engine.wake_up()
    assert fake_of(engine).methods() == []
    assert "release_memory_occupation skipped: CUDA is not available" in caplog.text


def test_patches_disabled_uses_raw_engine(fake_sglang, sglang_config, monkeypatch):
    monkeypatch.setenv("MONORL_DISABLE_PATCHES", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    eng = SGLangEngine(sglang_config)
    assert "__wrapped__" not in vars(FakeEngine.generate)
    eng.sleep()
    assert fake_of(eng).methods() == ["release_memory_occupation"]


def test_engine_applies_patchset(engine):
    from monorl.patching import is_applied

    assert is_applied("sglang.seeded_parallel_sampling", "sglang")
    assert is_applied("sglang.release_memory_guard", "sglang")


def test_shutdown_once(engine):
    engine.shutdown()
    engine.shutdown()
    assert fake_of(engine).methods() == ["shutdown"]
    assert fake_of(engine).shut_down
