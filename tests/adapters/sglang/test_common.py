"""Pure helper tests: sampling mapping, output parsing, bucketing, serialisation, config mapping."""

from __future__ import annotations

import math
import pickle

import pytest
import torch
from fake_sglang import SamplingParams as SglangSamplingParams
from fake_sglang import make_config

from monorl.adapters.sglang.common import (
    DEFAULT_BUCKET_BYTES,
    SERVER_ARG_PASSTHROUGH,
    bucket_bytes,
    bucket_weights,
    chat_prompt_ids,
    completion_from_output,
    engine_kwargs,
    expand_requests,
    group_outputs,
    map_finish_reason,
    output_logprobs,
    prompt_token_ids,
    sampling_dict,
    serialize_for_sglang,
    serialized_payloads,
    server_url,
)
from monorl.core.errors import ConfigError
from monorl.core.types import SamplingParams
from monorl.rollout.hf_engine import load_tokenizer


# -- sampling ------------------------------------------------------------------------------------
def test_sampling_dict_maps_every_field_to_pinned_names():
    params = SamplingParams(
        n=3,
        max_new_tokens=17,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        min_p=0.05,
        repetition_penalty=1.1,
        stop=["</s>", "\n\n"],
        seed=None,
    )
    d = sampling_dict(params)
    assert d == {
        "n": 3,
        "max_new_tokens": 17,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "skip_special_tokens": True,
        "stop": ["</s>", "\n\n"],
        "no_stop_trim": True,  # the text must keep e.g. a closing </tool_call>
    }
    # every key is a real sglang SamplingParams keyword (sampling_params.py @ v0.5.23)
    SglangSamplingParams(**d)


def test_sampling_dict_top_k_disabled_and_no_stop():
    d = sampling_dict(SamplingParams(top_k=-1, stop=[]))
    assert d["top_k"] == -1
    assert "stop" not in d and "no_stop_trim" not in d
    assert "sampling_seed" not in d
    assert sampling_dict(SamplingParams(top_k=0))["top_k"] == -1


def test_sampling_dict_seed_overrides():
    d = sampling_dict(SamplingParams(n=4, seed=7), n=1, seed=9)
    assert d["n"] == 1 and d["sampling_seed"] == 9
    SglangSamplingParams(**d)


def test_expand_requests_without_seed_keeps_one_request_per_prompt():
    ids, sps = expand_requests([[1, 2], [3]], SamplingParams(n=4))
    assert ids == [[1, 2], [3]]
    assert [sp["n"] for sp in sps] == [4, 4]
    assert all("sampling_seed" not in sp for sp in sps)


def test_expand_requests_with_seed_fans_out_distinct_seeds():
    ids, sps = expand_requests([[1, 2], [3]], SamplingParams(n=3, seed=100))
    assert ids == [[1, 2], [1, 2], [1, 2], [3], [3], [3]]
    assert [sp["n"] for sp in sps] == [1] * 6
    assert [sp["sampling_seed"] for sp in sps] == [100, 101, 102, 100, 101, 102]


# -- outputs -------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "finish, expected",
    [
        ({"type": "stop", "matched": 2}, "stop"),
        ({"type": "length", "length": 8}, "length"),
        ({"type": "abort", "message": "x"}, "error"),
        ("stop", "stop"),
        ("length", "length"),
        (None, "error"),
        ({"type": "weird"}, "error"),
    ],
)
def test_map_finish_reason(finish, expected):
    assert map_finish_reason(finish) == expected


def test_output_logprobs_extracts_floats_from_tuples():
    meta = {"output_token_logprobs": [(-0.25, 5, None), [-1.5, 6, "x"], (None, 7, None)]}
    lps = output_logprobs(meta)
    assert lps[:2] == [-0.25, -1.5] and math.isnan(lps[2])
    assert all(type(v) is float for v in lps)
    assert output_logprobs({}) is None


def test_completion_from_output_full():
    out = {
        "text": "hello",
        "output_ids": [11, 12, 13],
        "meta_info": {
            "id": "r",
            "finish_reason": {"type": "length", "length": 3},
            "output_token_logprobs": [(-0.1, 11, None), (-0.2, 12, None), (-0.3, 13, None)],
        },
    }
    c = completion_from_output(out, prompt_ids=[1, 2], want_logprobs=True)
    assert c.text == "hello"
    assert c.token_ids == [11, 12, 13]
    assert c.logprobs == [-0.1, -0.2, -0.3]
    assert c.finish_reason == "length"
    assert c.prompt_token_ids == [1, 2]
    c2 = completion_from_output(out, prompt_ids=[1, 2], want_logprobs=False)
    assert c2.logprobs is None


def test_completion_from_output_falls_back_to_logprob_token_ids():
    out = {
        "text": "",
        "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [(-1.0, 9, None)]},
    }
    c = completion_from_output(out, prompt_ids=[], want_logprobs=True)
    assert c.token_ids == [9] and c.logprobs == [-1.0]
    with pytest.raises(ValueError, match="neither"):
        completion_from_output({"text": "", "meta_info": {}}, prompt_ids=[], want_logprobs=False)


def test_completion_from_output_rejects_logprob_length_mismatch():
    out = {
        "output_ids": [1, 2],
        "meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [(-1.0, 1, None)]},
    }
    with pytest.raises(ValueError, match="1 output logprobs for 2"):
        completion_from_output(out, prompt_ids=[], want_logprobs=True)


def test_group_outputs_prompt_major():
    outs = [
        {"text": f"{i}", "output_ids": [i], "meta_info": {"finish_reason": {"type": "stop"}}}
        for i in range(6)
    ]
    grouped = group_outputs(outs, [[1], [2], [3]], SamplingParams(n=2, logprobs=False))
    assert [[c.text for c in g] for g in grouped] == [["0", "1"], ["2", "3"], ["4", "5"]]
    assert all(c.prompt_token_ids == [i + 1] for i, g in enumerate(grouped) for c in g)
    with pytest.raises(ValueError, match="expected 6"):
        group_outputs(outs[:5], [[1], [2], [3]], SamplingParams(n=2, logprobs=False))


# -- prompts -------------------------------------------------------------------------------------
def test_prompt_token_ids_chat_and_raw(sglang_config):
    tok = load_tokenizer(sglang_config)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    ids = prompt_token_ids(tok, messages)
    assert ids == chat_prompt_ids(tok, messages)
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    assert ids == tok(text, add_special_tokens=False)["input_ids"]
    assert all(isinstance(t, int) for t in ids)
    assert prompt_token_ids(tok, [5, 6, 7]) == [5, 6, 7]
    assert prompt_token_ids(tok, [5, 6, 7]) is not ids


# -- weights -------------------------------------------------------------------------------------
def test_bucket_weights_bounds_bytes_and_makes_contiguous():
    big = torch.zeros(1024, dtype=torch.float32)  # 4096 bytes
    view = torch.arange(16, dtype=torch.float32).reshape(4, 4).t()  # non-contiguous
    weights = [("a", big), ("b", big.clone()), ("c", view), ("d", torch.ones(2, requires_grad=True))]
    buckets = list(bucket_weights(weights, max_bytes=4096))
    assert [[n for n, _ in b] for b in buckets] == [["a"], ["b"], ["c", "d"]]
    c = dict(buckets[2])["c"]
    assert c.is_contiguous() and torch.equal(c, view)
    assert not dict(buckets[2])["d"].requires_grad
    assert list(bucket_weights([])) == []
    assert DEFAULT_BUCKET_BYTES == 512 * 1024 * 1024


def test_serialize_for_sglang_roundtrips_like_multiprocessing_serializer():
    named = [("w", torch.arange(6, dtype=torch.float32).reshape(2, 3))]
    raw = serialize_for_sglang(named)
    assert isinstance(raw, bytes) and raw[:1] == b"\x80"  # pickle protocol marker
    back = pickle.loads(raw)
    assert back[0][0] == "w" and torch.equal(back[0][1], named[0][1])
    b64 = serialize_for_sglang(named, output_str=True)
    assert isinstance(b64, str)
    payloads = serialized_payloads(named, tp_size=2)
    assert len(payloads) == 2 and all(isinstance(p, str) for p in payloads)
    with pytest.raises(ValueError):
        serialized_payloads(named, tp_size=0)


# -- config --------------------------------------------------------------------------------------
def test_engine_kwargs_core_mapping(run_config_dict):
    cfg = make_config(
        run_config_dict,
        tensor_parallel_size=2,
        data_parallel_size=3,
        gpu_memory_utilization=0.45,
        max_model_len=4096,
        enable_sleep_mode=False,
        extra={
            "schedule_policy": "fcfs",
            "enable_metrics": True,
            "mode": "colocate",
            "bucket_bytes": 10,
            "unknown_key": 1,
        },
    )
    kw = engine_kwargs(cfg)
    assert kw["model_path"] == run_config_dict["model"]["name_or_path"]
    assert kw["tp_size"] == 2 and kw["dp_size"] == 1
    assert kw["mem_fraction_static"] == 0.45
    assert kw["context_length"] == 4096
    assert kw["dtype"] == "float32"
    assert kw["enable_memory_saver"] is False
    assert kw["log_level"] == "warning"
    assert kw["schedule_policy"] == "fcfs" and kw["enable_metrics"] is True
    for k in ("mode", "bucket_bytes", "unknown_key"):
        assert k not in kw
    assert "schedule_policy" in SERVER_ARG_PASSTHROUGH and "enable_metrics" in SERVER_ARG_PASSTHROUGH


def test_engine_kwargs_engine_kwargs_override(run_config_dict):
    cfg = make_config(run_config_dict, extra={"engine_kwargs": {"tp_size": 8, "disable_cuda_graph": True}})
    kw = engine_kwargs(cfg)
    assert kw["tp_size"] == 8 and kw["disable_cuda_graph"] is True
    with pytest.raises(ConfigError, match="engine_kwargs"):
        engine_kwargs(make_config(run_config_dict, extra={"engine_kwargs": [1]}))


def test_bucket_bytes_and_server_url(run_config_dict):
    assert bucket_bytes(make_config(run_config_dict)) == DEFAULT_BUCKET_BYTES
    assert bucket_bytes(make_config(run_config_dict, extra={"bucket_bytes": 123})) == 123
    with pytest.raises(ConfigError):
        bucket_bytes(make_config(run_config_dict, extra={"bucket_bytes": 0}))
    assert server_url(make_config(run_config_dict, extra={"url": "http://h:30000/"}), 0) == "http://h:30000"
    assert server_url(make_config(run_config_dict, extra={"urls": ["http://a", "http://b"]}), 1) == "http://b"
    with pytest.raises(ConfigError, match="urls has 2"):
        server_url(make_config(run_config_dict, extra={"urls": ["http://a", "http://b"]}), 2)
    with pytest.raises(ConfigError, match="rollout.extra.url"):
        server_url(make_config(run_config_dict), 0)
