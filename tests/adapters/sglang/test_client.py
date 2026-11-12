"""``SGLangServerClient`` request bodies against sglang's HTTP endpoints via ``httpx.MockTransport``."""

from __future__ import annotations

import base64
import json
import pickle
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
import torch
from fake_sglang import make_config

from monorl.adapters.sglang.client import SGLangServerClient
from monorl.core.errors import ConfigError, WeightSyncError
from monorl.core.interfaces import RolloutEngine
from monorl.core.types import SamplingParams


@dataclass
class FakeServer:
    """Minimal stand-in for ``sglang/srt/entrypoints/http_server.py`` routes used by the client."""

    tp_size: int = 2
    requests: list[tuple[str, str, Any]] = field(default_factory=list)
    update_ok: bool = True
    disk_ok: bool = True
    flush_ok: bool = True
    generate_status: int = 200
    finish: dict[str, Any] = field(default_factory=lambda: {"type": "stop", "matched": 1})

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.requests.append((request.method, path, body))
        if path == "/get_server_info":
            return httpx.Response(200, json={"tp_size": self.tp_size, "dp_size": 1, "model_path": "m"})
        if path == "/generate":
            if self.generate_status != 200:
                return httpx.Response(self.generate_status, json={"error": "bad request"})
            outs = []
            for i, (ids, sp) in enumerate(zip(body["input_ids"], body["sampling_params"], strict=True)):
                for k in range(sp.get("n", 1)):
                    toks = [500 + i * 10 + k, 501]
                    meta: dict[str, Any] = {
                        "id": f"r{i}{k}",
                        "finish_reason": self.finish,
                        "prompt_tokens": len(ids),
                    }
                    if body.get("return_logprob"):
                        meta["output_token_logprobs"] = [[-0.1, toks[0], None], [-0.2, toks[1], None]]
                    outs.append({"text": f"t{i}{k}", "output_ids": toks, "meta_info": meta})
            return httpx.Response(200, json=outs)
        if path == "/update_weights_from_tensor":
            ok = self.update_ok
            return httpx.Response(
                200 if ok else 400, json={"success": ok, "message": "Success." if ok else "bad tensor"}
            )
        if path == "/update_weights_from_disk":
            ok = self.disk_ok
            return httpx.Response(
                200 if ok else 400,
                json={"success": ok, "message": "ok" if ok else "missing", "num_paused_requests": 0},
            )
        if path == "/flush_cache":
            return httpx.Response(
                200 if self.flush_ok else 400, text="Cache flushed." if self.flush_ok else "busy"
            )
        if path in ("/release_memory_occupation", "/resume_memory_occupation"):
            return httpx.Response(200, json=None)
        return httpx.Response(404, text=f"no route {path}")

    def paths(self) -> list[str]:
        return [p for _, p, _ in self.requests]


@pytest.fixture
def server():
    return FakeServer()


@pytest.fixture
def client(server, run_config_dict):
    cfg = make_config(run_config_dict, extra={"mode": "server", "url": "http://sgl.test:30000/"})
    c = SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))
    yield c
    c.shutdown()


def test_client_is_rollout_engine_and_needs_url(client, run_config_dict):
    assert isinstance(client, RolloutEngine)
    assert client.url == "http://sgl.test:30000"
    with pytest.raises(ConfigError, match="rollout.extra.url"):
        SGLangServerClient(make_config(run_config_dict, extra={"mode": "server"}))


def test_generate_request_body_chat_and_tokens(client, server):
    messages = [{"role": "user", "content": "hi"}]
    params = SamplingParams(n=2, max_new_tokens=5, temperature=0.5, top_k=10, stop=["x"])
    groups = client.generate([messages, [1, 2, 3]], params)
    method, path, body = server.requests[-1]
    assert (method, path) == ("POST", "/generate")
    tok = client.tokenizer
    chat_ids = tok(
        tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
        add_special_tokens=False,
    )["input_ids"]
    assert body["input_ids"] == [chat_ids, [1, 2, 3]]
    assert body["return_logprob"] is True and body["logprob_start_len"] == -1
    assert (
        body["sampling_params"]
        == [
            {
                "n": 2,
                "max_new_tokens": 5,
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 10,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "skip_special_tokens": True,
                "stop": ["x"],
            }
        ]
        * 2
    )
    assert [len(g) for g in groups] == [2, 2]
    assert groups[0][0].prompt_token_ids == chat_ids and groups[1][1].prompt_token_ids == [1, 2, 3]
    assert groups[1][1].text == "t11" and groups[1][1].token_ids == [511, 501]
    assert groups[0][0].logprobs == [-0.1, -0.2] and groups[0][0].finish_reason == "stop"


def test_generate_seed_fanout_and_no_logprobs(client, server):
    groups = client.generate([[1]], SamplingParams(n=2, seed=5, logprobs=False))
    body = server.requests[-1][2]
    assert body["input_ids"] == [[1], [1]]
    assert body["return_logprob"] is False
    assert [sp["sampling_seed"] for sp in body["sampling_params"]] == [5, 6]
    assert len(groups[0]) == 2 and groups[0][0].logprobs is None


def test_generate_length_and_error_status(client, server):
    server.finish = {"type": "length", "length": 2}
    assert client.generate([[1]], SamplingParams())[0][0].finish_reason == "length"
    server.generate_status = 400
    with pytest.raises(RuntimeError, match="HTTP 400"):
        client.generate([[1]], SamplingParams())
    assert client.generate([], SamplingParams()) == []


def test_update_weights_posts_base64_payload_per_rank_then_flushes(client, server):
    w = torch.arange(4, dtype=torch.float32)
    client.update_weights([("layer.w", w)])
    assert server.paths() == ["/get_server_info", "/update_weights_from_tensor", "/flush_cache"]
    _, _, body = server.requests[1]
    assert body["load_format"] is None and body["flush_cache"] is False
    payloads = body["serialized_named_tensors"]
    assert len(payloads) == server.tp_size
    for p in payloads:
        raw = base64.b64decode(p, validate=True)
        back = pickle.loads(raw)  # what MultiprocessingSerializer.deserialize does after base64
        assert back[0][0] == "layer.w" and torch.equal(back[0][1], w)
    # tp_size is cached: a second update does not re-query server info
    client.update_weights([("layer.b", torch.zeros(1))])
    assert server.paths()[3:] == ["/update_weights_from_tensor", "/flush_cache"]


def test_update_weights_buckets(server, run_config_dict):
    cfg = make_config(run_config_dict, extra={"mode": "server", "url": "http://s", "bucket_bytes": 8})
    c = SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))
    c.update_weights([(f"w{i}", torch.zeros(2, dtype=torch.float32)) for i in range(3)])
    assert server.paths() == ["/get_server_info"] + ["/update_weights_from_tensor"] * 3 + ["/flush_cache"]


def test_update_weights_failure_and_flush_warning(client, server, caplog):
    server.update_ok = False
    with pytest.raises(WeightSyncError, match="bad tensor"):
        client.update_weights([("w", torch.zeros(1))])
    server.update_ok = True
    server.flush_ok = False
    with caplog.at_level("WARNING"):
        client.update_weights([("w", torch.zeros(1))])
    assert "/flush_cache did not run" in caplog.text


def test_load_weights_from_path(client, server, tmp_path):
    client.load_weights_from_path(tmp_path)
    assert server.paths() == ["/update_weights_from_disk", "/flush_cache"]
    assert server.requests[0][2] == {"model_path": str(tmp_path), "load_format": None}
    server.disk_ok = False
    with pytest.raises(WeightSyncError, match="missing"):
        client.load_weights_from_path(tmp_path)


def test_sleep_wake_endpoints(client, server):
    client.sleep()
    client.sleep()
    client.wake_up()
    client.wake_up()
    assert server.requests == [
        ("POST", "/release_memory_occupation", {}),
        ("POST", "/resume_memory_occupation", {}),
    ]


def test_sleep_tags_and_disabled(server, run_config_dict):
    cfg = make_config(run_config_dict, extra={"mode": "server", "url": "http://s", "sleep_tags": ["weights"]})
    c = SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))
    c.sleep()
    assert server.requests[-1] == ("POST", "/release_memory_occupation", {"tags": ["weights"]})
    off = SGLangServerClient(
        make_config(run_config_dict, enable_sleep_mode=False, extra={"mode": "server", "url": "http://s"}),
        transport=httpx.MockTransport(server.handler),
    )
    n = len(server.requests)
    off.sleep()
    off.wake_up()
    assert len(server.requests) == n


def test_server_info_and_tp_fallback(server, run_config_dict):
    cfg = make_config(run_config_dict, tensor_parallel_size=3, extra={"mode": "server", "url": "http://s"})
    c = SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))
    assert c.server_info()["tp_size"] == 2

    def no_tp(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_server_info":
            return httpx.Response(200, json={"model_path": "m"})
        return server.handler(request)

    c2 = SGLangServerClient(cfg, transport=httpx.MockTransport(no_tp))
    c2.update_weights([("w", torch.zeros(1))])
    assert len(server.requests[-2][2]["serialized_named_tensors"]) == 3
