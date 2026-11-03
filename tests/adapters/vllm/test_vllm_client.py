from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest
import torch

from monorl.adapters.vllm.client import VllmServerClient
from monorl.adapters.vllm.common import tensor_from_bytes
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.interfaces import RolloutEngine
from monorl.core.registry import registry
from monorl.core.types import SamplingParams
from monorl.rollout.hf_engine import chat_prompt_ids


class FakeServer:
    """Answers like a monorl vLLM server (vLLM v0.33.0 ``CompletionResponse`` shape)."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.fail_paths: set[str] = set()
        self.finish_reasons = ["stop"]
        self.shuffle = True

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.fail_paths:
            return httpx.Response(500, text="boom")
        if path == "/v1/completions":
            body = json.loads(request.content)
            n = body["n"]
            choices = []
            for p, ids in enumerate(body["prompt"]):
                for i in range(n):
                    toks = [p, i, 7]
                    choice: dict[str, Any] = {
                        "index": p * n + i,
                        "text": f"text-{p}-{i}",
                        "token_ids": toks,
                        "prompt_token_ids": list(ids),
                        "finish_reason": self.finish_reasons[(p * n + i) % len(self.finish_reasons)],
                        "logprobs": None,
                    }
                    if body.get("logprobs") is not None:
                        choice["logprobs"] = {
                            "text_offset": [0, 1, 2],
                            "token_logprobs": [-(k + 1) / 10 for k in range(3)],
                            "tokens": [f"t{t}" for t in toks],
                            "top_logprobs": [None] * 3,
                        }
                    choices.append(choice)
            if self.shuffle:
                choices.reverse()
            return httpx.Response(
                200,
                json={"id": "cmpl-1", "object": "text_completion", "model": body["model"], "choices": choices,
                      "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            )
        if path == "/update_named_param":
            q = request.url.params
            t = tensor_from_bytes(request.content, q["dtype"], [int(s) for s in q["shape"].split(",") if s])
            return httpx.Response(200, json={"name": q["name"], "loaded": [1], "sum": float(t.float().sum())})
        return httpx.Response(200, json={"status": "ok", "path": path})

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def client(server: FakeServer, vllm_config: RunConfig) -> VllmServerClient:
    vllm_config.rollout.extra = {"mode": "server", "url": "http://vllm.test:8000/", "model_name": "policy"}
    eng = registry.get("rollout_engine", "vllm")(vllm_config, transport=httpx.MockTransport(server.handler))
    assert isinstance(eng, VllmServerClient) and isinstance(eng, RolloutEngine)
    return eng


def test_server_mode_needs_url(vllm_config: RunConfig) -> None:
    vllm_config.rollout.extra = {"mode": "server"}
    with pytest.raises(ConfigError, match="rollout.extra.url"):
        registry.get("rollout_engine", "vllm")(vllm_config)


def test_urls_per_replica(vllm_config: RunConfig, server: FakeServer, tiny_model_dir) -> None:
    vllm_config.rollout.extra = {"mode": "server", "urls": ["http://a:1", "http://b:2/"]}
    transport = httpx.MockTransport(server.handler)
    assert VllmServerClient(vllm_config, replica=1, transport=transport).url == "http://b:2"
    with pytest.raises(ConfigError, match="replica 2"):
        VllmServerClient(vllm_config, replica=2, transport=transport)


def test_completions_request_body_and_parsing(client: VllmServerClient, server: FakeServer) -> None:
    chat = [{"role": "user", "content": "hi"}]
    params = SamplingParams(n=2, max_new_tokens=9, temperature=0.5, top_k=-1, stop=["x"], seed=3, logprobs=True)
    out = client.generate([[4, 5], chat], params)
    req = server.requests[-1]
    assert req.url.path == "/v1/completions" and req.headers["content-type"].startswith("application/json")
    body = json.loads(req.content)
    chat_ids = chat_prompt_ids(client.tokenizer, chat)
    assert body["prompt"] == [[4, 5], chat_ids]
    assert body["model"] == "policy" and body["return_token_ids"] is True
    assert body["n"] == 2 and body["max_tokens"] == 9 and body["temperature"] == 0.5
    assert body["top_k"] == 0 and body["stop"] == ["x"] and body["seed"] == 3 and body["logprobs"] == 0
    assert len(out) == 2 and [len(g) for g in out] == [2, 2]
    assert [c.text for c in out[1]] == ["text-1-0", "text-1-1"]  # regrouped from shuffled choices
    assert out[1][1].token_ids == [1, 1, 7]
    assert out[0][0].logprobs == pytest.approx([-0.1, -0.2, -0.3])
    assert out[0][0].prompt_token_ids == [4, 5] and out[1][0].prompt_token_ids == chat_ids


def test_completions_without_logprobs_and_finish_reasons(client: VllmServerClient, server: FakeServer) -> None:
    server.finish_reasons = ["length", "abort"]
    out = client.generate([[1]], SamplingParams(n=2, logprobs=False))
    assert json.loads(server.requests[-1].content)["logprobs"] is None
    assert [c.logprobs for c in out[0]] == [None, None]
    assert [c.finish_reason for c in out[0]] == ["length", "error"]


def test_choice_count_mismatch_is_an_error(client: VllmServerClient, server: FakeServer) -> None:
    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client._client = httpx.Client(base_url=client.url, transport=httpx.MockTransport(bad))
    with pytest.raises(ValueError, match="expected 2"):
        client.generate([[1]], SamplingParams(n=2))


def test_null_logprob_becomes_nan(client: VllmServerClient) -> None:
    def resp(request: httpx.Request) -> httpx.Response:
        choice = {"index": 0, "text": "", "token_ids": [1, 2], "finish_reason": "stop",
                  "logprobs": {"token_logprobs": [None, -0.5], "tokens": ["a", "b"]}}
        return httpx.Response(200, json={"choices": [choice]})

    client._client = httpx.Client(base_url=client.url, transport=httpx.MockTransport(resp))
    lps = client.generate([[1]], SamplingParams(n=1))[0][0].logprobs
    assert math.isnan(lps[0]) and lps[1] == -0.5


def test_update_weights_streams_each_tensor(client: VllmServerClient, server: FakeServer) -> None:
    w = [("a.weight", torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)), ("b", torch.tensor(2.5))]
    client.update_weights(iter(w))
    reqs = [r for r in server.requests if r.url.path == "/update_named_param"]
    assert [r.url.params["name"] for r in reqs] == ["a.weight", "b"]
    assert reqs[0].url.params["dtype"] == "bfloat16" and reqs[0].url.params["shape"] == "2,3"
    assert reqs[0].headers["content-type"] == "application/octet-stream"
    assert len(reqs[0].content) == 12  # 6 x bf16
    assert reqs[1].url.params["shape"] == "" and len(reqs[1].content) == 4
    assert server.paths()[-1] == "/reset_prefix_cache"
    restored = tensor_from_bytes(reqs[0].content, "bfloat16", [2, 3])
    assert torch.equal(restored, w[0][1])


def test_load_weights_from_path(client: VllmServerClient, server: FakeServer, tmp_path) -> None:
    client.load_weights_from_path(tmp_path)
    req = server.requests[-2]
    assert req.url.path == "/load_weights_from_path" and json.loads(req.content) == {"path": str(tmp_path)}
    assert server.paths()[-1] == "/reset_prefix_cache"


def test_sleep_wake_requests(client: VllmServerClient, server: FakeServer) -> None:
    client.sleep()
    client.sleep()
    assert server.paths() == ["/sleep"] and server.requests[0].url.params["level"] == "1"
    client.update_weights([("w", torch.zeros(1))])  # wakes weights only
    wake = [r for r in server.requests if r.url.path == "/wake_up"]
    assert [r.url.params.get_list("tags") for r in wake] == [["weights"]]
    client.generate([[1]], SamplingParams())  # then the kv cache
    wake = [r for r in server.requests if r.url.path == "/wake_up"]
    assert [r.url.params.get_list("tags") for r in wake] == [["weights"], ["kv_cache"]]
    client.wake_up()
    assert len([r for r in server.requests if r.url.path == "/wake_up"]) == 2
    client.sleep()
    client.wake_up()
    assert server.requests[-1].url.params.get_list("tags") == ["weights", "kv_cache"]


def test_sleep_disabled_in_config(vllm_config: RunConfig, server: FakeServer) -> None:
    vllm_config.rollout.enable_sleep_mode = False
    vllm_config.rollout.extra = {"mode": "server", "url": "http://x"}
    c = VllmServerClient(vllm_config, transport=httpx.MockTransport(server.handler))
    c.sleep()
    c.wake_up()
    assert server.paths() == []


def test_http_errors_surface(client: VllmServerClient, server: FakeServer) -> None:
    server.fail_paths = {"/v1/completions"}
    with pytest.raises(RuntimeError, match="/v1/completions -> 500: boom"):
        client.generate([[1]], SamplingParams())


def test_shutdown_closes_client(client: VllmServerClient) -> None:
    client.shutdown()
    assert client._client.is_closed
