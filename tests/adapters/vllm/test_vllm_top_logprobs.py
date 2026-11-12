"""``SamplingParams.top_logprobs`` through the vLLM adapter (v0.33.0 shapes): the colocated engine,
``/v1/completions`` and ``/v1/chat/completions``."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from vllm_fakes import CompletionOutput, FakeLLM, Logprob, RequestOutput

from monorl.adapters.vllm.common import (
    chat_completion_kwargs,
    completions_from_chat_response,
    completions_from_request_output,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.registry import registry
from monorl.core.types import SamplingParams, TopLogprobs

# One position's distribution: tokens 11, 12, 13, 14 with these logprobs.
DIST = {11: -0.2, 12: -1.9, 13: -3.0, 14: -6.0}
TOP2 = [(11, -0.2), (12, -1.9)]


def _positions(top: TopLogprobs) -> list[list[tuple[int, float]]]:
    ids, lps = top.token_ids, [round(lp, 5) for lp in top.logprobs]
    return [list(zip(ids[i : i + top.k], lps[i : i + top.k], strict=True)) for i in range(0, len(ids), top.k)]


def _position(sampled: int, k: int) -> dict[int, Logprob]:
    """What vLLM returns for ``logprobs=k``: the sampled token first, then the top-k
    (``append_logprobs_for_next_position``); a sampled token inside the top-k appears once."""
    ranked = sorted(DIST, key=lambda t: -DIST[t])
    position = {sampled: Logprob(DIST[sampled], rank=ranked.index(sampled) + 1)}
    for rank, token in enumerate(ranked[:k], start=1):
        position[token] = Logprob(DIST[token], rank=rank)
    return position


def test_top_k_is_the_same_whichever_token_was_sampled() -> None:
    sampled = [11, 12, 14]  # most likely, inside the top-2, outside the top-2
    output = CompletionOutput(0, "abc", sampled, None, [_position(t, 2) for t in sampled], "stop")
    assert [len(p) for p in output.logprobs] == [2, 2, 3]
    (completion,) = completions_from_request_output(
        RequestOutput("0", None, [1], None, [output], True), [1], want_logprobs=True, top_logprobs=2
    )
    assert _positions(completion.top_logprobs) == [TOP2, TOP2, TOP2]
    assert completion.logprobs == [-0.2, -1.9, -6.0]


def test_colocated_engine_requests_and_returns_top_k(fake_vllm, vllm_config: RunConfig) -> None:
    engine = registry.get("rollout_engine", "vllm")(vllm_config)
    FakeLLM.extra_logprob_entries = 2  # the fake's top-3: the sampled token, then ids 1000 and 1001
    out = engine.generate([[5, 6]], SamplingParams(n=1, max_new_tokens=4, top_logprobs=3))
    assert FakeLLM.instances[-1].calls[-1][1]["sampling_params"].logprobs == 3
    completion = out[0][0]
    assert _positions(completion.top_logprobs) == [
        [(token, lp), (1000, -5.0), (1001, -6.0)]
        for token, lp in zip(completion.token_ids, completion.logprobs, strict=True)
    ]
    plain = engine.generate([[5, 6]], SamplingParams(n=1, max_new_tokens=4))
    assert FakeLLM.instances[-1].calls[-1][1]["sampling_params"].logprobs == 0
    assert plain[0][0].top_logprobs is None


def _score_centering(config: RunConfig, **extra: Any) -> RunConfig:
    data = config.to_dict()
    data["algorithm"]["off_policy"] = {"score_centering": True, "score_centering_top_k": 64}
    data["rollout"]["extra"] = extra
    return RunConfig.from_dict(data)


def test_score_centering_raises_the_engine_logprob_limit(fake_vllm, vllm_config: RunConfig) -> None:
    registry.get("rollout_engine", "vllm")(vllm_config)
    assert "max_logprobs" not in FakeLLM.instances[-1].kwargs
    registry.get("rollout_engine", "vllm")(_score_centering(vllm_config))
    assert FakeLLM.instances[-1].kwargs["max_logprobs"] == 64
    with pytest.raises(ConfigError, match="processed_logprobs"):
        registry.get("rollout_engine", "vllm")(_score_centering(vllm_config, logprobs_mode="raw_logprobs"))


def _completions_server(requests: list[dict[str, Any]], *, placeholders: bool = True):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        sampled = [11, 14]
        k = body["logprobs"]
        name = (lambda t: f"token_id:{t}") if placeholders else (lambda t: f"tok{t}")
        choice = {
            "index": 0,
            "text": "ab",
            "token_ids": sampled,
            "prompt_token_ids": body["prompt"][0],
            "finish_reason": "stop",
            "logprobs": {
                "tokens": [name(t) for t in sampled],
                "token_logprobs": [DIST[t] for t in sampled],
                # /v1/completions returns every entry of the position: k + 1 when the sampled
                # token is outside the top-k
                "top_logprobs": [{name(t): e.logprob for t, e in _position(s, k).items()} for s in sampled],
            },
        }
        return httpx.Response(200, json={"choices": [choice]})

    return handler


def _client(vllm_config: RunConfig, handler):
    vllm_config.rollout.extra = {"mode": "server", "url": "http://vllm.test:8000", "model_name": "policy"}
    return registry.get("rollout_engine", "vllm")(vllm_config, transport=httpx.MockTransport(handler))


def test_server_client_requests_and_parses_top_k(vllm_config: RunConfig) -> None:
    requests: list[dict[str, Any]] = []
    client = _client(vllm_config, _completions_server(requests))
    try:
        out = client.generate([[4, 5]], SamplingParams(n=1, max_new_tokens=2, top_logprobs=2))
    finally:
        client.shutdown()
    assert requests[-1]["logprobs"] == 2 and requests[-1]["return_tokens_as_token_ids"] is True
    completion = out[0][0]
    assert completion.token_ids == [11, 14] and completion.logprobs == [-0.2, -6.0]
    assert _positions(completion.top_logprobs) == [TOP2, TOP2]


def test_server_client_rejects_top_logprobs_keyed_by_token_text(vllm_config: RunConfig) -> None:
    client = _client(vllm_config, _completions_server([], placeholders=False))
    try:
        with pytest.raises(ValueError, match="return_tokens_as_token_ids"):
            client.generate([[4, 5]], SamplingParams(n=1, max_new_tokens=2, top_logprobs=2))
    finally:
        client.shutdown()


def test_chat_endpoint_asks_for_one_extra_candidate() -> None:
    # /v1/chat/completions returns the first `top_logprobs` entries of a position, sampled token
    # first: with k + 1 the top-k is always among them.
    body = chat_completion_kwargs(SamplingParams(top_logprobs=2))
    assert body["logprobs"] is True and body["top_logprobs"] == 3
    assert body["return_tokens_as_token_ids"] is True
    assert chat_completion_kwargs(SamplingParams())["top_logprobs"] == 0

    def content(sampled: int) -> dict[str, Any]:
        entries = list(_position(sampled, 3).items())[:3]
        return {
            "token": f"token_id:{sampled}",
            "logprob": DIST[sampled],
            "top_logprobs": [{"token": f"token_id:{t}", "logprob": e.logprob} for t, e in entries],
        }

    response = {
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "index": 0,
                "token_ids": [11, 14],
                "message": {"content": "ab"},
                "finish_reason": "stop",
                "logprobs": {"content": [content(11), content(14)]},
            }
        ],
    }
    (completion,) = completions_from_chat_response(response, 1, want_logprobs=True, top_logprobs=2)
    assert _positions(completion.top_logprobs) == [TOP2, TOP2]
    assert completion.logprobs == [-0.2, -6.0]
