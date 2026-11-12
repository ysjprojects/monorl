"""``SamplingParams.top_logprobs`` through the SGLang adapter (v0.5.23 shapes)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fake_sglang import make_config

from monorl.adapters.sglang.client import SGLangServerClient
from monorl.adapters.sglang.common import completion_from_output
from monorl.adapters.sglang.engine import SGLangEngine
from monorl.core.types import SamplingParams, TopLogprobs


def _top(token: int, k: int) -> list[tuple[float, int, None]]:
    """``meta_info.output_top_logprobs`` entry of one position: ``(logprob, token_id, text)``."""
    return [(-0.25 - j, token + j, None) for j in range(k)]


def _expected(token_ids: list[int], k: int) -> TopLogprobs:
    return TopLogprobs.from_positions([[(token + j, -0.25 - j) for j in range(k)] for token in token_ids], k)


def test_engine_requests_and_returns_top_k(fake_sglang, sglang_config, monkeypatch) -> None:
    engine = SGLangEngine(sglang_config)
    fake = type(engine.engine)
    original = fake.async_generate
    seen: list[dict[str, Any]] = []

    async def async_generate(self, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        outputs = await original(self, *args, **kwargs)
        for out in outputs:
            if kwargs.get("top_logprobs_num"):
                out["meta_info"]["output_top_logprobs"] = [
                    _top(t, kwargs["top_logprobs_num"]) for t in out["output_ids"]
                ]
        return outputs

    monkeypatch.setattr(fake, "async_generate", async_generate)
    try:
        out = engine.generate([[1, 2, 3]], SamplingParams(n=2, max_new_tokens=3, top_logprobs=4))
        assert seen[-1]["top_logprobs_num"] == 4 and seen[-1]["return_logprob"] is True
        for completion in out[0]:
            assert completion.top_logprobs == _expected(completion.token_ids, 4)
        plain = engine.generate([[1, 2, 3]], SamplingParams(n=1, max_new_tokens=3))
        assert "top_logprobs_num" not in seen[-1]
        assert plain[0][0].top_logprobs is None
    finally:
        engine.shutdown()


def test_server_client_requests_and_returns_top_k(run_config_dict) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        tokens = [500, 501]
        meta: dict[str, Any] = {
            "id": "r0",
            "finish_reason": {"type": "stop", "matched": 1},
            "output_token_logprobs": [[-0.1, 500, None], [-0.2, 501, None]],
        }
        if body.get("top_logprobs_num"):  # JSON turns the tuples into lists
            meta["output_top_logprobs"] = [
                [list(e) for e in _top(t, body["top_logprobs_num"])] for t in tokens
            ]
        return httpx.Response(200, json=[{"text": "t", "output_ids": tokens, "meta_info": meta}])

    config = make_config(run_config_dict, extra={"mode": "server", "url": "http://sglang.test:30000"})
    client = SGLangServerClient(config, transport=httpx.MockTransport(handler))
    try:
        out = client.generate([[7, 8]], SamplingParams(n=1, max_new_tokens=2, top_logprobs=3))
        assert bodies[-1]["top_logprobs_num"] == 3 and bodies[-1]["return_logprob"] is True
        assert out[0][0].top_logprobs == _expected([500, 501], 3)
        assert out[0][0].logprobs == [-0.1, -0.2]
        out = client.generate([[7, 8]], SamplingParams(n=1, max_new_tokens=2))
        assert "top_logprobs_num" not in bodies[-1] and out[0][0].top_logprobs is None
    finally:
        client.shutdown()


def test_missing_or_misaligned_top_logprobs() -> None:
    output = {
        "text": "t",
        "output_ids": [500, 501],
        "meta_info": {"output_top_logprobs": [_top(500, 2), None]},
    }
    # a position without candidates: no usable top-k for this completion
    completion = completion_from_output(output, prompt_ids=[1], want_logprobs=False, top_logprobs=2)
    assert completion.top_logprobs is None
    output["meta_info"]["output_top_logprobs"] = [_top(500, 2)]
    with pytest.raises(ValueError, match="top logprobs for 1 of 2"):
        completion_from_output(output, prompt_ids=[1], want_logprobs=False, top_logprobs=2)
