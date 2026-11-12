"""Multimodal prompts: ``TokensPrompt(multi_modal_data=...)`` in colocate mode and
``/v1/chat/completions`` with ``image_url`` parts in server mode (vLLM v0.33.0 shapes)."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from PIL import Image
from vllm_fakes import FakeLLM

from monorl.adapters.vllm.client import VllmServerClient
from monorl.adapters.vllm.common import (
    chat_completion_kwargs,
    chat_completion_messages,
    completions_from_chat_response,
    encode_prompt,
    limit_mm_per_prompt,
    tokens_prompt_kwargs,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.registry import registry
from monorl.core.types import SamplingParams

PLACEHOLDER_IDS = [101, 102, 103, 104]


class FakeProcessor:
    """``AutoProcessor`` stand-in: records the HF-shaped chat and returns fixed placeholder ids."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, torch.Tensor]:
        self.calls.append(messages)
        return {"input_ids": torch.tensor([PLACEHOLDER_IDS]), "pixel_values": torch.zeros(1, 3, 2, 2)}


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    p = tmp_path / "red.png"
    Image.new("RGB", (6, 4), (255, 0, 0)).save(p)
    return p


@pytest.fixture
def fake_processor(monkeypatch: pytest.MonkeyPatch) -> FakeProcessor:
    proc = FakeProcessor()
    import monorl.rollout.media as media

    monkeypatch.setattr(media, "load_processor", lambda config: proc)
    return proc


def _chat(image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": "colour?"}],
        }
    ]


# -- common helpers ---------------------------------------------------------------------------------


def test_encode_prompt_uses_processor_ids_and_images(
    image_path: Path, fake_processor: FakeProcessor, tiny_model_dir: Path
) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    ids, images = encode_prompt(tok, fake_processor, _chat(image_path))
    assert ids == PLACEHOLDER_IDS
    assert images is not None and len(images) == 1 and images[0].size == (6, 4)
    assert fake_processor.calls[0][0]["content"][0]["type"] == "image"
    text_ids, none = encode_prompt(tok, fake_processor, [{"role": "user", "content": "hi"}])
    assert none is None and text_ids and text_ids != PLACEHOLDER_IDS
    assert encode_prompt(tok, None, [1, 2, 3]) == ([1, 2, 3], None)
    with pytest.raises(ConfigError, match="AutoProcessor"):
        encode_prompt(tok, None, _chat(image_path))


def test_tokens_prompt_and_limit_mm() -> None:
    assert tokens_prompt_kwargs([1, 2], None) == {"prompt_token_ids": [1, 2]}
    assert tokens_prompt_kwargs([1, 2], ["img"]) == {
        "prompt_token_ids": [1, 2],
        "multi_modal_data": {"image": ["img"]},
    }
    assert limit_mm_per_prompt({}) is None
    assert limit_mm_per_prompt({"limit_mm_per_prompt": 3}) == {"image": 3}
    assert limit_mm_per_prompt({"limit_mm_per_prompt": {"image": 2, "video": 1}}) == {"image": 2, "video": 1}
    with pytest.raises(ConfigError):
        limit_mm_per_prompt({"limit_mm_per_prompt": "two"})


def test_chat_completion_messages_and_kwargs(image_path: Path) -> None:
    msgs = chat_completion_messages([{"role": "system", "content": "be brief"}, *_chat(image_path)])
    assert msgs[0] == {"role": "system", "content": "be brief"}
    parts = msgs[1]["content"]
    assert parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"
    assert parts[1] == {"type": "text", "text": "colour?"}
    data_url = "data:image/png;base64,AAAA"
    passthrough = chat_completion_messages(
        [{"role": "user", "content": [{"type": "image", "image": data_url}]}]
    )
    assert passthrough[0]["content"][0]["image_url"]["url"] == data_url
    with pytest.raises(ConfigError, match="audio"):
        chat_completion_messages([{"role": "user", "content": [{"type": "audio", "audio": "a.wav"}]}])

    kw = chat_completion_kwargs(SamplingParams(n=3, max_new_tokens=5, top_k=-1, logprobs=True, seed=7))
    assert kw["n"] == 3 and kw["max_completion_tokens"] == 5 and kw["top_k"] == 0 and kw["seed"] == 7
    assert kw["logprobs"] is True and kw["top_logprobs"] == 0 and kw["return_token_ids"] is True
    assert kw["include_stop_str_in_output"] is True
    assert chat_completion_kwargs(SamplingParams(logprobs=False))["logprobs"] is False


def test_completions_from_chat_response_validation() -> None:
    resp = {
        "prompt_token_ids": [9, 8],
        "choices": [
            {
                "index": 1,
                "message": {"role": "assistant", "content": "b"},
                "token_ids": [2],
                "finish_reason": "length",
                "logprobs": {"content": [{"token": "b", "logprob": None}]},
            },
            {
                "index": 0,
                "message": {"role": "assistant", "content": "a"},
                "token_ids": [1, 3],
                "finish_reason": "stop",
                "logprobs": {"content": [{"token": "a", "logprob": -0.1}, {"token": "c", "logprob": -0.2}]},
            },
        ],
    }
    comps = completions_from_chat_response(resp, 2, want_logprobs=True)
    assert [c.text for c in comps] == ["a", "b"] and comps[0].logprobs == [-0.1, -0.2]
    assert math.isnan(comps[1].logprobs[0]) and comps[1].finish_reason == "length"
    assert comps[0].prompt_token_ids == [9, 8]
    with pytest.raises(ValueError, match="expected 1"):
        completions_from_chat_response(resp, 1, want_logprobs=False)
    with pytest.raises(ValueError, match="prompt_token_ids"):
        completions_from_chat_response({"choices": resp["choices"]}, 2, want_logprobs=False)
    no_lp = {
        "prompt_token_ids": [1],
        "choices": [{"index": 0, "message": {"content": "x"}, "token_ids": [1]}],
    }
    with pytest.raises(ValueError, match="logprobs.content"):
        completions_from_chat_response(no_lp, 1, want_logprobs=True)


# -- colocated engine ---------------------------------------------------------------------------


def test_engine_builds_multi_modal_tokens_prompt(
    fake_vllm, vllm_config: RunConfig, image_path: Path, fake_processor: FakeProcessor
) -> None:
    vllm_config.rollout.extra = {"limit_mm_per_prompt": {"image": 2}}
    eng = registry.get("rollout_engine", "vllm")(vllm_config)
    assert FakeLLM.instances[-1].kwargs["limit_mm_per_prompt"] == {"image": 2}
    out = eng.generate(
        [_chat(image_path), [{"role": "user", "content": "text only"}]], SamplingParams(n=1, logprobs=True)
    )
    prompts = FakeLLM.instances[-1].calls[-1][1]["prompts"]
    assert prompts[0]["prompt_token_ids"] == PLACEHOLDER_IDS
    assert len(prompts[0]["multi_modal_data"]["image"]) == 1 and prompts[0]["multi_modal_data"]["image"][
        0
    ].size == (6, 4)
    assert "multi_modal_data" not in prompts[1]
    assert out[0][0].prompt_token_ids == PLACEHOLDER_IDS and out[0][0].logprobs == [-0.1, -0.2, -0.3]
    assert len(fake_processor.calls) == 1  # text prompt never touched the processor


def test_engine_text_only_never_loads_processor(
    fake_vllm, vllm_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monorl.rollout.media as media

    def boom(config: RunConfig) -> None:
        raise AssertionError("processor loaded for a text prompt")

    monkeypatch.setattr(media, "load_processor", boom)
    eng = registry.get("rollout_engine", "vllm")(vllm_config)
    assert eng.generate([[1, 2, 3]], SamplingParams(n=1))[0][0].prompt_token_ids == [1, 2, 3]


# -- server client ------------------------------------------------------------------------------


class ChatServer:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            n = body["n"]
            choices = []
            for i in reversed(range(n)):
                choice: dict[str, Any] = {
                    "index": i,
                    "message": {"role": "assistant", "content": f"chat-{i}"},
                    "token_ids": [i, 5],
                    "finish_reason": "stop",
                    "logprobs": None,
                }
                if body.get("logprobs"):
                    choice["logprobs"] = {
                        "content": [{"token": "a", "logprob": -0.5}, {"token": "b", "logprob": -0.25}]
                    }
                choices.append(choice)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl",
                    "model": body["model"],
                    "choices": choices,
                    "prompt_token_ids": [7, 7, 7],
                },
            )
        if request.url.path == "/v1/completions":
            choices = [
                {
                    "index": p * body["n"] + i,
                    "text": f"t-{p}-{i}",
                    "token_ids": [p, i],
                    "prompt_token_ids": ids,
                    "finish_reason": "stop",
                    "logprobs": {"token_logprobs": [-0.1, -0.2]}
                    if body.get("logprobs") is not None
                    else None,
                }
                for p, ids in enumerate(body["prompt"])
                for i in range(body["n"])
            ]
            return httpx.Response(200, json={"choices": choices, "model": body["model"]})
        return httpx.Response(200, json={"ok": True})


def test_client_routes_image_prompts_through_chat_completions(
    vllm_config: RunConfig, image_path: Path
) -> None:
    vllm_config.rollout.extra = {"mode": "server", "url": "http://vllm.test:8000", "model_name": "policy"}
    server = ChatServer()
    client = VllmServerClient(vllm_config, transport=httpx.MockTransport(server.handler))
    text = [{"role": "user", "content": "hello"}]
    out = client.generate(
        [_chat(image_path), text, _chat(image_path)], SamplingParams(n=2, logprobs=True, max_new_tokens=4)
    )
    client.shutdown()

    paths = [r.url.path for r in server.requests]
    assert paths == ["/v1/completions", "/v1/chat/completions", "/v1/chat/completions"]
    chat_body = json.loads(server.requests[1].content)
    assert chat_body["model"] == "policy" and chat_body["n"] == 2 and chat_body["max_completion_tokens"] == 4
    assert (
        chat_body["logprobs"] is True
        and chat_body["top_logprobs"] == 0
        and chat_body["return_token_ids"] is True
    )
    parts = chat_body["messages"][0]["content"]
    assert parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert parts[1] == {"type": "text", "text": "colour?"}

    assert len(out) == 3 and all(len(group) == 2 for group in out)
    assert [c.text for c in out[0]] == ["chat-0", "chat-1"]  # sorted by index
    assert out[0][0].prompt_token_ids == [7, 7, 7] and out[0][0].logprobs == [-0.5, -0.25]
    assert out[1][0].text == "t-0-0" and out[1][0].logprobs == [
        -0.1,
        -0.2,
    ]  # text prompt took /v1/completions
    assert out[2][1].token_ids == [1, 5]
