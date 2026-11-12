"""Multimodal prompts: ``Engine.generate(input_ids=..., image_data=...)`` and ``POST /generate``
``image_data`` data URLs (SGLang v0.5.23 ``GenerateReqInput``)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from fake_sglang import Engine as FakeEngine
from fake_sglang import make_config
from PIL import Image

from monorl.adapters.sglang.client import SGLangServerClient
from monorl.adapters.sglang.common import encode_prompt, expand_image_data, image_data_urls, is_media_prompt
from monorl.adapters.sglang.engine import SGLangEngine
from monorl.core.errors import ConfigError
from monorl.core.types import SamplingParams

PLACEHOLDER_IDS = [201, 202, 203]


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def apply_chat_template(self, messages: Any, **kw: Any) -> dict[str, torch.Tensor]:
        self.calls.append(messages)
        return {"input_ids": torch.tensor([PLACEHOLDER_IDS])}


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    p = tmp_path / "blue.png"
    Image.new("RGB", (5, 3), (0, 0, 255)).save(p)
    return p


@pytest.fixture
def fake_processor(monkeypatch: pytest.MonkeyPatch) -> FakeProcessor:
    import monorl.rollout.media as media

    proc = FakeProcessor()
    monkeypatch.setattr(media, "load_processor", lambda config: proc)
    return proc


def _chat(image_path: Path) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, {"type": "image", "image": str(image_path)}]}]


def test_common_helpers(image_path: Path, fake_processor: FakeProcessor, tiny_model_dir: Path) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    assert is_media_prompt(_chat(image_path)) and not is_media_prompt([1, 2]) and not is_media_prompt([{"role": "user", "content": "x"}])
    ids, images = encode_prompt(tok, fake_processor, _chat(image_path))
    assert ids == PLACEHOLDER_IDS and images is not None and images[0].size == (5, 3)
    with pytest.raises(ConfigError, match="AutoProcessor"):
        encode_prompt(tok, None, _chat(image_path))

    assert expand_image_data([None, None], SamplingParams(n=2)) is None
    img = images[0]
    assert expand_image_data([[img], None], SamplingParams(n=2)) == [[img], None]
    assert expand_image_data([[img], None], SamplingParams(n=2, seed=1)) == [[img], [img], None, None]
    urls = image_data_urls([[img, "http://x/y.png"], None])
    assert urls is not None and urls[1] is None and urls[0][1] == "http://x/y.png"
    assert urls[0][0].startswith("data:image/png;base64,")
    assert base64.b64decode(urls[0][0].split(",", 1)[1])[:4] == b"\x89PNG"
    assert image_data_urls(None) is None


def test_engine_passes_image_data(fake_sglang, sglang_config, image_path: Path, fake_processor: FakeProcessor) -> None:
    eng = SGLangEngine(sglang_config)
    try:
        out = eng.generate([_chat(image_path), [{"role": "user", "content": "text"}]], SamplingParams(n=2, logprobs=True))
        call = eng.engine.calls[-1].kwargs
        assert call["input_ids"][0] == PLACEHOLDER_IDS
        assert len(call["image_data"]) == 2 and call["image_data"][1] is None
        assert call["image_data"][0][0].size == (5, 3)
        assert out[0][0].prompt_token_ids == PLACEHOLDER_IDS and len(out[0]) == 2
        assert len(fake_processor.calls) == 1
        eng.generate([[1, 2, 3]], SamplingParams(n=1))
        assert eng.engine.calls[-1].kwargs["image_data"] is None
    finally:
        eng.shutdown()
    assert isinstance(eng.engine, FakeEngine)


def test_client_sends_image_data_urls(run_config_dict, image_path: Path, fake_processor: FakeProcessor) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        outs = []
        for i in range(len(body["input_ids"])):
            for k in range(body["sampling_params"][i].get("n", 1)):
                outs.append({"text": f"t{i}{k}", "output_ids": [i, k], "meta_info": {"finish_reason": {"type": "stop"}}})
        return httpx.Response(200, json=outs)

    cfg = make_config(run_config_dict, extra={"mode": "server", "url": "http://sgl.test:30000"})
    client = SGLangServerClient(cfg, transport=httpx.MockTransport(handler))
    try:
        out = client.generate([_chat(image_path), [{"role": "user", "content": "plain"}]], SamplingParams(n=1))
    finally:
        client.shutdown()
    body = bodies[-1]
    assert body["input_ids"][0] == PLACEHOLDER_IDS
    assert body["image_data"][1] is None and len(body["image_data"][0]) == 1
    assert body["image_data"][0][0].startswith("data:image/png;base64,")
    json.dumps(body)  # whole body is JSON-serialisable
    assert out[0][0].prompt_token_ids == PLACEHOLDER_IDS and out[1][0].text == "t10"
