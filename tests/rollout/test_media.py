"""``monorl.rollout.media`` loaders and message conversion (no model needed)."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from monorl.rollout.media import (
    chat_images,
    hf_content_parts,
    image_data_url,
    is_vlm_config,
    is_vlm_processor,
    load_image,
    needs_processor,
)


@pytest.fixture
def png(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "dot.png"
    Image.new("RGB", (8, 6), (10, 20, 30)).save(path)
    return path


def test_load_image_path_file_uri_and_pil(png: Path) -> None:
    from PIL import Image

    img = load_image(str(png))
    assert img.mode == "RGB" and img.size == (8, 6)
    assert load_image(png.as_uri()).size == (8, 6)
    assert load_image(Image.new("L", (3, 3))).mode == "RGB"


def test_load_image_data_url_roundtrip(png: Path) -> None:
    payload = base64.b64encode(png.read_bytes()).decode()
    img = load_image(f"data:image/png;base64,{payload}")
    assert img.size == (8, 6) and img.getpixel((0, 0)) == (10, 20, 30)
    url = image_data_url(img)
    assert url.startswith("data:image/png;base64,")
    assert load_image(url).getpixel((0, 0)) == (10, 20, 30)
    assert image_data_url(img, format="JPEG").startswith("data:image/jpeg;base64,")
    with pytest.raises(ValueError):
        load_image("data:image/png,notbase64")


def test_load_image_http(png: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **kwargs) -> httpx.Response:
        assert url == "https://example.test/dot.png"
        return httpx.Response(200, content=png.read_bytes(), request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    assert load_image("https://example.test/dot.png").size == (8, 6)


def test_load_image_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(str(tmp_path / "nope.png"))


def test_hf_content_parts_and_chat_images(png: Path) -> None:
    messages = [
        {"role": "system", "content": "Be brief."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(png)},
                {"type": "text", "text": "What?"},
                {"type": "video", "video": "v.mp4"},
            ],
        },
    ]
    assert needs_processor(messages)
    assert not needs_processor([{"role": "user", "content": [{"type": "text", "text": "x"}]}])
    assert len(chat_images(messages)) == 1
    loaded = hf_content_parts(messages)
    assert loaded[0] == {"role": "system", "content": "Be brief."}
    img_part, text_part, video_part = loaded[1]["content"]
    assert img_part["type"] == "image" and img_part["image"].size == (8, 6)
    assert text_part == {"type": "text", "text": "What?"}
    assert video_part == {"type": "video", "video": "v.mp4"}
    bare = hf_content_parts(messages, load_media=False)
    assert bare[1]["content"][0] == {"type": "image"}
    assert messages[1]["content"][0] == {"type": "image", "image": str(png)}  # input untouched


def test_vlm_detection_predicates() -> None:
    assert is_vlm_config(SimpleNamespace(model_type="qwen2_vl"))
    assert is_vlm_config(SimpleNamespace(model_type="custom", vision_config=object()))
    assert is_vlm_config(SimpleNamespace(model_type="custom", image_token_index=5))
    assert not is_vlm_config(SimpleNamespace(model_type="llama"))
    assert is_vlm_processor(
        SimpleNamespace(image_processor=object(), apply_chat_template=lambda *a, **k: None)
    )
    assert not is_vlm_processor(SimpleNamespace(apply_chat_template=lambda *a, **k: None))
