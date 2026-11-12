"""Builds a random tiny ``Qwen2VLForConditionalGeneration`` (+ the real Qwen2-VL processor) for
tests. Only the processor/tokenizer files of ``Qwen/Qwen2-VL-2B-Instruct`` are needed (json/txt,
a few MB); they are fetched once into the HF cache and reused offline afterwards. When they are
not available (no cache, no network) the dependent tests skip.

Image size limits are shrunk so a 56x56 image is 4 placeholder tokens (16 patches) and a 112x56
image 8 tokens, keeping forward passes fast on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

VLM_PROCESSOR_ID = "Qwen/Qwen2-VL-2B-Instruct"
PATCH = 14
MERGE = 2
IMAGE_SIZE = {"shortest_edge": (PATCH * MERGE) ** 2 * 4, "longest_edge": (PATCH * MERGE) ** 2 * 16}


def load_vlm_processor():
    from transformers import AutoProcessor

    last: Exception | None = None
    for local_only in (True, False):
        try:
            return AutoProcessor.from_pretrained(
                VLM_PROCESSOR_ID, local_files_only=local_only, size=dict(IMAGE_SIZE)
            )
        except Exception as e:  # noqa: BLE001 - try the network next, then skip
            last = e
    pytest.skip(f"Qwen2-VL processor files unavailable (needs one-time network access): {last}")


def build_tiny_vlm(out: Path) -> Path:
    """Random tiny Qwen2-VL saved with its processor to ``out``; returns ``out``."""
    import torch
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

    processor = load_vlm_processor()
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    cfg = Qwen2VLConfig(
        text_config=dict(
            vocab_size=len(tok),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
            rope_scaling={"type": "mrope", "mrope_section": [2, 2, 4]},  # sums to head_dim // 2
            tie_word_embeddings=True,
        ),
        vision_config=dict(
            depth=2,
            embed_dim=32,
            hidden_size=64,
            num_heads=4,
            in_channels=3,
            patch_size=PATCH,
            spatial_patch_size=PATCH,
            spatial_merge_size=MERGE,
            temporal_patch_size=2,
            mlp_ratio=2,
        ),
        image_token_id=tok.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=tok.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=tok.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=tok.convert_tokens_to_ids("<|vision_end|>"),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = Qwen2VLForConditionalGeneration(cfg)
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    processor.save_pretrained(out)
    return out


def write_test_images(out: Path) -> dict[str, Path]:
    """Two PNGs with different patch counts: ``red`` 56x56 (4 image tokens), ``blue`` 112x56 (8)."""
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, size, colour in (("red", (56, 56), (220, 30, 30)), ("blue", (112, 56), (30, 30, 220))):
        img = Image.new("RGB", size, colour)
        for x in range(0, size[0], 7):  # some structure so patches differ
            for y in range(0, size[1], 7):
                if (x // 7 + y // 7) % 2 == 0:
                    img.putpixel((x, y), (255, 255, 255))
        path = out / f"{name}.png"
        img.save(path)
        paths[name] = path
    return paths


def image_chat(image_path: Path | str, question: str = "What colour is this?") -> list[dict]:
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": question}],
        }
    ]
