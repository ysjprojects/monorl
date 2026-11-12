"""HF engine on a (random, tiny) Qwen2-VL: processor-aware prompt encoding, batched generation
with pixel inputs, and logprobs that match an independent forward pass."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.types import SamplingParams
from monorl.rollout.hf_engine import HFRolloutEngine, chat_prompt_ids, encode_prompt
from monorl.rollout.media import (
    collate_media,
    hf_content_parts,
    is_vlm_model,
    load_processor,
    vlm_encode,
    vlm_prompt_ids,
)
from tests.rollout.tiny_vlm import image_chat

TEXT_CHAT = [{"role": "user", "content": "Say hello."}]


def _image_token_id(engine: HFRolloutEngine) -> int:
    return int(engine.model.config.image_token_id)


def test_detection_and_processor(
    vlm_config: RunConfig, tiny_config: RunConfig, vlm_engine: HFRolloutEngine
) -> None:
    assert is_vlm_model(vlm_config)
    assert not is_vlm_model(tiny_config)
    assert load_processor(tiny_config) is None
    assert vlm_engine.processor is not None
    assert type(vlm_engine.model).__name__ == "Qwen2VLForConditionalGeneration"
    assert vlm_engine.tokenizer.pad_token_id is not None


def test_text_prompt_on_vlm_uses_tokenizer_path(vlm_engine: HFRolloutEngine) -> None:
    enc = encode_prompt(TEXT_CHAT, vlm_engine.tokenizer, vlm_engine.processor)
    assert not enc.multimodal
    assert enc.ids == chat_prompt_ids(vlm_engine.tokenizer, TEXT_CHAT)
    assert _image_token_id(vlm_engine) not in enc.ids


def test_media_prompt_ids_include_placeholders(
    vlm_engine: HFRolloutEngine, test_images: dict[str, Path]
) -> None:
    proc = vlm_engine.processor
    chat = image_chat(test_images["red"])
    enc = encode_prompt(chat, vlm_engine.tokenizer, proc)
    assert enc.multimodal
    image_id = _image_token_id(vlm_engine)
    assert enc.ids.count(image_id) == 4  # 56x56 → 16 patches / merge 2x2
    assert enc.ids == vlm_prompt_ids(proc, chat)
    assert set(enc.media) >= {"pixel_values", "image_grid_thw"}
    assert enc.media["pixel_values"].shape[0] == 16
    assert "mm_token_type_ids" in enc.token_aligned
    assert enc.token_aligned["mm_token_type_ids"].shape == (1, len(enc.ids))
    # the processor sees PIL images in HF-shaped parts; text parts untouched
    parts = hf_content_parts(chat)
    assert parts[0]["content"][0]["type"] == "image" and hasattr(parts[0]["content"][0]["image"], "size")
    assert parts[0]["content"][1] == {"type": "text", "text": "What colour is this?"}
    wide = encode_prompt(image_chat(test_images["blue"]), vlm_engine.tokenizer, proc)
    assert wide.ids.count(image_id) == 8


def test_generate_n_completions_with_images(
    vlm_engine: HFRolloutEngine, test_images: dict[str, Path]
) -> None:
    params = SamplingParams(n=3, max_new_tokens=5, temperature=1.0, seed=1)
    prompts = [image_chat(test_images["red"]), image_chat(test_images["blue"], "Describe it."), TEXT_CHAT]
    out = vlm_engine.generate(prompts, params)
    assert [len(g) for g in out] == [3, 3, 3]
    image_id = _image_token_id(vlm_engine)
    expected = [encode_prompt(p, vlm_engine.tokenizer, vlm_engine.processor).ids for p in prompts]
    for group, ids, n_img in zip(out, expected, (4, 8, 0), strict=True):
        for c in group:
            assert c.prompt_token_ids == ids
            assert c.prompt_token_ids.count(image_id) == n_img
            assert 1 <= len(c.token_ids) <= 5
            assert len(c.logprobs) == len(c.token_ids)
            assert c.finish_reason in ("stop", "length")
    assert len({tuple(c.token_ids) for c in out[0]}) > 1


def test_seed_is_deterministic_with_images(vlm_engine: HFRolloutEngine, test_images: dict[str, Path]) -> None:
    params = SamplingParams(n=2, max_new_tokens=4, temperature=1.0, seed=9)
    a = vlm_engine.generate([image_chat(test_images["red"])], params)
    b = vlm_engine.generate([image_chat(test_images["red"])], params)
    assert [[c.token_ids for c in g] for g in a] == [[c.token_ids for c in g] for g in b]
    assert [[c.logprobs for c in g] for g in a] == [[c.logprobs for c in g] for g in b]


def _reference_logprobs(
    engine: HFRolloutEngine, chat: list[dict], completion_ids: list[int], temperature: float
) -> torch.Tensor:
    enc = vlm_encode(engine.processor, chat)
    prompt = enc["input_ids"]
    full = torch.cat([prompt, torch.tensor([completion_ids])], dim=1)
    kwargs = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
    mm = torch.cat([enc["mm_token_type_ids"], torch.zeros((1, len(completion_ids)), dtype=torch.long)], dim=1)
    with torch.no_grad():
        logits = engine.model(
            input_ids=full, attention_mask=torch.ones_like(full), mm_token_type_ids=mm, **kwargs
        ).logits
    start = prompt.shape[1]
    lp = torch.log_softmax(logits[0, start - 1 : -1].float() / temperature, dim=-1)
    return lp.gather(1, torch.tensor(completion_ids)[:, None]).squeeze(1)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_logprobs_match_forward_pass(
    vlm_engine: HFRolloutEngine, test_images: dict[str, Path], temperature: float
) -> None:
    params = SamplingParams(n=2, max_new_tokens=6, temperature=temperature, seed=5, logprobs=True)
    prompts = [image_chat(test_images["blue"]), image_chat(test_images["red"], "Short answer please.")]
    out = vlm_engine.generate(prompts, params)
    for chat, group in zip(prompts, out, strict=True):
        for c in group:
            ref = _reference_logprobs(vlm_engine, chat, c.token_ids, temperature)
            torch.testing.assert_close(torch.tensor(c.logprobs), ref, atol=1e-4, rtol=0.0)


def test_greedy_replicates_n_with_images(vlm_engine: HFRolloutEngine, test_images: dict[str, Path]) -> None:
    out = vlm_engine.generate(
        [image_chat(test_images["red"])], SamplingParams(n=3, max_new_tokens=4, temperature=0.0)
    )
    assert len(out[0]) == 3
    assert len({tuple(c.token_ids) for c in out[0]}) == 1


def test_multi_turn_chat_is_prefix_consistent(
    vlm_engine: HFRolloutEngine, test_images: dict[str, Path]
) -> None:
    """Re-encoding the full chat after an assistant turn extends the first prompt token-exactly,
    which is what lets the batch builder train every turn of a multimodal episode. (A fixed
    assistant text: the random model's samples may start with newlines that BPE-merge with the
    template's ``\\n`` — the same re-tokenisation caveat the text path reports as
    ``tokens/prefix_mismatch``.)"""
    first = vlm_engine.generate(
        [image_chat(test_images["red"])], SamplingParams(n=1, max_new_tokens=4, temperature=0.0)
    )[0][0]
    chat = image_chat(test_images["red"]) + [
        {"role": "assistant", "content": "Red."},
        {"role": "user", "content": "Sure?"},
    ]
    second = vlm_engine.generate([chat], SamplingParams(n=1, max_new_tokens=2, temperature=0.0))[0][0]
    head = first.prompt_token_ids
    assert second.prompt_token_ids[: len(head)] == head
    assert second.prompt_token_ids.count(_image_token_id(vlm_engine)) == 4


def test_collate_media_concatenates_in_order(
    vlm_engine: HFRolloutEngine, test_images: dict[str, Path]
) -> None:
    red = encode_prompt(image_chat(test_images["red"]), vlm_engine.tokenizer, vlm_engine.processor).media
    blue = encode_prompt(image_chat(test_images["blue"]), vlm_engine.tokenizer, vlm_engine.processor).media
    batched = collate_media([red, None, blue])
    assert batched["pixel_values"].shape[0] == red["pixel_values"].shape[0] + blue["pixel_values"].shape[0]
    assert torch.equal(batched["image_grid_thw"], torch.cat([red["image_grid_thw"], blue["image_grid_thw"]]))
    assert collate_media([None, {}]) == {}
