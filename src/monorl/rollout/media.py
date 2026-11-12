"""Media loading and processor-aware chat encoding shared by engines, trainers and adapters.

A model is treated as a vision-language model when its ``AutoConfig`` carries a vision tower
(``vision_config`` / ``image_token_id``) or a known VLM ``model_type`` **and** its
``AutoProcessor`` exposes an image processor. Text-only models never pay for a processor load:
:func:`load_processor` returns ``None`` after inspecting the config alone.

Encoding recipe (identical in the HF engine and the batch builder, so prompt ids recorded at
rollout time are what the trainer sees)::

    processor.apply_chat_template(hf_content_parts(messages), add_generation_prompt=True,
                                  tokenize=True, return_dict=True, return_tensors="pt")

where every image part has been loaded into a PIL image. The returned dict holds ``input_ids``
(with media placeholder tokens), token-aligned tensors (``attention_mask``,
``mm_token_type_ids`` for Qwen2-VL-style models) and per-sequence media tensors
(``pixel_values``, ``image_grid_thw``, ``image_sizes``, ...), all with a leading batch dim of 1.
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from monorl.core.types import MEDIA_PART_TYPES, Message, messages_have_media

if TYPE_CHECKING:
    import torch
    from PIL import Image

    from monorl.config.schema import RunConfig

log = logging.getLogger(__name__)

KNOWN_VLM_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_vl_moe",
        "llava",
        "llava_next",
        "llava_next_video",
        "llava_onevision",
        "idefics2",
        "idefics3",
        "smolvlm",
        "paligemma",
        "gemma3",
        "gemma3n",
        "mllama",
        "pixtral",
        "mistral3",
        "internvl",
        "kimi_vl",
        "glm4v",
        "aya_vision",
        "phi4_multimodal",
        "cohere2_vision",
        "lfm2_vl",
        "perception_lm",
        "got_ocr2",
    }
)
"""``config.model_type`` values of transformers VLMs; the structural checks in
:func:`is_vlm_config` catch families not listed here."""

TOKEN_ALIGNED_KEYS: frozenset[str] = frozenset(
    {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
)
"""Processor outputs shaped ``[1, T]`` and aligned with ``input_ids``; everything else returned by
the processor is a per-sequence media tensor concatenated along dim 0 when batching."""

_HTTP_TIMEOUT = 30.0


# -- images -------------------------------------------------------------------------------------
def load_image(ref: str | Image.Image) -> Image.Image:
    """Load one image reference as an RGB PIL image.

    Accepts a PIL image (returned converted), a local path, ``file://``, ``data:image/...;base64,``,
    ``http(s)://`` (fetched with ``httpx``) and any other URI (``s3://``, ``hf://``, ...) through
    :class:`monorl.data.media.MediaResolver`, which caches it under ``$MONORL_HOME/cache/media``.
    """
    from PIL import Image

    if isinstance(ref, Image.Image):
        return ref.convert("RGB")
    ref = str(ref)
    if ref.startswith("data:"):
        return _decode_data_url(ref)
    parsed = urlparse(ref)
    if parsed.scheme in ("http", "https"):
        import httpx

        resp = httpx.get(ref, timeout=_HTTP_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme and len(parsed.scheme) > 1:  # a real scheme, not a Windows drive letter
        from monorl.data.media import MediaResolver

        refs = MediaResolver().image_refs(ref, where="load_image")
        if not refs:
            raise FileNotFoundError(f"no image behind {ref!r}")
        return load_image(refs[0])
    else:
        path = Path(ref)
    with Image.open(path) as img:
        return img.convert("RGB")


def _decode_data_url(ref: str) -> Image.Image:
    from PIL import Image

    header, sep, payload = ref.partition(",")
    if not sep or ";base64" not in header:
        raise ValueError("only base64 data: URLs are supported for images")
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"invalid base64 image data URL: {e}") from e
    return Image.open(io.BytesIO(raw)).convert("RGB")


def image_data_url(image: Image.Image, *, format: str = "PNG") -> str:
    """``data:image/<fmt>;base64,...`` for an in-memory image (OpenAI-style ``image_url`` parts)."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    mime = "image/jpeg" if format.upper() in ("JPG", "JPEG") else f"image/{format.lower()}"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def chat_images(messages: Sequence[Message]) -> list[Image.Image]:
    """Every image part of ``messages``, loaded, in message/part order."""
    images = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    images.append(load_image(part.get("image", "")))
    return images


def hf_content_parts(messages: Sequence[Message], *, load_media: bool = True) -> list[dict[str, Any]]:
    """monorl messages → the shape HF processor chat templates expect: media parts become
    ``{"type": "image", "image": <PIL image>}`` (``load_media=False`` keeps the bare
    ``{"type": "image"}`` marker, enough for ids-only rendering); text parts and string content
    pass through unchanged."""
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(dict(m))
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append({"type": "text", "text": str(part)})
                continue
            kind = part.get("type")
            if kind in MEDIA_PART_TYPES:
                if kind == "image" and load_media:
                    parts.append({"type": "image", "image": load_image(part.get("image", ""))})
                elif kind == "image":
                    parts.append({"type": "image"})
                else:  # audio/video: hand the reference to the processor as-is
                    parts.append({"type": kind, kind: part.get(kind)})
            else:
                parts.append(dict(part))
        new = dict(m)
        new["content"] = parts
        out.append(new)
    return out


# -- processors -----------------------------------------------------------------------------------
def is_vlm_config(hf_config: Any) -> bool:
    model_type = str(getattr(hf_config, "model_type", "") or "")
    if model_type in KNOWN_VLM_MODEL_TYPES:
        return True
    if getattr(hf_config, "vision_config", None) is not None:
        return True
    return any(getattr(hf_config, k, None) is not None for k in ("image_token_id", "image_token_index"))


def is_vlm_processor(obj: Any) -> bool:
    return getattr(obj, "image_processor", None) is not None and callable(
        getattr(obj, "apply_chat_template", None)
    )


def _read_hf_config(config: RunConfig) -> Any | None:
    from transformers import AutoConfig

    model = config.model
    try:
        return AutoConfig.from_pretrained(
            model.name_or_path, revision=model.revision, trust_remote_code=model.trust_remote_code
        )
    except Exception as e:  # noqa: BLE001 - unreadable config: behave as text-only, model loading reports the real error
        log.debug("cannot read config for %s: %s", model.name_or_path, e)
        return None


def is_vlm_model(config: RunConfig) -> bool:
    """True when ``config.model`` is a vision-language checkpoint (see module docstring)."""
    hf_config = _read_hf_config(config)
    return hf_config is not None and is_vlm_config(hf_config)


def auto_model_class(config: RunConfig) -> Any:
    """``AutoModelForImageTextToText`` for VLM checkpoints (``AutoModelForCausalLM`` does not map
    most of them, or maps them to their text-only backbone), else ``AutoModelForCausalLM``."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    return AutoModelForImageTextToText if is_vlm_model(config) else AutoModelForCausalLM


def load_processor(config: RunConfig) -> Any | None:
    """``AutoProcessor`` for VLM checkpoints, ``None`` for text-only models (decided from the
    model config, so text-only runs never load a second tokenizer). Applies
    ``model.chat_template`` and the pad-token fix to the processor's tokenizer."""
    from transformers import AutoProcessor

    if not is_vlm_model(config):
        return None
    model = config.model
    try:
        processor = AutoProcessor.from_pretrained(
            model.tokenizer_name_or_path or model.name_or_path,
            revision=model.revision,
            trust_remote_code=model.trust_remote_code,
        )
    except Exception as e:  # noqa: BLE001 - e.g. missing preprocessor_config.json or image backend
        log.warning(
            "load_processor: %s looks like a VLM but AutoProcessor failed (%s); using text-only path",
            model.name_or_path,
            e,
        )
        return None
    if not is_vlm_processor(processor):
        return None
    tok = getattr(processor, "tokenizer", None)
    if model.chat_template is not None:
        processor.chat_template = model.chat_template
        if tok is not None:
            tok.chat_template = model.chat_template
    if tok is not None and tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return processor


def vlm_encode(
    processor: Any, messages: Sequence[Message], *, add_generation_prompt: bool = True
) -> dict[str, torch.Tensor]:
    """Processor outputs for one chat (batch dim 1): ``input_ids`` with media placeholders plus
    every media tensor the model forward needs. See the module docstring for the recipe."""
    enc = processor.apply_chat_template(
        hf_content_parts(messages),
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {k: v for k, v in dict(enc).items() if hasattr(v, "shape")}


def vlm_prompt_ids(processor: Any, messages: Sequence[Message]) -> list[int]:
    """Prompt ids (placeholder tokens included) the policy conditions on for a multimodal chat."""
    return [int(t) for t in vlm_encode(processor, messages)["input_ids"][0].tolist()]


def needs_processor(messages: Sequence[Message]) -> bool:
    """True when a chat prompt must go through the processor (it carries media parts)."""
    return messages_have_media(list(messages))


def split_processor_outputs(
    enc: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``(token_aligned, per_sequence)``: ``[1, T]`` tensors aligned with ``input_ids`` (minus
    ``input_ids``/``attention_mask`` themselves) and the remaining media tensors."""
    width = int(enc["input_ids"].shape[1])
    aligned: dict[str, torch.Tensor] = {}
    media: dict[str, torch.Tensor] = {}
    for k, v in enc.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if k in TOKEN_ALIGNED_KEYS or (v.ndim == 2 and tuple(v.shape) == (1, width) and k.endswith("_ids")):
            aligned[k] = v
        else:
            media[k] = v
    return aligned, media


def collate_media(per_sequence: Sequence[dict[str, torch.Tensor] | None]) -> dict[str, torch.Tensor]:
    """Concatenate per-sequence media tensors along dim 0 in sequence order (how HF processors
    batch them: Qwen2-VL stacks patches ``[N, D]`` + ``image_grid_thw [n_img, 3]``, LLaVA stacks
    images ``[n_img, C, H, W]``). Sequences without media contribute nothing."""
    import torch

    keys: list[str] = []
    for d in per_sequence:
        for k in d or ():
            if k not in keys:
                keys.append(k)
    return {k: torch.cat([d[k] for d in per_sequence if d and k in d], dim=0) for k in keys}
