"""monorl :class:`~monorl.core.interfaces.DataSource` → ``datasets.Dataset`` for TRL trainers.

GRPO columns: ``prompt`` (chat messages), ``sample_id``, ``reference`` (string form for TRL-native
reward functions) and ``sample_json`` (the full :class:`~monorl.core.types.Sample`, used to rebuild
exact samples on the reward side). SFT columns (``algorithm.name: sft``): conversational
``prompt`` + ``completion`` (one assistant turn) so ``SFTTrainer`` masks the prompt with
``completion_only_loss``.

Multimodal datasets (``data.image_field`` or image parts in the messages) add an ``images`` column
(list of PIL images, ``datasets.Image`` feature) and render every ``content`` as a list of parts
where images are the ``{"type": "image"}`` placeholders ``trl.data_utils.prepare_multimodal_messages``
fills from ``images`` (TRL 1.17 ``GRPOTrainer._generate_and_score_completions`` /
``SFTTrainer`` ``DataCollatorForVisionLanguageModeling``). Text-only datasets keep plain string
content so Arrow gets one schema.
"""

from __future__ import annotations

import json
from typing import Any

from monorl.config.schema import DataConfig, RunConfig
from monorl.core.errors import DataError
from monorl.core.types import Message, Sample, message_text, messages_have_media

PROMPT_COLUMN = "prompt"
SAMPLE_COLUMN = "sample_json"
IMAGES_COLUMN = "images"
COMPLETION_COLUMN = "completion"


def text_messages(messages: list[Message]) -> list[dict[str, str]]:
    """``role``/``content`` only, media dropped (one Arrow schema for text runs)."""
    return [{"role": str(m.get("role", "user")), "content": message_text(m)} for m in messages]


def structured_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Every ``content`` as a list of TRL parts; image parts become unfilled placeholders."""
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content", "")
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        else:
            for part in content:
                kind = part.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                elif kind == "image":
                    parts.append({"type": "image"})
                else:
                    raise DataError(f"trainer.backend='trl' supports text and image parts, got {kind!r}")
        out.append({"role": str(m.get("role", "user")), "content": parts})
    return out


def sample_images(sample: Sample) -> list[Any]:
    """Decoded PIL images of ``sample`` in message order (``monorl.rollout.media.chat_images``)."""
    from monorl.rollout.media import chat_images

    return chat_images(sample.messages)


def sample_to_row(sample: Sample, *, multimodal: bool = False) -> dict[str, Any]:
    ref = sample.reference
    row: dict[str, Any] = {
        PROMPT_COLUMN: structured_messages(sample.messages) if multimodal else text_messages(sample.messages),
        "sample_id": str(sample.id),
        "reference": None if ref is None else (ref if isinstance(ref, str) else json.dumps(ref, default=str)),
        SAMPLE_COLUMN: json.dumps(
            {"id": sample.id, "messages": sample.messages, "reference": ref, "metadata": sample.metadata},
            default=str,
        ),
    }
    if multimodal:
        row[IMAGES_COLUMN] = sample_images(sample)
    return row


def sft_sample_to_row(sample: Sample, *, response_field: str | None, multimodal: bool = False) -> dict[str, Any]:
    """``prompt``/``completion`` conversational row; the response comes from
    :func:`monorl.orchestrator.sft.sft_response` (``data.response_field`` → ``answer_field`` →
    trailing assistant turn)."""
    from monorl.orchestrator.sft import sft_response

    response = sft_response(sample, response_field=response_field)
    completion: list[dict[str, Any]] = [{"role": "assistant", "content": response}]
    if multimodal:
        completion = structured_messages(completion)  # type: ignore[arg-type]
    row: dict[str, Any] = {
        PROMPT_COLUMN: structured_messages(sample.messages) if multimodal else text_messages(sample.messages),
        COMPLETION_COLUMN: completion,
    }
    if multimodal:
        row[IMAGES_COLUMN] = sample_images(sample)
    return row


def sample_from_row(row: dict[str, Any]) -> Sample:
    """Inverse of :func:`sample_to_row` (works on TRL's per-example input dicts)."""
    raw = row.get(SAMPLE_COLUMN)
    if raw is None:
        return Sample(id=str(row.get("sample_id", "")), messages=list(row[PROMPT_COLUMN]), reference=row.get("reference"))
    data = json.loads(raw)
    return Sample(
        id=str(data["id"]),
        messages=list(data["messages"]),
        reference=data.get("reference"),
        metadata=dict(data.get("metadata") or {}),
    )


def is_multimodal(data_config: DataConfig, samples: list[Sample]) -> bool:
    return data_config.image_field is not None or any(messages_have_media(s.messages) for s in samples)


def _to_dataset(rows: list[dict[str, Any]], *, multimodal: bool) -> Any:
    from datasets import Dataset, Image, Sequence

    ds = Dataset.from_list(rows)
    if multimodal:
        ds = ds.cast_column(IMAGES_COLUMN, Sequence(Image()))
    return ds


def build_dataset(
    data_config: DataConfig,
    *,
    split: str | None = None,
    tokenizer: Any = None,
    seed: int = 0,
    sft: bool = False,
) -> Any:
    """Materialise ``data_config`` through :func:`monorl.data.open_source` into a ``datasets.Dataset``."""
    from monorl.data import open_source

    samples = list(open_source(data_config, split=split, tokenizer=tokenizer, seed=seed))
    if not samples:
        raise DataError(f"dataset {data_config.uri!r} yielded no samples")
    multimodal = is_multimodal(data_config, samples)
    if sft:
        rows = [sft_sample_to_row(s, response_field=data_config.response_field, multimodal=multimodal) for s in samples]
    else:
        rows = [sample_to_row(s, multimodal=multimodal) for s in samples]
    return _to_dataset(rows, multimodal=multimodal)


def build_train_and_eval(config: RunConfig, *, tokenizer: Any = None) -> tuple[Any, Any | None]:
    sft = config.is_sft
    train = build_dataset(config.data, tokenizer=tokenizer, seed=config.seed, sft=sft)
    eval_ds = None
    if config.data.eval_uri:
        eval_cfg = config.data.model_copy(update={"uri": config.data.eval_uri, "split": config.data.eval_split})
        eval_ds = build_dataset(eval_cfg, tokenizer=tokenizer, seed=config.seed, sft=sft)
    return train, eval_ds
