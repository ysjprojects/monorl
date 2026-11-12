"""monorl :class:`~monorl.core.interfaces.DataSource` → ``datasets.Dataset`` for TRL trainers.

GRPO columns: ``prompt`` (chat messages), ``sample_id``, ``reference`` (string form for TRL-native
reward functions) and ``sample_json`` (the full :class:`~monorl.core.types.Sample`, used to rebuild
exact samples on the reward side). SFT columns (``algorithm.name: sft``): conversational
``prompt`` + ``completion`` (one assistant turn) so ``SFTTrainer`` masks the prompt with
``completion_only_loss``; with ``algorithm.sft_turns: all`` the whole conversation as ``messages``
(tool results keep ``name`` / ``tool_call_id``), which ``assistant_only_loss`` masks down to its
assistant turns. A dataset whose rows carry tool schemas gets a ``tools`` column (the JSON string
``SFTTrainer`` decodes and hands to the chat template; Arrow could not hold differing schemas).

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
MESSAGES_COLUMN = "messages"
TOOLS_COLUMN = "tools"


def text_messages(messages: list[Message]) -> list[dict[str, str]]:
    """``role``/``content`` only, media dropped (one Arrow schema for text runs)."""
    return [{"role": str(m.get("role", "user")), "content": message_text(m)} for m in messages]


def chat_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """:func:`text_messages` that keeps what ties a tool result to its call (``name``,
    ``tool_call_id``)."""
    out: list[dict[str, Any]] = list(text_messages(messages))
    for message, row in zip(messages, out, strict=True):
        for key in ("name", "tool_call_id"):
            if message.get(key) is not None:
                row[key] = str(message[key])  # type: ignore[literal-required]
    return out


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


def sft_sample_to_row(
    sample: Sample, *, response_field: str | None, multimodal: bool = False, turns: str = "last"
) -> dict[str, Any]:
    """``prompt``/``completion`` conversational row (``turns="last"``) or the whole conversation
    as ``messages`` (``"all"``); the response comes from
    :func:`monorl.orchestrator.sft.sft_response` (``data.response_field`` → ``answer_field`` →
    trailing assistant turn). ``tools`` (JSON) is set when the sample has tool schemas."""
    from monorl.orchestrator.sft import require_every_turn_trained, sft_messages, sft_response, sft_tools

    row: dict[str, Any]
    if turns == "all":
        messages = sft_messages(sample, response_field=response_field)
        require_every_turn_trained(messages, backend="trl", sample_id=sample.id)
        row = {MESSAGES_COLUMN: structured_messages(messages) if multimodal else chat_messages(messages)}
    else:
        response = sft_response(sample, response_field=response_field)
        completion: list[dict[str, Any]] = [{"role": "assistant", "content": response}]
        if multimodal:
            completion = structured_messages(completion)  # type: ignore[arg-type]
        prompt = structured_messages(sample.messages) if multimodal else text_messages(sample.messages)
        row = {PROMPT_COLUMN: prompt, COMPLETION_COLUMN: completion}
    tools = sft_tools(sample)
    if tools:
        row[TOOLS_COLUMN] = json.dumps(tools)
    if multimodal:
        row[IMAGES_COLUMN] = sample_images(sample)
    return row


def sample_from_row(row: dict[str, Any]) -> Sample:
    """Inverse of :func:`sample_to_row` (works on TRL's per-example input dicts)."""
    raw = row.get(SAMPLE_COLUMN)
    if raw is None:
        return Sample(
            id=str(row.get("sample_id", "")),
            messages=list(row[PROMPT_COLUMN]),
            reference=row.get("reference"),
        )
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
    sft_turns: str = "last",
) -> Any:
    """Materialise ``data_config`` through :func:`monorl.data.open_source` into a ``datasets.Dataset``."""
    from monorl.data import open_source

    samples = list(open_source(data_config, split=split, tokenizer=tokenizer, seed=seed))
    if not samples:
        raise DataError(f"dataset {data_config.uri!r} yielded no samples")
    multimodal = is_multimodal(data_config, samples)
    if sft:
        rows = [
            sft_sample_to_row(
                s, response_field=data_config.response_field, multimodal=multimodal, turns=sft_turns
            )
            for s in samples
        ]
        if any(TOOLS_COLUMN in row for row in rows):  # one schema: Arrow takes the columns of the first row
            for row in rows:
                row.setdefault(TOOLS_COLUMN, None)
    else:
        rows = [sample_to_row(s, multimodal=multimodal) for s in samples]
    return _to_dataset(rows, multimodal=multimodal)


def build_train_and_eval(config: RunConfig, *, tokenizer: Any = None) -> tuple[Any, Any | None]:
    sft, turns = config.is_sft, config.algorithm.sft_turns
    train = build_dataset(config.data, tokenizer=tokenizer, seed=config.seed, sft=sft, sft_turns=turns)
    eval_ds = None
    if config.data.eval_uri:
        eval_cfg = config.data.model_copy(
            update={"uri": config.data.eval_uri, "split": config.data.eval_split}
        )
        eval_ds = build_dataset(eval_cfg, tokenizer=tokenizer, seed=config.seed, sft=sft, sft_turns=turns)
    return train, eval_ds
