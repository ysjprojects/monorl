"""Row → :class:`Sample` mapping and the :class:`DataSource` implementations.

``open_source`` (in :mod:`monorl.data`) resolves a :class:`RowSource`, builds a
:class:`SampleMapper` from the config + sniffed layout, then wraps both in either
:class:`MaterializedSource` (list in memory, ``len()`` works, re-iterable) or
:class:`StreamingSource` (lazy, ``len()`` raises :class:`DataError` unless the count is known).

Pipeline per row: layout → messages (+ ``system_prompt``, ``prompt_template``) → image parts
(``image_field``) → tool schemas (``tools_field``) → reference (``#### answer`` extraction) + SFT
``metadata["response"]`` → ``max_prompt_tokens`` filter (text tokens only) → shuffle/limit.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.core.types import ContentPart, Message, Sample, message_text
from monorl.data import formats
from monorl.data.media import MediaResolver, normalize_content_parts, place_images
from monorl.data.rows import RowRef, RowSource

GSM8K_SEPARATOR = "####"
_SHAREGPT_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}
_MESSAGE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name", "weight")
TOOLS_COLUMN = "tools"
"""Column a chat dataset keeps its tool schemas in when ``data.tools_field`` is not set."""
_JSON_FORMATS = ("jsonl", "json")


class _RowFormatMap(dict):
    """``str.format_map`` mapping that names the missing column instead of a bare KeyError."""

    def __missing__(self, key: str) -> Any:
        raise DataError(f"prompt_template references {{{key}}} but the row has columns {sorted(self)}")


def normalize_messages(value: Any, *, where: str, resolver: MediaResolver | None = None) -> list[Message]:
    """Accept OpenAI (``role``/``content``) or ShareGPT (``from``/``value``) lists, or their JSON
    string encoding (CSV/parquet string columns). List-shaped ``content`` (OpenAI image parts,
    HF image structs) is normalised to monorl parts through ``resolver``."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise DataError(f"{where}: messages column is a string that is not JSON") from e
    if not isinstance(value, list) or not value:
        raise DataError(f"{where}: messages column must be a non-empty list of messages")
    resolver = resolver or MediaResolver()
    out: list[Message] = []
    for i, m in enumerate(value):
        if not isinstance(m, dict):
            raise DataError(f"{where}: message {i} is {type(m).__name__}, expected an object")
        if "from" in m and "role" not in m:
            role = _SHAREGPT_ROLES.get(str(m["from"]).lower())
            if role is None:
                raise DataError(f"{where}: message {i} has unknown ShareGPT speaker {m['from']!r}")
            content = m.get("value")
        else:
            role = m.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                raise DataError(f"{where}: message {i} has unsupported role {role!r}")
            content = m.get("content")
        msg: Message = {
            "role": role,
            "content": normalize_content_parts(content, resolver, where=f"{where}.messages[{i}]"),
        }
        if "role" in m:
            for key in _MESSAGE_KEYS[2:]:
                if m.get(key) is not None:
                    msg[key] = m[key]  # type: ignore[literal-required]
        out.append(msg)
    return out


def extract_reference(answer: Any) -> tuple[Any, bool]:
    """GSM8K-style ``...#### 42`` → ``"42"``. Returns ``(reference, extracted)``."""
    if isinstance(answer, str) and GSM8K_SEPARATOR in answer:
        tail = answer.rsplit(GSM8K_SEPARATOR, 1)[1].strip()
        if tail:
            return re.sub(r"(?<=\d),(?=\d{3})", "", tail), True
    return answer, False


def normalize_tools(
    value: Any, *, where: str, strict: bool, drop_nulls: bool = False
) -> list[dict[str, Any]] | None:
    """Tool schemas of a row: a list of objects, or its JSON string (how CSV and many Hub datasets
    store them). ``None`` for a missing or empty value. Anything else is a :class:`DataError` when
    the column was named in the config (``strict``) and is left alone (``None``) when the column is
    only a chat dataset's conventional ``tools``. ``drop_nulls`` removes ``null``-valued keys at
    every depth: columnar stores fill struct keys a row does not have with ``null``."""
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            if strict:
                raise DataError(f"{where}: tools column is a string that is not JSON") from e
            return None
    if value is None or value == "":
        return None
    if not isinstance(value, (list, tuple)) or not all(isinstance(t, dict) for t in value):
        if strict:
            raise DataError(f"{where}: tools must be a list of tool schema objects")
        return None
    if not value:
        return None
    return [_without_nulls(t) if drop_nulls else dict(t) for t in value]


def _without_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_without_nulls(v) for v in value]
    return value


@dataclass(slots=True)
class Layout:
    format: str  # chat | prompt_answer | text
    prompt_field: str | None = None
    answer_field: str | None = None
    messages_field: str | None = None
    response_field: str | None = None
    image_fields: list[str] = field(default_factory=list)
    """Columns attached to the user turn as image parts (``DataConfig.image_field``; explicit opt-in,
    ``sniff_format`` only reports candidates)."""
    tools_field: str | None = None
    """Column holding the row's tool schemas (``DataConfig.tools_field``). ``None``: a chat row's
    :data:`TOOLS_COLUMN` is used when it holds schemas."""

    @classmethod
    def resolve(cls, config: DataConfig, columns: dict[str, str], sample_rows: list[dict]) -> Layout:
        """Explicit config fields win over sniffing; an explicit ``messages_field`` implies chat and an
        explicit ``prompt_field`` (without ``messages_field``) implies prompt_answer/text."""
        sniffed = formats.sniff_format(columns, sample_rows)
        fmt = config.format
        if fmt == "auto":
            if config.messages_field and not config.prompt_field:
                fmt = "chat"
            elif config.prompt_field and not config.messages_field:
                fmt = "prompt_answer" if (config.answer_field or sniffed["answer_field"]) else "text"
            else:
                fmt = sniffed["format"]
        image_fields = config.image_field if isinstance(config.image_field, list) else [config.image_field]
        layout = cls(
            format=fmt,
            prompt_field=config.prompt_field or sniffed["prompt_field"],
            answer_field=config.answer_field or sniffed["answer_field"],
            messages_field=config.messages_field or sniffed["messages_field"],
            response_field=config.response_field,
            tools_field=config.tools_field,
            image_fields=[f for f in image_fields if f],
        )
        if layout.format not in ("chat", "prompt_answer", "text"):
            raise DataError(f"could not infer dataset format from columns {sorted(columns)}; set data.format")
        if layout.format == "chat" and not layout.messages_field:
            raise DataError(f"format=chat needs messages_field; columns: {sorted(columns)}")
        if layout.format in ("prompt_answer", "text") and not layout.prompt_field:
            raise DataError(f"format={layout.format} needs prompt_field; columns: {sorted(columns)}")
        if layout.format == "prompt_answer" and not layout.answer_field:
            raise DataError(f"format=prompt_answer needs answer_field; columns: {sorted(columns)}")
        for name in (
            layout.prompt_field,
            layout.answer_field,
            layout.messages_field,
            layout.response_field,
            layout.tools_field,
            *layout.image_fields,
        ):
            if name and columns and name not in columns:
                raise DataError(f"column {name!r} not found; available: {sorted(columns)}")
        return layout

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "prompt_field": self.prompt_field,
            "answer_field": self.answer_field,
            "messages_field": self.messages_field,
            "response_field": self.response_field,
            "tools_field": self.tools_field,
            "image_fields": list(self.image_fields),
        }


@dataclass(slots=True)
class SampleMapper:
    layout: Layout
    system_prompt: str | None = None
    prompt_template: str | None = None
    tokenizer: Any = None
    max_prompt_tokens: int | None = None
    media_base: Callable[[str], str | None] | None = None
    """``row_id -> base location`` for relative image paths (``RowSource.media_base``)."""
    null_filled: bool = False
    """Rows come from a columnar store (parquet, arrow, Hub datasets), where a struct column holds
    ``null`` for every key a row does not have: such keys are dropped from tool schemas."""
    dropped: int = 0
    """Rows dropped by the token filter so far (diagnostic)."""
    _resolvers: dict[str | None, MediaResolver] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls, config: DataConfig, layout: Layout, tokenizer: Any = None, rows: RowSource | None = None
    ) -> SampleMapper:
        return cls(
            layout=layout,
            system_prompt=config.system_prompt,
            prompt_template=config.prompt_template,
            tokenizer=tokenizer,
            max_prompt_tokens=config.max_prompt_tokens if tokenizer is not None else None,
            media_base=rows.media_base if rows is not None else None,
            null_filled=rows is not None and getattr(rows, "format", None) not in _JSON_FORMATS,
        )

    def resolver_for(self, row_id: str) -> MediaResolver:
        base = self.media_base(row_id) if self.media_base is not None else None
        resolver = self._resolvers.get(base)
        if resolver is None:
            resolver = self._resolvers[base] = MediaResolver(base)
        return resolver

    # -- row → sample
    def to_sample(self, row_id: str, row: Mapping[str, Any]) -> Sample:
        lay = self.layout
        metadata: dict[str, Any] = {}
        resolver = self.resolver_for(row_id)
        if lay.format == "chat":
            consumed = lay.messages_field
            messages = normalize_messages(row[lay.messages_field], where=row_id, resolver=resolver)
        else:
            consumed = lay.prompt_field
            prompt = row[lay.prompt_field]
            if formats.looks_like_chat(prompt):
                messages = normalize_messages(prompt, where=row_id, resolver=resolver)
            else:
                messages = [{"role": "user", "content": "" if prompt is None else str(prompt)}]

        reference: Any = None
        response: str | None = None
        if lay.answer_field and lay.answer_field in row:
            raw = row[lay.answer_field]
            reference, extracted = extract_reference(raw)
            if extracted:
                metadata["answer_raw"] = raw
            if lay.format != "chat" and raw is not None:
                response = str(raw)
        if messages and messages[-1]["role"] == "assistant":
            trailing = messages.pop()
            metadata["assistant_turn"] = trailing
            if reference is None:
                reference = message_text(trailing)
            if response is None:
                response = message_text(trailing)
        if lay.response_field and row.get(lay.response_field) is not None:
            response = str(row[lay.response_field])
        metadata["response"] = response
        if not messages:
            raise DataError(f"{row_id}: no prompt messages left after removing the assistant turn")

        images = [
            part
            for name in lay.image_fields
            for part in resolver.image_parts(row.get(name), where=f"{row_id}.{name}")
        ]
        if self.prompt_template is not None or images:
            self._apply_user_turn(messages, row, images, row_id)

        if self.system_prompt is not None:
            if messages[0]["role"] == "system":
                messages[0] = {"role": "system", "content": self.system_prompt}
            else:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

        skip = {consumed, *lay.image_fields}
        tools_field = lay.tools_field or (TOOLS_COLUMN if lay.format == "chat" else None)
        if tools_field is not None:
            tools = normalize_tools(
                row.get(tools_field),
                where=f"{row_id}.{tools_field}",
                strict=lay.tools_field is not None,
                drop_nulls=self.null_filled,
            )
            if tools is not None:
                metadata["tools"] = tools
                skip.add(tools_field)
        metadata["row"] = {k: v for k, v in row.items() if k not in skip}
        sample_id = str(row["id"]) if row.get("id") not in (None, "") else row_id
        return Sample(id=sample_id, messages=messages, reference=reference, metadata=metadata)

    def _apply_user_turn(
        self, messages: list[Message], row: Mapping[str, Any], images: list, row_id: str
    ) -> None:
        """Render ``prompt_template`` on the last user turn and attach ``images`` to it."""
        index = next((i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "user"), None)
        if index is None:
            if images:
                raise DataError(f"{row_id}: image_field set but the row has no user turn to attach images to")
            return
        if index != len(messages) - 1 and self.prompt_template is not None and not images:
            return  # template applies to a trailing user turn only (unchanged behaviour)
        current = messages[index]
        content = current["content"]
        existing: list[ContentPart] | None = content if isinstance(content, list) else None
        text = message_text(current)
        if self.prompt_template is not None and index == len(messages) - 1:
            text = self._render(row, text)
        if not images and existing is None:
            messages[index] = {**current, "content": text}
            return
        messages[index] = {**current, "content": place_images(text, images, existing)}

    def _render(self, row: Mapping[str, Any], prompt_text: str) -> str:
        scope = _RowFormatMap(row)
        scope.setdefault("prompt", prompt_text)
        try:
            return self.prompt_template.format_map(scope)  # type: ignore[union-attr]
        except (IndexError, ValueError) as e:
            raise DataError(f"invalid prompt_template {self.prompt_template!r}: {e}") from e

    # -- filtering
    def prompt_tokens(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> int:
        """Text tokens of the prompt, rendered with the row's ``tools`` when it has any (media
        parts are dropped: image placeholder tokens are model-specific and only known to the
        processor, so ``max_prompt_tokens`` bounds text)."""
        tok = self.tokenizer
        text_only = [{**m, "content": message_text(m)} for m in messages]
        if getattr(tok, "chat_template", None):
            kwargs = {"tools": tools} if tools else {}
            text = tok.apply_chat_template(text_only, tokenize=False, add_generation_prompt=True, **kwargs)
        else:
            text = "\n".join(m["content"] for m in text_only)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def keep(self, sample: Sample) -> bool:
        if self.max_prompt_tokens is None:
            return True
        n = self.prompt_tokens(sample.messages, sample.metadata.get("tools"))
        sample.metadata["prompt_tokens"] = n
        if n > self.max_prompt_tokens:
            self.dropped += 1
            return False
        return True

    def map_rows(self, rows: Iterator[RowRef], limit: int | None) -> Iterator[Sample]:
        samples = (self.to_sample(rid, row) for rid, row in rows)
        kept = (s for s in samples if self.keep(s))
        return itertools.islice(kept, limit) if limit is not None else kept


# -- data sources -----------------------------------------------------------------------------


class _BaseSource:
    def __init__(
        self, name: str, rows: RowSource, mapper: SampleMapper, *, seed: int, shuffle: bool, limit: int | None
    ) -> None:
        self.name = name
        self.uri = rows.uri
        self.rows = rows
        self.mapper = mapper
        self.seed = seed
        self.shuffle = shuffle
        self.limit = limit

    @property
    def layout(self) -> Layout:
        return self.mapper.layout

    def schema(self) -> dict[str, str]:
        return dict(self.rows.columns())

    def _mapped(self) -> Iterator[Sample]:
        return self.mapper.map_rows(
            self.rows.iter_rows(shuffle_seed=self.seed if self.shuffle else None), self.limit
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, uri={self.uri!r}, format={self.layout.format!r})"


class MaterializedSource(_BaseSource):
    """All samples held in memory: ``len()`` is exact (after the token filter) and iteration is
    repeatable."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.samples: list[Sample] = list(self._mapped())

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


class UnknownLengthError(DataError, TypeError):
    """``len()`` of a streaming source whose size is unknown. Also a ``TypeError`` so that
    ``list(source)`` / ``operator.length_hint`` fall back to plain iteration instead of failing."""


class StreamingSource(_BaseSource):
    """Lazy source: each ``iter()`` re-opens the stream. ``len()`` is known only when the backend
    reports a row count and no filtering/limit can change it."""

    def __len__(self) -> int:
        n = self.rows.known_len()
        if n is None:
            raise UnknownLengthError(f"{self.uri}: length unknown for a streaming source")
        if self.mapper.max_prompt_tokens is not None:
            raise UnknownLengthError(
                f"{self.uri}: length unknown (streaming with max_prompt_tokens filtering)"
            )
        return min(n, self.limit) if self.limit is not None else n

    def __iter__(self) -> Iterator[Sample]:
        return self._mapped()
