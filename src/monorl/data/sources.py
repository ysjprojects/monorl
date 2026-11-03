"""Row → :class:`Sample` mapping and the :class:`DataSource` implementations.

``open_source`` (in :mod:`monorl.data`) resolves a :class:`RowSource`, builds a
:class:`SampleMapper` from the config + sniffed layout, then wraps both in either
:class:`MaterializedSource` (list in memory, ``len()`` works, re-iterable) or
:class:`StreamingSource` (lazy, ``len()`` raises :class:`DataError` unless the count is known).

Pipeline per row: layout → messages (+ ``system_prompt``, ``prompt_template``) → reference
(``#### answer`` extraction) → ``max_prompt_tokens`` filter → shuffle/limit.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.core.types import Message, Sample
from monorl.data import formats
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
_MESSAGE_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")


class _RowFormatMap(dict):
    """``str.format_map`` mapping that names the missing column instead of a bare KeyError."""

    def __missing__(self, key: str) -> Any:
        raise DataError(f"prompt_template references {{{key}}} but the row has columns {sorted(self)}")


def normalize_messages(value: Any, *, where: str) -> list[Message]:
    """Accept OpenAI (``role``/``content``) or ShareGPT (``from``/``value``) lists, or their JSON
    string encoding (CSV/parquet string columns)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise DataError(f"{where}: messages column is a string that is not JSON") from e
    if not isinstance(value, list) or not value:
        raise DataError(f"{where}: messages column must be a non-empty list of messages")
    out: list[Message] = []
    for i, m in enumerate(value):
        if not isinstance(m, dict):
            raise DataError(f"{where}: message {i} is {type(m).__name__}, expected an object")
        if "from" in m and "role" not in m:
            role = _SHAREGPT_ROLES.get(str(m["from"]).lower())
            if role is None:
                raise DataError(f"{where}: message {i} has unknown ShareGPT speaker {m['from']!r}")
            msg: Message = {"role": role, "content": m.get("value") or ""}
        else:
            role = m.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                raise DataError(f"{where}: message {i} has unsupported role {role!r}")
            msg = {"role": role, "content": m.get("content") if m.get("content") is not None else ""}
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


@dataclass(slots=True)
class Layout:
    format: str  # chat | prompt_answer | text
    prompt_field: str | None = None
    answer_field: str | None = None
    messages_field: str | None = None

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
        layout = cls(
            format=fmt,
            prompt_field=config.prompt_field or sniffed["prompt_field"],
            answer_field=config.answer_field or sniffed["answer_field"],
            messages_field=config.messages_field or sniffed["messages_field"],
        )
        if layout.format not in ("chat", "prompt_answer", "text"):
            raise DataError(f"could not infer dataset format from columns {sorted(columns)}; set data.format")
        if layout.format == "chat" and not layout.messages_field:
            raise DataError(f"format=chat needs messages_field; columns: {sorted(columns)}")
        if layout.format in ("prompt_answer", "text") and not layout.prompt_field:
            raise DataError(f"format={layout.format} needs prompt_field; columns: {sorted(columns)}")
        if layout.format == "prompt_answer" and not layout.answer_field:
            raise DataError(f"format=prompt_answer needs answer_field; columns: {sorted(columns)}")
        for name in (layout.prompt_field, layout.answer_field, layout.messages_field):
            if name and columns and name not in columns:
                raise DataError(f"column {name!r} not found; available: {sorted(columns)}")
        return layout

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "prompt_field": self.prompt_field,
            "answer_field": self.answer_field,
            "messages_field": self.messages_field,
        }


@dataclass(slots=True)
class SampleMapper:
    layout: Layout
    system_prompt: str | None = None
    prompt_template: str | None = None
    tokenizer: Any = None
    max_prompt_tokens: int | None = None
    dropped: int = 0
    """Rows dropped by the token filter so far (diagnostic)."""

    @classmethod
    def from_config(cls, config: DataConfig, layout: Layout, tokenizer: Any = None) -> SampleMapper:
        return cls(
            layout=layout,
            system_prompt=config.system_prompt,
            prompt_template=config.prompt_template,
            tokenizer=tokenizer,
            max_prompt_tokens=config.max_prompt_tokens if tokenizer is not None else None,
        )

    # -- row → sample
    def to_sample(self, row_id: str, row: Mapping[str, Any]) -> Sample:
        lay = self.layout
        metadata: dict[str, Any] = {}
        if lay.format == "chat":
            consumed = lay.messages_field
            messages = normalize_messages(row[lay.messages_field], where=row_id)
        else:
            consumed = lay.prompt_field
            prompt = row[lay.prompt_field]
            if formats.looks_like_chat(prompt):
                messages = normalize_messages(prompt, where=row_id)
            else:
                messages = [{"role": "user", "content": "" if prompt is None else str(prompt)}]

        reference: Any = None
        if lay.answer_field and lay.answer_field in row:
            raw = row[lay.answer_field]
            reference, extracted = extract_reference(raw)
            if extracted:
                metadata["answer_raw"] = raw
        if messages and messages[-1]["role"] == "assistant":
            trailing = messages.pop()
            metadata["assistant_turn"] = trailing
            if reference is None:
                reference = trailing.get("content")
        if not messages:
            raise DataError(f"{row_id}: no prompt messages left after removing the assistant turn")
        if self.prompt_template is not None and messages[-1]["role"] == "user":
            messages[-1] = {**messages[-1], "content": self._render(row, messages[-1]["content"])}

        if self.system_prompt is not None:
            if messages[0]["role"] == "system":
                messages[0] = {"role": "system", "content": self.system_prompt}
            else:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

        metadata["row"] = {k: v for k, v in row.items() if k != consumed}
        sample_id = str(row["id"]) if row.get("id") not in (None, "") else row_id
        return Sample(id=sample_id, messages=messages, reference=reference, metadata=metadata)

    def _render(self, row: Mapping[str, Any], prompt_text: str) -> str:
        scope = _RowFormatMap(row)
        scope.setdefault("prompt", prompt_text)
        try:
            return self.prompt_template.format_map(scope)  # type: ignore[union-attr]
        except (IndexError, ValueError) as e:
            raise DataError(f"invalid prompt_template {self.prompt_template!r}: {e}") from e

    # -- filtering
    def prompt_tokens(self, messages: list[Message]) -> int:
        tok = self.tokenizer
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n".join(str(m.get("content", "")) for m in messages)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def keep(self, sample: Sample) -> bool:
        if self.max_prompt_tokens is None:
            return True
        n = self.prompt_tokens(sample.messages)
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
    def __init__(self, name: str, rows: RowSource, mapper: SampleMapper, *, seed: int, shuffle: bool, limit: int | None) -> None:
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
        return self.mapper.map_rows(self.rows.iter_rows(shuffle_seed=self.seed if self.shuffle else None), self.limit)

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
            raise UnknownLengthError(f"{self.uri}: length unknown (streaming with max_prompt_tokens filtering)")
        return min(n, self.limit) if self.limit is not None else n

    def __iter__(self) -> Iterator[Sample]:
        return self._mapped()
