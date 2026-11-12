"""Data sources: local files, ``hf://`` Hub datasets and ``s3://`` / fsspec objects → :class:`Sample`.

Public surface::

    open_source(data_config, *, split=None, tokenizer=None, seed=0) -> DataSource
    sniff_format(columns, sample_rows) -> {format, prompt_field, answer_field, messages_field, image_fields}
    browse.browse_uri(uri) / browse.preview_uri(uri, split=, hf_config_name=, limit=)
    register(registry)   # data_source: local | hf | s3
"""

from __future__ import annotations

from typing import Any

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.core.interfaces import DataSource
from monorl.data.formats import SNIFF_ROWS, sniff_format
from monorl.data.rows import open_rows
from monorl.data.sources import Layout, MaterializedSource, SampleMapper, StreamingSource
from monorl.data.uri import ParsedURI, parse_uri

__all__ = ["open_source", "sniff_format", "register", "source_kind"]

_FORBIDDEN_EXTRA = ("token", "storage_options", "key", "secret", "aws_access_key_id", "aws_secret_access_key")


def source_kind(parsed: ParsedURI) -> str:
    """Registry name for a URI: ``local`` | ``hf`` | ``s3`` (any remote fsspec scheme)."""
    return {"local": "local", "hf": "hf"}.get(parsed.kind, "s3")


def open_source(
    data_config: DataConfig, *, split: str | None = None, tokenizer: Any = None, seed: int = 0
) -> DataSource:
    """Resolve ``data_config.uri`` into a :class:`DataSource` of :class:`Sample` objects.

    * scheme → local / hf / s3 (fsspec) row source; ``streaming`` keeps hf/s3/local lazy
    * layout from ``format``/field overrides, else :func:`sniff_format` on the first rows
    * ``prompt_template`` (``str.format_map`` over the row + ``{prompt}``), ``system_prompt``
    * ``image_field`` column(s) → ``{"type": "image", "image": <path|uri>}`` parts on the user
      turn (``'<image>'`` in ``prompt_template`` positions them; default before the text); chat
      rows with OpenAI-style parts are normalised the same way (:mod:`monorl.data.media`)
    * ``#### answer`` extraction into ``Sample.reference`` (raw under ``metadata["answer_raw"]``);
      the SFT target under ``metadata["response"]`` (``response_field`` → ``answer_field`` →
      trailing assistant turn, else ``None``)
    * ``max_prompt_tokens`` filter when ``tokenizer`` is given (chat template applied to the text
      parts only; media placeholder tokens are not counted)
    * deterministic ``shuffle(seed)`` then ``limit``
    """
    parsed = parse_uri(data_config.uri)
    for key in _FORBIDDEN_EXTRA:
        if key in data_config.extra:
            raise DataError(f"data.extra[{key!r}] is not allowed: credentials come from the environment only")
    rows = open_rows(
        parsed,
        split=split or data_config.split,
        config_name=data_config.hf_config_name,
        streaming=data_config.streaming,
        extra=data_config.extra or None,
    )
    layout = Layout.resolve(data_config, rows.columns(), rows.head(SNIFF_ROWS))
    mapper = SampleMapper.from_config(data_config, layout, tokenizer, rows=rows)
    cls = StreamingSource if data_config.streaming else MaterializedSource
    return cls(
        source_kind(parsed),
        rows,
        mapper,
        seed=seed,
        shuffle=data_config.shuffle,
        limit=data_config.limit,
    )


def _factory_for(kind: str):
    def factory(data_config: DataConfig, **kw: Any) -> DataSource:
        actual = source_kind(parse_uri(data_config.uri))
        if actual != kind:
            raise DataError(f"data_source {kind!r} cannot open {data_config.uri!r} (it is a {actual!r} uri)")
        return open_source(data_config, **kw)

    factory.__name__ = f"open_{kind}_source"
    return factory


def register(registry) -> None:
    for kind in ("local", "hf", "s3"):
        registry.register("data_source", kind, _factory_for(kind))
