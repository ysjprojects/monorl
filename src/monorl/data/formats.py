"""File-format readers, column dtype inference and layout sniffing.

Readers are streaming generators over fsspec file objects so previews and ``streaming=True``
sources never read more than they need (``.json`` is the exception: a JSON array has to be
parsed whole).
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterator
from typing import Any

from monorl.core.errors import DataError

Format = str  # "jsonl" | "json" | "csv" | "parquet" | "arrow"

FORMAT_EXTENSIONS: dict[str, Format] = {
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".arrow": "arrow",
}
COMPRESSION_SUFFIXES: tuple[str, ...] = (".gz", ".bz2", ".xz", ".zst")
# When a directory mixes formats (e.g. ``save_to_disk`` dirs with ``*.arrow`` + ``state.json``)
# only the highest-priority format is read.
FORMAT_PRIORITY: tuple[Format, ...] = ("parquet", "arrow", "jsonl", "csv", "json")

ARROW_BATCH_ROWS = 1024
SNIFF_ROWS = 50

MESSAGES_COLUMNS: tuple[str, ...] = ("messages", "conversations", "conversation")
PROMPT_COLUMNS: tuple[str, ...] = ("prompt", "question", "instruction", "input", "problem", "query")
ANSWER_COLUMNS: tuple[str, ...] = (
    "answer",
    "solution",
    "response",
    "output",
    "completion",
    "target",
    "label",
    "final_answer",
)
TEXT_COLUMNS: tuple[str, ...] = ("text",)


def format_of(path: str) -> Format | None:
    """Format implied by the file extension (compression suffix stripped), or ``None``."""
    name = os.path.basename(path).lower()
    for suffix in COMPRESSION_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    _, ext = os.path.splitext(name)
    return FORMAT_EXTENSIONS.get(ext)


def is_data_file(path: str) -> bool:
    base = os.path.basename(path)
    return format_of(path) is not None and not base.startswith((".", "_"))


def select_files(paths: list[str]) -> list[str]:
    """Keep supported data files; if several formats are present keep only the best one."""
    candidates = sorted(p for p in paths if is_data_file(p))
    by_format: dict[Format, list[str]] = {}
    for p in candidates:
        by_format.setdefault(format_of(p) or "", []).append(p)
    for fmt in FORMAT_PRIORITY:
        if fmt in by_format:
            return by_format[fmt]
    return []


# -- readers ----------------------------------------------------------------------------------


def iter_file_rows(fs, path: str, fmt: Format | None = None) -> Iterator[dict[str, Any]]:
    fmt = fmt or format_of(path)
    if fmt is None:
        raise DataError(f"unsupported file type: {path} (expected {sorted(FORMAT_EXTENSIONS)})")
    reader = _READERS[fmt]
    yield from reader(fs, path)


def _text_handle(fs, path: str) -> io.TextIOBase:
    raw = fs.open(path, "rb", compression="infer")
    return io.TextIOWrapper(raw, encoding="utf-8", newline="")


def _read_jsonl(fs, path: str) -> Iterator[dict[str, Any]]:
    with _text_handle(fs, path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DataError(f"{path}:{lineno}: invalid JSON ({e.msg})") from e
            yield _as_row(row, f"{path}:{lineno}")


def _read_json(fs, path: str) -> Iterator[dict[str, Any]]:
    with _text_handle(fs, path) as fh:
        try:
            payload = json.load(fh)
        except json.JSONDecodeError as e:
            raise DataError(f"{path}: invalid JSON ({e.msg})") from e
    rows = _unwrap_json(payload, path)
    for i, row in enumerate(rows):
        yield _as_row(row, f"{path}[{i}]")


def _unwrap_json(payload: Any, path: str) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # {"data": [...]} / {"rows": [...]} / {"train": [...]} / single-key wrappers
        lists = {k: v for k, v in payload.items() if isinstance(v, list) and (not v or isinstance(v[0], dict))}
        for key in ("data", "rows", "examples", "train"):
            if key in lists:
                return lists[key]
        if len(lists) == 1:
            return next(iter(lists.values()))
        # columnar {"col": [..], "col2": [..]}
        if payload and all(isinstance(v, list) for v in payload.values()):
            lengths = {len(v) for v in payload.values()}
            if len(lengths) == 1:
                n = lengths.pop()
                return [{k: payload[k][i] for k in payload} for i in range(n)]
        if not any(isinstance(v, list) for v in payload.values()):
            return [payload]
    raise DataError(f"{path}: expected a JSON array of objects (or an object wrapping one)")


def _as_row(row: Any, where: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise DataError(f"{where}: expected a JSON object per row, got {type(row).__name__}")
    return row


def _read_csv(fs, path: str) -> Iterator[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.csv as pcsv

    delimiter = "\t" if format_of(path) == "csv" and path.lower().split(".gz")[0].endswith(".tsv") else ","
    with fs.open(path, "rb", compression="infer") as fh:
        try:
            reader = pcsv.open_csv(
                fh,
                read_options=pcsv.ReadOptions(block_size=1 << 20),
                parse_options=pcsv.ParseOptions(delimiter=delimiter),
            )
            for batch in reader:
                yield from batch.to_pylist()
        except pa.ArrowInvalid as e:
            raise DataError(f"{path}: {e}") from e


def _read_parquet(fs, path: str) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    with fs.open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(batch_size=ARROW_BATCH_ROWS):
            yield from batch.to_pylist()


def _open_arrow(fh):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    try:
        return ipc.open_stream(fh)
    except pa.ArrowInvalid:
        fh.seek(0)
        return ipc.open_file(fh)


def _read_arrow(fs, path: str) -> Iterator[dict[str, Any]]:
    import pyarrow.ipc as ipc

    with fs.open(path, "rb") as fh:
        reader = _open_arrow(fh)
        if isinstance(reader, ipc.RecordBatchFileReader):
            for i in range(reader.num_record_batches):
                yield from reader.get_batch(i).to_pylist()
        else:
            for batch in reader:
                yield from batch.to_pylist()


_READERS = {
    "jsonl": _read_jsonl,
    "json": _read_json,
    "csv": _read_csv,
    "parquet": _read_parquet,
    "arrow": _read_arrow,
}


# -- schema -----------------------------------------------------------------------------------


def arrow_schema_of(fs, path: str, fmt: Format | None = None) -> dict[str, str] | None:
    """Column dtypes from file metadata when the format carries a schema (parquet/arrow/csv)."""
    fmt = fmt or format_of(path)
    if fmt == "parquet":
        import pyarrow.parquet as pq

        with fs.open(path, "rb") as fh:
            return {f.name: str(f.type) for f in pq.ParquetFile(fh).schema_arrow}
    if fmt == "arrow":
        with fs.open(path, "rb") as fh:
            return {f.name: str(f.type) for f in _open_arrow(fh).schema}
    return None


def parquet_num_rows(fs, path: str) -> int:
    import pyarrow.parquet as pq

    with fs.open(path, "rb") as fh:
        return pq.ParquetFile(fh).metadata.num_rows


def dtype_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bytes):
        return "binary"
    if isinstance(value, (list, tuple)):
        inner = {dtype_of(v) for v in value[:8]} - {"null"}
        return f"list<{inner.pop()}>" if len(inner) == 1 else "list"
    if isinstance(value, dict):
        return "struct"
    return type(value).__name__


def infer_columns(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Merge python value types over sample rows into ``{column: dtype}`` (insertion ordered)."""
    columns: dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            dt = dtype_of(value)
            prev = columns.get(key)
            if prev is None or prev == "null":
                columns[key] = dt
            elif dt != "null" and dt != prev:
                if {prev, dt} == {"int64", "double"}:
                    columns[key] = "double"
                elif prev.startswith("list") and dt.startswith("list"):
                    columns[key] = "list"
                else:
                    columns[key] = "mixed"
    return columns


def features_to_columns(features) -> dict[str, str]:
    """``datasets.Features`` -> ``{column: dtype}`` using the arrow rendering."""
    try:
        return {f.name: str(f.type) for f in features.arrow_schema}
    except Exception:  # exotic features (Audio/Image) may not render; fall back to repr
        return {k: getattr(v, "dtype", None) or type(v).__name__ for k, v in features.items()}


# -- sniffing ---------------------------------------------------------------------------------


def _find(columns: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def looks_like_chat(value: Any) -> bool:
    if isinstance(value, str) and value.lstrip().startswith("["):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(m, dict) and (("role" in m and "content" in m) or ("from" in m and "value" in m)) for m in value)
    )


def _column_holds_chat(column: str, rows: list[dict[str, Any]]) -> bool:
    values = [r[column] for r in rows if r.get(column) is not None]
    return bool(values) and all(looks_like_chat(v) for v in values)


def sniff_format(columns: dict[str, str], sample_rows: list[dict]) -> dict:
    """Guess the dataset layout.

    Returns ``{"format": "chat"|"prompt_answer"|"text"|"unknown", "prompt_field", "answer_field",
    "messages_field"}``. ``chat`` wins when a messages-like column holds role/content lists;
    ``prompt_answer`` needs a prompt column plus an answer column; ``text`` is a lone prompt/text
    column; ``unknown`` means no string column at all (preview only; ``open_source`` rejects it).
    """
    result: dict[str, Any] = {"format": "unknown", "prompt_field": None, "answer_field": None, "messages_field": None}
    if not columns:
        return result
    answer = _find(columns, ANSWER_COLUMNS)

    messages = _find(columns, MESSAGES_COLUMNS)
    if messages is not None and (not sample_rows or _column_holds_chat(messages, sample_rows)):
        result.update(format="chat", messages_field=messages, answer_field=answer)
        return result
    prompt = _find(columns, PROMPT_COLUMNS)
    if prompt is not None and sample_rows and _column_holds_chat(prompt, sample_rows):
        result.update(format="chat", messages_field=prompt, answer_field=answer)
        return result
    if prompt is not None:
        result.update(prompt_field=prompt, answer_field=answer)
        result["format"] = "prompt_answer" if answer is not None else "text"
        return result
    text = _find(columns, TEXT_COLUMNS)
    if text is None:
        text = next((c for c, dt in columns.items() if dt == "string" and c != answer), None)
    if text is not None:
        result.update(prompt_field=text, answer_field=answer)
        result["format"] = "prompt_answer" if answer is not None else "text"
    return result
