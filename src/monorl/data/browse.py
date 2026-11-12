"""Cheap dataset inspection for the web UI / CLI: list entries and preview a few rows.

Neither function downloads a whole dataset: ``browse_uri`` only lists; ``preview_uri`` opens the
source in streaming mode and reads at most ``limit`` rows (plus the sniffing head).
"""

from __future__ import annotations

import base64
import datetime as _dt
import io
import posixpath
from typing import Any

from monorl.core.errors import DataError
from monorl.data.formats import SNIFF_ROWS, infer_columns, sniff_format
from monorl.data.media import image_extension, is_pil_image
from monorl.data.rows import open_rows
from monorl.data.uri import ParsedURI, filesystem_for, parse_uri

__all__ = ["browse_uri", "preview_uri"]

PREVIEW_MAX_STRING = 4000


def browse_uri(uri: str) -> list[dict[str, Any]]:
    """List a local directory, an ``s3://bucket/prefix`` (``s3://`` alone lists buckets) or the
    file tree of an ``hf://org/name[/path]`` dataset repo. A file URI lists itself.

    Returns ``[{name, uri, is_dir, size}]`` (directories first, then by name).
    """
    parsed = parse_uri(uri)
    if parsed.kind == "hf":
        entries = _browse_hf(parsed)
    else:
        entries = _browse_fs(parsed)
    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    return entries


def _browse_fs(parsed: ParsedURI) -> list[dict[str, Any]]:
    fs = filesystem_for(parsed)
    path = parsed.path
    fs.invalidate_cache()
    if parsed.kind == "remote" and path == "":
        listing = fs.ls("", detail=True)
    else:
        if not fs.exists(path):
            raise FileNotFoundError(f"path not found: {parsed.raw}")
        if fs.isfile(path):
            info = fs.info(path)
            return [_entry(parsed, info["name"], False, info.get("size"))]
        listing = fs.ls(path, detail=True)
    entries = []
    for info in listing:
        name = info["name"].rstrip("/")
        base = posixpath.basename(name)
        if not base or base.startswith("."):
            continue
        entries.append(_entry(parsed, name, info.get("type") == "directory", info.get("size")))
    return entries


def _entry(parsed: ParsedURI, path: str, is_dir: bool, size: int | None) -> dict[str, Any]:
    name = posixpath.basename(path.rstrip("/"))
    return {
        "name": name,
        "uri": parsed.with_path(path),
        "is_dir": is_dir,
        "size": None if is_dir else (int(size) if size is not None else None),
    }


def _browse_hf(parsed: ParsedURI) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    try:
        tree = api.list_repo_tree(
            parsed.repo_id, path_in_repo=parsed.subpath, repo_type="dataset", revision=parsed.revision
        )
        items = list(tree)
    except HfHubHTTPError as e:
        raise DataError(f"cannot list {parsed.raw!r}: {e}") from e
    rev = f"@{parsed.revision}" if parsed.revision else ""
    entries = []
    for item in items:
        is_dir = type(item).__name__ == "RepoFolder"
        entries.append(
            {
                "name": posixpath.basename(item.path),
                "uri": f"hf://{parsed.repo_id}{rev}/{item.path}",
                "is_dir": is_dir,
                "size": None if is_dir else getattr(item, "size", None),
            }
        )
    return entries


def preview_uri(
    uri: str, *, split: str | None = None, hf_config_name: str | None = None, limit: int = 20
) -> dict[str, Any]:
    """Read at most ``limit`` rows in streaming mode.

    Returns ``{"columns": {name: dtype}, "rows": [...], "inferred": sniff_format(...),
    "num_rows": int | None}``; ``num_rows`` is filled only when it is cheap (parquet metadata,
    Hub split info, small local files already materialised). Image values (PIL objects, HF
    ``Image`` structs, image bytes) are rendered as ``{"__image__": true, "width", "height"}``
    — never the bytes — and path/URI strings are passed through; ``inferred["image_fields"]``
    lists the candidate image columns.
    """
    if limit < 1:
        raise ValueError("limit must be >= 1")
    parsed = parse_uri(uri)
    rows = open_rows(parsed, split=split or "train", config_name=hf_config_name, streaming=True)
    window = max(limit, SNIFF_ROWS)
    head = rows.head(window)
    columns = rows.columns()
    if not columns and head:
        columns = infer_columns(head)
    inferred = sniff_format(columns, head)
    num_rows = rows.known_len()
    if num_rows is None and len(head) < window:
        num_rows = len(head)  # the whole source fit in the sniff window
    return {
        "columns": columns,
        "rows": [_jsonable(r) for r in head[:limit]],
        "inferred": inferred,
        "num_rows": num_rows,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= PREVIEW_MAX_STRING else value[:PREVIEW_MAX_STRING] + "…"
    if is_pil_image(value):
        return _image_summary(value.size)
    if isinstance(value, dict):
        if set(value) == {"bytes", "path"}:
            return _image_struct_summary(value)
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        if image_extension(bytes(value[:16])) is not None:
            return _image_summary(_image_size(bytes(value)))
        return {"__bytes__": len(value), "base64_head": base64.b64encode(bytes(value[:48])).decode()}
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if hasattr(value, "tolist"):  # numpy scalars/arrays
        return _jsonable(value.tolist())
    return str(value)


def _image_summary(size: tuple[int, int] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"__image__": True}
    if size is not None:
        out["width"], out["height"] = int(size[0]), int(size[1])
    return out


def _image_struct_summary(value: dict[str, Any]) -> Any:
    """HF ``Image`` struct: bytes → size summary (never the bytes), path-only → the path/URI."""
    data = value.get("bytes")
    if data:
        summary = _image_summary(_image_size(bytes(data)))
        if value.get("path"):
            summary["path"] = str(value["path"])
        return summary
    return str(value["path"]) if value.get("path") else None


def _image_size(data: bytes) -> tuple[int, int] | None:
    """Image dimensions from the header only (PIL is lazy; the pixels are never decoded)."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except Exception:
        return None
