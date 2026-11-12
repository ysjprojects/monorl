"""Image column values → :class:`~monorl.core.types.ImagePart` references.

A dataset can hand us an image in many shapes: an HF ``datasets.Image`` feature (a PIL image, or
the ``{"bytes", "path"}`` struct parquet stores), raw bytes, a base64 string or ``data:`` URL, a
file path (relative to the dataset file / directory, or absolute), an ``s3://``/``hf://``/other
fsspec URI, or an ``http(s)://`` URL. :func:`MediaResolver.image_refs` turns any of them into
a list of references that are safe to keep inside a JSON sample:

* ``http(s)://`` URLs are kept verbatim (engines fetch them through the processor);
* existing local files are referenced by absolute path;
* everything else (bytes, PIL, base64, remote objects) is materialised **once** into
  ``$MONORL_HOME/cache/media/<sha256>.<ext>`` (``MONORL_HOME`` defaults to ``~/.monorl``) and
  referenced by that absolute path. The hash is over the stored bytes, so duplicates across rows
  and datasets share one file.

Materialisation is lazy: nothing is written until a row is mapped, so streaming sources only pay
for the rows they actually yield.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import posixpath
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from monorl.core.errors import DataError
from monorl.core.types import ContentPart, ImagePart
from monorl.data.uri import storage_options_for

__all__ = ["MediaResolver", "image_extension", "is_pil_image", "media_cache_dir", "normalize_content_parts"]

IMAGE_PLACEHOLDER = "<image>"
HTTP_SCHEMES = ("http://", "https://")
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/\s]+={0,2}$")
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<params>(?:;[\w.+-]+=[^;,]*)*)(?P<b64>;base64)?,(?P<payload>.*)$", re.DOTALL)
_PART_TYPE_ALIASES = {
    "text": "text",
    "input_text": "text",
    "image": "image",
    "image_url": "image",
    "input_image": "image",
    "audio": "audio",
    "input_audio": "audio",
    "audio_url": "audio",
    "video": "video",
    "video_url": "video",
}


def media_cache_dir() -> Path:
    home = Path(os.environ.get("MONORL_HOME") or "~/.monorl").expanduser()
    return home / "cache" / "media"


def is_pil_image(value: Any) -> bool:
    """``True`` for ``PIL.Image.Image`` instances without importing PIL."""
    return type(value).__module__.startswith("PIL.") and hasattr(value, "save") and hasattr(value, "size")


def image_extension(data: bytes) -> str | None:
    """Extension implied by the magic bytes (``png``/``jpg``/``gif``/``webp``/``bmp``), or ``None``."""
    for magic, ext in _IMAGE_MAGIC:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _pil_png_bytes(image: Any) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _decode_to_png(data: bytes, where: str) -> bytes:
    """Re-encode an image whose container we cannot name (TIFF, ...) as PNG so the cache file's
    extension always matches its content."""
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover - PIL is a datasets dependency
        raise DataError(f"{where}: cannot decode image bytes (Pillow not installed)") from e
    try:
        with Image.open(io.BytesIO(data)) as img:
            return _pil_png_bytes(img)
    except Exception as e:
        raise DataError(f"{where}: value is not a decodable image ({e})") from e


def cache_image_bytes(data: bytes, *, where: str) -> str:
    """Write ``data`` to the media cache (once) and return the absolute path."""
    if not data:
        raise DataError(f"{where}: empty image bytes")
    ext = image_extension(data)
    if ext is None:
        data = _decode_to_png(data, where)
        ext = "png"
    digest = hashlib.sha256(data).hexdigest()
    root = media_cache_dir()
    target = root / f"{digest}.{ext}"
    if target.exists():
        return str(target)
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=root, prefix=".tmp-", suffix=f".{ext}")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return str(target)


def _decode_base64(payload: str, where: str) -> bytes:
    try:
        return base64.b64decode("".join(payload.split()), validate=True)
    except (binascii.Error, ValueError) as e:
        raise DataError(f"{where}: invalid base64 image data ({e})") from e


def _looks_like_base64_image(value: str) -> bytes | None:
    """Decode ``value`` when it is a base64 string whose bytes start with an image signature."""
    if len(value) < 32 or not _BASE64_RE.match(value):
        return None
    try:
        data = base64.b64decode("".join(value.split()), validate=True)
    except (binascii.Error, ValueError):
        return None
    return data if image_extension(data) is not None else None


def _split_scheme(uri: str) -> tuple[str, str] | None:
    scheme, sep, rest = uri.partition("://")
    if sep and scheme and scheme.isalnum():
        return scheme, rest
    return None


class MediaResolver:
    """Resolve raw image column values for one dataset into part references.

    ``base`` is the directory relative paths are resolved against: a local directory, or a
    ``scheme://bucket/prefix`` / ``hf://datasets/org/name[@rev]`` location for remote sources.
    """

    def __init__(self, base: str | None = None) -> None:
        self.base = base

    # -- public
    def image_refs(self, value: Any, *, where: str) -> list[str]:
        """All image references held by ``value`` (lists are flattened; ``None`` yields nothing)."""
        return list(self._iter_refs(value, where))

    def image_parts(self, value: Any, *, where: str) -> list[ImagePart]:
        return [{"type": "image", "image": ref} for ref in self.image_refs(value, where=where)]

    def resolve_part_ref(self, value: Any, *, where: str) -> str:
        refs = self.image_refs(value, where=where)
        if len(refs) != 1:
            raise DataError(f"{where}: an image part must hold exactly one image, got {len(refs)}")
        return refs[0]

    # -- dispatch
    def _iter_refs(self, value: Any, where: str) -> Iterator[str]:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                yield from self._iter_refs(item, f"{where}[{i}]")
            return
        if hasattr(value, "tolist") and not isinstance(value, (bytes, str)):  # numpy arrays of refs
            yield from self._iter_refs(value.tolist(), where)
            return
        if is_pil_image(value):
            yield cache_image_bytes(_pil_png_bytes(value), where=where)
            return
        if isinstance(value, (bytes, bytearray, memoryview)):
            yield cache_image_bytes(bytes(value), where=where)
            return
        if isinstance(value, dict):
            yield self._from_dict(value, where)
            return
        if isinstance(value, str):
            yield self._from_str(value, where)
            return
        raise DataError(f"{where}: unsupported image value of type {type(value).__name__}")

    def _from_dict(self, value: dict[str, Any], where: str) -> str:
        # HF ``Image`` feature as stored in parquet / returned undecoded: {"bytes", "path"}
        if "bytes" in value or "path" in value:
            data = value.get("bytes")
            if data:
                return cache_image_bytes(bytes(data), where=where)
            path = value.get("path")
            if path:
                return self._from_str(str(path), where)
            raise DataError(f"{where}: image struct has neither bytes nor path")
        # OpenAI style part dicts
        kind = _PART_TYPE_ALIASES.get(str(value.get("type", "")))
        if kind == "image":
            inner = value.get("image_url", value.get("image", value.get("url")))
            if isinstance(inner, dict):
                inner = inner.get("url", inner.get("image"))
            return self.resolve_part_ref(inner, where=where)
        if "url" in value:
            return self._from_str(str(value["url"]), where)
        raise DataError(f"{where}: unsupported image object with keys {sorted(value)}")

    def _from_str(self, value: str, where: str) -> str:
        text = value.strip()
        if not text:
            raise DataError(f"{where}: empty image reference")
        if text.startswith("data:"):
            m = _DATA_URL_RE.match(text)
            if m is None:
                raise DataError(f"{where}: malformed data: URL")
            payload = m.group("payload")
            data = _decode_base64(payload, where) if m.group("b64") else payload.encode("latin-1")
            return cache_image_bytes(data, where=where)
        if text.startswith(HTTP_SCHEMES):
            return text
        if text.startswith("file://"):
            return self._local(Path(text[len("file://") :]).expanduser(), where)
        split = _split_scheme(text)
        if split is not None:
            return self._fetch_remote(text, where)
        decoded = _looks_like_base64_image(text)
        if decoded is not None:
            return cache_image_bytes(decoded, where=where)
        return self._from_path(text, where)

    # -- paths
    def _from_path(self, path: str, where: str) -> str:
        p = Path(path).expanduser()
        if p.is_absolute():
            return self._local(p, where)
        base = self.base
        if base is None:
            return self._local(p, where)
        split = _split_scheme(base)
        if split is None:
            return self._local(Path(base) / p, where)
        scheme, rest = split
        return self._fetch_remote(f"{scheme}://{posixpath.join(rest, path)}", where)

    @staticmethod
    def _local(path: Path, where: str) -> str:
        resolved = path.absolute()
        if not resolved.is_file():
            raise DataError(f"{where}: image file not found: {resolved}")
        return str(resolved)

    @staticmethod
    def _fetch_remote(uri: str, where: str) -> str:
        import fsspec

        scheme, _, rest = uri.partition("://")
        try:
            fs = fsspec.filesystem(scheme, **storage_options_for(scheme))
            data = fs.cat_file(rest)
        except (ValueError, ImportError) as e:
            raise DataError(f"{where}: no fsspec driver for {scheme!r} ({e})") from e
        except FileNotFoundError as e:
            raise DataError(f"{where}: image not found: {uri}") from e
        return cache_image_bytes(bytes(data), where=where)


# -- chat content normalisation ----------------------------------------------------------------


def normalize_content_parts(content: Any, resolver: MediaResolver, *, where: str) -> str | list[ContentPart]:
    """Normalise one message's ``content``.

    Strings are returned unchanged. Lists are mapped part by part: bare strings become text
    parts; OpenAI shapes (``{"type": "image_url", "image_url": {"url"}}``, ``{"type": "image",
    "image": ...}``, ``{"type": "input_text"/"text", "text"}``) become monorl parts with every
    image reference resolved through ``resolver``. Dicts that are not parts (an HF ``Image``
    struct, a PIL image) are treated as a single image.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if is_pil_image(content) or isinstance(content, (bytes, bytearray)):
        return [{"type": "image", "image": resolver.resolve_part_ref(content, where=where)}]
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, (list, tuple)):
        return str(content)
    parts: list[ContentPart] = []
    for i, part in enumerate(content):
        here = f"{where}.content[{i}]"
        if part is None:
            continue
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            parts.append({"type": "image", "image": resolver.resolve_part_ref(part, where=here)})
            continue
        if "bytes" in part or "path" in part:
            parts.append({"type": "image", "image": resolver.resolve_part_ref(part, where=here)})
            continue
        kind = _PART_TYPE_ALIASES.get(str(part.get("type", "")))
        if kind == "text":
            text = part.get("text")
            parts.append({"type": "text", "text": "" if text is None else str(text)})
        elif kind == "image":
            out: ImagePart = {"type": "image", "image": resolver.resolve_part_ref(part, where=here)}
            inner = part.get("image_url")
            detail = part.get("detail") or (inner.get("detail") if isinstance(inner, dict) else None)
            if detail:
                out["detail"] = str(detail)
            parts.append(out)
        elif kind in ("audio", "video"):
            ref = part.get(kind, part.get(f"{kind}_url", part.get("url")))
            if isinstance(ref, dict):
                ref = ref.get("url")
            if not ref:
                raise DataError(f"{here}: {kind} part without a reference")
            parts.append({"type": kind, kind: str(ref)})  # type: ignore[misc]
        else:
            raise DataError(f"{here}: unsupported content part type {part.get('type')!r}")
    return parts


def place_images(text: str, images: list[ImagePart], existing: list[ContentPart] | None = None) -> list[ContentPart]:
    """Build the user content: ``<image>`` placeholders in ``text`` are replaced in order by
    ``images``; images without a placeholder go first. ``existing`` media parts (from an
    already multimodal message) are kept ahead of everything."""
    parts: list[ContentPart] = [p for p in (existing or []) if p.get("type") != "text"]
    segments = text.split(IMAGE_PLACEHOLDER)
    n_slots = len(segments) - 1
    parts.extend(images[n_slots:])
    for i, segment in enumerate(segments):
        if segment:
            parts.append({"type": "text", "text": segment})
        if i < n_slots and i < len(images):
            parts.append(images[i])
    return parts
