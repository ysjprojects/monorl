"""Image inputs of gateway requests, kept as local files.

An agent sends an image as a base64 ``data:`` URL, as bare base64 (Anthropic) or as an
``http(s)`` URL. The gateway stores the bytes once in monorl's media cache
(``$MONORL_HOME/cache/media/<sha256>.<ext>``, :mod:`monorl.data.media`) and the chat refers to
that file: the rollout engine, the batch builder and the trainer then all read the same pixels
(a remote image is fetched exactly once), recorded trajectories stay small JSON, and the same
image sent again is the same reference, so a request that repeats its history still continues
the earlier call.

Torch-free; the media cache and ``httpx`` are imported with the first image.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

MAX_IMAGE_BYTES = 20 * 1024 * 1024
"""Largest remote image the gateway downloads."""

_HTTP_TIMEOUT = 30.0


class ImageError(ValueError):
    """An image input the gateway cannot use. ``code`` is one of OpenAI's image error codes."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def cache_image_url(url: Any, *, where: str, fetch: bool = True) -> str:
    """Local file for an image given as a base64 ``data:`` URL or — when the session fetches
    remote images (``fetch``) — an ``http(s)`` URL."""
    if not isinstance(url, str) or not url.strip():
        raise ImageError(f"{where}: the image has no URL", "invalid_image_url")
    url = url.strip()
    if url.startswith("data:"):
        return _cache(url, where, "invalid_base64_image")
    if url.startswith(("http://", "https://")):
        if not fetch:
            raise ImageError(
                f"{where}: this session does not fetch remote images; send the image as a base64 data: URL",
                "invalid_image_url",
            )
        return _cache(_download(url, where), where, "invalid_image")
    raise ImageError(f"{where}: an image must be an http(s) URL or a base64 data: URL", "invalid_image_url")


def cache_image_base64(data: Any, *, where: str) -> str:
    """Local file for an image given as bare base64."""
    if not isinstance(data, str) or not data:
        raise ImageError(f"{where}: the image has no base64 data", "invalid_base64_image")
    try:
        raw = base64.b64decode("".join(data.split()), validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageError(f"{where}: invalid base64 image data ({e})", "invalid_base64_image") from e
    return _cache(raw, where, "invalid_image")


def _cache(value: str | bytes, where: str, code: str) -> str:
    from monorl.core.errors import DataError
    from monorl.data.media import MediaResolver

    try:
        return MediaResolver().resolve_part_ref(value, where=where)
    except DataError as e:
        raise ImageError(str(e), code) from e


def _download(url: str, where: str) -> bytes:
    import httpx

    data = bytearray()
    try:
        with httpx.stream("GET", url, timeout=_HTTP_TIMEOUT, follow_redirects=True) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                data += chunk
                if len(data) > MAX_IMAGE_BYTES:
                    raise ImageError(
                        f"{where}: the image at {url} is larger than {MAX_IMAGE_BYTES >> 20} MB",
                        "image_too_large",
                    )
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        raise ImageError(
            f"{where}: could not download the image at {url}: {e}", "failed_to_download_image"
        ) from e
    return bytes(data)


__all__ = ["MAX_IMAGE_BYTES", "ImageError", "cache_image_base64", "cache_image_url"]
