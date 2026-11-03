"""Dataset URI parsing and filesystem resolution.

Supported shapes:

* ``hf://org/name[@revision][/path-in-repo]`` — Hugging Face Hub dataset repo (``hf://datasets/``
  prefix tolerated). An optional path narrows ``load_dataset`` to those ``data_files``.
* ``s3://bucket/prefix-or-file`` and any other fsspec scheme (``gs://``, ``az://``, ...) when the
  driver is installed. S3 credentials come **only** from the environment
  (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_SESSION_TOKEN``,
  ``AWS_DEFAULT_REGION``/``AWS_REGION``, ``AWS_ENDPOINT_URL``), never from config.
* local path or ``file://`` — file, directory or glob.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from monorl.core.errors import DataError

Kind = Literal["local", "hf", "remote"]

S3_ENV_VARS: tuple[str, ...] = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_ENDPOINT_URL",
)


@dataclass(frozen=True, slots=True)
class ParsedURI:
    raw: str
    kind: Kind
    scheme: str
    """``file``, ``hf`` or the fsspec protocol (``s3``, ``gs``...)."""
    path: str
    """Local: absolute path. Remote: ``bucket/key`` without the scheme. HF: ``org/name``."""
    revision: str | None = None
    """HF only: ``@revision``."""
    subpath: str | None = None
    """HF only: path inside the repo (used as ``data_files``)."""

    @property
    def repo_id(self) -> str:
        return self.path

    def with_path(self, path: str) -> str:
        """Render a URI for a path on the same filesystem (used for per-file ids and browse)."""
        if self.kind == "local":
            return path
        return f"{self.scheme}://{path}"


def parse_uri(uri: str) -> ParsedURI:
    if not uri or not uri.strip():
        raise DataError("dataset uri is empty")
    uri = uri.strip()
    if uri.startswith("hf://"):
        return _parse_hf(uri)
    if uri.startswith("file://"):
        local = uri[len("file://") :]
        return ParsedURI(raw=uri, kind="local", scheme="file", path=str(Path(local).expanduser().absolute()))
    if "://" in uri:
        scheme, _, rest = uri.partition("://")
        if not scheme.isidentifier():
            raise DataError(f"unsupported dataset uri {uri!r}")
        return ParsedURI(raw=uri, kind="remote", scheme=scheme, path=rest.strip("/") if rest != "" else "")
    return ParsedURI(raw=uri, kind="local", scheme="file", path=str(Path(uri).expanduser().absolute()))


def _parse_hf(uri: str) -> ParsedURI:
    body = uri[len("hf://") :].strip("/")
    parts = [p for p in body.split("/") if p]
    if len(parts) >= 3 and parts[0] == "datasets":
        parts = parts[1:]
    if len(parts) < 2:
        raise DataError(f"hf uri must look like hf://org/name[@revision]; got {uri!r}")
    org, name = parts[0], parts[1]
    revision = None
    if "@" in name:
        name, _, revision = name.partition("@")
        revision = revision or None
    subpath = "/".join(parts[2:]) or None
    if not org or not name:
        raise DataError(f"hf uri must look like hf://org/name[@revision]; got {uri!r}")
    return ParsedURI(raw=uri, kind="hf", scheme="hf", path=f"{org}/{name}", revision=revision, subpath=subpath)


# -- credentials / storage options -----------------------------------------------------------


def s3_storage_options() -> dict[str, Any]:
    """s3fs constructor kwargs built from the AWS_* environment. Empty values are omitted so
    botocore's default chain (IAM role, config files) still applies when nothing is set."""
    env = os.environ
    opts: dict[str, Any] = {}
    if env.get("AWS_ACCESS_KEY_ID"):
        opts["key"] = env["AWS_ACCESS_KEY_ID"]
    if env.get("AWS_SECRET_ACCESS_KEY"):
        opts["secret"] = env["AWS_SECRET_ACCESS_KEY"]
    if env.get("AWS_SESSION_TOKEN"):
        opts["token"] = env["AWS_SESSION_TOKEN"]
    client_kwargs: dict[str, Any] = {}
    region = env.get("AWS_DEFAULT_REGION") or env.get("AWS_REGION")
    if region:
        client_kwargs["region_name"] = region
    if env.get("AWS_ENDPOINT_URL"):
        client_kwargs["endpoint_url"] = env["AWS_ENDPOINT_URL"]
    if client_kwargs:
        opts["client_kwargs"] = client_kwargs
    return opts


def s3_env_status() -> dict[str, Any]:
    """Non-secret summary of the S3 connector for the UI (``/api/connectors``)."""
    env = os.environ
    present = [v for v in S3_ENV_VARS if env.get(v)]
    return {
        "configured": bool(env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY")),
        "region": env.get("AWS_DEFAULT_REGION") or env.get("AWS_REGION"),
        "endpoint_url": env.get("AWS_ENDPOINT_URL"),
        "vars_present": present,
    }


def storage_options_for(scheme: str) -> dict[str, Any]:
    if scheme in ("s3", "s3a"):
        return s3_storage_options()
    return {}


def filesystem_for(parsed: ParsedURI):
    """Return an fsspec filesystem for a local or remote URI."""
    import fsspec

    if parsed.kind == "hf":
        raise DataError("hf:// datasets are loaded through `datasets`, not fsspec")
    scheme = "file" if parsed.kind == "local" else parsed.scheme
    try:
        return fsspec.filesystem(scheme, **storage_options_for(scheme))
    except (ValueError, ImportError) as e:  # unknown protocol or driver missing
        raise DataError(f"no fsspec driver for scheme {scheme!r} ({e}); install the matching package") from e
