"""YAML loading with ``${VAR}`` / ``${VAR:-default}`` environment interpolation and dotted
``key.path=value`` overrides (CLI ``--set``)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from monorl.core.errors import ConfigError

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def interpolate_env(value: Any) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is not set (referenced as ${{{name}}})")

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {p} must be a mapping")
    return interpolate_env(data)


def _coerce(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """``apply_overrides(cfg, ["trainer.max_steps=10", "rewards=[{type: exact_match}]"])``."""
    if not overrides:
        return data
    out = dict(data)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like key.path=value, got {item!r}")
        key, _, raw = item.partition("=")
        parts = key.strip().split(".")
        node: dict[str, Any] = out
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = _coerce(raw)
    return out
