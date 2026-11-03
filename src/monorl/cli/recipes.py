"""Recipe discovery: ``configs/recipes/*.yaml`` with a leading ``# `` description block.

Shared by ``monorl recipes`` and the server's ``GET /api/recipes``. Torch-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monorl.config.loader import load_yaml
from monorl.core.errors import ConfigError

RECIPES_DIR_ENV = "MONORL_RECIPES_DIR"


def recipes_dir() -> Path:
    """``$MONORL_RECIPES_DIR`` or ``<repo>/configs/recipes`` next to the source tree."""
    override = os.environ.get(RECIPES_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[3] / "configs" / "recipes"


@dataclass(slots=True)
class Recipe:
    name: str
    path: Path
    description: str

    def load(self) -> dict[str, Any]:
        return load_yaml(self.path)


def read_description(path: Path) -> str:
    """Leading ``# ...`` comment lines, joined with spaces; stops at the first non-comment line."""
    lines: list[str] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line.startswith("#"):
                break
            text = line.lstrip("#").strip()
            if text:
                lines.append(text)
    return " ".join(lines)


def list_recipes(directory: Path | None = None) -> list[Recipe]:
    root = directory or recipes_dir()
    if not root.is_dir():
        return []
    out: list[Recipe] = []
    for path in sorted(root.glob("*.yaml")):
        out.append(Recipe(name=path.stem, path=path, description=read_description(path)))
    return out


def get_recipe(name: str, directory: Path | None = None) -> Recipe:
    root = directory or recipes_dir()
    stem = name[:-5] if name.endswith(".yaml") else name
    path = root / f"{stem}.yaml"
    if not path.is_file():
        available = ", ".join(r.name for r in list_recipes(root)) or "(none)"
        raise ConfigError(f"no recipe named {name!r} in {root}; available: {available}")
    return Recipe(name=stem, path=path, description=read_description(path))


__all__ = ["RECIPES_DIR_ENV", "Recipe", "get_recipe", "list_recipes", "read_description", "recipes_dir"]
