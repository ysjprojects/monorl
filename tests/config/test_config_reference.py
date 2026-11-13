"""``docs/CONFIG.md`` is the rendering of the config schema, and the rendering drops no field."""

from __future__ import annotations

from pathlib import Path
from typing import get_args, get_origin

import pytest
from pydantic import BaseModel

from monorl.config.reference import REGENERATE_COMMAND, main, render_reference
from monorl.config.schema import RunConfig

CONFIG_MD = Path(__file__).resolve().parents[2] / "docs" / "CONFIG.md"


def _leaf_keys(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Walk the models independently of the renderer; list entries are keyed ``name[].field``."""
    keys: list[str] = []
    for name, field in model.model_fields.items():
        tp = field.annotation
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            keys += _leaf_keys(tp, f"{prefix}{name}.")
        elif (
            get_origin(tp) is list
            and isinstance(get_args(tp)[0], type)
            and issubclass(get_args(tp)[0], BaseModel)
        ):
            keys += _leaf_keys(get_args(tp)[0], f"{prefix}{name}[].")
        else:
            keys.append(f"{prefix}{name}")
    return keys


def test_the_checked_in_reference_matches_the_schema() -> None:
    assert CONFIG_MD.read_text(encoding="utf-8") == render_reference(), (
        f"docs/CONFIG.md is out of date with the config schema; run: {REGENERATE_COMMAND}"
    )


def test_rendering_is_deterministic() -> None:
    assert render_reference() == render_reference()


def test_every_run_config_field_has_a_table_row() -> None:
    keys = _leaf_keys(RunConfig)
    # the walk itself reaches top-level scalars, nested models and the reward entries
    assert {"seed", "model.lora.r", "algorithm.off_policy.anchor", "rewards[].type", "env.params"} <= set(
        keys
    )
    row_keys = [
        line.split("|")[1].strip() for line in render_reference().splitlines() if line.startswith("| `")
    ]
    missing = [k for k in keys if f"`{k}`" not in row_keys]
    assert not missing, f"fields without a row in the reference: {missing}"
    run_config_rows = [k for k in row_keys if not k.startswith(("`stages", "`name`", "`base`"))]
    assert len(run_config_rows) == len(set(run_config_rows)) == len(keys)


def test_check_detects_a_modified_copy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--check", str(CONFIG_MD)]) == 0
    assert capsys.readouterr().err == ""

    stale = tmp_path / "CONFIG.md"
    stale.write_text(CONFIG_MD.read_text(encoding="utf-8").replace("`seed`", "`sead`"), encoding="utf-8")
    assert main(["--check", str(stale)]) == 1
    err = capsys.readouterr().err
    assert REGENERATE_COMMAND in err and len(err.strip().splitlines()) == 1

    fresh = tmp_path / "fresh.md"
    assert main(["--write", str(fresh)]) == 0
    assert main(["--check", str(fresh)]) == 0
