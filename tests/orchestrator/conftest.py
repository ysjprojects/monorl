"""Orchestrator fixtures: fake components (see ``fakes.py``) wired into the registry and the
orchestrator's lazy sibling-package lookups."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fakes  # noqa: E402

from monorl.config.schema import RunConfig  # noqa: E402


@pytest.fixture
def fake_components(monkeypatch: pytest.MonkeyPatch) -> Any:
    fakes.install_fakes(monkeypatch)
    yield fakes
    fakes.reset_recorders()


@pytest.fixture
def make_config(tmp_path: Path, fake_components: Any):
    def _make(**overrides: Any) -> RunConfig:
        return RunConfig.from_dict(fakes.fake_config_dict(tmp_path, **overrides))

    return _make
