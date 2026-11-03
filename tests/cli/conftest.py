from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

MEP_SERVER = Path(__file__).parent / "mep_echo_server.py"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mep_server_script() -> Path:
    return MEP_SERVER
