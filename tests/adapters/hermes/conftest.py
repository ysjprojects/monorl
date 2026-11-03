from __future__ import annotations

import os
import shutil
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

# Isolate Hermes state (config, skills, sessions) from the developer's ~/.hermes. Must happen
# before any hermes-agent import: a few modules capture HERMES_HOME at import time.
_HERMES_TEST_HOME = Path(os.environ.get("MONORL_HOME", "/tmp/monorl-tests")) / "hermes-home"
os.environ.setdefault("HERMES_HOME", str(_HERMES_TEST_HOME))
os.environ.setdefault("TERMINAL_ENV", "local")

pytest.importorskip("run_agent")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


@pytest.fixture(scope="session")
def skills_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("skills")
    skill = root / "arith" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: arith\ndescription: arithmetic via the terminal\n---\n"
        "# Arithmetic\nPipe expressions into `bc`: `echo '1+1' | bc`.\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def workspace() -> Iterator[Path]:
    # Not pytest's tmp_path: on macOS it lives under /private/var, which Hermes' file tools refuse
    # to write to (tools.file_tools._SENSITIVE_PATH_PREFIXES).
    ws = Path(os.environ["MONORL_HOME"]) / "hermes-test-ws" / uuid.uuid4().hex
    ws.mkdir(parents=True)
    try:
        yield ws
    finally:
        shutil.rmtree(ws, ignore_errors=True)
