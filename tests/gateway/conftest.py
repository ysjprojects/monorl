"""Gateway fixtures: the orchestrator fakes (``tests/orchestrator/fakes.py``) wired into the
registry, the process-wide gateway, and sessions that are closed again after the test."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "orchestrator"))

import fakes  # noqa: E402

from monorl.config.schema import RunConfig  # noqa: E402
from monorl.core.types import SamplingParams  # noqa: E402
from monorl.gateway import Gateway, Session, get_gateway  # noqa: E402


@pytest.fixture
def fake_components(monkeypatch: pytest.MonkeyPatch) -> Any:
    fakes.install_fakes(monkeypatch)
    yield fakes
    fakes.reset_recorders()


@pytest.fixture
def make_config(tmp_path: Path, fake_components: Any) -> Callable[..., RunConfig]:
    def _make(**overrides: Any) -> RunConfig:
        return RunConfig.from_dict(fakes.fake_config_dict(tmp_path, **overrides))

    return _make


@pytest.fixture
def gateway() -> Gateway:
    return get_gateway()


@pytest.fixture
def open_session(gateway: Gateway) -> Iterator[Callable[..., Session]]:
    """``open_session(engine, **session_options)``: a session served by ``engine``."""
    sessions: list[Session] = []

    def _open(engine: Any, *, params: SamplingParams | None = None, **options: Any) -> Session:
        session = gateway.open_session(
            engine=engine,
            tokenizer=engine.tokenizer,
            params=params or SamplingParams(n=1, max_new_tokens=32),
            **options,
        )
        sessions.append(session)
        return session

    yield _open
    for session in sessions:
        gateway.close_session(session)
