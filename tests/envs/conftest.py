"""Fixtures for the ``harness`` environment's sandbox tests: the orchestrator fakes
(``tests/orchestrator/fakes.py``) wired into the registry, and a stand-in ``docker`` executable
on ``PATH`` (``fake_docker_cli.py``) whose containers are directories of the test."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from monorl.envs import sandbox

sys.path.insert(0, str(Path(__file__).parent.parent / "orchestrator"))


@pytest.fixture
def fake_components(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    import fakes

    fakes.install_fakes(monkeypatch)
    yield fakes
    fakes.reset_recorders()


@pytest.fixture
def make_config(tmp_path: Path, fake_components: Any) -> Callable[..., Any]:
    from monorl.config.schema import RunConfig

    def _make(**overrides: Any) -> RunConfig:
        return RunConfig.from_dict(fake_components.fake_config_dict(tmp_path, **overrides))

    return _make


class FakeDocker:
    """What the stand-in ``docker`` did and holds (``fake_docker_cli.py``)."""

    def __init__(self, state: Path) -> None:
        self.state = state

    def calls(self, *commands: str) -> list[dict[str, Any]]:
        """The recorded invocations, in order (only ``commands`` when given)."""
        log = self.state / "calls.jsonl"
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return [call for call in calls if not commands or call["command"] in commands]

    def add_image(self, name: str, workdir: str = "") -> None:
        """Make ``name`` a local image whose ``WORKDIR`` is ``workdir``."""
        path = self.state / "images.json"
        images = json.loads(path.read_text()) if path.exists() else {}
        images[name] = {"workdir": workdir, "context": None}
        path.write_text(json.dumps(images))

    def containers(self) -> list[str]:
        return sorted(path.name for path in (self.state / "containers").glob("*"))


@pytest.fixture(scope="session")
def fake_docker_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding ``docker``: a shim that runs ``fake_docker_cli.py``. One for the whole
    session, so that the sandbox's per-CLI watchdog process is started once."""
    directory = tmp_path_factory.mktemp("fake-docker-bin")
    shim = directory / "docker"
    cli = Path(__file__).with_name("fake_docker_cli.py")
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{cli}" "$@"\n')
    shim.chmod(0o755)
    return directory


@pytest.fixture
def docker(fake_docker_bin: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    """The stand-in ``docker`` first on ``PATH``, with a fresh daemon state: image ``box`` only."""
    state = tmp_path / "docker-state"
    (state / "containers").mkdir(parents=True)
    monkeypatch.setenv("PATH", f"{fake_docker_bin}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    for name in ("FAKE_DOCKER_DAEMON", "FAKE_DOCKER_OS", "FAKE_DOCKER_FAIL", "FAKE_DOCKER_CGROUP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    # what the sandbox module remembers about a daemon and its images belongs to the real one
    monkeypatch.setattr(sandbox, "_daemons", {})
    monkeypatch.setattr(sandbox, "_images", {})
    fake = FakeDocker(state)
    fake.add_image("box")
    return fake
