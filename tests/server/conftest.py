from __future__ import annotations

import shlex
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from monorl.server.app import ServerSettings, create_app
from monorl.server.jobs import JobManager

FAKE_TRAINER = Path(__file__).with_name("fake_trainer.py")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    monkeypatch.setenv("MONORL_HOME", str(h))
    monkeypatch.setenv("MONORL_TRAIN_CMD", shlex.join([sys.executable, str(FAKE_TRAINER)]))
    monkeypatch.setenv("MONORL_CANCEL_GRACE", "1.0")
    monkeypatch.delenv("FAKE_TRAINER_MODE", raising=False)
    return h


@pytest.fixture
def recipes_dir(tmp_path: Path) -> Path:
    d = tmp_path / "recipes"
    d.mkdir()
    return d


@pytest.fixture
def envs_bin(tmp_path: Path) -> Path:
    d = tmp_path / "envs-bin"
    d.mkdir()
    return d


@pytest.fixture
def settings(home: Path, recipes_dir: Path, envs_bin: Path, tmp_path: Path) -> ServerSettings:
    return ServerSettings(
        home=home,
        recipes_dir=recipes_dir,
        web_dist=tmp_path / "web-dist",
        envs_bin=envs_bin,
        envs_sdk=tmp_path / "envs-sdk",
    )


@pytest.fixture
def jobs(settings: ServerSettings) -> JobManager:
    return JobManager(settings.home)


@pytest.fixture
def client(settings: ServerSettings, jobs: JobManager) -> Iterator[TestClient]:
    with TestClient(create_app(settings, jobs=jobs)) as c:
        yield c


@pytest.fixture
def config_dict(tmp_path: Path) -> dict[str, Any]:
    """Valid RunConfig; the fake trainer never loads the model or the data."""
    data = tmp_path / "data.jsonl"
    data.write_text('{"prompt": "2+2?", "answer": "4"}\n')
    return {
        "run_name": "srv-test",
        "model": {"name_or_path": "org/tiny-model"},
        "data": {"uri": str(data), "prompt_field": "prompt", "answer_field": "answer"},
        "algorithm": {"name": "grpo"},
        "rollout": {"group_size": 2, "max_new_tokens": 8},
        "trainer": {"max_steps": 3},
        "rewards": [{"type": "exact_match"}],
        "logging": {"run_root": str(tmp_path / "runs")},
    }


def _wait_for(predicate, *, timeout: float = 15.0, interval: float = 0.05):  # noqa: ANN001
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met within timeout")


@pytest.fixture
def wait_for():  # noqa: ANN201
    return _wait_for


@pytest.fixture
def fake_trainer_path() -> Path:
    return FAKE_TRAINER
