from __future__ import annotations

import asyncio
import json
import shlex
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from monorl.server import routes_jobs
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


class SseStream:
    """One live SSE request against the ASGI app, read event by event (no sockets).

    ``TestClient`` buffers a response until the app returns, so it can only show a stream that has
    already ended. This runs the request as a task on the test's event loop instead, which lets a
    test act between events. ``async with`` opens it and disconnects on exit.
    """

    def __init__(self, app: Any, path: str, params: dict[str, Any]) -> None:
        self._app = app
        self._scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": urlencode(params).encode(),
            "headers": [(b"host", b"testserver"), (b"accept", b"text/event-stream")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        self._events: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()
        self._disconnect = asyncio.Event()
        self._request_sent = False
        self._buffer = ""
        self._task: asyncio.Task[None] | None = None
        self.status: int | None = None

    async def _receive(self) -> dict[str, Any]:
        if not self._request_sent:
            self._request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await self._disconnect.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            self.status = message["status"]
            return
        self._buffer = (self._buffer + message.get("body", b"").decode()).replace("\r\n", "\n")
        while "\n\n" in self._buffer:
            block, self._buffer = self._buffer.split("\n\n", 1)
            name, data = "message", []
            for line in block.split("\n"):
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
            if data:  # comment-only blocks are keep-alive pings
                self._events.put_nowait((name, json.loads("\n".join(data))))

    async def _run(self) -> None:
        try:
            await self._app(self._scope, self._receive, self._send)
        finally:
            self._events.put_nowait(None)

    async def __aenter__(self) -> SseStream:
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._task is not None
        self._disconnect.set()
        await asyncio.wait_for(self._task, 10)

    async def event(self, timeout: float = 15.0) -> tuple[str, Any]:
        """The next ``(event name, decoded data)``."""
        item = await asyncio.wait_for(self._events.get(), timeout)
        assert item is not None, f"stream closed (HTTP {self.status}) without the expected event"
        return item

    async def until_end(self) -> list[tuple[str, Any]]:
        """Every remaining event up to and including ``end``."""
        events = [await self.event()]
        while events[-1][0] != "end":
            events.append(await self.event())
        return events

    async def idle(self, seconds: float) -> bool:
        """Whether the stream is still open ``seconds`` from now and has sent nothing."""
        assert self._task is not None
        await asyncio.sleep(seconds)
        return self._events.empty() and not self._task.done()


@pytest.fixture
def sse(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> Callable[..., SseStream]:
    """``sse(path, **query)`` -> :class:`SseStream` on the ``client`` app; the server-side tail
    polls fast so streaming tests take milliseconds per step."""
    monkeypatch.setattr(routes_jobs, "POLL_INTERVAL", 0.02)
    return lambda path, **params: SseStream(client.app, path, params)
