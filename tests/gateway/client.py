"""Test helpers: a JSON-over-HTTP client for the gateway and tiny script agents for the harness."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# no proxy, whatever the machine is configured with: the gateway is on localhost
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(
    method: str,
    url: str,
    payload: Any = None,
    *,
    key: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> tuple[int, str]:
    """``(status, body text)`` of one request; error statuses are returned, not raised. ``key``
    is sent as bearer token, ``headers`` on top."""
    headers = {"Content-Type": "application/json", **(headers or {})}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def post(
    url: str, payload: Any, *, key: str | None = None, headers: dict[str, str] | None = None
) -> tuple[int, Any]:
    status, text = request("POST", url, payload, key=key, headers=headers)
    return status, json.loads(text)


def get(url: str, *, key: str | None = None) -> tuple[int, Any]:
    status, text = request("GET", url, key=key)
    return status, json.loads(text)


def sse_events(text: str) -> list[Any]:
    """The ``data:`` payloads of an SSE body, JSON-decoded except for the ``[DONE]`` marker."""
    payloads = [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]
    return [p if p == "[DONE]" else json.loads(p) for p in payloads]


def sse_named(text: str) -> list[tuple[str | None, Any]]:
    """``(event name, JSON data)`` of every event in an SSE body; comments and the ``[DONE]``
    marker are skipped."""
    events: list[tuple[str | None, Any]] = []
    for block in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if fields.get("data", "[DONE]") != "[DONE]":
            events.append((fields.get("event"), json.loads(fields["data"])))
    return events


AGENT_PRELUDE = '''
import json, os, subprocess, sys, time, urllib.error, urllib.request


def post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def chat(messages, **extra):
    payload = {"model": os.environ["MONORL_MODEL"], "messages": messages, **extra}
    return post(os.environ["OPENAI_BASE_URL"] + "/chat/completions", payload)["choices"][0]["message"]


def report(path, payload):
    """Write ``payload`` where the test looks for it, atomically."""
    with open(path + ".tmp", "w") as f:
        json.dump(payload, f)
    os.replace(path + ".tmp", path)


task = json.load(open(os.environ["MONORL_TASK_FILE"]))
'''


def write_agent(directory: Path, body: str, name: str = "agent.py") -> Path:
    """A script agent: :data:`AGENT_PRELUDE` (``post``, ``chat``, ``report``, ``task``) + ``body``."""
    path = directory / name
    path.write_text(AGENT_PRELUDE + body)
    return path


def wait_for_file(path: Path, timeout: float = 30.0) -> dict[str, Any]:
    """The JSON an agent wrote with ``report``, once it is there."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"{path} was not written")
        time.sleep(0.01)
    return json.loads(path.read_text())


def wait_dead(pid: int, timeout: float = 10.0) -> bool:
    """True once no process ``pid`` exists (a killed orphan is reaped by init a moment later)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False
