"""``monorl env check``: drive any MEP/1 server through spec → reset → step×N → close using the
raw protocol helpers in :mod:`monorl.envs.protocol`. Independent of the in-process env client so
it exercises exactly what a Go/C++ author shipped. Torch-free.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from monorl.core.types import Completion, Sample
from monorl.envs.protocol import (
    EnvSpec,
    ProtocolError,
    close_request,
    health_request,
    parse_response,
    reset_request,
    spec_request,
    step_request,
    validate_reset_response,
    validate_step_response,
)

CANNED_SAMPLE = Sample(
    id="env-check-0",
    messages=[{"role": "user", "content": "Hello from `monorl env check`. Reply with anything."}],
    reference="hello",
    metadata={"source": "monorl env check"},
)
CANNED_ACTION = Completion(text="hello", token_ids=[], finish_reason="stop")


@dataclass(slots=True)
class StepRecord:
    turn: int
    reward: float
    done: bool
    messages: list[dict[str, Any]]
    info: dict[str, Any]


@dataclass(slots=True)
class CheckReport:
    transport: str
    target: str
    spec: dict[str, Any] | None = None
    health: dict[str, Any] | None = None
    reset_messages: list[dict[str, Any]] = field(default_factory=list)
    reset_info: dict[str, Any] = field(default_factory=dict)
    steps: list[StepRecord] = field(default_factory=list)
    closed: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "target": self.target,
            "ok": self.ok,
            "spec": self.spec,
            "health": self.health,
            "reset": {"messages": self.reset_messages, "info": self.reset_info},
            "steps": [
                {"turn": s.turn, "reward": s.reward, "done": s.done, "messages": s.messages, "info": s.info}
                for s in self.steps
            ],
            "closed": self.closed,
            "error": self.error,
        }


class StdioTransport:
    """One JSON object per line on the child's stdin/stdout; stderr passes through.

    stdout is read unbuffered with ``select`` so a hung server surfaces as a ``ProtocolError``
    after ``timeout`` seconds instead of blocking forever.
    """

    def __init__(self, command: list[str], *, cwd: str | None = None, timeout: float = 30.0) -> None:
        self.command = command
        self.timeout = timeout
        self._buf = b""
        try:
            self.proc = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        except OSError as e:
            raise ProtocolError("spawn", f"cannot start {command!r}: {e}") from e

    def _readline(self, op: str) -> bytes | None:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProtocolError("transport", f"timed out after {self.timeout}s awaiting {op}")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                return None
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.proc.stdin is not None
        op = payload["op"]
        try:
            self.proc.stdin.write((json.dumps(payload) + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise ProtocolError("transport", f"server closed stdin: {e}") from e
        while True:
            raw = self._readline(op)
            if raw is None:
                code = self.proc.poll()
                raise ProtocolError(
                    "transport", f"server closed stdout (exit code {code}) while awaiting {op}"
                )
            line = raw.strip()
            if not line:
                continue
            try:
                return parse_response(line)
            except json.JSONDecodeError as e:
                raise ProtocolError("protocol", f"non-JSON line on stdout: {line[:200]!r} ({e})") from e

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait(timeout=5)


class HttpTransport:
    """``POST /<op>`` JSON; ``GET /spec`` and ``GET /health``."""

    def __init__(self, url: str, *, timeout: float = 30.0) -> None:
        import httpx

        self.base = url.rstrip("/")
        self.client = httpx.Client(base_url=self.base, timeout=timeout)
        self._httpx = httpx

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        op = payload["op"]
        body = {k: v for k, v in payload.items() if k != "op"}
        try:
            if op in ("spec", "health"):
                resp = self.client.get(f"/{op}")
            else:
                resp = self.client.post(f"/{op}", json=body)
        except self._httpx.HTTPError as e:
            raise ProtocolError("transport", f"{op}: {e}") from e
        try:
            data = resp.json()
        except ValueError as e:
            raise ProtocolError(
                "protocol", f"{op}: non-JSON response (HTTP {resp.status_code}): {resp.text[:200]!r}"
            ) from e
        if resp.status_code >= 400 and not (isinstance(data, dict) and data.get("error")):
            raise ProtocolError("http", f"{op}: HTTP {resp.status_code}: {resp.text[:200]}")
        return parse_response(data)

    def close(self) -> None:
        self.client.close()


def run_check(
    *,
    command: list[str] | None = None,
    url: str | None = None,
    turns: int = 1,
    cwd: str | None = None,
    timeout: float = 30.0,
    sample: Sample = CANNED_SAMPLE,
    action: Completion = CANNED_ACTION,
) -> CheckReport:
    if (command is None) == (url is None):
        raise ValueError("exactly one of command= or url= is required")
    transport: StdioTransport | HttpTransport
    if command is not None:
        report = CheckReport(transport="stdio", target=" ".join(command))
        try:
            transport = StdioTransport(command, cwd=cwd, timeout=timeout)
        except ProtocolError as e:
            report.error = str(e)
            return report
    else:
        assert url is not None
        report = CheckReport(transport="http", target=url)
        transport = HttpTransport(url, timeout=timeout)

    episode_id = f"check-{uuid.uuid4().hex[:8]}"
    try:
        spec_payload = transport.request(spec_request())
        if not spec_payload.get("name"):
            raise ProtocolError("protocol", "spec response must carry a non-empty 'name'")
        try:
            spec = EnvSpec.from_dict(spec_payload)
        except (TypeError, ValueError) as e:
            raise ProtocolError("protocol", f"malformed spec: {e}") from e
        report.spec = spec.to_dict()
        if spec.protocol != "MEP/1":
            raise ProtocolError("protocol", f"unsupported protocol {spec.protocol!r}; expected 'MEP/1'")
        health = transport.request(health_request())
        if not health.get("ok"):
            raise ProtocolError("protocol", f"health must return {{ok: true}}, got {health!r}")
        report.health = health

        reset_payload = transport.request(reset_request(episode_id, sample, seed=0))
        msgs, info = validate_reset_response(reset_payload)
        if not msgs:
            raise ProtocolError("protocol", "reset returned no messages")
        report.reset_messages, report.reset_info = list(msgs), info

        for turn in range(1, max(turns, 1) + 1):
            step_payload = transport.request(step_request(episode_id, action))
            s_msgs, reward, done, s_info = validate_step_response(step_payload)
            report.steps.append(
                StepRecord(turn=turn, reward=reward, done=done, messages=list(s_msgs), info=s_info)
            )
            if done:
                break

        transport.request(close_request(episode_id))
        report.closed = True
    except ProtocolError as e:
        report.error = str(e)
    finally:
        transport.close()
    return report


__all__ = [
    "CANNED_ACTION",
    "CANNED_SAMPLE",
    "CheckReport",
    "HttpTransport",
    "StdioTransport",
    "StepRecord",
    "run_check",
]
