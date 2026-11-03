"""MEP/1 remote environment client (``env.type = "remote"``).

``env.params``::

    {"transport": "stdio", "command": ["./my-env", "--stdio"], "cwd": ".", "env": {...},
     "parallel": 4, "timeout_s": 60}
    {"transport": "http", "url": "http://127.0.0.1:8080", "timeout_s": 60}

Stdio: the orchestrator spawns the env binary and speaks JSON lines over its stdin/stdout; stderr
is forwarded to the ``monorl.envs.remote`` logger. One process serves up to
``spec.parallel_episodes`` (or ``params.parallel``) concurrent episodes; a process-pool keyed by
``(command, cwd, env)`` is shared by all :class:`RemoteEnv` instances created from the same
params, spawning more processes when more episodes run concurrently. Requests to one process are
serialised (the protocol is line-oriented) and multiplexed by ``episode_id``. If the process dies
or a request times out, it is restarted; the in-flight episode fails with
:class:`~monorl.core.errors.EnvironmentError_` and later episodes land on the fresh process.

Http: a shared :class:`httpx.Client` per url; ``GET /spec`` is fetched once and cached.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from monorl.core.errors import EnvironmentError_
from monorl.core.types import Completion, Message, Sample, StepResult
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
from monorl.envs.tool_calling import parse_tool_calls

log = logging.getLogger("monorl.envs.remote")

_EOF = object()


class Transport(Protocol):
    def spec(self) -> EnvSpec: ...

    def request(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


# -- stdio ------------------------------------------------------------------------------------


class StdioTransport:
    """One env process. Thread-safe; one request in flight at a time."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 60.0,
        start_timeout_s: float | None = None,
    ) -> None:
        if not command:
            raise ValueError("stdio transport needs a non-empty command")
        self.command = [str(c) for c in command]
        self.cwd = cwd
        self.env = dict(env) if env else None
        self.timeout_s = float(timeout_s)
        self.start_timeout_s = float(start_timeout_s) if start_timeout_s is not None else max(self.timeout_s, 30.0)
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[Any] = queue.Queue()
        self._spec: EnvSpec | None = None
        self.generation = 0
        """Incremented on every (re)start; episodes started on an older generation are gone."""
        self.restarts = 0

    # -- process lifecycle ----------------------------------------------------------------
    def _start(self) -> None:
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        try:
            self._proc = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            raise EnvironmentError_(f"cannot spawn env process {self.command!r}: {e}") from e
        self._lines = queue.Queue()
        self.generation += 1
        proc = self._proc
        threading.Thread(target=self._pump_stdout, args=(proc, self._lines), daemon=True, name="mep-stdout").start()
        threading.Thread(target=self._pump_stderr, args=(proc,), daemon=True, name="mep-stderr").start()
        log.info("started env process (pid %s, generation %d): %s", proc.pid, self.generation, " ".join(self.command))

    @staticmethod
    def _pump_stdout(proc: subprocess.Popen[str], lines: queue.Queue[Any]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(_EOF)

    @staticmethod
    def _pump_stderr(proc: subprocess.Popen[str]) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            log.info("[env pid %s] %s", proc.pid, line.rstrip("\n"))

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            if self._proc is not None:
                self.restarts += 1
                log.warning("env process exited with code %s; restarting", self._proc.returncode)
            self._start()
        assert self._proc is not None
        return self._proc

    def _kill(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass

    def _crash(self, reason: str) -> EnvironmentError_:
        code = self._proc.poll() if self._proc is not None else None
        self._kill()
        self.restarts += 1
        log.error("env process failure (%s, exit code %s); restarting", reason, code)
        try:
            self._start()
        except EnvironmentError_ as e:
            return EnvironmentError_(f"{reason}; restart failed: {e}")
        return EnvironmentError_(f"env process {self.command[0]!r} failed: {reason} (exit code {code}); restarted")

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    # -- requests -------------------------------------------------------------------------
    def request(self, payload: dict[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        timeout = self.timeout_s if timeout_s is None else timeout_s
        with self._lock:
            proc = self._ensure_process()
            assert proc.stdin is not None
            try:
                proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise self._crash(f"write failed: {e}") from e
            while True:
                try:
                    line = self._lines.get(timeout=timeout)
                except queue.Empty:
                    raise self._crash(f"no response within {timeout}s") from None
                if line is _EOF:
                    raise self._crash("process closed stdout")
                if line.strip():
                    break
            try:
                return parse_response(line)
            except json.JSONDecodeError as e:
                raise self._crash(f"non-JSON response line {line[:200]!r}: {e.msg}") from e

    def spec(self) -> EnvSpec:
        if self._spec is None:
            self._spec = EnvSpec.from_dict(self.request(spec_request(), timeout_s=self.start_timeout_s))
        return self._spec

    def health(self) -> bool:
        return bool(self.request(health_request()).get("ok"))

    def close(self) -> None:
        with self._lock:
            proc = self._proc
            if proc is None:
                return
            if proc.poll() is None:
                try:
                    if proc.stdin:
                        proc.stdin.close()
                    proc.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            self._kill()
            self._proc = None


# -- http -------------------------------------------------------------------------------------


class HttpTransport:
    """Shared client for one MEP/1 http server; safe to use from many threads."""

    def __init__(self, url: str, *, timeout_s: float = 60.0, headers: Mapping[str, str] | None = None) -> None:
        import httpx

        self.url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._client = httpx.Client(base_url=self.url, timeout=self.timeout_s, headers=dict(headers or {}))
        self._spec: EnvSpec | None = None
        self._spec_lock = threading.Lock()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        body = dict(payload)
        op = body.pop("op")
        try:
            if op in ("spec", "health"):
                resp = self._client.get(f"/{op}")
            else:
                resp = self._client.post(f"/{op}", json=body)
        except httpx.HTTPError as e:
            raise EnvironmentError_(f"MEP http request {op} to {self.url} failed: {e}") from e
        try:
            payload_out = resp.json()
        except ValueError as e:
            raise ProtocolError("protocol", f"non-JSON response (HTTP {resp.status_code}): {resp.text[:200]!r}") from e
        if resp.status_code >= 400 and not (isinstance(payload_out, dict) and payload_out.get("error")):
            raise ProtocolError("http", f"HTTP {resp.status_code}: {resp.text[:200]}")
        return parse_response(payload_out)

    def spec(self) -> EnvSpec:
        with self._spec_lock:
            if self._spec is None:
                self._spec = EnvSpec.from_dict(self.request(spec_request()))
            return self._spec

    def health(self) -> bool:
        return bool(self.request(health_request()).get("ok"))

    def close(self) -> None:
        self._client.close()


# -- slot sources -----------------------------------------------------------------------------


class SlotSource(Protocol):
    """Hands a transport to an episode and takes it back when the episode ends."""

    def spec(self) -> EnvSpec: ...

    def acquire(self) -> Transport: ...

    def release(self, transport: Transport) -> None: ...

    def close(self) -> None: ...


class HttpSlotSource:
    def __init__(self, transport: HttpTransport) -> None:
        self.transport = transport

    def spec(self) -> EnvSpec:
        return self.transport.spec()

    def acquire(self) -> HttpTransport:
        return self.transport

    def release(self, transport: Transport) -> None:
        return None

    def close(self) -> None:
        self.transport.close()


@dataclass(slots=True)
class _Slot:
    transport: StdioTransport
    active: int = 0


class StdioPool:
    """Processes for one command; each serves up to ``parallel`` concurrent episodes."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 60.0,
        parallel: int | None = None,
    ) -> None:
        self._kwargs = {"cwd": cwd, "env": env, "timeout_s": timeout_s}
        self.command = list(command)
        self._parallel_override = parallel
        self._slots: list[_Slot] = []
        self._lock = threading.Lock()
        self._spec: EnvSpec | None = None

    def _spawn(self) -> _Slot:
        slot = _Slot(StdioTransport(self.command, **self._kwargs))
        self._slots.append(slot)
        return slot

    def spec(self) -> EnvSpec:
        with self._lock:
            if self._spec is None:
                slot = self._slots[0] if self._slots else self._spawn()
                self._spec = slot.transport.spec()
            return self._spec

    @property
    def parallel(self) -> int:
        if self._parallel_override is not None:
            return max(1, int(self._parallel_override))
        return max(1, self.spec().parallel_episodes)

    @property
    def processes(self) -> list[StdioTransport]:
        return [s.transport for s in self._slots]

    def acquire(self) -> StdioTransport:
        cap = self.parallel
        with self._lock:
            for slot in self._slots:
                if slot.active < cap:
                    slot.active += 1
                    return slot.transport
            slot = self._spawn()
            slot.active = 1
            return slot.transport

    def release(self, transport: Transport) -> None:
        with self._lock:
            for slot in self._slots:
                if slot.transport is transport:
                    slot.active = max(0, slot.active - 1)
                    return

    def close(self) -> None:
        with self._lock:
            slots, self._slots = self._slots, []
        for slot in slots:
            slot.transport.close()


_POOLS: dict[tuple[Any, ...], SlotSource] = {}
_POOLS_LOCK = threading.Lock()


def _pool_key(params: Mapping[str, Any]) -> tuple[Any, ...]:
    transport = params.get("transport", "stdio")
    if transport == "http":
        return ("http", str(params["url"]), float(params.get("timeout_s", 60.0)))
    env_items = tuple(sorted((str(k), str(v)) for k, v in dict(params.get("env") or {}).items()))
    return (
        "stdio",
        tuple(str(c) for c in params.get("command") or []),
        params.get("cwd"),
        env_items,
        float(params.get("timeout_s", 60.0)),
        params.get("parallel"),
    )


def slot_source_for(params: Mapping[str, Any]) -> SlotSource:
    """Shared transport pool for ``params`` (created on first use)."""
    transport = params.get("transport", "stdio")
    if transport not in ("stdio", "http"):
        raise ValueError(f"remote env: unknown transport {transport!r}; expected 'stdio' or 'http'")
    if transport == "http" and not params.get("url"):
        raise ValueError("remote env: http transport needs params.url")
    if transport == "stdio" and not params.get("command"):
        raise ValueError("remote env: stdio transport needs params.command")
    key = _pool_key(params)
    with _POOLS_LOCK:
        source = _POOLS.get(key)
        if source is None:
            timeout_s = float(params.get("timeout_s", 60.0))
            if transport == "http":
                source = HttpSlotSource(HttpTransport(str(params["url"]), timeout_s=timeout_s, headers=params.get("headers")))
            else:
                source = StdioPool(
                    [str(c) for c in params["command"]],
                    cwd=params.get("cwd"),
                    env=params.get("env"),
                    timeout_s=timeout_s,
                    parallel=params.get("parallel"),
                )
            _POOLS[key] = source
        return source


def shutdown_pools() -> None:
    with _POOLS_LOCK:
        sources = list(_POOLS.values())
        _POOLS.clear()
    for s in sources:
        s.close()


# -- environment ----------------------------------------------------------------------------


@dataclass(slots=True)
class _Episode:
    id: str
    transport: Transport
    generation: int = 0
    turn: int = 0
    info: dict[str, Any] = field(default_factory=dict)


class RemoteEnv:
    """An :class:`~monorl.core.interfaces.Environment` backed by a MEP/1 server."""

    def __init__(
        self,
        source: SlotSource,
        *,
        max_turns: int | None = None,
        seed: int | None = None,
        attach_tool_calls: bool = True,
    ) -> None:
        self._source = source
        self.spec: EnvSpec = source.spec()
        self.tools: list[dict[str, Any]] = list(self.spec.tools)
        self.max_turns: int = int(max_turns) if max_turns is not None else max(1, int(self.spec.max_turns))
        self.seed = seed
        self.attach_tool_calls = attach_tool_calls and bool(self.spec.tools)
        self._episode: _Episode | None = None

    @classmethod
    def from_params(cls, params: Mapping[str, Any], *, max_turns: int | None = None, seed: int | None = None) -> RemoteEnv:
        return cls(slot_source_for(params), max_turns=max_turns, seed=seed)

    @property
    def episode_id(self) -> str | None:
        return self._episode.id if self._episode else None

    def _call(self, ep: _Episode, payload: dict[str, Any]) -> dict[str, Any]:
        gen = getattr(ep.transport, "generation", 0)
        if gen != ep.generation:
            self._drop(ep)
            raise EnvironmentError_(f"env process restarted; episode {ep.id} is lost")
        try:
            return ep.transport.request(payload)
        except ProtocolError as e:
            if e.type in ("unknown_episode", "capacity"):
                self._drop(ep)
            raise EnvironmentError_(f"MEP {payload.get('op')} failed: {e}") from e
        except EnvironmentError_:
            self._drop(ep)
            raise

    def _drop(self, ep: _Episode) -> None:
        if self._episode is ep:
            self._episode = None
            self._source.release(ep.transport)

    def reset(self, sample: Sample) -> list[Message]:
        if self._episode is not None:
            self.close()
        transport = self._source.acquire()
        ep = _Episode(id=f"{sample.id}-{uuid.uuid4().hex[:8]}", transport=transport)
        self._episode = ep
        try:
            # A dead process is restarted inside request(); skip the generation check here and
            # record the generation that actually answered.
            payload = transport.request(reset_request(ep.id, sample, seed=self.seed))
        except ProtocolError as e:
            self._drop(ep)
            raise EnvironmentError_(f"MEP reset failed: {e}") from e
        except EnvironmentError_:
            self._drop(ep)
            raise
        ep.generation = getattr(transport, "generation", 0)
        messages, ep.info = validate_reset_response(payload)
        return messages

    def step(self, action: Completion) -> StepResult:
        ep = self._episode
        if ep is None:
            raise EnvironmentError_("step() before reset()")
        tool_calls = None
        if self.attach_tool_calls:
            parsed = parse_tool_calls(action.text)
            if parsed.calls:
                tool_calls = [dict(c) for c in parsed.calls]
        payload = self._call(ep, step_request(ep.id, action, tool_calls))
        messages, reward, done, info = validate_step_response(payload)
        ep.turn += 1
        if not done and ep.turn >= self.max_turns:
            done = True
            info = {**info, "truncated": True}
            self.close()
        elif done:
            self._drop(ep)
        return StepResult(messages=messages, reward=reward, done=done, info=info)

    def close(self) -> None:
        ep = self._episode
        if ep is None:
            return
        try:
            self._call(ep, close_request(ep.id))
        except EnvironmentError_ as e:
            log.debug("close for episode %s failed: %s", ep.id, e)
        finally:
            self._drop(ep)


def make_remote_env(params: Mapping[str, Any], *, max_turns: int | None = None, seed: int | None = None) -> RemoteEnv:
    return RemoteEnv.from_params(params, max_turns=max_turns, seed=seed)


__all__ = [
    "HttpSlotSource",
    "HttpTransport",
    "RemoteEnv",
    "SlotSource",
    "StdioPool",
    "StdioTransport",
    "Transport",
    "make_remote_env",
    "shutdown_pools",
    "slot_source_for",
]
