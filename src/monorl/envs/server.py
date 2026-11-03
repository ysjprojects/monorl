"""Run any Python :class:`~monorl.core.interfaces.Environment` as a MEP/1 server.

``serve(env_factory, transport="stdio")`` reads one JSON request per line on stdin and writes one
JSON response per line on stdout (logging goes to stderr). ``transport="http"`` serves the same
operations with FastAPI/uvicorn in-process (``POST /<op>``, ``GET /spec``, ``GET /health``).
Used by ``monorl env serve`` and by the remote-env tests. :class:`MEPServer` holds the episode
table and is transport-agnostic; :func:`start_http_server` runs the http transport in a daemon
thread for tests and embedding.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from monorl.core.interfaces import Environment
from monorl.core.types import Completion
from monorl.envs.protocol import OPS, EnvSpec, ProtocolError, decode_sample

log = logging.getLogger("monorl.envs.server")

_FINISH_REASONS = {"stop", "length", "tool_call", "error"}


class MEPServer:
    """Transport-agnostic MEP/1 request handler over a zero-arg environment factory."""

    def __init__(
        self,
        env_factory: Callable[[], Environment],
        *,
        name: str | None = None,
        parallel_episodes: int = 8,
        version: str = "0",
        description: str = "",
    ) -> None:
        self._factory = env_factory
        self._episodes: dict[str, Environment] = {}
        self._lock = threading.Lock()
        probe = env_factory()
        try:
            doc = (type(probe).__doc__ or "").strip().splitlines()
            self.spec = EnvSpec(
                name=name or type(probe).__name__,
                version=version,
                max_turns=int(getattr(probe, "max_turns", 1)),
                parallel_episodes=max(1, int(parallel_episodes)),
                tools=list(getattr(probe, "tools", []) or []),
                description=description or (doc[0] if doc else ""),
            )
        finally:
            probe.close()

    @property
    def active_episodes(self) -> int:
        return len(self._episodes)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one request; returns the response object (error envelopes included)."""
        try:
            return self._dispatch(request)
        except ProtocolError as e:
            return e.to_dict()
        except Exception as e:  # env-side failure: report, keep serving
            log.exception("MEP request failed")
            return ProtocolError(type(e).__name__, str(e)).to_dict()

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ProtocolError("protocol", "request must be a JSON object")
        op = request.get("op")
        if op not in OPS:
            raise ProtocolError("protocol", f"unknown op {op!r}; expected one of {OPS}")
        if op == "spec":
            return self.spec.to_dict()
        if op == "health":
            return {"ok": True}
        episode_id = request.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ProtocolError("protocol", f"{op} needs a string episode_id")
        if op == "reset":
            return self._reset(episode_id, request)
        if op == "step":
            return self._step(episode_id, request)
        return self._close(episode_id)

    def _reset(self, episode_id: str, request: dict[str, Any]) -> dict[str, Any]:
        sample_payload = request.get("sample")
        if not isinstance(sample_payload, dict) or "id" not in sample_payload:
            raise ProtocolError("protocol", "reset needs sample {id, messages, reference, metadata}")
        sample = decode_sample(sample_payload)
        with self._lock:
            if len(self._episodes) >= self.spec.parallel_episodes and episode_id not in self._episodes:
                raise ProtocolError("capacity", f"{self.spec.parallel_episodes} episodes already active")
            previous = self._episodes.pop(episode_id, None)
        if previous is not None:
            previous.close()
        env = self._factory()
        messages = env.reset(sample)
        with self._lock:
            self._episodes[episode_id] = env
        return {"messages": messages, "info": {"episode_id": episode_id}}

    def _step(self, episode_id: str, request: dict[str, Any]) -> dict[str, Any]:
        action = request.get("action")
        if not isinstance(action, dict) or not isinstance(action.get("text"), str):
            raise ProtocolError("protocol", "step needs action {text, finish_reason, tool_calls?}")
        with self._lock:
            env = self._episodes.get(episode_id)
        if env is None:
            raise ProtocolError("unknown_episode", f"no active episode {episode_id!r}")
        finish = action.get("finish_reason", "stop")
        completion = Completion(
            text=action["text"],
            token_ids=[],
            finish_reason=finish if finish in _FINISH_REASONS else "stop",  # type: ignore[arg-type]
        )
        result = env.step(completion)
        if result.done:
            with self._lock:
                self._episodes.pop(episode_id, None)
            env.close()
        return {"messages": result.messages, "reward": float(result.reward), "done": bool(result.done), "info": dict(result.info)}

    def _close(self, episode_id: str) -> dict[str, Any]:
        with self._lock:
            env = self._episodes.pop(episode_id, None)
        if env is not None:
            env.close()
        return {}

    def close_all(self) -> None:
        with self._lock:
            envs = list(self._episodes.values())
            self._episodes.clear()
        for env in envs:
            env.close()


# -- stdio ------------------------------------------------------------------------------------


def serve_stdio(server: MEPServer, stdin: Any = None, stdout: Any = None) -> None:
    """Block reading JSON lines until EOF. Blank lines are ignored."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response = ProtocolError("protocol", f"invalid JSON: {e.msg}").to_dict()
        else:
            response = server.handle(request)
        stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        stdout.flush()
    server.close_all()


# -- http -------------------------------------------------------------------------------------


def create_app(server: MEPServer) -> FastAPI:
    app = FastAPI(title=f"monorl MEP/1: {server.spec.name}")

    def _respond(payload: dict[str, Any]) -> JSONResponse:
        if "error" in payload:
            etype = payload["error"].get("type", "")
            status = 400 if etype in {"protocol", "unknown_episode"} else 503 if etype == "capacity" else 500
            return JSONResponse(payload, status_code=status)
        return JSONResponse(payload)

    @app.get("/spec")
    def get_spec() -> JSONResponse:
        return _respond(server.handle({"op": "spec"}))

    @app.get("/health")
    def get_health() -> JSONResponse:
        return _respond(server.handle({"op": "health"}))

    @app.post("/{op}")
    async def post_op(op: str, request: Request) -> JSONResponse:
        if op not in OPS:
            return _respond(ProtocolError("protocol", f"unknown op {op!r}").to_dict())
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            return _respond(ProtocolError("protocol", f"invalid JSON: {e.msg}").to_dict())
        if not isinstance(body, dict):
            return _respond(ProtocolError("protocol", "body must be a JSON object").to_dict())
        body["op"] = op
        # env.step/reset are synchronous; run them off the event loop so episodes overlap.
        return _respond(await run_in_threadpool(server.handle, body))

    return app


class HttpServerHandle:
    """A uvicorn server running in a daemon thread (tests / embedding)."""

    def __init__(self, server: MEPServer, host: str, port: int) -> None:
        import uvicorn

        self.mep = server
        self._config = uvicorn.Config(create_app(server), host=host, port=port, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(self._config)
        self._thread = threading.Thread(target=self._server.run, name="monorl-mep-http", daemon=True)
        self.host = host
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout_s: float = 10.0) -> HttpServerHandle:
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("MEP http server thread exited before starting")
            if time.monotonic() > deadline:
                raise TimeoutError("MEP http server did not start in time")
            time.sleep(0.02)
        # uvicorn binds before `started`; resolve the real port when 0 was requested.
        for srv in self._server.servers:
            for sock in srv.sockets:
                self.port = sock.getsockname()[1]
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=timeout_s)
        self.mep.close_all()


def start_http_server(server: MEPServer, host: str = "127.0.0.1", port: int = 0) -> HttpServerHandle:
    return HttpServerHandle(server, host, port).start()


def serve(
    env_factory: Callable[[], Environment],
    *,
    transport: Literal["stdio", "http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    name: str | None = None,
    parallel_episodes: int = 8,
) -> None:
    """Serve ``env_factory`` (a fresh Environment per episode) until EOF / shutdown."""
    server = MEPServer(env_factory, name=name, parallel_episodes=parallel_episodes)
    if transport == "stdio":
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        # Responses own the real stdout; anything the env prints must not corrupt the stream.
        out, sys.stdout = sys.stdout, sys.stderr
        try:
            serve_stdio(server, stdin=sys.stdin, stdout=out)
        finally:
            sys.stdout = out
    elif transport == "http":
        import uvicorn

        uvicorn.run(create_app(server), host=host, port=port, log_level="info")
    else:
        raise ValueError(f"unknown transport {transport!r}; expected 'stdio' or 'http'")


__all__ = ["HttpServerHandle", "MEPServer", "create_app", "serve", "serve_stdio", "start_http_server"]
