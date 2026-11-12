"""The capture gateway's HTTP server: an OpenAI-compatible endpoint served from the policy being
trained.

:class:`Gateway` holds the session table and runs a FastAPI app with uvicorn in a daemon thread,
bound to localhost on a free port. :func:`get_gateway` returns the process-wide instance, started
on first use and shared by every episode. A session is addressed by path or by API key::

    POST /s/{session}/v1/chat/completions      POST /v1/chat/completions   (Authorization: Bearer <session>)
    GET  /s/{session}/v1/models                GET  /v1/models             (Authorization: Bearer <session>)
    POST /s/{session}/reward   {"reward": float}   -> {"reward": <total posted so far>}
    GET  /health

Errors use OpenAI's JSON shape (``{"error": {"message", "type", "param", "code"}}``): 400 for a
request the gateway does not serve, 401/404 for an unknown session, 503 once the session is closed
or its generation was cancelled.

A completion blocks in ``engine.generate`` until its batch is served, so the calls run on worker
threads under the gateway's own limiter (:data:`MAX_IN_FLIGHT`) instead of the event loop's
default 40-thread pool: the calls of all parallel episodes are in flight together and the run's
``BatchingEngine`` coalesces them.
"""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from monorl.core.types import SamplingParams
from monorl.gateway.protocol import (
    GatewayError,
    completion_body,
    parse_chat_request,
    stream_events,
    usage,
)
from monorl.gateway.session import Session

if TYPE_CHECKING:
    from monorl.core.interfaces import RolloutEngine

log = logging.getLogger("monorl.gateway")

MAX_IN_FLIGHT = 512
"""Completions one gateway serves at once; further calls wait for a free slot."""

_CLOSED_REMEMBERED = 4096
"""How many closed session ids keep answering 503 (instead of "unknown session")."""


class Gateway:
    """Session table plus the HTTP server in front of it (module docstring)."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0, max_in_flight: int = MAX_IN_FLIGHT) -> None:
        import uvicorn

        self.host = host
        self.port = port
        self.max_in_flight = max(1, int(max_in_flight))
        self._sessions: dict[str, Session] = {}
        self._closed: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        self._limiter: anyio.CapacityLimiter | None = None
        self._pid = os.getpid()
        self._server = uvicorn.Server(
            uvicorn.Config(
                create_app(self),
                host=host,
                port=port,
                log_level="warning",
                lifespan="off",
                timeout_graceful_shutdown=2,
            )
        )
        self._thread = threading.Thread(target=self._server.run, name="monorl-gateway", daemon=True)

    # -- server lifecycle -----------------------------------------------------------------
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def running(self) -> bool:
        """The server thread is alive in this process (a forked child has none)."""
        return self._thread.is_alive() and self._pid == os.getpid()

    def start(self, timeout_s: float = 10.0) -> Gateway:
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("gateway server thread exited before starting")
            if time.monotonic() > deadline:
                raise TimeoutError("gateway server did not start in time")
            time.sleep(0.01)
        # uvicorn binds before `started`; resolve the real port when 0 was requested.
        for server in self._server.servers:
            for sock in server.sockets:
                self.port = sock.getsockname()[1]
        log.info("capture gateway listening on %s", self.url)
        return self

    def stop(self, timeout_s: float = 5.0) -> None:
        """Close every session and shut the server down."""
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            self.close_session(session)
        self._server.should_exit = True
        self._thread.join(timeout=timeout_s)

    # -- sessions ---------------------------------------------------------------------------
    def open_session(
        self,
        *,
        engine: RolloutEngine,
        tokenizer: Any,
        params: SamplingParams,
        turn_rendering: str = "tokens",
        model: str = "monorl",
        max_calls: int | None = None,
        max_context_tokens: int | None = None,
    ) -> Session:
        """A new session (one episode) served with ``engine`` under ``params``. Its id is also
        its API key."""
        session = Session(
            secrets.token_hex(16),
            engine=engine,
            tokenizer=tokenizer,
            params=params,
            turn_rendering=turn_rendering,
            model=model,
            max_calls=max_calls,
            max_context_tokens=max_context_tokens,
        )
        with self._lock:
            self._sessions[session.id] = session
        return session

    def close_session(self, session: Session) -> None:
        """Stop serving ``session``; its recorded calls stay readable. Idempotent."""
        session.close()
        with self._lock:
            if self._sessions.pop(session.id, None) is not None:
                self._closed[session.id] = None
                while len(self._closed) > _CLOSED_REMEMBERED:
                    self._closed.popitem(last=False)

    def base_url(self, session: Session) -> str:
        """``OPENAI_BASE_URL`` for ``session``; any API key is accepted on it."""
        return f"{self.url}/s/{session.id}/v1"

    def reward_url(self, session: Session) -> str:
        return f"{self.url}/s/{session.id}/reward"

    @property
    def num_sessions(self) -> int:
        return len(self._sessions)

    def session(self, session_id: str) -> Session:
        """The open session addressed by path."""
        return self._lookup(
            session_id, GatewayError(404, f"unknown session {session_id!r}", code="session_not_found")
        )

    def session_for_key(self, authorization: str | None) -> Session:
        """The open session addressed by an ``Authorization: Bearer <session>`` header."""
        scheme, _, key = (authorization or "").partition(" ")
        key = key.strip() if scheme.lower() == "bearer" else ""
        return self._lookup(
            key,
            GatewayError(
                401,
                "unknown session: send the session id as the API key (Authorization: Bearer <session>) "
                "or use the /s/<session>/v1 base URL",
                code="invalid_api_key",
            ),
        )

    def _lookup(self, session_id: str, unknown: GatewayError) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
            closed = session_id in self._closed
        if session is not None:
            return session
        if closed:
            raise GatewayError(503, "the session is closed", type="server_error", code="session_closed")
        raise unknown

    def limiter(self) -> anyio.CapacityLimiter:
        """Bounds the worker threads blocked in generations. Created on the server's event loop."""
        if self._limiter is None:
            self._limiter = anyio.CapacityLimiter(self.max_in_flight)
        return self._limiter


def create_app(gateway: Gateway) -> FastAPI:
    app = FastAPI(title="monorl capture gateway", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(GatewayError)
    async def gateway_error(_: Request, error: GatewayError) -> JSONResponse:
        return JSONResponse(error.body(), status_code=error.status)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        body = GatewayError(error.status_code, str(error.detail)).body()
        return JSONResponse(body, status_code=error.status_code)

    @app.exception_handler(Exception)
    async def server_error(_: Request, error: Exception) -> JSONResponse:
        body = GatewayError(500, f"{type(error).__name__}: {error}", type="server_error").body()
        return JSONResponse(body, status_code=500)

    async def chat(session: Session, request: Request) -> Response:
        chat_request = parse_chat_request(await _json_body(request))
        record = await anyio.to_thread.run_sync(session.complete, chat_request, limiter=gateway.limiter())
        payload = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "model": session.model,
            "message": record.answer,
            "finish_reason": record.finish_reason,
            "usage": usage(len(record.prompt_ids), len(record.completion.token_ids)),
        }
        if chat_request.stream:
            events = stream_events(**payload, include_usage=chat_request.include_usage)
            return Response("".join(events), media_type="text/event-stream")
        return JSONResponse(completion_body(**payload))

    def models(session: Session) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": session.model, "object": "model", "created": 0, "owned_by": "monorl"}],
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "sessions": gateway.num_sessions}

    @app.post("/s/{session_id}/v1/chat/completions")
    async def chat_by_path(session_id: str, request: Request) -> Response:
        return await chat(gateway.session(session_id), request)

    @app.post("/v1/chat/completions")
    async def chat_by_key(request: Request) -> Response:
        return await chat(gateway.session_for_key(request.headers.get("authorization")), request)

    @app.get("/s/{session_id}/v1/models")
    async def models_by_path(session_id: str) -> dict[str, Any]:
        return models(gateway.session(session_id))

    @app.get("/v1/models")
    async def models_by_key(request: Request) -> dict[str, Any]:
        return models(gateway.session_for_key(request.headers.get("authorization")))

    @app.post("/s/{session_id}/reward")
    async def post_reward(session_id: str, request: Request) -> dict[str, float]:
        session = gateway.session(session_id)
        body = await _json_body(request)
        reward = body.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise GatewayError(400, "reward must be a finite number", param="reward")
        return {"reward": session.add_reward(float(reward))}

    return app


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise GatewayError(400, f"the request body is not valid JSON: {e}") from e
    if not isinstance(body, dict):
        raise GatewayError(400, "the request body must be a JSON object")
    return body


_shared: Gateway | None = None
_shared_lock = threading.Lock()


def get_gateway() -> Gateway:
    """The process-wide gateway, started on first use."""
    global _shared
    with _shared_lock:
        if _shared is None or not _shared.running:
            _shared = Gateway().start()
        return _shared


__all__ = ["MAX_IN_FLIGHT", "Gateway", "create_app", "get_gateway"]
