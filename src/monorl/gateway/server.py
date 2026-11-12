"""The capture gateway's HTTP server: OpenAI- and Anthropic-compatible endpoints served from the
policy being trained.

:class:`Gateway` holds the session table and runs a FastAPI app with uvicorn in a daemon thread,
bound to localhost on a free port unless told otherwise. :func:`get_gateway` returns the
process-wide instance for a bind address, started on first use and shared by every episode. Every
route but ``/health`` needs the session id, in the path or as the API key (``Authorization:
Bearer <session>`` or ``x-api-key: <session>``)::

    POST /s/{session}/v1/chat/completions        POST /v1/chat/completions        OpenAI chat completions
    POST /s/{session}/v1/responses               POST /v1/responses               OpenAI Responses
    POST /s/{session}/v1/messages                POST /v1/messages                Anthropic Messages
    POST /s/{session}/v1/messages/count_tokens   POST /v1/messages/count_tokens
    GET  /s/{session}/v1/models                  GET  /v1/models
    POST /s/{session}/reward   {"reward": float}   -> {"reward": <total posted so far>}
    GET  /health

The three APIs are :class:`~monorl.gateway.protocol.Wire` objects over one session model. Errors
have the shape of the API the request came in on: 400 for a request the gateway does not serve,
401/404 for an unknown session, 503 once the session is closed or its generation was cancelled.

A completion blocks in ``engine.generate`` until its batch is served, so the calls run on worker
threads under the gateway's own limiter (:data:`MAX_IN_FLIGHT`) instead of the event loop's
default 40-thread pool: the calls of all parallel episodes are in flight together and the run's
``BatchingEngine`` coalesces them.

A streamed call is admitted and given its prompt first (so everything it can be refused for is
still a plain HTTP error), then the stream opens at once and sends the API's keep-alive every
:data:`KEEPALIVE_S` seconds until the generation is done; a client that hangs up does not stop
the generation, as for a call that is not streamed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from monorl.core.types import SamplingParams
from monorl.gateway.anthropic_messages import MessagesWire
from monorl.gateway.openai_responses import ResponsesWire
from monorl.gateway.protocol import ChatCompletionsWire, GatewayError, Wire
from monorl.gateway.session import CallRecord, PreparedCall, Session

if TYPE_CHECKING:
    from monorl.core.interfaces import RolloutEngine

log = logging.getLogger("monorl.gateway")

MAX_IN_FLIGHT = 512
"""Completions one gateway serves at once; further calls wait for a free slot."""

KEEPALIVE_S = 10.0
"""Seconds between the keep-alives of a stream whose generation is still running."""

_CLOSED_REMEMBERED = 4096
"""How many closed session ids keep answering 503 (instead of "unknown session")."""

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

CHAT_COMPLETIONS: Wire = ChatCompletionsWire()
RESPONSES: Wire = ResponsesWire()
MESSAGES: Wire = MessagesWire()


class Gateway:
    """Session table plus the HTTP server in front of it (module docstring)."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_in_flight: int = MAX_IN_FLIGHT,
        keepalive_s: float = KEEPALIVE_S,
    ) -> None:
        import uvicorn

        self.host = host
        self.port = port
        self.max_in_flight = max(1, int(max_in_flight))
        self.keepalive_s = float(keepalive_s)
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
        """Where this machine reaches the gateway."""
        return self._origin(None)

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
        processor: Any | None = None,
        fetch_images: bool = True,
    ) -> Session:
        """A new session (one episode) served with ``engine`` under ``params``
        (:class:`~monorl.gateway.session.Session` describes the options). Its id is also its API
        key."""
        session = Session(
            secrets.token_hex(16),
            engine=engine,
            tokenizer=tokenizer,
            params=params,
            turn_rendering=turn_rendering,
            model=model,
            max_calls=max_calls,
            max_context_tokens=max_context_tokens,
            processor=processor,
            fetch_images=fetch_images,
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

    def _origin(self, host: str | None) -> str:
        """``http://host:port`` of the gateway as a client names it: ``host``, else the bind
        address, with loopback standing in for "every interface"."""
        name = host or ("127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host)
        if ":" in name and not name.startswith("["):  # an IPv6 literal
            name = f"[{name}]"
        return f"http://{name}:{self.port}"

    def session_url(self, session: Session, host: str | None = None) -> str:
        """Root of ``session``'s routes. ``host`` replaces the bind address in the URL for
        clients that reach this machine under another name (``host.docker.internal`` from a
        container); a gateway bound to every interface is addressed through loopback by default."""
        return f"{self._origin(host)}/s/{session.id}"

    def base_url(self, session: Session, host: str | None = None) -> str:
        """``OPENAI_BASE_URL`` for ``session`` (chat completions and Responses); any API key is
        accepted on it."""
        return f"{self.session_url(session, host)}/v1"

    def anthropic_base_url(self, session: Session, host: str | None = None) -> str:
        """``ANTHROPIC_BASE_URL`` for ``session``: Anthropic clients add ``/v1/messages``
        themselves. Any API key is accepted on it."""
        return self.session_url(session, host)

    def reward_url(self, session: Session, host: str | None = None) -> str:
        return f"{self.session_url(session, host)}/reward"

    @property
    def num_sessions(self) -> int:
        return len(self._sessions)

    def session(self, session_id: str) -> Session:
        """The open session addressed by path."""
        session = self._lookup(session_id)
        if session is None:
            raise GatewayError(404, f"unknown session {session_id!r}", code="session_not_found")
        return session

    def session_for_key(self, authorization: str | None, api_key: str | None = None) -> Session:
        """The open session addressed by its id as the API key: an ``Authorization: Bearer
        <session>`` header (OpenAI clients, Anthropic auth tokens) or ``x-api-key: <session>``
        (Anthropic API keys)."""
        scheme, _, token = (authorization or "").partition(" ")
        for key in (token.strip() if scheme.lower() == "bearer" else "", (api_key or "").strip()):
            session = self._lookup(key) if key else None
            if session is not None:
                return session
        raise GatewayError(
            401,
            "unknown session: send the session id as the API key (Authorization: Bearer <session> or "
            "x-api-key: <session>) or use the session's /s/<session> base URL",
            code="invalid_api_key",
        )

    def _lookup(self, session_id: str) -> Session | None:
        """The open session ``session_id``; ``None`` when there never was one. A closed one
        answers 503."""
        with self._lock:
            session = self._sessions.get(session_id)
            closed = session_id in self._closed
        if session is None and closed:
            raise GatewayError(503, "the session is closed", type="server_error", code="session_closed")
        return session

    def limiter(self) -> anyio.CapacityLimiter:
        """Bounds the worker threads blocked in generations. Created on the server's event loop."""
        if self._limiter is None:
            self._limiter = anyio.CapacityLimiter(self.max_in_flight)
        return self._limiter


def _wire_of(path: str) -> Wire:
    """The API a request path belongs to, for the shape of its errors."""
    if "/v1/messages" in path:
        return MESSAGES
    return RESPONSES if path.endswith("/v1/responses") else CHAT_COMPLETIONS


def create_app(gateway: Gateway) -> FastAPI:
    app = FastAPI(title="monorl capture gateway", docs_url=None, redoc_url=None, openapi_url=None)

    def failed(request: Request, error: GatewayError) -> JSONResponse:
        return JSONResponse(_wire_of(request.url.path).error_body(error), status_code=error.status)

    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, error: GatewayError) -> JSONResponse:
        return failed(request, error)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        return failed(request, GatewayError(error.status_code, str(error.detail)))

    @app.exception_handler(Exception)
    async def server_error(request: Request, error: Exception) -> JSONResponse:
        return failed(request, GatewayError(500, f"{type(error).__name__}: {error}", type="server_error"))

    async def in_thread(function: Callable[[], Any]) -> Any:
        return await anyio.to_thread.run_sync(function, limiter=gateway.limiter())

    def call(wire: Wire) -> Callable[[Session, Request], Awaitable[Response]]:
        """Handler of the route on which ``wire`` answers calls with the policy."""

        async def handle(session: Session, request: Request) -> Response:
            body = await _json_body(request)
            if not body.get("stream"):
                record = await in_thread(lambda: session.complete(wire.parse(body, session)))
                return JSONResponse(wire.body(record, session))
            prepared = await in_thread(lambda: session.prepare(wire.parse(body, session)))
            # Its own task: the generation is under way before the client reads the first event
            # and is recorded whether or not the client stays for the stream.
            generation = asyncio.ensure_future(in_thread(lambda: session.finish(prepared)))
            generation.add_done_callback(_retrieve)
            events = _stream(wire, session, prepared, generation, gateway.keepalive_s)
            return StreamingResponse(events, media_type="text/event-stream", headers=_SSE_HEADERS)

        return handle

    async def count_tokens(session: Session, request: Request) -> Response:
        body = await _json_body(request)
        tokens = await in_thread(lambda: session.count_tokens(MESSAGES.parse(body, session)))
        return JSONResponse({"input_tokens": tokens})

    async def models(session: Session, request: Request) -> Response:
        model = {"id": session.model, "object": "model", "created": 0, "owned_by": "monorl"}
        return JSONResponse({"object": "list", "data": [model]})

    def route(
        path: str, handler: Callable[[Session, Request], Awaitable[Response]], method: str = "POST"
    ) -> None:
        """Serve ``handler`` for the session in the path and for the session given as API key."""

        async def by_path(session_id: str, request: Request) -> Response:
            return await handler(gateway.session(session_id), request)

        async def by_key(request: Request) -> Response:
            headers = request.headers
            session = gateway.session_for_key(headers.get("authorization"), headers.get("x-api-key"))
            return await handler(session, request)

        app.add_api_route(f"/s/{{session_id}}{path}", by_path, methods=[method], response_model=None)
        app.add_api_route(path, by_key, methods=[method], response_model=None)

    route("/v1/chat/completions", call(CHAT_COMPLETIONS))
    route("/v1/responses", call(RESPONSES))
    route("/v1/messages", call(MESSAGES))
    route("/v1/messages/count_tokens", count_tokens)
    route("/v1/models", models, method="GET")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "sessions": gateway.num_sessions}

    @app.post("/s/{session_id}/reward")
    async def post_reward(session_id: str, request: Request) -> dict[str, float]:
        session = gateway.session(session_id)
        body = await _json_body(request)
        reward = body.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise GatewayError(400, "reward must be a finite number", param="reward")
        return {"reward": session.add_reward(float(reward))}

    return app


async def _stream(
    wire: Wire,
    session: Session,
    call: PreparedCall,
    generation: asyncio.Future[CallRecord],
    keepalive_s: float,
) -> AsyncIterator[str]:
    """Server-sent events of one streamed call: the API's opening events at once, its keep-alive
    while ``generation`` runs, then the answer as its delta events — or its error event."""
    for event in wire.stream_head(call, session):
        yield event
    while not generation.done():
        await asyncio.wait({generation}, timeout=keepalive_s)
        if not generation.done():
            yield wire.keepalive
    try:
        events = wire.stream_tail(generation.result(), session)
    except GatewayError as error:
        events = wire.stream_error(error, call, session)
    except Exception as error:  # noqa: BLE001 - the stream is open: report it in-band
        unexpected = GatewayError(500, f"{type(error).__name__}: {error}", type="server_error")
        events = wire.stream_error(unexpected, call, session)
    for event in events:
        yield event


def _retrieve(generation: asyncio.Future[CallRecord]) -> None:
    """Take a streamed generation's error when no stream is left to report it to."""
    if not generation.cancelled():
        generation.exception()


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = json.loads(await request.body())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise GatewayError(400, f"the request body is not valid JSON: {e}") from e
    if not isinstance(body, dict):
        raise GatewayError(400, "the request body must be a JSON object")
    return body


_shared: dict[tuple[str, int], Gateway] = {}
_shared_lock = threading.Lock()


def get_gateway(*, host: str = "127.0.0.1", port: int = 0) -> Gateway:
    """The process-wide gateway bound to ``host`` (``port`` 0: any free port), started on first
    use. Bind to another address than localhost only for agents that run elsewhere (containers):
    the unguessable session id is then the only credential."""
    with _shared_lock:
        gateway = _shared.get((host, port))
        if gateway is None or not gateway.running:
            gateway = _shared[(host, port)] = Gateway(host=host, port=port).start()
        return gateway


__all__ = ["KEEPALIVE_S", "MAX_IN_FLIGHT", "Gateway", "create_app", "get_gateway"]
