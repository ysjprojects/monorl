"""What the capture gateway's wire formats share, and the OpenAI chat-completions format.

The gateway speaks three APIs — OpenAI chat completions (this module), OpenAI Responses
(:mod:`monorl.gateway.openai_responses`) and Anthropic Messages
(:mod:`monorl.gateway.anthropic_messages`). Each is a :class:`Wire`: it turns a request body into
a :class:`ChatRequest` — the chat in the one normalised form sessions compare and chat templates
render — and an answered call back into that API's response object, stream events and errors.
Whatever the API, a call is one policy turn of the session.

Normalised messages: ``content`` is a string, or — only when a user message carries images — a
list of monorl parts (``{"type": "text", "text"}``, ``{"type": "image", "image": <local file>}``);
assistant ``tool_calls`` are ``{"id", "type": "function", "function": {"name", "arguments":
<dict>}}``; ``tool`` messages keep ``tool_call_id`` and ``name``. Tool schemas are OpenAI
chat-completions function tools, the form HF chat templates take.

Torch-free and free of the HTTP stack; :mod:`monorl.gateway.session` and
:mod:`monorl.gateway.server` build on it.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from monorl.core.errors import MonorlError
from monorl.core.types import Message, ToolCall
from monorl.envs.tool_calling import TOOL_CALL_RE, parse_tool_calls
from monorl.gateway.images import ImageError, cache_image_base64, cache_image_url

if TYPE_CHECKING:
    from monorl.gateway.session import CallRecord, PreparedCall, Session

_ROLES = ("system", "user", "assistant", "tool")
_TEXT_PARTS = ("text", "input_text", "output_text")
_FENCE_RE = re.compile(r"```(?:json)?\s*.*?\s*```", re.DOTALL)

SSE_COMMENT = ": keep-alive\n\n"
"""Keep-alive of the OpenAI streams: a server-sent-events comment, which SSE clients skip."""


class GatewayError(MonorlError):
    """A request the gateway does not serve; answered with ``status`` in the error shape of the
    API it came in on (:meth:`Wire.error_body`; :meth:`body` is OpenAI's)."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.param = param
        self.code = code

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


class ContextLengthExceeded(GatewayError):
    """The prompt of a call leaves no room for a completion under the session's context budget.
    The message is OpenAI's chat-completions wording; the other wires reword it the way their
    vendor does, so that agents which detect the condition (and compact their history) do."""

    def __init__(self, prompt_tokens: int, max_context_tokens: int) -> None:
        super().__init__(
            400,
            f"This model's maximum context length is {max_context_tokens} tokens. However, your "
            f"messages resulted in {prompt_tokens} tokens. Please reduce the length of the messages.",
            param="messages",
            code="context_length_exceeded",
        )
        self.prompt_tokens = prompt_tokens
        self.max_context_tokens = max_context_tokens


@dataclass(slots=True)
class ChatRequest:
    """What the gateway honours of a request, whichever API it came in on."""

    messages: list[Message]
    """The chat, normalised (module docstring)."""
    tools: list[dict[str, Any]] | None = None
    """Function schemas in chat-completions form (the chat template renders them); ``None``
    without tools."""
    parse_tool_calls: bool = False
    """Tools were sent and the tool choice is not "none": answer with tool calls."""
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    include_usage: bool = False
    """Chat completions' ``stream_options.include_usage``: end the stream with a usage chunk."""
    id: str = ""
    """Id the answer is returned under (``chatcmpl-…``, ``resp_…``, ``msg_…``)."""
    created: int = 0
    """Unix time the request arrived."""
    instructions: bool = False
    """Responses API: the first message is the request's ``instructions`` (which a
    ``previous_response_id`` continuation does not inherit)."""
    echo: dict[str, Any] = field(default_factory=dict)
    """Request fields the API's response object repeats."""


class Wire(Protocol):
    """One API of the gateway (module docstring). Stateless; the methods taking a ``session``
    read its model name and sampling parameters only."""

    keepalive: str
    """What a stream sends while the generation is running, so idle timeouts do not fire."""

    def parse(self, body: Any, session: Session) -> ChatRequest:
        """Validate a request body. Raises :class:`GatewayError`."""
        ...

    def body(self, record: CallRecord, session: Session) -> dict[str, Any]:
        """The response object of an answered call."""
        ...

    def stream_head(self, call: PreparedCall, session: Session) -> Iterator[str]:
        """The events a stream opens with, before the generation has produced anything."""
        ...

    def stream_tail(self, record: CallRecord, session: Session) -> Iterator[str]:
        """The answered call as the API's delta events, after :meth:`stream_head`."""
        ...

    def stream_error(self, error: GatewayError, call: PreparedCall, session: Session) -> Iterator[str]:
        """How a stream that already began reports that the call failed."""
        ...

    def error_body(self, error: GatewayError) -> dict[str, Any]:
        """``error`` as the API's error response."""
        ...


# -- normalisation shared by the wires --------------------------------------------------------


def request_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def content_value(parts: Sequence[dict[str, Any]], role: Any, where: str, param: str | None) -> Any:
    """Message content from monorl parts: the text parts joined, or — with images — the parts
    themselves, which only user messages may carry."""
    if all(part["type"] == "text" for part in parts):
        return "".join(part["text"] for part in parts)
    if role != "user":
        raise GatewayError(
            400,
            f"{where}: images are accepted in user messages only",
            param=param,
            code="unsupported_content",
        )
    return list(parts)


def image_part(
    source: Any, where: str, param: str | None, *, fetch: bool, base64_data: bool = False
) -> dict[str, Any]:
    """The monorl image part for an image sent as an ``http(s)`` or base64 ``data:`` URL, or
    (``base64_data``) as bare base64. The image is kept as a local file
    (:mod:`monorl.gateway.images`)."""
    try:
        if base64_data:
            path = cache_image_base64(source, where=where)
        else:
            path = cache_image_url(source, where=where, fetch=fetch)
    except ImageError as e:
        raise GatewayError(400, str(e), param=param, code=e.code) from e
    return {"type": "image", "image": path}


def tool_call_entry(name: Any, arguments: Any, call_id: Any, where: str, param: str | None) -> dict[str, Any]:
    """One normalised assistant tool call; ``arguments`` is a JSON object or its JSON text."""
    if not isinstance(name, str) or not name:
        raise GatewayError(400, f"{where} needs a function name", param=param)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            raise GatewayError(400, f"{where}: the arguments are not valid JSON: {e.msg}", param=param) from e
    elif arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise GatewayError(400, f"{where}: the arguments must be a JSON object", param=param)
    entry: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if isinstance(call_id, str) and call_id:
        entry = {"id": call_id, **entry}
    return entry


def name_tool_results(messages: list[Message]) -> list[Message]:
    """Give every ``tool`` message without a ``name`` the name of the call it answers (chat
    templates render it). In place; returns ``messages``."""
    called: dict[str, str] = {}  # tool call id -> function name
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or ():
                if "id" in call:
                    called[call["id"]] = call["function"]["name"]  # type: ignore[typeddict-item]
        elif message.get("role") == "tool" and "name" not in message:
            name = called.get(message.get("tool_call_id", ""))
            if name:
                message["name"] = name
    return messages


def function_tool(name: Any, description: Any, parameters: Any, where: str) -> dict[str, Any]:
    """A function schema in chat-completions form from its flat fields."""
    if not isinstance(name, str) or not name:
        raise GatewayError(400, f"{where} needs a name", param="tools")
    function: dict[str, Any] = {"name": name}
    if isinstance(description, str):
        function["description"] = description
    if parameters is not None:
        if not isinstance(parameters, dict):
            raise GatewayError(400, f"{where}: the parameter schema must be a JSON object", param="tools")
        function["parameters"] = parameters
    return {"type": "function", "function": function}


def token_limit(body: dict[str, Any], keys: Sequence[str]) -> int | None:
    """The lowest of the request's completion-length fields ``keys``; ``None`` when none is set."""
    limits: list[int] = []
    for key in keys:
        value = body.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GatewayError(400, f"{key} must be a positive integer", param=key)
        limits.append(value)
    return min(limits) if limits else None


def stop_strings(raw: Any, key: str) -> list[str]:
    if raw is None:
        return []
    stops = [raw] if isinstance(raw, str) else raw
    if not isinstance(stops, list) or not all(isinstance(s, str) for s in stops):
        raise GatewayError(400, f"{key} must be a string or a list of strings", param=key)
    return [s for s in stops if s]


def split_tool_calls(text: str) -> tuple[list[ToolCall], int, str]:
    """``(calls, malformed, rest)`` of a completion: the tool calls
    :func:`monorl.envs.tool_calling.parse_tool_calls` finds in it, how many calls it found
    malformed, and the text around the calls (the whole text when there are none)."""
    parsed = parse_tool_calls(text)
    if not parsed.calls:
        return [], len(parsed.errors), text
    if TOOL_CALL_RE.search(text):
        rest = TOOL_CALL_RE.sub("", text)
    else:
        # JSON syntax: the calls are the fenced blocks that parse as calls, else the whole text
        rest = _FENCE_RE.sub(lambda m: "" if parse_tool_calls(m.group(0)).calls else m.group(0), text)
        if rest == text:
            rest = ""
    return list(parsed.calls), len(parsed.errors), rest.strip()


def split_stop(text: str, stops: Sequence[str]) -> tuple[str, str | None]:
    """``(text without a trailing stop string, that stop string)``. Engines keep the stop string
    that ended a completion; API clients expect the returned text to end before it."""
    for stop in stops:
        if stop and text.endswith(stop):
            return text[: -len(stop)], stop
    return text, None


def response_message(text: str, calls: Sequence[ToolCall]) -> Message:
    """The answer to a call as a normalised assistant message: what a request that continues
    the conversation repeats, whichever API it uses."""
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in calls
        ]
    return message  # type: ignore[return-value]


def answer_calls(response: Message) -> list[tuple[str, str, dict[str, Any]]]:
    """``(id, name, arguments)`` of the tool calls in a call's answer."""
    return [
        (call["id"], call["function"]["name"], call["function"]["arguments"])  # type: ignore[typeddict-item]
        for call in response.get("tool_calls") or ()
    ]


def sse(data: dict[str, Any], event: str | None = None) -> str:
    """One server-sent event carrying ``data`` as JSON."""
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, ensure_ascii=False)}\n\n"


# -- chat completions: requests ----------------------------------------------------------------


def _content(content: Any, role: Any, where: str, *, fetch_images: bool) -> Any:
    """Chat-completions message content: ``None`` is empty, text parts are joined, ``image_url``
    parts become monorl image parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise GatewayError(
            400, f"{where}: content must be a string, a list of parts or null", param="messages"
        )
    parts: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in _TEXT_PARTS and isinstance(part.get("text"), str):
            parts.append({"type": "text", "text": part["text"]})
        elif kind == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            parts.append(image_part(url, where, "messages", fetch=fetch_images))
        else:
            raise GatewayError(
                400,
                f"{where}: unsupported content part of type {kind!r}; text and image_url parts are accepted",
                param="messages",
                code="unsupported_content",
            )
    return content_value(parts, role, where, "messages")


def _tool_calls(raw: Any, where: str) -> list[dict[str, Any]]:
    """An assistant message's ``tool_calls`` with ``function.arguments`` decoded to a dict."""
    if not isinstance(raw, list):
        raise GatewayError(400, f"{where}: tool_calls must be a list", param="messages")
    calls: list[dict[str, Any]] = []
    for i, call in enumerate(raw):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            raise GatewayError(400, f"{where}: tool_calls[{i}] needs function.name", param="messages")
        calls.append(
            tool_call_entry(
                function.get("name"),
                function.get("arguments"),
                call.get("id"),
                f"{where}: tool_calls[{i}]",
                "messages",
            )
        )
    return calls


def normalize_messages(raw: Any, *, fetch_images: bool = True) -> list[Message]:
    """OpenAI chat messages in normalised form (module docstring): ``null`` content is empty,
    text parts are joined, ``developer`` becomes ``system``, ``tool`` messages take their
    ``name`` from the call they answer when not sent. Every other field is dropped, so a message
    the gateway answered with and the client's copy of it in a later request normalise to equal
    dicts."""
    if not isinstance(raw, list) or not raw:
        raise GatewayError(400, "messages must be a non-empty list", param="messages")
    messages: list[Message] = []
    for i, item in enumerate(raw):
        where = f"messages[{i}]"
        if not isinstance(item, dict):
            raise GatewayError(400, f"{where} must be an object", param="messages")
        role = "system" if item.get("role") == "developer" else item.get("role")
        if role not in _ROLES:
            raise GatewayError(
                400,
                f"{where}: unsupported role {item.get('role')!r}; expected one of {', '.join(_ROLES)}",
                param="messages",
            )
        message: dict[str, Any] = {
            "role": role,
            "content": _content(item.get("content"), role, where, fetch_images=fetch_images),
        }
        if role == "assistant" and item.get("tool_calls"):
            message["tool_calls"] = _tool_calls(item["tool_calls"], where)
        elif role == "tool":
            call_id = item.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                message["tool_call_id"] = call_id
            if isinstance(item.get("name"), str) and item["name"]:
                message["name"] = item["name"]
        messages.append(message)  # type: ignore[arg-type]
    return name_tool_results(messages)


def _tools(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None or raw == []:
        return None
    if not isinstance(raw, list):
        raise GatewayError(400, "tools must be a list", param="tools")
    for i, entry in enumerate(raw):
        function = entry.get("function") if isinstance(entry, dict) else None
        if (
            not isinstance(function, dict)
            or entry.get("type") != "function"
            or not isinstance(function.get("name"), str)
        ):
            raise GatewayError(
                400,
                f"tools[{i}] must be {{'type': 'function', 'function': {{'name': ...}}}}; "
                "other tool types are not supported",
                param="tools",
            )
    return raw


def parse_chat_request(body: Any, *, fetch_images: bool = True) -> ChatRequest:
    """Validate a ``POST /v1/chat/completions`` body. Sampling fields (``temperature``, ``top_p``,
    ``seed``, penalties, ...), ``model`` and everything unknown are ignored: the run's sampling
    parameters define the distribution the policy is trained on."""
    if not isinstance(body, dict):
        raise GatewayError(400, "the request body must be a JSON object")
    if body.get("n") not in (None, 1):
        raise GatewayError(
            400,
            "n must be 1: every call is one recorded policy turn; send separate requests for more samples",
            param="n",
        )
    if body.get("functions") or body.get("function_call"):
        raise GatewayError(
            400, "the legacy functions/function_call fields are not supported; use tools", param="functions"
        )
    tools = _tools(body.get("tools"))
    options = body.get("stream_options")
    return ChatRequest(
        messages=normalize_messages(body.get("messages"), fetch_images=fetch_images),
        tools=tools,
        parse_tool_calls=tools is not None and body.get("tool_choice") != "none",
        max_tokens=token_limit(body, ("max_tokens", "max_completion_tokens")),
        stop=stop_strings(body.get("stop"), "stop"),
        stream=bool(body.get("stream")),
        include_usage=isinstance(options, dict) and bool(options.get("include_usage")),
        id=request_id("chatcmpl-"),
        created=int(time.time()),
    )


# -- chat completions: responses ---------------------------------------------------------------


def assistant_message(response: Message) -> dict[str, Any]:
    """The OpenAI assistant message for a call's answer (``content`` is ``null`` when there is
    nothing but tool calls; ``function.arguments`` is a JSON string)."""
    calls = answer_calls(response)
    text = response.get("content")
    message: dict[str, Any] = {"role": "assistant", "content": text if text or not calls else None}
    if calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
            for call_id, name, arguments in calls
        ]
    return message


def usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class ChatCompletionsWire:
    """``POST /v1/chat/completions`` (:class:`Wire`)."""

    keepalive = SSE_COMMENT

    def parse(self, body: Any, session: Session) -> ChatRequest:
        return parse_chat_request(body, fetch_images=session.fetch_images)

    def body(self, record: CallRecord, session: Session) -> dict[str, Any]:
        """A ``chat.completion`` object."""
        request = record.request
        choice = {
            "index": 0,
            "message": assistant_message(record.response),
            "finish_reason": record.finish_reason,
            "logprobs": None,
        }
        return {
            "id": request.id,
            "object": "chat.completion",
            "created": request.created,
            "model": session.model,
            "choices": [choice],
            "usage": usage(len(record.prompt_ids), len(record.completion.token_ids)),
        }

    def stream_head(self, call: PreparedCall, session: Session) -> Iterator[str]:
        yield self._delta(call.request, session, {"role": "assistant", "content": ""})

    def stream_tail(self, record: CallRecord, session: Session) -> Iterator[str]:
        """``chat.completion.chunk`` events after the role chunk: the content, the tool calls,
        the finish reason, optionally the usage, then ``[DONE]``."""
        request = record.request
        message = assistant_message(record.response)
        if message["content"]:
            yield self._delta(request, session, {"content": message["content"]})
        if message.get("tool_calls"):
            calls = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
            yield self._delta(request, session, {"tool_calls": calls})
        yield self._delta(request, session, {}, record.finish_reason)
        if request.include_usage:
            tokens = usage(len(record.prompt_ids), len(record.completion.token_ids))
            yield sse(self._chunk(request, session, [], usage=tokens))
        yield "data: [DONE]\n\n"

    def stream_error(self, error: GatewayError, call: PreparedCall, session: Session) -> Iterator[str]:
        yield sse(self.error_body(error))

    def error_body(self, error: GatewayError) -> dict[str, Any]:
        return error.body()

    @staticmethod
    def _chunk(
        request: ChatRequest, session: Session, choices: list[dict[str, Any]], **extra: Any
    ) -> dict[str, Any]:
        return {
            "id": request.id,
            "object": "chat.completion.chunk",
            "created": request.created,
            "model": session.model,
            "choices": choices,
            **extra,
        }

    def _delta(
        self, request: ChatRequest, session: Session, payload: dict[str, Any], finish: str | None = None
    ) -> str:
        choice = {"index": 0, "delta": payload, "finish_reason": finish, "logprobs": None}
        return sse(self._chunk(request, session, [choice]))


__all__ = [
    "SSE_COMMENT",
    "ChatCompletionsWire",
    "ChatRequest",
    "ContextLengthExceeded",
    "GatewayError",
    "Wire",
    "answer_calls",
    "assistant_message",
    "content_value",
    "function_tool",
    "image_part",
    "name_tool_results",
    "normalize_messages",
    "parse_chat_request",
    "request_id",
    "response_message",
    "split_stop",
    "split_tool_calls",
    "sse",
    "stop_strings",
    "token_limit",
    "tool_call_entry",
    "usage",
]
