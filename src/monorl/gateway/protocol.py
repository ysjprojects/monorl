"""OpenAI chat-completions wire format of the capture gateway: request validation, message
normalisation, tool-call extraction, response and SSE bodies, and the error shape.

Torch-free and free of the HTTP stack; :mod:`monorl.gateway.session` and
:mod:`monorl.gateway.server` build on it.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from monorl.core.errors import MonorlError
from monorl.core.types import Message, ToolCall
from monorl.envs.tool_calling import TOOL_CALL_RE, parse_tool_calls

_ROLES = ("system", "user", "assistant", "tool")
_TEXT_PARTS = ("text", "input_text", "output_text")
_IMAGE_PARTS = ("image_url", "image", "input_image")
_FENCE_RE = re.compile(r"```(?:json)?\s*.*?\s*```", re.DOTALL)


class GatewayError(MonorlError):
    """A request the gateway does not serve; answered as OpenAI's error JSON with ``status``."""

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


@dataclass(slots=True)
class ChatRequest:
    """What the gateway honours of a chat-completions request body."""

    messages: list[Message]
    """The chat, normalised by :func:`normalize_messages`."""
    tools: list[dict[str, Any]] | None = None
    """Function schemas as sent (the chat template renders them); ``None`` without tools."""
    parse_tool_calls: bool = False
    """Tools were sent and ``tool_choice`` is not ``"none"``: answer with ``tool_calls``."""
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    include_usage: bool = False
    """``stream_options.include_usage``: end the stream with a usage chunk."""


# -- requests ---------------------------------------------------------------------------------


def _text(content: Any, where: str) -> str:
    """Message content as one string: ``None`` is empty, text parts are joined."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise GatewayError(
            400, f"{where}: content must be a string, a list of parts or null", param="messages"
        )
    texts: list[str] = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in _TEXT_PARTS and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif kind in _IMAGE_PARTS:
            raise GatewayError(
                400,
                f"{where}: image parts are not supported; the gateway serves text-only chats",
                param="messages",
                code="unsupported_content",
            )
        else:
            raise GatewayError(
                400,
                f"{where}: unsupported content part of type {kind!r}; only text parts are accepted",
                param="messages",
                code="unsupported_content",
            )
    return "".join(texts)


def _tool_calls(raw: Any, where: str) -> list[dict[str, Any]]:
    """An assistant message's ``tool_calls`` with ``function.arguments`` decoded to a dict."""
    if not isinstance(raw, list):
        raise GatewayError(400, f"{where}: tool_calls must be a list", param="messages")
    calls: list[dict[str, Any]] = []
    for i, call in enumerate(raw):
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            raise GatewayError(400, f"{where}: tool_calls[{i}] needs function.name", param="messages")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as e:
                raise GatewayError(
                    400,
                    f"{where}: arguments of tool_calls[{i}] are not valid JSON: {e.msg}",
                    param="messages",
                ) from e
        elif arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise GatewayError(
                400, f"{where}: arguments of tool_calls[{i}] must be a JSON object", param="messages"
            )
        entry: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": arguments}}
        if isinstance(call.get("id"), str) and call["id"]:
            entry = {"id": call["id"], **entry}
        calls.append(entry)
    return calls


def normalize_messages(raw: Any) -> list[Message]:
    """OpenAI chat messages in the form HF chat templates render.

    ``content`` is always a string (``null`` is empty, text parts are joined; other parts are
    rejected), ``developer`` becomes ``system``, assistant ``tool_calls`` are
    ``{"id", "type": "function", "function": {"name", "arguments": <dict>}}``, ``tool`` messages
    keep ``tool_call_id`` and ``name`` (taken from the call they answer when not sent). Every
    other field is dropped, so a message the gateway answered with and the client's copy of it in
    a later request normalise to equal dicts.
    """
    if not isinstance(raw, list) or not raw:
        raise GatewayError(400, "messages must be a non-empty list", param="messages")
    messages: list[Message] = []
    called: dict[str, str] = {}  # tool call id -> function name
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
        message: dict[str, Any] = {"role": role, "content": _text(item.get("content"), where)}
        if role == "assistant" and item.get("tool_calls"):
            message["tool_calls"] = _tool_calls(item["tool_calls"], where)
            called.update((c["id"], c["function"]["name"]) for c in message["tool_calls"] if "id" in c)
        elif role == "tool":
            call_id = item.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                message["tool_call_id"] = call_id
            name = item.get("name")
            if not isinstance(name, str) or not name:
                name = called.get(call_id) if isinstance(call_id, str) else None
            if name:
                message["name"] = name
        messages.append(message)  # type: ignore[arg-type]
    return messages


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


def _max_tokens(body: dict[str, Any]) -> int | None:
    limits: list[int] = []
    for key in ("max_tokens", "max_completion_tokens"):
        value = body.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GatewayError(400, f"{key} must be a positive integer", param=key)
        limits.append(value)
    return min(limits) if limits else None


def _stop(raw: Any) -> list[str]:
    if raw is None:
        return []
    stops = [raw] if isinstance(raw, str) else raw
    if not isinstance(stops, list) or not all(isinstance(s, str) for s in stops):
        raise GatewayError(400, "stop must be a string or a list of strings", param="stop")
    return [s for s in stops if s]


def parse_chat_request(body: Any) -> ChatRequest:
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
        messages=normalize_messages(body.get("messages")),
        tools=tools,
        parse_tool_calls=tools is not None and body.get("tool_choice") != "none",
        max_tokens=_max_tokens(body),
        stop=_stop(body.get("stop")),
        stream=bool(body.get("stream")),
        include_usage=isinstance(options, dict) and bool(options.get("include_usage")),
    )


# -- responses --------------------------------------------------------------------------------


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


def strip_stop(text: str, stops: Sequence[str]) -> str:
    """``text`` without a trailing stop string. Engines keep the stop string that ended a
    completion; OpenAI clients expect the returned text to end before it."""
    for stop in stops:
        if stop and text.endswith(stop):
            return text[: -len(stop)]
    return text


def assistant_message(content: str, calls: Sequence[ToolCall]) -> dict[str, Any]:
    """The OpenAI assistant message for ``content`` and parsed ``calls`` (``content`` is ``null``
    when there is nothing but tool calls; ``function.arguments`` is a JSON string)."""
    message: dict[str, Any] = {"role": "assistant", "content": content if content or not calls else None}
    if calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                },
            }
            for call in calls
        ]
    return message


def usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def completion_body(
    *, id: str, model: str, message: dict[str, Any], finish_reason: str, usage: dict[str, int]
) -> dict[str, Any]:
    """A ``chat.completion`` response object."""
    return {
        "id": id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": None}],
        "usage": usage,
    }


def stream_events(
    *,
    id: str,
    model: str,
    message: dict[str, Any],
    finish_reason: str,
    usage: dict[str, int],
    include_usage: bool,
) -> Iterator[str]:
    """The finished completion as server-sent ``chat.completion.chunk`` events: the role, the
    content, the tool calls, the finish reason, optionally the usage, then ``[DONE]``."""
    created = int(time.time())

    def event(choices: list[dict[str, Any]], **extra: Any) -> str:
        chunk = {
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": choices,
            **extra,
        }
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    def delta(payload: dict[str, Any], finish: str | None = None) -> str:
        return event([{"index": 0, "delta": payload, "finish_reason": finish, "logprobs": None}])

    yield delta({"role": "assistant", "content": ""})
    if message.get("content"):
        yield delta({"content": message["content"]})
    if message.get("tool_calls"):
        yield delta({"tool_calls": [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]})
    yield delta({}, finish_reason)
    if include_usage:
        yield event([], usage=usage)
    yield "data: [DONE]\n\n"


__all__ = [
    "ChatRequest",
    "GatewayError",
    "assistant_message",
    "completion_body",
    "normalize_messages",
    "parse_chat_request",
    "split_tool_calls",
    "stream_events",
    "strip_stop",
    "usage",
]
