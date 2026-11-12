"""Anthropic Messages wire format of the capture gateway (``POST /v1/messages`` and
``POST /v1/messages/count_tokens``).

Written against the API reference and the ``anthropic`` SDK's type definitions
(``anthropic.types``). A request is turned into the same normalised chat every API of the gateway
uses (:mod:`monorl.gateway.protocol`):

* ``system`` (a string or text blocks) is a leading ``system`` message;
* a ``user`` message's ``text`` and ``image`` blocks are one ``user`` message, each of its
  ``tool_result`` blocks a ``tool`` message, in the order sent;
* an ``assistant`` message's ``text`` blocks are its content, its ``tool_use`` blocks its tool
  calls;
* tools (``name`` / ``description`` / ``input_schema``) become chat-completions function
  schemas, the form chat templates render.

The answer is a ``message`` whose content is a ``text`` block and one ``tool_use`` block per
parsed tool call; sent back in an ``assistant`` message it normalises to the recorded answer, so
the next request continues the call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from monorl.core.types import Message
from monorl.gateway.protocol import (
    ChatRequest,
    ContextLengthExceeded,
    GatewayError,
    answer_calls,
    content_value,
    function_tool,
    image_part,
    name_tool_results,
    request_id,
    sse,
    stop_strings,
    token_limit,
    tool_call_entry,
)

if TYPE_CHECKING:
    from monorl.gateway.session import CallRecord, PreparedCall, Session

_ERROR_TYPES = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
}

PING = sse({"type": "ping"}, "ping")
"""Keep-alive of a Messages stream."""


# -- requests ---------------------------------------------------------------------------------


def _text_blocks(content: Any, where: str, what: str) -> str:
    """Content that may only be text: a string, or ``text`` blocks joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        for block in content
    ):
        return "".join(block["text"] for block in content)
    raise GatewayError(
        400,
        f"{where}: {what} must be a string or a list of text blocks",
        param="messages",
        code="unsupported_content",
    )


def _image(source: Any, where: str, *, fetch_images: bool) -> dict[str, Any]:
    kind = source.get("type") if isinstance(source, dict) else None
    if kind == "base64":
        return image_part(source.get("data"), where, "messages", fetch=fetch_images, base64_data=True)
    if kind == "url":
        return image_part(source.get("url"), where, "messages", fetch=fetch_images)
    raise GatewayError(
        400,
        f"{where}: unsupported image source of type {kind!r}; base64 and url sources are accepted",
        param="messages",
        code="unsupported_content",
    )


def _user(content: Any, where: str, *, fetch_images: bool) -> list[Message]:
    """A ``user`` message as normalised messages: its text and image blocks as one ``user``
    message, every ``tool_result`` block as a ``tool`` message, in the order sent."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        raise GatewayError(
            400, f"{where}: content must be a string or a list of content blocks", param="messages"
        )
    messages: list[Any] = []
    parts: list[dict[str, Any]] = []
    for j, block in enumerate(content):
        here = f"{where}.content.{j}"
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text" and isinstance(block.get("text"), str):
            parts.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            parts.append(_image(block.get("source"), here, fetch_images=fetch_images))
        elif kind == "tool_result":
            if not isinstance(block.get("tool_use_id"), str) or not block["tool_use_id"]:
                raise GatewayError(400, f"{here}: a tool_result needs tool_use_id", param="messages")
            if parts:
                messages.append({"role": "user", "content": content_value(parts, "user", where, "messages")})
                parts = []
            result = block.get("content")
            text = "" if result is None else _text_blocks(result, here, "a tool result")
            messages.append({"role": "tool", "content": text, "tool_call_id": block["tool_use_id"]})
        else:
            raise GatewayError(
                400,
                f"{here}: unsupported content block of type {kind!r} in a user message; text, image "
                "and tool_result blocks are accepted",
                param="messages",
                code="unsupported_content",
            )
    if parts or not messages:
        messages.append({"role": "user", "content": content_value(parts, "user", where, "messages")})
    return messages


def _assistant(content: Any, where: str) -> Message:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        raise GatewayError(
            400, f"{where}: content must be a string or a list of content blocks", param="messages"
        )
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    for j, block in enumerate(content):
        here = f"{where}.content.{j}"
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif kind == "tool_use":
            if not isinstance(block.get("id"), str) or not block["id"]:
                raise GatewayError(400, f"{here}: a tool_use needs an id", param="messages")
            calls.append(
                tool_call_entry(block.get("name"), block.get("input"), block["id"], here, "messages")
            )
        else:
            raise GatewayError(
                400,
                f"{here}: unsupported content block of type {kind!r} in an assistant message; text "
                "and tool_use blocks are accepted",
                param="messages",
                code="unsupported_content",
            )
    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
    if calls:
        message["tool_calls"] = calls
    return message  # type: ignore[return-value]


def _tools(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None or raw == []:
        return None
    if not isinstance(raw, list):
        raise GatewayError(400, "tools must be a list", param="tools")
    tools: list[dict[str, Any]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or entry.get("type") not in (None, "custom"):
            kind = entry.get("type") if isinstance(entry, dict) else None
            raise GatewayError(
                400,
                f"tools.{i}: only client tools (name, description, input_schema) are supported, not "
                f"{kind!r}: the policy being trained has no server tools",
                param="tools",
            )
        tools.append(
            function_tool(
                entry.get("name"), entry.get("description"), entry.get("input_schema"), f"tools.{i}"
            )
        )
    return tools


def parse_messages_request(body: Any, *, fetch_images: bool = True) -> ChatRequest:
    """Validate a ``POST /v1/messages`` (or ``count_tokens``) body. Sampling fields
    (``temperature``, ``top_p``, ``top_k``), ``thinking``, ``cache_control`` markers,
    ``metadata``, ``model`` and everything unknown are ignored: the run's sampling parameters
    define the distribution the policy is trained on."""
    if not isinstance(body, dict):
        raise GatewayError(400, "the request body must be a JSON object")
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise GatewayError(400, "messages: at least one message is required", param="messages")
    messages: list[Message] = []
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _text_blocks(system, "system", "the system prompt")})
    for i, item in enumerate(raw):
        where = f"messages.{i}"
        role = item.get("role") if isinstance(item, dict) else None
        if role == "user":
            messages += _user(item.get("content"), where, fetch_images=fetch_images)
        elif role == "assistant":
            messages.append(_assistant(item.get("content"), where))
        elif role == "system":
            text = _text_blocks(item.get("content"), where, "a system message")
            messages.append({"role": "system", "content": text})
        else:
            raise GatewayError(
                400, f"{where}: unsupported role {role!r}; expected user or assistant", param="messages"
            )
    if messages[-1]["role"] == "assistant":
        raise GatewayError(
            400,
            "the conversation must end with a user message: continuing a partial assistant message "
            "(prefill) is not supported",
            param="messages",
        )
    tools = _tools(body.get("tools"))
    choice = body.get("tool_choice")
    return ChatRequest(
        messages=name_tool_results(messages),
        tools=tools,
        parse_tool_calls=tools is not None
        and not (isinstance(choice, dict) and choice.get("type") == "none"),
        max_tokens=token_limit(body, ("max_tokens",)),
        stop=stop_strings(body.get("stop_sequences"), "stop_sequences"),
        stream=bool(body.get("stream")),
        id=request_id("msg_"),
        created=int(time.time()),
    )


# -- responses --------------------------------------------------------------------------------


def content_blocks(record: CallRecord) -> list[dict[str, Any]]:
    """The content of an answered call: a ``text`` block (left out when there is nothing but
    tool calls) and one ``tool_use`` block per call."""
    text = record.response.get("content") or ""
    calls = answer_calls(record.response)
    blocks: list[dict[str, Any]] = []
    if text or not calls:
        blocks.append({"type": "text", "text": text})
    for call_id, name, arguments in calls:
        blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": arguments})
    return blocks


def stop_reason(record: CallRecord) -> str:
    if record.finish_reason == "tool_calls":
        return "tool_use"
    if record.finish_reason == "length":
        return "max_tokens"
    return "end_turn" if record.stop_sequence is None else "stop_sequence"


def _message(
    request: ChatRequest,
    session: Session,
    *,
    content: list[dict[str, Any]],
    reason: str | None,
    stop_sequence: str | None,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    return {
        "id": request.id,
        "type": "message",
        "role": "assistant",
        "model": session.model,
        "content": content,
        "stop_reason": reason,
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": output_tokens,
        },
    }


class MessagesWire:
    """``POST /v1/messages`` (:class:`~monorl.gateway.protocol.Wire`)."""

    keepalive = PING

    def parse(self, body: Any, session: Session) -> ChatRequest:
        return parse_messages_request(body, fetch_images=session.fetch_images)

    def body(self, record: CallRecord, session: Session) -> dict[str, Any]:
        reason = stop_reason(record)
        return _message(
            record.request,
            session,
            content=content_blocks(record),
            reason=reason,
            stop_sequence=record.stop_sequence if reason == "stop_sequence" else None,
            input_tokens=len(record.prompt_ids),
            output_tokens=len(record.completion.token_ids),
        )

    def stream_head(self, call: PreparedCall, session: Session) -> Iterator[str]:
        message = _message(
            call.request,
            session,
            content=[],
            reason=None,
            stop_sequence=None,
            input_tokens=len(call.prompt_ids),
            output_tokens=0,
        )
        yield sse({"type": "message_start", "message": message}, "message_start")

    def stream_tail(self, record: CallRecord, session: Session) -> Iterator[str]:
        """The finished answer block by block — a text block as one ``text_delta``, a tool use as
        one ``input_json_delta`` — then ``message_delta`` with the stop reason and the usage, and
        ``message_stop``."""
        message = self.body(record, session)
        for index, block in enumerate(message["content"]):
            if block["type"] == "text":
                opened, delta = {"type": "text", "text": ""}, {"type": "text_delta", "text": block["text"]}
                empty = not block["text"]
            else:
                opened = {**block, "input": {}}
                arguments = json.dumps(block["input"], ensure_ascii=False)
                delta, empty = {"type": "input_json_delta", "partial_json": arguments}, False
            start = {"type": "content_block_start", "index": index, "content_block": opened}
            yield sse(start, "content_block_start")
            if not empty:
                yield sse(
                    {"type": "content_block_delta", "index": index, "delta": delta}, "content_block_delta"
                )
            yield sse({"type": "content_block_stop", "index": index}, "content_block_stop")
        end = {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": message["stop_sequence"]},
            "usage": message["usage"],
        }
        yield sse(end, "message_delta")
        yield sse({"type": "message_stop"}, "message_stop")

    def stream_error(self, error: GatewayError, call: PreparedCall, session: Session) -> Iterator[str]:
        yield sse(self.error_body(error), "error")

    def error_body(self, error: GatewayError) -> dict[str, Any]:
        """Anthropic's error shape. A prompt over the context budget is reported in the words
        the Messages API uses for it."""
        message = error.message
        if isinstance(error, ContextLengthExceeded):
            message = f"prompt is too long: {error.prompt_tokens} tokens > {error.max_context_tokens} maximum"
        default = "invalid_request_error" if error.status < 500 else "api_error"
        kind = _ERROR_TYPES.get(error.status, default)
        return {"type": "error", "error": {"type": kind, "message": message}}


__all__ = ["PING", "MessagesWire", "content_blocks", "parse_messages_request", "stop_reason"]
