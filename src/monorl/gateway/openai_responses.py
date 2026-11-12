"""OpenAI Responses wire format of the capture gateway (``POST /v1/responses``).

Written against the API reference and the ``openai`` SDK's type definitions
(``openai.types.responses``). A request is turned into the same normalised chat every API of the
gateway uses (:mod:`monorl.gateway.protocol`):

* ``instructions`` is a leading ``system`` message; ``input`` is one ``user`` message (a string)
  or a list of items — messages (``input_text`` / ``output_text`` / ``input_image`` parts),
  ``function_call`` items (they become the tool calls of the assistant message they follow) and
  ``function_call_output`` items (``tool`` messages);
* ``previous_response_id`` names an earlier call of the session: the chat is that call's chat,
  its answer and then ``input``. As with OpenAI, the earlier call's ``instructions`` are not
  inherited — a request that repeats them continues the call token for token, one that changes
  or drops them starts a new context;
* function tools (flat ``name`` / ``description`` / ``parameters``) become chat-completions
  function schemas, the form chat templates render.

The answer is a ``response`` object whose ``output`` holds a ``message`` item (the text) and one
``function_call`` item per parsed tool call; sent back as input items they normalise to the
recorded answer, so the next request continues the call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from monorl.core.types import Message
from monorl.gateway.protocol import (
    SSE_COMMENT,
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
    token_limit,
    tool_call_entry,
)

if TYPE_CHECKING:
    from monorl.gateway.session import CallRecord, PreparedCall, Session

_MESSAGE_ROLES = {"user": "user", "assistant": "assistant", "system": "system", "developer": "system"}
_TEXT_PARTS = ("input_text", "output_text")
_UNSUPPORTED = {
    "conversation": "conversations are not stored; send the history as input or use previous_response_id",
    "prompt": "stored prompt templates are not available",
}


# -- requests ---------------------------------------------------------------------------------


def _content(content: Any, role: str, where: str, *, fetch_images: bool) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise GatewayError(400, f"{where}: content must be a string or a list of parts", param="input")
    parts: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind in _TEXT_PARTS and isinstance(part.get("text"), str):
            parts.append({"type": "text", "text": part["text"]})
        elif kind == "input_image":
            if part.get("image_url") is None:
                raise GatewayError(
                    400,
                    f"{where}: an input_image needs image_url; uploaded files (file_id) are not available",
                    param="input",
                    code="unsupported_content",
                )
            parts.append(image_part(part["image_url"], where, "input", fetch=fetch_images))
        else:
            raise GatewayError(
                400,
                f"{where}: unsupported content part of type {kind!r}; input_text, output_text and "
                "input_image parts are accepted",
                param="input",
                code="unsupported_content",
            )
    return content_value(parts, role, where, "input")


def _output_text(output: Any, where: str) -> str:
    """The text of a ``function_call_output``: a string, or text parts joined."""
    if isinstance(output, str):
        return output
    if isinstance(output, list) and all(
        isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str)
        for part in output
    ):
        return "".join(part["text"] for part in output)
    raise GatewayError(
        400,
        f"{where}: output must be a string or a list of input_text parts; tool results are text",
        param="input",
        code="unsupported_content",
    )


def _input_messages(raw: Any, *, fetch_images: bool) -> list[Message]:
    """The ``input`` of a request as normalised messages."""
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    if not isinstance(raw, list) or not raw:
        raise GatewayError(400, "input must be a string or a non-empty list of items", param="input")
    messages: list[Any] = []
    calling: dict[str, Any] | None = None  # the assistant message function_call items add to
    for i, item in enumerate(raw):
        where = f"input[{i}]"
        if not isinstance(item, dict):
            raise GatewayError(400, f"{where} must be an object", param="input")
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            role = _MESSAGE_ROLES.get(item["role"]) if isinstance(item.get("role"), str) else None
            if role is None:
                raise GatewayError(
                    400,
                    f"{where}: unsupported role {item.get('role')!r}; expected one of "
                    f"{', '.join(_MESSAGE_ROLES)}",
                    param="input",
                )
            content = _content(item.get("content"), role, where, fetch_images=fetch_images)
            messages.append({"role": role, "content": content})
            calling = messages[-1] if role == "assistant" else None
        elif kind == "function_call":
            if not isinstance(item.get("call_id"), str) or not item["call_id"]:
                raise GatewayError(400, f"{where}: a function_call needs call_id", param="input")
            call = tool_call_entry(item.get("name"), item.get("arguments"), item["call_id"], where, "input")
            if calling is None:
                calling = {"role": "assistant", "content": ""}
                messages.append(calling)
            calling.setdefault("tool_calls", []).append(call)
        elif kind == "function_call_output":
            if not isinstance(item.get("call_id"), str) or not item["call_id"]:
                raise GatewayError(400, f"{where}: a function_call_output needs call_id", param="input")
            text = _output_text(item.get("output"), where)
            messages.append({"role": "tool", "content": text, "tool_call_id": item["call_id"]})
            calling = None
        else:
            raise GatewayError(
                400,
                f"{where}: unsupported item of type {kind!r}; message, function_call and "
                "function_call_output items are accepted",
                param="input",
                code="unsupported_content",
            )
    return messages


def _tools(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None or raw == []:
        return None
    if not isinstance(raw, list):
        raise GatewayError(400, "tools must be a list", param="tools")
    tools: list[dict[str, Any]] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or entry.get("type") != "function":
            kind = entry.get("type") if isinstance(entry, dict) else None
            raise GatewayError(
                400,
                f"tools[{i}]: only function tools are supported, not {kind!r}: the policy being "
                "trained has no hosted tools",
                param="tools",
            )
        tools.append(
            function_tool(entry.get("name"), entry.get("description"), entry.get("parameters"), f"tools[{i}]")
        )
    return tools


def _previous(response_id: Any, calls: Sequence[CallRecord]) -> CallRecord | None:
    if response_id is None:
        return None
    for call in reversed(calls):
        if call.request.id == response_id:
            return call
    raise GatewayError(
        400,
        f"Previous response with id '{response_id}' not found.",
        param="previous_response_id",
        code="previous_response_not_found",
    )


def parse_responses_request(
    body: Any, calls: Sequence[CallRecord] = (), *, fetch_images: bool = True
) -> ChatRequest:
    """Validate a ``POST /v1/responses`` body; ``calls`` are the session's calls so far (for
    ``previous_response_id``). Sampling fields (``temperature``, ``top_p``, ``top_logprobs``),
    ``reasoning``, ``text`` formats, ``truncation``, ``store``, ``model`` and everything unknown
    are ignored: the run's sampling parameters define the distribution the policy is trained
    on."""
    if not isinstance(body, dict):
        raise GatewayError(400, "the request body must be a JSON object")
    if body.get("background"):
        raise GatewayError(400, "background responses are not supported", param="background")
    for key, why in _UNSUPPORTED.items():
        if body.get(key) is not None:
            raise GatewayError(400, f"{key} is not supported: {why}", param=key)
    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise GatewayError(400, "instructions must be a string", param="instructions")
    previous = _previous(body.get("previous_response_id"), calls)
    messages: list[Message] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    if previous is not None:
        messages += previous.messages[int(previous.request.instructions) :]
        messages.append(previous.response)
    messages += _input_messages(body.get("input"), fetch_images=fetch_images)
    tools = _tools(body.get("tools"))
    return ChatRequest(
        messages=name_tool_results(messages),
        tools=tools,
        parse_tool_calls=tools is not None and body.get("tool_choice") != "none",
        max_tokens=token_limit(body, ("max_output_tokens",)),
        stream=bool(body.get("stream")),
        id=request_id("resp_"),
        created=int(time.time()),
        instructions=bool(instructions),
        echo={
            "instructions": instructions,
            "max_output_tokens": body.get("max_output_tokens"),
            "previous_response_id": body.get("previous_response_id"),
            "tools": body.get("tools") or [],
            "tool_choice": body.get("tool_choice") or "auto",
            "metadata": body.get("metadata") or {},
        },
    )


# -- responses --------------------------------------------------------------------------------


def output_items(record: CallRecord) -> list[dict[str, Any]]:
    """The ``output`` of an answered call: a ``message`` item with the text (left out when there
    is nothing but tool calls) and one ``function_call`` item per call. Item ids derive from the
    response id, so a stream and the final object name the same items."""
    stem = record.request.id.removeprefix("resp_")
    text = record.response.get("content") or ""
    calls = answer_calls(record.response)
    items: list[dict[str, Any]] = []
    if text or not calls:
        items.append(
            {
                "id": f"msg_{stem}",
                "type": "message",
                "status": "incomplete" if record.finish_reason == "length" else "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
            }
        )
    for i, (call_id, name, arguments) in enumerate(calls):
        items.append(
            {
                "id": f"fc_{stem}_{i}",
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            }
        )
    return items


def _usage(record: CallRecord) -> dict[str, Any]:
    prompt, completion = len(record.prompt_ids), len(record.completion.token_ids)
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt + completion,
    }


def _response(
    request: ChatRequest,
    session: Session,
    *,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A ``response`` object. ``temperature`` and ``top_p`` are the run's: what was sampled
    with, not what the request asked for."""
    echo = request.echo
    return {
        "id": request.id,
        "object": "response",
        "created_at": request.created,
        "completed_at": int(time.time()) if status == "completed" else None,
        "status": status,
        "background": False,
        "error": error,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "instructions": echo["instructions"],
        "max_output_tokens": echo["max_output_tokens"],
        "metadata": echo["metadata"],
        "model": session.model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": echo["previous_response_id"],
        "temperature": session.params.temperature,
        "text": {"format": {"type": "text"}},
        "tool_choice": echo["tool_choice"],
        "tools": echo["tools"],
        "top_p": session.params.top_p,
        "truncation": "disabled",
        "usage": usage,
    }


class ResponsesWire:
    """``POST /v1/responses`` (:class:`~monorl.gateway.protocol.Wire`)."""

    keepalive = SSE_COMMENT

    def parse(self, body: Any, session: Session) -> ChatRequest:
        return parse_responses_request(body, list(session.calls), fetch_images=session.fetch_images)

    def body(self, record: CallRecord, session: Session) -> dict[str, Any]:
        return _response(
            record.request,
            session,
            status="incomplete" if record.finish_reason == "length" else "completed",
            output=output_items(record),
            usage=_usage(record),
        )

    def stream_head(self, call: PreparedCall, session: Session) -> Iterator[str]:
        response = _response(call.request, session, status="in_progress", output=[])
        yield _event("response.created", 0, response=response)
        yield _event("response.in_progress", 1, response=response)

    def stream_tail(self, record: CallRecord, session: Session) -> Iterator[str]:
        """The finished answer item by item — for a message its content part and one text delta,
        for a function call one arguments delta — then ``response.completed`` (or
        ``response.incomplete``) with the whole response."""
        response = self.body(record, session)
        number = iter(range(2, 1 << 30))  # stream_head sent events 0 and 1
        for index, item in enumerate(response["output"]):
            at = {"item_id": item["id"], "output_index": index}
            if item["type"] == "message":
                (part,) = item["content"]
                text = {**at, "content_index": 0}
                opened = {**item, "status": "in_progress", "content": []}
                yield _event("response.output_item.added", next(number), output_index=index, item=opened)
                yield _event("response.content_part.added", next(number), **text, part={**part, "text": ""})
                if part["text"]:
                    yield _event(
                        "response.output_text.delta", next(number), **text, delta=part["text"], logprobs=[]
                    )
                yield _event(
                    "response.output_text.done", next(number), **text, text=part["text"], logprobs=[]
                )
                yield _event("response.content_part.done", next(number), **text, part=part)
            else:
                opened = {**item, "status": "in_progress", "arguments": ""}
                yield _event("response.output_item.added", next(number), output_index=index, item=opened)
                yield _event(
                    "response.function_call_arguments.delta", next(number), **at, delta=item["arguments"]
                )
                yield _event(
                    "response.function_call_arguments.done",
                    next(number),
                    **at,
                    name=item["name"],
                    arguments=item["arguments"],
                )
            yield _event("response.output_item.done", next(number), output_index=index, item=item)
        yield _event(f"response.{response['status']}", next(number), response=response)

    def stream_error(self, error: GatewayError, call: PreparedCall, session: Session) -> Iterator[str]:
        failure = {"code": "server_error", "message": error.message}
        response = _response(call.request, session, status="failed", output=[], error=failure)
        yield _event("response.failed", 2, response=response)

    def error_body(self, error: GatewayError) -> dict[str, Any]:
        """OpenAI's error shape; the chat is ``input`` here, and a prompt over the context budget
        is reported in the words the Responses API uses for it."""
        body = error.body()
        if isinstance(error, ContextLengthExceeded):
            body["error"]["message"] = (
                "Your input exceeds the context window of this model. Please adjust your input and try again."
            )
        if error.param == "messages":
            body["error"]["param"] = "input"
        return body


def _event(kind: str, sequence_number: int, **fields: Any) -> str:
    return sse({"type": kind, "sequence_number": sequence_number, **fields}, kind)


__all__ = ["ResponsesWire", "output_items", "parse_responses_request"]
