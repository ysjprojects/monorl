"""From a session's call records to a :class:`~monorl.core.types.Trajectory`.

Every call is one :class:`~monorl.core.types.Turn`, in the order the calls were answered. The
calls form a forest through :attr:`CallRecord.parent` (a call continues the conversation of an
earlier one); each call no later call continued ends a conversation, and each conversation — that
call's request messages plus its answer — is a thread:

* thread 0 (``Trajectory.messages``) is the conversation of the last call;
* every other conversation is an extra thread, in the order they ended (the context before the
  agent compacted or rewrote its history, a sub-agent's chat, a side question);
* a call belongs to thread 0 when the last call descends from it, otherwise to the thread of the
  latest conversation that descends from it. ``Turn.message_index`` is the position of its answer
  there.

Threads hold monorl chat messages: assistant ``tool_calls`` are ``{id, name, arguments}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from monorl.core.types import Message, Sample, ToolCall, Trajectory, Turn
from monorl.gateway.session import CallRecord


def build_trajectory(calls: Sequence[CallRecord], sample: Sample, group_id: str) -> Trajectory:
    """The episode recorded in ``calls`` (a session's calls, at least one) as an unscored
    trajectory."""
    if not calls:
        raise ValueError(f"episode {group_id} recorded no policy calls")
    continued = {call.parent for call in calls if call.parent is not None}
    ends = [call.index for call in calls if call.index not in continued]  # the last call is one
    order = [ends[-1], *ends[:-1]]
    threads = [[_message(m) for m in (*calls[end].messages, calls[end].response)] for end in order]
    thread_of: dict[int, int] = {}
    for thread, end in sorted(enumerate(order), key=lambda item: -item[1]):
        index: int | None = end
        while index is not None and index not in thread_of:
            thread_of[index] = thread
            index = calls[index].parent
    turns = [
        Turn(
            prompt_token_ids=list(call.prompt_ids),
            completion=call.completion,
            info=_turn_info(call, threads[thread_of[call.index]]),
            thread=thread_of[call.index],
            message_index=len(call.messages),
        )
        for call in calls
    ]
    return Trajectory(
        sample=sample,
        group_id=group_id,
        messages=threads[0],
        turns=turns,
        info={"finish_reason": calls[-1].completion.finish_reason, "num_turns": len(turns)},
        extra_threads=threads[1:],
    )


def _message(message: Message) -> Message:
    """A normalised gateway message as a monorl chat message."""
    calls = message.get("tool_calls")
    if not calls:
        return dict(message)  # type: ignore[return-value]
    flat: list[ToolCall] = []
    for call in calls:
        function = call["function"]  # type: ignore[typeddict-item]
        entry = ToolCall(name=function["name"], arguments=function["arguments"])
        if "id" in call:
            entry = ToolCall(id=call["id"], **entry)
        flat.append(entry)
    return {**message, "tool_calls": flat}  # type: ignore[return-value]


def _turn_info(call: CallRecord, thread: Sequence[Message]) -> dict[str, Any]:
    """``call``/``parent``/``bridged``, and for a request with tools the parsed ``tool_calls``
    (``{id, name, arguments, result}``; ``result`` is the content of the ``tool`` message that
    answered the call in ``thread``, ``None`` without one) and ``invalid_calls``."""
    info: dict[str, Any] = {"call": call.index, "parent": call.parent, "bridged": call.bridged}
    if call.tool_calls is None:
        return info
    results: dict[str, Any] = {}
    for message in thread[len(call.messages) + 1 :]:
        if message.get("role") != "tool":
            break
        if "tool_call_id" in message:
            results.setdefault(message["tool_call_id"], message.get("content", ""))
    info["tool_calls"] = [
        {"id": c["id"], "name": c["name"], "arguments": c.get("arguments"), "result": results.get(c["id"])}
        for c in call.tool_calls
    ]
    info["invalid_calls"] = call.invalid_calls
    return info


__all__ = ["build_trajectory"]
