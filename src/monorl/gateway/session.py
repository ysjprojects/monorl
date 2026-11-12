"""Gateway sessions and call records.

A :class:`Session` is one episode's view of the policy: the rollout engine, tokenizer and
sampling parameters of the run, plus every chat-completions call the episode's agent made. A
call is answered by :meth:`Session.complete` and kept as a :class:`CallRecord` with the exact
prompt ids and the engine's :class:`~monorl.core.types.Completion`, which is what
:func:`monorl.gateway.trajectory.build_trajectory` turns into a trainable trajectory.

Prompts follow "token in, token out" (:mod:`monorl.rollout.render`): when a request repeats an
earlier call's messages and its answer and then adds messages, its prompt is that call's prompt
ids, the sampled completion ids and only the new messages rendered — the agent's copy of the
assistant message is never re-tokenised. A request whose history matches no earlier call starts a
new context and is rendered in full.

Torch-free; sessions are thread-safe (an agent may call the policy concurrently).
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from monorl.core.errors import GenerationCancelled
from monorl.core.types import Completion, Message, SamplingParams, ToolCall
from monorl.gateway.protocol import (
    ChatRequest,
    GatewayError,
    assistant_message,
    normalize_messages,
    split_tool_calls,
    strip_stop,
)
from monorl.rollout.render import chat_prompt_ids, continue_prompt_ids

if TYPE_CHECKING:
    from monorl.core.interfaces import RolloutEngine


@dataclass(slots=True)
class CallRecord:
    """One answered chat-completions call, i.e. one policy turn."""

    index: int
    """Position among the session's calls, in the order they were answered."""
    messages: list[Message]
    """The request's chat (normalised)."""
    tools: list[dict[str, Any]] | None
    prompt_ids: list[int]
    """Token ids the policy conditioned on."""
    completion: Completion
    """The engine's completion, unchanged."""
    answer: dict[str, Any]
    """The OpenAI assistant message the call was answered with."""
    response: Message
    """``answer`` normalised like request messages: what a continuing request repeats."""
    finish_reason: str
    """OpenAI finish reason of the answer: ``stop`` | ``length`` | ``tool_calls``."""
    parent: int | None
    """The call whose conversation this one continues (``None``: a new context)."""
    bridged: bool
    """The prompt extends the parent's prompt and completion token for token."""
    tool_calls: list[ToolCall] | None
    """Calls parsed out of the completion; ``None`` when the request asked for none."""
    invalid_calls: int
    """Malformed tool calls in the completion."""


class Session:
    """See module docstring. ``max_calls`` caps the answered calls; ``model`` is the name the
    session reports. ``cancelled`` is set when a generation was cancelled (the episode is being
    aborted), ``failure`` holds the first engine error; the episode's owner watches both."""

    def __init__(
        self,
        session_id: str,
        *,
        engine: RolloutEngine,
        tokenizer: Any,
        params: SamplingParams,
        turn_rendering: str = "tokens",
        model: str = "monorl",
        max_calls: int | None = None,
    ) -> None:
        if turn_rendering not in ("tokens", "template"):
            raise ValueError(
                f"unknown turn rendering mode {turn_rendering!r}; expected 'tokens' or 'template'"
            )
        self.id = session_id
        self.engine = engine
        self.tokenizer = tokenizer
        self.params = params
        self.turn_rendering = turn_rendering
        self.model = model
        self.max_calls = max_calls
        self.calls: list[CallRecord] = []
        self.refused = 0
        """Calls turned away because ``max_calls`` was reached."""
        self.reward: float | None = None
        """Sum of the rewards posted to the session; ``None`` when none was posted."""
        self.cancelled = threading.Event()
        self.failure: BaseException | None = None
        self._closed = False
        self._in_flight = 0
        self._lock = threading.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Stop serving: later requests and generations still in flight answer 503 and are not
        recorded, so ``calls`` is final."""
        with self._lock:
            self._closed = True

    def add_reward(self, reward: float) -> float:
        """Add ``reward`` to the session's posted reward; returns the total."""
        with self._lock:
            self._require_open()
            self.reward = (self.reward or 0.0) + float(reward)
            return self.reward

    def complete(self, request: ChatRequest) -> CallRecord:
        """Answer one chat-completions request with the policy and record it. Blocks for the
        generation; raises :class:`GatewayError` for what the HTTP layer reports to the agent."""
        with self._lock:
            self._require_open()
            if self.max_calls is not None and len(self.calls) + self._in_flight >= self.max_calls:
                self.refused += 1
                raise GatewayError(
                    400,
                    f"this episode is limited to {self.max_calls} policy calls (env.max_turns)",
                    code="max_turns_exceeded",
                )
            self._in_flight += 1
            earlier = list(self.calls)
        try:
            parent = _find_parent(earlier, request)
            prompt_ids, bridged = self._prompt(parent, request)
            completion = self._generate(prompt_ids, request)
            calls, invalid, text = (
                split_tool_calls(completion.text) if request.parse_tool_calls else ([], 0, completion.text)
            )
            if completion.finish_reason == "stop":
                text = strip_stop(text, request.stop)
            answer = assistant_message(text, calls)
            with self._lock:
                self._require_open()
                record = CallRecord(
                    index=len(self.calls),
                    messages=request.messages,
                    tools=request.tools,
                    prompt_ids=(
                        prompt_ids
                        if completion.prompt_token_ids is None
                        else [int(t) for t in completion.prompt_token_ids]
                    ),
                    completion=completion,
                    answer=answer,
                    response=normalize_messages([answer])[0],
                    finish_reason=(
                        "tool_calls"
                        if calls
                        else "length"
                        if completion.finish_reason == "length"
                        else "stop"
                    ),
                    parent=None if parent is None else parent.index,
                    bridged=bridged,
                    tool_calls=calls if request.parse_tool_calls else None,
                    invalid_calls=invalid,
                )
                self.calls.append(record)
            return record
        finally:
            with self._lock:
                self._in_flight -= 1

    def _require_open(self) -> None:
        """Caller holds the lock."""
        if self._closed:
            raise GatewayError(503, "the session is closed", type="server_error", code="session_closed")
        if self.cancelled.is_set():
            raise GatewayError(
                503, "the episode is being aborted", type="server_error", code="generation_cancelled"
            )

    def _prompt(self, parent: CallRecord | None, request: ChatRequest) -> tuple[list[int], bool]:
        """``(prompt ids, bridged)`` for ``request``, continuing ``parent`` when there is one."""
        try:
            if parent is None:
                return chat_prompt_ids(self.tokenizer, request.messages, request.tools), False
            continuation = continue_prompt_ids(
                self.tokenizer,
                previous_messages=parent.messages,
                previous_prompt_ids=parent.prompt_ids,
                completion=parent.completion,
                new_messages=request.messages[len(parent.messages) + 1 :],
                tools=request.tools,
                assistant_message=parent.response,
                mode=self.turn_rendering,
            )
            return continuation.ids, continuation.bridged
        except Exception as e:  # noqa: BLE001 - whatever the chat template raises is the agent's to fix
            raise GatewayError(
                400,
                f"the chat template cannot render these messages: {e}",
                param="messages",
                code="template_error",
            ) from e

    def _generate(self, prompt_ids: Sequence[int], request: ChatRequest) -> Completion:
        """One sample under the run's sampling parameters; the request may only shorten it."""
        params = self.params
        if request.max_tokens is not None:
            max_new_tokens = min(params.max_new_tokens, request.max_tokens)
        else:
            max_new_tokens = params.max_new_tokens
        stop = [*params.stop, *(s for s in request.stop if s not in params.stop)]
        params = dataclasses.replace(params, n=1, max_new_tokens=max_new_tokens, stop=stop)
        try:
            return self.engine.generate([list(prompt_ids)], params)[0][0]
        except GenerationCancelled as e:
            self.cancelled.set()
            raise GatewayError(
                503, "the generation was cancelled", type="server_error", code="generation_cancelled"
            ) from e
        except Exception as e:  # noqa: BLE001 - reported to the agent and kept for the episode's owner
            with self._lock:
                if self.failure is None:
                    self.failure = e
            raise GatewayError(500, f"generation failed: {e}", type="server_error") from e


def _find_parent(calls: Sequence[CallRecord], request: ChatRequest) -> CallRecord | None:
    """The latest call whose chat plus answer is a message-level prefix of ``request`` and whose
    tools are the request's."""
    messages = request.messages
    for call in reversed(calls):
        n = len(call.messages)
        if (
            n < len(messages)
            and call.tools == request.tools
            and messages[n] == call.response
            and messages[:n] == call.messages
        ):
            return call
    return None


__all__ = ["CallRecord", "Session"]
