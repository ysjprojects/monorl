"""Gateway sessions and call records.

A :class:`Session` is one episode's view of the policy: the rollout engine, tokenizer and
sampling parameters of the run, plus every call the episode's agent made, over whichever API
(:class:`~monorl.gateway.protocol.Wire`). A call is admitted and given its prompt by
:meth:`Session.prepare`, answered by :meth:`Session.finish` (:meth:`Session.complete` does both)
and kept as a :class:`CallRecord` with the exact prompt ids and the engine's
:class:`~monorl.core.types.Completion`, which is what
:func:`monorl.gateway.trajectory.build_trajectory` turns into a trainable trajectory.

Prompts follow "token in, token out" (:mod:`monorl.rollout.render`): when a request repeats an
earlier call's messages and its answer and then adds messages, its prompt is that call's prompt
ids, the sampled completion ids and only the new messages rendered — the agent's copy of the
assistant message is never re-tokenised. A request whose history matches no earlier call starts a
new context and is rendered in full.

A chat with images cannot be sent as token ids (they carry no pixels): it reaches the engine as
messages, the engine encodes it with the model's processor and reports the ids it conditioned on,
as for stepped multimodal episodes. That path has no room for tool schemas, so images and tools
exclude each other. The batch builder trains a multimodal trajectory as the conversation the
episode ended in; once a session has made a call with images, calls without images are therefore
refused — they would end the episode in a text-only conversation whose earlier image turns
could not be trained on.

Torch-free; sessions are thread-safe (an agent may call the policy concurrently).
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from monorl.core.errors import GenerationCancelled
from monorl.core.types import Completion, Message, PromptInput, SamplingParams, ToolCall, messages_have_media
from monorl.gateway.protocol import (
    ChatRequest,
    ContextLengthExceeded,
    GatewayError,
    response_message,
    split_stop,
    split_tool_calls,
)
from monorl.rollout.render import chat_prompt_ids, continue_prompt_ids

if TYPE_CHECKING:
    from monorl.core.interfaces import RolloutEngine


@dataclass(slots=True)
class PreparedCall:
    """A call the session admitted, with the prompt it will be answered from."""

    request: ChatRequest
    parent: CallRecord | None
    """The call whose conversation this one continues (``None``: a new context)."""
    prompt_ids: list[int]
    bridged: bool
    room: int | None
    """Tokens the context budget leaves for the completion (``None``: no budget)."""
    media: bool
    """The chat has images: the engine gets it as messages, not as ``prompt_ids``."""


@dataclass(slots=True)
class CallRecord:
    """One answered call, i.e. one policy turn."""

    index: int
    """Position among the session's calls, in the order they were answered."""
    request: ChatRequest
    messages: list[Message]
    """The request's chat (normalised)."""
    tools: list[dict[str, Any]] | None
    prompt_ids: list[int]
    """Token ids the policy conditioned on."""
    completion: Completion
    """The engine's completion, unchanged."""
    response: Message
    """The answer as a normalised assistant message — the text around the parsed tool calls,
    without a stop string the request added, and the calls: what each API's response is built
    from and what a continuing request repeats."""
    finish_reason: str
    """OpenAI finish reason of the answer: ``stop`` | ``length`` | ``tool_calls``."""
    stop_sequence: str | None
    """The stop string of the request that ended the completion, if one did."""
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
    session reports. ``max_context_tokens`` is the episode's context budget: a call whose prompt
    leaves no room for a completion is refused (:class:`ContextLengthExceeded`, worded by each
    API the way its vendor words it), so an agent that compacts its history can react, and every
    completion is capped to the room that is left. ``processor`` is the policy's HF
    ``AutoProcessor`` when it is a vision-language model; without one the engine's is used
    (``engine.processor``, looked up through engine wrappers), and a policy that has none cannot
    take images. ``fetch_images=False`` refuses images sent as ``http(s)`` URLs instead of
    downloading them. ``cancelled`` is set when a generation was cancelled (the episode is being
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
        max_context_tokens: int | None = None,
        processor: Any | None = None,
        fetch_images: bool = True,
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
        self.max_context_tokens = max_context_tokens
        self.processor = processor if processor is not None else _engine_processor(engine)
        self.fetch_images = fetch_images
        self.calls: list[CallRecord] = []
        self.refused = 0
        """Calls turned away because ``max_calls`` was reached."""
        self.context_exceeded = 0
        """Calls turned away because their prompt did not fit ``max_context_tokens``."""
        self.reward: float | None = None
        """Sum of the rewards posted to the session; ``None`` when none was posted."""
        self.cancelled = threading.Event()
        self.failure: BaseException | None = None
        self._closed = False
        self._in_flight = 0
        self._media = False
        """A call with images was admitted (module docstring)."""
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
        """Answer one request with the policy and record it. Blocks for the generation; raises
        :class:`GatewayError` for what the HTTP layer reports to the agent."""
        return self.finish(self.prepare(request))

    def prepare(self, request: ChatRequest) -> PreparedCall:
        """Admit ``request`` and build its prompt, without generating. Everything a request can
        be refused for is raised here (:class:`GatewayError`), before a stream has to begin. The
        call counts as in flight until :meth:`finish` has run, which must follow."""
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
            parent, prompt_ids, bridged, media = self._render(earlier, request)
            if self._media and not media:
                raise _text_after_images()
            room = self._room(len(prompt_ids))
            if media:
                with self._lock:
                    self._media = True
        except BaseException:
            with self._lock:
                self._in_flight -= 1
            raise
        return PreparedCall(request, parent, prompt_ids, bridged, room, media)

    def finish(self, call: PreparedCall) -> CallRecord:
        """Generate the answer to a prepared call and record it. Blocks for the generation."""
        request = call.request
        try:
            completion = self._generate(call)
            calls, invalid, text = (
                split_tool_calls(completion.text) if request.parse_tool_calls else ([], 0, completion.text)
            )
            stop = None
            if completion.finish_reason == "stop":
                text, stop = split_stop(text, request.stop)
            with self._lock:
                self._require_open()
                if self._media and not call.media:  # a call with images was admitted meanwhile
                    raise _text_after_images()
                record = CallRecord(
                    index=len(self.calls),
                    request=request,
                    messages=request.messages,
                    tools=request.tools,
                    prompt_ids=(
                        call.prompt_ids
                        if completion.prompt_token_ids is None
                        else [int(t) for t in completion.prompt_token_ids]
                    ),
                    completion=completion,
                    response=response_message(text, calls),
                    finish_reason=(
                        "tool_calls"
                        if calls
                        else "length"
                        if completion.finish_reason == "length"
                        else "stop"
                    ),
                    stop_sequence=stop,
                    parent=None if call.parent is None else call.parent.index,
                    bridged=call.bridged,
                    tool_calls=calls if request.parse_tool_calls else None,
                    invalid_calls=invalid,
                )
                self.calls.append(record)
            return record
        finally:
            with self._lock:
                self._in_flight -= 1

    def count_tokens(self, request: ChatRequest) -> int:
        """Prompt tokens ``request`` would be answered from, were it made now. Nothing is
        recorded and the call does not count."""
        with self._lock:
            self._require_open()
            earlier = list(self.calls)
        return len(self._render(earlier, request)[1])

    def _require_open(self) -> None:
        """Caller holds the lock."""
        if self._closed:
            raise GatewayError(503, "the session is closed", type="server_error", code="session_closed")
        if self.cancelled.is_set():
            raise GatewayError(
                503, "the episode is being aborted", type="server_error", code="generation_cancelled"
            )

    def _render(
        self, earlier: Sequence[CallRecord], request: ChatRequest
    ) -> tuple[CallRecord | None, list[int], bool, bool]:
        """``(parent, prompt ids, bridged, media)`` for ``request`` after the calls ``earlier``."""
        parent = _find_parent(earlier, request)
        if messages_have_media(request.messages):
            return parent, self._media_prompt(request), False, True
        prompt_ids, bridged = self._prompt(parent, request)
        return parent, prompt_ids, bridged, False

    def _prompt(self, parent: CallRecord | None, request: ChatRequest) -> tuple[list[int], bool]:
        """``(prompt ids, bridged)`` for a text-only ``request``, continuing ``parent`` when
        there is one."""
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

    def _media_prompt(self, request: ChatRequest) -> list[int]:
        """Prompt ids of a chat with images, as the engine's processor path will produce them:
        the whole chat encoded by the model's processor, image placeholder tokens included."""
        if self.processor is None:
            raise GatewayError(
                400,
                "the policy being trained is a text-only model and cannot take image inputs",
                param="messages",
                code="unsupported_content",
            )
        if request.tools is not None:
            raise GatewayError(
                400,
                "images cannot be combined with tools: a chat with images reaches the rollout engine as "
                "messages, which cannot carry tool schemas. Send the request without tools",
                param="tools",
                code="unsupported_content",
            )
        from monorl.rollout.media import vlm_prompt_ids

        try:
            return vlm_prompt_ids(self.processor, request.messages)
        except Exception as e:  # noqa: BLE001 - an unreadable image or a chat the processor rejects
            raise GatewayError(
                400,
                f"the model's processor cannot encode these messages: {e}",
                param="messages",
                code="template_error",
            ) from e

    def _room(self, prompt_tokens: int) -> int | None:
        """Tokens left for the completion under the context budget (``None``: no budget). Raises
        :class:`ContextLengthExceeded` when the prompt leaves none."""
        if self.max_context_tokens is None:
            return None
        room = self.max_context_tokens - prompt_tokens
        if room < 1:
            with self._lock:
                self.context_exceeded += 1
            raise ContextLengthExceeded(prompt_tokens, self.max_context_tokens)
        return room

    def _generate(self, call: PreparedCall) -> Completion:
        """One sample under the run's sampling parameters; the request and the context budget
        (``call.room``) may only shorten it."""
        request = call.request
        params = self.params
        limits = [params.max_new_tokens, request.max_tokens, call.room]
        max_new_tokens = min(limit for limit in limits if limit is not None)
        stop = [*params.stop, *(s for s in request.stop if s not in params.stop)]
        params = dataclasses.replace(params, n=1, max_new_tokens=max_new_tokens, stop=stop)
        prompt: PromptInput = list(request.messages) if call.media else list(call.prompt_ids)
        try:
            return self.engine.generate([prompt], params)[0][0]
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


def _engine_processor(engine: Any) -> Any | None:
    """The ``AutoProcessor`` a vision-language engine exposes as ``processor``, looked up through
    engine wrappers (``BatchingEngine.engine``); ``None`` for text-only engines."""
    for _ in range(4):
        if engine is None:
            return None
        processor = getattr(engine, "processor", None)
        if processor is not None:
            return processor
        engine = getattr(engine, "engine", None)
    return None


def _text_after_images() -> GatewayError:
    return GatewayError(
        400,
        "this episode has made calls with images; a call without images would end it in a "
        "conversation that leaves those calls untrainable. Keep the images in the conversation",
        param="messages",
        code="unsupported_content",
    )


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


__all__ = ["CallRecord", "PreparedCall", "Session"]
