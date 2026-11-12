"""Chat rendering for multi-turn prompts: full template renders and token-level continuation.

:func:`chat_prompt_ids` renders a whole conversation with the tokenizer's chat template. Doing that
again for turn ``k+1`` re-encodes what the policy already produced through text, and that round
trip is not exact: special tokens are dropped by decoding, whitespace is normalised, and templates
may drop earlier reasoning or re-serialise tool calls. The new prompt then no longer starts with
the tokens the policy was sampled on, which are the tokens it is trained on.

:func:`continue_prompt_ids` keeps the sampled tokens instead ("token in, token out"): the next
prompt is the previous prompt, the completion exactly as sampled, the template's end-of-turn text
when the completion did not close the turn itself, and the rendering of only the new messages plus
the generation prompt. Both template pieces come from rendering the conversation with a sentinel
in place of the assistant turn, so nothing depends on how the template would have re-tokenised the
sampled text.

This only fixes tokens, never what the context says: a turn is continued this way only when the
template itself would carry the assistant's content over unchanged (surrounding whitespace
aside). A template that rewrites history — dropping the reasoning of earlier turns, rendering a
turn differently once more messages follow it — is followed instead: the conversation is rendered
in full, as with ``mode="template"``, and the result says so (``bridged=False``). The batch
builder then trains such a turn as a sequence of its own.

Torch-free; shared by the episode runner and the capture gateway.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from monorl.core.types import Completion, Message

SENTINEL = "MONORLTURNSENTINEL"
"""Stand-in assistant content used to locate the end of a turn in rendered chat text."""


def render_chat(
    tokenizer: Any, messages: Sequence[Message], tools: Any = None, *, add_generation_prompt: bool = True
) -> str:
    """``messages`` as chat-template text (``tools`` is only passed when given)."""
    kwargs: dict[str, Any] = {"add_generation_prompt": add_generation_prompt, "tokenize": False}
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(list(messages), **kwargs)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """Token ids of already-templated text: no special tokens are added, the template owns them."""
    if not text:
        return []
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


def chat_prompt_ids(tokenizer: Any, messages: Sequence[Message], tools: Any = None) -> list[int]:
    """Token ids of ``messages`` rendered with the chat template and a generation prompt.

    Rendered as text first and tokenised without special tokens, so the template alone decides
    about BOS/EOS; rollout engines use the same recipe for chat prompts.
    """
    return encode_text(tokenizer, render_chat(tokenizer, messages, tools))


def special_token_ids(tokenizer: Any) -> frozenset[int]:
    """Ids the tokenizer treats as special: its named special tokens, added tokens flagged special
    (end-of-turn markers are often only that) and the EOS id(s)."""
    ids = {int(t) for t in (getattr(tokenizer, "all_special_ids", None) or ())}
    for token_id, token in (getattr(tokenizer, "added_tokens_decoder", None) or {}).items():
        if getattr(token, "special", False):
            ids.add(int(token_id))
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif eos is not None:
        ids.update(int(t) for t in eos)
    return frozenset(ids)


@dataclass(slots=True)
class Continuation:
    ids: list[int]
    """Prompt token ids for the next turn."""
    bridged: bool
    """``True``: ``ids`` extend the previous prompt and its sampled completion token for token.
    ``False``: the conversation was rendered again from its messages."""


def continue_prompt_ids(
    tokenizer: Any,
    *,
    previous_messages: Sequence[Message],
    previous_prompt_ids: Sequence[int],
    completion: Completion,
    new_messages: Sequence[Message],
    tools: Any = None,
    assistant_message: Message | None = None,
    mode: str = "tokens",
) -> Continuation:
    """Prompt ids for the turn after ``completion``.

    ``previous_messages`` is the chat the previous turn was prompted with, ``previous_prompt_ids``
    the ids it conditioned on, ``new_messages`` what follows the assistant turn (tool results, the
    next user message). ``assistant_message`` is the completion as a chat message when it differs
    from plain ``{"role": "assistant", "content": completion.text}`` (structured ``tool_calls``).

    ``mode="tokens"`` continues the token stream (module docstring); ``mode="template"`` always
    re-renders ``previous_messages + [assistant_message] + new_messages``.
    """
    if mode not in ("tokens", "template"):
        raise ValueError(f"unknown turn rendering mode {mode!r}; expected 'tokens' or 'template'")
    assistant: Message = (
        dict(assistant_message)  # type: ignore[assignment]
        if assistant_message is not None
        else {"role": "assistant", "content": completion.text}
    )
    if mode == "tokens":
        pieces = _turn_pieces(tokenizer, previous_messages, assistant, new_messages, tools)
        if pieces is not None:
            closing, tail = pieces
            sampled = [int(t) for t in completion.token_ids]
            rest = _after_emitted_end(tokenizer, closing, tail, sampled[-1] if sampled else None)
            return Continuation(ids=[*(int(t) for t in previous_prompt_ids), *sampled, *rest], bridged=True)
    full = [*previous_messages, assistant, *new_messages]
    return Continuation(ids=chat_prompt_ids(tokenizer, full, tools), bridged=False)


def _turn_pieces(
    tokenizer: Any,
    previous_messages: Sequence[Message],
    assistant: Message,
    new_messages: Sequence[Message],
    tools: Any,
) -> tuple[str, str] | None:
    """``(closing, tail)``: the text the template puts after an assistant turn's content, and its
    rendering of ``new_messages`` plus the generation prompt. ``None`` when the template cannot be
    taken apart like that, renders the earlier messages differently now that more follow, or
    would not carry the assistant's content over as it is."""
    content = assistant.get("content") or ""
    if not isinstance(content, str):
        return None
    plain: Message = {"role": "assistant", "content": SENTINEL}
    # keeps tool_calls: templates may need them to render the tool messages that follow
    marked: Message = {**assistant, "content": SENTINEL}
    try:
        prompt = render_chat(tokenizer, previous_messages, tools)
        closed_plain = render_chat(tokenizer, [*previous_messages, plain], tools, add_generation_prompt=False)
        closed = (
            closed_plain
            if marked == plain
            else render_chat(tokenizer, [*previous_messages, marked], tools, add_generation_prompt=False)
        )
        extended = render_chat(tokenizer, [*previous_messages, marked, *new_messages], tools)
        actual = (
            render_chat(tokenizer, [*previous_messages, assistant, *new_messages], tools)
            if content.strip()
            else None
        )
    except Exception:  # noqa: BLE001 - a template may reject the probe; the full render reports real errors
        return None
    plain_at = closed_plain.rfind(SENTINEL)
    if plain_at < 0 or not extended.startswith(prompt):
        return None
    closing = closed_plain[plain_at + len(SENTINEL) :]
    # The assistant turn must follow the previous prompt as the template renders it today ...
    at = extended.find(SENTINEL, len(prompt))
    if at < 0:
        # A template that renders a turn from its tool_calls alone (Llama 3.1, Mistral) leaves the
        # content out; that only holds the sampled turn when there was no other content.
        if content.strip() or not assistant.get("tool_calls"):
            return None
        if not (closed.startswith(prompt) and closed.endswith(closing) and extended.startswith(closed)):
            return None
        return closing, extended[len(closed) :]
    if extended[len(prompt) : at].strip():
        return None
    head, rest = extended[:at], extended[at + len(SENTINEL) :]
    # ... end the way a finished turn ends (after the content: tool calls if any, then the closing) ...
    closed_at = closed.rfind(SENTINEL)
    if closed_at < 0:
        return None
    turn_end = closed[closed_at + len(SENTINEL) :]
    if not rest.startswith(turn_end) or not turn_end.endswith(closing):
        return None
    # ... and hold the real content as it is.
    if actual is not None:
        if len(actual) < len(head) + len(rest) or not (actual.startswith(head) and actual.endswith(rest)):
            return None
        if actual[len(head) : len(actual) - len(rest)].strip() != content.strip():
            return None
    return closing, rest[len(turn_end) :]


def _after_emitted_end(tokenizer: Any, closing: str, tail: str, last_sampled: int | None) -> list[int]:
    """Ids of ``closing + tail`` without what the policy already produced: when its last sampled
    token is the closing's end-of-turn marker (the first special token of ``closing``), the ids
    resume after that marker. A completion cut off by the length limit or a stop string gets the
    whole closing."""
    rest = encode_text(tokenizer, closing + tail)
    if last_sampled is None:
        return rest
    specials = special_token_ids(tokenizer)
    closing_ids = encode_text(tokenizer, closing)
    end = next((i for i, token_id in enumerate(closing_ids) if token_id in specials), None)
    if end is None or closing_ids[end] != last_sampled or rest[: end + 1] != closing_ids[: end + 1]:
        return rest
    return rest[end + 1 :]


__all__ = [
    "Continuation",
    "chat_prompt_ids",
    "continue_prompt_ids",
    "encode_text",
    "render_chat",
    "special_token_ids",
]
