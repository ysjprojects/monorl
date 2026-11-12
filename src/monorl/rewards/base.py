"""Helpers shared by every reward function.

A reward receives :class:`~monorl.core.types.Trajectory` objects; it scores
``last_response_text`` (the final policy turn) against ``sample.reference`` unless configured
otherwise via ``params["scope"] = "all"`` (``response_text`` across turns).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from monorl.core.types import Message, Trajectory, message_text

ResponseScope = str  # "last" | "all"


def reward_name(params: dict[str, Any], default: str) -> str:
    return str(params.get("name") or default)


def response_text(trajectory: Trajectory, scope: ResponseScope = "last") -> str:
    if scope == "all":
        return trajectory.response_text
    if scope == "last":
        return trajectory.last_response_text
    raise ValueError(f"unknown response scope {scope!r}; expected 'last' or 'all'")


def reference_value(trajectory: Trajectory, key: str | None = None) -> Any:
    """``sample.reference`` (or ``reference[key]`` / ``sample.metadata[key]`` when ``key`` given)."""
    ref = trajectory.sample.reference
    if key is None:
        return ref
    if isinstance(ref, dict) and key in ref:
        return ref[key]
    if key in trajectory.sample.metadata:
        return trajectory.sample.metadata[key]
    return ref


def conversation(trajectory: Trajectory) -> list[Message]:
    """Full chat including the final assistant turn (appended if the episode messages lack it)."""
    msgs = list(trajectory.messages) if trajectory.messages else list(trajectory.sample.messages)
    if not trajectory.turns:
        return msgs
    if (
        msgs
        and msgs[-1].get("role") == "assistant"
        and message_text(msgs[-1]) == trajectory.last_response_text
    ):
        return msgs
    return [*msgs, {"role": "assistant", "content": trajectory.last_response_text}]


def prompt_text(trajectory: Trajectory) -> str:
    """Plain-text rendering of the user-facing prompt (for judge templates)."""
    parts = []
    for m in trajectory.sample.messages:
        if m.get("role") in ("user", "system"):
            parts.append(message_text(m))
    return "\n\n".join(parts)


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_BOXED_RE = re.compile(r"\\boxed\s*\{")
_HASH_RE = re.compile(r"####\s*(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"```(?P<lang>[\w+-]*)[ \t]*\n(?P<body>.*?)```", re.DOTALL)


def extract_last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    return matches[-1].replace(",", "") if matches else None


def extract_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` with balanced braces, or ``None``."""
    last: str | None = None
    for m in _BOXED_RE.finditer(text):
        depth = 1
        i = m.end()
        start = i
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[start : i - 1]
    return last


def extract_after_hash(text: str) -> str | None:
    """GSM8K-style ``#### answer`` (last occurrence)."""
    matches = _HASH_RE.findall(text)
    return matches[-1].strip() if matches else None


def extract_code_block(text: str, language: str = "python") -> str | None:
    """Body of the last fenced code block tagged ``language`` (or untagged); ``None`` if absent."""
    chosen: str | None = None
    for m in _FENCE_RE.finditer(text):
        lang = m.group("lang").lower()
        if lang in ("", language):
            chosen = m.group("body")
    return chosen


EXTRACTORS = {
    "last_number": extract_last_number,
    "boxed": extract_boxed,
    "after_hash": extract_after_hash,
    "none": lambda text: text,
}


def extract_answer(text: str, mode: str) -> str | None:
    try:
        fn = EXTRACTORS[mode]
    except KeyError:
        raise ValueError(f"unknown extract mode {mode!r}; expected one of {sorted(EXTRACTORS)}") from None
    return fn(text)


def references_as_list(ref: Any) -> list[Any]:
    """A reference may be a scalar, a list of acceptable answers, or ``{"answer": ...}``."""
    if isinstance(ref, dict):
        ref = ref.get("answer", ref.get("reference", ref))
    if isinstance(ref, (list, tuple)):
        return list(ref)
    return [ref]


def check_trajectories(trajectories: Sequence[Trajectory]) -> None:
    for t in trajectories:
        if not isinstance(t, Trajectory):
            raise TypeError(f"reward functions take Trajectory objects, got {type(t).__name__}")
