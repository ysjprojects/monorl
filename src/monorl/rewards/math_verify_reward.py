"""``math_verify``: symbolic answer checking with the ``math-verify`` library.

Params:
    timeout_s: per-parse/verify timeout in seconds (default 5). math-verify implements the
        timeout with ``signal.alarm``, which only works on the main thread; off the main thread
        monorl disables it (reward threads then rely on sympy finishing).
    float_rounding / numeric_precision / strict: forwarded to ``math_verify.verify``.
    scope / reference_key: as for ``exact_match``.

The reference may be a plain answer (``"42"``, ``"\\frac{1}{2}"``), a ``$...$``/``\\boxed{}``
expression, or a list of acceptable answers.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from monorl.core.errors import BackendUnavailable
from monorl.core.types import RewardResult, Trajectory
from monorl.rewards.base import (
    check_trajectories,
    reference_value,
    references_as_list,
    response_text,
    reward_name,
)


def _import_math_verify() -> Any:
    try:
        import math_verify
    except ImportError as e:  # pragma: no cover - exercised only when the extra is missing
        raise BackendUnavailable("math_verify reward needs the 'math-verify' package (pip install monorl[rewards])") from e
    return math_verify


class MathVerifyReward:
    def __init__(self, params: dict[str, Any]) -> None:
        self._mv = _import_math_verify()
        self.name = reward_name(params, "math_verify")
        self.timeout_s = int(params.get("timeout_s", 5))
        self.float_rounding = int(params.get("float_rounding", 6))
        self.numeric_precision = int(params.get("numeric_precision", 15))
        self.strict = bool(params.get("strict", True))
        self.scope = str(params.get("scope", "last"))
        self.reference_key = params.get("reference_key")

    def _timeout(self) -> int | None:
        return self.timeout_s if threading.current_thread() is threading.main_thread() else None

    def _parse_reference(self, ref: Any) -> list[Any]:
        """Bare answers ("42", "(1, 2)", "x+1") are parsed as one LaTeX expression so that the
        whole string is the gold value; references that already carry ``$``/``\\boxed`` anchors
        are parsed as-is."""
        text = str(ref).strip()
        timeout = self._timeout()
        if "$" not in text and "\\boxed" not in text:
            parsed = self._mv.parse(f"${text}$", parsing_timeout=timeout)
            if parsed:
                return parsed
        return self._mv.parse(text, parsing_timeout=timeout)

    def score_text(self, text: str, reference: Any) -> tuple[float, list[str]]:
        timeout = self._timeout()
        prediction = self._mv.parse(text, parsing_timeout=timeout)
        extracted = [str(p) for p in prediction]
        if not prediction:
            return 0.0, extracted
        for ref in references_as_list(reference):
            if ref is None:
                continue
            gold = self._parse_reference(ref)
            if not gold:
                continue
            if self._mv.verify(
                gold,
                prediction,
                float_rounding=self.float_rounding,
                numeric_precision=self.numeric_precision,
                strict=self.strict,
                timeout_seconds=timeout,
            ):
                return 1.0, extracted
        return 0.0, extracted

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        check_trajectories(trajectories)
        out = []
        for t in trajectories:
            score, extracted = self.score_text(response_text(t, self.scope), reference_value(t, self.reference_key))
            out.append(RewardResult(score=score, breakdown={self.name: score}, info={"extracted": extracted}))
        return out


def make_math_verify(params: dict[str, Any], **_: Any) -> MathVerifyReward:
    return MathVerifyReward(params)
