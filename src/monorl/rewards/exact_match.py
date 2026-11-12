"""``exact_match``: 1.0 when the extracted answer equals the reference, else 0.0.

Params:
    extract: ``last_number`` | ``boxed`` | ``after_hash`` | ``none`` (default ``last_number``).
        Applied to the response; the reference goes through the same extractor and falls back
        to its raw text when nothing is found.
    normalize: list of ``"whitespace"`` / ``"case"`` (default both). ``whitespace`` strips and
        collapses runs of whitespace; ``case`` lower-cases.
    numeric_tolerance: absolute tolerance when both sides parse as numbers (default 1e-6).
    scope: ``last`` (final turn, default) or ``all`` (concatenated turns).
    reference_key: pick ``reference[key]``/``metadata[key]`` instead of the whole reference.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from monorl.core.types import RewardResult, Trajectory
from monorl.rewards.base import (
    check_trajectories,
    extract_answer,
    reference_value,
    references_as_list,
    response_text,
    reward_name,
)

_WS_RE = re.compile(r"\s+")


def _to_number(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").strip()
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


class ExactMatchReward:
    def __init__(self, params: dict[str, Any]) -> None:
        self.name = reward_name(params, "exact_match")
        self.extract = str(params.get("extract", "last_number"))
        extract_answer("", self.extract)  # validate eagerly
        normalize = params.get("normalize", ["whitespace", "case"])
        if isinstance(normalize, str):
            normalize = [normalize]
        unknown = set(normalize) - {"whitespace", "case"}
        if unknown:
            raise ValueError(f"exact_match: unknown normalize options {sorted(unknown)}")
        self.strip_ws = "whitespace" in normalize
        self.fold_case = "case" in normalize
        self.tolerance = float(params.get("numeric_tolerance", 1e-6))
        self.scope = str(params.get("scope", "last"))
        self.reference_key = params.get("reference_key")

    def _normalize(self, text: str) -> str:
        if self.strip_ws:
            text = _WS_RE.sub(" ", text).strip()
        if self.fold_case:
            text = text.lower()
        return text

    def _extract_reference(self, ref: Any) -> str:
        text = str(ref)
        extracted = extract_answer(text, self.extract)
        return extracted if extracted is not None else text

    def match(self, prediction: str | None, reference: str) -> bool:
        if prediction is None:
            return False
        p_num, r_num = _to_number(prediction), _to_number(reference)
        if p_num is not None and r_num is not None:
            return abs(p_num - r_num) <= self.tolerance
        return self._normalize(prediction) == self._normalize(reference)

    def score_text(self, text: str, reference: Any) -> tuple[float, str | None]:
        prediction = extract_answer(text, self.extract)
        for ref in references_as_list(reference):
            if ref is None:
                continue
            if self.match(prediction, self._extract_reference(ref)):
                return 1.0, prediction
        return 0.0, prediction

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        check_trajectories(trajectories)
        out = []
        for t in trajectories:
            score, prediction = self.score_text(
                response_text(t, self.scope), reference_value(t, self.reference_key)
            )
            out.append(
                RewardResult(score=score, breakdown={self.name: score}, info={"extracted": prediction})
            )
        return out


def make_exact_match(params: dict[str, Any], **_: Any) -> ExactMatchReward:
    return ExactMatchReward(params)
