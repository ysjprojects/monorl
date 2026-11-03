"""``format``: regex-based structural check.

Params:
    patterns: list of regular expressions (or a single ``pattern`` string).
    require_all: ``True`` (default) → 1.0 only when every pattern matches, else 0.0;
        ``False`` → fraction of patterns that matched.
    flags: list of ``re`` flag names (default ``["DOTALL"]``); add ``"IGNORECASE"``, ``"MULTILINE"``.
    fullmatch: ``True`` to require the pattern to match the whole response (default ``search``).
    scope: ``last`` / ``all``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from monorl.core.types import RewardResult, Trajectory
from monorl.rewards.base import check_trajectories, response_text, reward_name


class FormatReward:
    def __init__(self, params: dict[str, Any]) -> None:
        self.name = reward_name(params, "format")
        patterns = params.get("patterns")
        if patterns is None and "pattern" in params:
            patterns = [params["pattern"]]
        if not patterns:
            raise ValueError("format reward needs params.patterns (list of regexes)")
        if isinstance(patterns, str):
            patterns = [patterns]
        flags = 0
        for flag_name in params.get("flags", ["DOTALL"]):
            flags |= getattr(re, str(flag_name).upper())
        self.patterns = [re.compile(p, flags) for p in patterns]
        self.require_all = bool(params.get("require_all", True))
        self.fullmatch = bool(params.get("fullmatch", False))
        self.scope = str(params.get("scope", "last"))

    def score_text(self, text: str) -> tuple[float, list[bool]]:
        hits = [bool(p.fullmatch(text) if self.fullmatch else p.search(text)) for p in self.patterns]
        if self.require_all:
            score = 1.0 if all(hits) else 0.0
        else:
            score = sum(hits) / len(hits)
        return score, hits

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        check_trajectories(trajectories)
        out = []
        for t in trajectories:
            score, hits = self.score_text(response_text(t, self.scope))
            out.append(RewardResult(score=score, breakdown={self.name: score}, info={"matched": hits}))
        return out


def make_format(params: dict[str, Any], **_: Any) -> FormatReward:
    return FormatReward(params)
