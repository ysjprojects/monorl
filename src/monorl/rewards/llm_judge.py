"""``llm_judge``: score responses with an OpenAI-compatible chat-completions endpoint.

Params:
    base_url: e.g. ``https://api.openai.com/v1`` or a local vLLM/SGLang server (required).
    model: judge model name (required).
    api_key_env: environment variable holding the bearer token (default ``OPENAI_API_KEY``;
        missing variable → no ``Authorization`` header, which local servers accept).
    rubric: prompt template with ``{prompt}``, ``{response}``, ``{reference}`` placeholders;
        the judge must answer with ``<score>N</score>``.
    system_prompt: optional system message.    max_score: scale of ``N`` (default 10).
    normalize: divide the score by ``max_score`` (default ``True`` → reward in ``[0, 1]``).
    temperature (0.0), max_tokens (512), timeout_s (60), max_workers (8),
    extra_body: dict merged into the request body.
    on_error: ``zero`` (default; score 0 with ``info["error"]``) or ``raise``.

Factory kwargs: ``client=httpx.Client(...)`` injects a transport (tests use ``MockTransport``).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from monorl.core.types import RewardResult, Trajectory
from monorl.rewards.base import check_trajectories, prompt_text, reference_value, response_text, reward_name

DEFAULT_RUBRIC = """You are a strict grader. Evaluate the response to the task below.

[Task]
{prompt}

[Reference answer]
{reference}

[Response]
{response}

Judge correctness and quality against the reference. Reply with a brief justification followed by
a final line of the form <score>N</score> where N is an integer from 0 to {max_score}."""

_SCORE_RE = re.compile(r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)


def parse_score(text: str) -> float | None:
    matches = _SCORE_RE.findall(text)
    return float(matches[-1]) if matches else None


class LLMJudgeReward:
    def __init__(self, params: dict[str, Any], *, client: httpx.Client | None = None) -> None:
        self.name = reward_name(params, "llm_judge")
        base_url = params.get("base_url")
        model = params.get("model")
        if not base_url or not model:
            raise ValueError("llm_judge needs params.base_url and params.model")
        self.model = str(model)
        self.rubric = str(params.get("rubric", DEFAULT_RUBRIC))
        self.system_prompt = params.get("system_prompt")
        self.max_score = float(params.get("max_score", 10))
        self.normalize = bool(params.get("normalize", True))
        self.temperature = float(params.get("temperature", 0.0))
        self.max_tokens = int(params.get("max_tokens", 512))
        self.max_workers = int(params.get("max_workers", 8))
        self.extra_body = dict(params.get("extra_body", {}))
        self.on_error = str(params.get("on_error", "zero"))
        if self.on_error not in ("zero", "raise"):
            raise ValueError("llm_judge: on_error must be 'zero' or 'raise'")
        self.scope = str(params.get("scope", "last"))
        self.reference_key = params.get("reference_key")

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(str(params.get("api_key_env", "OPENAI_API_KEY")))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if client is None:
            client = httpx.Client(
                base_url=str(base_url).rstrip("/"), timeout=float(params.get("timeout_s", 60))
            )
        else:
            client.base_url = httpx.URL(str(base_url).rstrip("/"))
        client.headers.update(headers)
        self.client = client

    def build_messages(self, trajectory: Trajectory) -> list[dict[str, str]]:
        reference = reference_value(trajectory, self.reference_key)
        user = self.rubric.format(
            prompt=prompt_text(trajectory),
            response=response_text(trajectory, self.scope),
            reference="" if reference is None else str(reference),
            max_score=int(self.max_score) if self.max_score.is_integer() else self.max_score,
        )
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": str(self.system_prompt)})
        messages.append({"role": "user", "content": user})
        return messages

    def judge(self, trajectory: Trajectory) -> RewardResult:
        body = {
            "model": self.model,
            "messages": self.build_messages(trajectory),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        try:
            resp = self.client.post("/chat/completions", json=body)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
            if self.on_error == "raise":
                raise
            return RewardResult(
                score=0.0, breakdown={self.name: 0.0}, info={"error": f"{type(e).__name__}: {e}"}
            )
        raw = parse_score(content)
        if raw is None:
            if self.on_error == "raise":
                raise ValueError(f"llm_judge: no <score> tag in judge output: {content[:200]!r}")
            return RewardResult(
                score=0.0, breakdown={self.name: 0.0}, info={"error": "no_score", "judge_output": content}
            )
        raw = min(max(raw, 0.0), self.max_score)
        score = raw / self.max_score if self.normalize else raw
        return RewardResult(
            score=score, breakdown={self.name: score}, info={"raw_score": raw, "judge_output": content}
        )

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        check_trajectories(trajectories)
        if len(trajectories) <= 1 or self.max_workers <= 1:
            return [self.judge(t) for t in trajectories]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self.judge, trajectories))


def make_llm_judge(params: dict[str, Any], *, client: httpx.Client | None = None, **_: Any) -> LLMJudgeReward:
    return LLMJudgeReward(params, client=client)
