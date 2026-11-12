from __future__ import annotations

import json

import httpx
import pytest

from monorl.core.registry import registry
from monorl.rewards.llm_judge import parse_score


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://placeholder")


def test_parse_score():
    assert parse_score("Reasoning...\n<score>7</score>") == 7.0
    assert parse_score("<SCORE> 3.5 </SCORE>") == 3.5
    assert parse_score("<score>1</score> then <score>9</score>") == 9.0
    assert parse_score("no tag") is None


def test_llm_judge_scores_via_mock_endpoint(traj, monkeypatch):
    monkeypatch.setenv("JUDGE_KEY", "sk-test")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        content = body["messages"][-1]["content"]
        score = 10 if "correct answer" in content else 2
        return httpx.Response(
            200, json={"choices": [{"message": {"content": f"Looks fine.\n<score>{score}</score>"}}]}
        )

    fn = registry.get("reward", "llm_judge")(
        {
            "base_url": "http://judge.local/v1/",
            "model": "judge-1",
            "api_key_env": "JUDGE_KEY",
            "rubric": "Q: {prompt}\nRef: {reference}\nA: {response}\nScore 0-{max_score}.",
            "system_prompt": "Be strict.",
            "max_workers": 2,
        },
        client=_mock_client(handler),
    )
    results = fn([traj("the correct answer", "42", prompt="What?"), traj("garbage", "42", prompt="What?")])
    assert [r.score for r in results] == [1.0, 0.2]
    assert results[0].breakdown == {"llm_judge": 1.0}
    assert results[0].info["raw_score"] == 10.0
    assert len(seen) == 2
    req = seen[0]
    assert str(req.url) == "http://judge.local/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["model"] == "judge-1" and body["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": "Be strict."}
    assert "Q: What?\nRef: 42\nA: " in body["messages"][1]["content"]
    assert "Score 0-10." in body["messages"][1]["content"]


def test_llm_judge_raw_scores_and_clamping(traj):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "<score>12</score>"}}]})

    fn = registry.get("reward", "llm_judge")(
        {"base_url": "http://j", "model": "m", "normalize": False, "max_score": 5},
        client=_mock_client(handler),
    )
    assert fn([traj("x")])[0].score == 5.0


def test_llm_judge_error_handling(traj):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"choices": [{"message": {"content": "no tag here"}}]})

    zero = registry.get("reward", "llm_judge")(
        {"base_url": "http://j", "model": "m", "max_workers": 1}, client=_mock_client(handler)
    )
    results = zero([traj("a"), traj("b")])
    assert [r.score for r in results] == [0.0, 0.0]
    assert "HTTPStatusError" in results[0].info["error"]
    assert results[1].info["error"] == "no_score"

    strict = registry.get("reward", "llm_judge")(
        {"base_url": "http://j", "model": "m", "on_error": "raise"},
        client=_mock_client(lambda r: httpx.Response(503)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        strict([traj("a")])


def test_llm_judge_requires_endpoint_params():
    with pytest.raises(ValueError):
        registry.get("reward", "llm_judge")({"model": "m"})
