"""The capture gateway over real localhost HTTP: wire format, recorded tokens, continuation and
threads, tool calls, streaming, concurrency and errors."""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import fakes
import pytest

from monorl.core.types import Completion, Sample, SamplingParams
from monorl.gateway import Gateway, Session, build_trajectory
from monorl.orchestrator.batching import continuation_parents, trajectory_rows
from monorl.rollout import BatchingEngine
from monorl.rollout.render import chat_prompt_ids
from tests.gateway.client import get, post, request, sse_events

CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}
CALL_TEXT = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
SAMPLE = Sample(id="s", messages=[{"role": "user", "content": "abc"}], reference="ABC")


class ParamsEngine(fakes.FakeEngine):
    """:class:`fakes.FakeEngine` that keeps the sampling parameters of every ``generate`` call."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.seen: list[SamplingParams] = []

    def generate(self, prompts: Sequence[Any], params: SamplingParams) -> list[list[Completion]]:
        self.seen.append(params)
        return super().generate(prompts, params)


def _echo_engine(make_config: Callable[..., Any], cls: type = fakes.FakeEngine) -> Any:
    """Answers every prompt with its last user message in upper case."""
    return cls(make_config(rollout={"extra": {"fake_mode": "always_right"}}))


def _scripted_engine(make_config: Callable[..., Any], script: list[str]) -> fakes.FakeEngine:
    return fakes.FakeEngine(make_config(rollout={"extra": {"fake_mode": "script", "fake_script": script}}))


def _chat_url(gateway: Gateway, session: Session) -> str:
    return f"{gateway.base_url(session)}/chat/completions"


def test_completion_is_openai_shaped_and_recorded_token_for_token(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    engine = _echo_engine(make_config, ParamsEngine)
    run_params = SamplingParams(n=1, max_new_tokens=32, temperature=0.7, stop=["<end>"])
    session = open_session(engine, params=run_params, model="policy-in-training")
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "c"}]},
        ],
        "temperature": 0.0,
        "top_p": 0.1,
        "max_tokens": 5,
        "stop": "\n\n",
    }
    status, body = post(_chat_url(gateway, session), payload)

    assert status == 200
    assert body["object"] == "chat.completion" and body["model"] == "policy-in-training"
    assert body["id"].startswith("chatcmpl-")
    (choice,) = body["choices"]
    assert choice["index"] == 0 and choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": "ABC"}

    (record,) = session.calls
    chat = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "abc"}]
    tokenizer = engine.tokenizer
    assert record.messages == chat
    assert record.prompt_ids == chat_prompt_ids(tokenizer, chat)
    assert record.completion.token_ids == [*tokenizer.encode("ABC"), tokenizer.eos_token_id]
    assert record.completion.logprobs == [-(0.1 + 0.01 * j) for j in range(4)]  # the engine's
    assert body["usage"] == {
        "prompt_tokens": len(record.prompt_ids),
        "completion_tokens": 4,
        "total_tokens": len(record.prompt_ids) + 4,
    }
    # The run owns the sampling distribution: the request only shortens the completion.
    (params,) = engine.seen
    assert (params.n, params.temperature, params.top_p) == (1, 0.7, 1.0)
    assert (params.max_new_tokens, params.stop) == (5, ["<end>", "\n\n"])

    # The same session by API key, and a request that asks for more tokens than the run allows.
    status, body = post(
        f"{gateway.url}/v1/chat/completions", {**payload, "max_completion_tokens": 999}, key=session.id
    )
    assert status == 200 and body["choices"][0]["message"]["content"] == "ABC"
    assert len(session.calls) == 2 and engine.seen[-1].max_new_tokens == 5
    post(_chat_url(gateway, session), {"messages": payload["messages"], "max_tokens": 999})
    assert engine.seen[-1].max_new_tokens == 32
    for status, models in (
        get(f"{gateway.url}/v1/models", key=session.id),
        get(f"{gateway.base_url(session)}/models"),
    ):
        assert status == 200 and [m["id"] for m in models["data"]] == ["policy-in-training"]
    assert get(f"{gateway.url}/health") == (200, {"status": "ok", "sessions": gateway.num_sessions})


def test_follow_up_is_bridged_and_a_rewritten_history_starts_a_thread(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    engine = _echo_engine(make_config)
    session = open_session(engine)
    url = _chat_url(gateway, session)

    opening = [{"role": "user", "content": "abc"}]
    answer = post(url, {"messages": opening})[1]["choices"][0]["message"]
    follow_up = [*opening, answer, {"role": "user", "content": "def"}]
    assert post(url, {"messages": follow_up})[1]["choices"][0]["message"]["content"] == "DEF"
    first, second = session.calls
    assert (second.parent, second.bridged) == (0, True)
    sampled = first.prompt_ids + first.completion.token_ids
    assert second.prompt_ids[: len(sampled)] == sampled
    assert engine.tokenizer.decode(second.prompt_ids[len(sampled) :]) == "<|user|>def<|end|><|assistant|>"

    # The agent compacts its history: no earlier call had this conversation.
    compacted = [
        {"role": "user", "content": "we spoke about abc and def"},
        {"role": "assistant", "content": "noted"},
        {"role": "user", "content": "ghi"},
    ]
    post(url, {"messages": compacted})
    third = session.calls[2]
    assert (third.parent, third.bridged) == (None, False)
    assert third.prompt_ids == chat_prompt_ids(engine.tokenizer, compacted)

    trajectory = build_trajectory(session.calls, SAMPLE, "g")
    assert trajectory.messages == [*compacted, {"role": "assistant", "content": "GHI"}]
    assert trajectory.extra_threads == [[*follow_up, {"role": "assistant", "content": "DEF"}]]
    assert [(t.thread, t.message_index) for t in trajectory.turns] == [(1, 1), (1, 3), (0, 3)]
    assert [t.info for t in trajectory.turns] == [
        {"call": 0, "parent": None, "bridged": False},
        {"call": 1, "parent": 0, "bridged": True},
        {"call": 2, "parent": None, "bridged": False},
    ]
    assert [t.completion for t in trajectory.turns] == [c.completion for c in session.calls]
    # Training sees two sequences: the bridged pair as one, the compacted context as another.
    assert continuation_parents(trajectory) == [None, 0, None]
    assert [sum(row.response_mask) for row in trajectory_rows(trajectory)] == [8, 4]

    # rollout.turn_rendering: template re-renders the continued conversation instead.
    templated = open_session(engine, turn_rendering="template")
    url = _chat_url(gateway, templated)
    answer = post(url, {"messages": opening})[1]["choices"][0]["message"]
    post(url, {"messages": [*opening, answer, {"role": "user", "content": "def"}]})
    assert (templated.calls[1].parent, templated.calls[1].bridged) == (0, False)


def test_tool_calls_are_parsed_and_their_round_trip_continues_the_call(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    engine = _scripted_engine(make_config, [f"Let me compute. {CALL_TEXT}", "The answer is 4"])
    session = open_session(engine)
    url = _chat_url(gateway, session)
    question = [{"role": "user", "content": "What is 2+2?"}]

    status, body = post(url, {"messages": question, "tools": [CALCULATOR]})
    assert status == 200 and body["choices"][0]["finish_reason"] == "tool_calls"
    message = body["choices"][0]["message"]
    assert message["content"] == "Let me compute."
    (call,) = message["tool_calls"]
    assert call["type"] == "function" and call["function"]["name"] == "calculator"
    assert json.loads(call["function"]["arguments"]) == {"expression": "2+2"}
    first = session.calls[0]
    assert first.completion.text == f"Let me compute. {CALL_TEXT}"  # the turn keeps what was sampled

    # The agent sends the assistant message back as it got it, plus the tool's result.
    history = [*question, message, {"role": "tool", "tool_call_id": call["id"], "content": "4"}]
    status, body = post(url, {"messages": history, "tools": [CALCULATOR]})
    assert body["choices"][0] == {
        "index": 0,
        "message": {"role": "assistant", "content": "The answer is 4"},
        "finish_reason": "stop",
        "logprobs": None,
    }
    second = session.calls[1]
    assert (second.parent, second.bridged) == (0, True)
    sampled = first.prompt_ids + first.completion.token_ids
    assert second.prompt_ids[: len(sampled)] == sampled
    assert engine.tokenizer.decode(second.prompt_ids[len(sampled) :]) == "<|tool|>4<|end|><|assistant|>"

    trajectory = build_trajectory(session.calls, SAMPLE, "g")
    assert trajectory.turns[0].info == {
        "call": 0,
        "parent": None,
        "bridged": False,
        "tool_calls": [
            {"id": call["id"], "name": "calculator", "arguments": {"expression": "2+2"}, "result": "4"}
        ],
        "invalid_calls": 0,
    }
    assert trajectory.turns[1].info["tool_calls"] == []
    assert trajectory.messages == [
        *question,
        {
            "role": "assistant",
            "content": "Let me compute.",
            "tool_calls": [{"id": call["id"], "name": "calculator", "arguments": {"expression": "2+2"}}],
        },
        {"role": "tool", "content": "4", "tool_call_id": call["id"], "name": "calculator"},
        {"role": "assistant", "content": "The answer is 4"},
    ]

    # Without tools the same history is another context, and tool-call markup stays text.
    status, body = post(url, {"messages": [*history, body["choices"][0]["message"], question[0]]})
    assert body["choices"][0]["message"] == {"role": "assistant", "content": f"Let me compute. {CALL_TEXT}"}
    assert body["choices"][0]["finish_reason"] == "stop" and session.calls[2].parent is None


def test_tool_round_trip_with_a_real_chat_template(
    tiny_model_dir: Path, gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    """ChatML (SmolLM) concatenates ``message['content']``: the ``null`` content of a tool-call
    answer has to reach the template as text, and the sampled call tokens must stay in context."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)

    class Scripted:
        def __init__(self, script: list[str]) -> None:
            self.tokenizer = tokenizer
            self.script = script

        def generate(self, prompts: Sequence[Any], params: SamplingParams) -> list[list[Completion]]:
            out = []
            for prompt in prompts:
                text = self.script.pop(0)
                ids = [*tokenizer(text, add_special_tokens=False)["input_ids"], tokenizer.eos_token_id]
                out.append([Completion(text, ids, [-0.5] * len(ids), "stop", list(prompt))])
            return out

    session = open_session(Scripted([CALL_TEXT, "It is 4."]))
    url = _chat_url(gateway, session)
    question = [{"role": "system", "content": "Use the calculator."}, {"role": "user", "content": "2+2?"}]

    message = post(url, {"messages": question, "tools": [CALCULATOR]})[1]["choices"][0]["message"]
    assert message["content"] is None and message["tool_calls"][0]["function"]["name"] == "calculator"
    result = {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "4"}
    status, body = post(url, {"messages": [*question, message, result], "tools": [CALCULATOR]})
    assert status == 200 and body["choices"][0]["message"]["content"] == "It is 4."

    first, second = session.calls
    assert first.prompt_ids == chat_prompt_ids(tokenizer, question, [CALCULATOR])
    sampled = first.prompt_ids + first.completion.token_ids
    assert second.bridged and second.prompt_ids[: len(sampled)] == sampled
    assert tokenizer.decode(second.prompt_ids[len(sampled) :]) == (
        "\n<|im_start|>tool\n4<|im_end|>\n<|im_start|>assistant\n"
    )


def test_openai_sdk_agent_round_trip(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    openai = pytest.importorskip("openai")
    import httpx

    engine = _scripted_engine(make_config, [f"Let me compute. {CALL_TEXT}", "The answer is 4", "streamed"])
    session = open_session(engine, model="policy-in-training")

    def client(base_url: str, api_key: str) -> Any:
        http = httpx.Client(trust_env=False)  # no proxy, whatever the machine is configured with
        return openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0, http_client=http)

    by_path = client(gateway.base_url(session), "unused")
    messages: list[Any] = [{"role": "user", "content": "What is 2+2?"}]
    response = by_path.chat.completions.create(model="any", messages=messages, tools=[CALCULATOR])
    message = response.choices[0].message
    assert response.choices[0].finish_reason == "tool_calls" and message.content == "Let me compute."
    (call,) = message.tool_calls
    assert call.function.name == "calculator" and json.loads(call.function.arguments) == {"expression": "2+2"}

    # Agents written against the SDK put its message object straight back into the history.
    messages += [message, {"role": "tool", "tool_call_id": call.id, "content": "4"}]
    response = by_path.chat.completions.create(model="any", messages=messages, tools=[CALCULATOR])
    assert response.choices[0].message.content == "The answer is 4"
    first, second = session.calls
    assert (second.parent, second.bridged) == (0, True)
    assert response.usage.prompt_tokens == len(second.prompt_ids)
    assert second.prompt_ids[: len(first.prompt_ids)] == first.prompt_ids

    by_key = client(f"{gateway.url}/v1", session.id)
    stream = by_key.chat.completions.create(
        model="any", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    assert "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "streamed"
    assert [model.id for model in by_key.models.list()] == ["policy-in-training"]
    with pytest.raises(openai.BadRequestError, match="n must be 1"):
        by_path.chat.completions.create(model="any", messages=messages, n=2)
    with pytest.raises(openai.AuthenticationError):
        client(f"{gateway.url}/v1", "not-a-session").models.list()


def test_streaming_is_the_finished_completion_as_sse(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    session = open_session(_echo_engine(make_config))
    payload = {"messages": [{"role": "user", "content": "stream me"}]}
    plain = post(_chat_url(gateway, session), payload)[1]
    status, text = request(
        "POST",
        _chat_url(gateway, session),
        {**payload, "stream": True, "stream_options": {"include_usage": True}},
    )
    assert status == 200
    *chunks, done = sse_events(text)
    assert done == "[DONE]" and all(c["object"] == "chat.completion.chunk" for c in chunks)
    deltas = [c["choices"][0] for c in chunks if c["choices"]]
    assert deltas[0]["delta"]["role"] == "assistant"
    assert "".join(d["delta"].get("content") or "" for d in deltas) == "STREAM ME"
    assert plain["choices"][0]["message"]["content"] == "STREAM ME"
    assert [d["finish_reason"] for d in deltas] == [None] * (len(deltas) - 1) + ["stop"]
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == plain["usage"]

    # Tool calls are streamed whole, as in the plain answer; no usage chunk unless asked for.
    session = open_session(_scripted_engine(make_config, [f"On it. {CALL_TEXT}"]))
    payload = {"messages": [{"role": "user", "content": "2+2?"}], "tools": [CALCULATOR]}
    plain = post(_chat_url(gateway, session), payload)[1]["choices"][0]["message"]
    *chunks, done = sse_events(request("POST", _chat_url(gateway, session), {**payload, "stream": True})[1])
    deltas = [c["choices"][0] for c in chunks]
    assert "".join(d["delta"].get("content") or "" for d in deltas) == plain["content"] == "On it."
    (streamed,) = [call for d in deltas for call in d["delta"].get("tool_calls", [])]
    assert streamed["index"] == 0 and streamed["function"] == plain["tool_calls"][0]["function"]
    assert deltas[-1]["finish_reason"] == "tool_calls" and all("usage" not in c for c in chunks)


def test_128_calls_in_flight_reach_the_engine_as_one_batch(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    fake = _echo_engine(make_config)
    # The batch is only released early when 128 prompts are waiting at the same time.
    engine = BatchingEngine(fake, max_batch_size=128, max_wait_ms=30_000)
    try:
        sessions = [open_session(engine) for _ in range(128)]

        def call(i: int) -> tuple[int, Any]:
            return post(_chat_url(gateway, sessions[i]), {"messages": [{"role": "user", "content": f"q{i}"}]})

        with ThreadPoolExecutor(max_workers=128) as pool:
            results = list(pool.map(call, range(128)))
    finally:
        engine.shutdown()
    assert [status for status, _ in results] == [200] * 128
    assert [body["choices"][0]["message"]["content"] for _, body in results] == [f"Q{i}" for i in range(128)]
    assert [c["num_prompts"] for c in fake.calls] == [128]
    assert all(len(s.calls) == 1 for s in sessions)


def test_unserved_requests_get_openai_errors(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    engine = _echo_engine(make_config)
    session = open_session(engine, max_calls=1)
    url = _chat_url(gateway, session)
    messages = [{"role": "user", "content": "abc"}]

    status, body = post(f"{gateway.url}/s/nope/v1/chat/completions", {"messages": messages})
    assert status == 404 and body["error"]["code"] == "session_not_found"
    assert set(body["error"]) == {"message", "type", "param", "code"}
    status, body = post(f"{gateway.url}/v1/chat/completions", {"messages": messages}, key="nope")
    assert status == 401 and body["error"]["code"] == "invalid_api_key"
    assert get(f"{gateway.url}/v1/models")[0] == 401

    status, body = post(url, {"messages": messages, "n": 2})
    assert (
        status == 400 and body["error"]["param"] == "n" and body["error"]["type"] == "invalid_request_error"
    )
    picture = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    status, body = post(url, {"messages": [{"role": "user", "content": [picture]}]})
    assert status == 400 and "image" in body["error"]["message"]
    status, body = post(url, {"messages": []})
    assert status == 400 and body["error"]["param"] == "messages"
    assert request("POST", url, None)[0] == 400  # no JSON body
    assert get(f"{gateway.url}/v1/nothing")[1]["error"]["message"] == "Not Found"
    assert session.calls == [] and engine.calls == []

    # max_calls: the first call is served, the next one refused without reaching the engine.
    assert post(url, {"messages": messages})[0] == 200
    status, body = post(url, {"messages": messages})
    assert status == 400 and body["error"]["code"] == "max_turns_exceeded"
    assert len(session.calls) == 1 and session.refused == 1 and len(engine.calls) == 1

    gateway.close_session(session)
    status, body = post(url, {"messages": messages})
    assert status == 503 and body["error"]["code"] == "session_closed"
    status, body = post(gateway.reward_url(session), {"reward": 1.0})
    assert status == 503 and session.reward is None


def test_cancelled_generation_answers_503(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    engine = fakes.SlowCancellableEngine(make_config())  # generates for 60 s unless cancelled
    session = open_session(engine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(post, _chat_url(gateway, session), {"messages": SAMPLE.messages})
        assert engine.generating.wait(30)
        engine.cancel()
        status, body = pending.result(timeout=30)
    assert status == 503 and body["error"]["code"] == "generation_cancelled"
    assert session.cancelled.is_set() and session.calls == []
    # The episode is being aborted: a retry does not start another generation.
    assert post(_chat_url(gateway, session), {"messages": SAMPLE.messages})[0] == 503


def test_posted_rewards_add_up(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    session = open_session(_echo_engine(make_config))
    assert post(gateway.reward_url(session), {"reward": 0.5}) == (200, {"reward": 0.5})
    assert post(gateway.reward_url(session), {"reward": 1}) == (200, {"reward": 1.5})
    status, body = post(gateway.reward_url(session), {"reward": "high"})
    assert status == 400 and body["error"]["param"] == "reward"
    assert session.reward == 1.5


def test_a_stopped_gateway_closes_its_sessions_and_its_port(make_config: Callable[..., Any]) -> None:
    engine = _echo_engine(make_config)
    own = Gateway().start()
    session = own.open_session(engine=engine, tokenizer=engine.tokenizer, params=SamplingParams())
    assert post(_chat_url(own, session), {"messages": SAMPLE.messages})[0] == 200
    own.stop()
    assert session.closed and not own.running
    with pytest.raises(urllib.error.URLError):
        get(f"{own.url}/health")
