"""The OpenAI Responses and Anthropic Messages APIs of the gateway, driven by the vendors' SDK
clients and by raw HTTP: tool round trips that continue the earlier call token for token,
``previous_response_id``, streaming and its keep-alives, the context budget and the error shapes."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import fakes
import httpx
import pytest

from monorl.core.types import Completion, Sample, SamplingParams
from monorl.gateway import Gateway, Session, build_trajectory
from monorl.rollout.render import chat_prompt_ids
from tests.gateway.client import post, request, sse_events, sse_named

PARAMETERS = {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
DESCRIPTION = "Evaluate an arithmetic expression."
CHAT_TOOL = {
    "type": "function",
    "function": {"name": "calculator", "description": DESCRIPTION, "parameters": PARAMETERS},
}
RESPONSES_TOOL = {
    "type": "function",
    "name": "calculator",
    "description": DESCRIPTION,
    "parameters": PARAMETERS,
}
MESSAGES_TOOL = {"name": "calculator", "description": DESCRIPTION, "input_schema": PARAMETERS}
CALL_TEXT = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
SYSTEM = {"role": "system", "content": "Use the calculator."}
QUESTION = {"role": "user", "content": "What is 2+2?"}
SAMPLE = Sample(id="s", messages=[QUESTION], reference="4")


class ParamsEngine(fakes.FakeEngine):
    """:class:`fakes.FakeEngine` that keeps the sampling parameters of every ``generate`` call."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.seen: list[SamplingParams] = []

    def generate(self, prompts: Sequence[Any], params: SamplingParams) -> list[list[Completion]]:
        self.seen.append(params)
        return super().generate(prompts, params)


class GatedEngine(fakes.FakeEngine):
    """:class:`fakes.FakeEngine` whose generations wait until ``release`` is set."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.release = threading.Event()

    def generate(self, prompts: Sequence[Any], params: SamplingParams) -> list[list[Completion]]:
        assert self.release.wait(60)
        return super().generate(prompts, params)


class FailingEngine(fakes.FakeEngine):
    def generate(self, prompts: Sequence[Any], params: SamplingParams) -> list[list[Completion]]:
        raise RuntimeError("boom")


def _echo_engine(make_config: Callable[..., Any], cls: type = fakes.FakeEngine, **extra: Any) -> Any:
    """Answers every prompt with its last user message in upper case."""
    return cls(make_config(rollout={"extra": {"fake_mode": "always_right", **extra}}))


def _scripted_engine(make_config: Callable[..., Any], script: list[str]) -> fakes.FakeEngine:
    return fakes.FakeEngine(make_config(rollout={"extra": {"fake_mode": "script", "fake_script": script}}))


@pytest.fixture
def openai_client() -> Callable[..., Any]:
    """``openai_client(base_url, api_key="unused", **options)``: the OpenAI SDK's client."""
    openai = pytest.importorskip("openai")

    def client(base_url: str, api_key: str = "unused", **options: Any) -> Any:
        http = httpx.Client(trust_env=False)  # no proxy, whatever the machine is configured with
        return openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0, http_client=http, **options)

    return client


@pytest.fixture
def anthropic_client() -> Callable[..., Any]:
    """``anthropic_client(base_url, api_key="unused", **options)``: the Anthropic SDK's client."""
    anthropic = pytest.importorskip("anthropic")

    def client(base_url: str, api_key: str = "unused", **options: Any) -> Any:
        http = anthropic.DefaultHttpxClient(trust_env=False)
        return anthropic.Anthropic(
            base_url=base_url, api_key=api_key, max_retries=0, http_client=http, **options
        )

    return client


@pytest.fixture
def quick_gateway() -> Iterator[Gateway]:
    """A gateway of its own whose streams send a keep-alive every 50 ms."""
    gateway = Gateway(keepalive_s=0.05).start()
    yield gateway
    gateway.stop()


def _continues(session: Session, tail: str) -> None:
    """The session's second call continues its first token for token, adding ``tail``."""
    first, second = session.calls
    assert (second.parent, second.bridged) == (0, True)
    sampled = first.prompt_ids + first.completion.token_ids
    assert second.prompt_ids[: len(sampled)] == sampled
    assert session.tokenizer.decode(second.prompt_ids[len(sampled) :]) == tail


# -- OpenAI Responses ---------------------------------------------------------------------------


def test_responses_tool_round_trip_continues_the_call(
    make_config: Callable[..., Any],
    gateway: Gateway,
    open_session: Callable[..., Session],
    openai_client: Callable[..., Any],
) -> None:
    script = [f"Let me compute. {CALL_TEXT}", "The answer is 4"]
    engine = _scripted_engine(make_config, script)
    session = open_session(engine, model="policy-in-training")
    client = openai_client(gateway.base_url(session))
    asked = {"model": "any", "instructions": SYSTEM["content"], "tools": [RESPONSES_TOOL]}

    response = client.responses.create(input=QUESTION["content"], temperature=0.0, top_p=0.1, **asked)
    assert response.object == "response" and response.status == "completed"
    assert response.id.startswith("resp_") and response.model == "policy-in-training"
    assert response.temperature == 1.0 and response.top_p == 1.0  # the run's: what was sampled with
    message, call = response.output
    assert message.type == "message" and message.role == "assistant" and message.status == "completed"
    assert response.output_text == "Let me compute."
    assert call.type == "function_call" and call.name == "calculator"
    assert json.loads(call.arguments) == {"expression": "2+2"}
    first = session.calls[0]
    assert first.completion.text == script[0]  # the turn keeps what was sampled
    # instructions and flat function tools are the system message and tool schemas of a chat
    assert first.prompt_ids == chat_prompt_ids(engine.tokenizer, [SYSTEM, QUESTION], [CHAT_TOOL])

    # Stateless agents send the output items back, followed by the function's output.
    history = [
        QUESTION,
        *response.output,
        {"type": "function_call_output", "call_id": call.call_id, "output": "4"},
    ]
    final = client.responses.create(input=history, **asked)
    assert final.output_text == "The answer is 4" and [item.type for item in final.output] == ["message"]
    _continues(session, "<|tool|>4<|end|><|assistant|>")
    second = session.calls[1]
    assert (final.usage.input_tokens, final.usage.output_tokens) == (
        len(second.prompt_ids),
        len(second.completion.token_ids),
    )
    assert final.usage.total_tokens == final.usage.input_tokens + final.usage.output_tokens

    trajectory = build_trajectory(session.calls, SAMPLE, "g")
    assert trajectory.info["tools"] == [CHAT_TOOL]
    assert trajectory.turns[0].info["tool_calls"] == [
        {"id": call.call_id, "name": "calculator", "arguments": {"expression": "2+2"}, "result": "4"}
    ]
    assert trajectory.messages[2:] == [
        {
            "role": "assistant",
            "content": "Let me compute.",
            "tool_calls": [{"id": call.call_id, "name": "calculator", "arguments": {"expression": "2+2"}}],
        },
        {"role": "tool", "content": "4", "tool_call_id": call.call_id, "name": "calculator"},
        {"role": "assistant", "content": "The answer is 4"},
    ]

    # The same round trip through previous_response_id reaches the policy with the same tokens.
    stateful = open_session(_scripted_engine(make_config, script))
    client = openai_client(gateway.base_url(stateful))
    response = client.responses.create(input=[QUESTION], **asked)
    output = {"type": "function_call_output", "call_id": response.output[1].call_id, "output": "4"}
    final = client.responses.create(previous_response_id=response.id, input=[output], **asked)
    assert final.output_text == "The answer is 4" and final.previous_response_id == response.id
    _continues(stateful, "<|tool|>4<|end|><|assistant|>")
    assert [c.prompt_ids for c in stateful.calls] == [c.prompt_ids for c in session.calls]


def test_previous_response_id_continues_an_earlier_call(
    make_config: Callable[..., Any],
    gateway: Gateway,
    open_session: Callable[..., Session],
    openai_client: Callable[..., Any],
) -> None:
    openai = pytest.importorskip("openai")
    session = open_session(_echo_engine(make_config))
    client = openai_client(f"{gateway.url}/v1", session.id)  # the session as API key
    brief = {"role": "system", "content": "be brief"}

    first = client.responses.create(model="any", instructions="be brief", input="abc")
    second = client.responses.create(
        model="any", instructions="be brief", previous_response_id=first.id, input="def"
    )
    assert (first.output_text, second.output_text) == ("ABC", "DEF")
    _continues(session, "<|user|>def<|end|><|assistant|>")
    spoken = [
        {"role": "user", "content": "abc"},
        {"role": "assistant", "content": "ABC"},
        {"role": "user", "content": "def"},
    ]
    assert session.calls[1].messages == [brief, *spoken]

    # As with OpenAI, the earlier instructions are not inherited: without them the conversation
    # no longer starts the way the earlier calls did, and it is rendered as a context of its own.
    third = client.responses.create(model="any", previous_response_id=second.id, input="ghi")
    assert third.output_text == "GHI" and third.instructions is None
    record = session.calls[2]
    assert record.messages == [
        *spoken,
        {"role": "assistant", "content": "DEF"},
        {"role": "user", "content": "ghi"},
    ]
    assert record.parent is None and record.prompt_ids == chat_prompt_ids(session.tokenizer, record.messages)

    with pytest.raises(openai.BadRequestError) as refused:
        client.responses.create(model="any", previous_response_id="resp_unknown", input="x")
    assert refused.value.code == "previous_response_not_found"
    assert refused.value.param == "previous_response_id" and len(session.calls) == 3


def test_responses_stream_is_the_plain_response_as_events(
    make_config: Callable[..., Any],
    gateway: Gateway,
    open_session: Callable[..., Session],
    openai_client: Callable[..., Any],
) -> None:
    session = open_session(_scripted_engine(make_config, [f"On it. {CALL_TEXT}"]))
    client = openai_client(gateway.base_url(session))
    asked = {"model": "any", "input": "2+2?", "tools": [RESPONSES_TOOL], "max_output_tokens": 20}

    plain = client.responses.create(**asked)
    events = list(client.responses.create(stream=True, **asked))
    assert [event.sequence_number for event in events] == list(range(len(events)))
    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[0].response.status == "in_progress" and events[0].response.output == []
    streamed = events[-1].response
    assert streamed.id == events[0].response.id != plain.id

    def answer(response: Any) -> Any:
        message, call = response.output
        return (response.status, response.output_text, call.name, call.arguments, response.usage)

    assert answer(streamed) == answer(plain)
    assert answer(plain)[:4] == ("completed", "On it.", "calculator", '{"expression": "2+2"}')
    assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == "On it."
    arguments = [e.delta for e in events if e.type == "response.function_call_arguments.delta"]
    assert "".join(arguments) == plain.output[1].arguments
    assert events[7].item == streamed.output[0] and events[11].item == streamed.output[1]

    # The SDK's stream helper builds the same response from the events.
    with client.responses.stream(**asked) as stream:
        deltas = [event.delta for event in stream if event.type == "response.output_text.delta"]
        assert "".join(deltas) == "On it."
        assert answer(stream.get_final_response()) == answer(plain)

    # A completion cut off by the length limit is an incomplete response, streamed or not.
    cut = open_session(_echo_engine(make_config, fake_finish="length"))
    client = openai_client(gateway.base_url(cut))
    plain = client.responses.create(model="any", input="abc")
    assert plain.status == "incomplete" and plain.incomplete_details.reason == "max_output_tokens"
    assert plain.output_text == "ABC" and plain.output[0].status == "incomplete"
    last = list(client.responses.create(model="any", input="abc", stream=True))[-1]
    assert last.type == "response.incomplete" and last.response.output_text == "ABC"


# -- Anthropic Messages -------------------------------------------------------------------------


def test_messages_tool_round_trip_continues_the_call(
    make_config: Callable[..., Any],
    gateway: Gateway,
    open_session: Callable[..., Session],
    anthropic_client: Callable[..., Any],
) -> None:
    script = [f"Let me compute. {CALL_TEXT}", "The answer is 4"]
    engine = _scripted_engine(make_config, script)
    session = open_session(engine, model="policy-in-training")
    client = anthropic_client(gateway.anthropic_base_url(session))
    asked = {"model": "any", "max_tokens": 64, "system": SYSTEM["content"], "tools": [MESSAGES_TOOL]}

    sampling = {"temperature": 0.0, "top_k": 1}  # not in the SDK's typed parameters; ignored anyway
    message = client.messages.create(messages=[QUESTION], extra_body=sampling, **asked)
    assert message.type == "message" and message.role == "assistant"
    assert message.id.startswith("msg_") and message.model == "policy-in-training"
    assert message.stop_reason == "tool_use" and message.stop_sequence is None
    text, tool_use = message.content
    assert text.type == "text" and text.text == "Let me compute."
    assert tool_use.type == "tool_use" and tool_use.name == "calculator"
    assert tool_use.input == {"expression": "2+2"}
    first = session.calls[0]
    assert first.completion.text == script[0]
    # system and input_schema tools are the system message and tool schemas of a chat
    assert first.prompt_ids == chat_prompt_ids(engine.tokenizer, [SYSTEM, QUESTION], [CHAT_TOOL])

    # The agent sends the content blocks back as it got them, plus the tool's result.
    result = {"type": "tool_result", "tool_use_id": tool_use.id, "content": [{"type": "text", "text": "4"}]}
    history = [
        QUESTION,
        {"role": "assistant", "content": message.content},
        {"role": "user", "content": [result]},
    ]
    counted = client.messages.count_tokens(
        model="any", system=SYSTEM["content"], tools=[MESSAGES_TOOL], messages=history
    )
    assert len(session.calls) == 1  # counting is not a call
    final = client.messages.create(messages=history, **asked)
    assert [block.text for block in final.content] == ["The answer is 4"]
    assert final.stop_reason == "end_turn"
    _continues(session, "<|tool|>4<|end|><|assistant|>")
    second = session.calls[1]
    assert counted.input_tokens == final.usage.input_tokens == len(second.prompt_ids)
    assert final.usage.output_tokens == len(second.completion.token_ids)

    trajectory = build_trajectory(session.calls, SAMPLE, "g")
    assert trajectory.info["tools"] == [CHAT_TOOL]
    assert trajectory.turns[0].info["tool_calls"] == [
        {"id": tool_use.id, "name": "calculator", "arguments": {"expression": "2+2"}, "result": "4"}
    ]
    assert trajectory.messages[3] == {
        "role": "tool",
        "content": "4",
        "tool_call_id": tool_use.id,
        "name": "calculator",
    }


def test_messages_honours_stop_sequences_and_max_tokens_only(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    engine = _echo_engine(make_config, ParamsEngine)
    run_params = SamplingParams(n=1, max_new_tokens=32, temperature=0.7, stop=["<end>"])
    session = open_session(engine, params=run_params)
    payload = {
        "model": "claude-any",
        "max_tokens": 5,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "c"}]}
        ],
        "stop_sequences": ["C"],
        "temperature": 0.0,
        "top_k": 1,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    # the session as x-api-key, as an Anthropic client configured with an API key sends it
    status, body = post(f"{gateway.url}/v1/messages", payload, headers={"x-api-key": session.id})
    assert status == 200
    # the answer ends before the stop sequence, which is named
    assert body["content"] == [{"type": "text", "text": "AB"}]
    assert (body["stop_reason"], body["stop_sequence"]) == ("stop_sequence", "C")
    (record,) = session.calls
    assert record.messages == [{"role": "user", "content": "abc"}] and record.completion.text == "ABC"
    assert body["usage"]["input_tokens"] == len(record.prompt_ids)
    assert body["usage"]["output_tokens"] == len(record.completion.token_ids)
    # The run owns the sampling distribution: the request only shortens the completion.
    (params,) = engine.seen
    assert (params.n, params.temperature, params.top_k) == (1, 0.7, -1)
    assert (params.max_new_tokens, params.stop) == (5, ["<end>", "C"])

    status, body = post(f"{gateway.anthropic_base_url(session)}/v1/messages", {**payload, "max_tokens": 999})
    assert status == 200 and engine.seen[-1].max_new_tokens == 32


def test_messages_stream_is_the_plain_message_as_events(
    make_config: Callable[..., Any],
    gateway: Gateway,
    open_session: Callable[..., Session],
    anthropic_client: Callable[..., Any],
) -> None:
    session = open_session(_scripted_engine(make_config, [f"On it. {CALL_TEXT}"]))
    client = anthropic_client(f"{gateway.url}", session.id)  # the session as API key
    asked = {
        "model": "any",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "2+2?"}],
        "tools": [MESSAGES_TOOL],
    }

    plain = client.messages.create(**asked)
    events = list(client.messages.create(stream=True, **asked))
    assert [event.type for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[0].message
    assert start.content == [] and start.stop_reason is None
    assert start.usage.input_tokens == plain.usage.input_tokens
    assert events[2].delta.text == "On it." and events[4].content_block.input == {}
    assert json.loads(events[5].delta.partial_json) == {"expression": "2+2"}
    assert events[7].delta.stop_reason == "tool_use"
    assert events[7].usage.output_tokens == plain.usage.output_tokens

    def answer(message: Any) -> Any:
        text, tool_use = message.content
        usage = (message.usage.input_tokens, message.usage.output_tokens)
        return (text.text, tool_use.name, tool_use.input, message.stop_reason, usage)

    with client.messages.stream(**asked) as stream:
        assert "".join(stream.text_stream) == "On it."
        final = stream.get_final_message()
    assert answer(final) == answer(plain)
    assert answer(plain)[:4] == ("On it.", "calculator", {"expression": "2+2"}, "tool_use")


# -- streaming, the context budget, errors ------------------------------------------------------


def test_chat_stream_is_the_plain_completion_as_chunks(
    make_config: Callable[..., Any],
    gateway: Gateway,
    open_session: Callable[..., Session],
    openai_client: Callable[..., Any],
) -> None:
    session = open_session(_scripted_engine(make_config, [f"On it. {CALL_TEXT}"]))
    client = openai_client(gateway.base_url(session))
    asked = {"model": "any", "messages": [{"role": "user", "content": "2+2?"}], "tools": [CHAT_TOOL]}

    plain = client.chat.completions.create(**asked)
    chunks = list(
        client.chat.completions.create(stream=True, stream_options={"include_usage": True}, **asked)
    )
    deltas = [chunk.choices[0] for chunk in chunks if chunk.choices]
    (streamed,) = [call for choice in deltas for call in choice.delta.tool_calls or []]
    message = plain.choices[0].message
    assert "".join(choice.delta.content or "" for choice in deltas) == message.content == "On it."
    called = message.tool_calls[0].function
    assert (streamed.function.name, streamed.function.arguments) == (called.name, called.arguments)
    assert called.name == "calculator"
    assert deltas[-1].finish_reason == plain.choices[0].finish_reason == "tool_calls"
    assert chunks[-1].usage == plain.usage


@pytest.mark.parametrize(
    ("path", "payload", "opening", "keepalive"),
    [
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "abc"}]},
            "data: ",
            ": keep-alive",
        ),
        ("/v1/responses", {"input": "abc"}, "event: response.created", ": keep-alive"),
        (
            "/v1/messages",
            {"max_tokens": 16, "messages": [{"role": "user", "content": "abc"}]},
            "event: message_start",
            "event: ping",
        ),
    ],
)
def test_a_stream_opens_at_once_and_is_kept_alive_until_the_generation_is_done(
    make_config: Callable[..., Any],
    quick_gateway: Gateway,
    path: str,
    payload: dict[str, Any],
    opening: str,
    keepalive: str,
) -> None:
    engine = _echo_engine(make_config, GatedEngine)
    session = quick_gateway.open_session(
        engine=engine, tokenizer=engine.tokenizer, params=SamplingParams(n=1, max_new_tokens=32)
    )
    url = f"{quick_gateway.session_url(session)}{path}"
    with (
        httpx.Client(trust_env=False, timeout=60) as http,
        http.stream("POST", url, json={**payload, "stream": True}) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = response.iter_lines()
        waiting: list[str] = []
        while waiting.count(keepalive) < 3:  # nothing was generated yet: the engine is held back
            waiting.append(next(lines))
        assert waiting[0].startswith(opening) and session.calls == []
        engine.release.set()
        rest = "\n".join(lines)
    (record,) = session.calls
    assert record.completion.text == "ABC" and '"ABC"' in rest


def test_sdk_streams_outlive_a_generation_longer_than_the_read_timeout(
    make_config: Callable[..., Any],
    quick_gateway: Gateway,
    openai_client: Callable[..., Any],
    anthropic_client: Callable[..., Any],
) -> None:
    """Clients give up on a connection that stays silent for their read timeout (0.3 s here); the
    generation takes twice that."""
    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")

    def session_with_slow_engine() -> Session:
        engine = _echo_engine(make_config, fake_latency=0.6)
        return quick_gateway.open_session(
            engine=engine, tokenizer=engine.tokenizer, params=SamplingParams(n=1, max_new_tokens=32)
        )

    session = session_with_slow_engine()
    client = openai_client(quick_gateway.base_url(session), timeout=openai.Timeout(30.0, read=0.3))
    messages = [{"role": "user", "content": "abc"}]
    with pytest.raises(openai.APITimeoutError):  # not streamed: nothing can keep it alive
        client.chat.completions.create(model="any", messages=messages)
    chunks = client.chat.completions.create(model="any", messages=messages, stream=True)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "ABC"
    last = list(client.responses.create(model="any", input="abc", stream=True))[-1]
    assert last.type == "response.completed" and last.response.output_text == "ABC"

    session = session_with_slow_engine()
    client = anthropic_client(
        quick_gateway.anthropic_base_url(session), timeout=anthropic.Timeout(30.0, read=0.3)
    )
    with client.messages.stream(model="any", max_tokens=16, messages=messages) as stream:
        assert stream.get_final_text() == "ABC"


def test_a_prompt_over_the_context_budget_is_refused_the_way_each_vendor_does(
    make_config: Callable[..., Any],
    gateway: Gateway,
    open_session: Callable[..., Session],
    openai_client: Callable[..., Any],
    anthropic_client: Callable[..., Any],
) -> None:
    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")
    engine = _echo_engine(make_config, ParamsEngine)
    session = open_session(engine, params=SamplingParams(n=1, max_new_tokens=64), max_context_tokens=40)
    long = "x" * 60  # 63 prompt tokens with the fake tokenizer's chat markers
    by_openai = openai_client(gateway.base_url(session))
    by_anthropic = anthropic_client(gateway.anthropic_base_url(session))

    with pytest.raises(openai.BadRequestError) as chat:
        by_openai.chat.completions.create(model="any", messages=[{"role": "user", "content": long}])
    assert chat.value.code == "context_length_exceeded" and chat.value.param == "messages"
    assert "maximum context length is 40 tokens" in chat.value.message
    assert "resulted in 63 tokens" in chat.value.message

    for stream in (False, True):  # a stream that cannot begin is a plain HTTP error as well
        with pytest.raises(openai.BadRequestError) as responses:
            by_openai.responses.create(model="any", input=long, stream=stream)
        assert responses.value.code == "context_length_exceeded" and responses.value.param == "input"
        assert "Your input exceeds the context window of this model" in responses.value.message

        with pytest.raises(anthropic.BadRequestError) as messages:
            by_anthropic.messages.create(
                model="any", max_tokens=16, messages=[{"role": "user", "content": long}], stream=stream
            )
        assert messages.value.status_code == 400
        assert messages.value.body == {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 63 tokens > 40 maximum",
            },
        }
    assert session.context_exceeded == 5 and session.calls == [] and engine.seen == []

    # A prompt that fits is answered, with the completion capped to the room the budget leaves.
    message = by_anthropic.messages.create(
        model="any", max_tokens=64, messages=[{"role": "user", "content": "abc"}]
    )
    assert message.content[0].text == "ABC"
    assert engine.seen[-1].max_new_tokens == 40 - len(session.calls[0].prompt_ids) == 34


def test_a_generation_that_fails_after_the_stream_began_is_reported_in_band(
    make_config: Callable[..., Any],
    gateway: Gateway,
    open_session: Callable[..., Session],
    openai_client: Callable[..., Any],
    anthropic_client: Callable[..., Any],
) -> None:
    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")
    session = open_session(FailingEngine(make_config()))
    by_openai = openai_client(gateway.base_url(session))
    by_anthropic = anthropic_client(gateway.anthropic_base_url(session))
    messages = [{"role": "user", "content": "abc"}]

    with pytest.raises(openai.APIError, match="generation failed: boom"):
        list(by_openai.chat.completions.create(model="any", messages=messages, stream=True))
    *_, last = by_openai.responses.create(model="any", input="abc", stream=True)
    assert last.type == "response.failed" and last.response.status == "failed"
    assert last.response.error.code == "server_error" and "boom" in last.response.error.message
    with pytest.raises(anthropic.APIStatusError, match="generation failed: boom"):
        list(by_anthropic.messages.create(model="any", max_tokens=16, messages=messages, stream=True))
    # not streamed, the same failure is a 500 in the API's error shape
    with pytest.raises(anthropic.InternalServerError) as failed:
        by_anthropic.messages.create(model="any", max_tokens=16, messages=messages)
    assert failed.value.body["error"] == {"type": "api_error", "message": "generation failed: boom"}
    assert isinstance(session.failure, RuntimeError) and session.calls == []


def test_unserved_requests_get_each_apis_errors(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    engine = _echo_engine(make_config)
    session = open_session(engine, max_calls=1)
    base = gateway.session_url(session)
    user = {"role": "user", "content": "abc"}

    def responses(payload: Any, **options: Any) -> tuple[int, Any]:
        status, body = post(options.pop("url", f"{base}/v1/responses"), payload, **options)
        assert set(body["error"]) == {"message", "type", "param", "code"}  # OpenAI's shape
        return status, body["error"]

    def messages(payload: Any, **options: Any) -> tuple[int, Any]:
        payload = {"max_tokens": 16, **payload}
        status, body = post(options.pop("url", f"{base}/v1/messages"), payload, **options)
        assert set(body) == {"type", "error"} and body["type"] == "error"  # Anthropic's shape
        assert set(body["error"]) == {"type", "message"}
        return status, body["error"]

    # every route needs the session
    for path in ("/v1/responses", "/v1/messages", "/v1/messages/count_tokens", "/v1/chat/completions"):
        assert request("POST", f"{gateway.url}{path}", {})[0] == 401
        assert request("POST", f"{gateway.url}/s/nope{path}", {})[0] == 404
    status, error = responses({"input": "abc"}, url=f"{gateway.url}/v1/responses", key="nope")
    assert status == 401 and error["code"] == "invalid_api_key"
    status, error = messages(
        {"messages": [user]}, url=f"{gateway.url}/v1/messages", headers={"x-api-key": "x"}
    )
    assert (status, error["type"]) == (401, "authentication_error")
    status, error = messages({"messages": [user]}, url=f"{gateway.url}/s/nope/v1/messages")
    assert (status, error["type"]) == (404, "not_found_error")

    # Responses: what the policy cannot do is refused, not ignored
    status, error = responses({"input": "abc", "tools": [{"type": "web_search"}]})
    assert status == 400 and error["param"] == "tools" and "web_search" in error["message"]
    status, error = responses({"instructions": "be brief"})
    assert status == 400 and error["param"] == "input" and error["type"] == "invalid_request_error"
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": []}
    status, error = responses({"input": [user, reasoning]})
    assert status == 400 and "input[1]" in error["message"] and "'reasoning'" in error["message"]
    status, error = responses(
        {"input": [{"type": "function_call", "name": "f", "arguments": "{", "call_id": "c"}]}
    )
    assert status == 400 and "not valid JSON" in error["message"]
    status, error = responses({"input": "abc", "conversation": "conv_1"})
    assert status == 400 and error["param"] == "conversation"
    status, error = responses({"input": "abc", "background": True})
    assert status == 400 and error["param"] == "background"

    # Messages
    status, error = messages(
        {"messages": [user], "tools": [{"type": "web_search_20250305", "name": "web_search"}]}
    )
    assert (status, error["type"]) == (400, "invalid_request_error") and "web_search_20250305" in error[
        "message"
    ]
    status, error = messages({"messages": [user, {"role": "assistant", "content": "The answer is ("}]})
    assert status == 400 and "prefill" in error["message"]
    thinking = {"type": "thinking", "thinking": "hm", "signature": "sig"}
    status, error = messages({"messages": [user, {"role": "assistant", "content": [thinking]}, user]})
    assert status == 400 and "messages.1.content.0" in error["message"] and "'thinking'" in error["message"]
    status, error = messages({"messages": []})
    assert (status, error["type"]) == (400, "invalid_request_error")
    document = {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "x"}}
    status, error = messages({"messages": [{"role": "user", "content": [document]}]})
    assert status == 400 and "'document'" in error["message"]
    assert request("POST", f"{base}/v1/messages", None)[0] == 400  # no JSON body
    assert session.calls == [] and engine.calls == []

    # max_calls: the first call is served, the next one refused, whichever API it uses
    assert post(f"{base}/v1/responses", {"input": "abc"})[0] == 200
    status, error = messages({"messages": [user]})
    assert status == 400 and "limited to 1 policy calls" in error["message"]
    status, error = responses({"input": "abc", "stream": True})
    assert status == 400 and error["code"] == "max_turns_exceeded"
    assert len(session.calls) == 1 and session.refused == 2
    # counting tokens is not a call
    assert post(f"{base}/v1/messages/count_tokens", {"messages": [user]}) == (
        200,
        {"input_tokens": len(session.calls[0].prompt_ids)},
    )

    gateway.close_session(session)
    status, error = messages({"messages": [user]})
    assert (status, error["type"]) == (503, "api_error")
    status, error = responses({"input": "abc"})
    assert status == 503 and error["code"] == "session_closed"


def test_the_three_apis_share_one_session(
    make_config: Callable[..., Any], gateway: Gateway, open_session: Callable[..., Session]
) -> None:
    """A conversation begun over one API is continued token for token over another: the calls of
    a session are compared as chats, not as wire formats."""
    engine = _scripted_engine(make_config, [f"Let me compute. {CALL_TEXT}", "The answer is 4", "Sure."])
    session = open_session(engine)
    base = gateway.session_url(session)

    status, body = post(f"{base}/v1/chat/completions", {"messages": [SYSTEM, QUESTION], "tools": [CHAT_TOOL]})
    (call,) = body["choices"][0]["message"]["tool_calls"]
    payload = {
        "max_tokens": 64,
        "system": [{"type": "text", "text": SYSTEM["content"], "cache_control": {"type": "ephemeral"}}],
        "tools": [MESSAGES_TOOL],
        "messages": [
            QUESTION,
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me compute."},
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": "4"}]},
        ],
    }
    status, body = post(f"{base}/v1/messages", payload)
    assert status == 200 and body["content"] == [{"type": "text", "text": "The answer is 4"}]
    _continues(session, "<|tool|>4<|end|><|assistant|>")

    items = [
        QUESTION,
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Let me compute."}],
        },
        {
            "type": "function_call",
            "call_id": call["id"],
            "name": "calculator",
            "arguments": '{"expression": "2+2"}',
        },
        {"type": "function_call_output", "call_id": call["id"], "output": "4"},
        {"role": "assistant", "content": "The answer is 4"},
        {"role": "user", "content": [{"type": "input_text", "text": "Thanks."}]},
    ]
    status, text = request(
        "POST",
        f"{base}/v1/responses",
        {"instructions": SYSTEM["content"], "tools": [RESPONSES_TOOL], "input": items, "stream": True},
    )
    events = sse_named(text)
    assert all(name == data["type"] for name, data in events)
    assert events[-1][1]["response"]["output"][0]["content"][0]["text"] == "Sure."
    third = session.calls[2]
    assert (third.parent, third.bridged) == (1, True)
    assert [turn.info["parent"] for turn in build_trajectory(session.calls, SAMPLE, "g").turns] == [
        None,
        0,
        1,
    ]
    # chat completions' own stream still ends with [DONE]
    text = request("POST", f"{base}/v1/chat/completions", {"messages": [QUESTION], "stream": True})[1]
    assert sse_events(text)[-1] == "[DONE]"
