"""Image inputs of the gateway against a tiny random Qwen2-VL on CPU: the three APIs' image
shapes reach the policy as the same chat, the recorded calls become a batch the trainer can
reproduce the sampling logprobs from, and what cannot work is refused."""

from __future__ import annotations

import base64
import socket
import threading
from collections.abc import Callable, Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import fakes
import httpx
import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.types import Sample, SamplingParams
from monorl.data.media import media_cache_dir
from monorl.gateway import Gateway, Session, build_trajectory
from monorl.orchestrator.batching import build_train_batch
from monorl.rollout.media import collate_media, vlm_prompt_ids
from tests.gateway.client import post
from tests.rollout.tiny_vlm import build_tiny_vlm, write_test_images

QUESTION = "What colour is this?"
SAMPLE = Sample(id="s", messages=[{"role": "user", "content": QUESTION}], reference="red")
TOOL = {"type": "function", "function": {"name": "zoom", "parameters": {"type": "object", "properties": {}}}}


@pytest.fixture(scope="module")
def vlm_engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    from monorl.rollout.hf_engine import HFRolloutEngine

    model_dir = build_tiny_vlm(tmp_path_factory.mktemp("tiny-qwen2vl"))
    config = RunConfig.from_dict(
        {
            "model": {"name_or_path": str(model_dir), "dtype": "float32"},
            "data": {"uri": "unused.jsonl"},
            "rollout": {"backend": "hf", "group_size": 2, "max_new_tokens": 8, "device": "cpu"},
            "trainer": {"backend": "reference", "device": "cpu", "precision": "fp32"},
            "rewards": [{"type": "exact_match"}],
        }
    )
    engine = HFRolloutEngine(config)
    yield engine
    engine.shutdown()


@pytest.fixture(scope="module")
def images(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """``red``: 56x56 (4 image tokens), ``blue``: 112x56 (8)."""
    return write_test_images(tmp_path_factory.mktemp("images"))


@pytest.fixture(autouse=True)
def media_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The media cache in a directory of the test; localhost reached without a proxy."""
    monkeypatch.setenv("MONORL_HOME", str(tmp_path / "home"))
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    return media_cache_dir()


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def image_server(images: dict[str, Path]) -> Iterator[str]:
    """Base URL of an HTTP server on localhost that serves the test images."""
    handler = partial(_QuietHandler, directory=str(images["red"].parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join()


def _base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def test_image_calls_over_each_api_end_in_a_trainable_batch(
    vlm_engine: Any,
    images: dict[str, Path],
    image_server: str,
    media_cache: Path,
    gateway: Gateway,
    open_session: Callable[..., Session],
) -> None:
    openai = pytest.importorskip("openai")
    anthropic = pytest.importorskip("anthropic")
    processor = vlm_engine.processor
    image_id = int(vlm_engine.model.config.image_token_id)
    params = SamplingParams(n=1, max_new_tokens=4, seed=3)
    red = _base64(images["red"])
    again = {"role": "user", "content": "Are you sure?"}

    # Chat completions: a base64 data URL; the SDK's message object goes back into the history.
    chat = open_session(vlm_engine, params=params)
    http = httpx.Client(trust_env=False)
    client = openai.OpenAI(base_url=gateway.base_url(chat), api_key="unused", max_retries=0, http_client=http)
    picture = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{red}", "detail": "low"}}
    messages: list[Any] = [{"role": "user", "content": [picture, {"type": "text", "text": QUESTION}]}]
    reply = client.chat.completions.create(model="any", messages=messages)
    messages += [reply.choices[0].message, again]
    client.chat.completions.create(model="any", messages=messages)

    # Responses: an http URL, which the gateway fetches once; the follow-up names the response.
    responses = open_session(vlm_engine, params=params)
    http = httpx.Client(trust_env=False)
    client = openai.OpenAI(
        base_url=gateway.base_url(responses), api_key="unused", max_retries=0, http_client=http
    )
    content = [
        {"type": "input_image", "image_url": f"{image_server}/blue.png", "detail": "auto"},
        {"type": "input_text", "text": QUESTION},
    ]
    first = client.responses.create(model="any", input=[{"role": "user", "content": content}])
    client.responses.create(model="any", previous_response_id=first.id, input=again["content"])

    # Messages: bare base64; the follow-up is streamed.
    anthropic_session = open_session(vlm_engine, params=params)
    client = anthropic.Anthropic(
        base_url=gateway.anthropic_base_url(anthropic_session),
        api_key="unused",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(trust_env=False),
    )
    source = {"type": "base64", "media_type": "image/png", "data": red}
    user = {
        "role": "user",
        "content": [{"type": "image", "source": source}, {"type": "text", "text": QUESTION}],
    }
    message = client.messages.create(model="any", max_tokens=4, messages=[user])
    history = [user, {"role": "assistant", "content": message.content}, again]
    with client.messages.stream(model="any", max_tokens=4, messages=history) as stream:
        streamed = stream.get_final_message()

    sessions = (chat, responses, anthropic_session)
    for session, image_tokens in zip(sessions, (4, 8, 4), strict=True):
        first_call, second_call = session.calls
        # the image is a file of the media cache; the chat is monorl's, whatever the API
        path = Path(first_call.messages[0]["content"][0]["image"])
        assert path.parent == media_cache and path.is_file()
        assert first_call.messages == [
            {
                "role": "user",
                "content": [{"type": "image", "image": str(path)}, {"type": "text", "text": QUESTION}],
            }
        ]
        # the policy conditioned on the processor's encoding, image placeholders included
        assert first_call.prompt_ids == vlm_prompt_ids(processor, first_call.messages)
        assert first_call.prompt_ids.count(image_id) == image_tokens
        # the follow-up continues the call and is encoded again from its chat
        assert second_call.messages == [*first_call.messages, first_call.response, again]
        assert (second_call.parent, second_call.bridged) == (0, False)
        assert second_call.prompt_ids == vlm_prompt_ids(processor, second_call.messages)
    assert chat.calls[0].messages == anthropic_session.calls[0].messages  # same bytes, same file
    assert (
        Path(responses.calls[0].messages[0]["content"][0]["image"]).read_bytes()
        == images["blue"].read_bytes()
    )
    record = anthropic_session.calls[1]
    assert streamed.usage.input_tokens == len(record.prompt_ids)
    assert streamed.usage.output_tokens == len(record.completion.token_ids)

    trajectories = [build_trajectory(session.calls, SAMPLE, f"g{i}") for i, session in enumerate(sessions)]
    assert all("tools" not in trajectory.info for trajectory in trajectories)
    batch, _ = build_train_batch(
        trajectories,
        [1.0, -1.0, 0.5],
        pad_token_id=processor.tokenizer.pad_token_id,
        max_seq_len=256,
        processor=processor,
    )
    assert batch.input_ids.shape[0] == 3 and batch.old_logprobs is not None
    for row, trajectory in enumerate(trajectories):
        last = trajectory.turns[-1]
        sequence = [*last.prompt_token_ids, *last.completion.token_ids]
        assert batch.input_ids[row, : len(sequence)].tolist() == sequence
        assert batch.response_mask[row, len(last.prompt_token_ids) : len(sequence)].all()
    per_sequence = batch.extra["per_sequence"]
    assert [media["pixel_values"].shape[0] for media in per_sequence] == [16, 32, 16]
    # Trainable: a forward pass over the batch gives the logprobs the completions were sampled with.
    with torch.no_grad():
        logits = vlm_engine.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            mm_token_type_ids=batch.extra["mm_token_type_ids"],
            **collate_media(per_sequence),
        ).logits
    logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    logprobs = logprobs.gather(2, batch.input_ids[:, 1:, None]).squeeze(2)
    trained = batch.response_mask[:, 1:] == 1
    torch.testing.assert_close(logprobs[trained], batch.old_logprobs[:, 1:][trained], atol=1e-3, rtol=0.0)


def test_what_cannot_work_with_images_is_refused(
    make_config: Callable[..., Any],
    vlm_engine: Any,
    images: dict[str, Path],
    image_server: str,
    gateway: Gateway,
    open_session: Callable[..., Session],
) -> None:
    red = _base64(images["red"])
    data_url = f"data:image/png;base64,{red}"
    picture = {"type": "image_url", "image_url": {"url": data_url}}
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": red}}

    # A text-only policy cannot take images, whichever API they come in on.
    engine = fakes.FakeEngine(make_config())
    text_only = open_session(engine)
    base = gateway.session_url(text_only)
    for path, payload in (
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": [picture]}]}),
        (
            "/v1/responses",
            {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": data_url}]}]},
        ),
        ("/v1/messages", {"max_tokens": 8, "messages": [{"role": "user", "content": [block]}]}),
    ):
        status, body = post(f"{base}{path}", payload)
        assert status == 400 and "text-only model" in body["error"]["message"]
    assert text_only.calls == [] and engine.calls == []

    session = open_session(vlm_engine, params=SamplingParams(n=1, max_new_tokens=2, seed=0))
    base = gateway.session_url(session)
    chat_url = f"{base}/v1/chat/completions"
    user = {"role": "user", "content": [picture, {"type": "text", "text": QUESTION}]}

    def refused(payload: Any, path: str = "/v1/chat/completions") -> dict[str, Any]:
        status, body = post(f"{base}{path}", payload)
        assert status == 400
        return body["error"]

    # Tool schemas cannot travel with a chat that has images.
    error = refused({"messages": [user], "tools": [TOOL]})
    assert error["param"] == "tools" and "cannot be combined with tools" in error["message"]
    error = refused(
        {
            "max_tokens": 8,
            "messages": [{"role": "user", "content": [block]}],
            "tools": [{"name": "zoom", "input_schema": {"type": "object"}}],
        },
        "/v1/messages",
    )
    assert error["type"] == "invalid_request_error" and "cannot be combined with tools" in error["message"]
    # Images belong to user messages.
    error = refused({"messages": [{"role": "system", "content": [picture]}, user]})
    assert "user messages only" in error["message"]
    result = {"type": "tool_result", "tool_use_id": "t1", "content": [block]}
    error = refused({"max_tokens": 8, "messages": [{"role": "user", "content": [result]}]}, "/v1/messages")
    assert "a tool result must be a string or a list of text blocks" in error["message"]
    # What is not an image the gateway can get at: broken data, files of the trainer's host,
    # uploaded files, a URL nobody serves.
    for url, code in (
        ("data:image/png;base64,AAAA", "invalid_base64_image"),
        (f"file://{images['red']}", "invalid_image_url"),
        (str(images["red"]), "invalid_image_url"),
        (f"{image_server}/missing.png", "failed_to_download_image"),
        (_closed_port_url(), "failed_to_download_image"),
    ):
        error = refused(
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]}
        )
        assert error["code"] == code, url
    error = refused(
        {"input": [{"role": "user", "content": [{"type": "input_image", "file_id": "file-1"}]}]},
        "/v1/responses",
    )
    assert error["param"] == "input" and "file_id" in error["message"]
    source = {"type": "file", "file_id": "file-1"}
    error = refused(
        {"max_tokens": 8, "messages": [{"role": "user", "content": [{"type": "image", "source": source}]}]},
        "/v1/messages",
    )
    assert "unsupported image source of type 'file'" in error["message"]
    assert session.calls == []

    # A session can be told not to fetch remote images.
    offline = open_session(vlm_engine, fetch_images=False)
    remote = {"type": "image_url", "image_url": {"url": f"{image_server}/red.png"}}
    status, body = post(
        f"{gateway.base_url(offline)}/chat/completions", {"messages": [{"role": "user", "content": [remote]}]}
    )
    assert status == 400 and "does not fetch remote images" in body["error"]["message"]
    # The context budget counts the image's tokens, before anything is generated.
    tight = open_session(vlm_engine, max_context_tokens=10)
    status, body = post(f"{gateway.base_url(tight)}/chat/completions", {"messages": [user]})
    assert status == 400 and body["error"]["code"] == "context_length_exceeded"
    assert tight.context_exceeded == 1 and tight.calls == []

    # Once a call had images, a call without them would end the episode in a conversation that
    # leaves the image turns untrainable: it is refused, and the episode stays trainable.
    assert post(chat_url, {"messages": [user]})[0] == 200
    error = refused({"messages": [{"role": "user", "content": "And without the picture?"}]})
    assert "calls with images" in error["message"]
    assert len(session.calls) == 1
    # The other order is fine: the image conversation is the one the episode ends in.
    mixed = open_session(vlm_engine, params=SamplingParams(n=1, max_new_tokens=2, seed=0))
    mixed_url = f"{gateway.base_url(mixed)}/chat/completions"
    assert (
        post(mixed_url, {"messages": [{"role": "user", "content": "A question without a picture."}]})[0]
        == 200
    )
    assert post(mixed_url, {"messages": [user]})[0] == 200
    processor = vlm_engine.processor
    for recorded in (session, mixed):
        trajectory = build_trajectory(recorded.calls, SAMPLE, "g")
        batch, _ = build_train_batch(
            [trajectory],
            [1.0],
            pad_token_id=processor.tokenizer.pad_token_id,
            max_seq_len=256,
            processor=processor,
        )
        last = trajectory.turns[-1]
        assert batch.input_ids.shape[0] == 1 and batch.extra["per_sequence"][0]["pixel_values"].shape[0] == 16
        assert batch.input_ids[0, : len(last.prompt_token_ids)].tolist() == last.prompt_token_ids


def _closed_port_url() -> str:
    """An http URL on localhost that nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}/red.png"
