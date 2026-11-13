"""``SGLangServerClient`` request bodies against sglang's HTTP endpoints via ``httpx.MockTransport``."""

from __future__ import annotations

import base64
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
import torch
from fake_sglang import LoopbackServer, WeightUpdateSession, make_config, spec_meta

from monorl.adapters.sglang.client import SGLangServerClient
from monorl.core.errors import ConfigError, GenerationCancelled, WeightSyncError
from monorl.core.interfaces import RolloutEngine
from monorl.core.types import SamplingParams, SpeculationStats


@dataclass
class FakeServer:
    """Minimal stand-in for ``sglang/srt/entrypoints/http_server.py`` routes used by the client."""

    tp_size: int = 2
    requests: list[tuple[str, str, Any]] = field(default_factory=list)
    update_ok: bool = True
    disk_ok: bool = True
    flush_ok: bool = True
    generate_status: int = 200
    finish: dict[str, Any] = field(default_factory=lambda: {"type": "stop", "matched": 1})
    server_args: dict[str, Any] = field(default_factory=dict)  # launch arguments besides tp_size
    session: WeightUpdateSession = field(default_factory=WeightUpdateSession)
    update_fail_on: str | None = None  # a bucket carrying this tensor name fails to load
    spec_verify_ct: int = 0
    spec_correct_drafts: int = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        path = request.url.path
        self.requests.append((request.method, path, body))
        if path == "/get_server_info":
            info = {
                "tp_size": self.tp_size,
                "dp_size": 1,
                "model_path": "m",
                "speculative_algorithm": None,
                "speculative_accept_threshold_single": 1.0,
                "speculative_accept_threshold_acc": 1.0,
                **self.server_args,
            }
            return httpx.Response(200, json=info)
        if path == "/generate":
            if self.generate_status != 200:
                return httpx.Response(self.generate_status, json={"error": "bad request"})
            outs = []
            for i, (ids, sp) in enumerate(zip(body["input_ids"], body["sampling_params"], strict=True)):
                for k in range(sp.get("n", 1)):
                    toks = [500 + i * 10 + k, 501]
                    meta: dict[str, Any] = {
                        "id": f"r{i}{k}",
                        "finish_reason": self.finish,
                        "prompt_tokens": len(ids),
                        "completion_tokens": len(toks),
                        **spec_meta(
                            self.server_args,
                            verify_ct=self.spec_verify_ct,
                            correct_drafts=self.spec_correct_drafts,
                            completion_tokens=len(toks),
                        ),
                    }
                    if body.get("return_logprob"):
                        meta["output_token_logprobs"] = [[-0.1, toks[0], None], [-0.2, toks[1], None]]
                    outs.append({"text": f"t{i}{k}", "output_ids": toks, "meta_info": meta})
            return httpx.Response(200, json=outs)
        if path == "/begin_weight_update":
            return self.session_response(self.session.begin(body["selector"]))
        if path == "/end_weight_update":
            return self.session_response(self.session.end())
        if path == "/update_weights_from_tensor":
            refused = self.session.refuse_update()
            if refused is not None:
                return self.session_response(refused)
            names = [n for n, _ in pickle.loads(base64.b64decode(body["serialized_named_tensors"][0]))]
            if self.update_fail_on in names:
                return self.session_response((False, f"failed to load {self.update_fail_on}"))
            ok = self.update_ok
            return httpx.Response(
                200 if ok else 400, json={"success": ok, "message": "Success." if ok else "bad tensor"}
            )
        if path == "/update_weights_from_disk":
            ok = self.disk_ok
            return httpx.Response(
                200 if ok else 400,
                json={"success": ok, "message": "ok" if ok else "missing", "num_paused_requests": 0},
            )
        if path == "/flush_cache":
            return httpx.Response(
                200 if self.flush_ok else 400, text="Cache flushed." if self.flush_ok else "busy"
            )
        if path in ("/release_memory_occupation", "/resume_memory_occupation"):
            return httpx.Response(200, json=None)
        return httpx.Response(404, text=f"no route {path}")

    @staticmethod
    def session_response(result: tuple[bool, str]) -> httpx.Response:
        success, message = result
        return httpx.Response(200 if success else 400, json={"success": success, "message": message})

    def paths(self) -> list[str]:
        return [p for _, p, _ in self.requests]


@pytest.fixture
def server():
    return FakeServer()


@pytest.fixture
def client(server, run_config_dict):
    cfg = make_config(run_config_dict, extra={"mode": "server", "url": "http://sgl.test:30000/"})
    c = SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))
    yield c
    c.shutdown()


@pytest.fixture
def loopback(server, run_config_dict):
    """A client talking to ``server`` over a real loopback connection."""
    web = LoopbackServer(server.handler)
    c = SGLangServerClient(make_config(run_config_dict, extra={"mode": "server", "url": web.url}))
    yield web, c
    c.shutdown()
    web.close()


def test_client_is_rollout_engine_and_needs_url(client, run_config_dict):
    assert isinstance(client, RolloutEngine)
    assert client.url == "http://sgl.test:30000"
    with pytest.raises(ConfigError, match="rollout.extra.url"):
        SGLangServerClient(make_config(run_config_dict, extra={"mode": "server"}))


def test_generate_request_body_chat_and_tokens(client, server):
    messages = [{"role": "user", "content": "hi"}]
    params = SamplingParams(n=2, max_new_tokens=5, temperature=0.5, top_k=10, stop=["x"])
    groups = client.generate([messages, [1, 2, 3]], params)
    method, path, body = server.requests[-1]
    assert (method, path) == ("POST", "/generate")
    tok = client.tokenizer
    chat_ids = tok(
        tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
        add_special_tokens=False,
    )["input_ids"]
    assert body["input_ids"] == [chat_ids, [1, 2, 3]]
    assert body["return_logprob"] is True and body["logprob_start_len"] == -1
    assert (
        body["sampling_params"]
        == [
            {
                "n": 2,
                "max_new_tokens": 5,
                "temperature": 0.5,
                "top_p": 1.0,
                "top_k": 10,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "skip_special_tokens": True,
                "stop": ["x"],
                "no_stop_trim": True,
            }
        ]
        * 2
    )
    assert [len(g) for g in groups] == [2, 2]
    assert groups[0][0].prompt_token_ids == chat_ids and groups[1][1].prompt_token_ids == [1, 2, 3]
    assert groups[1][1].text == "t11" and groups[1][1].token_ids == [511, 501]
    assert groups[0][0].logprobs == [-0.1, -0.2] and groups[0][0].finish_reason == "stop"


def test_generate_seed_fanout_and_no_logprobs(client, server):
    groups = client.generate([[1]], SamplingParams(n=2, seed=5, logprobs=False))
    body = server.requests[-1][2]
    assert body["input_ids"] == [[1], [1]]
    assert body["return_logprob"] is False
    assert [sp["sampling_seed"] for sp in body["sampling_params"]] == [5, 6]
    assert len(groups[0]) == 2 and groups[0][0].logprobs is None


def test_generate_length_and_error_status(client, server):
    server.finish = {"type": "length", "length": 2}
    assert client.generate([[1]], SamplingParams())[0][0].finish_reason == "length"
    server.generate_status = 400
    with pytest.raises(RuntimeError, match="HTTP 400"):
        client.generate([[1]], SamplingParams())
    assert client.generate([], SamplingParams()) == []


def test_cancel_disconnects_every_running_generate_request(loopback):
    web, client = loopback
    web.stall = True  # the server holds the requests: long generations
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = [pool.submit(client.generate, [[1, 2]], SamplingParams(n=1)) for _ in range(2)]
        assert [web.stalled.get(timeout=5) for _ in running] == ["/generate"] * 2
        client.cancel()
        client.cancel()  # repeated calls are harmless
        for call in running:
            with pytest.raises(GenerationCancelled):
                call.result(timeout=5)
    # the server saw both clients leave
    assert [web.disconnects.get(timeout=5) for _ in running] == ["/generate"] * 2
    web.stall = False
    groups = client.generate([[1, 2, 3]], SamplingParams(n=2))
    assert [c.text for c in groups[0]] == ["t00", "t01"]


def test_cancel_while_idle_does_not_affect_the_next_generate(loopback):
    _, client = loopback
    client.cancel()
    groups = client.generate([[1, 2, 3]], SamplingParams(n=2))
    assert [c.prompt_token_ids for c in groups[0]] == [[1, 2, 3], [1, 2, 3]]


def test_update_weights_posts_base64_payload_per_rank_then_flushes(client, server):
    w = torch.arange(4, dtype=torch.float32)
    client.update_weights([("layer.w", w)])
    assert server.paths() == ["/get_server_info"] + [
        "/begin_weight_update",
        "/update_weights_from_tensor",
        "/end_weight_update",
        "/flush_cache",
    ]
    assert server.requests[1][2] == {"selector": "all"}  # target and draft runner
    _, _, body = server.requests[2]
    assert body["load_format"] is None and body["flush_cache"] is False
    payloads = body["serialized_named_tensors"]
    assert len(payloads) == server.tp_size
    for p in payloads:
        raw = base64.b64decode(p, validate=True)
        back = pickle.loads(raw)  # what MultiprocessingSerializer.deserialize does after base64
        assert back[0][0] == "layer.w" and torch.equal(back[0][1], w)
    # tp_size is cached: a second update does not re-query server info
    client.update_weights([("layer.b", torch.zeros(1))])
    assert server.paths()[5:] == [
        "/begin_weight_update",
        "/update_weights_from_tensor",
        "/end_weight_update",
        "/flush_cache",
    ]


def test_update_weights_buckets(server, run_config_dict):
    cfg = make_config(run_config_dict, extra={"mode": "server", "url": "http://s", "bucket_bytes": 8})
    c = SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))
    c.update_weights([(f"w{i}", torch.zeros(2, dtype=torch.float32)) for i in range(3)])
    c.shutdown()
    assert server.paths() == (
        ["/get_server_info", "/begin_weight_update"]
        + ["/update_weights_from_tensor"] * 3
        + ["/end_weight_update", "/flush_cache"]
    )
    assert server.session.opened == ["all"] and server.session.closed == 1


def test_failed_bucket_closes_the_session_so_the_next_push_works(server, run_config_dict):
    cfg = make_config(run_config_dict, extra={"mode": "server", "url": "http://s", "bucket_bytes": 8})
    c = SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))
    weights = [(f"w{i}", torch.zeros(2, dtype=torch.float32)) for i in range(3)]
    server.update_fail_on = "w1"
    with pytest.raises(WeightSyncError, match="failed to load w1"):
        c.update_weights(weights)
    # w2 was not sent, the session was closed and the cache was not flushed
    assert server.paths()[1:] == (
        ["/begin_weight_update"] + ["/update_weights_from_tensor"] * 2 + ["/end_weight_update"]
    )
    assert server.session.selector is None
    server.update_fail_on = None
    c.update_weights(weights)
    c.shutdown()
    assert server.session.opened == ["all", "all"] and server.session.closed == 2
    assert server.paths()[-2:] == ["/end_weight_update", "/flush_cache"]


def test_update_weights_empty_sends_nothing(client, server):
    client.update_weights([])
    assert server.requests == []


def test_update_weights_failure_and_flush_warning(client, server, caplog):
    server.update_ok = False
    with pytest.raises(WeightSyncError, match="bad tensor"):
        client.update_weights([("w", torch.zeros(1))])
    server.update_ok = True
    server.flush_ok = False
    with caplog.at_level("WARNING"):
        client.update_weights([("w", torch.zeros(1))])
    assert "/flush_cache did not run" in caplog.text


def test_load_weights_from_path(client, server, tmp_path):
    client.load_weights_from_path(tmp_path)
    assert server.paths() == ["/update_weights_from_disk", "/flush_cache"]
    assert server.requests[0][2] == {"model_path": str(tmp_path), "load_format": None}
    server.disk_ok = False
    with pytest.raises(WeightSyncError, match="missing"):
        client.load_weights_from_path(tmp_path)


def test_sleep_wake_endpoints(client, server):
    client.sleep()
    client.sleep()
    client.wake_up()
    client.wake_up()
    assert server.requests == [
        ("POST", "/release_memory_occupation", {}),
        ("POST", "/resume_memory_occupation", {}),
    ]


def test_sleep_tags_and_disabled(server, run_config_dict):
    cfg = make_config(run_config_dict, extra={"mode": "server", "url": "http://s", "sleep_tags": ["weights"]})
    c = SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))
    c.sleep()
    assert server.requests[-1] == ("POST", "/release_memory_occupation", {"tags": ["weights"]})
    off = SGLangServerClient(
        make_config(run_config_dict, enable_sleep_mode=False, extra={"mode": "server", "url": "http://s"}),
        transport=httpx.MockTransport(server.handler),
    )
    n = len(server.requests)
    off.sleep()
    off.wake_up()
    assert len(server.requests) == n
    c.shutdown()
    off.shutdown()


def test_server_info_and_tp_fallback(server, run_config_dict):
    cfg = make_config(run_config_dict, tensor_parallel_size=3, extra={"mode": "server", "url": "http://s"})
    c = SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))
    assert c.server_info()["tp_size"] == 2

    def no_tp(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/get_server_info":
            return httpx.Response(200, json={"model_path": "m"})
        return server.handler(request)

    c2 = SGLangServerClient(cfg, transport=httpx.MockTransport(no_tp))
    c2.update_weights([("w", torch.zeros(1))])
    assert len(server.requests[-3][2]["serialized_named_tensors"]) == 3
    c.shutdown()
    c2.shutdown()


# -- MTP drafting --------------------------------------------------------------------------------
MTP_ROLLOUT = {"enabled": True, "rollout": True}


def mtp_client(server: FakeServer, run_config_dict: dict[str, Any], mtp: dict[str, Any] | None):
    cfg = make_config(run_config_dict, mtp=mtp, extra={"mode": "server", "url": "http://s"})
    return SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))


def test_speculation_stats_sum_meta_info_and_reset(server, run_config_dict):
    server.server_args = {"speculative_algorithm": "EAGLE", "speculative_num_draft_tokens": 3}
    server.spec_verify_ct, server.spec_correct_drafts = 2, 1
    c = mtp_client(server, run_config_dict, MTP_ROLLOUT)
    assert c.speculation_stats() == SpeculationStats()
    c.generate([[1, 2], [3]], SamplingParams(n=2))
    c.generate([[4]], SamplingParams(n=1))
    # 5 requests of 2 tokens, 2 verify passes each proposing num_draft_tokens - 1 = 2 drafts
    assert c.speculation_stats() == SpeculationStats(steps=10, drafted=20, accepted=5, emitted=10)
    assert c.speculation_stats() == SpeculationStats()
    assert server.paths().count("/get_server_info") == 1  # the server is checked once
    c.shutdown()


def test_speculation_stats_is_none_without_mtp_rollout(server, run_config_dict):
    server.server_args = {"speculative_algorithm": "EAGLE"}  # the server drafts, the run did not ask
    server.spec_verify_ct = 2
    c = mtp_client(server, run_config_dict, {"enabled": True})
    c.generate([[1]], SamplingParams(n=1))
    assert c.speculation_stats() is None
    assert "/get_server_info" not in server.paths()
    c.shutdown()


@pytest.mark.parametrize(
    ("server_args", "message"),
    [
        ({}, "speculative_algorithm=None"),
        ({"speculative_algorithm": "NGRAM"}, "speculative_algorithm='NGRAM'"),
        (
            {"speculative_algorithm": "EAGLE", "speculative_accept_threshold_single": 0.6},
            "speculative_accept_threshold_single=0.6",
        ),
        (
            {"speculative_algorithm": "EAGLE", "speculative_accept_threshold_acc": 0.9},
            "speculative_accept_threshold_acc=0.9",
        ),
    ],
)
def test_mtp_rollout_refuses_a_server_that_does_not_draft_losslessly(
    server, run_config_dict, server_args, message
):
    server.server_args = server_args
    c = mtp_client(server, run_config_dict, MTP_ROLLOUT)
    with pytest.raises(ConfigError, match=message):
        c.generate([[1]], SamplingParams(n=1))
    assert "/generate" not in server.paths()
    c.shutdown()


MTP_STREAM = [
    ("model.layers.0.w", torch.ones(2)),
    ("@mtp/mtp.fc.weight", torch.full((2,), 2.0)),
    ("lm_head.weight", torch.full((2,), 3.0)),
    ("@mtp/mtp.layers.0.mlp.experts.3.down_proj.weight", torch.full((2,), 4.0)),
]


def pushed(server: FakeServer) -> dict[str, float]:
    """Name → value of every tensor the server received, which must all lie inside one session."""
    paths = server.paths()
    updates = [i for i, p in enumerate(paths) if p == "/update_weights_from_tensor"]
    assert paths.count("/begin_weight_update") == 1
    assert paths.index("/begin_weight_update") < updates[0] and updates[-1] < paths.index(
        "/end_weight_update"
    )
    return {
        name: float(tensor[0])
        for i in updates
        for name, tensor in pickle.loads(
            base64.b64decode(server.requests[i][2]["serialized_named_tensors"][0])
        )
    }


def test_mtp_weights_reach_a_drafting_server_under_their_checkpoint_keys(server, run_config_dict):
    cfg = make_config(
        run_config_dict, mtp=MTP_ROLLOUT, extra={"mode": "server", "url": "http://s", "bucket_bytes": 8}
    )
    c = SGLangServerClient(cfg, transport=httpx.MockTransport(server.handler))
    c.update_weights(iter(MTP_STREAM))
    c.shutdown()
    assert pushed(server) == {
        "model.layers.0.w": 1.0,
        "mtp.fc.weight": 2.0,
        "lm_head.weight": 3.0,
        "mtp.layers.0.mlp.experts.3.down_proj.weight": 4.0,
    }


def test_mtp_weights_are_not_sent_to_a_server_that_does_not_draft(server, run_config_dict):
    c = mtp_client(server, run_config_dict, {"enabled": True, "train": True})
    c.update_weights(iter(MTP_STREAM))
    assert pushed(server) == {"model.layers.0.w": 1.0, "lm_head.weight": 3.0}
    n = len(server.requests)
    c.update_weights([MTP_STREAM[1]])  # nothing for this server: no session, no cache flush
    c.shutdown()
    assert len(server.requests) == n
