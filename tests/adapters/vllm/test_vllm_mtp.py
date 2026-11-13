"""MTP drafting with vLLM: the engine arguments ``model.mtp`` compiles to, weight sync into the
drafter, and the acceptance statistics — against the fake engine and worker of ``vllm_fakes``."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from safetensors.torch import save_file
from vllm_fakes import FakeLLM, SamplingParams, TokensPrompt

from monorl.adapters.vllm.client import VllmServerClient
from monorl.adapters.vllm.engine import VllmEngine
from monorl.adapters.vllm.server import attach_router
from monorl.core.errors import ConfigError
from monorl.core.types import SamplingParams as MonorlSamplingParams
from monorl.core.types import SpeculationStats
from monorl.mtp import WIRE_PREFIX

ROLLOUT = {"enabled": True, "rollout": True}


LOOKALIKE = "mtp.gate.weight"  # a policy parameter that merely looks like an MTP checkpoint key


def _weights() -> list[tuple[str, torch.Tensor]]:
    """An update as the trainer exports it: the policy under its state_dict names, then the MTP
    layer's checkpoint keys behind ``WIRE_PREFIX``."""
    return [
        ("model.layers.0.self_attn.q_proj.weight", torch.full((2, 2), 1.0)),
        ("lm_head.weight", torch.full((2, 2), 2.0)),
        (LOOKALIKE, torch.full((2, 2), 5.0)),
        (WIRE_PREFIX + "mtp.fc.weight", torch.full((2, 2), 3.0)),
        (WIRE_PREFIX + "mtp.layers.0.mlp.experts.3.down_proj.weight", torch.full((2, 2), 4.0)),
    ]


POLICY_NAMES = [name for name, _ in _weights()[:3]]
MTP_KEYS = ["mtp.fc.weight", "mtp.layers.0.mlp.experts.3.down_proj.weight"]  # what the drafter loads


def _checkpoint(weights: list[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The same weights as a checkpoint file holds them: MTP layers under their bare keys."""
    return {name.removeprefix(WIRE_PREFIX): tensor for name, tensor in weights}


# -- engine arguments -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mtp", "user", "expected"),
    [
        ({}, None, None),
        ({"enabled": True, "train": True}, None, None),
        (ROLLOUT, None, {"method": "mtp", "num_speculative_tokens": 1}),
        (ROLLOUT | {"num_speculative_tokens": 3}, None, {"method": "mtp", "num_speculative_tokens": 3}),
        (
            ROLLOUT | {"num_speculative_tokens": 3},
            {"num-speculative-tokens": 2, "draft_lm_head_quantization": "fp8", "method": "deepseek_mtp"},
            {"method": "deepseek_mtp", "num_speculative_tokens": 2, "draft_lm_head_quantization": "fp8"},
        ),
    ],
)
def test_speculative_config_passed_to_the_engine(fake_vllm, mtp_config, mtp, user, expected) -> None:
    extra = {} if user is None else {"engine_kwargs": {"speculative_config": user}}
    engine = VllmEngine(mtp_config(mtp, extra))
    kwargs = FakeLLM.instances[-1].kwargs
    assert kwargs.get("speculative_config") == expected
    # Engine statistics are switched on for the acceptance counters only.
    assert kwargs.get("disable_log_stats", True) is (expected is None)
    assert (engine.speculation_stats() is None) is (expected is None)


@pytest.mark.parametrize(
    ("extra", "sleep", "message"),
    [
        (
            {"engine_kwargs": {"speculative_config": {"rejection_sample_method": "synthetic"}}},
            False,
            "logprobs",
        ),
        ({"engine_kwargs": {"speculative_config": {"rejection-sample-method": "fly"}}}, False, "logprobs"),
        ({"engine_kwargs": {"speculative_config": {"method": "ngram"}}}, False, "MTP layers"),
        ({"sleep_level": 2}, True, "sleep_level"),
    ],
)
@pytest.mark.parametrize("mode", ["colocate", "server"])
def test_settings_that_break_mtp_rollouts_are_rejected(
    fake_vllm, mtp_config, extra, sleep, message, mode
) -> None:
    extra = extra | {"mode": mode, "url": "http://vllm.test"}
    with pytest.raises(ConfigError, match=message):
        (VllmEngine if mode == "colocate" else VllmServerClient)(mtp_config(ROLLOUT, extra, sleep=sleep))
    assert FakeLLM.instances == []
    # The same settings are the user's own business without MTP rollouts.
    if mode == "colocate":
        VllmEngine(mtp_config({}, extra, sleep=sleep))


def test_level_one_sleep_is_allowed_with_mtp_rollouts(fake_vllm, mtp_config) -> None:
    engine = VllmEngine(mtp_config(ROLLOUT, {"sleep_level": 1}, sleep=True))
    engine.sleep()
    assert ("sleep", {"level": 1, "mode": "abort"}) in FakeLLM.instances[-1].calls


# -- weight sync ----------------------------------------------------------------------------------


@pytest.mark.parametrize("quantization", [None, "fp8"])
def test_update_sends_mtp_weights_to_the_drafter_only(fake_vllm, mtp_config, quantization) -> None:
    user = {} if quantization is None else {"draft_lm_head_quantization": quantization}
    # 16-byte tensors, 32-byte chunks: the first chunk holds no MTP weight.
    engine = VllmEngine(
        mtp_config(ROLLOUT, {"chunk_bytes": 32, "engine_kwargs": {"speculative_config": user}})
    )
    engine.update_weights(iter(_weights()))
    for worker in FakeLLM.instances[-1].workers:
        target, drafter = worker.model_runner.model, worker.get_draft_model()
        assert [name for name, _ in target.loaded] == POLICY_NAMES
        assert [name for name, _ in drafter.loaded] == MTP_KEYS
        assert torch.equal(drafter.loaded[0][1], torch.full((2, 2), 3.0))
        if quantization is None:
            assert drafter.quantized_head is None
        else:  # the drafter verifies nothing, but its copy of the head must be the head just pushed
            assert torch.equal(drafter.quantized_head, torch.full((2, 2), 2.0))
            assert not drafter.source_head_shared
    assert FakeLLM.instances[-1].names()[-1] == "reset_prefix_cache"


def test_update_without_mtp_weights_keeps_the_drafter_and_refreshes_its_head(fake_vllm, mtp_config) -> None:
    user = {"draft_lm_head_quantization": "nvfp4"}
    engine = VllmEngine(mtp_config(ROLLOUT, {"engine_kwargs": {"speculative_config": user}}))
    engine.update_weights(_weights())
    engine.update_weights([("lm_head.weight", torch.full((2, 2), 9.0))])
    for worker in FakeLLM.instances[-1].workers:
        target, drafter = worker.model_runner.model, worker.get_draft_model()
        assert [name for name, _ in target.loaded] == [*POLICY_NAMES, "lm_head.weight"]
        assert not set(drafter.offered) & {name for name, _ in _weights() if name.startswith(WIRE_PREFIX)}
        assert [name for name, _ in drafter.loaded] == MTP_KEYS
        assert torch.equal(drafter.quantized_head, torch.full((2, 2), 9.0))


def test_engine_that_does_not_draft_ignores_mtp_weights(fake_vllm, mtp_config) -> None:
    engine = VllmEngine(mtp_config({"enabled": True, "train": True}))
    engine.update_weights(_weights())
    for worker in FakeLLM.instances[-1].workers:
        assert worker.get_draft_model() is None
        assert [name for name, _ in worker.model_runner.model.loaded] == POLICY_NAMES


def test_checkpoint_reload_reaches_the_drafter(fake_vllm, mtp_config, tmp_path) -> None:
    user = {"draft_lm_head_quantization": "fp8"}
    engine = VllmEngine(mtp_config(ROLLOUT, {"engine_kwargs": {"speculative_config": user}}))
    worker = FakeLLM.instances[-1].workers[0]
    drafter = worker.get_draft_model()

    full = tmp_path / "step-1"
    full.mkdir()
    save_file(_checkpoint(_weights()), str(full / "model.safetensors"))
    engine.load_weights_from_path(full)
    assert worker.reload_calls == [{"args": (), "weights_path": str(full), "is_checkpoint_format": True}]
    assert sorted(name for name, _ in drafter.loaded) == sorted(MTP_KEYS)
    assert not drafter.source_head_shared

    # A checkpoint of the policy alone (a trainer that holds no MTP layers) reloads too.
    policy_only = tmp_path / "step-2"
    policy_only.mkdir()
    save_file(_checkpoint(_weights()[:3]), str(policy_only / "model.safetensors"))
    engine.load_weights_from_path(policy_only)
    assert len(worker.reload_calls) == 2 and len(drafter.loaded) == len(MTP_KEYS)


# -- statistics -----------------------------------------------------------------------------------


def test_speculation_stats_are_deltas(fake_vllm, mtp_config) -> None:
    engine = VllmEngine(mtp_config(ROLLOUT | {"num_speculative_tokens": 2}))
    assert engine.speculation_stats() == SpeculationStats()
    params = MonorlSamplingParams(n=2, max_new_tokens=4)
    engine.generate([[1, 2], [3]], params)  # 4 completions, 2 passes each: 2 drafted, 1 accepted
    assert engine.speculation_stats() == SpeculationStats(steps=8, drafted=16, accepted=8, emitted=16)
    assert engine.speculation_stats() == SpeculationStats()
    engine.generate([[1]], params)
    stats = engine.speculation_stats()
    assert stats == SpeculationStats(steps=4, drafted=8, accepted=4, emitted=8)
    assert stats.metrics() == {"mtp/accept_rate": 0.5, "mtp/accept_length": 2.0}


# -- server mode ----------------------------------------------------------------------------------


class _EngineClient:
    """``AsyncLLM.collective_rpc`` over the fake workers of ``llm``."""

    def __init__(self, llm: FakeLLM) -> None:
        self.llm = llm

    async def collective_rpc(
        self, method: str, timeout: float | None = None, args: tuple = (), kwargs: Any = None
    ) -> list[Any]:
        return self.llm.collective_rpc(method, args=args, kwargs=kwargs)


def _server(fake_vllm, **llm_kwargs: Any) -> tuple[FakeLLM, httpx.ASGITransport]:
    """The monorl routes on an app that also serves vLLM's ``/metrics`` and ``/reset_prefix_cache``."""
    llm = FakeLLM("policy", **llm_kwargs)
    app = FastAPI()
    attach_router(app)
    app.state.engine_client = _EngineClient(llm)

    def metrics() -> PlainTextResponse:  # without engine statistics only the process metrics remain
        process = 'python_gc_collections_total{generation="0"} 7.0\n'
        return PlainTextResponse(llm.metrics_text() if llm.log_stats else process)

    app.add_api_route("/metrics", metrics, methods=["GET"])
    app.add_api_route("/reset_prefix_cache", lambda: JSONResponse({}), methods=["POST"])
    return llm, httpx.ASGITransport(app=app)


SERVER = {"mode": "server", "url": "http://vllm.test"}


def test_server_mode_syncs_the_drafter_and_reports_stats(fake_vllm, mtp_config) -> None:
    speculative = {"method": "mtp", "num_speculative_tokens": 2, "draft_lm_head_quantization": "fp8"}
    llm, transport = _server(fake_vllm, speculative_config=speculative, disable_log_stats=False)
    llm.enqueue([TokensPrompt(prompt_token_ids=[1])], SamplingParams(n=1))  # traffic before the run
    client = VllmServerClient(
        mtp_config(ROLLOUT | {"num_speculative_tokens": 2}, SERVER), transport=transport
    )
    try:
        client.update_weights(_weights())
        target, drafter = llm.workers[0].model_runner.model, llm.workers[0].get_draft_model()
        assert [name for name, _ in target.loaded] == POLICY_NAMES
        assert [name for name, _ in drafter.loaded] == MTP_KEYS
        assert torch.equal(drafter.loaded[1][1], torch.full((2, 2), 4.0))
        assert torch.equal(drafter.quantized_head, torch.full((2, 2), 2.0))

        assert client.speculation_stats() == SpeculationStats()
        llm.enqueue([TokensPrompt(prompt_token_ids=[1])], SamplingParams(n=3))
        assert client.speculation_stats() == SpeculationStats(steps=6, drafted=12, accepted=6, emitted=12)
        assert client.speculation_stats() == SpeculationStats()
    finally:
        client.shutdown()


def test_server_mode_leaves_mtp_weights_out_when_not_drafting(fake_vllm, mtp_config) -> None:
    llm, transport = _server(fake_vllm)
    client = VllmServerClient(mtp_config({"enabled": True}, SERVER), transport=transport)
    try:
        client.update_weights(_weights())
        assert [name for name, _ in llm.workers[0].model_runner.model.loaded] == POLICY_NAMES
        assert client.speculation_stats() is None
    finally:
        client.shutdown()


@pytest.mark.parametrize(
    ("llm_kwargs", "mtp", "message"),
    [
        ({}, {}, "does not draft"),
        ({"speculative_config": {"method": "ngram"}}, {}, "not with the model's MTP layers"),
        (
            {"speculative_config": {"method": "mtp", "rejection_sample_method": "synthetic"}},
            {},
            "rejection_sample_method='synthetic'",
        ),
        (
            {"speculative_config": {"method": "mtp", "num_speculative_tokens": 1}},
            {"num_speculative_tokens": 2},
            "drafts 1 tokens",
        ),
        ({"speculative_config": {"method": "mtp"}}, {}, "disable-log-stats"),  # the server keeps no stats
    ],
)
def test_server_that_does_not_draft_as_configured_is_refused(
    fake_vllm, mtp_config, llm_kwargs, mtp, message
) -> None:
    if "disable-log-stats" not in message:
        llm_kwargs = llm_kwargs | {"disable_log_stats": False}
    _, transport = _server(fake_vllm, **llm_kwargs)
    with pytest.raises(ConfigError, match=message) as error:
        VllmServerClient(mtp_config(ROLLOUT | mtp, SERVER), transport=transport)
    if "disable-log-stats" not in message:  # the error says how to start the server
        assert '--speculative-config \'{"method": "mtp", "num_speculative_tokens": ' in str(error.value)
