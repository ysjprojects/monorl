from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from vllm_fakes import CompletionOutput, FakeLLM, Logprob

from monorl.adapters.vllm import PATCHES
from monorl.adapters.vllm.common import VLLM_VERSIONS
from monorl.adapters.vllm.patches import (
    build_monorl_routes,
    build_sampled_logprobs,
    build_sleep_guard,
    build_wake_up_guard,
)
from monorl.patching import apply, is_registered, revert, status

PATCH_IDS = {
    "vllm.llm.sleep_guard",
    "vllm.llm.wake_up_guard",
    "vllm.outputs.sampled_logprobs",
    "vllm.server.monorl_routes",
}


def test_patchset_declaration() -> None:
    assert PATCHES.framework == "vllm"
    assert {p.id for p in PATCHES.patches} == PATCH_IDS
    for p in PATCHES.patches:
        assert p.versions == VLLM_VERSIONS == ">=0.33,<0.34"
        assert p.description
        assert p.version_applicable("0.33.0")[0] and p.version_applicable("0.33.2")[0]
        assert not p.version_applicable("0.34.0")[0] and not p.version_applicable("0.32.0")[0]
    assert is_registered("vllm")


# -- builders against fakes -------------------------------------------------------------------


def _llm(enable_sleep_mode: bool) -> FakeLLM:
    return FakeLLM("m", enable_sleep_mode=enable_sleep_mode)


def test_sleep_guard_forwards_or_noops() -> None:
    sleep = build_sleep_guard(FakeLLM.sleep)
    on = _llm(True)
    sleep(on, level=2, mode="wait")
    assert on.calls == [("sleep", {"level": 2, "mode": "wait"})]
    off = _llm(False)
    assert sleep(off, level=1) is None and off.calls == []
    assert sleep.__wrapped__ is FakeLLM.sleep and "level" in inspect.signature(sleep).parameters


def test_wake_up_guard() -> None:
    wake = build_wake_up_guard(FakeLLM.wake_up)
    on = _llm(True)
    assert wake(on, ["weights"]) is True and on.calls == [("wake_up", {"tags": ["weights"]})]
    off = _llm(False)
    assert wake(off) is True and off.calls == []


def test_sampled_logprobs_property_on_dataclass() -> None:
    @dataclass
    class Out:  # minimal CompletionOutput look-alike
        token_ids: list[int]
        logprobs: list[dict[int, Logprob]] | None

    prop = build_sampled_logprobs(None)
    assert isinstance(prop, property)
    Out.sampled_logprobs = prop
    assert Out([3, 4], [{3: Logprob(-0.25), 9: Logprob(-3.0)}, {4: Logprob(-1.5)}]).sampled_logprobs == [
        -0.25,
        -1.5,
    ]
    assert Out([3], None).sampled_logprobs is None


def test_monorl_routes_wrapper_attaches_router() -> None:
    seen = {}

    def original(args, app, supported_tasks, model_config=None):
        seen["call"] = (args, supported_tasks, model_config)
        return "orig"

    wrapped = build_monorl_routes(original)
    app = FastAPI()
    assert wrapped("args", app, ("generate",), None) == "orig"
    assert seen["call"] == ("args", ("generate",), None)
    paths = set(app.openapi()["paths"])
    assert {"/update_named_param", "/load_weights_from_path", "/monorl/info"} <= paths


# -- apply through the patch engine against the fake module tree ---------------------------------


def test_apply_and_revert_against_fake_tree(fake_vllm) -> None:
    report = apply("vllm")
    assert report.installed_version == "0.33.0"
    assert report.failed == [] and set(report.applied) == PATCH_IDS, report
    llm_mod = fake_vllm["vllm.entrypoints.llm"]
    out_mod = fake_vllm["vllm.outputs"]
    app_mod = fake_vllm["vllm.entrypoints.launchers.app"]
    assert llm_mod.LLM.sleep.__monorl_patch__ == "vllm.llm.sleep_guard"
    assert isinstance(out_mod.CompletionOutput.__dict__["sampled_logprobs"], property)
    assert app_mod.register_api_routers.__monorl_patch__ == "vllm.server.monorl_routes"

    # patched behaviour end to end on the fakes
    llm = llm_mod.LLM("m", enable_sleep_mode=False)
    llm.sleep(level=1)
    assert llm.wake_up() is True and llm.calls == []
    out = CompletionOutput(0, "", [5, 6], None, [{5: Logprob(-0.5)}, {6: Logprob(-0.7)}], "stop")
    assert out.sampled_logprobs == [-0.5, -0.7]
    app = FastAPI()
    app_mod.register_api_routers("args", app, ("generate",))
    assert app.state.registered == ("args", ("generate",), None)
    assert "/update_named_param" in app.openapi()["paths"]

    assert {s.id: s.applied for s in status("vllm")} == dict.fromkeys(PATCH_IDS, True)
    second = apply("vllm")  # idempotent
    assert second.applied == [] and all(reason == "already applied" for _, reason in second.skipped)

    reverted = revert("vllm")
    assert set(reverted) == PATCH_IDS
    assert llm_mod.LLM.sleep is FakeLLM.sleep
    assert "sampled_logprobs" not in out_mod.CompletionOutput.__dict__


def test_patches_skip_on_unsupported_version(fake_vllm) -> None:
    fake_vllm["vllm"].__version__ = "0.34.0"
    report = apply("vllm")
    assert report.applied == [] and len(report.skipped) == 4
    assert all("0.34.0" in reason for _, reason in report.skipped)


def test_patches_disabled_by_env(fake_vllm, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONORL_DISABLE_PATCHES", "1")
    report = apply("vllm")
    assert report.applied == [] and all("disabled" in r for _, r in report.skipped)
