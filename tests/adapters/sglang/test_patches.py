"""Patch builders (direct) and the ``sglang`` PatchSet applied to the fake module tree."""

from __future__ import annotations

import types

import numpy as np
import torch
from fake_sglang import Engine as FakeEngine
from fake_sglang import TokenizerManager as FakeTokenizerManager

from monorl.adapters.sglang import PATCHES
from monorl.adapters.sglang.common import SGLANG_VERSIONS
from monorl.adapters.sglang.patches import (
    build_logprob_floats,
    build_memory_guard,
    build_seeded_parallel_sampling,
    build_update_weights_named_tensors,
    has_params,
    normalize_named_tensors,
)
from monorl.patching import apply, get_patchsets, is_applied, revert, status

EXPECTED_IDS = {
    "sglang.logprob_floats",
    "sglang.seeded_parallel_sampling",
    "sglang.release_memory_guard",
    "sglang.resume_memory_guard",
    "sglang.update_weights_named_tensors",
}


# -- patch set -----------------------------------------------------------------------------------
def test_patchset_declaration():
    assert PATCHES.framework == "sglang"
    assert {p.id for p in PATCHES.patches} == EXPECTED_IDS
    for p in PATCHES.patches:
        assert p.versions == SGLANG_VERSIONS == ">=0.5.23,<0.6"
        assert p.mode == "wrap"
        assert p.description and p.verify is not None
        assert p.target.startswith("sglang.srt.")
    assert PATCHES in get_patchsets("sglang")


def test_version_gate():
    p = PATCHES.patches[0]
    assert p.version_applicable("0.5.23") == (True, "")
    assert p.version_applicable("0.5.99")[0]
    assert not p.version_applicable("0.5.22")[0]
    assert not p.version_applicable("0.6.0")[0]
    assert not p.version_applicable(None)[0]


def test_verify_against_fake_originals():
    by_id = {p.id: p for p in PATCHES.patches}
    assert by_id["sglang.logprob_floats"].verify(FakeTokenizerManager.detokenize_logprob_tokens)
    assert by_id["sglang.seeded_parallel_sampling"].verify(FakeEngine.generate)
    assert by_id["sglang.release_memory_guard"].verify(FakeEngine.release_memory_occupation)
    assert by_id["sglang.resume_memory_guard"].verify(FakeEngine.resume_memory_occupation)
    assert by_id["sglang.update_weights_named_tensors"].verify(FakeEngine.update_weights_from_tensor)
    assert not by_id["sglang.seeded_parallel_sampling"].verify(lambda self, x: None)
    assert not has_params(object(), "x")


# -- builders ------------------------------------------------------------------------------------
def test_logprob_floats_coerces_scalars():
    seen = {}

    def original(self, vals, idxs, decode):
        seen["vals"], seen["idxs"], seen["decode"] = vals, idxs, decode
        return list(zip(vals, idxs, [None] * len(vals), strict=True))

    patched = build_logprob_floats(original)
    out = patched(
        None,
        [np.float32(-0.5), torch.tensor(-1.25), -2.0, None],
        [np.int64(3), torch.tensor(4), 5, None],
        False,
    )
    assert seen["vals"] == [-0.5, -1.25, -2.0, None] and all(type(v) is float for v in seen["vals"][:3])
    assert seen["idxs"] == [3, 4, 5, None] and all(type(i) is int for i in seen["idxs"][:3])
    assert out[0] == (-0.5, 3, None)
    assert patched.__wrapped__ is original


class _Gen:
    """Carrier for ``build_seeded_parallel_sampling``: records what reached the original."""

    def __init__(self):
        self.calls = []

    def generate(
        self,
        prompt=None,
        sampling_params=None,
        input_ids=None,
        return_logprob=False,
        logprob_start_len=None,
        stream=False,
        rid=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "input_ids": input_ids,
                "return_logprob": return_logprob,
                "logprob_start_len": logprob_start_len,
                "stream": stream,
                "rid": rid,
            }
        )
        return "ok"


def test_seeded_parallel_sampling_fans_out_input_ids():
    g = _Gen()
    patched = build_seeded_parallel_sampling(_Gen.generate)
    assert (
        patched(
            g,
            input_ids=[[1, 2], [3]],
            sampling_params={"n": 2, "sampling_seed": 10, "temperature": 0.9},
            return_logprob=True,
        )
        == "ok"
    )
    call = g.calls[-1]
    assert call["input_ids"] == [[1, 2], [1, 2], [3], [3]] and call["prompt"] is None
    assert call["sampling_params"] == [
        {"n": 1, "sampling_seed": 10, "temperature": 0.9},
        {"n": 1, "sampling_seed": 11, "temperature": 0.9},
        {"n": 1, "sampling_seed": 10, "temperature": 0.9},
        {"n": 1, "sampling_seed": 11, "temperature": 0.9},
    ]
    assert call["return_logprob"] is True


def test_seeded_parallel_sampling_positional_prompt_and_per_request_params():
    g = _Gen()
    patched = build_seeded_parallel_sampling(_Gen.generate)
    patched(g, "hello", [{"n": 3, "sampling_seed": 1}])
    call = g.calls[-1]
    assert call["prompt"] == ["hello"] * 3 and call["input_ids"] is None
    assert [sp["sampling_seed"] for sp in call["sampling_params"]] == [1, 2, 3]
    patched(g, ["a", "b"], [{"n": 2, "sampling_seed": 7}, {"n": 2}])
    call = g.calls[-1]
    assert call["prompt"] == ["a", "a", "b"]
    assert call["sampling_params"] == [{"n": 1, "sampling_seed": 7}, {"n": 1, "sampling_seed": 8}, {"n": 2}]


def test_seeded_parallel_sampling_leaves_other_calls_alone():
    g = _Gen()
    patched = build_seeded_parallel_sampling(_Gen.generate)
    patched(g, input_ids=[[1]], sampling_params={"n": 2})  # no seed
    patched(g, input_ids=[[1]], sampling_params={"n": 1, "sampling_seed": 3})  # n == 1
    patched(g, input_ids=[[1]], sampling_params={"n": 2, "sampling_seed": 3}, stream=True)  # streaming
    patched(
        g, input_ids=[[1]], sampling_params={"n": 2, "sampling_seed": 3}, rid=["r"]
    )  # per-request list arg
    patched(g, input_ids=[[1]], sampling_params=[{"n": 2, "sampling_seed": 3}, {"n": 1}])  # length mismatch
    patched(g, input_ids=[[1]], sampling_params=None)
    for call in g.calls:
        assert call["input_ids"] == [[1]]
    assert g.calls[0]["sampling_params"] == {"n": 2}
    assert g.calls[3]["rid"] == ["r"]


def test_seeded_parallel_sampling_end_to_end_on_fake_engine():
    patched = build_seeded_parallel_sampling(FakeEngine.generate)
    eng = FakeEngine(model_path="m")
    outs = patched(eng, input_ids=[[1, 2]], sampling_params={"n": 3, "sampling_seed": 5})
    assert len(outs) == 3
    assert len({tuple(o["output_ids"]) for o in outs}) == 3  # distinct samples
    unpatched = FakeEngine.generate(eng, input_ids=[[1, 2]], sampling_params={"n": 3, "sampling_seed": 5})
    assert len({tuple(o["output_ids"]) for o in unpatched}) == 1  # the upstream behaviour being fixed


def test_memory_guard_no_cuda_is_noop(monkeypatch, caplog):
    calls = []

    def original(self, tags=None):
        calls.append(tags)
        return "released"

    guarded = build_memory_guard(original, "release_memory_occupation")
    eng = types.SimpleNamespace(server_args=types.SimpleNamespace(enable_memory_saver=True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level("WARNING"):
        assert guarded(eng, tags=["weights"]) is None
    assert calls == [] and "CUDA is not available" in caplog.text
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert guarded(eng, tags=["weights"]) == "released" and calls == [["weights"]]
    eng.server_args.enable_memory_saver = False
    assert guarded(eng) is None and len(calls) == 1
    assert "enable_memory_saver" in caplog.text


def test_normalize_named_tensors_and_wrapper():
    view = torch.arange(16, dtype=torch.float32).reshape(4, 4)[:, 1]
    grad = torch.ones(2, requires_grad=True)
    out = normalize_named_tensors({"a": view, "b": grad})
    assert [n for n, _ in out] == ["a", "b"]
    assert out[0][1].is_contiguous() and torch.equal(out[0][1], view) and not out[1][1].requires_grad
    gen = normalize_named_tensors(("c", torch.zeros(1)) for _ in range(1))
    assert gen[0][0] == "c"
    assert normalize_named_tensors([b"payload", "b64"]) == [b"payload", "b64"]
    assert normalize_named_tensors(7) == 7

    seen = {}

    def original(self, named_tensors, load_format=None, flush_cache=True):
        seen["named"], seen["lf"], seen["fc"] = named_tensors, load_format, flush_cache
        return True, "ok"

    wrapped = build_update_weights_named_tensors(original)
    assert wrapped(None, {"a": view}, flush_cache=False) == (True, "ok")
    assert seen["named"][0][0] == "a" and seen["named"][0][1].is_contiguous() and seen["fc"] is False


# -- apply / revert on the fake tree ---------------------------------------------------------------
def test_apply_and_revert_on_fake_sglang(fake_sglang, monkeypatch):
    engine_mod = fake_sglang["sglang.srt.entrypoints.engine"]
    tm_mod = fake_sglang["sglang.srt.managers.tokenizer_manager"]
    originals = {
        "generate": engine_mod.Engine.generate,
        "release": engine_mod.Engine.release_memory_occupation,
        "update": engine_mod.Engine.update_weights_from_tensor,
        "detok": tm_mod.TokenizerManager.detokenize_logprob_tokens,
    }
    report = apply("sglang")
    assert set(report.applied) == EXPECTED_IDS, report
    assert all(is_applied(pid, "sglang") for pid in EXPECTED_IDS)
    for s in status("sglang"):
        assert s.installed_version == "0.5.23" and s.applicable and s.applied, s
    assert engine_mod.Engine.generate is not originals["generate"]
    assert engine_mod.Engine.generate.__wrapped__ is originals["generate"]

    # behaviour through the patched fake: seed fan-out, float coercion, memory guard, tensor normalisation
    eng = engine_mod.Engine(model_path="m", enable_memory_saver=True)
    outs = eng.generate(input_ids=[[1]], sampling_params={"n": 2, "sampling_seed": 9})
    assert len({tuple(o["output_ids"]) for o in outs}) == 2
    tm = tm_mod.TokenizerManager()
    assert tm.detokenize_logprob_tokens([np.float32(-1.0)], [np.int64(2)], False) == [(-1.0, 2, None)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert eng.release_memory_occupation() is None and not any(
        c.method == "release_memory_occupation" for c in eng.calls
    )
    eng.update_weights_from_tensor({"w": torch.zeros(2)}, flush_cache=False)
    assert eng.calls[-1].kwargs["named_tensors"][0][0] == "w"

    # idempotent re-apply, then revert restores pristine originals
    assert apply("sglang").applied == []
    reverted = revert("sglang")
    assert set(reverted) == EXPECTED_IDS
    assert engine_mod.Engine.generate is originals["generate"]
    assert engine_mod.Engine.release_memory_occupation is originals["release"]
    assert engine_mod.Engine.update_weights_from_tensor is originals["update"]
    assert tm_mod.TokenizerManager.detokenize_logprob_tokens is originals["detok"]


def test_apply_skips_out_of_range_version(fake_sglang):
    fake_sglang["sglang"].__version__ = "0.6.0"
    report = apply("sglang")
    assert report.applied == []
    assert len(report.skipped) == len(EXPECTED_IDS)
    assert all("does not satisfy" in s.reason for s in status("sglang"))


def test_status_without_sglang(no_sglang):
    for s in status("sglang"):
        assert s.installed_version is None and not s.applicable and not s.applied
