from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from monorl.core.errors import BackendUnavailable, ConfigError
from monorl.core.interfaces import Trainer
from monorl.core.registry import Registry
from tests.adapters.unsloth.conftest import FakeCalls, unsloth_config

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _make_batch(tokenizer: Any, *, batch_size: int = 4, response_len: int = 5) -> Any:
    from monorl.core.batch import TrainBatch
    from monorl.rollout.hf_engine import chat_prompt_ids

    prompts = [chat_prompt_ids(tokenizer, [{"role": "user", "content": f"Q{i}: " + "pad " * (i % 3)}]) for i in range(batch_size)]
    responses = [[1234] * (response_len - (i % 2)) + [tokenizer.eos_token_id] for i in range(batch_size)]
    width = max(len(p) + len(r) for p, r in zip(prompts, responses, strict=True))
    input_ids = torch.full((batch_size, width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((batch_size, width), dtype=torch.long)
    response_mask = torch.zeros((batch_size, width), dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompts, responses, strict=True)):
        seq = p + r
        input_ids[i, : len(seq)] = torch.tensor(seq)
        attention[i, : len(seq)] = 1
        response_mask[i, len(p) : len(seq)] = 1
    return TrainBatch(
        input_ids=input_ids,
        attention_mask=attention,
        response_mask=response_mask,
        advantages=response_mask.float(),
        rewards=torch.ones(batch_size),
        group_ids=[f"g{i // 2}" for i in range(batch_size)],
    )


def _build(config: Any) -> Any:
    from monorl.adapters.unsloth import register

    registry = Registry()
    register(registry)
    return registry.get("trainer", "unsloth")(config)


# -- availability gates --------------------------------------------------------------------------


def test_backend_unavailable_without_cuda(tiny_model_dir: Path, fake_unsloth: types.ModuleType) -> None:
    """Real environment: no CUDA → actionable error before any model loads, even with unsloth importable."""
    from monorl.adapters.unsloth import is_available
    from monorl.adapters.unsloth.trainer import make_unsloth_trainer

    assert not torch.cuda.is_available()
    assert is_available() is False
    with pytest.raises(BackendUnavailable, match="CUDA"):
        make_unsloth_trainer(unsloth_config(tiny_model_dir))
    assert fake_unsloth._fake_calls.from_pretrained == []


def test_backend_unavailable_without_unsloth(tiny_model_dir: Path, fake_cuda: None, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [m for m in sys.modules if m == "unsloth" or m.startswith("unsloth.")]:
        monkeypatch.delitem(sys.modules, name)
    from monorl.adapters.unsloth.trainer import make_unsloth_trainer

    with pytest.raises(BackendUnavailable, match="monorl\\[unsloth\\]"):
        make_unsloth_trainer(unsloth_config(tiny_model_dir))


class _LazyFakeFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves the fake ``unsloth`` through a real ``import`` so pre-import detection runs."""

    def __init__(self, module: types.ModuleType) -> None:
        self.module = module

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if name == "unsloth":
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec: Any) -> Any:
        return self.module

    def exec_module(self, module: Any) -> None:
        pass


def test_import_order_warning_and_strict_mode(
    tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from monorl.adapters.unsloth.trainer import import_unsloth

    monkeypatch.delitem(sys.modules, "unsloth")
    monkeypatch.setitem(sys.modules, "trl", types.ModuleType("trl"))
    finder = _LazyFakeFinder(fake_unsloth)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    with pytest.raises(BackendUnavailable, match="imported after"):
        import_unsloth(strict_import_order=True)
    assert "unsloth" not in sys.modules or sys.modules["unsloth"] is fake_unsloth
    monkeypatch.delitem(sys.modules, "unsloth", raising=False)

    with caplog.at_level(logging.WARNING, logger="monorl.adapters.unsloth.trainer"):
        mod = import_unsloth()
    assert mod is fake_unsloth
    assert any("trl" in rec.getMessage() and "imported after" in rec.getMessage() for rec in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="monorl.adapters.unsloth.trainer"):
        import_unsloth(strict_import_order=True)  # already imported: nothing to enforce
    assert not caplog.records


# -- factory / argument forwarding ---------------------------------------------------------------


def test_factory_builds_trainer_and_forwards_lora_args(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType, fake_calls: FakeCalls) -> None:
    trainer = _build(unsloth_config(tiny_model_dir))

    assert isinstance(trainer, Trainer)
    assert trainer.step == 0
    assert os.environ["UNSLOTH_RETURN_LOGITS"] == "1"
    assert os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") != "1"

    (load,) = fake_calls.from_pretrained
    assert load["model_name"] == str(tiny_model_dir)
    assert load["max_seq_length"] == 64
    assert load["dtype"] is torch.float32
    assert load["load_in_4bit"] is False and load["load_in_8bit"] is False
    assert load["fast_inference"] is False
    assert load["full_finetuning"] is False
    assert load["use_gradient_checkpointing"] is False
    assert load["device_map"] == {"": "cpu"}
    assert load["trust_remote_code"] is False and load["revision"] is None

    (peft,) = fake_calls.get_peft_model
    assert peft["r"] == 4 and peft["lora_alpha"] == 8 and peft["lora_dropout"] == 0.0
    assert peft["target_modules"] == ["q_proj", "v_proj"]
    assert peft["bias"] == "none"
    assert peft["use_gradient_checkpointing"] is False
    assert peft["random_state"] == 7

    assert fake_calls.for_training == [trainer.model]
    assert trainer.lora_enabled and hasattr(trainer.model, "peft_config")
    assert trainer.hf_model is trainer.model.get_base_model()
    assert trainer.inner.model is trainer.model


def test_gradient_checkpointing_4bit_and_passthrough_kwargs(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType, fake_calls: FakeCalls) -> None:
    config = unsloth_config(
        tiny_model_dir,
        gradient_checkpointing=True,
        extra={"load_in_4bit": True, "unsloth_kwargs": {"token": "hf_x"}, "unsloth_lora_kwargs": {"use_rslora": True}},
    )
    config.model.dtype = "auto"
    trainer = _build(config)

    (load,) = fake_calls.from_pretrained
    assert load["load_in_4bit"] is True
    assert load["dtype"] is None
    assert load["use_gradient_checkpointing"] == "unsloth"
    assert load["token"] == "hf_x"
    (peft,) = fake_calls.get_peft_model
    assert peft["use_gradient_checkpointing"] == "unsloth"
    assert peft["use_rslora"] is True
    assert trainer.hf_model.is_gradient_checkpointing  # enabled by Unsloth, not re-enabled by the reference loop


def test_full_finetuning_without_lora(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType, fake_calls: FakeCalls) -> None:
    trainer = _build(unsloth_config(tiny_model_dir, lora=False))
    (load,) = fake_calls.from_pretrained
    assert load["full_finetuning"] is True
    assert fake_calls.get_peft_model == []
    assert not trainer.lora_enabled and trainer.model is trainer.hf_model
    names = dict(trainer.export_weights())
    assert set(names) == set(trainer.hf_model.state_dict())


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"load_in_4bit": True, "load_in_8bit": True}, "mutually exclusive"),
        ({"unsloth_kwargs": [1]}, "must be a mapping"),
    ],
)
def test_config_errors(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType, extra: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        _build(unsloth_config(tiny_model_dir, extra=extra))


def test_quantised_load_requires_lora(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType) -> None:
    with pytest.raises(ConfigError, match="lora.enabled"):
        _build(unsloth_config(tiny_model_dir, lora=False, extra={"load_in_4bit": True}))


# -- training loop through the reference trainer ----------------------------------------------


def test_train_step_and_logprobs(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType, fake_calls: FakeCalls, tokenizer: Any) -> None:
    trainer = _build(unsloth_config(tiny_model_dir))
    batch = _make_batch(tokenizer)
    base_before = {n: p.detach().clone() for n, p in trainer.hf_model.named_parameters() if "lora_" not in n}
    lora_before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if "lora_" in n}

    before = trainer.compute_logprobs(batch)
    ref_before = trainer.compute_logprobs(batch, reference=True)
    metrics = trainer.train_step(batch)
    after = trainer.compute_logprobs(batch)
    ref_after = trainer.compute_logprobs(batch, reference=True)

    assert trainer.step == 1
    assert {"loss/policy", "grad_norm", "lr", "tokens/response_mean"} <= set(metrics)
    assert before.shape == tuple(batch.input_ids.shape)
    mask = batch.response_mask.bool()
    assert (after[mask] - before[mask]).abs().sum() > 0  # the policy moved
    assert torch.allclose(ref_before, ref_after)  # the reference (adapters disabled) did not
    assert all(torch.equal(base_before[n], p.detach()) for n, p in trainer.hf_model.named_parameters() if "lora_" not in n)
    assert any(not torch.equal(lora_before[n], p.detach()) for n, p in trainer.model.named_parameters() if "lora_" in n)
    assert len(fake_calls.for_training) == 1  # training mode set once, not per step


def test_for_training_reapplied_after_inference_mode(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType, fake_calls: FakeCalls, tokenizer: Any) -> None:
    trainer = _build(unsloth_config(tiny_model_dir))
    fake_unsloth.FastLanguageModel.for_inference(trainer.model)  # e.g. a rollout engine sharing the model
    assert trainer.model._flag_for_generation
    trainer.train_step(_make_batch(tokenizer))
    assert len(fake_calls.for_training) == 2
    assert not getattr(trainer.model, "_flag_for_generation", False)


# -- weight export ----------------------------------------------------------------------------


def _lora_delta(layer: Any) -> torch.Tensor:
    adapter = layer.active_adapters[0]
    return (layer.lora_B[adapter].weight @ layer.lora_A[adapter].weight) * layer.scaling[adapter]


def test_export_weights_merges_lora_into_hf_names(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType) -> None:
    from transformers import AutoModelForCausalLM

    trainer = _build(unsloth_config(tiny_model_dir))
    exported = dict(trainer.export_weights())
    plain_keys = set(AutoModelForCausalLM.from_pretrained(tiny_model_dir).state_dict())
    assert set(exported) == plain_keys
    assert not any(".lora_" in k or ".base_layer" in k for k in exported)

    q_proj = trainer.hf_model.model.layers[0].self_attn.q_proj
    expected = q_proj.base_layer.weight.detach() + _lora_delta(q_proj)
    assert torch.allclose(exported["model.layers.0.self_attn.q_proj.weight"], expected, atol=1e-6)
    assert not torch.allclose(exported["model.layers.0.self_attn.q_proj.weight"], q_proj.base_layer.weight.detach())
    # untouched layers are exported as-is
    k_proj = trainer.hf_model.model.layers[0].self_attn.k_proj
    assert torch.equal(exported["model.layers.0.self_attn.k_proj.weight"], k_proj.weight.detach())
    # identical to the reference trainer's own merge
    reference = dict(trainer.inner.export_weights())
    assert set(reference) == set(exported)
    assert all(torch.allclose(reference[k], exported[k], atol=1e-6) for k in exported)


def test_export_weights_dequantises_4bit(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType) -> None:
    from transformers import AutoModelForCausalLM

    trainer = _build(unsloth_config(tiny_model_dir, extra={"load_in_4bit": True}))
    exported = dict(trainer.export_weights())
    plain_keys = set(AutoModelForCausalLM.from_pretrained(tiny_model_dir).state_dict())
    assert set(exported) == plain_keys

    layer0 = trainer.hf_model.model.layers[0]
    gate = layer0.mlp.gate_proj  # 4-bit, not LoRA-targeted → dequantised verbatim
    assert torch.allclose(exported["model.layers.0.mlp.gate_proj.weight"], gate.weight.quant_state.dequantized)
    assert not torch.allclose(exported["model.layers.0.mlp.gate_proj.weight"], gate.weight.detach())

    q_proj = layer0.self_attn.q_proj  # 4-bit *and* LoRA-targeted → dequantise, then merge
    expected = q_proj.base_layer.weight.quant_state.dequantized + _lora_delta(q_proj)
    assert torch.allclose(exported["model.layers.0.self_attn.q_proj.weight"], expected, atol=1e-6)
    assert exported["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float32


def test_merge_helpers_on_tiny_tensors() -> None:
    """Pure merge arithmetic on a hand-built LoRA layer: W + scaling * B @ A."""
    from peft import LoraConfig, get_peft_model

    from monorl.adapters.unsloth.trainer import iter_merged_weights, merge_lora_layer

    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=True))
    net[0].weight.data.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    peft_model = get_peft_model(net, LoraConfig(r=1, lora_alpha=2, target_modules=["0"]))
    layer = peft_model.base_model.model[0]
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(torch.tensor([[1.0, 0.0, -1.0]]))
        layer.lora_B["default"].weight.copy_(torch.tensor([[1.0], [2.0]]))
    # scaling = alpha / r = 2 → delta = 2 * [[1,0,-1],[2,0,-2]]
    expected = torch.tensor([[3.0, 2.0, 1.0], [8.0, 5.0, 2.0]])
    assert torch.equal(merge_lora_layer(layer), expected)
    exported = dict(iter_merged_weights(peft_model.get_base_model()))
    assert set(exported) == {"0.weight", "0.bias"}
    assert torch.equal(exported["0.weight"], expected)
    assert torch.equal(exported["0.bias"], layer.base_layer.bias.detach())


def test_merge_rejects_non_finite() -> None:
    from peft import LoraConfig, get_peft_model

    from monorl.adapters.unsloth.trainer import merge_lora_layer

    peft_model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(2, 2)), LoraConfig(r=1, lora_alpha=1, target_modules=["0"]))
    layer = peft_model.base_model.model[0]
    with torch.no_grad():
        layer.lora_B["default"].weight.fill_(float("inf"))
        layer.lora_A["default"].weight.fill_(1.0)
    with pytest.raises(RuntimeError, match="non-finite"):
        merge_lora_layer(layer)


# -- persistence -------------------------------------------------------------------------------


def test_save_pretrained_uses_merged_16bit(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType, fake_calls: FakeCalls, tmp_path: Path, tokenizer: Any) -> None:
    trainer = _build(unsloth_config(tiny_model_dir))
    out = tmp_path / "export"
    trainer.save_pretrained(out)

    (call,) = fake_calls.save_pretrained_merged
    assert call["save_directory"] == str(out)
    assert call["save_method"] == "merged_16bit"
    assert call["tokenizer"] is trainer.tokenizer
    assert (out / "merged_16bit.marker").exists()
    assert (out / "config.json").exists()
    assert (out / "tokenizer_config.json").exists()
    # the save path may have flipped the model to inference mode: next step restores training mode
    trainer.train_step(_make_batch(tokenizer))
    assert len(fake_calls.for_training) == 2


def test_save_pretrained_without_bound_method_uses_generic_saver(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType, fake_calls: FakeCalls, tmp_path: Path) -> None:
    trainer = _build(unsloth_config(tiny_model_dir))
    del trainer.model.save_pretrained_merged
    trainer.save_pretrained(tmp_path / "generic")
    (call,) = fake_calls.save_pretrained_merged
    assert call["save_method"] == "merged_16bit" and (tmp_path / "generic" / "merged_16bit.marker").exists()


def test_checkpoint_roundtrip(tiny_model_dir: Path, fake_cuda: None, fake_unsloth: types.ModuleType, fake_calls: FakeCalls, tmp_path: Path, tokenizer: Any) -> None:
    trainer = _build(unsloth_config(tiny_model_dir))
    batch = _make_batch(tokenizer)
    trainer.train_step(batch)
    snapshot = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if "lora_" in n}
    ckpt = tmp_path / "ckpt"
    trainer.save_checkpoint(ckpt)
    assert (ckpt / "adapter").is_dir() and (ckpt / "trainer_state.pt").exists()

    trainer.train_step(batch)
    assert trainer.step == 2
    trainer.load_checkpoint(ckpt)
    assert trainer.step == 1
    assert all(torch.equal(snapshot[n], p.detach()) for n, p in trainer.model.named_parameters() if "lora_" in n)
    trainer.train_step(batch)  # resumes training; for_training re-armed after the load
    assert trainer.step == 2 and len(fake_calls.for_training) == 2
