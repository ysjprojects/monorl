"""Fixtures: a fake ``unsloth`` package in ``sys.modules`` and a fake CUDA flag.

The fake mirrors the upstream surface the adapter uses (unsloth 2026.10.11, tag v0.1.913-beta):

* ``unsloth.FastLanguageModel.from_pretrained`` — ``unsloth/models/loader.py`` (parameters
  ``model_name, max_seq_length, dtype, load_in_4bit, load_in_8bit, load_in_16bit, full_finetuning,
  token, device_map, ..., trust_remote_code, use_gradient_checkpointing, ..., revision, ...,
  fast_inference, ...``) returning ``(model, tokenizer)``;
* ``FastLanguageModel.get_peft_model`` — ``unsloth/models/llama.py`` (``r, target_modules,
  lora_alpha, lora_dropout, bias, ..., use_gradient_checkpointing, random_state, ...``) returning a
  PEFT model (here: real ``peft.get_peft_model``);
* ``FastLanguageModel.for_training(model, use_gradient_checkpointing=None)`` /
  ``for_inference(model)`` — ``unsloth/models/llama.py`` (sets ``training``/``gradient_checkpointing``
  flags and ``_flag_for_generation``);
* ``model.save_pretrained_merged(save_directory, tokenizer, save_method="merged_16bit", ...)`` —
  bound by ``unsloth.save.patch_saving_functions`` on loaded models;
* ``unsloth.kernels.utils.fast_dequantize(W, quant_state)`` — ``unsloth/kernels/utils.py``;
* ``unsloth.models.rl._wrap_grpo_generate_and_score`` marks its wrapper with
  ``_unsloth_restore_training_wrapped = True`` and rebinds ``trl.trainer.grpo_trainer.GRPOTrainer``
  to a regenerated ``UnslothGRPOTrainer`` — ``unsloth/models/rl.py``.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch

from monorl.config.schema import RunConfig

FAKE_VERSION = "2026.10.11"


@dataclass
class FakeQuantState:
    """Stands in for ``bitsandbytes.functional.QuantState``: remembers the fp tensor it 'quantised'."""

    dequantized: torch.Tensor
    dtype: torch.dtype = torch.float32


def fake_fast_dequantize(
    W: Any, quant_state: Any = None, out: Any = None, use_global_buffer: bool = False, dtype: Any = None
) -> torch.Tensor:
    assert quant_state is not None, "fast_dequantize needs the quant_state"
    return quant_state.dequantized.clone()


def quantize_fake_4bit(param: torch.nn.Parameter) -> torch.nn.Parameter:
    """Hang a ``quant_state`` off ``param`` the way bnb ``Params4bit`` do. The 'dequantised' value is
    deliberately *different* from ``.data`` (which stays usable by ``F.linear``) so a test can tell
    whether the export went through ``fast_dequantize``."""
    true = param.detach().clone()
    packed = torch.nn.Parameter(true.clone(), requires_grad=False)
    packed.quant_state = FakeQuantState(dequantized=true + 1.0, dtype=true.dtype)
    return packed


@dataclass
class FakeCalls:
    from_pretrained: list[dict[str, Any]] = field(default_factory=list)
    get_peft_model: list[dict[str, Any]] = field(default_factory=list)
    for_training: list[Any] = field(default_factory=list)
    for_inference: list[Any] = field(default_factory=list)
    save_pretrained_merged: list[dict[str, Any]] = field(default_factory=list)


def _bind_save_pretrained_merged(model: Any, calls: FakeCalls) -> None:
    def save_pretrained_merged(
        save_directory: str,
        tokenizer: Any = None,
        save_method: str = "merged_16bit",
        push_to_hub: bool = False,
        token: Any = None,
        **kwargs: Any,
    ) -> None:
        calls.save_pretrained_merged.append(
            {"save_directory": save_directory, "tokenizer": tokenizer, "save_method": save_method, **kwargs}
        )
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        base.config.save_pretrained(save_directory)
        (Path(save_directory) / f"{save_method}.marker").write_text(save_method)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)

    model.save_pretrained_merged = save_pretrained_merged


def _module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def make_fake_unsloth(calls: FakeCalls, *, load_dtype: torch.dtype = torch.float32) -> types.ModuleType:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(
            model_name: str = "unsloth/Llama-3.2-1B-Instruct",
            max_seq_length: int = 2048,
            dtype: Any = None,
            load_in_4bit: bool = True,
            load_in_8bit: bool = False,
            load_in_16bit: bool = False,
            full_finetuning: bool = False,
            token: Any = None,
            device_map: Any = "sequential",
            rope_scaling: Any = None,
            fix_tokenizer: bool = True,
            trust_remote_code: bool = False,
            use_gradient_checkpointing: Any = "unsloth",
            resize_model_vocab: Any = None,
            revision: Any = None,
            use_exact_model_name: bool = False,
            fast_inference: bool = False,
            gpu_memory_utilization: float = 0.5,
            random_state: int = 3407,
            max_lora_rank: int = 64,
            **kwargs: Any,
        ) -> tuple[Any, Any]:
            calls.from_pretrained.append(
                {
                    "model_name": model_name,
                    "max_seq_length": max_seq_length,
                    "dtype": dtype,
                    "token": token,
                    "load_in_4bit": load_in_4bit,
                    "load_in_8bit": load_in_8bit,
                    "full_finetuning": full_finetuning,
                    "device_map": device_map,
                    "trust_remote_code": trust_remote_code,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                    "revision": revision,
                    "fast_inference": fast_inference,
                    **kwargs,
                }
            )
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=load_dtype)
            model.config.use_cache = False
            if load_in_4bit:  # QLoRA: one LoRA-targeted and one untouched linear carry a quant_state
                layer0 = model.model.layers[0]
                layer0.self_attn.q_proj.weight = quantize_fake_4bit(layer0.self_attn.q_proj.weight)
                layer0.mlp.gate_proj.weight = quantize_fake_4bit(layer0.mlp.gate_proj.weight)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            _bind_save_pretrained_merged(model, calls)
            return model, tokenizer

        @staticmethod
        def get_peft_model(
            model: Any,
            r: int = 16,
            target_modules: Any = (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
            lora_alpha: int = 16,
            lora_dropout: float = 0.0,
            bias: str = "none",
            layers_to_transform: Any = None,
            layers_pattern: Any = None,
            use_gradient_checkpointing: Any = "unsloth",
            random_state: int = 3407,
            max_seq_length: int = 2048,
            use_rslora: bool = False,
            modules_to_save: Any = None,
            init_lora_weights: Any = True,
            **kwargs: Any,
        ) -> Any:
            calls.get_peft_model.append(
                {
                    "r": r,
                    "target_modules": target_modules,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "bias": bias,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                    "random_state": random_state,
                    "use_rslora": use_rslora,
                    **kwargs,
                }
            )
            torch.manual_seed(random_state)
            targets = list(target_modules) if not isinstance(target_modules, str) else target_modules
            peft_model = get_peft_model(
                model,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=targets,
                    bias=bias,
                ),
            )
            with torch.no_grad():  # non-zero B so merging is observable
                for name, p in peft_model.named_parameters():
                    if "lora_B" in name:
                        p.normal_(std=0.05)
            if use_gradient_checkpointing:
                peft_model.get_base_model().gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                peft_model.enable_input_require_grads()
            _bind_save_pretrained_merged(peft_model, calls)
            return peft_model

        @staticmethod
        def for_training(model: Any, use_gradient_checkpointing: Any = None) -> Any:
            calls.for_training.append(model)
            model.train()
            for m in model.modules():
                if "_flag_for_generation" in m.__dict__:
                    del m._flag_for_generation
            return model

        @staticmethod
        def for_inference(model: Any) -> Any:
            calls.for_inference.append(model)
            model.eval()
            model._flag_for_generation = True
            return model

    kernels_utils = _module("unsloth.kernels.utils", fast_dequantize=fake_fast_dequantize)
    kernels = _module("unsloth.kernels", utils=kernels_utils, __path__=[])

    def unsloth_generic_save_pretrained_merged(
        self: Any,
        save_directory: str,
        tokenizer: Any = None,
        save_method: str = "merged_16bit",
        **kwargs: Any,
    ) -> None:
        _bind_save_pretrained_merged(self, calls)
        self.save_pretrained_merged(save_directory, tokenizer, save_method=save_method, **kwargs)

    save = _module(
        "unsloth.save", unsloth_generic_save_pretrained_merged=unsloth_generic_save_pretrained_merged
    )
    rl = _module("unsloth.models.rl")
    models = _module("unsloth.models", rl=rl, __version__=FAKE_VERSION, __path__=[])
    root = _module(
        "unsloth",
        __version__=FAKE_VERSION,
        __path__=[],
        FastLanguageModel=FastLanguageModel,
        kernels=kernels,
        save=save,
        models=models,
        _fake_calls=calls,
    )
    root.__spec__.submodule_search_locations = []
    root._submodules = {m.__name__: m for m in (kernels_utils, kernels, save, rl, models)}
    return root


@pytest.fixture
def fake_calls() -> FakeCalls:
    return FakeCalls()


@pytest.fixture
def fake_unsloth(monkeypatch: pytest.MonkeyPatch, fake_calls: FakeCalls) -> types.ModuleType:
    """Install the fake ``unsloth`` package tree in ``sys.modules`` for the duration of a test."""
    from monorl.patching import revert

    for name in [m for m in sys.modules if m == "unsloth" or m.startswith("unsloth.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    root = make_fake_unsloth(fake_calls)
    monkeypatch.setitem(sys.modules, "unsloth", root)
    for name, mod in root._submodules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    yield root
    revert("unsloth")


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend a CUDA device exists (only for the adapter's availability gate)."""
    import monorl.adapters.unsloth.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "cuda_available", lambda: True)


def unsloth_config(
    tiny_model_dir: Path, *, lora: bool = True, extra: dict[str, Any] | None = None, **trainer: Any
) -> RunConfig:
    trainer_cfg: dict[str, Any] = {
        "backend": "unsloth",
        "device": "cpu",
        "precision": "fp32",
        "micro_batch_size": 2,
        "max_steps": 10,
        "learning_rate": 1e-2,
        "max_seq_len": 64,
        "extra": extra or {},
    }
    trainer_cfg.update(trainer)
    return RunConfig.from_dict(
        {
            "seed": 7,
            "model": {
                "name_or_path": str(tiny_model_dir),
                "dtype": "float32",
                "lora": {
                    "enabled": lora,
                    "r": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
            "data": {"uri": "unused.jsonl"},
            "rollout": {"backend": "hf", "group_size": 2, "max_new_tokens": 8, "device": "cpu"},
            "trainer": trainer_cfg,
            "rewards": [{"type": "exact_match"}],
        }
    )


@pytest.fixture(scope="session")
def tokenizer(tiny_model_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model_dir)
