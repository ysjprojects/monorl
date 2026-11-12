"""``trainer: unsloth`` — Unsloth ``FastLanguageModel`` inside the reference trainer loop.

Upstream API used (unsloth 2026.10.11, ``unsloth/models/loader.py``, ``unsloth/models/llama.py``,
``unsloth/save.py``; git tag ``v0.1.913-beta``, see ``third_party/pins.toml``):

* ``FastLanguageModel.from_pretrained(model_name, max_seq_length, dtype, load_in_4bit, load_in_8bit,
  full_finetuning, device_map, revision, trust_remote_code, use_gradient_checkpointing,
  fast_inference=False, **kwargs) -> (model, tokenizer)``. ``dtype=None`` lets Unsloth pick
  bf16/fp16 for the GPU; ``fast_inference`` stays ``False`` because sampling is the job of the
  monorl rollout engine, not an Unsloth-owned vLLM.
* ``FastLanguageModel.get_peft_model(model, r, target_modules, lora_alpha, lora_dropout, bias,
  use_gradient_checkpointing, random_state, ...) -> PeftModel``. ``use_gradient_checkpointing=
  "unsloth"`` selects Unsloth's offloaded checkpointing; PEFT's ``lora_A``/``lora_B``/``scaling``
  layout is unchanged, which is what :func:`merge_lora_layer` relies on.
* ``FastLanguageModel.for_training(model)`` / ``for_inference(model)`` flip training mode,
  gradient checkpointing and ``use_cache`` on every nested module; the trainer calls
  ``for_training`` before the first optimizer step and again after anything that may have
  switched the model to inference mode.
* ``model.save_pretrained_merged(save_directory, tokenizer, save_method="merged_16bit")`` —
  bound by ``unsloth.save.patch_saving_functions`` on every model Unsloth loads; it dequantises
  4-bit weights and merges the LoRA into a plain HF checkpoint.
* ``unsloth.kernels.utils.fast_dequantize(W, quant_state)`` (fallback:
  ``bitsandbytes.functional.dequantize_4bit``) to turn bnb ``Params4bit`` back into 16-bit weights
  for :meth:`UnslothTrainer.export_weights`.
* Environment contract of the patched forward (``unsloth/models/llama.py``): ``UNSLOTH_RETURN_LOGITS=1``
  keeps raw logits (otherwise the fused cross-entropy path hands back an ``EmptyLogits`` sentinel
  when labels are given) and ``UNSLOTH_RETURN_HIDDEN_STATES`` must not be ``"1"`` (that returns the
  last hidden state under ``.logits``). The reference loop computes logprobs from ``.logits``, so
  both are pinned before every forward.

The composition: :class:`UnslothTrainer` builds the model/tokenizer with Unsloth and hands them to
:class:`monorl.trainer.ReferenceTrainer` (``model=``/``tokenizer=`` injection), which owns the
optimizer, micro-batching, loss, checkpoints and logprob computation. Only ``export_weights``
(dequantise + merge) and ``save_pretrained`` (Unsloth's merged export) are Unsloth-specific.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch
from monorl.core.errors import BackendUnavailable, ConfigError
from monorl.rollout.hf_engine import resolve_device

log = logging.getLogger(__name__)

#: Modules Unsloth patches at import; importing them first loses optimisations (``unsloth/_gpu_init.py``).
CRITICAL_MODULES: tuple[str, ...] = ("trl", "transformers", "peft")

INSTALL_HINT = "pip install 'monorl[unsloth]' (CUDA build of torch + unsloth 2026.10.x + unsloth_zoo)"


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def ensure_cuda() -> None:
    """Unsloth's kernels (Triton, bitsandbytes) are CUDA/ROCm-only; fail before touching the model."""
    if not cuda_available():
        raise BackendUnavailable(
            "trainer.backend='unsloth' needs a CUDA device but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}). Use trainer.backend='reference' on CPU/MPS, or run on a GPU host "
            f"with {INSTALL_HINT}."
        )


def import_unsloth(*, strict_import_order: bool = False) -> Any:
    """``import unsloth`` and report which of :data:`CRITICAL_MODULES` were imported before it.

    Unsloth patches ``trl``/``transformers``/``peft`` at import time and warns when they were imported
    first. monorl cannot avoid ``transformers`` (the reference loop needs it), but ``trl`` must not be
    imported before Unsloth when the TRL backend is combined with it; ``strict_import_order=True``
    turns that case into :class:`BackendUnavailable`.
    """
    pre_imported = [m for m in CRITICAL_MODULES if m in sys.modules] if "unsloth" not in sys.modules else []
    try:
        import unsloth
    except ImportError as e:
        raise BackendUnavailable(
            f"trainer.backend='unsloth' needs the unsloth package: {e}. Install with {INSTALL_HINT}."
        ) from e
    if pre_imported:
        msg = (
            f"unsloth was imported after {pre_imported}; its import-time patches for those modules are "
            "partially skipped. Import unsloth first (`import monorl.adapters.unsloth.preload`, or "
            "`python -c 'import unsloth'` style launchers) for the full set of optimisations."
        )
        if strict_import_order and "trl" in pre_imported:
            raise BackendUnavailable(msg + " (trainer.extra.strict_import_order=true)")
        log.warning(msg)
    return unsloth


def pin_logits_env() -> None:
    """Make Unsloth's patched forward return real logits (see module docstring)."""
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES") == "1":
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"


def unsloth_dtype(dtype: str) -> torch.dtype | None:
    """``model.dtype`` → ``FastLanguageModel.from_pretrained(dtype=...)``; ``auto`` lets Unsloth choose."""
    return None if dtype == "auto" else getattr(torch, dtype)


def from_pretrained_kwargs(config: RunConfig, device: torch.device) -> dict[str, Any]:
    """Keyword arguments for ``FastLanguageModel.from_pretrained`` derived from the run config.

    ``trainer.extra``: ``load_in_4bit``/``load_in_8bit`` (default False), ``unsloth_kwargs`` (dict
    passed through verbatim, last).
    """
    tc, mc = config.trainer, config.model
    load_in_4bit = bool(tc.extra.get("load_in_4bit", False))
    load_in_8bit = bool(tc.extra.get("load_in_8bit", False))
    if load_in_4bit and load_in_8bit:
        raise ConfigError("trainer.extra: load_in_4bit and load_in_8bit are mutually exclusive")
    if (load_in_4bit or load_in_8bit) and not mc.lora.enabled:
        raise ConfigError(
            "trainer.extra.load_in_4bit/load_in_8bit need model.lora.enabled=true (QLoRA); quantised weights cannot be trained directly"
        )
    kwargs: dict[str, Any] = {
        "model_name": mc.name_or_path,
        "max_seq_length": tc.max_seq_len,
        "dtype": unsloth_dtype(mc.dtype),
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "full_finetuning": not mc.lora.enabled,
        "device_map": {"": str(device)},
        "revision": mc.revision,
        "trust_remote_code": mc.trust_remote_code,
        "use_gradient_checkpointing": "unsloth" if tc.gradient_checkpointing else False,
        "fast_inference": False,
    }
    extra = tc.extra.get("unsloth_kwargs", {})
    if not isinstance(extra, dict):
        raise ConfigError("trainer.extra.unsloth_kwargs must be a mapping")
    kwargs.update(extra)
    return kwargs


def get_peft_model_kwargs(config: RunConfig) -> dict[str, Any]:
    """Keyword arguments for ``FastLanguageModel.get_peft_model`` from ``model.lora``."""
    lora, tc = config.model.lora, config.trainer
    kwargs: dict[str, Any] = {
        "r": lora.r,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "target_modules": lora.target_modules,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth" if tc.gradient_checkpointing else False,
        "random_state": config.seed,
    }
    extra = tc.extra.get("unsloth_lora_kwargs", {})
    if not isinstance(extra, dict):
        raise ConfigError("trainer.extra.unsloth_lora_kwargs must be a mapping")
    kwargs.update(extra)
    return kwargs


def load_unsloth_model(config: RunConfig, *, device: torch.device) -> tuple[nn.Module, Any, Any]:
    """``(model, tokenizer, FastLanguageModel)``; LoRA applied through Unsloth when enabled."""
    unsloth = import_unsloth(strict_import_order=bool(config.trainer.extra.get("strict_import_order", False)))
    fast = unsloth.FastLanguageModel
    model, tokenizer = fast.from_pretrained(**from_pretrained_kwargs(config, device))
    if config.model.lora.enabled:
        model = fast.get_peft_model(model, **get_peft_model_kwargs(config))
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.model.chat_template is not None:
        tokenizer.chat_template = config.model.chat_template
    return model, tokenizer, fast


# -- weight export ----------------------------------------------------------------------------


def is_4bit_weight(weight: Any) -> bool:
    """bitsandbytes ``Params4bit`` carry a ``quant_state`` (``bnb.nn.Params4bit``)."""
    return getattr(weight, "quant_state", None) is not None


def dequantize_4bit(weight: Any) -> torch.Tensor:
    """16-bit tensor for a bnb 4-bit parameter: Unsloth's ``fast_dequantize`` when importable,
    else ``bitsandbytes.functional.dequantize_4bit`` (both take ``(W, quant_state)``)."""
    quant_state = weight.quant_state
    try:
        from unsloth.kernels.utils import fast_dequantize
    except ImportError:
        from bitsandbytes.functional import dequantize_4bit as fast_dequantize
    return fast_dequantize(weight, quant_state)


def base_weight(layer: Any) -> torch.Tensor:
    weight = layer.weight
    return dequantize_4bit(weight) if is_4bit_weight(weight) else weight.detach()


def merge_lora_layer(layer: Any) -> torch.Tensor:
    """``W + sum_a scaling_a * B_a @ A_a`` for a PEFT ``LoraLayer`` (4-bit base dequantised first),
    accumulated in fp32 and returned in the base weight's dtype. Mirrors ``unsloth.save._merge_lora``."""
    base = layer.get_base_layer()
    w = base_weight(base)
    out_dtype = w.dtype
    merged = w.to(torch.float32).clone()
    for adapter in layer.active_adapters:
        if adapter not in layer.lora_A:
            continue
        a = layer.lora_A[adapter].weight.to(device=merged.device, dtype=torch.float32)
        b = layer.lora_B[adapter].weight.to(device=merged.device, dtype=torch.float32)
        merged.addmm_(b, a, alpha=float(layer.scaling[adapter]))
    if not torch.isfinite(merged).all():
        raise RuntimeError("LoRA merge produced non-finite values")
    return merged.to(out_dtype)


def merge_lora_bias(layer: Any) -> torch.Tensor | None:
    base = layer.get_base_layer()
    bias = getattr(base, "bias", None)
    if bias is None:
        return None
    merged = bias.detach().clone()
    for adapter in layer.active_adapters:
        if getattr(layer, "lora_bias", {}).get(adapter, False):
            merged += layer.lora_B[adapter].bias.to(merged.dtype) * layer.scaling[adapter]
    return merged


def iter_merged_weights(hf_model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """HF-named, merged, dequantised ``(name, tensor)`` pairs for ``hf_model`` (the base model
    PEFT injected its layers into, i.e. ``peft_model.get_base_model()``). Non-LoRA 4-bit linears are
    dequantised too; bnb quant-state entries and ``lora_*`` parameters are not exported."""
    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:  # no peft: nothing is a LoRA layer
        LoraLayer = ()  # type: ignore[assignment]

    persistent = set(hf_model.state_dict().keys())
    lora_prefixes: list[str] = []
    with torch.no_grad():
        for mod_name, module in hf_model.named_modules():
            prefix = f"{mod_name}." if mod_name else ""
            if any(mod_name.startswith(p) for p in lora_prefixes):
                continue  # base_layer / lora_A / lora_B / lora_dropout children of a merged layer
            if LoraLayer and isinstance(module, LoraLayer):
                lora_prefixes.append(prefix)
                yield f"{prefix}weight", merge_lora_layer(module)
                bias = merge_lora_bias(module)
                if bias is not None:
                    yield f"{prefix}bias", bias
                continue
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and is_4bit_weight(weight):
                yield f"{prefix}weight", dequantize_4bit(weight)
                bias = getattr(module, "bias", None)
                if bias is not None:
                    yield f"{prefix}bias", bias.detach()
                continue
            for pname, param in module.named_parameters(recurse=False):
                yield f"{prefix}{pname}", param.detach()
            for bname, buf in module.named_buffers(recurse=False):
                if f"{prefix}{bname}" in persistent:
                    yield f"{prefix}{bname}", buf.detach()


# -- trainer -----------------------------------------------------------------------------------


class UnslothTrainer:
    """:class:`~monorl.core.interfaces.Trainer` backed by an Unsloth-loaded model inside
    :class:`~monorl.trainer.ReferenceTrainer`."""

    def __init__(self, config: RunConfig, *, loss_fn: Any | None = None) -> None:
        from monorl.trainer.reference import ReferenceTrainer

        ensure_cuda()
        self.config = config
        self.device = resolve_device(config.trainer.device)
        pin_logits_env()
        model, tokenizer, self._fast = load_unsloth_model(config, device=self.device)
        self.inner = ReferenceTrainer(config, loss_fn=loss_fn, model=model, tokenizer=tokenizer)
        self.model: nn.Module = self.inner.model
        self.hf_model: nn.Module = self.inner.hf_model
        self.tokenizer = self.inner.tokenizer
        self.lora_enabled = self.inner.lora_enabled
        self._training_mode = False
        self._for_training()

    def _for_training(self) -> None:
        pin_logits_env()
        self._fast.for_training(self.model)
        self._training_mode = True

    # -- Trainer ----------------------------------------------------------------------------
    @property
    def step(self) -> int:
        return self.inner.step

    def train_step(self, batch: TrainBatch) -> dict[str, float]:
        if not self._training_mode or getattr(self.model, "_flag_for_generation", False):
            self._for_training()
        pin_logits_env()
        return self.inner.train_step(batch)

    def compute_logprobs(self, batch: TrainBatch, *, reference: bool = False) -> torch.Tensor:
        pin_logits_env()
        return self.inner.compute_logprobs(batch, reference=reference)

    def compute_values(self, batch: TrainBatch) -> torch.Tensor:
        """The reference trainer's value head (``algorithm.name: ppo``). It reads the input of
        ``lm_head``, which Unsloth's forward calls as a module whenever it returns logits
        (``CausalLM_fast_forward`` with ``UNSLOTH_RETURN_LOGITS=1`` and no labels)."""
        pin_logits_env()
        return self.inner.compute_values(batch)

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter_merged_weights(self.hf_model)

    def save_checkpoint(self, path: Path) -> None:
        self.inner.save_checkpoint(path)

    def load_checkpoint(self, path: Path) -> None:
        self.inner.load_checkpoint(path)
        self._training_mode = False

    def save_pretrained(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_merged = getattr(self.model, "save_pretrained_merged", None)
        if save_merged is None:  # model not loaded through Unsloth's loader (patch_saving_functions not run)
            from unsloth.save import unsloth_generic_save_pretrained_merged

            unsloth_generic_save_pretrained_merged(
                self.model, str(path), self.tokenizer, save_method="merged_16bit"
            )
        else:
            save_merged(str(path), self.tokenizer, save_method="merged_16bit")
        self._training_mode = False  # Unsloth's save path may switch the model to inference mode


def make_unsloth_trainer(config: RunConfig, **kwargs: Any) -> UnslothTrainer:
    return UnslothTrainer(config, loss_fn=kwargs.get("loss_fn"))
