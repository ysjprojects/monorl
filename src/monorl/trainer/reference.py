"""Reference :class:`~monorl.core.interfaces.Trainer`: plain torch + transformers (+ PEFT LoRA).

Runs on CPU/MPS/CUDA. Logprobs are computed as ``log_softmax(logits / rollout.temperature)`` so
they are directly comparable with the rollout engines' temperature-scaled sampling logprobs
(importance ratio 1 on-policy). The loss is any callable with the
``monorl.algorithms.policy_loss`` signature; by default that function is imported lazily.

Data parallelism
----------------
With ``world_size > 1`` (``rank``/``world_size`` kwargs, else ``RANK``/``WORLD_SIZE`` env) every
rank holds a full replica and trains on its shard of the :class:`TrainBatch` (see
:class:`~monorl.distributed.ray_runtime.RayTrainer`). Gradients are synchronised by a manual
bucketed all-reduce **sum** of the trainable parameters' grads rather than by
``DistributedDataParallel``:

* micro-batch losses are weighted by the *global* response-token (or sequence) count, so the
  summed gradient is bit-for-bit the single-process gradient over the whole batch even when
  shards hold different numbers of tokens (DDP's mean-of-per-rank-means is not);
* no module wrapper: ``compute_logprobs``, ``export_weights``, PEFT ``disable_adapter`` and the
  ``save_*`` paths see the plain (PEFT) model, and there is no ``no_sync``/unused-parameter
  bookkeeping for gradient accumulation or LoRA's frozen base weights.

Trainable parameters are broadcast from rank 0 after construction (LoRA adapters are randomly
initialised), loss metrics are reduced to their global weighted means, ``grad_norm`` is the global
norm (grads are identical after the all-reduce), rank 0 writes checkpoints/exports behind a barrier
and every rank loads them. The single-process path never touches ``torch.distributed``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from transformers import get_scheduler

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch
from monorl.rollout.hf_engine import iter_hf_weights, load_tokenizer, resolve_device
from monorl.rollout.media import auto_model_class, collate_media

log = logging.getLogger(__name__)

CHECKPOINT_STATE = "trainer_state.pt"
CHECKPOINT_MODEL = "model"
CHECKPOINT_ADAPTER = "adapter"
GRAD_BUCKET_BYTES = 64 << 20


def _resolve_rank(rank: int | None, world_size: int | None) -> tuple[int, int]:
    ws = int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else int(world_size)
    r = int(os.environ.get("RANK", "0")) if rank is None else int(rank)
    if ws < 1 or not 0 <= r < ws:
        raise ValueError(f"invalid trainer rank {r} for world_size {ws}")
    return r, ws


class _Collective:
    """Process-group operations for a data-parallel trainer; only built when ``world_size > 1``."""

    def __init__(self, rank: int, world_size: int, device: torch.device) -> None:
        import torch.distributed as dist

        self.dist = dist
        self.rank = rank
        self.world_size = world_size
        self.owns_group = not dist.is_initialized()
        if self.owns_group:
            backend = "nccl" if device.type == "cuda" else "gloo"
            if device.type == "cuda" and device.index is not None:
                torch.cuda.set_device(device)
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        # NCCL needs device tensors; gloo wants CPU tensors (MPS grads are staged through CPU).
        self.comm_device = device if dist.get_backend() == "nccl" else torch.device("cpu")

    def all_gather_object(self, obj: Any) -> list[Any]:
        out: list[Any] = [None] * self.world_size
        self.dist.all_gather_object(out, obj)
        return out

    def broadcast_params(self, params: list[torch.Tensor]) -> None:
        with torch.no_grad():
            for flat, views in self._buckets([p.data for p in params]):
                self.dist.broadcast(flat, src=0)
                self._scatter(flat, views)

    def all_reduce_grads(self, params: list[torch.Tensor]) -> None:
        grads: list[torch.Tensor] = []
        for p in params:
            if p.grad is None:  # parameter unused on this rank: contribute zeros, keep collectives aligned
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)
        with torch.no_grad():
            for flat, views in self._buckets(grads):
                self.dist.all_reduce(flat, op=self.dist.ReduceOp.SUM)
                self._scatter(flat, views)

    def barrier(self) -> None:
        self.dist.barrier()

    def close(self) -> None:
        if self.owns_group and self.dist.is_initialized():
            self.dist.destroy_process_group()

    def _buckets(self, tensors: list[torch.Tensor]) -> Iterator[tuple[torch.Tensor, list[torch.Tensor]]]:
        """Yields ``(flat, views)``: ``flat`` is a contiguous same-dtype bucket of at most
        :data:`GRAD_BUCKET_BYTES` on ``comm_device``; copy it back with :meth:`_scatter`."""
        views: list[torch.Tensor] = []
        pending = 0
        for t in tensors:
            size = t.numel() * t.element_size()
            if views and (t.dtype != views[0].dtype or pending + size > GRAD_BUCKET_BYTES):
                yield self._flatten(views), views
                views, pending = [], 0
            views.append(t)
            pending += size
        if views:
            yield self._flatten(views), views

    def _flatten(self, views: list[torch.Tensor]) -> torch.Tensor:
        flat = views[0].reshape(-1) if len(views) == 1 else torch.cat([v.reshape(-1) for v in views])
        return flat.to(self.comm_device)

    @staticmethod
    def _scatter(flat: torch.Tensor, views: list[torch.Tensor]) -> None:
        offset = 0
        for v in views:
            n = v.numel()
            v.copy_(flat[offset : offset + n].view_as(v))
            offset += n


class PolicyLossOutput(Protocol):
    loss: torch.Tensor
    metrics: dict[str, float]


LossFn = Callable[..., PolicyLossOutput]
"""``(logprobs, old_logprobs, advantages, response_mask, *, algorithm, ref_logprobs=None, entropy=None)``."""


def _default_loss_fn(algorithm: Any) -> LossFn:
    from monorl.algorithms import select_loss_fn

    return select_loss_fn(algorithm)


def sequence_logprobs_and_entropy(
    logits: torch.Tensor, input_ids: torch.Tensor, temperature: float, *, entropy: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``[B,T]`` logprob of ``input_ids[:, t]`` given the prefix (position 0 is 0) and, optionally,
    the entropy of the predictive distribution at every position (position 0 is 0)."""
    scaled = logits[:, :-1].float() / temperature
    logp_all = torch.log_softmax(scaled, dim=-1)
    lp = logp_all.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    lp = torch.nn.functional.pad(lp, (1, 0))
    ent = None
    if entropy:
        ent = -(logp_all.exp() * logp_all).sum(-1)
        ent = torch.nn.functional.pad(ent, (1, 0))
    return lp, ent


class ReferenceTrainer:
    """Full fine-tuning or LoRA on a HF causal LM with micro-batched gradient accumulation."""

    def __init__(
        self,
        config: RunConfig,
        *,
        loss_fn: LossFn | None = None,
        model: nn.Module | None = None,
        tokenizer: Any | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        """``model``/``tokenizer`` inject pre-built objects (e.g. Unsloth's ``FastLanguageModel``):
        an injected model is assumed to be on its device already and, when LoRA is enabled, may be
        a ready ``PeftModel`` (used as-is) or a base model (wrapped here). ``rank``/``world_size``
        default to ``RANK``/``WORLD_SIZE`` (then ``0``/``1``); ``world_size > 1`` joins (or
        initialises, from ``MASTER_ADDR``/``MASTER_PORT``) the ``torch.distributed`` group."""
        self.config = config
        self.max_seq_len = config.trainer.max_seq_len
        self.device = resolve_device(config.trainer.device)
        self.temperature = config.rollout.temperature if config.rollout.temperature > 0 else 1.0
        self._loss_fn = loss_fn
        self._step = 0
        self.rank, self.world_size = _resolve_rank(rank, world_size)
        self._dist = _Collective(self.rank, self.world_size, self.device) if self.world_size > 1 else None

        self.precision = config.trainer.precision if self.device.type != "cpu" else "fp32"
        self.param_dtype = (
            torch.float32 if config.model.dtype in ("auto", "float32") else getattr(torch, config.model.dtype)
        )
        if self.device.type == "cpu":
            self.param_dtype = torch.float32
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.precision)
        self.scaler = (
            torch.amp.GradScaler("cuda") if self.precision == "fp16" and self.device.type == "cuda" else None
        )

        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(config)
        self.lora_enabled = config.model.lora.enabled
        if model is None:
            self.hf_model: nn.Module = self._load_base()
            self.hf_model.to(self.device)
            self.model: nn.Module = self._wrap_lora(self.hf_model) if self.lora_enabled else self.hf_model
        elif self.lora_enabled and hasattr(model, "peft_config") and hasattr(model, "get_base_model"):
            self.model = model
            self.hf_model = model.get_base_model()
        else:
            self.hf_model = model
            self.model = self._wrap_lora(self.hf_model) if self.lora_enabled else self.hf_model
        if config.trainer.gradient_checkpointing and not getattr(
            self.hf_model, "is_gradient_checkpointing", False
        ):
            self.hf_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if self.lora_enabled:
                self.model.enable_input_require_grads()
        self.model.train()

        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self._dist is not None:
            self._dist.broadcast_params(self.trainable_params)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self._reference_model: nn.Module | None = None

    def shutdown(self) -> None:
        """Leaves the process group this trainer created (no-op for single-process runs)."""
        if self._dist is not None:
            self._dist.close()

    # -- construction ---------------------------------------------------------------------
    def _load_base(self) -> nn.Module:
        model = auto_model_class(self.config).from_pretrained(
            self.config.model.name_or_path,
            dtype=self.param_dtype,
            revision=self.config.model.revision,
            trust_remote_code=self.config.model.trust_remote_code,
            attn_implementation=self.config.model.attn_implementation,
        )
        model.config.use_cache = False
        return model

    def _wrap_lora(self, model: nn.Module) -> nn.Module:
        from peft import LoraConfig, get_peft_model

        lora = self.config.model.lora
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules,
        )
        return get_peft_model(model, peft_config)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        tc = self.config.trainer
        if tc.optimizer == "adamw":
            return torch.optim.AdamW(self.trainable_params, lr=tc.learning_rate, weight_decay=tc.weight_decay)
        if tc.optimizer == "adam":
            return torch.optim.Adam(self.trainable_params, lr=tc.learning_rate, weight_decay=tc.weight_decay)
        return torch.optim.SGD(self.trainable_params, lr=tc.learning_rate, weight_decay=tc.weight_decay)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        tc = self.config.trainer
        total = tc.max_steps * self.config.algorithm.num_epochs
        name = tc.scheduler
        if name == "constant":
            name = "constant_with_warmup" if tc.warmup_steps > 0 else "constant"
        return get_scheduler(name, self.optimizer, num_warmup_steps=tc.warmup_steps, num_training_steps=total)

    @property
    def loss_fn(self) -> LossFn:
        if self._loss_fn is None:
            self._loss_fn = _default_loss_fn(self.config.algorithm)
        return self._loss_fn

    # -- Trainer --------------------------------------------------------------------------
    @property
    def step(self) -> int:
        return self._step

    def train_step(self, batch: TrainBatch) -> dict[str, float]:
        batch = self._prepare(batch)
        local_tokens = int(batch.response_mask.sum().item())
        local_seqs = len(batch)
        # Sequence-mode loss averages over sequences that have at least one response token (the
        # loss skips empty ones), so the micro-batch weights must use that count too.
        local_scored = int((batch.response_mask.sum(dim=1) > 0).sum().item())
        total_tokens, total_seqs, total_scored = local_tokens, local_seqs, local_scored
        if (
            self._dist is not None
        ):  # weights use global counts so the summed gradient is the full-batch gradient
            counts = self._dist.all_gather_object((local_tokens, local_seqs, local_scored))
            total_tokens, total_seqs, total_scored = (sum(c[i] for c in counts) for i in range(3))
        total_tokens = max(total_tokens, 1)
        total_scored = max(total_scored, 1)
        mbs = self.config.trainer.micro_batch_size
        token_agg = self.config.algorithm.loss_agg == "token"
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        sums: dict[str, float] = {}
        weights_total = 0.0
        for start in range(0, local_seqs, mbs):
            micro = batch.slice(start, start + mbs)
            micro_tokens = int(micro.response_mask.sum().item())
            micro_scored = int((micro.response_mask.sum(dim=1) > 0).sum().item())
            weight = micro_tokens / total_tokens if token_agg else micro_scored / total_scored
            if micro_tokens == 0:
                continue
            with self._autocast():
                logits = self._forward(self.model, micro)
            logprobs, entropy = sequence_logprobs_and_entropy(
                logits, micro.input_ids, self.temperature, entropy=True
            )
            loss_kwargs: dict[str, Any] = {}
            if self.config.algorithm.name == "sft" and self.config.algorithm.sft_label_smoothing > 0:
                from monorl.algorithms import mean_logprob_all_tokens

                loss_kwargs["mean_logprob_all_tokens"] = mean_logprob_all_tokens(logits, self.temperature)
            del logits
            mask = micro.response_mask.float()
            logprobs = logprobs * mask
            out = self.loss_fn(
                logprobs,
                micro.old_logprobs,
                micro.advantages,
                micro.response_mask,
                algorithm=self.config.algorithm,
                ref_logprobs=micro.ref_logprobs,
                entropy=entropy,
                **loss_kwargs,
            )
            scaled = out.loss * weight
            if self.scaler is not None:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            metrics = dict(out.metrics)
            metrics.setdefault(
                "loss/entropy", float((entropy.detach() * mask).sum().item() / max(micro_tokens, 1))
            )
            metrics["loss/total"] = float(out.loss.detach().item())
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v) * weight
            weights_total += weight

        if self._dist is not None:
            self._dist.all_reduce_grads(self.trainable_params)
            gathered = self._dist.all_gather_object((sums, weights_total))
            sums, weights_total = {}, 0.0
            for other_sums, other_total in gathered:
                for k, v in other_sums.items():
                    sums[k] = sums.get(k, 0.0) + v
                weights_total += other_total
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.trainer.max_grad_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._step += 1

        result = {k: v / weights_total for k, v in sums.items()} if weights_total > 0 else {}
        result["grad_norm"] = float(grad_norm.item())
        result["lr"] = float(self.scheduler.get_last_lr()[0])
        result["tokens/response_mean"] = float(total_tokens / total_seqs)
        return result

    @torch.no_grad()
    def compute_logprobs(self, batch: TrainBatch, *, reference: bool = False) -> torch.Tensor:
        src_device = batch.input_ids.device
        prepared = self._prepare(batch)
        mbs = self.config.trainer.micro_batch_size
        out = torch.zeros(prepared.input_ids.shape, dtype=torch.float32, device=self.device)
        with self._reference_context(reference) as model:
            was_training = model.training
            model.eval()
            for start in range(0, len(prepared), mbs):
                micro = prepared.slice(start, start + mbs)
                with self._autocast():
                    logits = self._forward(model, micro)
                lp, _ = sequence_logprobs_and_entropy(
                    logits, micro.input_ids, self.temperature, entropy=False
                )
                out[start : start + len(micro)] = lp * micro.response_mask.float()
            model.train(was_training)
        return out.to(src_device)

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        if not self.lora_enabled:
            for name, tensor in self.hf_model.state_dict().items():
                yield name, tensor.detach()
            return
        from peft.tuners.lora import LoraLayer

        lora_layers = {name: m for name, m in self.hf_model.named_modules() if isinstance(m, LoraLayer)}
        with torch.no_grad():
            for name, tensor in self.hf_model.state_dict().items():
                if ".lora_" in name:
                    continue
                if name.endswith(".base_layer.weight"):
                    prefix = name[: -len(".base_layer.weight")]
                    layer = lora_layers[prefix]
                    merged = tensor.detach().clone()
                    for adapter in layer.active_adapters:
                        if adapter in layer.lora_A:
                            merged += layer.get_delta_weight(adapter).to(merged.dtype)
                    yield prefix + ".weight", merged
                elif name.endswith(".base_layer.bias"):
                    prefix = name[: -len(".base_layer.bias")]
                    layer = lora_layers[prefix]
                    merged = tensor.detach().clone()
                    for adapter in layer.active_adapters:
                        if layer.lora_bias.get(adapter, False):
                            merged += layer.lora_B[adapter].bias * layer.scaling[adapter]
                    yield prefix + ".bias", merged
                else:
                    yield name.replace(".base_layer", ""), tensor.detach()

    def save_checkpoint(self, path: Path) -> None:
        """Rank 0 writes; every rank returns once the files are on disk."""
        path = Path(path)
        if self._writer:
            path.mkdir(parents=True, exist_ok=True)
            if self.lora_enabled:
                self.model.save_pretrained(str(path / CHECKPOINT_ADAPTER))
            else:
                self.hf_model.save_pretrained(str(path / CHECKPOINT_MODEL), safe_serialization=True)
            state: dict[str, Any] = {
                "step": self._step,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            }
            torch.save(state, path / CHECKPOINT_STATE)
        self._sync_writers()

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        if self.lora_enabled:
            from peft import load_peft_weights, set_peft_model_state_dict

            weights = load_peft_weights(str(path / CHECKPOINT_ADAPTER), device=str(self.device))
            result = set_peft_model_state_dict(self.model, weights)
            unexpected = getattr(result, "unexpected_keys", [])
            if unexpected:
                raise RuntimeError(f"checkpoint adapter has unexpected keys: {sorted(unexpected)[:5]}")
        else:
            state = self.hf_model.state_dict()
            with torch.no_grad():
                for name, tensor in iter_hf_weights(path / CHECKPOINT_MODEL):
                    state[name].copy_(tensor.to(device=state[name].device, dtype=state[name].dtype))
        state = torch.load(path / CHECKPOINT_STATE, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        self._step = int(state["step"])
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

    def save_pretrained(self, path: Path) -> None:
        path = Path(path)
        if self._writer:
            path.mkdir(parents=True, exist_ok=True)
            if self.lora_enabled:
                merged = {name: tensor.detach().to("cpu") for name, tensor in self.export_weights()}
                self.hf_model.save_pretrained(str(path), state_dict=merged, safe_serialization=True)
            else:
                self.hf_model.save_pretrained(str(path), safe_serialization=True)
            self.tokenizer.save_pretrained(str(path))
        self._sync_writers()

    @property
    def _writer(self) -> bool:
        return self._dist is None or self.rank == 0

    def _sync_writers(self) -> None:
        if self._dist is not None:
            self._dist.barrier()

    # -- internals ------------------------------------------------------------------------
    def _prepare(self, batch: TrainBatch) -> TrainBatch:
        if batch.input_ids.shape[1] > self.max_seq_len:
            width = batch.input_ids.shape[1]
            batch = TrainBatch(
                input_ids=batch.input_ids[:, : self.max_seq_len],
                attention_mask=batch.attention_mask[:, : self.max_seq_len],
                response_mask=batch.response_mask[:, : self.max_seq_len],
                advantages=batch.advantages[:, : self.max_seq_len],
                rewards=batch.rewards,
                group_ids=batch.group_ids,
                old_logprobs=None
                if batch.old_logprobs is None
                else batch.old_logprobs[:, : self.max_seq_len],
                ref_logprobs=None
                if batch.ref_logprobs is None
                else batch.ref_logprobs[:, : self.max_seq_len],
                position_ids=None
                if batch.position_ids is None
                else batch.position_ids[:, : self.max_seq_len],
                meta=batch.meta,
                extra={
                    k: v[:, : self.max_seq_len]
                    if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] == width
                    else v
                    for k, v in batch.extra.items()
                },
            )
        return batch.to(self.device)

    def _forward(self, model: nn.Module, micro: TrainBatch) -> torch.Tensor:
        """Logits for one micro-batch. ``extra`` tensors (``[B, T]``-aligned, e.g.
        ``mm_token_type_ids``) are passed as-is; ``extra["per_sequence"]`` media tensors are
        concatenated for the micro-batch (``pixel_values``, ``image_grid_thw``, ...). Multimodal
        batches let the model derive its own position ids (Qwen2-VL M-RoPE needs 3-D positions,
        which the 1-D ``position_ids`` of a text batch would silently override)."""
        kwargs: dict[str, Any] = {}
        position_ids = micro.position_ids
        for k, v in micro.extra.items():
            if k == "per_sequence":
                kwargs.update(collate_media(v))
                position_ids = None
            elif isinstance(v, torch.Tensor):
                kwargs[k] = v
        return model(
            input_ids=micro.input_ids,
            attention_mask=micro.attention_mask,
            position_ids=position_ids,
            **kwargs,
        ).logits

    def _autocast(self) -> Any:
        if self.autocast_dtype is None:
            return torch.autocast(self.device.type, enabled=False)
        return torch.autocast(self.device.type, dtype=self.autocast_dtype)

    def _reference_context(self, reference: bool) -> Any:
        """Yields the module to run: the policy, the LoRA model with adapters disabled, or a frozen
        copy of the initial weights (loaded from ``model.name_or_path`` on first use)."""
        if not reference:
            return contextlib.nullcontext(self.model)
        if self.lora_enabled:

            @contextlib.contextmanager
            def disabled() -> Iterator[nn.Module]:
                with self.model.disable_adapter():
                    yield self.model

            return disabled()
        if self._reference_model is None:
            ref = self._load_base()
            ref.to(self.device)
            ref.eval()
            ref.requires_grad_(False)
            self._reference_model = ref
        return contextlib.nullcontext(self._reference_model)
