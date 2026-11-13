"""Reference :class:`~monorl.core.interfaces.Trainer`: plain torch + transformers (+ PEFT LoRA).

Runs on CPU/MPS/CUDA. Logprobs are computed as ``log_softmax(logits / rollout.temperature)`` so
they are directly comparable with the rollout engines' temperature-scaled sampling logprobs
(importance ratio 1 on-policy). The loss is any callable with the
``monorl.algorithms.policy_loss`` signature; by default that function is imported lazily.

Value baseline
--------------
``algorithm.name: ppo`` adds a critic: a zero-initialised linear value head on the policy's final
hidden states (the input of its output embedding), read at the positions
``TrainBatch.value_mask`` marks, i.e. the last prompt token before a turn's completion.
:meth:`ReferenceTrainer.compute_values` is the no-grad pass the orchestrator computes advantages
from; ``train_step`` adds ``algorithm.value_coef`` times the clipped value loss
(:func:`monorl.algorithms.value_loss`) of the batch's ``returns``, averaged over all value
positions of the batch. The head trains with the policy (its gradient reaches the policy's
trainable parameters) at ``VALUE_LR_SCALE`` times the learning rate, is part of the gradient
all-reduce and of ``trainer_state.pt``, and is never exported: ``export_weights`` and
``save_pretrained`` are the policy alone.

MTP layers
----------
``model.mtp.enabled`` holds the checkpoint's multi-token-prediction layers next to the policy
(:class:`monorl.mtp.MtpLayers`; they share its embedding and output head). They follow the policy
in ``export_weights`` (checkpoint keys behind ``monorl.mtp.WIRE_PREFIX``, so engines can draft with
them) and in ``save_pretrained`` (the checkpoint's own keys: ``final/`` is a complete checkpoint
again), and they are part of ``trainer_state.pt``.
``model.mtp.train`` trains them: ``train_step`` adds ``loss_coef`` times :meth:`MtpLayers.loss` on
the response tokens, computed from the hidden states of the policy's last decoder layer. They are
full parameters (also next to LoRA adapters) in an optimizer group of their own at the policy's
learning rate, schedule and weight decay. Their gradient is clipped to ``max_grad_norm`` on its
own: with ``detach_trunk`` the policy's update is exactly what it is without MTP training, and
``grad_norm`` stays the policy's. A micro-batch's MTP loss is its share of the batch's (the target
counts are global across ranks), so the accumulated gradient is the full-batch gradient. Without
``train`` the layers are frozen and no forward pass runs through them.

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
from monorl.core.errors import ConfigError
from monorl.mtp import MtpLayers, resolve_checkpoint_dir, write_checkpoint_weights
from monorl.rollout.hf_engine import hf_weight_names, iter_hf_weights, load_tokenizer, resolve_device
from monorl.rollout.media import auto_model_class, collate_media

log = logging.getLogger(__name__)

CHECKPOINT_STATE = "trainer_state.pt"
CHECKPOINT_MODEL = "model"
CHECKPOINT_ADAPTER = "adapter"
#: Learning rate of the value head relative to ``trainer.learning_rate`` (the verl compilation of
#: ``ppo`` gives its critic the same factor).
VALUE_LR_SCALE = 10.0
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
"""``(logprobs, old_logprobs, advantages, response_mask, *, algorithm, ref_logprobs=None, entropy=None)``;
``loss_weights``, ``behavior_logprobs`` and ``top_logprobs``/``behavior_top_logprobs`` are passed as
further keywords only for batches that carry them."""


def _default_loss_fn(algorithm: Any) -> LossFn:
    from monorl.algorithms import select_loss_fn

    return select_loss_fn(algorithm)


def sequence_logprobs_and_entropy(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float,
    *,
    entropy: bool,
    top_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """``[B,T]`` logprob of ``input_ids[:, t]`` given the prefix (position 0 is 0), optionally the
    entropy of the predictive distribution at every position (position 0 is 0) and, for
    ``top_ids`` (``[B,T,K]`` candidate tokens per position), their ``[B,T,K]`` logprobs under the
    same distribution (position 0 is 0)."""
    scaled = logits[:, :-1].float() / temperature
    logp_all = torch.log_softmax(scaled, dim=-1)
    lp = logp_all.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    lp = torch.nn.functional.pad(lp, (1, 0))
    ent = None
    if entropy:
        ent = -(logp_all.exp() * logp_all).sum(-1)
        ent = torch.nn.functional.pad(ent, (1, 0))
    top = None
    if top_ids is not None:
        top = torch.nn.functional.pad(logp_all.gather(-1, top_ids[:, 1:]), (0, 0, 1, 0))
    return lp, ent, top


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
        self.mtp: MtpLayers | None = None
        self._mtp_params: list[nn.Parameter] = []
        if config.model.mtp.enabled:
            # After the LoRA wrapping: the layers share the output head the policy really uses.
            self.mtp = self._load_mtp()
            self.mtp.model.train(config.model.mtp.train)
            for p in self.mtp.parameters():
                p.requires_grad_(config.model.mtp.train)
            if config.model.mtp.train:
                self._mtp_params = self.mtp.parameters()
        self.model.train()

        self._policy_params = [p for p in self.model.parameters() if p.requires_grad]
        self.value_head: nn.Linear | None = None
        self._value_params: list[nn.Parameter] = []
        if config.algorithm.name == "ppo":
            self.value_head = self._build_value_head()
            self._value_params = list(self.value_head.parameters())
        self.trainable_params = self._policy_params + self._value_params + self._mtp_params
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
    def _load_base(self, checkpoint: Path | None = None) -> nn.Module:
        """The configured model, or the model a trainer checkpoint holds (same loading options)."""
        model = auto_model_class(self.config).from_pretrained(
            checkpoint or self.config.model.name_or_path,
            dtype=self.param_dtype,
            revision=None if checkpoint else self.config.model.revision,
            trust_remote_code=self.config.model.trust_remote_code,
            attn_implementation=self.config.model.attn_implementation,
        )
        model.config.use_cache = False
        return model

    def _load_mtp(self) -> MtpLayers:
        model = self.config.model
        return MtpLayers(self.hf_model, resolve_checkpoint_dir(model.name_or_path, model.revision))

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

    def _output_embedding(self) -> nn.Module:
        module = self.hf_model.get_output_embeddings()
        if module is None:
            raise ConfigError(
                "algorithm.name='ppo' reads the value head's input at the model's output embedding (lm_head), "
                f"which {type(self.hf_model).__name__} does not expose"
            )
        return module

    def _build_value_head(self) -> nn.Linear:
        """``hidden -> 1`` in float32, zero-initialised: the first advantages are plain returns and
        the policy's parameters get no value gradient before the head has moved."""
        head = nn.Linear(self._output_embedding().weight.shape[-1], 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return head.to(self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        tc = self.config.trainer
        params: list[dict[str, Any]] = [{"params": self._policy_params}]
        if self._value_params:
            params.append({"params": self._value_params, "lr": tc.learning_rate * VALUE_LR_SCALE})
        if self._mtp_params:
            params.append({"params": self._mtp_params})
        if tc.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=tc.learning_rate, weight_decay=tc.weight_decay)
        if tc.optimizer == "adam":
            return torch.optim.Adam(params, lr=tc.learning_rate, weight_decay=tc.weight_decay)
        return torch.optim.SGD(params, lr=tc.learning_rate, weight_decay=tc.weight_decay)

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
        # With per-token loss weights (loss_agg="rollout") a micro-batch counts by its weight mass.
        local_mass = float(batch.loss_weights.sum().item()) if batch.loss_weights is not None else 0.0
        # The value loss is a mean over the batch's value positions, whatever loss_agg says.
        critic = self.value_head is not None and batch.value_mask is not None and batch.returns is not None
        local_states = int(batch.value_mask.sum().item()) if critic else 0
        # The MTP loss of every depth is a mean over the batch's targets of that depth.
        mtp_cfg = self.config.model.mtp
        mtp = self.mtp if mtp_cfg.train else None
        mtp_totals = mtp.target_counts(batch.response_mask) if mtp is not None else []
        total_tokens, total_seqs, total_scored, total_mass, total_states = (
            local_tokens,
            local_seqs,
            local_scored,
            local_mass,
            local_states,
        )
        if (
            self._dist is not None
        ):  # weights use global counts so the summed gradient is the full-batch gradient
            counts = self._dist.all_gather_object(
                (local_tokens, local_seqs, local_scored, local_mass, local_states, mtp_totals)
            )
            total_tokens, total_seqs, total_scored, total_mass, total_states = (
                sum(c[i] for c in counts) for i in range(5)
            )
            mtp_totals = [sum(depth) for depth in zip(*(c[5] for c in counts), strict=True)]
        total_tokens = max(total_tokens, 1)
        total_scored = max(total_scored, 1)
        mbs = self.config.trainer.micro_batch_size
        token_agg = self.config.algorithm.loss_agg == "token"
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        sums: dict[str, float] = {}
        weights_total = 0.0
        value_sums: dict[str, float] = {}
        value_weights_total = 0.0
        mtp_sums = [0.0] * len(mtp_totals)
        mtp_correct = 0
        # What generation hands to the MTP layers: the output of the last decoder layer, before the final norm.
        mtp_inputs: dict[str, Any] = {}
        if mtp is not None:
            num_layers = self.hf_model.config.get_text_config().num_hidden_layers
            mtp_inputs["output_hidden_states"] = [num_layers - 1]
        for start in range(0, local_seqs, mbs):
            micro = batch.slice(start, start + mbs)
            micro_tokens = int(micro.response_mask.sum().item())
            micro_scored = int((micro.response_mask.sum(dim=1) > 0).sum().item())
            micro_states = int(micro.value_mask.sum().item()) if critic else 0
            if micro.loss_weights is not None:
                weight = float(micro.loss_weights.sum().item()) / max(total_mass, 1e-8)
            else:
                weight = micro_tokens / total_tokens if token_agg else micro_scored / total_scored
            if micro_tokens == 0 and micro_states == 0:
                continue
            with contextlib.ExitStack() as stack:
                stack.enter_context(self._autocast())
                hidden = stack.enter_context(self._final_hidden()) if micro_states else None
                outputs = self._outputs(self.model, micro, **mtp_inputs)
            logits = outputs.logits
            last_layer = outputs.hidden_states[-1] if mtp is not None else None
            del outputs
            logprobs, entropy, top_logprobs = sequence_logprobs_and_entropy(
                logits, micro.input_ids, self.temperature, entropy=True, top_ids=micro.behavior_top_ids
            )
            loss_kwargs: dict[str, Any] = {}
            if micro.loss_weights is not None:
                loss_kwargs["loss_weights"] = micro.loss_weights
            if micro.behavior_logprobs is not None:
                loss_kwargs["behavior_logprobs"] = micro.behavior_logprobs
            if top_logprobs is not None:
                loss_kwargs["top_logprobs"] = top_logprobs
                loss_kwargs["behavior_top_logprobs"] = micro.behavior_top_logprobs
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
            if hidden is not None:
                from monorl.algorithms import value_loss

                states = micro.value_mask.bool()
                value_out = value_loss(
                    self._values(hidden, micro, states),
                    micro.returns[states],
                    None if micro.values is None else micro.values[states],
                    self.config.algorithm.value_clip or 0.0,
                )
                value_weight = micro_states / total_states
                scaled = scaled + self.config.algorithm.value_coef * value_out.loss * value_weight
                for k, v in value_out.metrics.items():
                    value_sums[k] = value_sums.get(k, 0.0) + float(v) * value_weight
                value_weights_total += value_weight
            del hidden
            if mtp is not None:
                with self._autocast():
                    mtp_out = mtp.loss(
                        last_layer,
                        micro.input_ids,
                        micro.response_mask,
                        detach_trunk=mtp_cfg.detach_trunk,
                        totals=mtp_totals,
                    )
                # already this micro-batch's share of the batch's loss
                scaled = scaled + mtp_cfg.loss_coef * mtp_out.loss
                mtp_sums = [a + b for a, b in zip(mtp_sums, mtp_out.sums, strict=True)]
                mtp_correct += mtp_out.correct
            del last_layer
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
            gathered = self._dist.all_gather_object(
                (sums, weights_total, value_sums, value_weights_total, mtp_sums, mtp_correct)
            )
            sums, weights_total, value_sums, value_weights_total = {}, 0.0, {}, 0.0
            mtp_sums, mtp_correct = [0.0] * len(mtp_totals), 0
            for (
                other_sums,
                other_total,
                other_value_sums,
                other_value_total,
                other_mtp,
                other_correct,
            ) in gathered:
                for k, v in other_sums.items():
                    sums[k] = sums.get(k, 0.0) + v
                weights_total += other_total
                for k, v in other_value_sums.items():
                    value_sums[k] = value_sums.get(k, 0.0) + v
                value_weights_total += other_value_total
                mtp_sums = [a + b for a, b in zip(mtp_sums, other_mtp, strict=True)]
                mtp_correct += other_correct
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        max_grad_norm = self.config.trainer.max_grad_norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self._policy_params + self._value_params, max_grad_norm)
        if self._mtp_params:  # on their own: the policy's clip does not depend on the MTP gradient
            torch.nn.utils.clip_grad_norm_(self._mtp_params, max_grad_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._step += 1

        result = {k: v / weights_total for k, v in sums.items()} if weights_total > 0 else {}
        if value_weights_total > 0:
            result.update({k: v / value_weights_total for k, v in value_sums.items()})
            result["loss/total"] = (
                result.get("loss/total", 0.0) + self.config.algorithm.value_coef * result["loss/value"]
            )
        depth_losses = {
            depth: entropy / count
            for depth, (entropy, count) in enumerate(zip(mtp_sums, mtp_totals, strict=True), start=1)
            if count > 0
        }
        if depth_losses:
            result["loss/mtp"] = sum(depth_losses.values()) / len(depth_losses)
            result["mtp/accuracy"] = mtp_correct / sum(mtp_totals)
            if len(mtp_totals) > 1:
                result.update({f"loss/mtp_{depth}": loss for depth, loss in depth_losses.items()})
            result["loss/total"] = result.get("loss/total", 0.0) + mtp_cfg.loss_coef * result["loss/mtp"]
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
                lp, _, _ = sequence_logprobs_and_entropy(
                    logits, micro.input_ids, self.temperature, entropy=False
                )
                out[start : start + len(micro)] = lp * micro.response_mask.float()
            model.train(was_training)
        return out.to(src_device)

    @torch.no_grad()
    def compute_values(self, batch: TrainBatch) -> torch.Tensor:
        """``[B, T]`` estimates of the value head (``algorithm.name: ppo``), no gradient: entry
        ``t`` values the state after ``input_ids[:, : t + 1]``, so a turn's value is the entry at
        its last prompt token (``TrainBatch.value_mask``). Zero at padding."""
        if self.value_head is None:
            raise ConfigError(
                f"the trainer has no value head: algorithm.name={self.config.algorithm.name!r}, not 'ppo'"
            )
        src_device = batch.input_ids.device
        prepared = self._prepare(batch)
        mbs = self.config.trainer.micro_batch_size
        out = torch.zeros(prepared.input_ids.shape, dtype=torch.float32, device=self.device)
        was_training = self.model.training
        self.model.eval()
        for start in range(0, len(prepared), mbs):
            micro = prepared.slice(start, start + mbs)
            with self._autocast(), self._final_hidden() as hidden:
                self._forward(self.model, micro)
            out[start : start + len(micro)] = self._values(hidden, micro) * micro.attention_mask
        self.model.train(was_training)
        return out.to(src_device)

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """The policy's weights (LoRA adapters merged) and then, with ``model.mtp.enabled``, the
        MTP layers under their checkpoint keys, tagged with ``monorl.mtp.WIRE_PREFIX``."""
        yield from self._policy_weights()
        if self.mtp is not None:
            yield from self.mtp.wire_weights()

    def _policy_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        if not self.lora_enabled:
            for name, tensor in self.hf_model.state_dict().items():
                yield name, tensor.detach()
            return
        from peft.tuners.lora import LoraLayer

        # state_dict name of every adapted tensor -> the layer that adapts it. PEFT wraps a module once
        # per adapted tensor: one wrapper around a linear layer, nested ones (`ParamWrapper`, which names
        # its tensor) around a module whose packed expert weights are adapted one by one.
        adapted: dict[str, Any] = {}
        biases: dict[str, Any] = {}
        for name, layer in self.hf_model.named_modules():
            if not isinstance(layer, LoraLayer):
                continue
            base, module = name, layer
            while isinstance(module, LoraLayer):
                base, module = base + ".base_layer", module.base_layer
            adapted[f"{base}.{getattr(layer, 'parameter_name', 'weight')}"] = layer
            if not hasattr(layer, "parameter_name"):
                biases[f"{base}.bias"] = layer
        with torch.no_grad():
            for name, tensor in self.hf_model.state_dict().items():
                if ".lora_" in name:
                    continue
                exported = name.replace(".base_layer", "")
                if name in adapted:
                    layer = adapted[name]
                    merged = tensor.detach().clone()
                    for adapter in layer.active_adapters:
                        if adapter in layer.lora_A:
                            merged += layer.get_delta_weight(adapter).to(merged.dtype)
                    yield exported, merged
                elif name in biases:
                    layer = biases[name]
                    merged = tensor.detach().clone()
                    for adapter in layer.active_adapters:
                        if layer.lora_bias.get(adapter, False):
                            merged += layer.lora_B[adapter].bias * layer.scaling[adapter]
                    yield exported, merged
                else:
                    yield exported, tensor.detach()

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
                "value_head": self.value_head.state_dict() if self.value_head is not None else None,
                "mtp": self.mtp.state_dict() if self.mtp is not None else None,
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
            stored = path / CHECKPOINT_MODEL
            with torch.no_grad():
                if all(name in state for name in hf_weight_names(stored)):
                    for name, tensor in iter_hf_weights(stored):
                        state[name].copy_(tensor.to(device=state[name].device, dtype=state[name].dtype))
                else:
                    # save_pretrained wrote the checkpoint's layout, which is not the state dict's for
                    # models whose weights are converted on load (packed expert weights): those are
                    # read by transformers' loader, at the price of a second copy for the moment.
                    saved = self._load_base(stored)
                    for name, tensor in saved.state_dict().items():
                        state[name].copy_(tensor.to(device=state[name].device, dtype=state[name].dtype))
                    del saved
        state = torch.load(path / CHECKPOINT_STATE, map_location="cpu", weights_only=False)
        if self.value_head is not None:
            if state.get("value_head") is None:
                raise RuntimeError(
                    f"checkpoint {path} holds no value head: it was not written by an algorithm.name='ppo' run"
                )
            self.value_head.load_state_dict(state["value_head"])
        if self.mtp is not None:
            if state.get("mtp") is None:
                raise RuntimeError(
                    f"checkpoint {path} holds no MTP layers: it was not written by a run with model.mtp.enabled"
                )
            self.mtp.load_state_dict(state["mtp"])
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
                policy = {name: tensor.detach().to("cpu") for name, tensor in self._policy_weights()}
                self.hf_model.save_pretrained(str(path), state_dict=policy, safe_serialization=True)
            else:
                policy = self.hf_model.state_dict()
                self.hf_model.save_pretrained(str(path), safe_serialization=True)
            if self.mtp is not None:  # save_pretrained leaves them out; with them it is a complete checkpoint
                write_checkpoint_weights(path, self.mtp.checkpoint_weights(tied_from=policy))
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

            def cut(t: torch.Tensor | None) -> torch.Tensor | None:
                return None if t is None else t[:, : self.max_seq_len]

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
                loss_weights=cut(batch.loss_weights),
                behavior_logprobs=cut(batch.behavior_logprobs),
                behavior_top_ids=cut(batch.behavior_top_ids),
                behavior_top_logprobs=cut(batch.behavior_top_logprobs),
                values=cut(batch.values),
                returns=cut(batch.returns),
                value_mask=cut(batch.value_mask),
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
        """Logits for one micro-batch."""
        return self._outputs(model, micro).logits

    def _outputs(self, model: nn.Module, micro: TrainBatch, **model_kwargs: Any) -> Any:
        """The model's output for one micro-batch. ``extra`` tensors (``[B, T]``-aligned, e.g.
        ``mm_token_type_ids``) are passed as-is; ``extra["per_sequence"]`` media tensors are
        concatenated for the micro-batch (``pixel_values``, ``image_grid_thw``, ...). Multimodal
        batches let the model derive its own position ids (Qwen2-VL M-RoPE needs 3-D positions,
        which the 1-D ``position_ids`` of a text batch would silently override)."""
        kwargs: dict[str, Any] = dict(model_kwargs)
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
        )

    @contextlib.contextmanager
    def _final_hidden(self) -> Iterator[list[torch.Tensor]]:
        """Collects the input of the policy's output embedding, i.e. the final hidden states
        ``[B, T, H]``, of every forward pass run inside the block. Reading it there costs no
        memory beyond the pass itself, unlike ``output_hidden_states``, which keeps every layer's
        output alive (also under gradient checkpointing)."""
        captured: list[torch.Tensor] = []
        handle = self._output_embedding().register_forward_pre_hook(
            lambda _module, args: captured.append(args[0])
        )
        try:
            yield captured
        finally:
            handle.remove()

    def _values(
        self, hidden: list[torch.Tensor], micro: TrainBatch, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Value-head estimates from the hidden states :meth:`_final_hidden` collected for
        ``micro``: ``[B, T]``, or one per ``True`` entry of ``positions`` (``[B, T]`` bool)."""
        if len(hidden) != 1 or hidden[0].shape[:2] != micro.input_ids.shape:
            raise RuntimeError(
                "the value head reads the input of the model's output embedding, but the forward pass called "
                f"it {len(hidden)} time(s) and not once on the whole {tuple(micro.input_ids.shape)} batch"
            )
        states = hidden[0] if positions is None else hidden[0][positions]
        return self.value_head(states.float()).squeeze(-1)

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
