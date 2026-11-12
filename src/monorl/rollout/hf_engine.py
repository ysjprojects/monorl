"""Reference :class:`~monorl.core.interfaces.RolloutEngine` on plain ``transformers``.

Slow but exact: it runs on CPU/MPS/CUDA, honours every :class:`SamplingParams` field and returns
per-token logprobs of the sampled tokens under the **temperature-scaled** policy distribution
(``log_softmax(logits / temperature)`` gathered at the sampled token), computed from the raw
logits ``generate`` saw at each step. Top-k/top-p/min-p/repetition-penalty truncations are not
folded into the logprobs, so trainer-side logprobs computed with the same temperature give
importance ratios of exactly 1 on-policy. vLLM (``processed_logprobs``) and SGLang report logprobs
*after* those processors; the orchestrator therefore recomputes ``old_logprobs`` with the trainer
for non-``hf`` engines whenever a truncating sampler knob is active.

Vision-language checkpoints (:func:`monorl.rollout.media.load_processor` finds a processor) encode
chats with image parts through ``processor.apply_chat_template`` so ``Completion.prompt_token_ids``
contains the media placeholder tokens, and ``generate`` receives ``pixel_values``/``image_grid_thw``/
``mm_token_type_ids``. Raw-id prompts, text-only chats and text-only models take the tokenizer path
unchanged.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from array import array
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation import StoppingCriteria, StoppingCriteriaList, StopStringCriteria

from monorl.config.schema import RunConfig
from monorl.core.errors import GenerationCancelled
from monorl.core.types import Completion, FinishReason, PromptInput, SamplingParams, TopLogprobs
from monorl.rollout.media import (
    auto_model_class,
    collate_media,
    load_processor,
    needs_processor,
    split_processor_outputs,
    vlm_encode,
)

log = logging.getLogger(__name__)


def resolve_device(requested: str | None) -> torch.device:
    """``cpu``/``mps``/``cuda[:i]`` as requested, else the best available accelerator."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu" or requested == "float32":
        return torch.float32
    if requested == "auto" or requested == "bfloat16":
        return torch.bfloat16
    return torch.float16


def load_tokenizer(config: RunConfig) -> Any:
    tok = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path or config.model.name_or_path,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if config.model.chat_template is not None:
        tok.chat_template = config.model.chat_template
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def chat_prompt_ids(tokenizer: Any, messages: Sequence[Any]) -> list[int]:
    """Token ids the policy conditions on for a chat prompt: chat template rendered to text, then
    tokenised **without** extra special tokens (so multi-turn callers can re-derive the same ids
    from the text and check token-exact continuation)."""
    text = tokenizer.apply_chat_template(list(messages), add_generation_prompt=True, tokenize=False)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _is_token_prompt(prompt: PromptInput) -> bool:
    return len(prompt) > 0 and isinstance(prompt[0], int)


@dataclass(slots=True)
class _EncodedPrompt:
    ids: list[int]
    token_aligned: dict[str, torch.Tensor] | None = None
    """Processor tensors aligned with ``ids`` (``[1, len(ids)]``), e.g. ``mm_token_type_ids``."""
    media: dict[str, torch.Tensor] | None = None
    """Per-prompt media tensors (``pixel_values``, ``image_grid_thw``, ...)."""

    @property
    def multimodal(self) -> bool:
        return bool(self.media) or bool(self.token_aligned)


def encode_prompt(prompt: PromptInput, tokenizer: Any, processor: Any | None) -> _EncodedPrompt:
    """Raw ids pass through; chats with media parts go through the processor (when the model has
    one), every other chat through :func:`chat_prompt_ids` — the text path is unchanged."""
    if _is_token_prompt(prompt):
        return _EncodedPrompt(ids=[int(t) for t in prompt])  # type: ignore[union-attr]
    if processor is not None and needs_processor(prompt):  # type: ignore[arg-type]
        enc = vlm_encode(processor, prompt)  # type: ignore[arg-type]
        aligned, media = split_processor_outputs(enc)
        return _EncodedPrompt(
            ids=[int(t) for t in enc["input_ids"][0].tolist()], token_aligned=aligned, media=media
        )
    return _EncodedPrompt(ids=chat_prompt_ids(tokenizer, prompt))


class _FinishRecorder(StoppingCriteria):
    """Stopping criterion that also remembers *when* and *why* each sequence finished.

    ``generate`` pads finished rows with ``pad_token_id`` (often equal to EOS), so the number of
    real tokens cannot be recovered from the output alone; this records it during the loop.
    """

    def __init__(self, batch_size: int, eos_ids: list[int], stop: StopStringCriteria | None) -> None:
        self.eos_ids = eos_ids
        self.stop = stop
        self.num_tokens = [-1] * batch_size
        self.reason: list[FinishReason | None] = [None] * batch_size
        self._step = 0
        self._eos_tensor: torch.Tensor | None = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> torch.BoolTensor:
        self._step += 1
        last = input_ids[:, -1]
        if self._eos_tensor is None or self._eos_tensor.device != last.device:
            self._eos_tensor = torch.tensor(self.eos_ids, device=last.device)
        is_eos = torch.isin(last, self._eos_tensor)
        is_stop = self.stop(input_ids, scores) if self.stop is not None else torch.zeros_like(is_eos)
        done = is_eos | is_stop
        for i in torch.nonzero(done, as_tuple=False).flatten().tolist():
            if self.reason[i] is None:
                self.reason[i] = "stop"
                self.num_tokens[i] = self._step
        return done

    def finalize(self, total_steps: int) -> None:
        for i, r in enumerate(self.reason):
            if r is None or self.num_tokens[i] > total_steps:
                self.reason[i] = "length"
                self.num_tokens[i] = total_steps


class _CancelSwitch(StoppingCriteria):
    """One per ``generate`` call: once ``cancelled`` is set, every row stops at the next decoding
    step, so ``model.generate`` unwinds through its normal exit path."""

    def __init__(self) -> None:
        self.cancelled = False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> torch.BoolTensor:
        return torch.full((input_ids.shape[0],), self.cancelled, dtype=torch.bool, device=input_ids.device)

    def check(self) -> None:
        if self.cancelled:
            raise GenerationCancelled("generation cancelled")


class HFRolloutEngine:
    """Transformers-backed sampler. One model per instance; ``generate`` is serialised by a lock
    (wrap in :class:`~monorl.rollout.BatchingEngine` to coalesce parallel episodes). ``cancel()``
    stops the running ``generate`` within one decoding step and fails the calls waiting for the
    lock; all of them raise :class:`GenerationCancelled`."""

    def __init__(self, config: RunConfig, *, replica: int = 0) -> None:
        self.config = config
        self.replica = replica
        self.device = resolve_device(config.rollout.device)
        self.dtype = resolve_dtype(config.model.dtype, self.device)
        self._processor = load_processor(config)
        self._tokenizer = self._processor.tokenizer if self._processor is not None else load_tokenizer(config)
        self.model = auto_model_class(config).from_pretrained(
            config.model.name_or_path,
            dtype=self.dtype,
            revision=config.model.revision,
            trust_remote_code=config.model.trust_remote_code,
            attn_implementation=config.model.attn_implementation,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self._init_call_state()
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self._tokenizer.eos_token_id
        self._eos_ids: list[int] = [eos] if isinstance(eos, int) else list(eos)
        self._pad_id: int = self._tokenizer.pad_token_id

    def _init_call_state(self) -> None:
        """Locks and per-call bookkeeping ``generate``/``cancel``/``sleep`` rely on. Subclasses that
        build the engine around an existing model without running ``__init__`` call this instead."""
        self._lock = threading.Lock()
        self._calls: set[_CancelSwitch] = set()
        """Switches of the ``generate`` calls that have not returned yet (running or waiting)."""
        self._calls_lock = threading.Lock()
        self._asleep = False

    # -- RolloutEngine --------------------------------------------------------------------
    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def processor(self) -> Any | None:
        """``AutoProcessor`` for VLM checkpoints (shared with the batch builder), else ``None``."""
        return self._processor

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        if not prompts:
            return []
        if params.n < 1:
            raise ValueError("SamplingParams.n must be >= 1")
        switch = _CancelSwitch()
        with self._calls_lock:
            self._calls.add(switch)
        try:
            encoded = [encode_prompt(p, self._tokenizer, self._processor) for p in prompts]
            with self._lock:
                if self._asleep:
                    raise RuntimeError("engine is asleep; call wake_up() before generate()")
                switch.check()  # cancelled while encoding or waiting for another generate
                with self._seeded(params.seed):
                    return self._generate_locked(encoded, params, switch)
        finally:
            with self._calls_lock:
                self._calls.discard(switch)

    def cancel(self) -> None:
        """Abort the ``generate`` calls in flight (they raise :class:`GenerationCancelled`); a
        no-op when idle. The engine stays usable: later calls run normally."""
        with self._calls_lock:
            for switch in self._calls:
                switch.cancelled = True

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        with self._lock, torch.no_grad():
            state = self.model.state_dict()
            for name, tensor in weights:
                target = state.get(name)
                if target is None:
                    raise KeyError(f"update_weights: unknown parameter {name!r}")
                if tuple(target.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"update_weights: shape mismatch for {name!r}: {tuple(tensor.shape)} vs {tuple(target.shape)}"
                    )
                target.copy_(tensor.to(device=target.device, dtype=target.dtype, non_blocking=True))

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        self.update_weights(iter_hf_weights(path))

    def sleep(self) -> None:
        with self._lock:
            if self.device.type != "cuda" or self._asleep:
                return
            self.model.to("cpu")
            torch.cuda.empty_cache()
            self._asleep = True

    def wake_up(self) -> None:
        with self._lock:
            if not self._asleep:
                return
            self.model.to(self.device)
            self._asleep = False

    def shutdown(self) -> None:
        with self._lock:
            model = self.__dict__.pop("model", None)
            if model is not None:
                model.to("cpu")
                del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    # -- internals ------------------------------------------------------------------------
    @contextlib.contextmanager
    def _seeded(self, seed: int | None) -> Iterator[None]:
        if seed is None:
            yield
            return
        if self.device.type == "cuda":
            fork = torch.random.fork_rng(devices=[self.device.index or 0])
        elif self.device.type == "mps":
            fork = torch.random.fork_rng(device_type="mps")
        else:
            fork = torch.random.fork_rng(devices=[])
        with fork:
            torch.manual_seed(seed)
            yield

    def _generation_config(self, params: SamplingParams, n: int) -> GenerationConfig:
        do_sample = params.temperature > 0.0
        gc = GenerationConfig(
            max_new_tokens=params.max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=n,
            pad_token_id=self._pad_id,
            eos_token_id=self._eos_ids,
            repetition_penalty=params.repetition_penalty,
            return_dict_in_generate=True,
            output_logits=params.logprobs or params.top_logprobs > 0,
            use_cache=True,
        )
        if do_sample:
            gc.temperature = params.temperature
            gc.top_p = params.top_p
            # 0 (not None) disables top-k: transformers backfills None from the model's
            # generation_config.json or its global default of 50, which would truncate sampling
            # while the returned logprobs stay full-softmax.
            gc.top_k = params.top_k if params.top_k > 0 else 0
            gc.min_p = params.min_p
        return gc

    def _left_pad(self, prompt_ids: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(p) for p in prompt_ids)
        ids = torch.full((len(prompt_ids), width), self._pad_id, dtype=torch.long)
        mask = torch.zeros((len(prompt_ids), width), dtype=torch.long)
        for i, p in enumerate(prompt_ids):
            if not p:
                raise ValueError("empty prompt")
            ids[i, width - len(p) :] = torch.tensor(p, dtype=torch.long)
            mask[i, width - len(p) :] = 1
        return ids.to(self.device), mask.to(self.device)

    def _multimodal_inputs(
        self, encoded: list[_EncodedPrompt], n: int, width: int
    ) -> dict[str, torch.Tensor]:
        """Processor tensors for ``len(encoded) * n`` rows (row ``i*n+j`` = prompt ``i``): token-aligned
        tensors left-padded with 0 to ``width``; media tensors concatenated in row order. Rows are
        expanded here rather than by ``num_return_sequences`` because ``generate`` would
        ``repeat_interleave`` patch tensors such as Qwen2-VL's ``pixel_values [patches, D]`` along
        dim 0, interleaving patches of different copies."""
        rows = len(encoded) * n
        aligned: dict[str, torch.Tensor] = {}
        for i, e in enumerate(encoded):
            for k, v in (e.token_aligned or {}).items():
                if k not in aligned:
                    aligned[k] = torch.zeros((rows, width), dtype=v.dtype)
                aligned[k][i * n : (i + 1) * n, width - v.shape[1] :] = v[0]
        media = collate_media([e.media for e in encoded for _ in range(n)])
        return {k: v.to(self.device) for k, v in (aligned | media).items()}

    @torch.no_grad()
    def _generate_locked(
        self, encoded: list[_EncodedPrompt], params: SamplingParams, switch: _CancelSwitch
    ) -> list[list[Completion]]:
        greedy = params.temperature <= 0.0
        n = 1 if greedy else params.n
        prompt_ids = [e.ids for e in encoded]
        input_ids, attention_mask = self._left_pad(prompt_ids)
        rows = input_ids.shape[0] * n
        stop = StopStringCriteria(self._tokenizer, list(params.stop)) if params.stop else None
        recorder = _FinishRecorder(rows, self._eos_ids, stop)
        model_kwargs: dict[str, Any] = {}
        expand = 1
        if any(e.multimodal for e in encoded):
            model_kwargs = self._multimodal_inputs(encoded, n, input_ids.shape[1])
            input_ids = input_ids.repeat_interleave(n, dim=0)
            attention_mask = attention_mask.repeat_interleave(n, dim=0)
            expand = n
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self._generation_config(params, n // expand),
            stopping_criteria=StoppingCriteriaList([recorder, switch]),
            **model_kwargs,
        )
        switch.check()  # the sequences were cut short, not finished
        prompt_width = input_ids.shape[1]
        generated = out.sequences[:, prompt_width:]
        total_steps = generated.shape[1]
        recorder.finalize(total_steps)

        logprobs: torch.Tensor | None = None
        top: tuple[torch.Tensor, torch.Tensor] | None = None  # ([rows, steps, k] ids, logprobs)
        if params.logprobs or params.top_logprobs > 0:
            temperature = params.temperature if not greedy else 1.0
            if params.logprobs:
                logprobs = torch.empty((rows, total_steps), dtype=torch.float32)
            for t, step_logits in enumerate(out.logits[:total_steps]):
                step_logprobs = torch.log_softmax(step_logits.float() / temperature, dim=-1)
                if logprobs is not None:
                    logprobs[:, t] = step_logprobs.gather(1, generated[:, t : t + 1]).squeeze(1).cpu()
                if params.top_logprobs > 0:
                    values, indices = step_logprobs.topk(min(params.top_logprobs, step_logprobs.shape[-1]))
                    if top is None:
                        top = (
                            torch.empty((rows, total_steps, indices.shape[-1]), dtype=torch.int32),
                            torch.empty((rows, total_steps, indices.shape[-1]), dtype=torch.float32),
                        )
                    top[0][:, t] = indices.cpu()
                    top[1][:, t] = values.cpu()
            if params.top_logprobs > 0 and top is None:  # nothing was generated
                top = (torch.empty((rows, 0, 0), dtype=torch.int32), torch.empty((rows, 0, 0)))
        generated = generated.cpu()

        results: list[list[Completion]] = []
        for i, pids in enumerate(prompt_ids):
            group: list[Completion] = []
            for j in range(n):
                row = i * n + j
                k = recorder.num_tokens[row]
                ids = generated[row, :k].tolist()
                group.append(
                    Completion(
                        text=self._tokenizer.decode(ids, skip_special_tokens=True),
                        token_ids=ids,
                        logprobs=logprobs[row, :k].tolist() if logprobs is not None else None,
                        finish_reason=recorder.reason[row] or "length",
                        prompt_token_ids=list(pids),
                        top_logprobs=None
                        if top is None
                        else TopLogprobs(
                            k=top[0].shape[-1],
                            token_ids=array("i", top[0][row, :k].reshape(-1).tolist()),
                            logprobs=array("f", top[1][row, :k].reshape(-1).tolist()),
                        ),
                    )
                )
            if greedy and params.n > 1:
                base = group[0]
                top_logprobs = base.top_logprobs
                group = [
                    Completion(
                        text=base.text,
                        token_ids=list(base.token_ids),
                        logprobs=None if base.logprobs is None else list(base.logprobs),
                        finish_reason=base.finish_reason,
                        prompt_token_ids=list(pids),
                        top_logprobs=None
                        if top_logprobs is None
                        else TopLogprobs(top_logprobs.k, top_logprobs.token_ids[:], top_logprobs.logprobs[:]),
                    )
                    for _ in range(params.n)
                ]
            results.append(group)
        return results


def iter_hf_weights(path: str | os.PathLike[str]) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream ``(name, tensor)`` from an HF-format directory (safetensors, sharded or not) on CPU."""
    from safetensors import safe_open

    root = Path(path)
    index = root / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (root / "model.safetensors").exists():
        shards = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"no model.safetensors(.index.json) under {root}")
    for shard in shards:
        with safe_open(str(root / shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)
