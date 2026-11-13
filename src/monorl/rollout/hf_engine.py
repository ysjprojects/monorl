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

With ``model.mtp.rollout`` the checkpoint's MTP layers (:class:`monorl.mtp.MtpLayers`) draft the
tokens after the next one and the policy verifies every draft (transformers' assisted decoding):
tokens and logprobs are distributed and recorded exactly as without drafting, one policy pass just
emits several of them. transformers drafts for one sequence per call, so the rows are decoded one
after another, and only for text prompts.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation import StoppingCriteria, StoppingCriteriaList, StopStringCriteria

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, GenerationCancelled
from monorl.core.types import (
    Completion,
    FinishReason,
    PromptInput,
    SamplingParams,
    SpeculationStats,
    TopLogprobs,
)
from monorl.mtp import (
    WIRE_PREFIX,
    MtpLayers,
    checkpoint_key,
    checkpoint_shards,
    is_wire_name,
    mtp_key_pattern,
    resolve_checkpoint_dir,
)
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


class _DraftFinishRecorder(StoppingCriteria):
    """:class:`_FinishRecorder` for assisted decoding of one sequence, where a decoding step appends
    several tokens (the accepted drafts and the policy's own) and the criteria run once per step.

    Plain decoding looks at the sequence after every token, so every prefix a step added is checked
    in order: the first that ends with EOS or a stop string is where the completion ends, and the
    engine drops what the step emitted after it."""

    def __init__(self, prompt_length: int, eos_ids: list[int], stop: StopStringCriteria | None) -> None:
        self.eos_ids = frozenset(eos_ids)
        self.stop = stop
        self.num_tokens: int | None = None
        """Generated tokens up to and including the one that finished the sequence."""
        self._prompt_length = prompt_length
        self._checked = prompt_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> torch.BoolTensor:
        if self.num_tokens is None:
            added = input_ids[0, self._checked :].tolist()
            for end, token in enumerate(added, start=self._checked + 1):
                if token in self.eos_ids or (
                    self.stop is not None and bool(self.stop(input_ids[:, :end], scores)[0])
                ):
                    self.num_tokens = end - self._prompt_length
                    break
            self._checked = input_ids.shape[1]
        finished = self.num_tokens is not None
        return torch.full((input_ids.shape[0],), finished, dtype=torch.bool, device=input_ids.device)


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

    mtp_layers: MtpLayers | None = None
    """The model's MTP layers when the engine drafts with them (``model.mtp.rollout``)."""

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
        if config.model.mtp.enabled and config.model.mtp.rollout:
            self.mtp_layers = self._load_mtp_layers(config)

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
            if self.mtp_layers is not None and any(e.multimodal for e in encoded):
                raise ValueError(
                    "model.mtp.rollout on the hf engine drafts for text prompts only; prompts with "
                    "images or other media need model.mtp.rollout: false"
                )
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
        """Copy ``weights`` into the policy. Names tagged as weights of the MTP layers
        (:data:`monorl.mtp.WIRE_PREFIX`) replace the drafter's weights once the policy's are in
        (all of them or none: see :meth:`MtpLayers.load_checkpoint_weights`); an engine that does
        not draft drops them."""
        with self._lock, torch.no_grad():
            state = self.model.state_dict()
            layers = self.mtp_layers
            drafter: dict[str, torch.Tensor] = {}
            for name, tensor in weights:
                if is_wire_name(name):
                    if layers is not None:
                        drafter[checkpoint_key(name)] = tensor
                    continue
                target = state.get(name)
                if target is None:
                    raise KeyError(f"update_weights: unknown parameter {name!r}")
                if tuple(target.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"update_weights: shape mismatch for {name!r}: {tuple(tensor.shape)} vs {tuple(target.shape)}"
                    )
                target.copy_(tensor.to(device=target.device, dtype=target.dtype, non_blocking=True))
            if layers is not None and drafter:
                layers.load_checkpoint_weights(drafter)
                if self._asleep:  # loaded onto the layers' device; they rest with the policy
                    layers.model.to("cpu")

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        """:meth:`update_weights` from an HF-format directory (``save_pretrained``, with the MTP
        layers when :func:`monorl.mtp.write_checkpoint_weights` added them)."""
        self.update_weights(self._directory_weights(path))

    def speculation_stats(self) -> SpeculationStats | None:
        """Drafting counters since the previous call; ``None`` when the engine does not draft."""
        return self.mtp_layers.take_stats() if self.mtp_layers is not None else None

    def sleep(self) -> None:
        with self._lock:
            if self.device.type != "cuda" or self._asleep:
                return
            self.model.to("cpu")
            if self.mtp_layers is not None:
                self.mtp_layers.model.to("cpu")
            torch.cuda.empty_cache()
            self._asleep = True

    def wake_up(self) -> None:
        with self._lock:
            if not self._asleep:
                return
            self.model.to(self.device)
            if self.mtp_layers is not None:
                self.mtp_layers.model.to(self.mtp_layers.device)
            self._asleep = False

    def shutdown(self) -> None:
        with self._lock:
            model = self.__dict__.pop("model", None)
            layers, self.mtp_layers = self.mtp_layers, None
            if layers is not None:
                layers.model.to("cpu")
                del layers
            if model is not None:
                model.to("cpu")
                del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    # -- internals ------------------------------------------------------------------------
    def _load_mtp_layers(self, config: RunConfig) -> MtpLayers:
        """The checkpoint's MTP layers, frozen and installed as the drafter of ``self.model``."""
        layers = MtpLayers(
            self.model, resolve_checkpoint_dir(config.model.name_or_path, config.model.revision)
        )
        wanted = config.model.mtp.num_speculative_tokens
        if wanted is not None and wanted != layers.num_layers:
            raise ConfigError(
                f"model.mtp.num_speculative_tokens={wanted} is not available with rollout.backend='hf': it "
                f"drafts one token per MTP layer and {config.model.name_or_path} has {layers.num_layers}; "
                "leave it unset"
            )
        layers.model.eval()
        layers.model.requires_grad_(False)
        layers.install_drafting()
        return layers

    def _directory_weights(self, path: str | os.PathLike[str]) -> Iterator[tuple[str, torch.Tensor]]:
        """The weights of an HF-format directory as a weight stream. Its keys are the checkpoint's
        own. Those of the model's MTP layers are found by the model's key pattern and tagged, or
        left unread when the engine does not draft. The policy's are its ``state_dict`` names for
        most models and streamed tensor by tensor; some models are stored in another layout that
        transformers converts when it loads them (Mellum: one tensor per expert, packed in the
        module), and such a directory is read by loading it as a second model on CPU."""
        pattern = mtp_key_pattern(self.model)
        known = self.model.state_dict().keys()
        drafting = self.mtp_layers is not None

        def is_mtp(name: str) -> bool:
            return pattern is not None and pattern.search(name) is not None

        as_stored = all(name in known for name in hf_weight_names(path) if not is_mtp(name))
        if not as_stored:
            policy, info = type(self.model).from_pretrained(path, dtype=self.dtype, output_loading_info=True)
            problems = {
                kind: sorted(map(str, info[kind]))[:3]
                for kind in ("missing_keys", "unexpected_keys", "mismatched_keys")
                if info[kind]
            }
            if problems:
                raise KeyError(
                    f"load_weights_from_path: {os.fspath(path)} does not hold the weights of "
                    f"{type(self.model).__name__}: {problems}"
                )
            yield from policy.state_dict().items()
        for name, tensor in iter_hf_weights(path, lambda name: drafting if is_mtp(name) else as_stored):
            yield (WIRE_PREFIX + name if is_mtp(name) else name), tensor

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
            use_mtp=self.mtp_layers is not None,
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
        decode = self._decode_batch if self.mtp_layers is None else self._decode_drafting
        rows = decode(encoded, params, n, switch)  # row i * n + j = sample j of prompt i
        results: list[list[Completion]] = []
        for i in range(len(encoded)):
            group = rows[i * n : (i + 1) * n]
            if greedy and params.n > 1:
                base = group[0]
                top_logprobs = base.top_logprobs
                group = [
                    Completion(
                        text=base.text,
                        token_ids=list(base.token_ids),
                        logprobs=None if base.logprobs is None else list(base.logprobs),
                        finish_reason=base.finish_reason,
                        prompt_token_ids=list(base.prompt_token_ids),
                        top_logprobs=None
                        if top_logprobs is None
                        else TopLogprobs(top_logprobs.k, top_logprobs.token_ids[:], top_logprobs.logprobs[:]),
                    )
                    for _ in range(params.n)
                ]
            results.append(group)
        return results

    def _decode_batch(
        self, encoded: list[_EncodedPrompt], params: SamplingParams, n: int, switch: _CancelSwitch
    ) -> list[Completion]:
        """``n`` completions of every prompt, all rows in one left-padded ``generate`` call."""
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
        generated = out.sequences[:, input_ids.shape[1] :]
        recorder.finalize(generated.shape[1])
        return self._completions(
            [ids for ids in prompt_ids for _ in range(n)],
            generated,
            recorder.num_tokens,
            recorder.reason,
            out.logits,
            params,
        )

    def _decode_drafting(
        self, encoded: list[_EncodedPrompt], params: SamplingParams, n: int, switch: _CancelSwitch
    ) -> list[Completion]:
        """:meth:`_decode_batch` with the MTP layers drafting. Assisted decoding takes one sequence
        per call, so the rows are decoded one after another; all of them draw from the one random
        stream the caller seeded, each from where the previous row left it."""
        layers = self.mtp_layers
        if layers is None:
            raise RuntimeError("the engine holds no MTP layers to draft with")
        generation_config = self._generation_config(params, 1)
        stop = StopStringCriteria(self._tokenizer, list(params.stop)) if params.stop else None
        completions: list[Completion] = []
        for e in encoded:
            if not e.ids:
                raise ValueError("empty prompt")
            input_ids = torch.tensor([e.ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)
            for _ in range(n):
                recorder = _DraftFinishRecorder(len(e.ids), self._eos_ids, stop)
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    stopping_criteria=StoppingCriteriaList([recorder, switch]),
                )
                switch.check()  # the sequence was cut short, not finished
                generated = out.sequences[:, len(e.ids) :]
                reason: FinishReason = "length"
                if recorder.num_tokens is not None:
                    # The step that finished the sequence may have emitted tokens after its end.
                    layers.drop_emitted(generated.shape[1] - recorder.num_tokens)
                    generated = generated[:, : recorder.num_tokens]
                    reason = "stop"
                completions += self._completions(
                    [e.ids], generated, [generated.shape[1]], [reason], out.logits, params
                )
        return completions

    def _completions(
        self,
        prompt_ids: Sequence[list[int]],
        generated: torch.Tensor,
        num_tokens: Sequence[int],
        reasons: Sequence[FinishReason | None],
        logits: Sequence[torch.Tensor],
        params: SamplingParams,
    ) -> list[Completion]:
        """One :class:`Completion` per row of ``generated`` ``[rows, steps]``: the row's first
        ``num_tokens[row]`` tokens as the continuation of ``prompt_ids[row]``. ``logits[t]``
        ``[rows, vocab]`` are the policy's raw logits the token at position ``t`` was chosen from
        (read only when ``params`` ask for logprobs)."""
        rows, total_steps = generated.shape
        logprobs: torch.Tensor | None = None
        top: tuple[torch.Tensor, torch.Tensor] | None = None  # ([rows, steps, k] ids, logprobs)
        if params.logprobs or params.top_logprobs > 0:
            temperature = params.temperature if params.temperature > 0.0 else 1.0
            if params.logprobs:
                logprobs = torch.empty((rows, total_steps), dtype=torch.float32)
            for t, step_logits in enumerate(logits[:total_steps]):
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

        completions: list[Completion] = []
        for row, pids in enumerate(prompt_ids):
            k = num_tokens[row]
            ids = generated[row, :k].tolist()
            completions.append(
                Completion(
                    text=self._tokenizer.decode(ids, skip_special_tokens=True),
                    token_ids=ids,
                    logprobs=logprobs[row, :k].tolist() if logprobs is not None else None,
                    finish_reason=reasons[row] or "length",
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
        return completions


def iter_hf_weights(
    path: str | os.PathLike[str], wanted: Callable[[str], bool] | None = None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream ``(name, tensor)`` from an HF-format directory (safetensors, sharded or not) on CPU,
    under the names it stores; only the names ``wanted`` accepts are read."""
    from safetensors import safe_open

    for shard in checkpoint_shards(path):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                if wanted is None or wanted(name):
                    yield name, f.get_tensor(name)


def hf_weight_names(path: str | os.PathLike[str]) -> list[str]:
    """The names :func:`iter_hf_weights` streams, without reading a tensor."""
    from safetensors import safe_open

    names: list[str] = []
    for shard in checkpoint_shards(path):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            names.extend(f.keys())
    return names
