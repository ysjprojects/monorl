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

Rows of a batch that have the same prompt (the ``n`` samples of a prompt, the equal prompts of
episodes that were batched together) compute it once and start from copies of its key-value cache
(:meth:`HFRolloutEngine._shared_prompt_cache`).

With ``model.mtp.rollout`` the checkpoint's MTP layers (:class:`monorl.mtp.MtpLayers`) draft the
tokens after the next one and the policy verifies every draft (transformers' assisted decoding):
tokens and logprobs are distributed and recorded exactly as without drafting, one policy pass just
emits several of them. transformers drafts for one sequence per call, so the rows are decoded one
after another, and only for text prompts.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import os
import threading
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer, DynamicCache, GenerationConfig
from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer
from transformers.generation import (
    GenerationMixin,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, GenerationCancelled
from monorl.core.types import (
    Completion,
    FinishReason,
    PromptInput,
    SamplingParams,
    SpeculationStats,
    TopLogprobs,
    prompt_caps,
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


class _LogprobRecorder(LogitsProcessor):
    """The log-probability of every token a ``generate`` call samples, and of the ``top`` most
    likely tokens at its position, under ``log_softmax(raw logits / temperature)``.

    ``generate`` can hand back the raw logits of every step (``output_logits``), but then it holds
    ``steps x rows x vocab`` floats until it returns. This keeps one step. :meth:`observe`, a
    forward hook on the model, sees a step's logits before any logits processor has touched them;
    the token that was chosen from them is the last one of ``input_ids`` when ``generate`` calls
    its logits processors for the following step, which is all this processor is there for.
    """

    def __init__(self, temperature: float, *, logprobs: bool, top: int) -> None:
        self.temperature = temperature
        self.top = top
        self._ahead: torch.Tensor | None = None  # log-softmax of the step whose token is not drawn yet
        self._drawn: torch.Tensor | None = None  # ... of the step whose token is drawn but not read
        self._logprobs: list[torch.Tensor] | None = [] if logprobs else None
        self._top_ids: list[torch.Tensor] = []
        self._top_logprobs: list[torch.Tensor] = []

    def observe(self, module: torch.nn.Module, args: Any, output: Any) -> None:
        # a new tensor, not a view: the logits of the first pass span the whole prompt
        self._ahead = torch.log_softmax(output.logits[:, -1].float() / self.temperature, dim=-1)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._read(input_ids[:, -1])
        step, self._ahead = self._ahead, None
        if step is None:
            raise RuntimeError("generate asked for a token without a forward pass of the model")
        if self.top > 0:
            values, indices = step.topk(min(self.top, step.shape[-1]))
            self._top_ids.append(indices.to("cpu", torch.int32))
            self._top_logprobs.append(values.cpu())
        if self._logprobs is not None:
            self._drawn = step
        return scores

    def _read(self, tokens: torch.Tensor) -> None:
        if self._drawn is not None and self._logprobs is not None:
            self._logprobs.append(self._drawn.gather(1, tokens[:, None]).squeeze(1).cpu())
            self._drawn = None

    def result(
        self, generated: torch.Tensor
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        """``[rows, steps]`` log-probabilities of ``generated`` (``None`` unless asked for) and the
        ``([rows, steps, k] token ids, log-probabilities)`` of the top tokens (``None`` unless
        ``top > 0``). ``generate`` may have run a step more than it returned (on MPS it learns a
        step late that every row has finished, and takes that step back): it is left out."""
        rows, steps = generated.shape
        if steps:
            self._read(generated[:, -1])
        logprobs = None
        if self._logprobs is not None:
            logprobs = torch.stack(self._logprobs[:steps], dim=1) if steps else torch.empty((rows, 0))
        top = None
        if self.top > 0:
            if steps:
                top = (
                    torch.stack(self._top_ids[:steps], dim=1),
                    torch.stack(self._top_logprobs[:steps], dim=1),
                )
            else:  # nothing was generated
                top = (torch.empty((rows, 0, 0), dtype=torch.int32), torch.empty((rows, 0, 0)))
        return logprobs, top


_ROW_LAYERS = (DynamicLayer, DynamicSlidingWindowLayer)
"""Cache layers whose whole state per row is the keys and values ``batch_select_indices`` selects."""


def _can_drop_rows(cache: Any) -> bool:
    """Whether ``generate`` can go on from ``cache`` with some of its rows taken out: the cache
    ``generate`` builds by default, with layers of :data:`_ROW_LAYERS`. Other caches (static,
    quantized, a model's own state) and layers with more state per row (linear attention, an
    index) stay whole."""
    return type(cache) is DynamicCache and all(type(layer) in _ROW_LAYERS for layer in cache.layers)


def _select_cache_rows(
    cache: DynamicCache,
    rows: Sequence[int],
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select live rows from a cache accepted by :func:`_can_drop_rows` and its inputs.

    Reuse one long index tensor per current storage device. Passing host indices to every
    accelerator K/V selection would promote the same indices separately for every tensor.
    Layer methods still own the cache selection and its bookkeeping.
    """
    indices: dict[torch.device, torch.Tensor] = {}

    def on_device(device: torch.device) -> torch.Tensor:
        index = indices.get(device)
        if index is None:
            index = torch.tensor(rows, dtype=torch.long, device=device)
            indices[device] = index
        return index

    for layer in cache.layers:
        if layer.get_seq_length() > 0:
            layer.batch_select_indices(on_device(layer.keys.device))
    return (
        sequences.index_select(0, on_device(sequences.device)),
        attention_mask.index_select(0, on_device(attention_mask.device)),
    )


class _FinishRecorder(StoppingCriteria):
    """Stopping criterion that remembers *when* each row finished, and that can end a ``generate``
    call before all of them have.

    ``generate`` pads finished rows with ``pad_token_id`` (often equal to EOS), so the number of
    real tokens cannot be recovered from the output alone; this records it during the loop.

    A row also finishes at a cap of its own: ``caps[row]`` is the most tokens the call may generate
    for it, where ``generate`` itself knows one ``max_new_tokens`` for all its rows. Such a row is
    not in ``stopped``, which holds the rows that ended with EOS or a stop string.

    ``generate`` also keeps computing every row until the last one has finished. With ``slack`` set
    the recorder reports all rows as finished as soon as the next forward pass would bring the
    finished rows the call has computed above ``slack`` (0: as soon as a row finishes), which makes
    ``generate`` return: ``cut`` then says that the call ended this way, and the rows without an
    entry in ``num_tokens`` are the ones to continue in a call of their own.
    """

    def __init__(
        self,
        caps: Sequence[int],
        eos_ids: list[int],
        stop: StopStringCriteria | None,
        *,
        slack: int | None = None,
    ) -> None:
        rows = len(caps)
        self.caps = torch.tensor(caps)
        self.eos_ids = eos_ids
        self.stop = stop
        self.slack = slack
        self.num_tokens: list[int | None] = [None] * rows
        """Tokens the call had generated when the row finished; ``None`` for a row that has not."""
        self.stopped: set[int] = set()
        """The finished rows that ended with EOS or a stop string, not at their cap."""
        self.cut = False
        self._unfinished = rows
        self._idle = 0  # finished rows in the forward passes of the call, the next one included
        self._step = 0
        self._closed = False
        self._eos_tensor: torch.Tensor | None = None

    def check_cache(self, slack: int, module: torch.nn.Module, args: Any, output: Any) -> None:
        """Forward hook on the model (``slack`` bound) for a call that builds its cache: the call
        gets that slack once a forward pass has left a cache its rows can be taken out of."""
        if _can_drop_rows(output.get("past_key_values")):
            self.slack = slack

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> torch.BoolTensor:
        if self._closed:
            # generate may run one more step before it reads that every row has finished (see
            # _LogprobRecorder.result); its tokens are not part of the output
            return torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        self._step += 1
        last = input_ids[:, -1]
        if self._eos_tensor is None or self._eos_tensor.device != last.device:
            self._eos_tensor = torch.tensor(self.eos_ids, device=last.device)
            self.caps = self.caps.to(last.device)
        is_eos = torch.isin(last, self._eos_tensor)
        is_stop = self.stop(input_ids, scores) if self.stop is not None else torch.zeros_like(is_eos)
        ended = is_eos | is_stop
        done = ended | (self.caps <= self._step)
        for i in torch.nonzero(done, as_tuple=False).flatten().tolist():
            if self.num_tokens[i] is None:
                self.num_tokens[i] = self._step
                self._unfinished -= 1
                if ended[i]:
                    self.stopped.add(i)
        self._idle += len(self.num_tokens) - self._unfinished
        if self._unfinished and (self.slack is None or self._idle <= self.slack):
            return done
        self._closed = True
        self.cut = self._unfinished > 0
        return torch.ones_like(done)


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

    accepts_partial_weights = True
    """``update_weights`` copies into the tensors it is given and leaves the others, and the model
    keeps its weights while the engine sleeps: weight sync may leave out what did not change."""

    shared_prefill_min_positions = 256
    """The fewest positions (rows times padded prompt tokens) that computing every distinct prompt
    of a batch once has to save before the engine does it (:meth:`_shared_prompt_cache`); below
    that the one forward pass more costs about what it saves."""

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

    def generate(
        self,
        prompts: Sequence[PromptInput],
        params: SamplingParams,
        *,
        max_new_tokens: Sequence[int] | None = None,
    ) -> list[list[Completion]]:
        if not prompts:
            return []
        if params.n < 1:
            raise ValueError("SamplingParams.n must be >= 1")
        caps = prompt_caps(params, len(prompts), max_new_tokens)
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
                    return self._generate_locked(encoded, params, caps, switch)
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

    def _generation_config(
        self, params: SamplingParams, max_new_tokens: int, *, output_logits: bool
    ) -> GenerationConfig:
        do_sample = params.temperature > 0.0
        gc = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._pad_id,
            eos_token_id=self._eos_ids,
            repetition_penalty=params.repetition_penalty,
            return_dict_in_generate=True,
            output_logits=output_logits,
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
        tensors left-padded with 0 to ``width``; media tensors concatenated in row order (``generate``
        cannot make the rows with ``num_return_sequences``: it would ``repeat_interleave`` patch
        tensors such as Qwen2-VL's ``pixel_values [patches, D]`` along dim 0, interleaving patches
        of different copies)."""
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
        self, encoded: list[_EncodedPrompt], params: SamplingParams, caps: list[int], switch: _CancelSwitch
    ) -> list[list[Completion]]:
        greedy = params.temperature <= 0.0
        n = 1 if greedy else params.n
        decode = self._decode_batch if self.mtp_layers is None else self._decode_drafting
        rows = decode(encoded, params, caps, n, switch)  # row i * n + j = sample j of prompt i
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

    def _generating_model(self) -> torch.nn.Module:
        """The module whose forward pass ``generate`` runs, which is where a forward hook sees the
        logits of a decoding step: ``self.model``, or the transformers model inside a wrapper that
        hands ``generate`` on to it (a PEFT model, whose own ``forward`` a ``generate`` call never
        goes through)."""
        return next(m for m in self.model.modules() if isinstance(m, GenerationMixin))

    def _slack(self, rows: int) -> int | None:
        """How many finished rows the forward passes of a ``generate`` call over ``rows`` rows may
        compute before the call is ended to go on without them (:class:`_FinishRecorder`); ``None``
        when it has to run to the end of all its rows.

        Ending a call costs no forward pass, so it ends when a row finishes. Except on MPS, where
        transformers reads the stopping criteria a step late: a call that is ended runs one more
        step with all its rows and takes it back, so the finished rows stay until they have cost as
        much as that. Checkpoints with a processor are decoded in one call: some keep state per row
        between the steps of a ``generate`` call (Qwen2-VL the offsets of its position ids) that
        would not match the rows of the next. So are models whose cache cannot give up rows
        (:func:`_can_drop_rows`, which the recorder checks at the first step)."""
        if self._processor is not None:
            return None
        return rows if self.device.type == "mps" else 0

    def _prefill_cache(self, model: torch.nn.Module, forward: Iterable[str]) -> DynamicCache | None:
        """An empty cache to prefill prompts into once per distinct prompt
        (:meth:`_shared_prompt_cache`), or ``None`` when every row has to compute its prompt: a
        forward pass over a prompt without its last token has to leave what ``generate`` computes
        for those tokens itself, in a cache whose rows can be copied. That is the case for a text
        model of transformers' own code in evaluation mode (dropout would be drawn once per prompt
        instead of once per row), whose default cache is a ``DynamicCache`` of plain layers (full
        attention in every layer: no sliding window, no chunks, no state besides keys and values,
        no layers that share theirs), and whose positions do not depend on how long the sequence
        of a forward pass is (the dynamic RoPE scalings do). ``model`` is the transformers model
        that generates and ``forward`` the names of the parameters of its forward pass."""
        text_config = model.config.get_text_config(decoder=True)
        rope = getattr(text_config, "rope_parameters", None) or vars(text_config).get("rope_scaling") or {}
        cache = DynamicCache(config=model.config)  # the one generate builds itself
        shared = (
            self._processor is None  # such checkpoints keep state per row between steps (_slack)
            and not self.model.training
            and not type(model).__module__.startswith("transformers_modules")  # trust_remote_code
            # an adapter of virtual tokens: its wrapper puts them in front of what generate is given
            and not getattr(getattr(self.model, "active_peft_config", None), "is_prompt_learning", False)
            and model._supports_default_dynamic_cache()
            and model.generation_config.cache_implementation is None
            and {"attention_mask", "past_key_values", "logits_to_keep"} <= set(forward)
            # no RoPE, or the plain one (a mapping per layer type has no type of its own)
            and (not rope or rope.get("rope_type", rope.get("type")) == "default")
            and bool(cache.layers)
            and all(type(layer) is DynamicLayer for layer in cache.layers)
        )
        return cache if shared else None

    def _shared_prompt_cache(
        self,
        prompt_ids: list[list[int]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n: int,
        switch: _CancelSwitch,
    ) -> DynamicCache | None:
        """The key-value cache of every row of the batch (row ``i * n + j``: sample ``j`` of prompt
        ``i``) for its prompt without the last token, when rows share prompts: each distinct
        prompt is computed once and its cache rows are copied to the rows that have it, the ``n``
        samples of a prompt and the equal prompts of other episodes that were batched with it.
        ``generate`` then starts at the last prompt token of every row, where the rows first
        differ, so sampling, stopping and the recorded log-probabilities are what they are without
        the cache. ``None`` when the batch is decoded the ordinary way: that would save fewer than
        :attr:`shared_prefill_min_positions` positions (none when no prompt repeats), or the
        model's prompts cannot be computed apart from their last token (:meth:`_prefill_cache`).

        ``input_ids`` and ``attention_mask`` are the left-padded prompts, one row per prompt. The
        cache lives for this call only: no other call and no later weights ever see it."""
        distinct: dict[tuple[int, ...], int] = {}  # prompt -> its index among the distinct ones
        first_rows: list[int] = []  # the row of input_ids that holds each distinct prompt
        sources: list[int] = []  # per row of the batch, the distinct prompt it has
        for row, ids in enumerate(prompt_ids):
            index = distinct.setdefault(tuple(ids), len(first_rows))
            if index == len(first_rows):
                first_rows.append(row)
            sources.extend([index] * n)
        # The rows that are copies do not compute what precedes the last token of their prompt.
        saved = (len(sources) - len(first_rows)) * (input_ids.shape[1] - 1)
        if saved < max(self.shared_prefill_min_positions, 1):
            return None
        model = self._generating_model()
        forward = inspect.signature(model.forward).parameters
        cache = self._prefill_cache(model, forward)
        if cache is None:
            return None
        select = torch.tensor(first_rows, device=input_ids.device)
        prefix_ids = input_ids.index_select(0, select)[:, :-1]
        prefix_mask = attention_mask.index_select(0, select)[:, :-1]
        inputs: dict[str, Any] = {"input_ids": prefix_ids, "attention_mask": prefix_mask}
        if "position_ids" in forward:
            # As generate numbers the tokens of a left-padded batch (prepare_inputs_for_generation).
            inputs["position_ids"] = (prefix_mask.cumsum(-1) - 1).masked_fill_(prefix_mask == 0, 1)
        switch.check()
        # The model itself, not prepare_inputs_for_generation: what that wants for a cache it is
        # handed differs between transformers releases, what forward takes does not.
        self.model(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
        switch.check()
        cache.batch_select_indices(torch.tensor(sources, device=input_ids.device))
        return cache

    def _decode_batch(
        self,
        encoded: list[_EncodedPrompt],
        params: SamplingParams,
        caps: list[int],
        n: int,
        switch: _CancelSwitch,
    ) -> list[Completion]:
        """``n`` completions of every prompt, of at most ``caps[prompt]`` tokens, all rows
        left-padded into one batch. Rows that have the same prompt start from one computation of
        it (:meth:`_shared_prompt_cache`).

        ``generate`` computes every row of its batch until the last one has finished, so the batch
        is decoded by a chain of ``generate`` calls: a call returns when rows have finished, with
        EOS, a stop string or at their cap (:meth:`_slack`), and the next one continues the others
        from the key-value cache the call left behind, the finished rows taken out of it. Every
        call is handed the whole sequences, so logits processors and stop strings see what they see
        in a single call. What changes is the batch ``torch.multinomial`` draws for: once a row of
        the batch has finished, the others get other random numbers than they would have (from the
        same distribution)."""
        prompt_ids = [e.ids for e in encoded]
        input_ids, attention_mask = self._left_pad(prompt_ids)
        model_kwargs: dict[str, Any] = {}
        if any(e.multimodal for e in encoded):
            model_kwargs = self._multimodal_inputs(encoded, n, input_ids.shape[1])
        else:
            cache = self._shared_prompt_cache(prompt_ids, input_ids, attention_mask, n, switch)
            if cache is not None:
                model_kwargs["past_key_values"] = cache
        input_ids = input_ids.repeat_interleave(n, dim=0)
        attention_mask = attention_mask.repeat_interleave(n, dim=0)
        rows = input_ids.shape[0]
        stop = StopStringCriteria(self._tokenizer, list(params.stop)) if params.stop else None
        temperature = params.temperature if params.temperature > 0.0 else 1.0
        recording = params.logprobs or params.top_logprobs > 0
        model = self._generating_model()
        row_caps = [cap for cap in caps for _ in range(n)]
        live = list(range(rows))  # the rows of the batch the next call decodes, in its row order
        num_tokens: list[int | None] = [None] * rows
        stopped = [False] * rows  # ended with EOS or a stop string, not at the cap
        tokens: list[torch.Tensor] = []  # per call [rows, steps of the call], as are the next three
        logprobs: list[torch.Tensor] = []
        top_ids: list[torch.Tensor] = []
        top_logprobs: list[torch.Tensor] = []
        steps = 0

        def of_batch(part: torch.Tensor, fill: float) -> torch.Tensor:
            """``part``, which has a row per live row, with a row per row of the batch."""
            whole = part.new_full((rows, *part.shape[1:]), fill)
            whole[live] = part
            return whole

        while True:
            slack = self._slack(len(live))
            # from a cache its rows can be taken out of: the shared prefill (a DynamicCache of
            # DynamicLayer) or the cache of a call that has given up rows
            continued = "past_key_values" in model_kwargs
            left = [row_caps[row] - steps for row in live]
            finish = _FinishRecorder(left, self._eos_ids, stop, slack=slack if continued else None)
            recorder = _LogprobRecorder(temperature, logprobs=params.logprobs, top=params.top_logprobs)
            hooks = [model.register_forward_hook(recorder.observe)] if recording else []
            if slack is not None and not continued:
                hooks.append(model.register_forward_hook(functools.partial(finish.check_cache, slack)))
            try:
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self._generation_config(params, max(left), output_logits=False),
                    stopping_criteria=StoppingCriteriaList([finish, switch]),
                    logits_processor=LogitsProcessorList([recorder] if recording else []),
                    **model_kwargs,
                )
            finally:
                for hook in hooks:
                    hook.remove()
            switch.check()  # the sequences were cut short, not finished
            generated = out.sequences[:, input_ids.shape[1] :]
            tokens.append(of_batch(generated.cpu(), self._pad_id))
            call_logprobs, call_top = recorder.result(generated)
            if call_logprobs is not None:
                logprobs.append(of_batch(call_logprobs, 0.0))
            if call_top is not None:
                top_ids.append(of_batch(call_top[0], 0))
                top_logprobs.append(of_batch(call_top[1], 0.0))
            for i, (row, finished_at) in enumerate(zip(live, finish.num_tokens, strict=True)):
                if finished_at is not None:
                    num_tokens[row] = steps + finished_at
                    stopped[row] = i in finish.stopped
            steps += generated.shape[1]
            if not finish.cut:
                break
            unfinished = [i for i, finished_at in enumerate(finish.num_tokens) if finished_at is None]
            cache = out.past_key_values
            input_ids, attention_mask = _select_cache_rows(cache, unfinished, out.sequences, attention_mask)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((len(unfinished), generated.shape[1]))],
                dim=1,
            )
            model_kwargs = {"past_key_values": cache}
            live = [live[i] for i in unfinished]
        return self._completions(
            [e.ids for e in encoded for _ in range(n)],
            torch.cat(tokens, dim=1),
            [steps if finished_at is None else finished_at for finished_at in num_tokens],
            ["stop" if row_stopped else "length" for row_stopped in stopped],
            torch.cat(logprobs, dim=1) if params.logprobs else None,
            (torch.cat(top_ids, dim=1), torch.cat(top_logprobs, dim=1)) if params.top_logprobs > 0 else None,
        )

    def _decode_drafting(
        self,
        encoded: list[_EncodedPrompt],
        params: SamplingParams,
        caps: list[int],
        n: int,
        switch: _CancelSwitch,
    ) -> list[Completion]:
        """:meth:`_decode_batch` with the MTP layers drafting. Assisted decoding takes one sequence
        per call, so the rows are decoded one after another, each call with the cap of its prompt;
        all of them draw from the one random stream the caller seeded, each from where the previous
        row left it."""
        layers = self.mtp_layers
        if layers is None:
            raise RuntimeError("the engine holds no MTP layers to draft with")
        # One row per call: the logits of its steps are small enough to be taken from ``generate``.
        generation_configs = {
            cap: self._generation_config(
                params, cap, output_logits=params.logprobs or params.top_logprobs > 0
            )
            for cap in set(caps)
        }
        stop = StopStringCriteria(self._tokenizer, list(params.stop)) if params.stop else None
        completions: list[Completion] = []
        for e, cap in zip(encoded, caps, strict=True):
            if not e.ids:
                raise ValueError("empty prompt")
            input_ids = torch.tensor([e.ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)
            for _ in range(n):
                recorder = _DraftFinishRecorder(len(e.ids), self._eos_ids, stop)
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_configs[cap],
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
                    [e.ids],
                    generated,
                    [generated.shape[1]],
                    [reason],
                    *self._logprobs_from_logits(out.logits, generated, params),
                )
        return completions

    @staticmethod
    def _logprobs_from_logits(
        logits: Sequence[torch.Tensor], generated: torch.Tensor, params: SamplingParams
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        """What :meth:`_LogprobRecorder.result` returns, from the raw logits ``generate`` returned:
        ``logits[t]`` ``[rows, vocab]`` are those the token at position ``t`` was chosen from."""
        rows, steps = generated.shape
        temperature = params.temperature if params.temperature > 0.0 else 1.0
        logprobs = torch.empty((rows, steps), dtype=torch.float32) if params.logprobs else None
        top: tuple[torch.Tensor, torch.Tensor] | None = None
        for t, step_logits in enumerate(
            logits[:steps] if logprobs is not None or params.top_logprobs else ()
        ):
            step_logprobs = torch.log_softmax(step_logits.float() / temperature, dim=-1)
            if logprobs is not None:
                logprobs[:, t] = step_logprobs.gather(1, generated[:, t : t + 1]).squeeze(1).cpu()
            if params.top_logprobs > 0:
                values, indices = step_logprobs.topk(min(params.top_logprobs, step_logprobs.shape[-1]))
                if top is None:
                    top = (
                        torch.empty((rows, steps, indices.shape[-1]), dtype=torch.int32),
                        torch.empty((rows, steps, indices.shape[-1]), dtype=torch.float32),
                    )
                top[0][:, t] = indices.cpu()
                top[1][:, t] = values.cpu()
        if params.top_logprobs > 0 and top is None:  # nothing was generated
            top = (torch.empty((rows, 0, 0), dtype=torch.int32), torch.empty((rows, 0, 0)))
        return logprobs, top

    def _completions(
        self,
        prompt_ids: Sequence[list[int]],
        generated: torch.Tensor,
        num_tokens: Sequence[int],
        reasons: Sequence[FinishReason | None],
        logprobs: torch.Tensor | None,
        top: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> list[Completion]:
        """One :class:`Completion` per row of ``generated`` ``[rows, steps]``: the row's first
        ``num_tokens[row]`` tokens as the continuation of ``prompt_ids[row]``, with their
        ``logprobs`` ``[rows, steps]`` and the ``top`` tokens of every position
        (``([rows, steps, k] ids, logprobs)``) where the caller asked for them."""
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
