"""Shared, sglang-free helpers for the colocated engine and the HTTP client.

Everything here is verified against SGLang v0.5.23:

* ``python/sglang/srt/sampling/sampling_params.py`` ``SamplingParams`` fields: ``max_new_tokens``,
  ``stop``, ``temperature``, ``top_p``, ``top_k`` (``-1`` = whole vocabulary), ``min_p``,
  ``repetition_penalty``, ``n``, ``sampling_seed``, ``skip_special_tokens``. ``temperature < 1e-6``
  is turned into greedy (``top_k=1``) by ``__post_init__``.
* ``python/sglang/srt/managers/tokenizer_manager.py`` ``_handle_batch_request``: with ``n > 1``
  each prompt is expanded into ``n`` requests **sharing the same sampling params** (so a
  ``sampling_seed`` would make all ``n`` samples identical) and results come back flat,
  prompt-major. ``_handle_batch_output`` emits ``{"text", "output_ids", "meta_info"}`` where
  ``meta_info.finish_reason`` is ``{"type": "stop"|"length"|"abort", ...}``
  (``schedule_batch.py`` ``FINISH_*.to_json``) and, with ``return_logprob``,
  ``meta_info.output_token_logprobs`` is a list of ``(logprob, token_id, text_or_None)``
  (``detokenize_logprob_tokens``). ``top_logprobs_num=k`` adds ``meta_info.output_top_logprobs``:
  per output token the ``k`` most likely tokens as such tuples (``detokenize_top_logprobs_tokens``),
  the ``topk`` of the distribution the token was sampled from (``srt/layers/sampler.py``,
  ``srt/layers/logprob_processor.py`` ``get_top_logprobs``), whatever token was sampled.
* ``python/sglang/srt/utils/common.py`` ``MultiprocessingSerializer.serialize(obj, output_str)``:
  ``ForkingPickler`` dump, optionally base64 — mirrored by :func:`serialize_for_sglang` so the
  HTTP client can build ``/update_weights_from_tensor`` payloads without importing sglang.
* ``python/sglang/srt/arg_groups/fields/*.py`` — the ``ServerArgs`` keyword names used in
  :func:`engine_kwargs` and :data:`SERVER_ARG_PASSTHROUGH`.
* ``python/sglang/srt/arg_groups/fields/spec.py`` and ``arg_groups/speculative_hook.py`` — drafting
  with a model's own MTP layers is ``speculative_algorithm="EAGLE"`` (``"NEXTN"`` is an alias,
  upper-cased before it is resolved) with the MTP weights taken from the target checkpoint.
  ``speculative_num_steps`` / ``speculative_eagle_topk`` / ``speculative_num_draft_tokens`` are set
  all three or not at all (SGLang then chooses them per architecture); with ``topk == 1`` the
  verify window is ``num_steps + 1``. ``speculative_accept_threshold_single`` / ``_acc`` (default
  1.0) below 1.0 accept drafts more often than the target would sample them. Verification applies
  temperature, ``top_k`` and ``top_p``; the logprobs returned are the target's
  (``srt/layers/logprob_processor.py``).
* ``python/sglang/srt/managers/tokenizer_manager.py`` ``_calculate_spec_decoding_metrics``: a
  finished request that was verified at least once carries ``meta_info.spec_verify_ct`` (verify
  passes), ``spec_num_proposed_drafts`` (``verify_ct * (num_draft_tokens - 1)``) and
  ``spec_num_correct_drafts`` (accepted drafts) next to ``completion_tokens``.
* ``python/sglang/srt/managers/scheduler_components/weight_updater.py``: tensor updates are only
  accepted inside a session (``begin_weight_update(selector)`` … ``end_weight_update()``); with
  selector ``"all"`` the same named tensors go to the target and the draft runner, whose
  ``load_weights`` each pick their own checkpoint keys. A failed update leaves the session open.
* ``python/sglang/srt/entrypoints/engine.py`` ``Engine.generate(input_ids=..., image_data=...)``
  and ``GenerateReqInput.image_data`` (``srt/managers/io_struct.py``): one entry per request, each
  a list of images (PIL instance, file path, URL or base64/data URL — ``srt/utils.py``
  ``load_image``); ``None`` entries mean text-only. With ``input_ids`` the multimodal processor
  decodes the ids back to text, re-expands the image placeholders and returns the final
  ``input_ids``, so sending the ``AutoProcessor`` chat-template ids keeps prompt tokens exact.
"""

from __future__ import annotations

import base64
import io
import math
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from multiprocessing.reduction import ForkingPickler
from typing import Any

import torch
import torch.multiprocessing  # noqa: F401  (registers tensor/storage reducers on ForkingPickler)

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.types import (
    Completion,
    FinishReason,
    PromptInput,
    SamplingParams,
    SpeculationStats,
    TopLogprobs,
    messages_have_media,
)
from monorl.mtp import checkpoint_key, is_wire_name

SGLANG_VERSIONS = ">=0.5.23,<0.6"
FRAMEWORK = "sglang"

DEFAULT_BUCKET_BYTES = 512 * 1024 * 1024
"""Upper bound on the bytes handed to one ``update_weights_from_tensor`` call."""

# ``rollout.extra`` keys interpreted by monorl itself (never forwarded to ``ServerArgs``).
MONORL_EXTRA_KEYS: frozenset[str] = frozenset(
    {"mode", "url", "urls", "bucket_bytes", "engine_kwargs", "timeout", "sleep_tags", "log_level"}
)

# ``rollout.extra`` keys forwarded verbatim to ``sglang.srt.entrypoints.engine.Engine(**kwargs)``
# (= ``ServerArgs`` fields, ``python/sglang/srt/arg_groups/fields/``).
SERVER_ARG_PASSTHROUGH: frozenset[str] = frozenset(
    {
        "schedule_policy",
        "enable_metrics",
        "random_seed",
        "disable_cuda_graph",
        "cuda_graph_max_bs",
        "attention_backend",
        "max_running_requests",
        "max_total_tokens",
        "disable_radix_cache",
        "disable_overlap_schedule",
        "base_gpu_id",
        "device",
        "load_format",
        "pp_size",
        "ep_size",
        "nnodes",
        "node_rank",
        "dist_init_addr",
        "enable_p2p_check",
        "chat_template",
        "weight_version",
        "host",
        "port",
    }
)

MTP_ALGORITHMS: tuple[str, ...] = ("EAGLE", "NEXTN")
"""``speculative_algorithm`` values that draft with the MTP layers of the target checkpoint."""

ACCEPT_THRESHOLDS: tuple[str, ...] = (
    "speculative_accept_threshold_single",
    "speculative_accept_threshold_acc",
)

WEIGHT_UPDATE_SELECTOR = "all"
"""Weight-update sessions cover the target and the draft runner (a no-op for an absent draft)."""

_FINISH: dict[str | None, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "abort": "error",
    None: "error",
}


# -- prompts -----------------------------------------------------------------------------------


def is_token_prompt(prompt: PromptInput) -> bool:
    return len(prompt) > 0 and isinstance(prompt[0], int)


def chat_prompt_ids(tokenizer: Any, messages: Sequence[Any]) -> list[int]:
    """Chat template rendered with ``add_generation_prompt=True`` and tokenised **without**
    extra special tokens, so multi-turn callers re-deriving ids from the text get the same
    prefix bytes and RadixAttention keeps hitting the cache."""
    text = tokenizer.apply_chat_template(list(messages), add_generation_prompt=True, tokenize=False)
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


def prompt_token_ids(tokenizer: Any, prompt: PromptInput) -> list[int]:
    """Exact ids the engine conditions on. Raw token prompts pass through untouched."""
    if is_token_prompt(prompt):
        return [int(t) for t in prompt]  # type: ignore[union-attr]
    return chat_prompt_ids(tokenizer, prompt)  # type: ignore[arg-type]


def is_media_prompt(prompt: PromptInput) -> bool:
    """Chat prompt carrying image/audio/video parts (raw token prompts are always text-only)."""
    return not is_token_prompt(prompt) and messages_have_media(prompt)  # type: ignore[arg-type]


def encode_prompt(
    tokenizer: Any, processor: Any | None, prompt: PromptInput
) -> tuple[list[int], list[Any] | None]:
    """``(prompt_token_ids, images)``: text prompts via :func:`prompt_token_ids`; multimodal chats
    through the model's ``AutoProcessor`` chat template (``monorl.rollout.media``) so the ids carry
    the image placeholder tokens sglang's multimodal processor expands."""
    if not is_media_prompt(prompt):
        return prompt_token_ids(tokenizer, prompt), None
    if processor is None:
        raise ConfigError(
            "prompt carries image parts but the model has no AutoProcessor (not a vision-language model); "
            "set data.image_field only with a VLM checkpoint"
        )
    from monorl.rollout.media import chat_images, vlm_prompt_ids

    return vlm_prompt_ids(processor, prompt), chat_images(prompt)  # type: ignore[arg-type]


def expand_image_data(
    images: Sequence[list[Any] | None], params: SamplingParams
) -> list[list[Any] | None] | None:
    """Per-request ``image_data`` aligned with :func:`expand_requests` (``None`` when no prompt
    has images; entries are repeated ``n`` times when a seed forces one request per sample)."""
    if all(imgs is None for imgs in images):
        return None
    repeat = 1 if params.seed is None else params.n
    out: list[list[Any] | None] = []
    for imgs in images:
        entry = list(imgs) if imgs else None
        out.extend([entry] * repeat)
    return out


def image_data_urls(image_data: list[list[Any] | None] | None) -> list[list[str] | None] | None:
    """JSON-safe ``image_data`` for ``POST /generate``: PIL images become ``data:image/png;base64``
    URLs (sglang ``load_image`` decodes them), strings pass through."""
    if image_data is None:
        return None
    from monorl.rollout.media import image_data_url

    return [
        None if entry is None else [img if isinstance(img, str) else image_data_url(img) for img in entry]
        for entry in image_data
    ]


# -- sampling ----------------------------------------------------------------------------------


def sampling_dict(params: SamplingParams, *, n: int | None = None, seed: int | None = None) -> dict[str, Any]:
    """``SamplingParams`` → sglang ``sampling_params`` dict (``SamplingParams.__init__`` keys).

    ``no_stop_trim`` keeps a matched stop string in the text (sglang's detokenizer trims it by
    default), as the ``RolloutEngine`` contract requires."""
    out: dict[str, Any] = {
        "n": params.n if n is None else n,
        "max_new_tokens": params.max_new_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k if params.top_k > 0 else -1,
        "min_p": params.min_p,
        "repetition_penalty": params.repetition_penalty,
        "skip_special_tokens": True,
    }
    if params.stop:
        out["stop"] = list(params.stop)
        out["no_stop_trim"] = True
    if seed is not None:
        out["sampling_seed"] = int(seed)
    return out


def expand_requests(
    prompt_ids: Sequence[list[int]], params: SamplingParams
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Per-request ``input_ids`` and ``sampling_params`` lists for one ``generate`` call.

    Without a seed each prompt is one request with ``n=params.n`` (sglang expands it and shares
    the prefill). With a seed sglang would reuse the same ``sampling_seed`` for all ``n`` samples
    and return ``n`` identical completions, so the prompt is sent ``n`` times with seeds
    ``seed, seed+1, ...`` — the duplicated prompts still share the radix prefix.
    """
    if params.seed is None:
        return [list(p) for p in prompt_ids], [sampling_dict(params) for _ in prompt_ids]
    ids: list[list[int]] = []
    sps: list[dict[str, Any]] = []
    for p in prompt_ids:
        for k in range(params.n):
            ids.append(list(p))
            sps.append(sampling_dict(params, n=1, seed=params.seed + k))
    return ids, sps


# -- outputs -----------------------------------------------------------------------------------


def map_finish_reason(finish: Any) -> FinishReason:
    """``meta_info.finish_reason`` (``{"type": ...}`` dict, plain string or ``None``)."""
    if isinstance(finish, Mapping):
        finish = finish.get("type")
    if finish is not None and not isinstance(finish, str):
        finish = str(finish)
    return _FINISH.get(finish, "error")


def output_logprobs(meta_info: Mapping[str, Any]) -> list[float] | None:
    """Float logprobs from ``meta_info.output_token_logprobs`` ``[(logprob, token_id, text), ...]``."""
    entries = meta_info.get("output_token_logprobs")
    if entries is None:
        return None
    out: list[float] = []
    for entry in entries:
        value = entry[0] if isinstance(entry, (list, tuple)) else entry
        out.append(math.nan if value is None else float(value))
    return out


def logprob_kwargs(params: SamplingParams) -> dict[str, Any]:
    """``Engine.generate`` / ``GenerateReqInput`` logprob arguments for ``params``: output-token
    logprobs only (``logprob_start_len=-1``), plus the top-k per token when requested."""
    kwargs: dict[str, Any] = {
        "return_logprob": params.logprobs or params.top_logprobs > 0,
        "logprob_start_len": -1,
    }
    if params.top_logprobs > 0:
        kwargs["top_logprobs_num"] = params.top_logprobs
    return kwargs


def output_top_logprobs(meta_info: Mapping[str, Any], k: int) -> TopLogprobs | None:
    """The top-``k`` per output token from ``meta_info.output_top_logprobs``
    ``[[(logprob, token_id, text), ...], ...]``; ``None`` when a position carries none."""
    positions = meta_info.get("output_top_logprobs")
    if positions is None or not all(positions):
        return None
    return TopLogprobs.from_positions(
        [[(int(entry[1]), float(entry[0])) for entry in position] for position in positions], k
    )


def output_token_ids(output: Mapping[str, Any]) -> list[int]:
    ids = output.get("output_ids")
    if ids is None:
        entries = output.get("meta_info", {}).get("output_token_logprobs")
        if entries is None:
            raise ValueError(
                "sglang output carries neither 'output_ids' nor 'meta_info.output_token_logprobs'"
            )
        ids = [entry[1] for entry in entries]
    return [int(t) for t in ids]


def completion_from_output(
    output: Mapping[str, Any], *, prompt_ids: list[int], want_logprobs: bool, top_logprobs: int = 0
) -> Completion:
    meta = output.get("meta_info") or {}
    token_ids = output_token_ids(output)
    logprobs = output_logprobs(meta) if want_logprobs else None
    if logprobs is not None and len(logprobs) != len(token_ids):
        raise ValueError(
            f"sglang returned {len(logprobs)} output logprobs for {len(token_ids)} output tokens "
            f"(request {meta.get('id')!r})"
        )
    top = output_top_logprobs(meta, top_logprobs) if top_logprobs > 0 else None
    if top is not None and len(top) != len(token_ids):
        raise ValueError(
            f"sglang returned top logprobs for {len(top)} of {len(token_ids)} output tokens "
            f"(request {meta.get('id')!r})"
        )
    return Completion(
        text=output.get("text") or "",
        token_ids=token_ids,
        logprobs=logprobs,
        finish_reason=map_finish_reason(meta.get("finish_reason")),
        prompt_token_ids=list(prompt_ids),
        top_logprobs=top,
    )


def group_outputs(
    outputs: Sequence[Mapping[str, Any]],
    prompt_ids: Sequence[list[int]],
    params: SamplingParams,
) -> list[list[Completion]]:
    """Flat, prompt-major sglang outputs → ``params.n`` completions per prompt, order preserved."""
    expected = len(prompt_ids) * params.n
    if len(outputs) != expected:
        raise ValueError(
            f"sglang returned {len(outputs)} outputs, expected {expected} ({len(prompt_ids)} x {params.n})"
        )
    grouped: list[list[Completion]] = []
    for i, ids in enumerate(prompt_ids):
        chunk = outputs[i * params.n : (i + 1) * params.n]
        grouped.append(
            [
                completion_from_output(
                    o,
                    prompt_ids=ids,
                    want_logprobs=params.logprobs,
                    top_logprobs=params.top_logprobs,
                )
                for o in chunk
            ]
        )
    return grouped


# -- speculative decoding ----------------------------------------------------------------------


def drafts_with_mtp(config: RunConfig) -> bool:
    """Whether rollouts draft with the model's MTP layers."""
    return config.model.mtp.enabled and config.model.mtp.rollout


def speculation_of(outputs: Iterable[Mapping[str, Any]]) -> SpeculationStats:
    """What speculative decoding did for these finished requests, from their ``meta_info``.
    Requests that were never verified (aborted before the first decode step, or served by an
    engine that does not draft) carry no ``spec_*`` entries and count nothing."""
    stats = SpeculationStats()
    for output in outputs:
        meta = output.get("meta_info") or {}
        steps = meta.get("spec_verify_ct")
        if not steps:
            continue
        stats.add(
            SpeculationStats(
                steps=int(steps),
                drafted=int(meta["spec_num_proposed_drafts"]),
                accepted=int(meta["spec_num_correct_drafts"]),
                emitted=int(meta["completion_tokens"]),
            )
        )
    return stats


class SpeculationCounter:
    """Thread-safe running total behind the engines' ``speculation_stats()``."""

    def __init__(self) -> None:
        self._total = SpeculationStats()
        self._lock = threading.Lock()

    def record(self, outputs: Iterable[Mapping[str, Any]]) -> None:
        stats = speculation_of(outputs)
        with self._lock:
            self._total.add(stats)

    def take(self) -> SpeculationStats:
        """The counters since the previous call; reading resets them."""
        with self._lock:
            total, self._total = self._total, SpeculationStats()
        return total


def check_speculative_args(config: RunConfig, args: Mapping[str, Any], *, source: str) -> None:
    """Reject speculative-decoding arguments (``Engine`` kwargs, or the resolved ``ServerArgs`` a
    server reports) under which rollouts would not be what the config asks for. ``source`` names
    where ``args`` come from, for the message."""
    algorithm = args.get("speculative_algorithm")
    if algorithm is not None:
        for key in ACCEPT_THRESHOLDS:
            value = args.get(key)
            if value is not None and float(value) < 1.0:
                raise ConfigError(
                    f"{source}: {key}={value} accepts draft tokens more often than the policy samples them, "
                    "so the recorded logprobs would no longer be those of the sampling distribution; "
                    "leave it at 1.0"
                )
    if drafts_with_mtp(config) and str(algorithm).upper() not in MTP_ALGORITHMS:
        raise ConfigError(
            f"{source}: speculative_algorithm={algorithm!r}, but model.mtp.rollout drafts with the model's "
            f"MTP layers, which is speculative_algorithm 'EAGLE' (alias 'NEXTN') in SGLang"
        )


def speculative_kwargs(config: RunConfig) -> dict[str, Any]:
    """``ServerArgs`` that make SGLang draft with the checkpoint's MTP layers; empty unless
    ``model.mtp.rollout`` is on."""
    if not drafts_with_mtp(config):
        return {}
    kwargs: dict[str, Any] = {"speculative_algorithm": "EAGLE"}
    steps = config.model.mtp.num_speculative_tokens
    if steps is not None:
        kwargs.update(
            speculative_num_steps=steps, speculative_eagle_topk=1, speculative_num_draft_tokens=steps + 1
        )
    if config.rollout.enable_sleep_mode:
        # On release the memory saver stashes the target's buffers only; the CPU backups keep the
        # weights of target and draft across release/resume (verl sets both for MTP rollouts).
        kwargs.update(enable_weights_cpu_backup=True, enable_draft_weights_cpu_backup=True)
    return kwargs


# -- weights -----------------------------------------------------------------------------------


def route_weights(
    weights: Iterable[tuple[str, torch.Tensor]], *, drafting: bool
) -> Iterator[tuple[str, torch.Tensor]]:
    """A trainer's weight stream as SGLang loads it. MTP weights (``monorl.mtp.WIRE_PREFIX``) go
    out under their checkpoint keys, which the draft runner's ``load_weights`` picks up and the
    target's skips; an engine that does not draft has no runner for them, so they are dropped.
    The policy's weights pass through."""
    for name, tensor in weights:
        if not is_wire_name(name):
            yield name, tensor
        elif drafting:
            yield checkpoint_key(name), tensor


def bucket_weights(
    weights: Iterable[tuple[str, torch.Tensor]], *, max_bytes: int = DEFAULT_BUCKET_BYTES
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Group a weight stream into lists bounded by ``max_bytes`` so one
    ``update_weights_from_tensor`` call never pickles the whole model at once. Tensors are
    detached and made contiguous: ``ForkingPickler`` shares the *whole* underlying storage of a
    view, so a narrow slice of a fused parameter would otherwise ship the fused tensor."""
    bucket: list[tuple[str, torch.Tensor]] = []
    size = 0
    for name, tensor in weights:
        t = tensor.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        nbytes = t.numel() * t.element_size()
        if bucket and size + nbytes > max_bytes:
            yield bucket
            bucket, size = [], 0
        bucket.append((name, t))
        size += nbytes
    if bucket:
        yield bucket


def serialize_for_sglang(obj: Any, *, output_str: bool = False) -> bytes | str:
    """Byte-for-byte what ``MultiprocessingSerializer.serialize(obj, output_str)`` produces
    (``sglang/srt/utils/common.py``): a ``ForkingPickler`` dump, base64 when ``output_str``.
    Used by the HTTP client; the colocated engine hands raw tensors to ``Engine`` which
    serialises them itself (``Engine._serialize_tensors_per_rank``)."""
    try:
        from sglang.srt.utils import MultiprocessingSerializer  # pinned location, see module doc
    except ImportError:
        MultiprocessingSerializer = None  # noqa: N806
    if MultiprocessingSerializer is not None:
        return MultiprocessingSerializer.serialize(obj, output_str=output_str)
    buf = io.BytesIO()
    ForkingPickler(buf).dump(obj)
    data = buf.getvalue()
    return base64.b64encode(data).decode("utf-8") if output_str else data


def serialized_payloads(named_tensors: Sequence[tuple[str, torch.Tensor]], *, tp_size: int) -> list[str]:
    """One base64 payload per TP rank, as ``UpdateWeightsFromTensorReqInput.serialized_named_tensors``
    expects over HTTP (``io_struct.py``: ``Annotated[List[bytes], Base64Bytes()]``)."""
    if tp_size < 1:
        raise ValueError(f"tp_size must be >= 1, got {tp_size}")
    return [str(serialize_for_sglang(list(named_tensors), output_str=True)) for _ in range(tp_size)]


# -- config → engine kwargs ----------------------------------------------------------------------


def bucket_bytes(config: RunConfig) -> int:
    value = config.rollout.extra.get("bucket_bytes", DEFAULT_BUCKET_BYTES)
    if not isinstance(value, int) or value <= 0:
        raise ConfigError(f"rollout.extra.bucket_bytes must be a positive int, got {value!r}")
    return value


def engine_kwargs(config: RunConfig) -> dict[str, Any]:
    """``RunConfig`` → ``Engine(**kwargs)`` / ``ServerArgs`` keyword arguments.

    ``rollout.extra`` keys in :data:`SERVER_ARG_PASSTHROUGH` are forwarded as-is;
    ``rollout.extra.engine_kwargs`` (a dict) is merged last and wins. Unknown keys are ignored
    here so other components can keep their own options in ``rollout.extra``. With
    ``model.mtp.rollout`` the engine drafts with the model's MTP layers
    (:func:`speculative_kwargs`); overrides that would make speculative decoding lossy, or draft
    with something else, raise :class:`ConfigError`.
    """
    rollout, model, extra = config.rollout, config.model, config.rollout.extra
    kwargs: dict[str, Any] = {
        "model_path": model.name_or_path,
        "tokenizer_path": model.tokenizer_name_or_path,
        "trust_remote_code": model.trust_remote_code,
        "revision": model.revision,
        "dtype": model.dtype,
        "tp_size": rollout.tensor_parallel_size,
        "dp_size": 1,  # replicas are separate Engine instances (``replica`` kw), not sglang dp
        "mem_fraction_static": rollout.gpu_memory_utilization,
        "context_length": rollout.max_model_len,
        "enable_memory_saver": rollout.enable_sleep_mode,
        "log_level": str(extra.get("log_level", "warning")),
        **speculative_kwargs(config),
    }
    for key in SERVER_ARG_PASSTHROUGH:
        if key in extra:
            kwargs[key] = extra[key]
    overrides = extra.get("engine_kwargs") or {}
    if not isinstance(overrides, Mapping):
        raise ConfigError(f"rollout.extra.engine_kwargs must be a mapping, got {type(overrides).__name__}")
    kwargs.update(overrides)
    check_speculative_args(config, kwargs, source="rollout.extra.engine_kwargs")
    return kwargs


def server_url(config: RunConfig, replica: int) -> str:
    """``rollout.extra.url`` (one server) or ``rollout.extra.urls[replica]`` (one per replica)."""
    extra = config.rollout.extra
    urls = extra.get("urls")
    if urls:
        if replica >= len(urls):
            raise ConfigError(
                f"rollout.extra.urls has {len(urls)} entries but replica {replica} was requested"
            )
        return str(urls[replica]).rstrip("/")
    url = extra.get("url")
    if not url:
        raise ConfigError(
            "rollout.extra.mode='server' needs rollout.extra.url (or urls) of an sglang HTTP server"
        )
    return str(url).rstrip("/")
