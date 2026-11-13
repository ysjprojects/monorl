"""Helpers shared by the colocated engine, the HTTP client and the server.

Everything here is pure Python/torch: no vLLM import, so it is unit-testable without vLLM and
usable from the server process (which has vLLM) and the trainer process (which may not).

Upstream references (vLLM v0.33.0):

* ``vllm/sampling_params.py`` ``SamplingParams`` fields ``n, max_tokens, temperature, top_p,
  top_k (0 disables, -1 accepted), min_p, repetition_penalty, stop, seed, logprobs``; with
  ``logprobs=0`` only the sampled token's logprob is returned (OpenAI semantics). ``logprobs=k``
  returns, per position, the sampled token followed by the ``k`` most likely tokens
  (``vllm/v1/sample/sampler.py`` ``gather_logprobs``, ``vllm/logprobs.py``
  ``append_logprobs_for_next_position``): ``k`` entries when the sampled token is among them, else
  ``k + 1``. ``k`` is bounded by the engine's ``max_logprobs`` (``vllm/config/model.py``, default 20).
* ``/v1/completions`` returns all of those entries as ``logprobs.top_logprobs[i]`` and
  ``/v1/chat/completions`` the first ``top_logprobs`` of them as
  ``logprobs.content[i].top_logprobs`` (``vllm/entrypoints/generate/base/serving.py``
  ``completion_top_logprobs_limit`` / ``chat_top_logprobs_limit``); ``return_tokens_as_token_ids``
  renders their tokens as ``"token_id:<id>"`` (``format_token_id_placeholder``).
* ``vllm/outputs.py`` ``CompletionOutput(index, text, token_ids, cumulative_logprob, logprobs,
  finish_reason, ...)``; ``logprobs`` is ``SampleLogprobs = list[dict[int, Logprob]]`` or a
  ``FlatLogprobs`` (``vllm/logprobs.py``) when ``flat_logprobs=True``.
* ``vllm/v1/engine/__init__.py`` ``FINISH_REASON_STRINGS = ("stop", "length", "abort", "error",
  "repetition")``.
* ``vllm/inputs/llm.py`` ``TokensPrompt(prompt_token_ids, multi_modal_data={"image": [PIL...]})``
  for in-process multimodal prompts; ``vllm/engine/arg_utils.py`` ``limit_mm_per_prompt``.
* ``vllm/entrypoints/openai/chat_completion/protocol.py`` ``ChatCompletionRequest`` (``messages``
  with ``image_url`` parts, ``logprobs``/``top_logprobs``, ``return_token_ids``,
  ``max_completion_tokens``) and ``ChatCompletionResponse`` (``choices[i].token_ids``,
  ``choices[i].logprobs.content[k].logprob``, response-level ``prompt_token_ids``).
* ``vllm/config/speculative.py`` ``SpeculativeConfig`` (``speculative_config`` engine argument,
  ``--speculative-config`` JSON on the server; unknown keys are rejected): ``method`` (``"mtp"``;
  the per-family names such as ``deepseek_mtp`` are rewritten to it), ``num_speculative_tokens``,
  ``rejection_sample_method`` (``"standard"`` is exact rejection sampling; ``"synthetic"`` and
  ``"fly"`` accept tokens the target would not have sampled), ``draft_lm_head_quantization``.
* Logprobs under speculative decoding are the target's for every emitted token, taken from the
  logits the token was verified against: after logits processors, temperature and top-k/top-p
  with ``logprobs_mode="processed_logprobs"``, raw otherwise
  (``vllm/v1/sample/rejection_sampler.py`` ``RejectionSampler.forward``,
  ``vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`` ``_verify_in_chunks``), which is what
  ``vllm/v1/sample/sampler.py`` ``Sampler.forward`` returns without a drafter. ``min_p`` and
  ``logit_bias`` are rejected per request (``SamplingParams._validate_spec_decode``).
* ``vllm/v1/spec_decode/metrics.py`` ``SpecDecodingProm``: Prometheus counters
  ``vllm:spec_decode_num_drafts`` (verification passes), ``..._num_draft_tokens``,
  ``..._num_accepted_tokens``, one sample per ``model_name``/``engine`` label set; a text
  exposition (``GET /metrics``) names the samples ``<counter>_total``.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import torch

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.types import (
    Completion,
    FinishReason,
    Message,
    PromptInput,
    SamplingParams,
    SpeculationStats,
    TopLogprobs,
    messages_have_media,
)

#: Fully-qualified name vLLM resolves via ``resolve_obj_by_qualname`` for ``worker_extension_cls``
#: (``vllm/v1/worker/worker_base.py`` ``WorkerWrapperBase.init_worker``).
WORKER_EXTENSION_CLS = "monorl.adapters.vllm.worker.MonorlWorkerExtension"

#: Supported upstream range; patches and the doctor report use the same specifier.
VLLM_VERSIONS = ">=0.33,<0.34"

_FINISH_REASONS: dict[str | None, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "repetition": "stop",  # vLLM's repetition detector ends the sequence like a stop string
    "abort": "error",
    "error": "error",
    None: "error",  # request never finished
}


def map_finish_reason(reason: str | None) -> FinishReason:
    return _FINISH_REASONS.get(reason, "error")


def sampling_kwargs(params: SamplingParams) -> dict[str, Any]:
    """Keyword arguments for ``vllm.SamplingParams`` (also the ``/v1/completions`` body subset).

    ``logprobs=0`` requests exactly the sampled token's logprob per position; ``None`` disables
    logprob computation entirely; ``params.top_logprobs`` requests that many most likely tokens
    too. ``include_stop_str_in_output`` keeps a matched stop string in the text (vLLM cuts it off
    by default), as the ``RolloutEngine`` contract requires.
    """
    if params.top_logprobs > 0:
        logprobs: int | None = params.top_logprobs
    else:
        logprobs = 0 if params.logprobs else None
    return {
        "n": params.n,
        "max_tokens": params.max_new_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": 0 if params.top_k < 0 else params.top_k,
        "min_p": params.min_p,
        "repetition_penalty": params.repetition_penalty,
        "stop": list(params.stop),
        "include_stop_str_in_output": True,
        "seed": params.seed,
        "logprobs": logprobs,
    }


#: ``format_token_id_placeholder`` (``vllm/entrypoints/generate/base/serving.py``).
TOKEN_ID_PREFIX = "token_id:"


def most_likely(candidates: Iterable[tuple[int, float]], k: int) -> list[tuple[int, float]]:
    """The ``k`` most likely ``(token_id, logprob)`` of one position, most likely first.

    vLLM reports the sampled token plus its top-k. A sampled token outside the top-k is the least
    likely entry, so cutting the sorted candidates at ``k`` leaves the top-k whatever was sampled.
    """
    return sorted(candidates, key=lambda c: -c[1])[:k]


def placeholder_token_id(token: str) -> int:
    """Token id of a ``"token_id:<id>"`` placeholder (``return_tokens_as_token_ids=true``)."""
    if not token.startswith(TOKEN_ID_PREFIX):
        raise ValueError(
            f"server returned the logprob token {token!r} instead of a 'token_id:<id>' placeholder; "
            "request must set return_tokens_as_token_ids=true"
        )
    return int(token[len(TOKEN_ID_PREFIX) :])


def is_token_prompt(prompt: PromptInput) -> bool:
    return len(prompt) > 0 and isinstance(prompt[0], int)


def prompt_token_ids(tokenizer: Any, prompt: PromptInput) -> list[int]:
    """Exact ids the engine conditions on: raw ids pass through, chats go through the HF chat
    template (``add_generation_prompt=True``) exactly like the ``hf`` engine so multi-turn
    callers can rely on token-exact continuation across backends."""
    if is_token_prompt(prompt):
        return [int(t) for t in prompt]
    from monorl.rollout.hf_engine import chat_prompt_ids

    return chat_prompt_ids(tokenizer, prompt)


def is_media_prompt(prompt: PromptInput) -> bool:
    """Chat prompt carrying image/audio/video parts (raw token prompts are always text-only)."""
    return not is_token_prompt(prompt) and messages_have_media(prompt)  # type: ignore[arg-type]


def encode_prompt(
    tokenizer: Any, processor: Any | None, prompt: PromptInput
) -> tuple[list[int], list[Any] | None]:
    """``(prompt_token_ids, images)`` for one prompt.

    Text prompts tokenise as :func:`prompt_token_ids`; multimodal chats go through the model's
    ``AutoProcessor`` chat template (``monorl.rollout.media.vlm_prompt_ids``) so the ids include
    the media placeholder tokens vLLM expands, and the decoded images are returned alongside.
    """
    if not is_media_prompt(prompt):
        return prompt_token_ids(tokenizer, prompt), None
    if processor is None:
        raise ConfigError(
            "prompt carries image parts but the model has no AutoProcessor (not a vision-language model); "
            "set data.image_field only with a VLM checkpoint"
        )
    from monorl.rollout.media import chat_images, vlm_prompt_ids

    return vlm_prompt_ids(processor, prompt), chat_images(prompt)  # type: ignore[arg-type]


def tokens_prompt_kwargs(ids: list[int], images: list[Any] | None) -> dict[str, Any]:
    """Keyword arguments for ``vllm.inputs.TokensPrompt``."""
    kwargs: dict[str, Any] = {"prompt_token_ids": ids}
    if images:
        kwargs["multi_modal_data"] = {"image": list(images)}
    return kwargs


def limit_mm_per_prompt(extra: dict[str, Any]) -> dict[str, int] | None:
    """``rollout.extra.limit_mm_per_prompt`` (``{"image": 4}`` or a bare int for images) as the
    ``EngineArgs.limit_mm_per_prompt`` mapping."""
    value = extra.get("limit_mm_per_prompt")
    if value is None:
        return None
    if isinstance(value, int):
        return {"image": int(value)}
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    raise ConfigError(f"rollout.extra.limit_mm_per_prompt must be an int or a mapping, got {value!r}")


# -- /v1/chat/completions (server mode, multimodal prompts) -------------------------------------


def chat_completion_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """monorl chat → OpenAI ``messages``: image parts become ``image_url`` data URLs (the server
    decodes them; vLLM accepts ``data:image/...;base64`` URLs), text parts stay ``text``."""
    from monorl.rollout.media import image_data_url, load_image

    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content", "")
        msg: dict[str, Any] = {"role": m.get("role", "user")}
        if isinstance(content, str):
            msg["content"] = content
        else:
            parts: list[dict[str, Any]] = []
            for part in content:
                kind = part.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": str(part.get("text", ""))})
                elif kind == "image":
                    url = str(part.get("image", ""))
                    if not url.startswith("data:"):
                        url = image_data_url(load_image(url))
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    raise ConfigError(f"vLLM server mode supports text and image parts, got {kind!r}")
            msg["content"] = parts
        for key in ("name", "tool_calls", "tool_call_id"):
            if key in m:
                msg[key] = m[key]
        out.append(msg)
    return out


def chat_completion_kwargs(params: SamplingParams) -> dict[str, Any]:
    """``ChatCompletionRequest`` body subset. ``logprobs=True, top_logprobs=0`` returns exactly the
    sampled token's logprob per position; ``return_token_ids=True`` adds ``choices[i].token_ids``
    and the response-level ``prompt_token_ids`` (the chat endpoint's logprob entries carry token
    *strings*, so the ids must come from ``return_token_ids``).

    With ``params.top_logprobs = k`` the request asks for ``k + 1`` candidates as token-id
    placeholders: the endpoint returns the first ``top_logprobs`` entries of a position, which
    start with the sampled token, so ``k + 1`` always contain the ``k`` most likely tokens."""
    top = params.top_logprobs
    body: dict[str, Any] = {
        "n": params.n,
        "max_completion_tokens": params.max_new_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": 0 if params.top_k < 0 else params.top_k,
        "min_p": params.min_p,
        "repetition_penalty": params.repetition_penalty,
        "stop": list(params.stop),
        "include_stop_str_in_output": True,
        "seed": params.seed,
        "logprobs": bool(params.logprobs) or top > 0,
        "top_logprobs": top + 1 if top > 0 else 0,
        "return_token_ids": True,
    }
    if top > 0:
        body["return_tokens_as_token_ids"] = True
    return body


def completions_from_chat_response(
    response: dict[str, Any], n: int, *, want_logprobs: bool, top_logprobs: int = 0
) -> list[Completion]:
    """One ``ChatCompletionResponse`` → ``n`` completions ordered by ``choices[i].index``."""
    choices = response.get("choices") or []
    if len(choices) != n:
        raise ValueError(f"server returned {len(choices)} chat choices, expected {n}")
    prompt_ids = response.get("prompt_token_ids")
    if prompt_ids is None:
        raise ValueError("chat response lacks prompt_token_ids; request must set return_token_ids=true")
    result: list[Completion] = []
    for choice in sorted(choices, key=lambda c: c["index"]):
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise ValueError(
                "chat response lacks choices[].token_ids; request must set return_token_ids=true"
            )
        lps: list[float] | None = None
        top: TopLogprobs | None = None
        if want_logprobs or top_logprobs > 0:
            content = (choice.get("logprobs") or {}).get("content")
            if content is None:
                raise ValueError("chat response lacks logprobs.content; request must set logprobs=true")
            if len(content) != len(token_ids):
                raise ValueError(f"logprobs length {len(content)} != token_ids length {len(token_ids)}")
            if want_logprobs:
                lps = [math.nan if e.get("logprob") is None else float(e["logprob"]) for e in content]
            if top_logprobs > 0 and all(e.get("top_logprobs") for e in content):
                top = TopLogprobs.from_positions(
                    [
                        most_likely(
                            (
                                (placeholder_token_id(c["token"]), float(c["logprob"]))
                                for c in e["top_logprobs"]
                            ),
                            top_logprobs,
                        )
                        for e in content
                    ],
                    top_logprobs,
                )
        message = choice.get("message") or {}
        result.append(
            Completion(
                text=str(message.get("content") or ""),
                token_ids=[int(t) for t in token_ids],
                logprobs=lps,
                finish_reason=map_finish_reason(choice.get("finish_reason")),
                prompt_token_ids=[int(t) for t in prompt_ids],
                top_logprobs=top,
            )
        )
    return result


def sampled_logprobs(output: Any) -> list[float] | None:
    """Per-token logprob of the *sampled* token for one ``CompletionOutput``.

    Handles both ``list[dict[int, Logprob]]`` and ``FlatLogprobs``; positions whose sampled token
    is missing from the returned dict (should not happen with ``logprobs>=0``) yield ``nan`` so
    misalignment is loud rather than silent.
    """
    lps = output.logprobs
    if lps is None:
        return None
    result: list[float] = []
    for token_id, position in zip(output.token_ids, lps, strict=True):
        entry = position.get(token_id) if position is not None else None
        lp = entry.logprob if entry is not None else math.nan
        result.append(float(lp))
    return result


def top_logprobs_of(output: Any, k: int) -> TopLogprobs | None:
    """The ``k`` most likely tokens at every position of one ``CompletionOutput`` sampled with
    ``logprobs=k``; ``None`` when a position carries no logprobs."""
    lps = output.logprobs
    if lps is None or any(position is None for position in lps):
        return None
    return TopLogprobs.from_positions(
        [
            most_likely(((int(token), float(entry.logprob)) for token, entry in position.items()), k)
            for position in lps
        ],
        k,
    )


def completions_from_request_output(
    request_output: Any, prompt_ids: list[int], *, want_logprobs: bool, top_logprobs: int = 0
) -> list[Completion]:
    """Convert one ``RequestOutput`` into ``n`` :class:`Completion` objects ordered by ``index``."""
    outs = sorted(request_output.outputs, key=lambda o: o.index)
    result: list[Completion] = []
    for out in outs:
        result.append(
            Completion(
                text=out.text,
                token_ids=[int(t) for t in out.token_ids],
                logprobs=sampled_logprobs(out) if want_logprobs else None,
                finish_reason=map_finish_reason(out.finish_reason),
                prompt_token_ids=list(prompt_ids),
                top_logprobs=top_logprobs_of(out, top_logprobs) if top_logprobs > 0 else None,
            )
        )
    return result


def completions_from_choices(
    choices: Sequence[dict[str, Any]],
    prompt_ids: Sequence[list[int]],
    n: int,
    *,
    want_logprobs: bool,
    top_logprobs: int = 0,
) -> list[list[Completion]]:
    """Group ``/v1/completions`` choices (``index = prompt_idx * n + i``) back per prompt.

    ``vllm/entrypoints/openai/completion/protocol.py`` ``CompletionResponseChoice``: ``text``,
    ``token_ids`` (``return_token_ids=True``), ``logprobs.token_logprobs`` (``logprobs>=0``),
    ``logprobs.top_logprobs`` (one ``{"token_id:<id>": logprob}`` mapping per position with
    ``logprobs=k`` and ``return_tokens_as_token_ids=True``), ``finish_reason``, ``prompt_token_ids``.
    """
    expected = len(prompt_ids) * n
    if len(choices) != expected:
        raise ValueError(f"server returned {len(choices)} choices, expected {expected}")
    ordered = sorted(choices, key=lambda c: c["index"])
    grouped: list[list[Completion]] = []
    for p_idx, ids in enumerate(prompt_ids):
        group: list[Completion] = []
        for choice in ordered[p_idx * n : (p_idx + 1) * n]:
            token_ids = choice.get("token_ids")
            if token_ids is None:
                raise ValueError("server response lacks token_ids; request must set return_token_ids=true")
            lps: list[float] | None = None
            lp_obj = choice.get("logprobs") or {}
            if want_logprobs:
                raw = lp_obj.get("token_logprobs")
                if raw is None:
                    raise ValueError(
                        "server response lacks logprobs.token_logprobs; request must set logprobs=0"
                    )
                lps = [math.nan if v is None else float(v) for v in raw]
                if len(lps) != len(token_ids):
                    raise ValueError(f"logprobs length {len(lps)} != token_ids length {len(token_ids)}")
            top: TopLogprobs | None = None
            if top_logprobs > 0:
                positions = lp_obj.get("top_logprobs")
                if positions is None or len(positions) != len(token_ids):
                    raise ValueError(
                        "server response lacks logprobs.top_logprobs for every token; request must set "
                        f"logprobs={top_logprobs}"
                    )
                if all(positions):
                    top = TopLogprobs.from_positions(
                        [
                            most_likely(
                                ((placeholder_token_id(t), float(lp)) for t, lp in position.items()),
                                top_logprobs,
                            )
                            for position in positions
                        ],
                        top_logprobs,
                    )
            group.append(
                Completion(
                    text=choice.get("text", ""),
                    token_ids=[int(t) for t in token_ids],
                    logprobs=lps,
                    finish_reason=map_finish_reason(choice.get("finish_reason")),
                    prompt_token_ids=list(choice.get("prompt_token_ids") or ids),
                    top_logprobs=top,
                )
            )
        grouped.append(group)
    return grouped


# -- MTP drafting (speculative decoding) -------------------------------------------------------


def speculative_config(config: RunConfig) -> dict[str, Any] | None:
    """``speculative_config`` of an engine that drafts with the model's MTP layers; ``None``
    when ``model.mtp.rollout`` is off. Keys of ``rollout.extra.engine_kwargs.speculative_config``
    are laid over ``{"method": "mtp", "num_speculative_tokens": N or 1}``.

    Raises :class:`ConfigError` for settings under which the rollout would not be what
    ``model.mtp.rollout`` promises: a drafter that is not the MTP layers, a rejection sampler
    that is not exact (the recorded logprobs would not be those of the distribution the tokens
    were sampled from), and sleep level 2, which discards the drafter's weights.
    """
    mtp = config.model.mtp
    if not (mtp.enabled and mtp.rollout):
        return None
    extra = config.rollout.extra
    user = (extra.get("engine_kwargs") or {}).get("speculative_config") or {}
    if not isinstance(user, dict):
        raise ConfigError(f"rollout.extra.engine_kwargs.speculative_config must be a mapping, got {user!r}")
    compiled: dict[str, Any] = {"method": "mtp", "num_speculative_tokens": mtp.num_speculative_tokens or 1}
    # vLLM accepts dashed keys (EngineArgs.create_speculative_config); compare one spelling.
    compiled.update({str(key).replace("-", "_"): value for key, value in user.items()})
    method = compiled["method"]
    if not (isinstance(method, str) and (method == "mtp" or method.endswith("_mtp"))):
        raise ConfigError(
            f"model.mtp.rollout drafts with the model's MTP layers, but "
            f"rollout.extra.engine_kwargs.speculative_config.method is {method!r}; remove it or set "
            "model.mtp.rollout: false"
        )
    rejection = compiled.get("rejection_sample_method", "standard")
    if rejection != "standard":
        raise ConfigError(
            f"speculative_config.rejection_sample_method={rejection!r} is not supported with "
            "model.mtp.rollout: only 'standard' samples every token from the policy's distribution, "
            "which is what the recorded logprobs describe"
        )
    if config.rollout.enable_sleep_mode and int(extra.get("sleep_level", 1)) == 2:
        raise ConfigError(
            "model.mtp.rollout cannot be combined with rollout.extra.sleep_level=2: vLLM's level-2 sleep "
            "discards the drafter's weights, after which no draft is accepted; use sleep_level=1"
        )
    return compiled


#: ``SpecDecodingProm`` counters, in the order of :meth:`SpeculationCounters.delta`'s totals.
SPEC_DECODE_COUNTERS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)


def spec_decode_totals(samples: Iterable[tuple[str, float]]) -> tuple[int, int, int] | None:
    """``(drafts, draft tokens, accepted tokens)`` summed over the label sets of
    ``(counter name, value)`` samples; ``None`` when the counters are not among them (the engine
    does not draft, or it keeps no statistics)."""
    totals = dict.fromkeys(SPEC_DECODE_COUNTERS, 0)
    seen = False
    for name, value in samples:
        if name in totals:
            totals[name] += int(value)
            seen = True
    return (
        (totals[SPEC_DECODE_COUNTERS[0]], totals[SPEC_DECODE_COUNTERS[1]], totals[SPEC_DECODE_COUNTERS[2]])
        if seen
        else None
    )


def prometheus_samples(text: str) -> Iterator[tuple[str, float]]:
    """``(counter name, value)`` of the counter samples in a Prometheus text exposition
    (``vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 12.0``)."""
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        sample, _, value = line.rpartition(" ")
        name = sample.split("{", 1)[0].strip()
        if not name.endswith("_total"):
            continue
        try:
            yield name.removesuffix("_total"), float(value)
        except ValueError:
            continue


class SpeculationCounters:
    """Turns vLLM's ever-growing speculative-decoding counters into the deltas
    ``speculation_stats()`` reports."""

    def __init__(self, totals: tuple[int, int, int] = (0, 0, 0)) -> None:
        self._seen = totals
        self._lock = threading.Lock()

    def delta(self, totals: tuple[int, int, int]) -> SpeculationStats:
        """What happened between the previous call (or construction) and ``totals``."""
        with self._lock:
            previous, self._seen = self._seen, totals
        if any(now < before for now, before in zip(totals, previous, strict=True)):
            previous = (0, 0, 0)  # the counters started over (a new engine registered them, a restart)
        steps, drafted, accepted = (now - before for now, before in zip(totals, previous, strict=True))
        # Every verification pass emits its accepted drafts and one token of the policy's own.
        return SpeculationStats(steps=steps, drafted=drafted, accepted=accepted, emitted=accepted + steps)


def speculative_serve_flag(compiled: dict[str, Any]) -> str:
    """The ``vllm serve`` / ``monorl.adapters.vllm.server`` argument for ``compiled``."""
    return f"--speculative-config '{json.dumps(compiled)}'"


# -- weight chunking / wire format -------------------------------------------------------------

DEFAULT_CHUNK_BYTES = 512 * 1024 * 1024


def chunk_weights(
    weights: Iterable[tuple[str, torch.Tensor]], *, max_bytes: int = DEFAULT_CHUNK_BYTES
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Group a weight stream into lists bounded by ``max_bytes`` so one ``collective_rpc`` call
    never has to serialise the whole model."""
    chunk: list[tuple[str, torch.Tensor]] = []
    size = 0
    for name, tensor in weights:
        t = tensor.detach()
        nbytes = t.numel() * t.element_size()
        if chunk and size + nbytes > max_bytes:
            yield chunk
            chunk, size = [], 0
        chunk.append((name, t))
        size += nbytes
    if chunk:
        yield chunk


_DTYPE_NAMES: dict[str, torch.dtype] = {
    name: getattr(torch, name)
    for name in (
        "float32",
        "float16",
        "bfloat16",
        "float64",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint8",
        "bool",
    )
}


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return _DTYPE_NAMES[name.removeprefix("torch.")]
    except KeyError as e:
        raise ValueError(f"unsupported tensor dtype {name!r}") from e


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Raw little-endian buffer of ``tensor`` (any dtype, incl. bf16 which numpy lacks)."""
    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).numpy().tobytes()


def tensor_from_bytes(data: bytes, dtype: str, shape: Sequence[int]) -> torch.Tensor:
    """Inverse of :func:`tensor_to_bytes`; copies so the result owns its memory."""
    torch_dtype = dtype_from_name(dtype)
    shape = tuple(int(s) for s in shape)
    numel = math.prod(shape) if shape else 1
    expected = numel * torch.empty((), dtype=torch_dtype).element_size()
    if len(data) != expected:
        raise ValueError(
            f"tensor payload has {len(data)} bytes, expected {expected} for {dtype}{list(shape)}"
        )
    if numel == 0:
        return torch.empty(shape, dtype=torch_dtype)
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return buf.view(torch_dtype).reshape(shape).clone()
