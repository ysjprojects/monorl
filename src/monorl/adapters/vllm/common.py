"""Helpers shared by the colocated engine, the HTTP client and the server.

Everything here is pure Python/torch: no vLLM import, so it is unit-testable without vLLM and
usable from the server process (which has vLLM) and the trainer process (which may not).

Upstream references (vLLM v0.33.0):

* ``vllm/sampling_params.py`` ``SamplingParams`` fields ``n, max_tokens, temperature, top_p,
  top_k (0 disables, -1 accepted), min_p, repetition_penalty, stop, seed, logprobs``; with
  ``logprobs=0`` only the sampled token's logprob is returned (OpenAI semantics).
* ``vllm/outputs.py`` ``CompletionOutput(index, text, token_ids, cumulative_logprob, logprobs,
  finish_reason, ...)``; ``logprobs`` is ``SampleLogprobs = list[dict[int, Logprob]]`` or a
  ``FlatLogprobs`` (``vllm/logprobs.py``) when ``flat_logprobs=True``.
* ``vllm/v1/engine/__init__.py`` ``FINISH_REASON_STRINGS = ("stop", "length", "abort", "error",
  "repetition")``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import torch

from monorl.core.types import Completion, FinishReason, PromptInput, SamplingParams

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
    logprob computation entirely.
    """
    return {
        "n": params.n,
        "max_tokens": params.max_new_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": 0 if params.top_k < 0 else params.top_k,
        "min_p": params.min_p,
        "repetition_penalty": params.repetition_penalty,
        "stop": list(params.stop),
        "seed": params.seed,
        "logprobs": 0 if params.logprobs else None,
    }


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


def completions_from_request_output(
    request_output: Any, prompt_ids: list[int], *, want_logprobs: bool
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
            )
        )
    return result


def completions_from_choices(
    choices: Sequence[dict[str, Any]], prompt_ids: Sequence[list[int]], n: int, *, want_logprobs: bool
) -> list[list[Completion]]:
    """Group ``/v1/completions`` choices (``index = prompt_idx * n + i``) back per prompt.

    ``vllm/entrypoints/openai/completion/protocol.py`` ``CompletionResponseChoice``: ``text``,
    ``token_ids`` (``return_token_ids=True``), ``logprobs.token_logprobs`` (``logprobs=0``),
    ``finish_reason``, ``prompt_token_ids``.
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
            if want_logprobs:
                lp_obj = choice.get("logprobs") or {}
                raw = lp_obj.get("token_logprobs")
                if raw is None:
                    raise ValueError("server response lacks logprobs.token_logprobs; request must set logprobs=0")
                lps = [math.nan if v is None else float(v) for v in raw]
                if len(lps) != len(token_ids):
                    raise ValueError(f"logprobs length {len(lps)} != token_ids length {len(token_ids)}")
            group.append(
                Completion(
                    text=choice.get("text", ""),
                    token_ids=[int(t) for t in token_ids],
                    logprobs=lps,
                    finish_reason=map_finish_reason(choice.get("finish_reason")),
                    prompt_token_ids=list(choice.get("prompt_token_ids") or ids),
                )
            )
        grouped.append(group)
    return grouped


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
    for name in ("float32", "float16", "bfloat16", "float64", "int64", "int32", "int16", "int8", "uint8", "bool")
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
        raise ValueError(f"tensor payload has {len(data)} bytes, expected {expected} for {dtype}{list(shape)}")
    if numel == 0:
        return torch.empty(shape, dtype=torch_dtype)
    buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return buf.view(torch_dtype).reshape(shape).clone()
