"""Versioned monkeypatches for SGLang ``>=0.5.23,<0.6``.

Each patch smooths one spot where the raw 0.5.23 API is awkward for the
:class:`~monorl.core.interfaces.RolloutEngine` protocol. Targets (verified at tag v0.5.23):

* ``sglang.srt.managers.tokenizer_manager.TokenizerManager.detokenize_logprob_tokens``
  ``(self, token_logprobs_val, token_logprobs_idx, decode_to_text)`` builds the
  ``(logprob, token_id, text)`` tuples that end up in ``meta_info.output_token_logprobs``.
* ``sglang.srt.entrypoints.engine.Engine.generate(self, prompt=None, sampling_params=None,
  input_ids=None, ...)`` — ``n > 1`` fans out in ``TokenizerManager._handle_batch_request`` with a
  shallow copy of the sampling params, so a ``sampling_seed`` yields ``n`` identical samples.
* ``Engine.release_memory_occupation(self, tags=None)`` / ``resume_memory_occupation`` — backed by
  ``torch_memory_saver``; without CUDA or ``enable_memory_saver`` the scheduler side raises.
* ``Engine.update_weights_from_tensor(self, named_tensors, load_format=None, flush_cache=True)``
  pickles ``named_tensors`` with ``ForkingPickler``; views ship their whole base storage and
  generators/dicts are not accepted.

The builders are plain functions so they can be unit-tested without sglang installed.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any

import torch

from monorl.adapters.sglang.common import FRAMEWORK, SGLANG_VERSIONS
from monorl.patching import Patch, PatchSet, register_patchset

log = logging.getLogger(__name__)

_ENGINE = "sglang.srt.entrypoints.engine.Engine"
_TOKENIZER_MANAGER = "sglang.srt.managers.tokenizer_manager.TokenizerManager"

# ``Engine.generate`` keyword arguments that may be per-request lists; the seed fan-out only
# rewrites calls where none of them is a list (otherwise it would have to expand them too).
_PER_REQUEST_ARGS: tuple[str, ...] = (
    "return_logprob",
    "logprob_start_len",
    "top_logprobs_num",
    "token_ids_logprob",
    "lora_path",
    "custom_logit_processor",
    "image_data",
    "audio_data",
    "video_data",
    "mm_hashes",
    "mm_content_hashes",
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "rid",
    "cache_salt",
    "return_hidden_states",
)


def has_params(original: Any, *names: str) -> bool:
    """``verify`` helper: the original callable accepts every named parameter."""
    try:
        params = inspect.signature(original).parameters
    except (TypeError, ValueError):
        return False
    return all(n in params for n in names)


# -- logprob float extraction --------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def build_logprob_floats(original: Callable[..., Any]) -> Callable[..., Any]:
    """Coerce logprob values / token ids to plain ``float`` / ``int`` before the tuples are built,
    so ``meta_info.output_token_logprobs`` never carries numpy or torch scalars (which ``orjson``
    in the HTTP server rejects and which break ``float()``-free consumers)."""

    @functools.wraps(original)
    def detokenize_logprob_tokens(
        self: Any, token_logprobs_val: Any, token_logprobs_idx: Any, decode_to_text: Any
    ) -> Any:
        vals = [_as_float(v) for v in token_logprobs_val]
        idxs = [_as_int(i) for i in token_logprobs_idx]
        return original(self, vals, idxs, decode_to_text)

    return detokenize_logprob_tokens


# -- seeded parallel sampling -----------------------------------------------------------------


def _seeded_fanout(
    sampling_params: Any, prompts: Any, input_ids: Any
) -> tuple[list[dict[str, Any]], Any, Any] | None:
    """Rewrite ``(sampling_params, prompt, input_ids)`` so each of the ``n`` samples per prompt
    gets its own ``sampling_seed``. Returns ``None`` when nothing needs rewriting."""
    if prompts is not None and input_ids is not None:
        return None
    if isinstance(sampling_params, Mapping):
        per_request = None
        template = sampling_params
    elif (
        isinstance(sampling_params, list)
        and sampling_params
        and all(isinstance(s, Mapping) for s in sampling_params)
    ):
        per_request = sampling_params
        template = None
    else:
        return None

    source = prompts if prompts is not None else input_ids
    if source is None:
        return None
    if isinstance(source, str) or (isinstance(source, list) and source and isinstance(source[0], int)):
        batch = [source]  # single prompt with n>1: sglang already returns a list for it, as the fan-out does
    elif isinstance(source, list):
        batch = source
    else:
        return None
    if per_request is not None and len(per_request) != len(batch):
        return None

    def sp_for(i: int) -> Mapping[str, Any]:
        return template if template is not None else per_request[i]  # type: ignore[index]

    needs = any(
        int(sp_for(i).get("n", 1) or 1) > 1 and sp_for(i).get("sampling_seed") is not None
        for i in range(len(batch))
    )
    if not needs:
        return None

    new_sps: list[dict[str, Any]] = []
    new_batch: list[Any] = []
    for i, item in enumerate(batch):
        sp = dict(sp_for(i))
        n = int(sp.get("n", 1) or 1)
        seed = sp.get("sampling_seed")
        if n > 1 and seed is not None:
            for k in range(n):
                new_batch.append(item)
                new_sps.append({**sp, "n": 1, "sampling_seed": int(seed) + k})
        else:
            new_batch.append(item)
            new_sps.append(sp)
    if prompts is not None:
        return new_sps, new_batch, None
    return new_sps, None, new_batch


def build_seeded_parallel_sampling(original: Callable[..., Any]) -> Callable[..., Any]:
    """``Engine.generate`` with ``sampling_seed`` and ``n > 1``: send ``n`` single-sample requests
    per prompt with seeds ``seed, seed+1, …`` (flat, prompt-major output order is unchanged).
    Calls with per-request list arguments other than ``sampling_params`` are left untouched."""
    signature = inspect.signature(original)

    @functools.wraps(original)
    def generate(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(self, *args, **kwargs)
        arguments = bound.arguments
        if arguments.get("stream") or any(isinstance(arguments.get(k), list) for k in _PER_REQUEST_ARGS):
            return original(self, *args, **kwargs)
        rewritten = _seeded_fanout(
            arguments.get("sampling_params"), arguments.get("prompt"), arguments.get("input_ids")
        )
        if rewritten is None:
            return original(self, *args, **kwargs)
        sps, prompt, input_ids = rewritten
        arguments["sampling_params"] = sps
        arguments["prompt"] = prompt
        arguments["input_ids"] = input_ids
        bound = signature.bind_partial(**arguments)
        return original(*bound.args, **bound.kwargs)

    return generate


# -- memory saver guards ----------------------------------------------------------------------


def _memory_saver_usable(engine: Any) -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    server_args = getattr(engine, "server_args", None)
    if server_args is not None and getattr(server_args, "enable_memory_saver", True) is False:
        return False, "Engine was started without enable_memory_saver=True"
    return True, ""


def build_memory_guard(original: Callable[..., Any], op: str) -> Callable[..., Any]:
    """Turn ``release/resume_memory_occupation`` into a logged no-op when the memory saver cannot
    work (no CUDA, or ``enable_memory_saver`` off) instead of failing inside the scheduler."""

    @functools.wraps(original)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        usable, reason = _memory_saver_usable(self)
        if not usable:
            log.warning("sglang %s skipped: %s", op, reason)
            return None
        return original(self, *args, **kwargs)

    return guarded


def build_release_memory_guard(original: Callable[..., Any]) -> Callable[..., Any]:
    return build_memory_guard(original, "release_memory_occupation")


def build_resume_memory_guard(original: Callable[..., Any]) -> Callable[..., Any]:
    return build_memory_guard(original, "resume_memory_occupation")


# -- update_weights_from_tensor input normalisation ----------------------------------------------


def normalize_named_tensors(named_tensors: Any) -> Any:
    """Mappings, generators and iterators → ``list[(name, tensor)]``; tensors detached and made
    contiguous. Pre-serialised payload lists (``bytes``/``str`` entries) pass through."""
    if isinstance(named_tensors, Mapping):
        items: Any = list(named_tensors.items())
    elif isinstance(named_tensors, (list, tuple)):
        items = list(named_tensors)
    elif hasattr(named_tensors, "__iter__"):
        items = list(named_tensors)
    else:
        return named_tensors
    out: list[Any] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], torch.Tensor):
            name, tensor = item
            t = tensor.detach()
            if not t.is_contiguous():
                t = t.contiguous()
            out.append((name, t))
        else:
            out.append(item)
    return out


def build_update_weights_named_tensors(original: Callable[..., Any]) -> Callable[..., Any]:
    """``Engine.update_weights_from_tensor`` accepting ``dict`` / generator inputs and never
    pickling a view's full base storage."""

    @functools.wraps(original)
    def update_weights_from_tensor(self: Any, named_tensors: Any, *args: Any, **kwargs: Any) -> Any:
        return original(self, normalize_named_tensors(named_tensors), *args, **kwargs)

    return update_weights_from_tensor


# -- the patch set ---------------------------------------------------------------------------

PATCHES = PatchSet(
    framework=FRAMEWORK,
    patches=[
        Patch(
            id="sglang.logprob_floats",
            target=f"{_TOKENIZER_MANAGER}.detokenize_logprob_tokens",
            versions=SGLANG_VERSIONS,
            mode="wrap",
            build=build_logprob_floats,
            description=(
                "meta_info.output_token_logprobs entries are (float, int, text): numpy/torch scalars "
                "coming out of the scheduler are coerced before the tuples are built."
            ),
            verify=lambda original: has_params(
                original, "token_logprobs_val", "token_logprobs_idx", "decode_to_text"
            ),
        ),
        Patch(
            id="sglang.seeded_parallel_sampling",
            target=f"{_ENGINE}.generate",
            versions=SGLANG_VERSIONS,
            mode="wrap",
            build=build_seeded_parallel_sampling,
            description=(
                "Engine.generate with sampling_seed and n>1 derives a distinct seed per sample instead of "
                "returning n identical completions."
            ),
            verify=lambda original: has_params(
                original, "prompt", "sampling_params", "input_ids", "return_logprob", "stream"
            ),
        ),
        Patch(
            id="sglang.release_memory_guard",
            target=f"{_ENGINE}.release_memory_occupation",
            versions=SGLANG_VERSIONS,
            mode="wrap",
            build=build_release_memory_guard,
            description="release_memory_occupation is a logged no-op without CUDA or enable_memory_saver.",
            verify=lambda original: has_params(original, "tags"),
        ),
        Patch(
            id="sglang.resume_memory_guard",
            target=f"{_ENGINE}.resume_memory_occupation",
            versions=SGLANG_VERSIONS,
            mode="wrap",
            build=build_resume_memory_guard,
            description="resume_memory_occupation is a logged no-op without CUDA or enable_memory_saver.",
            verify=lambda original: has_params(original, "tags"),
        ),
        Patch(
            id="sglang.update_weights_named_tensors",
            target=f"{_ENGINE}.update_weights_from_tensor",
            versions=SGLANG_VERSIONS,
            mode="wrap",
            build=build_update_weights_named_tensors,
            description=(
                "update_weights_from_tensor accepts dicts/generators and detaches + makes tensors contiguous "
                "so ForkingPickler never ships a view's whole base storage."
            ),
            verify=lambda original: has_params(original, "named_tensors", "load_format", "flush_cache"),
        ),
    ],
)

register_patchset(PATCHES)
