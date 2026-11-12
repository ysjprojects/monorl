"""Colocated (in-process) vLLM rollout engine.

Built against vLLM v0.33.0:

* ``vllm/entrypoints/llm.py`` ``LLM(model, *, tokenizer, trust_remote_code, tensor_parallel_size,
  dtype, revision, seed, gpu_memory_utilization, **EngineArgs)`` — ``max_model_len``,
  ``enable_sleep_mode``, ``worker_extension_cls``, ``logprobs_mode`` are ``EngineArgs`` fields
  (``vllm/engine/arg_utils.py``). ``LLM.collective_rpc(method, timeout, args, kwargs)``,
  ``LLM.sleep(level, mode)``, ``LLM.wake_up(tags)``, ``LLM.reset_prefix_cache()``.
* ``LLM.generate`` is ``LLM.enqueue(prompts, sampling_params, use_tqdm=...)`` (returns the engine
  request ids) followed by ``OfflineInferenceMixin._run_engine``
  (``vllm/entrypoints/common/offline.py``): ``while llm_engine.has_unfinished_requests():
  llm_engine.step()``, finished outputs sorted by ``int(request_id)``. That loop cannot be
  interrupted and ``LLMEngine`` is not thread-safe, so :meth:`VllmEngine.generate` enqueues and
  drives the same loop itself; between two steps it honours ``cancel()`` with
  ``LLMEngine.abort_request(request_ids, internal=True)`` (``vllm/v1/engine/llm_engine.py``), the
  call upstream uses to roll back a failed enqueue.
* ``vllm/inputs/llm.py`` ``TokensPrompt(prompt_token_ids=..., multi_modal_data={"image": [...]})``;
  multimodal chats are tokenised with the model's ``AutoProcessor`` chat template (same recipe as
  ``monorl.rollout.hf_engine``) so the ids carry the image placeholder tokens vLLM expands.
  ``EngineArgs.limit_mm_per_prompt`` comes from ``rollout.extra.limit_mm_per_prompt``.
* ``vllm/platforms`` ``current_platform.is_sleep_mode_available()`` — ``ModelConfig`` raises when
  ``enable_sleep_mode`` is set on a platform without it (``vllm/config/model.py``).
* ``vllm/v1/engine/llm_engine.py`` ``LLMEngine.engine_core.shutdown()`` releases the engine.
* ``vllm/v1/worker/gpu_worker.py`` ``wake_up(tags)``: tags are ``"weights"`` / ``"kv_cache"``;
  waking a tag that is already mapped is an error in the cumem allocator, so we track which
  tags are released.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from monorl.adapters.vllm.common import (
    VLLM_VERSIONS,
    WORKER_EXTENSION_CLS,
    chunk_weights,
    completions_from_request_output,
    encode_prompt,
    is_media_prompt,
    limit_mm_per_prompt,
    sampling_kwargs,
    tokens_prompt_kwargs,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import BackendUnavailable, GenerationCancelled
from monorl.core.types import Completion, PromptInput, SamplingParams

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "vLLM is not installed or failed to import. Install it with `uv pip install 'monorl[vllm]'` "
    "(vLLM needs a CUDA/ROCm GPU and Linux); on this machine use rollout.backend='hf', or "
    "rollout.extra.mode='server' with rollout.extra.url pointing at a remote monorl vLLM server."
)

_ALL_TAGS: tuple[str, ...] = ("weights", "kv_cache")


def import_vllm() -> Any:
    """Import vLLM or raise :class:`BackendUnavailable` with an actionable message."""
    try:
        import vllm
    except ImportError as e:
        raise BackendUnavailable(f"{INSTALL_HINT} ({e})") from e
    version = getattr(vllm, "__version__", None)
    if version is not None:
        try:
            if Version(version) not in SpecifierSet(VLLM_VERSIONS, prereleases=True):
                log.warning(
                    "vllm %s is outside the supported range %s; proceeding anyway", version, VLLM_VERSIONS
                )
        except InvalidVersion:
            log.warning("could not parse vllm version %r", version)
    return vllm


class VllmEngine:
    """``rollout_engine: vllm`` in colocate mode: one ``vllm.LLM`` per replica in this process.

    ``config.rollout.extra`` keys honoured: ``in_process`` (default ``True``: run the engine core
    in this process so ``collective_rpc`` hands tensors over by reference), ``enforce_eager``,
    ``max_num_seqs``, ``max_num_batched_tokens``, ``logprobs_mode`` (default
    ``processed_logprobs`` so logprobs reflect temperature/top-p, as TRL does),
    plus an ``engine_kwargs`` dict passed verbatim to ``vllm.LLM``. ``limit_mm_per_prompt``
    (``{"image": N}`` or an int) bounds images per prompt for VLM runs.

    ``cancel()`` stops the running ``generate`` within one engine step (its requests are aborted
    in the engine) and fails the calls waiting for the lock; all of them raise
    :class:`GenerationCancelled`.
    """

    def __init__(self, config: RunConfig, *, replica: int = 0) -> None:
        extra = dict(config.rollout.extra)
        if extra.get("in_process", True):
            # Engine core in-process: collective_rpc args are not serialised (UniProcExecutor).
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        vllm = import_vllm()
        from monorl.patching import apply

        apply("vllm")
        from vllm.platforms import current_platform

        from monorl.rollout.hf_engine import load_tokenizer

        self._config = config
        self._replica = replica
        self._tokenizer = load_tokenizer(config)
        self._processor: Any | None = None  # AutoProcessor, loaded on the first multimodal prompt
        self._processor_loaded = False
        self._lock = threading.RLock()
        self._calls: set[threading.Event] = set()
        """Cancel flags of the ``generate`` calls that have not returned yet (running or waiting)."""
        self._calls_lock = threading.Lock()
        self._chunk_bytes = int(extra.get("chunk_bytes", 512 * 1024 * 1024))
        self._sleep_level = int(extra.get("sleep_level", 1))
        self._sleep_enabled = bool(config.rollout.enable_sleep_mode) and bool(
            current_platform.is_sleep_mode_available()
        )
        if config.rollout.enable_sleep_mode and not self._sleep_enabled:
            log.warning("rollout.enable_sleep_mode requested but this platform has no sleep mode; disabled")
        self._sleeping_tags: set[str] = set()  # vLLM memory tags currently released

        engine_kwargs: dict[str, Any] = {
            "tokenizer": config.model.tokenizer_name_or_path,
            "trust_remote_code": config.model.trust_remote_code,
            "revision": config.model.revision,
            "tensor_parallel_size": config.rollout.tensor_parallel_size,
            "dtype": config.model.dtype,
            "seed": config.seed + replica,
            "gpu_memory_utilization": config.rollout.gpu_memory_utilization,
            "max_model_len": config.rollout.max_model_len,
            "enable_sleep_mode": self._sleep_enabled,
            "worker_extension_cls": WORKER_EXTENSION_CLS,
            "logprobs_mode": extra.get("logprobs_mode", "processed_logprobs"),
            "enforce_eager": bool(extra.get("enforce_eager", False)),
        }
        for key in ("max_num_seqs", "max_num_batched_tokens"):
            if key in extra:
                engine_kwargs[key] = extra[key]
        mm_limit = limit_mm_per_prompt(extra)
        if mm_limit is not None:
            engine_kwargs["limit_mm_per_prompt"] = mm_limit
        engine_kwargs.update(extra.get("engine_kwargs", {}))
        self._vllm = vllm
        self._llm = vllm.LLM(config.model.name_or_path, **engine_kwargs)

    # -- RolloutEngine --------------------------------------------------------------------
    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def llm(self) -> Any:
        """The underlying ``vllm.LLM`` (for weight-sync strategies that need ``collective_rpc``)."""
        return self._llm

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        if not prompts:
            return []
        from vllm.inputs import TokensPrompt

        cancelled = threading.Event()
        with self._calls_lock:
            self._calls.add(cancelled)
        try:
            processor = self._load_processor() if any(is_media_prompt(p) for p in prompts) else None
            encoded = [encode_prompt(self._tokenizer, processor, p) for p in prompts]
            ids = [e[0] for e in encoded]
            vllm_prompts = [TokensPrompt(**tokens_prompt_kwargs(p_ids, images)) for p_ids, images in encoded]
            sp = self._vllm.SamplingParams(**sampling_kwargs(params))
            with self._lock:
                if cancelled.is_set():  # while encoding or waiting for another generate
                    raise GenerationCancelled("generation cancelled")
                self._wake(_ALL_TAGS)
                outputs = self._run(vllm_prompts, sp, cancelled)
        finally:
            with self._calls_lock:
                self._calls.discard(cancelled)
        if len(outputs) != len(prompts):
            raise RuntimeError(f"vllm returned {len(outputs)} outputs for {len(prompts)} prompts")
        result: list[list[Completion]] = []
        for p_ids, out in zip(ids, outputs, strict=True):
            comps = completions_from_request_output(out, p_ids, want_logprobs=params.logprobs)
            if len(comps) != params.n:
                raise RuntimeError(f"vllm returned {len(comps)} completions for n={params.n}")
            result.append(comps)
        return result

    def cancel(self) -> None:
        """Abort the ``generate`` calls in flight (they raise :class:`GenerationCancelled`); a
        no-op when idle. The engine stays usable: later calls run normally."""
        with self._calls_lock:
            for cancelled in self._calls:
                cancelled.set()

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        with self._lock:
            self._wake(("weights",))
            for chunk in chunk_weights(weights, max_bytes=self._chunk_bytes):
                self._llm.collective_rpc("monorl_update_weights", args=(chunk,))
            self._llm.reset_prefix_cache()

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        with self._lock:
            self._wake(("weights",))
            self._llm.collective_rpc("monorl_load_weights_from_path", args=(os.fspath(path),))
            self._llm.reset_prefix_cache()

    def sleep(self) -> None:
        if not self._sleep_enabled:
            return
        with self._lock:
            if self._sleeping_tags:
                return
            self._llm.sleep(level=self._sleep_level)
            self._sleeping_tags = set(_ALL_TAGS)

    def wake_up(self) -> None:
        if not self._sleep_enabled:
            return
        with self._lock:
            self._wake(_ALL_TAGS)

    def shutdown(self) -> None:
        llm = self._llm
        if llm is None:
            return
        llm.llm_engine.engine_core.shutdown()
        self._llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- internals ------------------------------------------------------------------------
    def _run(self, prompts: list[Any], sampling_params: Any, cancelled: threading.Event) -> list[Any]:
        """``LLM.generate`` with a cancellation point between engine steps: the finished
        ``RequestOutput`` per prompt, in prompt order."""
        engine = self._llm.llm_engine
        request_ids = self._llm.enqueue(prompts, sampling_params=sampling_params, use_tqdm=False)
        outputs: list[Any] = []
        while engine.has_unfinished_requests():
            if cancelled.is_set():
                engine.abort_request(request_ids, internal=True)
                raise GenerationCancelled("generation cancelled")
            outputs.extend(out for out in engine.step() if out.finished)
        # Requests finish in any order; ids come from the LLM's request counter.
        outputs.sort(key=lambda out: int(out.request_id))
        return outputs

    def _wake(self, tags: tuple[str, ...]) -> None:
        """Re-map only the memory tags that are currently released."""
        pending = [t for t in tags if t in self._sleeping_tags]
        if not pending:
            return
        self._llm.wake_up(tags=pending)
        self._sleeping_tags.difference_update(pending)

    def _load_processor(self) -> Any | None:
        if not self._processor_loaded:
            from monorl.rollout.media import load_processor

            self._processor = load_processor(self._config)
            self._processor_loaded = True
        return self._processor
