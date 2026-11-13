"""Colocated (in-process) SGLang rollout engine.

Built against SGLang v0.5.23 (``python/sglang/srt/entrypoints/engine.py``):

* ``Engine(**ServerArgs kwargs)`` — ``model_path``, ``tokenizer_path``, ``tp_size``, ``dp_size``,
  ``mem_fraction_static``, ``context_length``, ``dtype``, ``enable_memory_saver``,
  ``trust_remote_code``, ``revision``, ``log_level`` … (``srt/arg_groups/fields/``).
* ``Engine.generate(prompt=None, sampling_params=None, input_ids=None, image_data=None, ...,
  return_logprob=False, logprob_start_len=None, ...)`` → list of ``{"text", "output_ids",
  "meta_info"}`` dicts, flat and prompt-major when ``n > 1``. ``logprob_start_len=-1`` asks for
  output-token logprobs only. ``image_data`` is one entry per request (list of PIL images, or
  ``None`` for text-only requests); ids of multimodal prompts come from the ``AutoProcessor``
  chat template (``monorl.rollout.media``) and are re-expanded by sglang's multimodal processor.
* ``Engine.generate`` blocks in ``Engine.loop.run_until_complete(...)`` on the caller's thread.
  :meth:`SGLangEngine.generate` runs the equivalent coroutine ``Engine.async_generate`` (same
  arguments and result) as a task on that loop instead, so ``cancel()`` can cancel the task from
  another thread. ``TokenizerManager.generate_request`` (``srt/managers/tokenizer_manager.py``)
  handles the cancellation itself: its ``except BaseException`` cleanup
  (``_release_req_states_on_failure``) sends ``abort_request(rid)`` to the scheduler for every
  request already dispatched and drops the rest.
* ``Engine.update_weights_from_tensor(named_tensors, load_format=None, flush_cache=True)`` takes
  ``List[Tuple[str, torch.Tensor]]`` and serialises them itself with
  ``MultiprocessingSerializer`` once per TP rank (``_serialize_tensors_per_rank``); returns
  ``(success, message)`` (``tokenizer_control_mixin.py``).
  The scheduler only accepts it inside a session: ``Engine.begin_weight_update(selector="all")``
  … ``Engine.end_weight_update()``, both ``(success, message)``
  (``srt/managers/scheduler_components/weight_updater.py``). With selector ``"all"`` the target
  and the draft runner get the same tensors and each loads its own checkpoint keys, so a stream
  that carries the MTP layers updates the drafter with the policy.
* ``Engine.update_weights_from_disk(model_path, load_format=None)`` → ``(success, message,
  num_paused_requests)``; needs no session and reloads target and draft.
* With ``speculative_algorithm`` set, finished requests report their verify passes and accepted
  drafts in ``meta_info`` (:func:`~monorl.adapters.sglang.common.speculation_of`).
* ``Engine.release_memory_occupation(tags=None)`` / ``resume_memory_occupation(tags=None)``
  (tags ``"weights"`` / ``"kv_cache"``; ``None`` = everything) need ``enable_memory_saver``.
* ``Engine.flush_cache()`` → ``FlushCacheReqOutput(success, message)``; ``Engine.shutdown()``.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import threading
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from monorl.adapters.sglang.common import (
    SGLANG_VERSIONS,
    WEIGHT_UPDATE_SELECTOR,
    SpeculationCounter,
    bucket_bytes,
    bucket_weights,
    drafts_with_mtp,
    encode_prompt,
    engine_kwargs,
    expand_image_data,
    expand_requests,
    group_outputs,
    is_media_prompt,
    logprob_kwargs,
    route_weights,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import BackendUnavailable, GenerationCancelled, WeightSyncError
from monorl.core.types import Completion, PromptInput, SamplingParams, SpeculationStats
from monorl.rollout.hf_engine import load_tokenizer

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "SGLang is not installed or failed to import. Install it with `uv pip install 'monorl[sglang]'` "
    "(sglang 0.5.x needs a CUDA GPU and Linux); on this machine use rollout.backend='hf', or "
    "rollout.extra.mode='server' with rollout.extra.url pointing at a running `sglang.launch_server`."
)


def import_sglang() -> Any:
    """Import sglang or raise :class:`BackendUnavailable` with an actionable message."""
    try:
        import sglang
    except ImportError as e:
        raise BackendUnavailable(f"{INSTALL_HINT} ({e})") from e
    version = getattr(sglang, "__version__", None)
    if version is not None:
        try:
            if Version(version) not in SpecifierSet(SGLANG_VERSIONS, prereleases=True):
                log.warning(
                    "sglang %s is outside the supported range %s; proceeding anyway", version, SGLANG_VERSIONS
                )
        except InvalidVersion:
            log.warning("could not parse sglang version %r", version)
    return sglang


class _GenerateCall:
    """One ``generate`` call that has not returned yet; fields are guarded by the engine's
    ``_calls_lock``."""

    def __init__(self) -> None:
        self.cancelled = False
        self.task: asyncio.Task[Any] | None = None
        """The ``Engine.async_generate`` task the call is running, if it got that far."""


def _engine_class() -> Any:
    """Import ``Engine`` with the ``sglang`` patchset applied (version-gated, idempotent)."""
    import_sglang()
    import monorl.adapters.sglang.patches  # noqa: F401  (registers PATCHES)
    from monorl.patching import apply

    report = apply("sglang")
    if report.failed:
        log.warning("sglang patches failed to apply: %s", report.failed)
    try:
        from sglang.srt.entrypoints.engine import Engine
    except ImportError as e:
        raise BackendUnavailable(f"{INSTALL_HINT} ({e})") from e
    return Engine


class SGLangEngine:
    """``rollout_engine: sglang`` in colocate mode: one ``sglang.srt.entrypoints.engine.Engine``
    per replica in this process.

    ``config.rollout.extra`` keys honoured: ``bucket_bytes`` (weight-update bucket size),
    ``sleep_tags`` (list passed to ``release_memory_occupation``; default all), ``log_level``,
    every ``ServerArgs`` name in :data:`~monorl.adapters.sglang.common.SERVER_ARG_PASSTHROUGH`
    (``schedule_policy``, ``enable_metrics``, ``random_seed``, ``attention_backend``, …) and an
    ``engine_kwargs`` dict passed verbatim to ``Engine``.

    Prompts are tokenised here with the HF tokenizer and always sent as ``input_ids`` so
    multi-turn continuations stay byte-exact and RadixAttention prefix cache hits are maximal.

    ``cancel()`` cancels the running ``generate`` (sglang aborts its requests in the scheduler)
    and fails the calls waiting for the lock; all of them raise :class:`GenerationCancelled`.

    With ``model.mtp.rollout`` the engine drafts with the checkpoint's MTP layers (lossless: the
    policy verifies every draft); ``update_weights`` routes the MTP weights of the stream to the
    drafter and ``speculation_stats()`` reports what the drafts achieved.
    """

    def __init__(self, config: RunConfig, *, replica: int = 0) -> None:
        engine_cls = _engine_class()
        self.config = config
        self.replica = replica
        self._tokenizer = load_tokenizer(config)
        self._processor: Any | None = None  # AutoProcessor, loaded on the first multimodal prompt
        self._processor_loaded = False
        self._bucket_bytes = bucket_bytes(config)
        self._sleep_tags: list[str] | None = config.rollout.extra.get("sleep_tags")
        self._kwargs = engine_kwargs(config)
        self._speculation = SpeculationCounter() if drafts_with_mtp(config) else None
        self._lock = threading.Lock()
        self._calls: set[_GenerateCall] = set()
        self._calls_lock = threading.Lock()
        self._asleep = False
        self._closed = False
        log.info("starting sglang Engine replica %d: %s", replica, self._kwargs)
        self._engine = engine_cls(**self._kwargs)

    # -- RolloutEngine --------------------------------------------------------------------
    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def engine(self) -> Any:
        return self._engine

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        if not prompts:
            return []
        call = _GenerateCall()
        with self._calls_lock:
            self._calls.add(call)
        try:
            processor = self._load_processor() if any(is_media_prompt(p) for p in prompts) else None
            encoded = [encode_prompt(self._tokenizer, processor, p) for p in prompts]
            prompt_ids = [ids for ids, _ in encoded]
            input_ids, sampling = expand_requests(prompt_ids, params)
            image_data = expand_image_data([images for _, images in encoded], params)
            kwargs: dict[str, Any] = {}
            if image_data is not None:
                kwargs["image_data"] = image_data
            with self._lock:
                if self._asleep:
                    raise RuntimeError("sglang engine is asleep; call wake_up() before generate()")
                loop = self._engine.loop
                with self._calls_lock:
                    if call.cancelled:  # while encoding or waiting for another generate
                        raise GenerationCancelled("generation cancelled")
                    call.task = loop.create_task(
                        self._engine.async_generate(
                            input_ids=input_ids,
                            sampling_params=sampling,
                            **logprob_kwargs(params),
                            **kwargs,
                        )
                    )
                try:
                    outputs = loop.run_until_complete(call.task)
                except asyncio.CancelledError:
                    if call.cancelled:
                        raise GenerationCancelled("generation cancelled") from None
                    raise
        finally:
            with self._calls_lock:
                self._calls.discard(call)
        if isinstance(outputs, dict):
            outputs = [outputs]
        if self._speculation is not None:
            self._speculation.record(outputs)
        return group_outputs(outputs, prompt_ids, params)

    def speculation_stats(self) -> SpeculationStats | None:
        """Speculative-decoding counters of the requests finished since the previous call;
        ``None`` when the engine does not draft."""
        return None if self._speculation is None else self._speculation.take()

    def cancel(self) -> None:
        """Abort the ``generate`` calls in flight (they raise :class:`GenerationCancelled`); a
        no-op when idle. The engine stays usable: later calls run normally."""
        with self._calls_lock:
            for call in self._calls:
                call.cancelled = True
                if call.task is not None:
                    # The task belongs to the thread running the engine's loop.
                    call.task.get_loop().call_soon_threadsafe(call.task.cancel)

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Push ``weights`` in one weight-update session covering the policy and the drafter, so
        both change together. MTP weights of the stream reach the drafter under their checkpoint
        keys; without drafting they are left out."""
        with self._lock:
            updated = 0
            routed = route_weights(weights, drafting=self._speculation is not None)
            buckets = bucket_weights(routed, max_bytes=self._bucket_bytes)
            first = next(buckets, None)
            if first is None:
                return
            _check_result(
                self._engine.begin_weight_update(selector=WEIGHT_UPDATE_SELECTOR), "begin_weight_update"
            )
            try:
                for bucket in itertools.chain([first], buckets):
                    result = self._engine.update_weights_from_tensor(
                        named_tensors=bucket, load_format=None, flush_cache=False
                    )
                    _check_result(result, "update_weights_from_tensor")
                    updated += len(bucket)
            except BaseException:
                # A failed update leaves the session open and the next push could not begin one.
                try:
                    _check_result(self._engine.end_weight_update(), "end_weight_update")
                except Exception:
                    log.exception(
                        "sglang replica %d: could not close the weight-update session", self.replica
                    )
                raise
            _check_result(self._engine.end_weight_update(), "end_weight_update")
            self._flush_cache()
        log.debug("sglang replica %d: updated %d tensors", self.replica, updated)

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        with self._lock:
            result = self._engine.update_weights_from_disk(model_path=os.fspath(path), load_format=None)
            _check_result(result, "update_weights_from_disk")
            self._flush_cache()

    def sleep(self) -> None:
        if not self.config.rollout.enable_sleep_mode:
            return
        with self._lock:
            if self._asleep:
                return
            self._engine.release_memory_occupation(tags=self._sleep_tags)
            self._asleep = True

    def wake_up(self) -> None:
        if not self.config.rollout.enable_sleep_mode:
            return
        with self._lock:
            if not self._asleep:
                return
            self._engine.resume_memory_occupation(tags=self._sleep_tags)
            self._asleep = False

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._engine.shutdown()

    # -- internals ----------------------------------------------------------------------
    def _load_processor(self) -> Any | None:
        if not self._processor_loaded:
            from monorl.rollout.media import load_processor

            self._processor = load_processor(self.config)
            self._processor_loaded = True
        return self._processor

    def _flush_cache(self) -> None:
        result = self._engine.flush_cache()
        success = getattr(result, "success", result)
        if success is False:
            message = getattr(result, "message", "") or ""
            log.warning("sglang flush_cache did not run (%s); stale radix entries may be served", message)


def _check_result(result: Any, op: str) -> None:
    """Upstream returns ``(success, message[, ...])`` tuples; anything falsy is a failed sync."""
    if isinstance(result, tuple):
        success, message = result[0], (result[1] if len(result) > 1 else "")
    elif isinstance(result, dict):
        success, message = result.get("success", True), result.get("message", "")
    else:
        success, message = getattr(result, "success", True), getattr(result, "message", "")
    if not success:
        raise WeightSyncError(f"sglang {op} failed: {message}")
