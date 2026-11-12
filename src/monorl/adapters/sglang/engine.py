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
* ``Engine.update_weights_from_tensor(named_tensors, load_format=None, flush_cache=True)`` takes
  ``List[Tuple[str, torch.Tensor]]`` and serialises them itself with
  ``MultiprocessingSerializer`` once per TP rank (``_serialize_tensors_per_rank``); returns
  ``(success, message)`` (``tokenizer_control_mixin.py``).
* ``Engine.update_weights_from_disk(model_path, load_format=None)`` → ``(success, message,
  num_paused_requests)``.
* ``Engine.release_memory_occupation(tags=None)`` / ``resume_memory_occupation(tags=None)``
  (tags ``"weights"`` / ``"kv_cache"``; ``None`` = everything) need ``enable_memory_saver``.
* ``Engine.flush_cache()`` → ``FlushCacheReqOutput(success, message)``; ``Engine.shutdown()``.
"""

from __future__ import annotations

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
    bucket_bytes,
    bucket_weights,
    encode_prompt,
    engine_kwargs,
    expand_image_data,
    expand_requests,
    group_outputs,
    is_media_prompt,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import BackendUnavailable, WeightSyncError
from monorl.core.types import Completion, PromptInput, SamplingParams
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
        self._lock = threading.Lock()
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
            outputs = self._engine.generate(
                input_ids=input_ids,
                sampling_params=sampling,
                return_logprob=params.logprobs,
                logprob_start_len=-1,
                **kwargs,
            )
        if isinstance(outputs, dict):
            outputs = [outputs]
        return group_outputs(outputs, prompt_ids, params)

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        with self._lock:
            updated = 0
            for bucket in bucket_weights(weights, max_bytes=self._bucket_bytes):
                result = self._engine.update_weights_from_tensor(
                    named_tensors=bucket, load_format=None, flush_cache=False
                )
                _check_result(result, "update_weights_from_tensor")
                updated += len(bucket)
            if updated:
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
