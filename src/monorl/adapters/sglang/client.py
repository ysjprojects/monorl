"""Server-mode SGLang rollout engine: an httpx client for ``python -m sglang.launch_server``.

Same :class:`~monorl.core.interfaces.RolloutEngine` protocol as the colocated engine, without
importing sglang. Endpoints (SGLang v0.5.23, ``python/sglang/srt/entrypoints/http_server.py``):

* ``POST /generate`` — body is ``GenerateReqInput`` (``srt/managers/io_struct.py`` /
  ``sglang/api/v1/api_types.py``): ``input_ids`` (list of token-id lists), ``sampling_params``
  (one dict per request), ``return_logprob``, ``logprob_start_len``, ``image_data`` (one entry per
  request: a list of ``data:image/png;base64`` URLs, or ``None`` for text-only); returns the same
  flat list of ``{"text", "output_ids", "meta_info"}`` dicts as ``Engine.generate``.
* ``POST /update_weights_from_tensor`` — ``UpdateWeightsFromTensorReqInput``:
  ``serialized_named_tensors`` (one base64 ``MultiprocessingSerializer`` payload per TP rank),
  ``load_format``, ``flush_cache``; ``{"success", "message"}`` with 400 on failure. Tensors are
  shared through ``ForkingPickler`` (CUDA IPC / shm), so the server must run on the same host.
* ``POST /update_weights_from_disk`` — ``{"model_path", "load_format"}`` →
  ``{"success", "message", "num_paused_requests"}``.
* ``POST /release_memory_occupation`` / ``POST /resume_memory_occupation`` — ``{"tags"}``.
* ``POST /flush_cache`` — text response, 200 when the radix cache was flushed.
* ``GET /get_server_info`` — resolved ``ServerArgs`` (``tp_size`` is read from here).
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable, Sequence
from typing import Any

import httpx
import torch

from monorl.adapters.sglang.common import (
    bucket_bytes,
    bucket_weights,
    encode_prompt,
    expand_image_data,
    expand_requests,
    group_outputs,
    image_data_urls,
    is_media_prompt,
    serialized_payloads,
    server_url,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import WeightSyncError
from monorl.core.types import Completion, PromptInput, SamplingParams
from monorl.rollout.hf_engine import load_tokenizer

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600.0


class SGLangServerClient:
    """``rollout_engine: sglang`` with ``rollout.extra.mode='server'``.

    ``rollout.extra`` keys: ``url``/``urls``, ``timeout`` (seconds, default 600), ``bucket_bytes``,
    ``sleep_tags``. ``transport`` (an ``httpx.BaseTransport``) is for tests.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        replica: int = 0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.replica = replica
        self._url = server_url(config, replica)
        self._tokenizer = load_tokenizer(config)
        self._processor: Any | None = None  # AutoProcessor, loaded on the first multimodal prompt
        self._processor_loaded = False
        self._bucket_bytes = bucket_bytes(config)
        self._sleep_tags: list[str] | None = config.rollout.extra.get("sleep_tags")
        timeout = float(config.rollout.extra.get("timeout", DEFAULT_TIMEOUT))
        self._http = httpx.Client(base_url=self._url, timeout=timeout, transport=transport)
        self._lock = threading.Lock()
        self._tp_size: int | None = None
        self._asleep = False
        self._closed = False

    # -- RolloutEngine --------------------------------------------------------------------
    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def url(self) -> str:
        return self._url

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        if not prompts:
            return []
        processor = self._load_processor() if any(is_media_prompt(p) for p in prompts) else None
        encoded = [encode_prompt(self._tokenizer, processor, p) for p in prompts]
        prompt_ids = [ids for ids, _ in encoded]
        input_ids, sampling = expand_requests(prompt_ids, params)
        body: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": sampling,
            "return_logprob": params.logprobs,
            "logprob_start_len": -1,
        }
        image_data = image_data_urls(expand_image_data([images for _, images in encoded], params))
        if image_data is not None:
            body["image_data"] = image_data
        response = self._post("/generate", json=body)
        if response.status_code >= 400:
            raise RuntimeError(f"sglang /generate failed (HTTP {response.status_code}): {response.text}")
        outputs = response.json()
        if isinstance(outputs, dict):
            outputs = [outputs]
        return group_outputs(outputs, prompt_ids, params)

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        with self._lock:
            tp_size = self._server_tp_size()
            updated = 0
            for bucket in bucket_weights(weights, max_bytes=self._bucket_bytes):
                body = {
                    "serialized_named_tensors": serialized_payloads(bucket, tp_size=tp_size),
                    "load_format": None,
                    "flush_cache": False,
                }
                self._check(
                    self._post("/update_weights_from_tensor", json=body), "update_weights_from_tensor"
                )
                updated += len(bucket)
            if updated:
                self._flush_cache()
        log.debug("sglang server %s: updated %d tensors", self._url, updated)

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        with self._lock:
            body = {"model_path": os.fspath(path), "load_format": None}
            self._check(self._post("/update_weights_from_disk", json=body), "update_weights_from_disk")
            self._flush_cache()

    def sleep(self) -> None:
        if not self.config.rollout.enable_sleep_mode:
            return
        with self._lock:
            if self._asleep:
                return
            self._post("/release_memory_occupation", json=self._tags_body()).raise_for_status()
            self._asleep = True

    def wake_up(self) -> None:
        if not self.config.rollout.enable_sleep_mode:
            return
        with self._lock:
            if not self._asleep:
                return
            self._post("/resume_memory_occupation", json=self._tags_body()).raise_for_status()
            self._asleep = False

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._http.close()

    # -- internals ----------------------------------------------------------------------
    def _load_processor(self) -> Any | None:
        if not self._processor_loaded:
            from monorl.rollout.media import load_processor

            self._processor = load_processor(self.config)
            self._processor_loaded = True
        return self._processor

    def server_info(self) -> dict[str, Any]:
        response = self._http.get("/get_server_info")
        response.raise_for_status()
        return response.json()

    def _server_tp_size(self) -> int:
        if self._tp_size is None:
            info = self.server_info()
            tp = info.get("tp_size")
            self._tp_size = int(tp) if tp is not None else int(self.config.rollout.tensor_parallel_size)
        return self._tp_size

    def _tags_body(self) -> dict[str, Any]:
        return {"tags": self._sleep_tags} if self._sleep_tags is not None else {}

    def _flush_cache(self) -> None:
        response = self._post("/flush_cache")
        if response.status_code >= 400:
            log.warning(
                "sglang /flush_cache did not run (%s); stale radix entries may be served", response.text
            )

    def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._http.post(path, **kwargs)

    @staticmethod
    def _check(response: httpx.Response, op: str) -> None:
        payload: Any
        try:
            payload = response.json()
        except ValueError:
            payload = {"success": response.status_code < 400, "message": response.text}
        if response.status_code >= 400 or not payload.get("success", True):
            raise WeightSyncError(
                f"sglang {op} failed (HTTP {response.status_code}): {payload.get('message', payload)}"
            )
