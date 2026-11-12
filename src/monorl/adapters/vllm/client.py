"""Server-mode vLLM rollout engine: an HTTP client for :mod:`monorl.adapters.vllm.server`.

Implements the same :class:`~monorl.core.interfaces.RolloutEngine` protocol as the colocated
engine without importing vLLM, so the trainer process can live on another machine or in an
environment where vLLM is not installable.

Endpoints used (vLLM v0.33.0 unless noted):

* ``POST /v1/completions`` (``vllm/entrypoints/openai/completion/protocol.py``): ``prompt`` as a
  list of token-id lists, ``return_token_ids=true`` → ``choices[i].token_ids`` /
  ``prompt_token_ids``; ``logprobs=0`` → ``choices[i].logprobs.token_logprobs``.
* ``POST /v1/chat/completions`` (``vllm/entrypoints/openai/chat_completion/protocol.py``) for
  prompts with image parts: ``messages`` with ``image_url`` data URLs, ``logprobs=true,
  top_logprobs=0`` and ``return_token_ids=true``. The chat endpoint's logprob entries only carry
  token *strings*, so token ids come from ``choices[i].token_ids`` and the prompt ids the server
  conditioned on (chat template + image placeholders rendered server-side) from the
  response-level ``prompt_token_ids``. One request per prompt (a chat request is one conversation).
* ``POST /sleep?level=N``, ``POST /wake_up``, ``GET /is_sleeping``
  (``vllm/entrypoints/serve/dev/sleep/api_router.py``, enabled by ``VLLM_SERVER_DEV_MODE=1``).
* ``POST /reset_prefix_cache`` (``vllm/entrypoints/serve/dev/cache/api_router.py``).
* ``POST /update_named_param``, ``POST /load_weights_from_path`` — monorl additions.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable, Sequence
from typing import Any

import httpx
import torch

from monorl.adapters.vllm.common import (
    chat_completion_kwargs,
    chat_completion_messages,
    completions_from_chat_response,
    completions_from_choices,
    dtype_name,
    is_media_prompt,
    prompt_token_ids,
    sampling_kwargs,
    tensor_to_bytes,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.types import Completion, PromptInput, SamplingParams

log = logging.getLogger(__name__)
_ALL_TAGS: tuple[str, ...] = ("weights", "kv_cache")


def server_url(config: RunConfig, replica: int) -> str:
    """``rollout.extra.url`` (one server) or ``rollout.extra.urls[replica]`` (one per replica)."""
    extra = config.rollout.extra
    urls = extra.get("urls")
    if urls:
        if replica >= len(urls):
            raise ConfigError(f"rollout.extra.urls has {len(urls)} entries but replica {replica} was requested")
        return str(urls[replica]).rstrip("/")
    url = extra.get("url")
    if not url:
        raise ConfigError("rollout.extra.mode='server' needs rollout.extra.url (or urls) of a monorl vLLM server")
    return str(url).rstrip("/")


class VllmServerClient:
    """``rollout_engine: vllm`` with ``rollout.extra.mode='server'``.

    ``rollout.extra`` keys: ``url``/``urls``, ``model_name`` (served model name, defaults to
    ``model.name_or_path``), ``timeout`` seconds per request (default: none), ``sleep_level``.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        replica: int = 0,
        transport: httpx.BaseTransport | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        extra = config.rollout.extra
        self._config = config
        self._url = server_url(config, replica)
        self._model_name = str(extra.get("model_name") or config.model.name_or_path)
        self._sleep_level = int(extra.get("sleep_level", 1))
        self._sleep_enabled = bool(config.rollout.enable_sleep_mode)
        timeout = extra.get("timeout")
        self._client = httpx.Client(
            base_url=self._url,
            timeout=httpx.Timeout(None if timeout is None else float(timeout)),
            transport=transport,
        )
        if tokenizer is None:
            from monorl.rollout.hf_engine import load_tokenizer

            tokenizer = load_tokenizer(config)
        self._tokenizer = tokenizer
        self._lock = threading.RLock()
        self._sleeping_tags: set[str] = set()  # vLLM memory tags currently released

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
        text_idx = [i for i, p in enumerate(prompts) if not is_media_prompt(p)]
        media_idx = [i for i, p in enumerate(prompts) if is_media_prompt(p)]
        result: list[list[Completion] | None] = [None] * len(prompts)
        with self._lock:
            self._wake(_ALL_TAGS)
            if text_idx:
                ids = [prompt_token_ids(self._tokenizer, prompts[i]) for i in text_idx]
                body: dict[str, Any] = {
                    "model": self._model_name,
                    "prompt": ids,
                    "return_token_ids": True,
                    **sampling_kwargs(params),
                }
                resp = self._post("/v1/completions", json=body)
                grouped = completions_from_choices(resp.json()["choices"], ids, params.n, want_logprobs=params.logprobs)
                for i, comps in zip(text_idx, grouped, strict=True):
                    result[i] = comps
            for i in media_idx:
                body = {
                    "model": self._model_name,
                    "messages": chat_completion_messages(prompts[i]),  # type: ignore[arg-type]
                    **chat_completion_kwargs(params),
                }
                resp = self._post("/v1/chat/completions", json=body)
                result[i] = completions_from_chat_response(resp.json(), params.n, want_logprobs=params.logprobs)
        return [r for r in result if r is not None]

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        with self._lock:
            self._wake(("weights",))
            for name, tensor in weights:
                t = tensor.detach()
                self._post(
                    "/update_named_param",
                    params={"name": name, "dtype": dtype_name(t.dtype), "shape": ",".join(str(s) for s in t.shape)},
                    content=tensor_to_bytes(t),
                    headers={"content-type": "application/octet-stream"},
                )
            self._post("/reset_prefix_cache")

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        with self._lock:
            self._wake(("weights",))
            self._post("/load_weights_from_path", json={"path": os.fspath(path)})
            self._post("/reset_prefix_cache")

    def sleep(self) -> None:
        if not self._sleep_enabled:
            return
        with self._lock:
            if self._sleeping_tags:
                return
            self._post("/sleep", params={"level": self._sleep_level})
            self._sleeping_tags = set(_ALL_TAGS)

    def wake_up(self) -> None:
        if not self._sleep_enabled:
            return
        with self._lock:
            self._wake(_ALL_TAGS)

    def shutdown(self) -> None:
        self._client.close()

    # -- internals ------------------------------------------------------------------------
    def _wake(self, tags: tuple[str, ...]) -> None:
        """Re-map only the tags currently released (``/wake_up?tags=weights&tags=kv_cache``);
        weights must be mapped before loading into them, the KV cache only before generating."""
        pending = [t for t in tags if t in self._sleeping_tags]
        if not pending:
            return
        self._post("/wake_up", params=[("tags", t) for t in pending])
        self._sleeping_tags.difference_update(pending)

    def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        resp = self._client.post(path, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"vLLM server {self._url}{path} -> {resp.status_code}: {resp.text[:500]}")
        return resp
