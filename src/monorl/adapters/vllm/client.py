"""Server-mode vLLM rollout engine: an HTTP client for :mod:`monorl.adapters.vllm.server`.

Implements the same :class:`~monorl.core.interfaces.RolloutEngine` protocol as the colocated
engine without importing vLLM, so the trainer process can live on another machine or in an
environment where vLLM is not installable.

Endpoints used (vLLM v0.33.0 unless noted):

* ``POST /v1/completions`` (``vllm/entrypoints/openai/completion/protocol.py``): ``prompt`` as a
  list of token-id lists, ``return_token_ids=true`` → ``choices[i].token_ids`` /
  ``prompt_token_ids``; ``logprobs=0`` → ``choices[i].logprobs.token_logprobs``. With
  ``SamplingParams.top_logprobs = k``: ``logprobs=k`` and ``return_tokens_as_token_ids=true`` →
  ``choices[i].logprobs.top_logprobs`` (``{"token_id:<id>": logprob}`` per position). The server
  must allow that many: ``--max-logprobs`` ≥ ``k`` (``k + 1`` for the chat endpoint; default 20)
  and ``--logprobs-mode processed_logprobs``.
* ``POST /v1/chat/completions`` (``vllm/entrypoints/openai/chat_completion/protocol.py``) for
  prompts with image parts: ``messages`` with ``image_url`` data URLs, ``logprobs=true,
  top_logprobs=0`` and ``return_token_ids=true``. The chat endpoint's logprob entries only carry
  token *strings*, so token ids come from ``choices[i].token_ids`` and the prompt ids the server
  conditioned on (chat template + image placeholders rendered server-side) from the
  response-level ``prompt_token_ids``. One request per prompt (a chat request is one conversation).
* ``POST /sleep?level=N``, ``POST /wake_up``, ``GET /is_sleeping``
  (``vllm/entrypoints/serve/dev/sleep/api_router.py``, enabled by ``VLLM_SERVER_DEV_MODE=1``).
* ``POST /reset_prefix_cache`` (``vllm/entrypoints/serve/dev/cache/api_router.py``).
* ``POST /update_named_param``, ``POST /load_weights_from_path``,
  ``POST /monorl/finish_weight_update``, ``GET /monorl/info`` — monorl additions.
* ``GET /metrics`` (``vllm/entrypoints/serve/instrumentator/metrics.py``): Prometheus text with
  the speculative-decoding counters (``vllm/v1/spec_decode/metrics.py``), absent when the server
  runs with ``--disable-log-stats`` or does not draft.

``cancel()`` relies on the server dropping a generation whose client went away: both completion
routes are wrapped in ``with_cancellation`` (``vllm/entrypoints/serve/utils/api_utils.py``), which
cancels the handler on ``http.disconnect``, and ``AsyncLLM.generate``
(``vllm/v1/engine/async_llm.py``) aborts its request in the engine when it is cancelled. The
requests therefore run as tasks of an ``httpx.AsyncClient`` on a private event-loop thread:
cancelling a task closes its connection, which a thread blocked in a synchronous ``httpx.Client``
call could not be made to do.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
from collections.abc import Iterable, Sequence
from typing import Any

import httpx
import torch

from monorl.adapters.vllm.common import (
    SpeculationCounters,
    chat_completion_kwargs,
    chat_completion_messages,
    completions_from_chat_response,
    completions_from_choices,
    dtype_name,
    is_media_prompt,
    prometheus_samples,
    prompt_token_ids,
    sampling_kwargs,
    spec_decode_totals,
    speculative_config,
    speculative_serve_flag,
    tensor_to_bytes,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, GenerationCancelled
from monorl.core.types import Completion, PromptInput, SamplingParams, SpeculationStats
from monorl.mtp import checkpoint_key, is_wire_name

log = logging.getLogger(__name__)
_ALL_TAGS: tuple[str, ...] = ("weights", "kv_cache")


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
            "rollout.extra.mode='server' needs rollout.extra.url (or urls) of a monorl vLLM server"
        )
    return str(url).rstrip("/")


class _GenerateCall:
    """One ``generate`` call that has not returned yet; fields are guarded by the client's
    ``_calls_lock``."""

    def __init__(self) -> None:
        self.cancelled = False
        self.request: concurrent.futures.Future[httpx.Response] | None = None
        """The HTTP request the call is blocked on, if any."""


class VllmServerClient:
    """``rollout_engine: vllm`` with ``rollout.extra.mode='server'``.

    ``rollout.extra`` keys: ``url``/``urls``, ``model_name`` (served model name, defaults to
    ``model.name_or_path``), ``timeout`` seconds per request (default: none), ``sleep_level``.

    ``cancel()`` disconnects the completion requests in flight, which makes the server abort
    their generation, and fails the ``generate`` calls waiting for the lock; all of them raise
    :class:`GenerationCancelled`.

    With ``model.mtp.rollout`` the server has to draft with the model's MTP layers: the client
    checks what ``/monorl/info`` reports against the config when it is created and raises
    :class:`ConfigError` naming the ``--speculative-config`` to start the server with. MTP weights
    of an update are then sent marked for the drafter; without ``model.mtp.rollout`` they are
    left out. ``speculation_stats()`` reports the drafting since its previous call, server-wide:
    replicas sharing one server each report it, which leaves the rates derived from the sum right.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        replica: int = 0,
        transport: httpx.AsyncBaseTransport | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        extra = config.rollout.extra
        speculative = speculative_config(config)
        self._drafting = speculative is not None
        self._config = config
        self._url = server_url(config, replica)
        self._model_name = str(extra.get("model_name") or config.model.name_or_path)
        self._sleep_level = int(extra.get("sleep_level", 1))
        self._sleep_enabled = bool(config.rollout.enable_sleep_mode)
        if tokenizer is None:
            from monorl.rollout.hf_engine import load_tokenizer

            tokenizer = load_tokenizer(config)
        self._tokenizer = tokenizer
        timeout = extra.get("timeout")
        self._client = httpx.AsyncClient(
            base_url=self._url,
            timeout=httpx.Timeout(None if timeout is None else float(timeout)),
            transport=transport,
        )
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name=f"monorl-vllm-client-{replica}", daemon=True
        )
        self._loop_thread.start()
        self._lock = threading.RLock()
        self._calls: set[_GenerateCall] = set()
        self._calls_lock = threading.Lock()
        self._sleeping_tags: set[str] = set()  # vLLM memory tags currently released
        self._speculation: SpeculationCounters | None = None
        if speculative is not None:
            try:
                self._check_server_drafts(speculative)
                self._speculation = SpeculationCounters(self._spec_decode_totals())
            except BaseException:
                self.shutdown()
                raise

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
        call = _GenerateCall()
        with self._calls_lock:
            self._calls.add(call)
        try:
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
                    if params.top_logprobs > 0:
                        body["return_tokens_as_token_ids"] = True
                    resp = self._post("/v1/completions", call=call, json=body)
                    grouped = completions_from_choices(
                        resp.json()["choices"],
                        ids,
                        params.n,
                        want_logprobs=params.logprobs,
                        top_logprobs=params.top_logprobs,
                    )
                    for i, comps in zip(text_idx, grouped, strict=True):
                        result[i] = comps
                for i in media_idx:
                    body = {
                        "model": self._model_name,
                        "messages": chat_completion_messages(prompts[i]),  # type: ignore[arg-type]
                        **chat_completion_kwargs(params),
                    }
                    resp = self._post("/v1/chat/completions", call=call, json=body)
                    result[i] = completions_from_chat_response(
                        resp.json(), params.n, want_logprobs=params.logprobs, top_logprobs=params.top_logprobs
                    )
        finally:
            with self._calls_lock:
                self._calls.discard(call)
        return [r for r in result if r is not None]

    def cancel(self) -> None:
        """Abort the ``generate`` calls in flight (they raise :class:`GenerationCancelled`); a
        no-op when idle. The client stays usable: later calls run normally."""
        with self._calls_lock:
            for call in self._calls:
                call.cancelled = True
                if call.request is not None:
                    call.request.cancel()

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        with self._lock:
            self._wake(("weights",))
            for name, tensor in weights:
                t = tensor.detach()
                params = {
                    "name": name,
                    "dtype": dtype_name(t.dtype),
                    "shape": ",".join(str(s) for s in t.shape),
                }
                if is_wire_name(name):  # an MTP weight: to the drafter, under its checkpoint key
                    if not self._drafting:
                        continue  # the server does not draft: MTP layers are not its business
                    params["name"] = checkpoint_key(name)
                    params["mtp"] = "true"
                self._post(
                    "/update_named_param",
                    params=params,
                    content=tensor_to_bytes(t),
                    headers={"content-type": "application/octet-stream"},
                )
            if self._drafting:
                self._post("/monorl/finish_weight_update")
            self._post("/reset_prefix_cache")

    def speculation_stats(self) -> SpeculationStats | None:
        """Drafting on the server since the previous call (reading resets it); ``None`` when
        ``model.mtp.rollout`` is off."""
        if self._speculation is None:
            return None
        return self._speculation.delta(self._spec_decode_totals())

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
        if self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._client.aclose(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._loop.close()

    # -- internals ------------------------------------------------------------------------
    def _check_server_drafts(self, speculative: dict[str, Any]) -> None:
        """Raise :class:`ConfigError` unless every worker of the server drafts the way
        ``model.mtp.rollout`` compiles to (``speculative``)."""
        hint = f"start the server with {speculative_serve_flag(speculative)}"
        workers = self._request("GET", "/monorl/info").json().get("workers") or []
        for worker in workers or [{}]:
            served = worker.get("speculative")
            if not served:
                raise ConfigError(
                    f"model.mtp.rollout is set but the vLLM server {self._url} does not draft; {hint}"
                )
            method = str(served.get("method"))
            if not (method == "mtp" or method.endswith("_mtp")):
                raise ConfigError(
                    f"model.mtp.rollout is set but the vLLM server {self._url} drafts with method "
                    f"{method!r}, not with the model's MTP layers; {hint}"
                )
            if served.get("rejection_sample_method", "standard") != "standard":
                raise ConfigError(
                    f"the vLLM server {self._url} runs rejection_sample_method="
                    f"{served['rejection_sample_method']!r}, under which the recorded logprobs are not those "
                    f"of the distribution the tokens are sampled from; {hint}"
                )
            wanted = self._config.model.mtp.num_speculative_tokens
            if wanted is not None and served.get("num_speculative_tokens") != wanted:
                raise ConfigError(
                    f"model.mtp.num_speculative_tokens={wanted} but the vLLM server {self._url} drafts "
                    f"{served.get('num_speculative_tokens')} tokens per step; {hint}"
                )

    def _spec_decode_totals(self) -> tuple[int, int, int]:
        totals = spec_decode_totals(prometheus_samples(self._request("GET", "/metrics").text))
        if totals is None:
            raise ConfigError(
                f"the vLLM server {self._url} exposes no speculative-decoding counters on /metrics; "
                "start it without --disable-log-stats"
            )
        return totals

    def _wake(self, tags: tuple[str, ...]) -> None:
        """Re-map only the tags currently released (``/wake_up?tags=weights&tags=kv_cache``);
        weights must be mapped before loading into them, the KV cache only before generating."""
        pending = [t for t in tags if t in self._sleeping_tags]
        if not pending:
            return
        self._post("/wake_up", params=[("tags", t) for t in pending])
        self._sleeping_tags.difference_update(pending)

    def _post(self, path: str, *, call: _GenerateCall | None = None, **kwargs: Any) -> httpx.Response:
        return self._request("POST", path, call=call, **kwargs)

    def _request(
        self, method: str, path: str, *, call: _GenerateCall | None = None, **kwargs: Any
    ) -> httpx.Response:
        """Send a request on the client's event loop and wait for the response. With ``call`` the
        request belongs to that ``generate`` call and :meth:`cancel` can abandon it."""
        with self._calls_lock:
            if call is not None and call.cancelled:
                raise GenerationCancelled("generation cancelled")
            request = asyncio.run_coroutine_threadsafe(
                self._client.request(method, path, **kwargs), self._loop
            )
            if call is not None:
                call.request = request
        try:
            resp = request.result()
        except concurrent.futures.CancelledError:  # only cancel() cancels a request
            raise GenerationCancelled("generation cancelled") from None
        except BaseException:
            request.cancel()  # interrupted while waiting (KeyboardInterrupt): drop the request too
            raise
        if resp.status_code >= 400:
            raise RuntimeError(f"vLLM server {self._url}{path} -> {resp.status_code}: {resp.text[:500]}")
        return resp
