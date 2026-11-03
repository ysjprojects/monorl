"""Dynamic batching proxy: many episode threads call ``generate`` concurrently; one engine call
serves a whole batch.

Requests are split into per-prompt items. A worker thread takes the oldest item, waits up to
``max_wait_ms`` for more items with the *same* :class:`SamplingParams` (up to ``max_batch_size``
prompts), runs ``engine.generate`` once and routes the ``n`` completions of each prompt back to
its caller in the caller's original order. Items with different params are never mixed; they are
served in subsequent engine calls.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from monorl.core.types import Completion, PromptInput, SamplingParams

if TYPE_CHECKING:
    import torch

    from monorl.core.interfaces import RolloutEngine


def params_key(params: SamplingParams) -> tuple[Any, ...]:
    """Hashable identity of a :class:`SamplingParams` (lists become tuples)."""
    out = []
    for f in fields(params):
        v = getattr(params, f.name)
        out.append(tuple(v) if isinstance(v, list) else v)
    return tuple(out)


@dataclass
class _Request:
    params: SamplingParams
    results: list[list[Completion] | None]
    remaining: int
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


@dataclass(slots=True)
class _Item:
    request: _Request
    index: int
    prompt: PromptInput
    key: tuple[Any, ...]


class BatchingEngine:
    """Wraps any :class:`RolloutEngine`; see module docstring."""

    def __init__(self, engine: RolloutEngine, *, max_batch_size: int, max_wait_ms: float) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait = max(0.0, max_wait_ms) / 1000.0
        self._queue: deque[_Item] = deque()
        self._cv = threading.Condition()
        self._engine_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._closed = False
        self.num_engine_calls = 0

    # -- RolloutEngine --------------------------------------------------------------------
    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        if not prompts:
            return []
        req = _Request(params=params, results=[None] * len(prompts), remaining=len(prompts))
        key = params_key(params)
        with self._cv:
            if self._closed:
                raise RuntimeError("BatchingEngine is shut down")
            for i, p in enumerate(prompts):
                self._queue.append(_Item(req, i, p, key))
            self._ensure_worker()
            self._cv.notify_all()
        req.done.wait()
        if req.error is not None:
            raise req.error
        return [r if r is not None else [] for r in req.results]

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        with self._engine_lock:
            self.engine.update_weights(weights)

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        with self._engine_lock:
            self.engine.load_weights_from_path(path)

    def sleep(self) -> None:
        with self._engine_lock:
            self.engine.sleep()

    def wake_up(self) -> None:
        with self._engine_lock:
            self.engine.wake_up()

    def shutdown(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        with self._engine_lock:
            self.engine.shutdown()

    @property
    def tokenizer(self) -> Any:
        return self.engine.tokenizer

    # -- worker ---------------------------------------------------------------------------
    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="monorl-batching", daemon=True)
            self._worker.start()

    def _collect(self) -> list[_Item] | None:
        """Block until a batch is ready (or the proxy is closed). Called without the lock."""
        with self._cv:
            while not self._queue and not self._closed:
                self._cv.wait()
            if not self._queue:
                return None
            head = self._queue[0]
            deadline = time.monotonic() + self.max_wait
            while self._count(head.key) < self.max_batch_size:
                left = deadline - time.monotonic()
                if left <= 0 or self._closed:
                    break
                self._cv.wait(timeout=left)
            batch: list[_Item] = []
            keep: deque[_Item] = deque()
            for item in self._queue:
                if item.key == head.key and len(batch) < self.max_batch_size:
                    batch.append(item)
                else:
                    keep.append(item)
            self._queue = keep
            return batch

    def _count(self, key: tuple[Any, ...]) -> int:
        return sum(1 for item in self._queue if item.key == key)

    def _run(self) -> None:
        while True:
            batch = self._collect()
            if batch is None:
                return
            self._serve(batch)

    def _serve(self, batch: list[_Item]) -> None:
        requests = {id(item.request): item.request for item in batch}
        try:
            with self._engine_lock:
                self.num_engine_calls += 1
                outputs = self.engine.generate([item.prompt for item in batch], batch[0].request.params)
            if len(outputs) != len(batch):
                raise RuntimeError(f"engine returned {len(outputs)} results for {len(batch)} prompts")
        except BaseException as e:  # noqa: BLE001 - propagate to every waiting caller
            for req in requests.values():
                req.error = e
                req.done.set()
            # Drop the rest of this request's items so no caller waits forever on a failed request.
            with self._cv:
                self._queue = deque(item for item in self._queue if id(item.request) not in requests)
            return
        for item, completions in zip(batch, outputs, strict=True):
            item.request.results[item.index] = completions
        for req in requests.values():
            req.remaining -= sum(1 for item in batch if item.request is req)
            if req.remaining == 0:
                req.done.set()
