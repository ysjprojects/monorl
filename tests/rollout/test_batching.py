from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest

from monorl.core.errors import GenerationCancelled
from monorl.core.interfaces import RolloutEngine
from monorl.core.types import Completion, PromptInput, SamplingParams
from monorl.rollout import BatchingEngine, params_key


class FakeEngine:
    """Echoes each prompt as text; records every call (prompt count + params)."""

    def __init__(self, delay: float = 0.02, fail_on: str | None = None) -> None:
        self.calls: list[tuple[int, SamplingParams]] = []
        self.delay = delay
        self.fail_on = fail_on
        self.lock = threading.Lock()
        self.events: list[str] = []
        self.tokenizer = object()

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        with self.lock:
            self.calls.append((len(prompts), params))
        time.sleep(self.delay)
        out = []
        for p in prompts:
            text = p if isinstance(p[0], int) else p[0]["content"]
            if self.fail_on is not None and text == self.fail_on:
                raise RuntimeError(f"boom {text}")
            out.append([Completion(text=f"{text}#{k}", token_ids=[k]) for k in range(params.n)])
        return out

    def update_weights(self, weights) -> None:
        self.events.append("update_weights")

    def load_weights_from_path(self, path) -> None:
        self.events.append(f"load:{path}")

    def sleep(self) -> None:
        self.events.append("sleep")

    def wake_up(self) -> None:
        self.events.append("wake_up")

    def shutdown(self) -> None:
        self.events.append("shutdown")


def chat(text: str) -> PromptInput:
    return [{"role": "user", "content": text}]


def test_is_rollout_engine() -> None:
    proxy = BatchingEngine(FakeEngine(), max_batch_size=4, max_wait_ms=10)
    assert isinstance(proxy, RolloutEngine)
    proxy.shutdown()


def test_coalesces_concurrent_callers() -> None:
    fake = FakeEngine(delay=0.02)
    proxy = BatchingEngine(fake, max_batch_size=8, max_wait_ms=100)
    params = SamplingParams(n=2, max_new_tokens=4)
    callers = 8

    def call(i: int):
        return proxy.generate([chat(f"p{i}")], params)

    with ThreadPoolExecutor(callers) as pool:
        results = list(pool.map(call, range(callers)))
    proxy.shutdown()
    for i, res in enumerate(results):
        assert len(res) == 1 and [c.text for c in res[0]] == [f"p{i}#0", f"p{i}#1"]
    assert len(fake.calls) < callers
    assert sum(n for n, _ in fake.calls) == callers
    assert proxy.num_engine_calls == len(fake.calls)


def test_preserves_order_within_call_and_chunks_oversized_requests() -> None:
    fake = FakeEngine(delay=0.0)
    proxy = BatchingEngine(fake, max_batch_size=2, max_wait_ms=0)
    prompts = [chat(f"q{i}") for i in range(5)]
    out = proxy.generate(prompts, SamplingParams(n=1))
    proxy.shutdown()
    assert [g[0].text for g in out] == [f"q{i}#0" for i in range(5)]
    assert [n for n, _ in fake.calls] == [2, 2, 1]


def test_different_params_are_not_mixed() -> None:
    fake = FakeEngine(delay=0.02)
    proxy = BatchingEngine(fake, max_batch_size=8, max_wait_ms=100)
    hot = SamplingParams(n=1, temperature=1.0, stop=["</a>"])
    cold = SamplingParams(n=1, temperature=0.3, stop=["</a>"])
    assert params_key(hot) != params_key(cold)
    assert params_key(hot) == params_key(SamplingParams(n=1, temperature=1.0, stop=["</a>"]))

    def call(i: int):
        return proxy.generate([chat(f"p{i}")], hot if i % 2 == 0 else cold)

    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(call, range(6)))
    proxy.shutdown()
    assert [r[0][0].text for r in results] == [f"p{i}#0" for i in range(6)]
    assert len(fake.calls) == 2
    assert {params_key(p) for _, p in fake.calls} == {params_key(hot), params_key(cold)}
    assert sorted(n for n, _ in fake.calls) == [3, 3]


def test_errors_propagate_to_callers_and_queue_recovers() -> None:
    fake = FakeEngine(delay=0.0, fail_on="bad")
    proxy = BatchingEngine(fake, max_batch_size=4, max_wait_ms=0)
    with pytest.raises(RuntimeError, match="boom bad"):
        proxy.generate([chat("ok"), chat("bad")], SamplingParams(n=1))
    out = proxy.generate([chat("fine")], SamplingParams(n=1))
    assert out[0][0].text == "fine#0"
    proxy.shutdown()


def test_forwards_lifecycle_calls() -> None:
    fake = FakeEngine()
    proxy = BatchingEngine(fake, max_batch_size=4, max_wait_ms=0)
    assert proxy.tokenizer is fake.tokenizer
    proxy.update_weights([])
    proxy.load_weights_from_path("/tmp/x")
    proxy.sleep()
    proxy.wake_up()
    proxy.shutdown()
    assert fake.events == ["update_weights", "load:/tmp/x", "sleep", "wake_up", "shutdown"]
    with pytest.raises(RuntimeError):
        proxy.generate([chat("late")], SamplingParams())


# -- cancel ------------------------------------------------------------------------------------


class BlockingEngine(FakeEngine):
    """``generate`` blocks until ``release`` is set; ``fail_first`` makes the first call raise then."""

    def __init__(self, fail_first: bool = False) -> None:
        super().__init__(delay=0.0)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_first = fail_first
        self.started = 0

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        self.started += 1
        first = self.started == 1
        self.entered.set()
        if not self.release.wait(10):
            raise TimeoutError("BlockingEngine was never released")
        if first and self.fail_first:
            raise RuntimeError("old call failed")
        return super().generate(prompts, params)


class CancellableEngine(FakeEngine):
    """``generate`` blocks until ``cancel()`` and then raises, like an engine implementing the
    optional ``cancel`` capability; with ``block = False`` it answers at once."""

    def __init__(self) -> None:
        super().__init__(delay=0.0)
        self.entered = threading.Event()
        self.block = True
        self.cancel_calls = 0
        self._cancelled = threading.Event()

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        if self.block:
            self._cancelled.clear()
            self.entered.set()
            if not self._cancelled.wait(10):
                raise TimeoutError("CancellableEngine was never cancelled")
            raise GenerationCancelled("engine call cancelled")
        return super().generate(prompts, params)

    def cancel(self) -> None:
        self.cancel_calls += 1
        self._cancelled.set()


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.002)


def _call_in_threads(proxy: BatchingEngine, requests: list[list[PromptInput]], params: SamplingParams):
    outcomes: dict[int, object] = {}

    def call(i: int) -> None:
        try:
            outcomes[i] = proxy.generate(requests[i], params)
        except BaseException as e:  # noqa: BLE001 - recorded for the assertions
            outcomes[i] = e

    threads = [threading.Thread(target=call, args=(i,), daemon=True) for i in range(len(requests))]
    for t in threads:
        t.start()
    return threads, outcomes


def _shutdown_returns(proxy: BatchingEngine) -> bool:
    t = threading.Thread(target=proxy.shutdown, daemon=True)
    t.start()
    t.join(5)
    return not t.is_alive()


@pytest.mark.parametrize("old_call_fails", [False, True])
def test_cancel_fails_queued_and_in_flight_callers_while_engine_is_busy(old_call_fails: bool) -> None:
    fake = BlockingEngine(fail_first=old_call_fails)
    proxy = BatchingEngine(fake, max_batch_size=2, max_wait_ms=5)
    params = SamplingParams(n=2, max_new_tokens=4)
    in_flight, served = _call_in_threads(proxy, [[chat("old0"), chat("old1")]], params)
    assert fake.entered.wait(5)  # a full batch is inside the engine ...
    queued, waiting = _call_in_threads(proxy, [[chat("old2")], [chat("old3")]], params)
    _wait_until(lambda: len(proxy._queue) == 2)  # ... and two more callers are queued behind it

    t0 = time.monotonic()
    proxy.cancel()
    for t in in_flight + queued:
        t.join(1.0)
    assert not any(t.is_alive() for t in in_flight + queued)
    assert time.monotonic() - t0 < 1.0
    outcomes = [served.get(0), waiting.get(0), waiting.get(1)]
    assert all(isinstance(o, GenerationCancelled) for o in outcomes), outcomes
    assert fake.started == 1 and not fake.release.is_set()  # the engine call is still blocked

    # A request made after the cancel: the old call's two results (or its error) are for prompts
    # nobody waits on any more and must not be routed to it.
    new_threads, new = _call_in_threads(proxy, [[chat("new0"), chat("new1")]], params)
    _wait_until(lambda: len(proxy._queue) == 2)
    fake.release.set()
    new_threads[0].join(5)
    assert not new_threads[0].is_alive()
    assert not isinstance(new[0], BaseException), new[0]
    assert [[c.text for c in group] for group in new[0]] == [["new0#0", "new0#1"], ["new1#0", "new1#1"]]
    assert fake.started == 2  # the cancelled queued prompts never reached the engine
    assert _shutdown_returns(proxy)
    assert fake.events == ["shutdown"]


def test_cancel_forwards_to_engine_cancel_and_worker_keeps_serving() -> None:
    fake = CancellableEngine()
    proxy = BatchingEngine(fake, max_batch_size=4, max_wait_ms=0)
    params = SamplingParams(n=1)
    proxy.cancel()  # idle: nothing to fail
    assert fake.cancel_calls == 1
    threads, outcomes = _call_in_threads(proxy, [[chat("a"), chat("b")]], params)
    assert fake.entered.wait(5)
    proxy.cancel()
    threads[0].join(1.0)
    assert isinstance(outcomes.get(0), GenerationCancelled)
    assert fake.cancel_calls == 2

    fake.block = False
    out = proxy.generate([chat("c")], params)
    assert [c.text for c in out[0]] == ["c#0"]
    assert [n for n, _ in fake.calls] == [1]
    assert _shutdown_returns(proxy)


def test_cancel_during_batching_window_drops_the_waiting_prompts() -> None:
    fake = FakeEngine(delay=0.0)
    proxy = BatchingEngine(fake, max_batch_size=2, max_wait_ms=3000)
    params = SamplingParams(n=1)
    threads, outcomes = _call_in_threads(proxy, [[chat("lonely")]], params)
    _wait_until(lambda: len(proxy._queue) == 1)
    time.sleep(0.05)  # let the worker open its batching window on that prompt
    t0 = time.monotonic()
    proxy.cancel()
    threads[0].join(1.0)
    assert isinstance(outcomes.get(0), GenerationCancelled)

    out = proxy.generate([chat("x"), chat("y")], params)  # a full batch: served without the window
    assert time.monotonic() - t0 < 2.0
    assert [g[0].text for g in out] == ["x#0", "y#0"]
    assert [n for n, _ in fake.calls] == [2]  # "lonely" was never sent to the engine
    assert _shutdown_returns(proxy)
