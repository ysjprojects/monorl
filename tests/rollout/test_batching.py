from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import pytest

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
