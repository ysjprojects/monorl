"""Fixtures for the bench tests: a deterministic fake rollout engine registered under a test-only
name, a MEP/1 stdio server script, and a typer CliRunner."""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from monorl.core.registry import registry
from monorl.core.types import Completion, PromptInput, SamplingParams

MEP_SERVER = Path(__file__).parent / "mep_bench_server.py"
FAKE_ENGINE_NAME = "bench-fake"


class FakeTokenizer:
    """Whitespace tokenizer: enough for ``open_source``'s ``max_prompt_tokens`` filter."""

    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [abs(hash(w)) % 1000 for w in text.split()]}


class FakeEngine:
    """Returns ``tokens_per_completion`` tokens per completion and sleeps ``seconds_per_call``
    (plus ``seconds_per_prompt`` per prompt) so throughput numbers are predictable."""

    def __init__(
        self,
        *,
        tokens_per_completion: int = 5,
        seconds_per_call: float = 0.0,
        seconds_per_prompt: float = 0.0,
    ) -> None:
        self.tokens_per_completion = tokens_per_completion
        self.seconds_per_call = seconds_per_call
        self.seconds_per_prompt = seconds_per_prompt
        self.calls: list[int] = []
        self.shutdowns = 0
        self._tok = FakeTokenizer()

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        self.calls.append(len(prompts))
        delay = self.seconds_per_call + self.seconds_per_prompt * len(prompts)
        if delay:
            time.sleep(delay)
        out: list[list[Completion]] = []
        for p in prompts:
            prompt_ids = list(p) if p and isinstance(p[0], int) else [7] * (len(p) + 3)
            out.append(
                [
                    Completion(
                        text="x " * self.tokens_per_completion,
                        token_ids=list(range(self.tokens_per_completion)),
                        logprobs=[-0.1] * self.tokens_per_completion,
                        finish_reason="stop",
                        prompt_token_ids=prompt_ids,
                    )
                    for _ in range(params.n)
                ]
            )
        return out

    def update_weights(self, weights: Any) -> None:
        for _ in weights:
            pass

    def load_weights_from_path(self, path: Any) -> None:
        pass

    def sleep(self) -> None:
        pass

    def wake_up(self) -> None:
        pass

    def shutdown(self) -> None:
        self.shutdowns += 1

    @property
    def tokenizer(self) -> Any:
        return self._tok


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def registered_fake_engine() -> FakeEngine:
    """Registers ``rollout_engine:bench-fake`` for the duration of a test."""
    engine = FakeEngine()
    registry.register("rollout_engine", FAKE_ENGINE_NAME, lambda config, **kw: engine, overwrite=True)
    yield engine
    registry._items["rollout_engine"].pop(FAKE_ENGINE_NAME, None)


@pytest.fixture
def mep_command() -> list[str]:
    return [sys.executable, str(MEP_SERVER)]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def chat_prompts() -> list[PromptInput]:
    return [[{"role": "user", "content": f"q{i}"}] for i in range(10)]
