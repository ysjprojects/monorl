"""Pipeline fixtures: the orchestrator fakes (``tests/orchestrator/fakes.py``) wired into the
registry plus a data-source fake that also understands the chat rows written by rejection sampling."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "orchestrator"))

import fakes  # noqa: E402

from monorl.config.pipeline import PipelineConfig  # noqa: E402
from monorl.config.schema import DataConfig  # noqa: E402
from monorl.core.types import Message, Sample  # noqa: E402


class ChatAwareSource:
    """Reads prompt/answer rows (like ``fakes.fake_open_source``) and chat rows
    (``{"messages": [...]}``), stripping a trailing assistant turn into ``metadata["assistant_turn"]``
    exactly like the real data layer."""

    name = "fake-chat-aware"

    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def schema(self) -> dict[str, str]:
        return {"messages": "list"}


def chat_aware_open_source(
    data_config: DataConfig, *, split: str | None = None, tokenizer: Any = None, seed: int = 0
) -> Any:
    prompt_field = data_config.prompt_field or "prompt"
    answer_field = data_config.answer_field or "answer"
    samples: list[Sample] = []
    with open(data_config.uri) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metadata: dict[str, Any] = {"row": row}
            reference: Any = None
            if "messages" in row:
                messages: list[Message] = [dict(m) for m in row["messages"]]
                reference = row.get("reference")
                if messages and messages[-1]["role"] == "assistant":
                    trailing = messages.pop()
                    metadata["assistant_turn"] = trailing
                    if reference is None:
                        reference = trailing.get("content")
            else:
                messages = []
                if data_config.system_prompt:
                    messages.append({"role": "system", "content": data_config.system_prompt})
                messages.append({"role": "user", "content": str(row[prompt_field])})
                reference = row.get(answer_field)
            samples.append(
                Sample(id=str(row.get("id", i)), messages=messages, reference=reference, metadata=metadata)
            )
    if data_config.limit:
        samples = samples[: data_config.limit]
    return ChatAwareSource(samples)


@pytest.fixture
def fake_components(monkeypatch: pytest.MonkeyPatch) -> Any:
    from monorl.orchestrator import loop

    fakes.install_fakes(monkeypatch)
    monkeypatch.setattr(loop, "_resolve_open_source", lambda: chat_aware_open_source)
    yield fakes
    fakes.reset_recorders()


def pipeline_dict(tmp_path: Path, stages: list[dict[str, Any]], **base_overrides: Any) -> dict[str, Any]:
    base = fakes.fake_config_dict(tmp_path, **base_overrides)
    base.pop("run_name", None)
    return {"name": "fake-pipeline", "base": base, "stages": stages}


DEFAULT_STAGES: list[dict[str, Any]] = [
    {"type": "train", "name": "warmup", "init_from": "base", "overrides": {"trainer": {"max_steps": 1}}},
    {"type": "train", "name": "grpo_1", "overrides": {"trainer": {"max_steps": 2}}},
    {
        "type": "rejection_sample",
        "name": "rs",
        "num_samples": 2,
        "min_reward": 1.0,
        "max_per_prompt": 1,
        "rollout_overrides": {"temperature": 0.7},
    },
    {
        "type": "train",
        "name": "sft_like",
        "data_from": "previous",
        "overrides": {"trainer": {"max_steps": 1}},
    },
    {"type": "eval", "name": "eval", "max_prompts": 3},
]


@pytest.fixture
def make_pipeline(tmp_path: Path, fake_components: Any):
    def _make(stages: list[dict[str, Any]] | None = None, **base_overrides: Any) -> PipelineConfig:
        return PipelineConfig.from_dict(pipeline_dict(tmp_path, stages or DEFAULT_STAGES, **base_overrides))

    return _make
