"""Agent conversations as chat rows: tool results, context turns and a ``tools`` column read the
same from jsonl, parquet, arrow and Hub datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.core.types import Sample
from monorl.data import open_source
from tests.data.conftest import write_arrow, write_jsonl, write_parquet

CALCULATOR = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate an expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. 2+2"}},
            "required": ["expression"],
        },
    },
}
LOOKUP = {
    "type": "function",
    "function": {
        "name": "lookup",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        "strict": True,
    },
}
CALL = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call>'

AGENT_ROWS: list[dict[str, Any]] = [
    {
        "messages": [
            {"role": "system", "content": "Use the tools."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": CALL},
            {"role": "tool", "name": "calculator", "tool_call_id": "call_1", "content": "4"},
            {"role": "assistant", "content": "It is 4."},
        ],
        "tools": [CALCULATOR],
        "reference": "4",
        "num_turns": 2,
    },
    {
        "messages": [
            {"role": "user", "content": "Capital of France?"},
            {"role": "assistant", "content": "Let me check.", "weight": 0},
            {"role": "user", "content": "Go on."},
            {"role": "assistant", "content": "Paris."},
        ],
        "tools": [LOOKUP, CALCULATOR],
        "reference": "Paris",
        "num_turns": 1,
    },
    {
        "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}],
        "reference": "Hello.",
        "num_turns": 1,
    },
]


def _content(samples: list[Sample]) -> list[dict[str, Any]]:
    """What SFT consumes of a sample (row ids name the file, so they differ by design)."""
    return [
        {
            "messages": s.messages,
            "tools": s.metadata.get("tools"),
            "target": s.metadata["assistant_turn"],
            "num_turns": s.metadata["row"]["num_turns"],
            "tools_left_in_row": "tools" in s.metadata["row"] and s.metadata["row"]["tools"] is not None,
        }
        for s in samples
    ]


def _read(uri: str | Path, **kw: Any) -> list[Sample]:
    return list(open_source(DataConfig(uri=str(uri), shuffle=False, **kw)))


def test_agent_rows_read_the_same_from_every_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from_jsonl = _content(_read(write_jsonl(tmp_path / "rows.jsonl", AGENT_ROWS)))
    assert (
        from_jsonl
        == [
            {
                "messages": AGENT_ROWS[0]["messages"][:-1],  # the tool result keeps its name and call id
                "tools": [CALCULATOR],
                "target": {"role": "assistant", "content": "It is 4."},
                "num_turns": 2,
                "tools_left_in_row": False,
            },
            {
                "messages": AGENT_ROWS[1]["messages"][:-1],  # ... and the context turn its weight
                "tools": [LOOKUP, CALCULATOR],
                "target": {"role": "assistant", "content": "Paris."},
                "num_turns": 1,
                "tools_left_in_row": False,
            },
            {
                "messages": [{"role": "user", "content": "Hi"}],
                "tools": None,
                "target": {"role": "assistant", "content": "Hello."},
                "num_turns": 1,
                "tools_left_in_row": False,
            },
        ]
    )
    # columnar stores give every row one struct type and fill what a row lacks with nulls
    assert _content(_read(write_parquet(tmp_path / "rows.parquet", AGENT_ROWS))) == from_jsonl
    assert _content(_read(write_arrow(tmp_path / "rows.arrow", AGENT_ROWS))) == from_jsonl

    import datasets

    hub = datasets.Dataset.from_list(AGENT_ROWS)
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: hub)
    assert _content(_read("hf://someone/agent-episodes")) == from_jsonl

    # tool schemas as a JSON string column (how CSV and many Hub datasets keep them)
    as_strings = [{**row, "tools": json.dumps(row["tools"])} if "tools" in row else row for row in AGENT_ROWS]
    assert _content(_read(write_jsonl(tmp_path / "strings.jsonl", as_strings))) == from_jsonl
    assert _content(_read(write_parquet(tmp_path / "strings.parquet", as_strings))) == from_jsonl


def test_tools_field_names_the_column_and_is_validated(tmp_path: Path) -> None:
    renamed = [
        {"question": "What is 2+2?", "answer": "4", "functions": [CALCULATOR]},
        {"question": "What is 3+3?", "answer": "6", "functions": None},
    ]
    first, second = _read(write_jsonl(tmp_path / "qa.jsonl", renamed), tools_field="functions")
    assert first.metadata["tools"] == [CALCULATOR] and "functions" not in first.metadata["row"]
    assert "tools" not in second.metadata
    with pytest.raises(DataError, match="'nope' not found"):
        _read(tmp_path / "qa.jsonl", tools_field="nope")

    # a chat dataset's own `tools` column that does not hold schemas is left alone ...
    names = [{"messages": [{"role": "user", "content": "q"}], "tools": ["search", "browser"]}]
    path = write_jsonl(tmp_path / "names.jsonl", names)
    (sample,) = _read(path)
    assert "tools" not in sample.metadata and sample.metadata["row"]["tools"] == ["search", "browser"]
    # ... unless the config says it does
    with pytest.raises(DataError, match="list of tool schema objects"):
        _read(path, tools_field="tools")
    with pytest.raises(DataError, match="not JSON"):
        _read(
            write_jsonl(tmp_path / "bad.jsonl", [{"messages": names[0]["messages"], "tools": "[{"}]),
            tools_field="tools",
        )


class _ToolAwareTokenizer:
    """One token per character; the chat template spells the tool schemas out before the chat."""

    chat_template = "tools-then-chat"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: Any = None,
    ) -> str:
        return (json.dumps(tools) if tools else "") + "".join(m["content"] for m in messages)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) for c in text]}


def test_prompt_token_filter_counts_the_tool_schemas(tmp_path: Path) -> None:
    rows = [
        {"messages": [{"role": "user", "content": "short"}], "tools": [CALCULATOR]},
        {"messages": [{"role": "user", "content": "short"}]},
    ]
    source = open_source(
        DataConfig(uri=str(write_jsonl(tmp_path / "rows.jsonl", rows)), shuffle=False, max_prompt_tokens=50),
        tokenizer=_ToolAwareTokenizer(),
    )
    (kept,) = list(source)
    assert "tools" not in kept.metadata and kept.metadata["prompt_tokens"] == len("short")
    assert source.mapper.dropped == 1  # the prompt the model would see includes the schemas
