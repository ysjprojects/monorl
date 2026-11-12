from __future__ import annotations

import pytest

from monorl.data import sniff_format
from monorl.data.formats import infer_columns
from monorl.data.sources import extract_reference


def _sniff(rows):
    return sniff_format(infer_columns(rows), rows)


def test_chat_messages_column():
    rows = [{"messages": [{"role": "user", "content": "hi"}], "answer": "x"}]
    assert _sniff(rows) == {"format": "chat", "messages_field": "messages", "prompt_field": None, "answer_field": "answer", "image_fields": []}


@pytest.mark.parametrize("col", ["conversations", "conversation", "Messages"])
def test_chat_aliases_and_sharegpt(col):
    rows = [{col: [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}]
    out = _sniff(rows)
    assert out["format"] == "chat" and out["messages_field"] == col


def test_messages_column_that_is_not_chat_falls_through():
    rows = [{"messages": "plain string", "prompt": "p", "completion": "c"}]
    out = _sniff(rows)
    assert out == {"format": "prompt_answer", "prompt_field": "prompt", "answer_field": "completion", "messages_field": None, "image_fields": []}


def test_trl_conversational_prompt_column_is_chat():
    rows = [{"prompt": [{"role": "user", "content": "hi"}], "label": 1}]
    out = _sniff(rows)
    assert out["format"] == "chat" and out["messages_field"] == "prompt" and out["answer_field"] == "label"


@pytest.mark.parametrize(
    ("prompt", "answer"),
    [("question", "answer"), ("instruction", "output"), ("problem", "solution"), ("query", "response"), ("input", "target")],
)
def test_prompt_answer_pairs(prompt, answer):
    out = _sniff([{prompt: "q", answer: "a", "misc": 1}])
    assert out == {"format": "prompt_answer", "prompt_field": prompt, "answer_field": answer, "messages_field": None, "image_fields": []}


def test_prompt_without_answer_is_text():
    assert _sniff([{"prompt": "q", "n": 3}])["format"] == "text"
    out = _sniff([{"text": "just text"}])
    assert out["format"] == "text" and out["prompt_field"] == "text"


def test_single_string_column_fallback_and_unknown():
    out = _sniff([{"sentence": "hello", "score": 0.5}])
    assert out["format"] == "text" and out["prompt_field"] == "sentence"
    assert _sniff([{"a": 1, "b": 2.0}])["format"] == "unknown"
    assert sniff_format({}, [])["format"] == "unknown"


def test_gsm8k_extraction():
    assert extract_reference("Natalia sold 48/2 = 24 clips.\n#### 72") == ("72", True)
    assert extract_reference("x #### 1,000") == ("1000", True)
    assert extract_reference("#### 3.5") == ("3.5", True)
    assert extract_reference("no marker") == ("no marker", False)
    assert extract_reference(42) == (42, False)
    assert extract_reference("ends with #### ") == ("ends with #### ", False)


def test_infer_columns_merges_types():
    cols = infer_columns([{"a": 1, "b": None, "c": [1, 2]}, {"a": 2.5, "b": "s", "c": ["x"]}, {"d": {"k": 1}}])
    assert cols == {"a": "double", "b": "string", "c": "list", "d": "struct"}
