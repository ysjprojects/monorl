from __future__ import annotations

from pathlib import Path

import pytest

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.data import open_source
from tests.data.conftest import QA_ROWS, write_jsonl


def _open(path: Path, **kw):
    kw.setdefault("shuffle", False)
    return open_source(DataConfig(uri=str(path), **kw))


def test_prompt_template_and_system_prompt(qa_jsonl: Path) -> None:
    src = _open(
        qa_jsonl,
        prompt_template="Q: {prompt}\nDifficulty {difficulty}. Think step by step.",
        system_prompt="You are a math tutor.",
    )
    s = src[2]
    assert s.messages == [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "Q: What is 2+2?\nDifficulty 2. Think step by step."},
    ]
    assert s.reference == "4"
    assert s.metadata["answer_raw"] == "Add them. #### 4"
    assert s.metadata["row"] == {"answer": "Add them. #### 4", "difficulty": 2}


def test_prompt_template_can_reference_the_prompt_column_by_name(qa_jsonl: Path) -> None:
    src = _open(qa_jsonl, prompt_template="{question} Answer briefly.")
    assert src[0].messages[-1]["content"] == "What is 0+0? Answer briefly."


def test_prompt_template_missing_column_is_a_data_error(qa_jsonl: Path) -> None:
    with pytest.raises(DataError, match="prompt_template references {nope}"):
        _open(qa_jsonl, prompt_template="{nope}")


def test_chat_rows_strip_trailing_assistant_and_override_system(chat_jsonl: Path) -> None:
    src = _open(chat_jsonl, system_prompt="New system.")
    s = src[1]
    assert s.id == "chat-1"
    assert s.messages == [
        {"role": "system", "content": "New system."},
        {"role": "user", "content": "Say the number 1."},
    ]
    assert s.reference == "1"
    assert s.metadata["assistant_turn"] == {"role": "assistant", "content": "1"}
    assert src.layout.format == "chat"
    plain = _open(chat_jsonl)
    assert plain[1].messages[0] == {"role": "system", "content": "You are terse."}


def test_chat_template_applies_to_last_user_turn(chat_jsonl: Path) -> None:
    src = _open(chat_jsonl, prompt_template="{prompt} (id={id})")
    assert src[3].messages[-1] == {"role": "user", "content": "Say the number 3. (id=chat-3)"}


def test_sharegpt_and_json_string_messages(tmp_path: Path) -> None:
    rows = [
        {
            "conversations": [
                {"from": "system", "value": "sys"},
                {"from": "human", "value": "hi"},
                {"from": "gpt", "value": "yo"},
            ]
        },
        {"conversations": '[{"role": "user", "content": "json-encoded"}]'},
    ]
    src = _open(write_jsonl(tmp_path / "sg.jsonl", rows))
    assert src[0].messages == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert src[0].reference == "yo"
    assert src[1].messages == [{"role": "user", "content": "json-encoded"}]
    assert src[1].reference is None


def test_explicit_fields_override_sniffing(tmp_path: Path) -> None:
    rows = [{"question": "q", "answer": "a", "hint": "h", "gold": "g"} for _ in range(3)]
    p = write_jsonl(tmp_path / "x.jsonl", rows)
    src = _open(p, prompt_field="hint", answer_field="gold")
    assert src.layout.format == "prompt_answer"
    assert src[0].messages == [{"role": "user", "content": "h"}] and src[0].reference == "g"
    src = _open(p, format="text", prompt_field="hint")
    assert src[0].reference == "a", "sniffed answer column still provides the reference"
    with pytest.raises(DataError, match="not found"):
        _open(p, prompt_field="missing")
    with pytest.raises(DataError, match="needs messages_field"):
        _open(p, format="chat")


def test_explicit_messages_field_implies_chat(tmp_path: Path) -> None:
    rows = [{"dialog": [{"role": "user", "content": "x"}], "question": "ignored"}]
    src = _open(write_jsonl(tmp_path / "d.jsonl", rows), messages_field="dialog")
    assert src.layout.format == "chat" and src[0].messages == [{"role": "user", "content": "x"}]


def test_unknown_layout_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="could not infer"):
        _open(write_jsonl(tmp_path / "n.jsonl", [{"a": 1, "b": 2}]))


def test_stable_ids_and_id_column(qa_jsonl: Path, chat_jsonl: Path) -> None:
    ids = [s.id for s in _open(qa_jsonl)]
    assert ids[0] == f"{qa_jsonl}#0" and ids[-1] == f"{qa_jsonl}#{len(QA_ROWS) - 1}"
    shuffled = {s.id: s.messages for s in _open(qa_jsonl, shuffle=True)}
    assert shuffled == {s.id: s.messages for s in _open(qa_jsonl)}, "ids follow rows, not positions"
    assert [s.id for s in _open(chat_jsonl)][:2] == ["chat-0", "chat-1"]


def test_shuffle_and_limit_are_deterministic(qa_jsonl: Path) -> None:
    a = [s.id for s in open_source(DataConfig(uri=str(qa_jsonl), shuffle=True), seed=7)]
    b = [s.id for s in open_source(DataConfig(uri=str(qa_jsonl), shuffle=True), seed=7)]
    c = [s.id for s in open_source(DataConfig(uri=str(qa_jsonl), shuffle=True), seed=8)]
    plain = [s.id for s in _open(qa_jsonl)]
    assert a == b and sorted(a) == sorted(plain) and a != plain and c != a
    limited = open_source(DataConfig(uri=str(qa_jsonl), shuffle=True, limit=4), seed=7)
    assert len(limited) == 4 and [s.id for s in limited] == a[:4]
    assert [s.id for s in _open(qa_jsonl, limit=3)] == plain[:3]


def test_streaming_shuffle_is_deterministic(qa_jsonl: Path) -> None:
    cfg = DataConfig(uri=str(qa_jsonl), shuffle=True, streaming=True, limit=5)
    a = [s.id for s in open_source(cfg, seed=1)]
    b = [s.id for s in open_source(cfg, seed=1)]
    assert a == b and len(a) == 5 and len(set(a)) == 5


def test_max_prompt_tokens_filter_with_tokenizer(tmp_path: Path, tiny_tokenizer) -> None:
    rows = [
        {"prompt": "short", "answer": "1"},
        {"prompt": "word " * 200, "answer": "2"},
        {"prompt": "also short", "answer": "3"},
    ]
    p = write_jsonl(tmp_path / "len.jsonl", rows)
    src = open_source(DataConfig(uri=str(p), shuffle=False, max_prompt_tokens=64), tokenizer=tiny_tokenizer)
    assert [s.reference for s in src] == ["1", "3"]
    assert src.mapper.dropped == 1
    assert all(0 < s.metadata["prompt_tokens"] <= 64 for s in src)
    chat_len = len(
        tiny_tokenizer.apply_chat_template(src[0].messages, tokenize=False, add_generation_prompt=True)
    )
    assert chat_len > len("short"), "count is taken after the chat template"
    assert len(open_source(DataConfig(uri=str(p), shuffle=False, max_prompt_tokens=64))) == 3, (
        "no tokenizer → no filter"
    )
    streaming = open_source(
        DataConfig(uri=str(p), shuffle=False, streaming=True, max_prompt_tokens=64), tokenizer=tiny_tokenizer
    )
    assert [s.reference for s in streaming] == ["1", "3"]
    with pytest.raises(DataError):
        len(streaming)


def test_credentials_in_extra_are_rejected(qa_jsonl: Path) -> None:
    with pytest.raises(DataError, match="environment only"):
        _open(qa_jsonl, extra={"token": "hf_xxx"})
