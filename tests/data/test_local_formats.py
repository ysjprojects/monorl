from __future__ import annotations

import json
from pathlib import Path

import pytest

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.core.interfaces import DataSource
from monorl.core.types import Sample
from monorl.data import open_source
from tests.data.conftest import CHAT_ROWS, QA_ROWS, write_arrow, write_csv, write_jsonl, write_parquet


def _open(path: Path, **kw) -> DataSource:
    return open_source(DataConfig(uri=str(path), shuffle=False, **kw))


def _check_qa(src: DataSource) -> None:
    assert isinstance(src, DataSource)
    assert len(src) == len(QA_ROWS)
    samples = list(src)
    assert all(isinstance(s, Sample) for s in samples)
    assert samples[3].messages == [{"role": "user", "content": "What is 3+3?"}]
    assert samples[3].reference == "6"
    assert list(src) == samples, "materialised sources iterate repeatably"


def test_jsonl(tmp_path: Path) -> None:
    src = _open(write_jsonl(tmp_path / "x.jsonl", QA_ROWS))
    _check_qa(src)
    assert src.schema() == {"question": "string", "answer": "string", "difficulty": "int64"}
    assert src.name == "local"


def test_json_array_and_wrapped(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps(QA_ROWS))
    _check_qa(_open(p))
    p.write_text(json.dumps({"data": QA_ROWS}))
    _check_qa(_open(p))
    p.write_text(
        json.dumps({"question": [r["question"] for r in QA_ROWS], "answer": [r["answer"] for r in QA_ROWS]})
    )
    _check_qa(_open(p))


def test_csv(tmp_path: Path) -> None:
    src = _open(write_csv(tmp_path / "x.csv", QA_ROWS))
    _check_qa(src)
    assert src.schema()["difficulty"] == "int64"


def test_parquet(tmp_path: Path) -> None:
    src = _open(write_parquet(tmp_path / "x.parquet", QA_ROWS))
    _check_qa(src)
    assert src.schema() == {"question": "string", "answer": "string", "difficulty": "int64"}


def test_arrow_stream(tmp_path: Path) -> None:
    src = _open(write_arrow(tmp_path / "x.arrow", QA_ROWS))
    _check_qa(src)


def test_parquet_chat_struct_column(tmp_path: Path) -> None:
    src = _open(write_parquet(tmp_path / "chat.parquet", CHAT_ROWS))
    assert src.layout.format == "chat"
    s = next(iter(src))
    assert s.id == "chat-0"
    assert s.messages == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say the number 0."},
    ]
    assert s.reference == "0"
    assert "struct<role: string, content: string>" in src.schema()["messages"]


def test_gzip_jsonl(tmp_path: Path) -> None:
    import gzip

    p = tmp_path / "x.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        for r in QA_ROWS:
            fh.write(json.dumps(r) + "\n")
    _check_qa(_open(p))


def test_directory_recurses_and_skips_non_data(jsonl_dir: Path) -> None:
    src = _open(jsonl_dir)
    assert len(src) == len(QA_ROWS)
    ids = [s.id for s in src]
    assert ids[0].endswith("shards/nested/part-001.jsonl#0") and ids[-1].endswith("shards/part-000.jsonl#4")
    assert len(set(ids)) == len(ids)


def test_glob(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "train-0.jsonl", QA_ROWS[:4])
    write_jsonl(tmp_path / "train-1.jsonl", QA_ROWS[4:8])
    write_jsonl(tmp_path / "test.jsonl", QA_ROWS[8:])
    assert len(_open(tmp_path / "train-*.jsonl")) == 8


def test_directory_prefers_data_format_over_metadata_json(tmp_path: Path) -> None:
    d = tmp_path / "saved"
    d.mkdir()
    write_arrow(d / "data-00000.arrow", QA_ROWS)
    (d / "state.json").write_text(json.dumps({"_fingerprint": "abc", "_format_type": None}))
    (d / "dataset_info.json").write_text(json.dumps({"features": {}}))
    assert len(_open(d)) == len(QA_ROWS)


def test_file_uri_scheme(qa_jsonl: Path) -> None:
    assert len(_open(f"file://{qa_jsonl}")) == len(QA_ROWS)


def test_missing_and_unsupported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _open(tmp_path / "nope.jsonl")
    (tmp_path / "x.txt").write_text("hello")
    with pytest.raises(DataError):
        _open(tmp_path / "x.txt")
    (tmp_path / "bad.jsonl").write_text('{"question": "a", "answer": "b"}\n{not json\n')
    with pytest.raises(DataError, match="invalid JSON"):
        _open(tmp_path / "bad.jsonl")


def test_local_streaming_is_lazy_and_len_raises(qa_jsonl: Path) -> None:
    src = _open(qa_jsonl, streaming=True)
    with pytest.raises(DataError):
        len(src)
    first = list(src)
    assert [s.reference for s in first] == [str(2 * i) for i in range(len(QA_ROWS))]
    assert [s.id for s in src] == [s.id for s in first], "streams re-open on each iteration"


def test_streaming_parquet_len_from_metadata(tmp_path: Path) -> None:
    src = _open(write_parquet(tmp_path / "x.parquet", QA_ROWS), streaming=True)
    assert len(src) == len(QA_ROWS)
    assert len(_open(tmp_path / "x.parquet", streaming=True, limit=3)) == 3
