from __future__ import annotations

import json
from pathlib import Path

import pytest

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.data import open_source
from monorl.data.browse import browse_uri, preview_uri
from monorl.data.uri import parse_uri, s3_env_status, s3_storage_options
from tests.data.conftest import CHAT_ROWS, QA_ROWS, write_jsonl, write_parquet

# -- local browse / preview --------------------------------------------------------------------


def test_browse_local_directory(jsonl_dir: Path) -> None:
    entries = browse_uri(str(jsonl_dir))
    assert [e["name"] for e in entries] == ["nested", "part-000.jsonl", "README.md"]
    assert entries[0]["is_dir"] is True and entries[0]["size"] is None
    assert entries[1] == {
        "name": "part-000.jsonl",
        "uri": str(jsonl_dir / "part-000.jsonl"),
        "is_dir": False,
        "size": (jsonl_dir / "part-000.jsonl").stat().st_size,
    }
    assert browse_uri(str(jsonl_dir / "part-000.jsonl"))[0]["name"] == "part-000.jsonl"
    with pytest.raises(FileNotFoundError):
        browse_uri(str(jsonl_dir / "missing"))


def test_preview_local_reads_only_limit_rows(tmp_path: Path) -> None:
    big = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(500)]
    p = write_jsonl(tmp_path / "big.jsonl", big)
    out = preview_uri(str(p), limit=3)
    assert out["columns"] == {"question": "string", "answer": "string"}
    assert out["rows"] == big[:3]
    assert out["inferred"]["format"] == "prompt_answer"
    assert out["num_rows"] is None, "jsonl length is not known without reading everything"
    small = preview_uri(str(write_jsonl(tmp_path / "small.jsonl", big[:7])), limit=3)
    assert small["num_rows"] == 7 and len(small["rows"]) == 3
    pq_out = preview_uri(str(write_parquet(tmp_path / "big.parquet", big)), limit=2)
    assert pq_out["num_rows"] == 500 and len(pq_out["rows"]) == 2
    with pytest.raises(ValueError):
        preview_uri(str(p), limit=0)


def test_preview_directory_and_json_serialisable(jsonl_dir: Path, tmp_path: Path) -> None:
    out = preview_uri(str(jsonl_dir))
    assert out["num_rows"] == len(QA_ROWS) and len(out["rows"]) == len(QA_ROWS)
    json.dumps(out)
    chat = preview_uri(str(write_parquet(tmp_path / "c.parquet", CHAT_ROWS)), limit=2)
    assert chat["inferred"]["messages_field"] == "messages" and chat["num_rows"] == len(CHAT_ROWS)
    json.dumps(chat)


# -- uri parsing / env ---------------------------------------------------------------------------


def test_parse_uri_shapes(tmp_path: Path) -> None:
    hf = parse_uri("hf://openai/gsm8k@v1/data/train.parquet")
    assert (hf.kind, hf.repo_id, hf.revision, hf.subpath) == ("hf", "openai/gsm8k", "v1", "data/train.parquet")
    assert parse_uri("hf://datasets/org/name").repo_id == "org/name"
    s3 = parse_uri("s3://bucket/prefix/")
    assert (s3.kind, s3.scheme, s3.path) == ("remote", "s3", "bucket/prefix")
    assert parse_uri("s3://").path == ""
    assert parse_uri("gs://b/k").scheme == "gs"
    local = parse_uri("~/x.jsonl")
    assert local.kind == "local" and local.path == str(Path("~/x.jsonl").expanduser())
    for bad in ("", "hf://only-org", "://x"):
        with pytest.raises(DataError):
            parse_uri(bad)


def test_s3_options_come_only_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION", "AWS_REGION", "AWS_ENDPOINT_URL"):
        monkeypatch.delenv(var, raising=False)
    assert s3_storage_options() == {}
    assert s3_env_status() == {"configured": False, "region": None, "endpoint_url": None, "vars_present": []}
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "t")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://minio:9000")
    assert s3_storage_options() == {
        "key": "k",
        "secret": "s",
        "token": "t",
        "client_kwargs": {"region_name": "eu-west-1", "endpoint_url": "http://minio:9000"},
    }
    status = s3_env_status()
    assert status["configured"] is True and status["endpoint_url"] == "http://minio:9000"
    assert "k" not in json.dumps(status) and "s" not in status.values()


def test_unknown_fsspec_driver_is_a_data_error() -> None:
    with pytest.raises(DataError, match="no fsspec driver"):
        open_source(DataConfig(uri="nosuchscheme://bucket/x.jsonl"))


# -- S3 end to end (moto) ------------------------------------------------------------------------


def test_s3_open_source_jsonl_prefix(moto_s3: dict) -> None:
    uri = f"s3://{moto_s3['bucket']}/data/jsonl/"
    src = open_source(DataConfig(uri=uri, shuffle=False))
    assert src.name == "s3"
    assert len(src) == len(QA_ROWS)
    assert src[0].id == f"s3://{moto_s3['bucket']}/data/jsonl/a.jsonl#0"
    assert src[7].id == f"s3://{moto_s3['bucket']}/data/jsonl/b.jsonl#1"
    assert src[7].messages == [{"role": "user", "content": "What is 7+7?"}] and src[7].reference == "14"
    assert src.schema() == {"question": "string", "answer": "string", "difficulty": "int64"}


def test_s3_single_parquet_file_and_streaming(moto_s3: dict) -> None:
    uri = f"s3://{moto_s3['bucket']}/data/parquet/chat.parquet"
    src = open_source(DataConfig(uri=uri, shuffle=False, streaming=True))
    assert len(src) == len(CHAT_ROWS), "parquet metadata gives the count without reading rows"
    samples = list(src)
    assert [s.id for s in samples] == [r["id"] for r in CHAT_ROWS]
    assert samples[2].messages[-1] == {"role": "user", "content": "Say the number 2."}
    assert samples[2].reference == "2"
    streamed_jsonl = open_source(DataConfig(uri=f"s3://{moto_s3['bucket']}/data/jsonl/*.jsonl", streaming=True, shuffle=True, limit=4))
    with pytest.raises(DataError):
        len(streamed_jsonl)
    assert len(list(streamed_jsonl)) == 4


def test_s3_browse_and_preview(moto_s3: dict) -> None:
    bucket = moto_s3["bucket"]
    buckets = browse_uri("s3://")
    assert {"name": bucket, "uri": f"s3://{bucket}", "is_dir": True, "size": None} in buckets
    root = browse_uri(f"s3://{bucket}/data")
    assert [(e["name"], e["is_dir"]) for e in root] == [("jsonl", True), ("parquet", True), ("notes.txt", False)]
    assert root[2]["uri"] == f"s3://{bucket}/data/notes.txt" and root[2]["size"] == len(b"not data")
    with pytest.raises(FileNotFoundError):
        browse_uri(f"s3://{bucket}/nope")

    out = preview_uri(f"s3://{bucket}/data/jsonl", limit=4)
    assert out["rows"] == QA_ROWS[:4]
    assert out["inferred"] == {"format": "prompt_answer", "prompt_field": "question", "answer_field": "answer", "messages_field": None, "image_fields": []}
    assert out["num_rows"] == len(QA_ROWS)
    pq_out = preview_uri(f"s3://{bucket}/data/parquet/chat.parquet", limit=2)
    assert pq_out["num_rows"] == len(CHAT_ROWS) and pq_out["inferred"]["format"] == "chat"
    assert [r["id"] for r in pq_out["rows"]] == ["chat-0", "chat-1"]
    json.dumps(pq_out)


def test_s3_sees_newly_written_objects(moto_s3: dict, tmp_path: Path) -> None:
    bucket = moto_s3["bucket"]
    assert browse_uri(f"s3://{bucket}/") and all(e["name"] != "late" for e in browse_uri(f"s3://{bucket}/"))
    moto_s3["client"].upload_file(str(write_jsonl(tmp_path / "late.jsonl", QA_ROWS[:2])), bucket, "late/late.jsonl")
    assert any(e["name"] == "late" for e in browse_uri(f"s3://{bucket}/"))
    assert len(open_source(DataConfig(uri=f"s3://{bucket}/late"))) == 2
