from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

QA_ROWS: list[dict[str, Any]] = [
    {"question": f"What is {i}+{i}?", "answer": f"Add them. #### {2 * i}", "difficulty": i % 3} for i in range(12)
]

CHAT_ROWS: list[dict[str, Any]] = [
    {
        "id": f"chat-{i}",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": f"Say the number {i}."},
            {"role": "assistant", "content": str(i)},
        ],
        "tags": ["a", "b"],
    }
    for i in range(6)
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def write_arrow(path: Path, rows: list[dict[str, Any]]) -> Path:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    table = pa.Table.from_pylist(rows)
    with pa.OSFile(str(path), "wb") as sink, ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    import csv

    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.fixture
def qa_jsonl(tmp_path: Path) -> Path:
    return write_jsonl(tmp_path / "qa.jsonl", QA_ROWS)


@pytest.fixture
def chat_jsonl(tmp_path: Path) -> Path:
    return write_jsonl(tmp_path / "chat.jsonl", CHAT_ROWS)


@pytest.fixture
def jsonl_dir(tmp_path: Path) -> Path:
    d = tmp_path / "shards"
    write_jsonl(d / "part-000.jsonl", QA_ROWS[:5])
    write_jsonl(d / "nested" / "part-001.jsonl", QA_ROWS[5:])
    (d / "README.md").write_text("ignore me")
    (d / ".hidden.jsonl").write_text(json.dumps({"question": "x", "answer": "y"}) + "\n")
    return d


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def moto_s3(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A ThreadedMotoServer on a free port with dummy env credentials + a seeded bucket."""
    pytest.importorskip("moto")
    from moto.server import ThreadedMotoServer

    port = free_port()
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()
    endpoint = f"http://127.0.0.1:{port}"
    mp = pytest.MonkeyPatch()
    mp.setenv("AWS_ACCESS_KEY_ID", "testing")
    mp.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    mp.setenv("AWS_DEFAULT_REGION", "us-east-1")
    mp.setenv("AWS_ENDPOINT_URL", endpoint)
    mp.delenv("AWS_SESSION_TOKEN", raising=False)
    mp.delenv("AWS_PROFILE", raising=False)

    import boto3

    client = boto3.client("s3", endpoint_url=endpoint, region_name="us-east-1")
    bucket = "monorl-test"
    client.create_bucket(Bucket=bucket)
    local = tmp_path_factory.mktemp("s3src")
    client.upload_file(str(write_jsonl(local / "a.jsonl", QA_ROWS[:6])), bucket, "data/jsonl/a.jsonl")
    client.upload_file(str(write_jsonl(local / "b.jsonl", QA_ROWS[6:])), bucket, "data/jsonl/b.jsonl")
    client.upload_file(str(write_parquet(local / "chat.parquet", CHAT_ROWS)), bucket, "data/parquet/chat.parquet")
    client.put_object(Bucket=bucket, Key="data/notes.txt", Body=b"not data")
    try:
        yield {"endpoint": endpoint, "bucket": bucket, "client": client}
    finally:
        mp.undo()
        server.stop()


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model_dir)
