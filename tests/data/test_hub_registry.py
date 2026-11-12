from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.core.registry import Registry
from monorl.data import open_source, register
from monorl.data.browse import browse_uri, preview_uri

GSM8K = "hf://openai/gsm8k"


def test_register_and_scheme_guard(qa_jsonl: Path) -> None:
    reg = Registry()
    register(reg)
    assert reg.names("data_source", load_builtins=False) == ["hf", "local", "s3"]
    src = reg.get("data_source", "local")(DataConfig(uri=str(qa_jsonl), shuffle=False), seed=3)
    assert len(src) == 12
    with pytest.raises(DataError, match="cannot open"):
        reg.get("data_source", "s3")(DataConfig(uri=str(qa_jsonl)))
    with pytest.raises(DataError, match="cannot open"):
        reg.get("data_source", "hf")(DataConfig(uri="s3://b/k.jsonl"))


def test_builtin_registry_resolves_data_sources() -> None:
    from monorl.core.registry import registry

    assert registry.contains("data_source", "local")
    assert registry.contains("data_source", "hf")
    assert registry.contains("data_source", "s3")


@pytest.mark.network
def test_hf_streaming_open_source() -> None:
    cfg = DataConfig(
        uri=GSM8K,
        hf_config_name="main",
        split="train",
        streaming=True,
        shuffle=False,
        system_prompt="Solve step by step.",
    )
    src = open_source(cfg)
    assert src.name == "hf"
    assert len(src) == 7473, "split size comes from Hub metadata, no download"
    assert src.schema() == {"question": "string", "answer": "string"}
    assert src.layout.format == "prompt_answer"
    first = list(itertools.islice(src, 3))
    assert first[0].id == f"{GSM8K}#train#0"
    assert first[0].messages[0] == {"role": "system", "content": "Solve step by step."}
    assert first[0].messages[1]["role"] == "user" and first[0].messages[1]["content"].startswith("Natalia")
    assert first[0].reference == "72"
    assert "####" in first[0].metadata["answer_raw"]
    shuffled = open_source(
        DataConfig(uri=GSM8K, hf_config_name="main", streaming=True, shuffle=True, limit=5)
    )
    assert len(shuffled) == 5, "known split size capped by limit"
    a = [s.id for s in shuffled]
    assert len(a) == 5 and a == [s.id for s in shuffled]
    with pytest.raises(DataError, match="cannot load"):
        open_source(DataConfig(uri=GSM8K, hf_config_name="no-such-config", streaming=True))


@pytest.mark.network
def test_hf_preview_and_browse() -> None:
    out = preview_uri(GSM8K, split="test", hf_config_name="main", limit=4)
    assert out["columns"] == {"question": "string", "answer": "string"}
    assert len(out["rows"]) == 4 and out["num_rows"] == 1319
    assert out["inferred"]["format"] == "prompt_answer"
    tree = browse_uri(GSM8K)
    names = {e["name"] for e in tree}
    assert "main" in names and next(e for e in tree if e["name"] == "main")["is_dir"] is True
    files = browse_uri(f"{GSM8K}/main")
    assert any(e["name"].endswith(".parquet") and e["size"] for e in files)
    assert files[0]["uri"].startswith("hf://openai/gsm8k/main/")
