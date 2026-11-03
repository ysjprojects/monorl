from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from monorl.adapters.nemo_rl.data import (
    MODULE_NAME,
    TASK_NAME,
    MonorlDataset,
    data_paths,
    export_dataset,
    export_jsonl,
    monorl_data_processor,
    row_to_sample,
    sample_to_row,
    write_bridge_module,
)
from monorl.config.schema import RunConfig
from monorl.core.types import Sample


class _Spec:
    def __init__(self, prompt=None, system_prompt=None):
        self.prompt = prompt
        self.system_prompt = system_prompt


def test_row_roundtrip_keeps_nested_fields() -> None:
    sample = Sample(
        id="s1",
        messages=[{"role": "system", "content": "be terse"}, {"role": "user", "content": "2+2?"}],
        reference={"answer": "4", "alts": ["four"]},
        metadata={"row": {"difficulty": 1}},
    )
    row = sample_to_row(sample)
    assert row["task_name"] == TASK_NAME
    assert row["input"] == "2+2?"  # stock ResponseDataset input_key
    assert json.loads(row["output"]) == {"answer": "4", "alts": ["four"]}
    assert row_to_sample(row) == sample

    plain = sample_to_row(Sample(id="s2", messages=[{"role": "user", "content": "q"}], reference="8"))
    assert plain["output"] == "8"
    assert row_to_sample(plain).reference == "8"


def test_export_dataset_writes_jsonl_and_paths(nemo_config: RunConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    exported = export_dataset(nemo_config, run_dir)
    paths = data_paths(run_dir)
    assert exported.paths == paths
    assert exported.num_train == 4 and exported.num_eval is None
    rows = [json.loads(line) for line in paths.train_jsonl.read_text().splitlines()]
    assert len(rows) == 4
    assert {r["task_name"] for r in rows} == {TASK_NAME}
    assert {row_to_sample(r).reference for r in rows} == {"4", "8", "6", "21"}

    data = {**nemo_config.data.model_dump(mode="json"), "eval_uri": nemo_config.data.uri}
    with_eval = RunConfig.from_dict({**nemo_config.to_dict(), "data": data})
    exported = export_dataset(with_eval, run_dir)
    assert exported.num_eval == 4 and paths.eval_jsonl.exists()


def test_processor_builds_datum_spec(tiny_model_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    sample = Sample(id="s1", messages=[{"role": "user", "content": "What is 2+2?"}], reference="4", metadata={"k": "v"})
    datum = monorl_data_processor(sample_to_row(sample), _Spec(), tokenizer, 512, 7)
    assert datum["task_name"] == TASK_NAME
    assert datum["idx"] == 7 and datum["loss_multiplier"] == 1.0
    assert len(datum["message_log"]) == 1
    msg = datum["message_log"][0]
    assert msg["role"] == "user" and isinstance(msg["token_ids"], torch.Tensor)
    assert msg["token_ids"].dtype == torch.int64
    assert datum["length"] == msg["token_ids"].shape[0] > 0
    expected = tokenizer.apply_chat_template(sample.messages, tokenize=False, add_generation_prompt=True)
    assert msg["content"] == expected
    assert datum["extra_env_info"] == {"sample_id": "s1", "messages": sample.messages, "reference": "4", "metadata": {"k": "v"}}


def test_processor_masks_overlong_and_applies_task_spec(tiny_model_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    sample = Sample(id="s1", messages=[{"role": "user", "content": "word " * 200}], reference="x")
    datum = monorl_data_processor(sample_to_row(sample), _Spec(), tokenizer, 32, 0)
    assert datum["loss_multiplier"] == 0.0
    assert datum["message_log"][0]["token_ids"].shape[0] == 4
    assert datum["length"] >= 32

    spec = _Spec(prompt="Q: {}\nThink step by step.", system_prompt="You are terse.")
    datum = monorl_data_processor(sample_to_row(Sample(id="s2", messages=[{"role": "user", "content": "2+2?"}], reference="4")), spec, tokenizer, 512, 1)
    rendered = datum["extra_env_info"]["messages"]
    assert rendered[0] == {"role": "system", "content": "You are terse."}
    assert rendered[1]["content"] == "Q: 2+2?\nThink step by step."
    assert "Think step by step." in datum["message_log"][0]["content"]


def test_dataset_loads_export_and_splits(nemo_config: RunConfig, tmp_path: Path, fake_nemo_rl) -> None:
    paths = export_dataset(nemo_config, tmp_path).paths
    ds = MonorlDataset(
        data_path=str(paths.train_jsonl),
        dataset_name=f"{MODULE_NAME}.MonorlDataset",
        prompt_file=None,
        system_prompt_file=None,
        processor=TASK_NAME,
        env_name="monorl",
        split_validation_size=0,
        seed=1,
    )
    assert ds.task_name == TASK_NAME and len(ds.dataset) == 4 and ds.val_dataset is None
    assert "task_name" in ds.dataset.column_names and ds.dataset[0]["task_name"] == TASK_NAME
    ds.set_task_spec({"prompt_file": None, "system_prompt_file": None, "env_name": "monorl"})
    ds.set_processor()
    assert ds.task_spec.task_name == TASK_NAME and ds.task_spec.prompt is None
    assert ds.processor is monorl_data_processor and ds.preprocessor is None

    split = MonorlDataset(data_path=str(paths.train_jsonl), split_validation_size=0.5, seed=0)
    assert len(split.dataset) == 2 and len(split.val_dataset) == 2


def test_bridge_module_importable_from_run_dir(tmp_path: Path) -> None:
    module_path = write_bridge_module(tmp_path)
    assert module_path == data_paths(tmp_path).module_path == tmp_path.resolve() / f"{MODULE_NAME}.py"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        assert module.MonorlDataset is MonorlDataset
        assert module.monorl_data_processor is monorl_data_processor
        assert module.TASK_NAME == TASK_NAME
    finally:
        sys.modules.pop(MODULE_NAME, None)


def test_export_jsonl_counts(tmp_path: Path) -> None:
    samples = [Sample(id=str(i), messages=[{"role": "user", "content": f"q{i}"}], reference=str(i)) for i in range(3)]
    assert export_jsonl(samples, tmp_path / "nested" / "x.jsonl") == 3
    assert (tmp_path / "nested" / "x.jsonl").read_text().count("\n") == 3


@pytest.mark.parametrize("bad", [{"sample_id": "x"}, {}])
def test_row_to_sample_requires_fields(bad: dict) -> None:
    with pytest.raises(KeyError):
        row_to_sample(bad)
