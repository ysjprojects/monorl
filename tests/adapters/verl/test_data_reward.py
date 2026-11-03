"""Parquet export columns and the generated custom_reward_function module."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from monorl.adapters.verl.data import (
    PARQUET_COLUMNS,
    decode_reference,
    encode_reference,
    export_datasets,
    sample_to_row,
    write_parquet,
)
from monorl.adapters.verl.reward import RewardBridge, trajectory_from_verl, write_reward_module
from monorl.core.types import Sample


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("monorl_reward_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sample_to_row_layout():
    sample = Sample(id="s1", messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}], reference={"a": 1}, metadata={"k": "v"})
    row = sample_to_row(sample, 3, agent_name="tool_agent")
    assert tuple(row) == PARQUET_COLUMNS
    assert row["prompt"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    assert row["data_source"] == "monorl"
    assert row["reward_model"] == {"style": "rule", "ground_truth": '{"a": 1}'}
    assert row["extra_info"]["index"] == 3
    assert row["extra_info"]["sample_id"] == "s1"
    assert row["extra_info"]["reference_encoding"] == "json"
    assert json.loads(row["extra_info"]["metadata_json"]) == {"k": "v"}
    assert row["agent_name"] == "tool_agent"


@pytest.mark.parametrize("ref", ["4", None, {"tests": [1, 2]}, [1, "a"], 3])
def test_reference_roundtrip(ref):
    assert decode_reference(*encode_reference(ref)) == ref


def test_export_datasets_columns(make_config, tmp_path: Path, data_file: Path):
    cfg = make_config(
        data={"uri": str(data_file), "prompt_field": "prompt", "answer_field": "answer", "eval_uri": str(data_file), "shuffle": False}
    )
    paths = export_datasets(cfg, tmp_path)
    assert set(paths) == {"train", "val"}
    table = pq.read_table(paths["train"])
    assert table.column_names == list(PARQUET_COLUMNS)
    assert table.num_rows == 3
    rows = table.to_pylist()
    assert rows[0]["prompt"][-1]["role"] == "user"
    assert {r["reward_model"]["ground_truth"] for r in rows} == {"4", "8", "6"}
    assert all(r["agent_name"] == "single_turn_agent" for r in rows)
    assert pq.read_table(paths["val"]).num_rows == 3


def test_write_parquet_rejects_empty(tmp_path: Path):
    with pytest.raises(ValueError):
        write_parquet([], tmp_path / "x.parquet")


def test_trajectory_from_verl_rebuilds_sample():
    extra = {"sample_id": "s9", "index": 2, "messages_json": json.dumps([{"role": "user", "content": "q"}]), "metadata_json": "{}", "reference_encoding": "text", "num_turns": 2}
    traj = trajectory_from_verl("resp", "4", extra)
    assert traj.sample.id == "s9"
    assert traj.sample.reference == "4"
    assert traj.sample.messages == [{"role": "user", "content": "q"}]
    assert traj.response_text == "resp"
    assert traj.messages[-1] == {"role": "assistant", "content": "resp"}
    assert traj.info["num_turns"] == 2


def test_generated_reward_module(make_config, tmp_path: Path):
    cfg = make_config(rewards=[{"type": "exact_match", "weight": 2.0}, {"type": "length", "weight": 0.0, "params": {"name": "len"}}])
    path = write_reward_module(cfg, tmp_path, config_path=None)
    assert path == tmp_path / "monorl_reward.py"
    module = _load_module(path)
    assert module.REWARD_SPECS[0]["type"] == "exact_match"
    extra = sample_to_row(Sample(id="a", messages=[{"role": "user", "content": "2+2?"}], reference="4"), 0)["extra_info"]
    hit = module.compute_score(data_source="monorl", solution_str="The answer is 4", ground_truth="4", extra_info=extra)
    miss = module.compute_score(data_source="monorl", solution_str="The answer is 5", ground_truth="4", extra_info=extra)
    assert hit["score"] == pytest.approx(2.0)
    assert hit["reward/exact_match"] == pytest.approx(1.0)
    assert "reward/len" in hit
    assert miss["score"] == pytest.approx(0.0)
    assert module._bridge is not None  # built once and cached


def test_reward_bridge_with_run_config(make_config, tmp_path: Path):
    cfg = make_config()
    config_path = tmp_path / "config.yaml"
    cfg.to_yaml(config_path)
    bridge = RewardBridge([s.model_dump(mode="json") for s in cfg.rewards], config_path)
    assert bridge.config is not None
    out = bridge("monorl", "4", "4", {"reference_encoding": "text"})
    assert out["score"] == pytest.approx(1.0)


def test_write_reward_module_needs_rewards(make_config, tmp_path: Path):
    cfg = make_config(rewards=[], env={"type": "tool_calling", "params": {"tools": ["calculator"]}})
    with pytest.raises(ValueError):
        write_reward_module(cfg, tmp_path)
