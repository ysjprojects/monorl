"""Real CPU mini-pipeline on the tiny random Llama: sft → grpo → rejection_sample → sft (on the
sampled data) → eval, with the ``hf`` engine, ``reference`` trainer and ``exact_match`` reward.
No fakes anywhere."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from monorl.config.pipeline import PipelineConfig
from monorl.core.reporting import iter_metrics, read_status
from monorl.pipeline import read_manifest, run_pipeline
from monorl.pipeline.stages import DATASET_FILE

pytestmark = pytest.mark.slow


@pytest.fixture
def real_pipeline(run_config_dict: dict[str, Any], tmp_path: Path) -> PipelineConfig:
    base = dict(run_config_dict)
    base.pop("run_name")
    base["data"] = {**base["data"], "limit": 4}
    base["trainer"] = {**base["trainer"], "save_every": 10}
    base["rollout"] = {**base["rollout"], "max_new_tokens": 6, "extra": {"max_parallel_episodes": 4}}
    return PipelineConfig.from_dict(
        {
            "name": "real-mini-pipeline",
            "base": base,
            "stages": [
                {
                    "type": "train",
                    "name": "cold_start_sft",
                    "init_from": "base",
                    "overrides": {
                        "algorithm": {"name": "sft"},
                        "data": {"response_field": "answer"},
                        "trainer": {"max_steps": 2, "prompts_per_step": 2},
                    },
                },
                {
                    "type": "train",
                    "name": "grpo",
                    "overrides": {"trainer": {"max_steps": 1}, "rollout": {"group_size": 2}},
                },
                {
                    "type": "rejection_sample",
                    "name": "rs",
                    "num_samples": 2,
                    "min_reward": 0.5,
                    "max_per_prompt": 1,
                    "require_finish": False,
                    # a random policy rarely matches the reference; a length reward that always fires keeps rows
                    "reward_overrides": [
                        {"type": "exact_match", "weight": 1.0},
                        {"type": "length", "weight": 1.0, "params": {"max_tokens": 64}},
                    ],
                    "mix": [str(Path(base["data"]["uri"]))],
                },
                {
                    "type": "train",
                    "name": "sft_on_samples",
                    "data_from": "previous",
                    "overrides": {
                        "algorithm": {"name": "sft"},
                        "trainer": {"max_steps": 1, "prompts_per_step": 2},
                    },
                },
                {"type": "eval", "name": "eval", "max_prompts": 2},
            ],
        }
    )


def test_real_mini_pipeline(real_pipeline: PipelineConfig) -> None:
    result = run_pipeline(real_pipeline)
    assert result.state == "succeeded", (
        result.error,
        [(s["name"], s["state"], s["error"]) for s in result.stages],
    )

    run_dir = Path(result.run_dir)
    manifest = read_manifest(run_dir)
    assert manifest["state"] == "succeeded"
    stages = {s["name"]: s for s in manifest["stages"]}
    assert all(s["state"] == "succeeded" for s in stages.values())
    assert sorted(Path(s["run_dir"]).name for s in stages.values()) == [
        "00-cold_start_sft",
        "01-grpo",
        "02-rs",
        "03-sft_on_samples",
        "04-eval",
    ]
    assert read_status(run_dir)["state"] == "succeeded" and read_status(run_dir)["step"] == 5

    # SFT cold start exported a real HF model that GRPO then loaded
    sft_final = Path(stages["cold_start_sft"]["run_dir"]) / "final"
    assert json.loads((sft_final / "config.json").read_text())["architectures"] == ["LlamaForCausalLM"]
    grpo_cfg = (Path(stages["grpo"]["run_dir"]) / "config.yaml").read_text()
    assert str(sft_final) in grpo_cfg
    grpo_rows = [r for _, r in iter_metrics(stages["grpo"]["run_dir"]) if r["phase"] == "train"]
    assert [r["step"] for r in grpo_rows] == [1] and "loss/policy" in grpo_rows[0]

    # rejection sampling wrote chat rows the next SFT stage consumed
    dataset = Path(stages["rs"]["run_dir"]) / DATASET_FILE
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    assert rows and stages["rs"]["summary"]["rows"] == len(rows)
    sampled = [r for r in rows if r["source"] == "sampled"]
    mixed = [r for r in rows if r["source"] == "mix"]
    assert sampled and len(mixed) == 4
    for row in rows:
        assert row["messages"][0]["role"] == "user" and row["messages"][-1]["role"] == "assistant"
        assert isinstance(row["messages"][-1]["content"], str)
    assert {r["reference"] for r in mixed} == {"4", "8", "6", "21"}
    assert all(r["messages"][-1]["content"] == r["reference"] for r in mixed)
    assert stages["rs"]["summary"]["sampled"] == 8 and stages["rs"]["summary"]["accepted"] == len(sampled)
    rs_metrics = [r for _, r in iter_metrics(stages["rs"]["run_dir"]) if r["phase"] == "rejection"]
    assert rs_metrics and "rs/accept_rate" in rs_metrics[0] and "reward/exact_match" in rs_metrics[0]

    sft2_cfg = (Path(stages["sft_on_samples"]["run_dir"]) / "config.yaml").read_text()
    assert str(dataset) in sft2_cfg and "format: chat" in sft2_cfg
    sft2_rows = [r for _, r in iter_metrics(stages["sft_on_samples"]["run_dir"]) if r["phase"] == "train"]
    assert [r["step"] for r in sft2_rows] == [1] and "loss/sft" in sft2_rows[0]

    # final model = last train stage's export; eval ran greedily on it
    assert result.final_model_path == str(Path(stages["sft_on_samples"]["run_dir"]) / "final")
    assert (Path(result.final_model_path) / "config.json").exists()
    assert stages["eval"]["summary"]["eval/num_samples"] == 2.0
    assert "eval/reward_mean" in stages["eval"]["summary"]
    assert str(result.final_model_path) in (Path(stages["eval"]["run_dir"]) / "config.yaml").read_text()
