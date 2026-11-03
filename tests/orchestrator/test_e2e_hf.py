"""End-to-end: real ``hf`` rollout engine + ``reference`` trainer + ``exact_match`` reward on the
tiny random Llama. No fakes anywhere."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from monorl.config.schema import RunConfig
from monorl.core.reporting import iter_metrics, iter_trajectories, list_checkpoints, read_status
from monorl.orchestrator import run

CONTRACT_KEYS = {
    "reward/mean",
    "reward/std",
    "reward/exact_match",
    "loss/policy",
    "loss/kl",
    "loss/entropy",
    "policy/clipfrac",
    "policy/approx_kl",
    "tokens/response_mean",
    "tokens/prompt_mean",
    "time/rollout",
    "time/train",
    "time/sync",
    "lr",
}


@pytest.mark.slow
def test_end_to_end_hf_engine_reference_trainer(run_config_dict: dict[str, Any]) -> None:
    config = RunConfig.from_dict(run_config_dict)
    result = run(config)

    assert result.state == "succeeded", result.error
    assert result.final_step == 2
    run_dir = Path(result.run_dir)
    assert read_status(run_dir)["state"] == "succeeded"

    rows = [row for _, row in iter_metrics(run_dir) if row["phase"] == "train"]
    assert [r["step"] for r in rows] == [1, 2]
    missing = CONTRACT_KEYS - set(rows[0])
    assert not missing, f"missing metric keys: {sorted(missing)}; have {sorted(rows[0])}"
    assert rows[0]["tokens/prefix_mismatch"] == 0.0
    assert rows[0]["policy/staleness"] == 0.0
    assert 0 < rows[0]["tokens/response_mean"] <= config.rollout.max_new_tokens
    assert rows[0]["tokens/prompt_mean"] > 0

    assert (run_dir / "checkpoints" / "step-1").is_dir()
    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-1", "step-2"]
    assert (run_dir / "final" / "config.json").exists()
    assert json.loads((run_dir / "final" / "config.json").read_text())["architectures"] == ["LlamaForCausalLM"]

    trajectories = [row for _, row in iter_trajectories(run_dir)]
    assert {t["step"] for t in trajectories} == {1, 2}
    assert len([t for t in trajectories if t["step"] == 1]) == config.sequences_per_step()
    first = trajectories[0]
    assert first["prompt"] == "What is 2+2?" or first["prompt"].startswith("What is")
    assert isinstance(first["response"], str) and first["num_response_tokens"] > 0
    assert "exact_match" in first["reward_breakdown"]
    assert (run_dir / "config.yaml").exists()


@pytest.mark.slow
def test_end_to_end_resume_with_real_trainer(run_config_dict: dict[str, Any]) -> None:
    config = RunConfig.from_dict({**run_config_dict, "trainer": {**run_config_dict["trainer"], "max_steps": 1}})
    first = run(config)
    assert first.state == "succeeded" and first.final_step == 1
    run_dir = Path(first.run_dir)

    config2 = RunConfig.from_dict({**run_config_dict, "trainer": {**run_config_dict["trainer"], "max_steps": 2}})
    resumed = run(config2, resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 2
    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-1", "step-2"]
    assert [row["step"] for _, row in iter_metrics(run_dir) if row["phase"] == "train"] == [1, 2]
    assert "resumed from" in (run_dir / "run.log").read_text()
