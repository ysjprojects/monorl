"""``run_pipeline`` against the orchestrator fakes: manifest states, model/data hand-off, resume,
cancellation."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any

import fakes
import pytest
import yaml

from monorl.core.reporting import iter_metrics, iter_trajectories, read_status
from monorl.pipeline import PipelineResult, read_manifest, run_pipeline
from monorl.pipeline.stages import DATASET_FILE, MANIFEST_FILE, PIPELINE_CONFIG_FILE


def _stage_config(stage: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load((Path(stage["run_dir"]) / "config.yaml").read_text())


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_full_pipeline_manifest_and_handoff(make_pipeline: Any, tmp_path: Path) -> None:
    config = make_pipeline()
    result = run_pipeline(config)

    assert isinstance(result, PipelineResult)
    assert result.state == "succeeded", result.error
    run_dir = Path(result.run_dir)
    assert run_dir == (tmp_path / "runs" / "fake-pipeline").resolve()
    assert (run_dir / PIPELINE_CONFIG_FILE).exists() and (run_dir / MANIFEST_FILE).exists()

    status = read_status(run_dir)
    assert status["state"] == "succeeded" and status["step"] == 5

    manifest = read_manifest(run_dir)
    assert manifest["name"] == "fake-pipeline" and manifest["state"] == "succeeded"
    stages = manifest["stages"]
    assert [s["name"] for s in stages] == ["warmup", "grpo_1", "rs", "sft_like", "eval"]
    assert [s["type"] for s in stages] == ["train", "train", "rejection_sample", "train", "eval"]
    assert all(s["state"] == "succeeded" for s in stages)
    assert [Path(s["run_dir"]).name for s in stages] == [
        "00-warmup",
        "01-grpo_1",
        "02-rs",
        "03-sft_like",
        "04-eval",
    ]
    assert all(Path(s["run_dir"]).parent == run_dir / "stages" for s in stages)
    assert all(s["started_at"] is not None and s["finished_at"] is not None for s in stages)
    assert result.stages == stages

    # model hand-off: warmup starts from the base model, every later train stage from the previous final/
    warmup, grpo_1, rs, sft_like, eval_stage = stages
    assert _stage_config(warmup)["model"]["name_or_path"] == config.base.model.name_or_path
    assert _stage_config(grpo_1)["model"]["name_or_path"] == str(Path(warmup["run_dir"]) / "final")
    assert _stage_config(sft_like)["model"]["name_or_path"] == str(Path(grpo_1["run_dir"]) / "final")
    assert (Path(sft_like["run_dir"]) / "final" / "config.json").exists()
    assert result.final_model_path == str(Path(sft_like["run_dir"]) / "final")
    assert manifest["final_model_path"] == result.final_model_path
    # sampling stages run the current policy
    assert _stage_config(rs)["model"]["name_or_path"] == str(Path(grpo_1["run_dir"]) / "final")
    assert _stage_config(rs)["rollout"]["temperature"] == 0.7
    assert _stage_config(eval_stage)["model"]["name_or_path"] == result.final_model_path
    engines_by_model = [e.config.model.name_or_path for e in fakes.ENGINES]
    assert engines_by_model[2] == _stage_config(rs)["model"]["name_or_path"]
    assert all(e.shutdown_count >= 1 for e in fakes.ENGINES)

    # train stages: standard run dirs with their own status + step counts
    assert warmup["step"] == 1 and grpo_1["step"] == 2 and sft_like["step"] == 1
    assert read_status(grpo_1["run_dir"])["state"] == "succeeded"
    assert "reward/mean" in grpo_1["summary"] and grpo_1["summary"]["final_model_path"].endswith("final")

    # rejection sampling: dataset + metrics + trajectories, summary counts
    dataset = Path(rs["run_dir"]) / DATASET_FILE
    assert dataset.exists() and rs["summary"]["dataset"] == str(dataset)
    rows = _rows(dataset)
    assert rows and rs["summary"]["rows"] == len(rows) == rs["summary"]["accepted"]
    assert rs["summary"]["sampled"] == 6 * 2 and rs["summary"]["prompts"] == 6
    assert 0 < rs["summary"]["accept_rate"] <= 1
    for row in rows:
        assert row["source"] == "sampled" and row["reward"] >= 1.0
        assert row["messages"][-1]["role"] == "assistant"
        assert row["messages"][-1]["content"] == row["reference"]  # exact-match accepted
        assert row["messages"][0]["role"] == "user"
    metric_rows = [r for _, r in iter_metrics(rs["run_dir"]) if r["phase"] == "rejection"]
    assert metric_rows and {
        "rs/prompts",
        "rs/sampled",
        "rs/accepted",
        "rs/accept_rate",
        "reward/mean",
    } <= set(metric_rows[0])
    assert sum(r["rs/accepted"] for r in metric_rows) == len(rows)
    assert [r for _, r in iter_trajectories(rs["run_dir"])]
    assert read_status(rs["run_dir"])["state"] == "succeeded"

    # data_from: previous → the SFT-like stage reads the rejection dataset in chat format
    sft_data = _stage_config(sft_like)["data"]
    assert (
        sft_data["uri"] == str(dataset)
        and sft_data["format"] == "chat"
        and sft_data["messages_field"] == "messages"
    )
    assert sft_data["prompt_field"] is None and sft_data["answer_field"] is None
    assert sft_data["max_prompt_tokens"] == config.base.data.max_prompt_tokens

    # eval: greedy, logs eval/* metrics
    assert eval_stage["summary"]["eval/num_samples"] == 3.0
    assert "eval/reward_mean" in eval_stage["summary"]
    eval_rows = [r for _, r in iter_metrics(eval_stage["run_dir"]) if r["phase"] == "eval"]
    assert eval_rows[-1]["eval/reward_mean"] == eval_stage["summary"]["eval/reward_mean"]
    eval_engine = fakes.ENGINES[-1]
    assert all(call["temperature"] == 0.0 for call in eval_engine.calls)

    log = (run_dir / "run.log").read_text()
    assert "stage 1/5" in log and "stage 5/5" in log


def test_mix_datasets_and_filters(make_pipeline: Any, tmp_path: Path) -> None:
    mix_file = fakes.write_jsonl(
        tmp_path / "mix.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ]
            },
            {"messages": [{"role": "user", "content": "no answer"}]},
        ],
    )
    stages = [
        {
            "type": "rejection_sample",
            "name": "rs",
            "num_samples": 2,
            "min_reward": 1.0,
            "max_prompts": 3,
            "mix": [str(mix_file)],
            "rollout_overrides": {"extra": {"fake_mode": "always_right", "max_parallel_episodes": 4}},
        },
        {
            "type": "train",
            "name": "sft_like",
            "data_from": "previous",
            "overrides": {"trainer": {"max_steps": 1}},
        },
    ]
    config = make_pipeline(stages)
    result = run_pipeline(config)
    assert result.state == "succeeded", result.error
    rs = result.stages[0]
    rows = _rows(Path(rs["summary"]["dataset"]))
    sampled = [r for r in rows if r["source"] == "sampled"]
    mixed = [r for r in rows if r["source"] == "mix"]
    # always_right + dedupe + max_per_prompt=1 → exactly one row per prompt (3 prompts)
    assert len(sampled) == 3 and rs["summary"]["accepted"] == 3 and rs["summary"]["sampled"] == 6
    assert len(mixed) == 1 and mixed[0]["messages"][-1] == {"role": "assistant", "content": "hi there"}
    assert mixed[0]["reward"] is None and rs["summary"]["mixed"] == 1 and rs["summary"]["rows"] == 4


def test_nothing_accepted_fails_stage(make_pipeline: Any) -> None:
    stages = [
        {"type": "rejection_sample", "name": "rs", "num_samples": 1, "min_reward": 5.0},
        {"type": "train", "name": "after", "data_from": "previous"},
    ]
    result = run_pipeline(make_pipeline(stages))
    assert result.state == "failed"
    assert result.stages[0]["state"] == "failed" and "accepted 0" in result.stages[0]["error"]
    assert result.stages[1]["state"] == "pending"
    assert read_status(result.run_dir)["state"] == "failed"
    assert "accepted 0" in read_status(result.run_dir)["message"]


def test_invalid_override_fails_before_running(make_pipeline: Any) -> None:
    stages = [{"type": "train", "name": "bad", "overrides": {"trainer": {"max_steps": "many"}}}]
    result = run_pipeline(make_pipeline(stages))
    assert result.state == "failed" and "invalid stage configuration" in (result.error or "")
    assert result.stages[0]["state"] == "pending"
    assert not fakes.TRAINERS


def test_explicit_run_dir_and_generated_name(make_pipeline: Any, tmp_path: Path) -> None:
    config = make_pipeline([{"type": "train", "name": "only", "overrides": {"trainer": {"max_steps": 1}}}])
    config = config.model_copy(update={"name": None})
    target = tmp_path / "elsewhere" / "p"
    result = run_pipeline(config, target)
    assert result.state == "succeeded"
    assert Path(result.run_dir) == target.resolve()
    manifest = read_manifest(target)
    assert manifest["name"].startswith("fake-model-pipeline-")
    saved = yaml.safe_load((target / PIPELINE_CONFIG_FILE).read_text())
    assert saved["name"] == manifest["name"]


def test_resume_skips_succeeded_stages(make_pipeline: Any, tmp_path: Path) -> None:
    stages = [
        {"type": "train", "name": "first", "overrides": {"trainer": {"max_steps": 1}}},
        {"type": "rejection_sample", "name": "rs", "num_samples": 2, "min_reward": 1.0},
        {
            "type": "train",
            "name": "second",
            "data_from": "previous",
            "overrides": {"trainer": {"max_steps": 2, "extra": {"fake_fail_at": 1}}},
        },
        {"type": "train", "name": "third", "overrides": {"trainer": {"max_steps": 1}}},
    ]
    first = run_pipeline(make_pipeline(stages))
    assert first.state == "failed"
    assert [s["state"] for s in first.stages] == ["succeeded", "succeeded", "failed", "pending"]
    assert "fake trainer failure" in first.stages[2]["error"]
    run_dir = Path(first.run_dir)
    dataset_before = first.stages[1]["summary"]["dataset"]
    first_final = str(Path(first.stages[0]["run_dir"]) / "final")

    fakes.reset_recorders()
    fixed = [*stages]
    fixed[2] = {
        "type": "train",
        "name": "second",
        "data_from": "previous",
        "overrides": {"trainer": {"max_steps": 2}},
    }
    resumed = run_pipeline(make_pipeline(fixed), run_dir, resume=True)
    assert resumed.state == "succeeded", resumed.error
    assert [s["state"] for s in resumed.stages] == ["succeeded"] * 4
    # the first two stages were not re-run: no new engine for them, outputs reused
    assert len(fakes.TRAINERS) == 2  # second + third
    assert resumed.stages[1]["summary"]["dataset"] == dataset_before
    second_cfg = _stage_config(resumed.stages[2])
    assert second_cfg["data"]["uri"] == dataset_before
    assert second_cfg["model"]["name_or_path"] == first_final
    assert _stage_config(resumed.stages[3])["model"]["name_or_path"] == str(
        Path(resumed.stages[2]["run_dir"]) / "final"
    )
    assert read_status(run_dir)["step"] == 4
    assert "already succeeded; reusing" in (run_dir / "run.log").read_text()


def test_resume_rejects_mismatched_stages(make_pipeline: Any) -> None:
    stages = [{"type": "train", "name": "a", "overrides": {"trainer": {"max_steps": 1}}}]
    first = run_pipeline(make_pipeline(stages))
    assert first.state == "succeeded"
    other = make_pipeline([{"type": "train", "name": "b", "overrides": {"trainer": {"max_steps": 1}}}])
    result = run_pipeline(other, Path(first.run_dir), resume=True)
    assert result.state == "failed" and "do not match" in (result.error or "")


def _sigterm_self() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


def test_sigterm_during_train_stage_cancels_pipeline(
    make_pipeline: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = [
        {"type": "train", "name": "quick", "overrides": {"trainer": {"max_steps": 1}}},
        {
            "type": "train",
            "name": "long",
            "overrides": {"trainer": {"max_steps": 10_000, "save_every": 1_000}},
        },
        {"type": "eval", "name": "eval"},
    ]
    config = make_pipeline(stages)
    original = fakes.FakeTrainer.train_step

    def train_step(self: Any, batch: Any) -> dict[str, float]:
        out = original(self, batch)
        # SIGTERM lands inside step 2 of the long stage, while the orchestrator's handler is active
        if self.config.run_name == "01-long" and self.step == 2:
            _sigterm_self()
        return out

    monkeypatch.setattr(fakes.FakeTrainer, "train_step", train_step)
    result = run_pipeline(config)
    assert result.state == "cancelled"
    assert [s["state"] for s in result.stages] == ["succeeded", "cancelled", "pending"]
    assert result.stages[1]["step"] == 2  # the interrupted step was finished
    assert read_status(result.run_dir)["state"] == "cancelled"
    assert "cancelled after 1 of 3 stages" in read_status(result.run_dir)["message"]
    assert read_status(result.stages[1]["run_dir"])["state"] == "cancelled"
    assert read_manifest(result.run_dir)["state"] == "cancelled"
    assert result.final_model_path is None
    assert "received SIGTERM" in (Path(result.stages[1]["run_dir"]) / "run.log").read_text()


def test_sigterm_during_rejection_stage_cancels_pipeline(
    make_pipeline: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = [
        {
            "type": "rejection_sample",
            "name": "rs",
            "num_samples": 2,
            "min_reward": 0.0,
            "rollout_overrides": {"extra": {"rejection_prompts_per_step": 1}},
        },
        {"type": "train", "name": "after", "data_from": "previous"},
    ]
    config = make_pipeline(stages)
    original = fakes.FakeEngine.generate
    fired = {"done": False}

    def generate(self: Any, prompts: Any, params: Any) -> Any:
        if not fired["done"]:
            fired["done"] = True
            _sigterm_self()  # inside the first chunk: the pipeline's own handler sets cancel
        return original(self, prompts, params)

    monkeypatch.setattr(fakes.FakeEngine, "generate", generate)
    result = run_pipeline(config)
    assert result.state == "cancelled"
    assert [s["state"] for s in result.stages] == ["cancelled", "pending"]
    rs = result.stages[0]
    assert rs["step"] <= 1  # episodes not yet started abort at once; the next chunk never starts
    assert not (Path(rs["run_dir"]) / DATASET_FILE).exists()
    assert read_status(rs["run_dir"])["state"] == "cancelled"
    assert read_status(result.run_dir)["state"] == "cancelled"
    assert "received SIGTERM" in (Path(result.run_dir) / "run.log").read_text()
    assert fakes.ENGINES[-1].shutdown_count >= 1


def test_signal_handlers_restored(make_pipeline: Any) -> None:
    before = signal.getsignal(signal.SIGTERM)
    run_pipeline(make_pipeline([{"type": "train", "name": "a", "overrides": {"trainer": {"max_steps": 1}}}]))
    assert signal.getsignal(signal.SIGTERM) == before


@pytest.mark.parametrize("init_from", ["base", "previous"])
def test_init_from_base_vs_previous(make_pipeline: Any, init_from: str) -> None:
    stages = [
        {"type": "train", "name": "a", "overrides": {"trainer": {"max_steps": 1}}},
        {"type": "train", "name": "b", "init_from": init_from, "overrides": {"trainer": {"max_steps": 1}}},
    ]
    config = make_pipeline(stages)
    result = run_pipeline(config)
    assert result.state == "succeeded"
    b_model = _stage_config(result.stages[1])["model"]["name_or_path"]
    if init_from == "base":
        assert b_model == config.base.model.name_or_path
    else:
        assert b_model == str(Path(result.stages[0]["run_dir"]) / "final")
