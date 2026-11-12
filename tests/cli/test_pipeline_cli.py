"""``monorl pipeline`` and the ``monorl train`` auto-dispatch, with ``run_pipeline`` monkeypatched."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click import unstyle
from typer.testing import CliRunner

from monorl.cli.main import app
from monorl.config.pipeline import PipelineConfig


@pytest.fixture
def pipeline_yaml(run_config_dict: dict[str, Any], tmp_path: Path) -> Path:
    base = dict(run_config_dict)
    base.pop("run_name")
    payload = {
        "name": "cli-pipeline",
        "base": base,
        "stages": [
            {
                "type": "train",
                "name": "sft",
                "init_from": "base",
                "overrides": {"algorithm": {"name": "sft"}},
            },
            {"type": "rejection_sample", "name": "rs", "num_samples": 2},
            {"type": "train", "name": "sft2", "data_from": "previous"},
            {"type": "eval", "name": "eval"},
        ],
    }
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


@pytest.fixture
def fake_run_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> list[tuple[PipelineConfig, Path | None, bool]]:
    import monorl.pipeline.runner as runner_mod
    from monorl.pipeline.runner import PipelineResult

    calls: list[tuple[PipelineConfig, Path | None, bool]] = []

    def fake(config: PipelineConfig, run_dir: Path | None = None, *, resume: bool = False) -> PipelineResult:
        calls.append((config, run_dir, resume))
        stages = [
            {
                "index": i,
                "name": s.name,
                "type": s.type,
                "state": "succeeded",
                "step": 3,
                "run_dir": str(tmp_path / s.name),
            }
            for i, s in enumerate(config.stages)
        ]
        return PipelineResult(
            state="succeeded",
            run_dir=str(tmp_path / "pd"),
            stages=stages,
            final_model_path=str(tmp_path / "final"),
        )

    monkeypatch.setattr(runner_mod, "run_pipeline", fake)
    return calls


def test_help_lists_pipeline(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0 and "pipeline" in result.output
    result = runner.invoke(app, ["pipeline", "--help"])
    assert result.exit_code == 0
    help_text = unstyle(result.output)  # typer styles its help when GITHUB_ACTIONS / FORCE_COLOR is set
    for flag in ("--config", "--set", "--run-dir", "--resume"):
        assert flag in help_text


def test_pipeline_command_with_overrides(
    runner: CliRunner, pipeline_yaml: Path, fake_run_pipeline, tmp_path: Path
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "--config",
            str(pipeline_yaml),
            "--set",
            "base.trainer.max_steps=9",
            "--set",
            "name=overridden",
            "--run-dir",
            str(tmp_path / "pd"),
            "--resume",
        ],
    )
    assert result.exit_code == 0, result.output
    config, run_dir, resume = fake_run_pipeline[0]
    assert isinstance(config, PipelineConfig)
    assert config.name == "overridden" and config.base.trainer.max_steps == 9
    assert [s.name for s in config.stages] == ["sft", "rs", "sft2", "eval"]
    assert run_dir == tmp_path / "pd" and resume is True
    assert "pipeline dir:" in result.output and "state: succeeded" in result.output
    for name in ("sft", "rs", "sft2", "eval"):
        assert name in result.output
    assert "final model:" in result.output


def test_train_dispatches_pipeline_yaml(
    runner: CliRunner, pipeline_yaml: Path, fake_run_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monorl.orchestrator

    def boom(*a: Any, **kw: Any) -> None:
        raise AssertionError("orchestrator.run must not be called for a pipeline yaml")

    monkeypatch.setattr(monorl.orchestrator, "run", boom)
    result = runner.invoke(app, ["train", "--config", str(pipeline_yaml), "--set", "base.seed=7"])
    assert result.exit_code == 0, result.output
    config, run_dir, resume = fake_run_pipeline[0]
    assert config.base.seed == 7 and run_dir is None and resume is False


@pytest.mark.parametrize("state,code", [("failed", 1), ("cancelled", 130)])
def test_pipeline_exit_codes(
    runner: CliRunner, pipeline_yaml: Path, monkeypatch: pytest.MonkeyPatch, state: str, code: int
) -> None:
    import monorl.pipeline.runner as runner_mod
    from monorl.pipeline.runner import PipelineResult

    monkeypatch.setattr(
        runner_mod,
        "run_pipeline",
        lambda config, run_dir=None, *, resume=False: PipelineResult(state=state, run_dir="x", error="boom"),
    )
    result = runner.invoke(app, ["pipeline", "--config", str(pipeline_yaml)])
    assert result.exit_code == code
    assert "boom" in result.output


def test_pipeline_rejects_run_config_and_invalid(
    runner: CliRunner, run_config_dict: dict[str, Any], tmp_path: Path
) -> None:
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text(yaml.safe_dump(run_config_dict))
    result = runner.invoke(app, ["pipeline", "--config", str(run_yaml)])
    assert result.exit_code == 1 and "not a pipeline config" in result.output

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {"base": run_config_dict, "stages": [{"type": "train", "name": "x", "data_from": "previous"}]}
        )
    )
    result = runner.invoke(app, ["pipeline", "--config", str(bad)])
    assert result.exit_code == 1 and "rejection_sample" in result.output

    result = runner.invoke(app, ["pipeline", "--config", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 1 and "not found" in result.output


def test_recipes_show_pipeline_recipe(runner: CliRunner) -> None:
    result = runner.invoke(app, ["recipes", "show", "r1_sft_grpo_loop_cpu", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["kind"] == "pipeline"
    assert [s["type"] for s in payload["config"]["stages"]] == [
        "train",
        "train",
        "rejection_sample",
        "train",
        "train",
        "eval",
    ]
    assert payload["config"]["stages"][0]["overrides"]["algorithm"]["name"] == "sft"
    assert payload["config"]["stages"][3]["data_from"] == "previous"
