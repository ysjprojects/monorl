"""``algorithm.name: sft`` → ``examples/run_sft.py`` config + launch; images rejected."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from monorl.adapters.nemo_rl.backend import NemoRLBackend, build_command
from monorl.adapters.nemo_rl.compile import SFT_DATASET, SFT_PROCESSOR, compile_config, compile_sft_config
from monorl.adapters.nemo_rl.data import export_dataset, sft_sample_to_row
from monorl.adapters.nemo_rl.entry import HOME_ENV, parse_args
from monorl.adapters.nemo_rl.patches import translate_metrics
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.reporting import RunReporter, iter_metrics, read_status
from monorl.core.types import Sample


@pytest.fixture
def sft_config(nemo_config: RunConfig) -> RunConfig:
    data = nemo_config.to_dict()
    data["algorithm"] = {"name": "sft"}
    data["rewards"] = []
    data["trainer"] = {**data["trainer"], "eval_every": 1}
    data["data"] = {**data["data"], "eval_uri": data["data"]["uri"]}
    return RunConfig.from_dict(data)


def test_sft_config_snapshot(sft_config: RunConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cfg = compile_config(sft_config, run_dir)
    assert cfg == compile_sft_config(sft_config, run_dir)
    assert set(cfg) == {"sft", "checkpointing", "policy", "data", "logger", "cluster"}
    tr = sft_config.trainer
    assert cfg["sft"] == {
        "max_num_epochs": 2,
        "max_num_steps": 2,
        "val_period": 1,
        "val_batches": 8,
        "val_global_batch_size": 2,
        "val_micro_batch_size": 2,
        "val_at_start": False,
        "val_at_end": True,
        "seed": 42,
        "only_unmask_final": False,
    }
    policy = cfg["policy"]
    assert policy["train_global_batch_size"] == tr.prompts_per_step == 2
    assert policy["train_micro_batch_size"] == tr.micro_batch_size and policy["precision"] == "bfloat16"
    assert "generation" not in policy and "logprob_batch_size" not in policy
    assert policy["optimizer"]["kwargs"]["lr"] == tr.learning_rate
    data = cfg["data"]
    assert data["train"] == {
        "dataset_name": SFT_DATASET,
        "data_path": str(run_dir.resolve() / "nemo_rl" / "sft_train.jsonl"),
        "split_validation_size": 0,
        "seed": 42,
    }
    assert data["validation"] == {
        "dataset_name": SFT_DATASET,
        "data_path": str(run_dir.resolve() / "nemo_rl" / "sft_eval.jsonl"),
    }
    assert data["default"] == {"prompt_file": None, "system_prompt_file": None, "processor": SFT_PROCESSOR}
    assert data["add_bos"] is True and data["add_eos"] is True and data["add_generation_prompt"] is False
    assert data["max_input_seq_length"] == tr.max_seq_len
    assert (
        cfg["checkpointing"]["metric_name"] == "val:val_loss"
        and cfg["checkpointing"]["higher_is_better"] is False
    )
    assert cfg["checkpointing"]["checkpoint_dir"] == str(run_dir.resolve() / "checkpoints")
    assert cfg["logger"]["monorl"]["run_dir"] == str(run_dir.resolve())
    yaml.safe_dump(cfg)

    no_eval = RunConfig.from_dict(
        {**sft_config.to_dict(), "data": {**sft_config.data.model_dump(mode="json"), "eval_uri": None}}
    )
    cfg2 = compile_sft_config(no_eval, run_dir)
    assert (
        cfg2["sft"]["val_period"] == 0
        and cfg2["data"]["validation"] is None
        and cfg2["checkpointing"]["metric_name"] is None
    )


def test_sft_skips_rl_only_checks_and_honours_overrides(sft_config: RunConfig, tmp_path: Path) -> None:
    data = sft_config.to_dict()
    data["algorithm"] = {"name": "sft", "num_epochs": 3, "entropy_coef": 0.1}
    data["rollout"] = {**data["rollout"], "backend": "sglang"}
    data["trainer"]["extra"] = {"nemo_rl": {"sft": {"val_batches": 2}}}
    cfg = compile_config(RunConfig.from_dict(data), tmp_path)
    assert cfg["sft"]["val_batches"] == 2


def test_images_are_rejected(nemo_config: RunConfig, sft_config: RunConfig, tmp_path: Path) -> None:
    for base in (nemo_config, sft_config):
        data = base.to_dict()
        data["data"] = {**data["data"], "image_field": "picture"}
        with pytest.raises(ConfigError, match="run_vlm_grpo.py"):
            compile_config(RunConfig.from_dict(data), tmp_path)
    img = Sample(
        id="i",
        messages=[{"role": "user", "content": [{"type": "image", "image": "x.png"}]}],
        metadata={"response": "r"},
    )
    with pytest.raises(ConfigError, match="run_vlm_sft.py"):
        sft_sample_to_row(img, response_field=None)


def test_sft_rows_and_export(sft_config: RunConfig, tmp_path: Path) -> None:
    sample = Sample(
        id="s", messages=[{"role": "user", "content": "2+2?"}], reference="4", metadata={"response": "4"}
    )
    assert sft_sample_to_row(sample, response_field=None) == {
        "messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
    }
    exported = export_dataset(sft_config, tmp_path)
    assert exported.num_train == 4 and exported.num_eval == 4
    rows = [json.loads(line) for line in exported.paths.sft_train_jsonl.read_text().splitlines()]
    assert all(set(r) == {"messages"} and r["messages"][-1]["role"] == "assistant" for r in rows)
    assert {r["messages"][-1]["content"] for r in rows} == {"4", "8", "6", "21"}
    assert not exported.paths.train_jsonl.exists()


def test_sft_metric_names() -> None:
    out = translate_metrics({"loss": 0.5, "lr": 1e-6, "global_valid_toks": 40}, "train", algorithm="sft")
    assert out == {"loss/sft": 0.5, "lr": 1e-6, "tokens/valid": 40.0}
    assert translate_metrics({"val_loss": 0.7}, "validation", algorithm="sft") == {"eval/loss": 0.7}
    assert translate_metrics({"loss": 0.5}, "train", algorithm="grpo") == {"loss/policy": 0.5}


def test_sft_launch_command_and_entry_args(
    sft_config: RunConfig, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("monorl.adapters.nemo_rl.backend.shutil.which", lambda name: None)
    cmd = build_command(sft_config, fake_home, tmp_path / "x.yaml", tmp_path)
    assert cmd[-2:] == ["--script", "examples/run_sft.py"]
    args, _ = parse_args(["--config", "c.yaml", "--script", "examples/run_sft.py"])
    assert args.script == "examples/run_sft.py"
    assert parse_args(["--config", "c.yaml"])[0].script == "examples/run_grpo.py"


def test_sft_run_end_to_end_with_fake_checkout(
    sft_config: RunConfig, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(HOME_ENV, str(fake_home))
    data = sft_config.to_dict()
    data["trainer"]["extra"] = {"nemo_rl_launcher": "python"}
    cfg = RunConfig.from_dict(data)
    run_dir = tmp_path / "run"
    result = NemoRLBackend(cfg).run(cfg, run_dir, reporter=RunReporter(run_dir, echo=False))

    assert result.state == "succeeded", (run_dir / "run.log").read_text()
    assert result.final_step == 2
    child: dict[str, Any] = json.loads((run_dir / "child_env.json").read_text())
    assert child["argv"][0].endswith("examples/run_sft.py") and child["algorithm"] == "sft"
    assert child["sections"] == ["checkpointing", "cluster", "data", "logger", "policy", "sft"]
    assert child["processor"] == SFT_PROCESSOR and child["dataset_name"] == SFT_DATASET
    assert len(child["rows"]) == 4 and child["rows"][0]["messages"][-1]["role"] == "assistant"

    rows = [r for _, r in iter_metrics(run_dir)]
    train = [r for r in rows if r["phase"] == "train"]
    assert [r["loss/sft"] for r in train] == [1.0, 0.5] and all("loss/policy" not in r for r in train)
    assert [r for r in rows if r["phase"] == "eval"][0]["eval/loss"] == 0.75
    assert read_status(run_dir)["state"] == "succeeded"
    assert "sft done" in (run_dir / "run.log").read_text()
