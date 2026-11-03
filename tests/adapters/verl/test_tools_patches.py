"""Tool-config generation, Tracking.log metric mapping, Ray runtime-env forwarding and the entry
wrapper, all against the fake ``verl`` package from ``conftest.py``."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from monorl.adapters.verl import PATCHES
from monorl.adapters.verl.patches import METRIC_MAP, map_metrics, phase_for
from monorl.adapters.verl.tools import TOOL_CLASS, build_tool_config, write_tool_config
from monorl.core.reporting import iter_metrics
from monorl.patching import apply, is_applied, register_patchset, revert, status


def test_build_tool_config_schema(make_config):
    cfg = make_config(env={"type": "tool_calling", "max_turns": 3, "params": {"tools": ["calculator", "lookup"], "corpus": {"a": "b"}}})
    tree = build_tool_config(cfg)
    names = [t["tool_schema"]["function"]["name"] for t in tree["tools"]]
    assert names == ["calculator", "lookup"]
    first = tree["tools"][0]
    assert first["class_name"] == TOOL_CLASS
    assert first["config"]["type"] == "native"
    assert first["config"]["tool_name"] == "calculator"
    assert first["config"]["env_params"] == {"tools": ["calculator", "lookup"], "corpus": {"a": "b"}}
    assert first["tool_schema"]["type"] == "function"
    assert first["tool_schema"]["function"]["parameters"]["type"] == "object"
    assert "expression" in first["tool_schema"]["function"]["parameters"]["properties"]


def test_write_tool_config_yaml(make_config, tmp_path: Path):
    cfg = make_config(env={"type": "tool_calling", "params": {"tools": ["calculator"]}})
    path = write_tool_config(cfg, tmp_path)
    loaded = yaml.safe_load(path.read_text())
    assert path.name == "verl_tools.yaml"
    assert [t["config"]["tool_name"] for t in loaded["tools"]] == ["calculator"]


def test_build_tool_config_requires_tool_calling(make_config):
    with pytest.raises(ValueError):
        build_tool_config(make_config())


def test_monorl_tool_executes(fake_verl, make_config):
    cfg = make_config(env={"type": "tool_calling", "params": {"tools": ["calculator"]}})
    spec = build_tool_config(cfg)["tools"][0]
    verl_tool = importlib.import_module("monorl.adapters.verl.verl_tool")
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import OpenAIFunctionToolSchema

    tool = verl_tool.MonorlTool(config=spec["config"], tool_schema=OpenAIFunctionToolSchema.model_validate(spec["tool_schema"]))
    assert isinstance(tool, BaseTool)
    assert tool.name == "calculator"

    async def drive():
        instance_id, _ = await tool.create(create_kwargs={})
        response, reward, metrics = await tool.execute(instance_id, {"expression": "2*(3+4)"}, agent_data=None)
        await tool.release(instance_id)
        return response, reward, metrics

    response, reward, metrics = asyncio.run(drive())
    assert response.text == "14"
    assert reward == 0.0
    assert "tool/latency_s" in metrics


def test_metric_mapping_table():
    data = {k: i + 1.0 for i, k in enumerate(METRIC_MAP)}
    data["timing_s/ref"] = 3.0
    data["some/string"] = "x"
    mapped = map_metrics(data)
    assert mapped["reward/mean"] == data["critic/score/mean"]
    assert mapped["reward/std"] == data["critic/score/std"]
    assert mapped["loss/policy"] == data["actor/pg_loss"]
    assert mapped["loss/kl"] == data["actor/kl_loss"]
    assert mapped["loss/entropy"] == data["actor/entropy"]
    assert mapped["policy/clipfrac"] == data["actor/pg_clipfrac"]
    assert mapped["tokens/response_mean"] == data["response_length/mean"]
    assert mapped["tokens/prompt_mean"] == data["prompt_length/mean"]
    assert mapped["lr"] == data["actor/lr"]
    assert mapped["time/rollout"] == data["timing_s/gen"]
    assert mapped["time/train"] == data["timing_s/update_actor"]
    assert mapped["time/ref"] == 3.0
    assert mapped["critic/score/mean"] == data["critic/score/mean"]  # raw key kept
    assert "some/string" not in mapped
    assert phase_for({"val-core/x": 1}) == "eval"
    assert phase_for({"actor/pg_loss": 1, "val-core/x": 1}) == "train"


def test_patchset_status_without_verl():
    if "verl" in sys.modules:
        pytest.skip("verl importable")
    register_patchset(PATCHES)
    report = apply("verl")
    assert not report.applied and not report.failed
    assert all("not installed" in reason for _, reason in report.skipped)
    ids = {s.id for s in status("verl", resolve=False)}
    assert ids == {"verl.tracking.run_dir", "verl.tracking.metrics_jsonl", "verl.ray_runtime_env.monorl"}


def test_tracking_patch_writes_metrics_jsonl(fake_verl, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    register_patchset(PATCHES)
    run_dir = tmp_path / "run"
    monkeypatch.setenv("MONORL_RUN_DIR", str(run_dir))
    report = apply("verl")
    assert report.installed_version == fake_verl.version
    assert set(report.applied) == {"verl.tracking.run_dir", "verl.tracking.metrics_jsonl", "verl.ray_runtime_env.monorl"}, report
    assert all(s.applied and s.verified for s in status("verl"))

    from verl.utils.tracking import Tracking

    tracking = Tracking("monorl", "exp", "console", config={"trainer": {"default_local_dir": str(run_dir / "checkpoints")}})
    tracking.log({"critic/score/mean": 0.75, "actor/pg_loss": -0.02, "timing_s/gen": 1.5, "actor/lr": 1e-6}, step=3)
    tracking.log(data={"val-core/monorl/reward/mean@1": 0.9}, step=3)
    assert tracking.console_logger.rows[0][1] == 3  # original still runs
    rows = [row for _, row in iter_metrics(run_dir)]
    assert len(rows) == 2
    assert rows[0]["step"] == 3 and rows[0]["phase"] == "train"
    assert rows[0]["reward/mean"] == 0.75
    assert rows[0]["loss/policy"] == -0.02
    assert rows[0]["time/rollout"] == 1.5
    assert rows[0]["lr"] == 1e-6
    assert rows[1]["phase"] == "eval"

    revert("verl")
    assert not is_applied("verl.tracking.metrics_jsonl", "verl")
    tracking.log({"critic/score/mean": 0.1}, step=4)
    assert len(list(iter_metrics(run_dir))) == 2


def test_tracking_patch_falls_back_to_default_local_dir(fake_verl, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    register_patchset(PATCHES)
    monkeypatch.delenv("MONORL_RUN_DIR", raising=False)
    apply("verl")
    from verl.utils.tracking import Tracking

    run_dir = tmp_path / "fallback"
    Tracking("p", "e", config={"trainer": {"default_local_dir": str(run_dir / "checkpoints")}}).log({"actor/entropy": 2.0}, 1)
    rows = [row for _, row in iter_metrics(run_dir)]
    assert rows and rows[0]["loss/entropy"] == 2.0


def test_runtime_env_patch_forwards_monorl_vars(fake_verl, monkeypatch: pytest.MonkeyPatch):
    register_patchset(PATCHES)
    monkeypatch.setenv("MONORL_RUN_DIR", "/runs/x")
    monkeypatch.setenv("PYTHONPATH", "/src")
    apply("verl")
    from verl.trainer import main_ppo

    env = main_ppo.get_ppo_ray_runtime_env()
    assert env["env_vars"]["MONORL_RUN_DIR"] == "/runs/x"
    assert env["env_vars"]["PYTHONPATH"] == "/src"
    assert env["env_vars"]["TOKENIZERS_PARALLELISM"] == "true"


def test_entry_wrapper_runs_patched_task_runner(fake_verl, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    monkeypatch.setenv("MONORL_RUN_DIR", str(run_dir))
    monkeypatch.setattr(sys, "argv", ["entry", "trainer.total_training_steps=1", "algorithm.adv_estimator=grpo"])
    entry = importlib.import_module("monorl.adapters.verl.entry")
    from verl.trainer import main_ppo
    from verl.utils import config as verl_config

    assert entry.CONFIG_PATH == "pkg://verl.trainer.config" and entry.CONFIG_NAME == "ppo_trainer"
    assert entry.MonorlTaskRunner.__ray_actor_class__.__mro__[1] is main_ppo.TaskRunnerV1.__ray_actor_class__
    entry.apply_patches()
    entry.main()
    assert main_ppo.CALLS[-1][1] is entry.MonorlTaskRunner
    assert verl_config.VALIDATED[-1][0]["algorithm"]["adv_estimator"] == "grpo"
    base = main_ppo.TaskRunnerV1.__ray_actor_class__
    assert base.RUNS, "TaskRunnerV1.run was not invoked"
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["reward/mean"] == 0.5 and rows[0]["loss/policy"] == -0.1 and rows[0]["step"] == 1
