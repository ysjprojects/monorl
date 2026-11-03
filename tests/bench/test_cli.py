from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from monorl.bench.cli import app
from monorl.distributed.cluster import ClusterInfo
from monorl.distributed.placement import PlacementPlan

from .conftest import FAKE_ENGINE_NAME


@pytest.fixture
def recipe(run_config_dict, tmp_path: Path) -> Path:
    cfg = dict(run_config_dict)
    cfg["rollout"] = {**cfg["rollout"], "backend": FAKE_ENGINE_NAME}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_help_lists_subcommands(runner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("rollout", "env", "sync", "plan"):
        assert name in result.output


def test_rollout_command_with_registered_fake_engine(runner, registered_fake_engine, recipe: Path, tmp_path: Path) -> None:
    out = tmp_path / "rollout.json"
    result = runner.invoke(
        app,
        ["rollout", "--config", str(recipe), "--prompts", "4", "--batch-sizes", "1,2,8", "--seconds", "0", "--json", str(out), "--no-warmup"],
    )
    assert result.exit_code == 0, result.output
    assert "rollout throughput" in result.output
    assert registered_fake_engine.calls == [1, 1, 1, 1, 2, 2, 4]
    assert registered_fake_engine.shutdowns == 1
    data = json.loads(out.read_text())
    assert data["kind"] == "rollout" and data["backend"] == FAKE_ENGINE_NAME
    assert [r["batch_size"] for r in data["results"]] == [1, 2, 8]
    assert data["results"][0]["requests"] == 4 * 2  # 4 prompts × group_size 2
    assert data["results"][2]["prompts"] == 4  # fewer prompts than batch size → one smaller batch
    assert data["sampling"]["n"] == 2 and data["sampling"]["max_new_tokens"] == 8
    assert data["best_batch_size"] in (1, 2, 8)


def test_rollout_rejects_bad_batch_sizes(runner, registered_fake_engine, recipe: Path) -> None:
    result = runner.invoke(app, ["rollout", "--config", str(recipe), "--batch-sizes", "1,x"])
    assert result.exit_code == 2
    result = runner.invoke(app, ["rollout", "--config", str(recipe), "--batch-sizes", "0"])
    assert result.exit_code == 2


def test_rollout_unknown_backend_fails_cleanly(runner, recipe: Path) -> None:
    result = runner.invoke(app, ["rollout", "--config", str(recipe), "--set", "rollout.backend=does-not-exist"])
    assert result.exit_code == 1
    assert "does-not-exist" in result.output


def test_env_command_stdio(runner, mep_command, tmp_path: Path) -> None:
    out = tmp_path / "env.json"
    args = ["env", "--parallel", "1,4", "--episodes", "6", "--turns", "2", "--json", str(out)]
    for c in mep_command:
        args += ["--command", c]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "env throughput" in result.output
    data = json.loads(out.read_text())
    assert [r["parallel"] for r in data["results"]] == [1, 4]
    assert all(r["episodes"] == 6 and r["steps"] == 12 for r in data["results"])
    assert data["spec"]["parallel_episodes"] == 2


def test_env_command_requires_one_target(runner) -> None:
    assert runner.invoke(app, ["env"]).exit_code == 2
    assert runner.invoke(app, ["env", "--command", "x", "--url", "http://h"]).exit_code == 2


def test_sync_command_fake_params(runner, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    out = tmp_path / "sync.json"
    result = runner.invoke(
        app, ["sync", "--fake-params", "100000", "--engines", "2", "--repeats", "2", "--strategy", "inprocess", "--strategy", "checkpoint", "--json", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "weight sync" in result.output
    data = json.loads(out.read_text())
    assert data["source"] == "synthetic" and data["num_params"] == 100_000 and data["engines"] == 2
    assert {r["strategy"] for r in data["results"]} == {"inprocess", "checkpoint"}
    assert all(not r["skipped"] and r["repeats"] == 2 for r in data["results"])
    assert data["fastest"] in ("inprocess", "checkpoint")


def test_sync_command_needs_source(runner) -> None:
    assert runner.invoke(app, ["sync"]).exit_code == 2


def test_plan_command_with_fake_plan_placement(runner, recipe: Path, tmp_path: Path, monkeypatch) -> None:
    import monorl.distributed

    def fake_plan(config, cluster: ClusterInfo) -> PlacementPlan:
        n = cluster.num_gpus
        return PlacementPlan(
            mode="local",
            placement="colocated",
            num_nodes=1,
            gpus_per_node=n,
            total_gpus=n,
            rollout_gpus=n,
            trainer_gpus=n,
            engine_replicas=1,
            engine_tensor_parallel=1,
            trainer_world_size=max(1, n),
            engine_sleep=n > 0,
            weight_sync="inprocess",
            device="cuda" if n else "cpu",
            rollout_devices=[f"cuda:{i}" for i in range(n)] or ["cpu"],
            trainer_devices=[f"cuda:{i}" for i in range(n)] or ["cpu"],
            model_estimate=None,
            notes=[f"fake plan for {n} gpus"],
        )

    monkeypatch.setattr(monorl.distributed, "plan_placement", fake_plan)
    out = tmp_path / "plan.json"
    result = runner.invoke(app, ["plan", "--config", str(recipe), "--gpus", "4", "--gpu-mem-gb", "40", "--json", str(out)])
    assert result.exit_code == 0, result.output
    assert "placement plan" in result.output
    assert "fake plan for 4 gpus" in result.output
    data = json.loads(out.read_text())
    assert data["plan"]["total_gpus"] == 4 and data["cluster"]["num_gpus"] == 4
    assert data["cluster"]["cuda_devices"][0]["total_memory_bytes"] == 40 << 30
    assert data["estimate"] is None
    assert any("no step-time estimate" in n for n in data["notes"])


def test_plan_command_rejects_unknown_preset(runner, recipe: Path, monkeypatch) -> None:
    import monorl.distributed

    calls: list[int] = []
    monkeypatch.setattr(monorl.distributed, "plan_placement", lambda c, cl: calls.append(1))
    result = runner.invoke(app, ["plan", "--config", str(recipe), "--gpus", "1", "--preset", "tpu"])
    assert result.exit_code == 1
    assert "unknown hardware preset" in result.output
    assert calls == []
