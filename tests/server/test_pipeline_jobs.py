"""Pipeline jobs through the API: creation (``{pipeline: ...}``), ``kind``, stage routes, recipes
and schema; driven by ``fake_pipeline.py`` via ``MONORL_PIPELINE_CMD``."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from monorl.server.jobs import TERMINAL_STATES, JobManager, pipeline_command
from monorl.server.routes_system import load_recipes

FAKE_PIPELINE = Path(__file__).with_name("fake_pipeline.py")


@pytest.fixture(autouse=True)
def pipeline_cmd(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONORL_PIPELINE_CMD", shlex.join([sys.executable, str(FAKE_PIPELINE)]))
    monkeypatch.delenv("FAKE_PIPELINE_MODE", raising=False)


@pytest.fixture
def pipeline_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    base = dict(config_dict)
    base.pop("run_name")
    return {
        "name": "srv-pipeline",
        "base": base,
        "stages": [
            {
                "type": "train",
                "name": "sft",
                "init_from": "base",
                "overrides": {"algorithm": {"name": "sft"}},
            },
            {"type": "train", "name": "grpo", "overrides": {"trainer": {"max_steps": 2}}},
            {"type": "rejection_sample", "name": "rs", "num_samples": 2},
            {"type": "eval", "name": "eval"},
        ],
    }


@pytest.fixture
def terminal(client: TestClient, wait_for):  # noqa: ANN001, ANN201
    def _terminal(job_id: str, timeout: float = 20.0) -> dict[str, Any]:
        def check():  # noqa: ANN202
            job = client.get(f"/api/jobs/{job_id}").json()
            return job if job["state"] in TERMINAL_STATES else None

        return wait_for(check, timeout=timeout)

    return _terminal


def test_pipeline_command_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONORL_PIPELINE_CMD", raising=False)
    assert pipeline_command()[-3:] == ["-m", "monorl.cli", "pipeline"]
    monkeypatch.setenv("MONORL_PIPELINE_CMD", "/usr/bin/env 'my pipeline' --flag")
    assert pipeline_command() == ["/usr/bin/env", "my pipeline", "--flag"]


def test_create_pipeline_job_lifecycle(
    client: TestClient, pipeline_dict: dict[str, Any], tmp_path: Path, terminal
) -> None:
    r = client.post("/api/jobs", json={"pipeline": pipeline_dict})
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["kind"] == "pipeline" and job["name"] == "srv-pipeline" and job["state"] == "running"
    assert job["config"]["name"] == "srv-pipeline" and [s["name"] for s in job["config"]["stages"]] == [
        "sft",
        "grpo",
        "rs",
        "eval",
    ]
    run_dir = Path(job["run_dir"])
    assert run_dir == tmp_path / "runs" / "srv-pipeline"
    assert job["command"][-4:] == ["--config", str(run_dir / "pipeline.yaml"), "--run-dir", str(run_dir)]
    saved = yaml.safe_load((run_dir / "pipeline.yaml").read_text())
    assert saved["name"] == "srv-pipeline" and len(saved["stages"]) == 4

    final = terminal(job["id"])
    assert final["state"] == "succeeded", final
    assert final["step"] == 4 and final["kind"] == "pipeline"
    assert final["status"]["state"] == "succeeded"
    listed = client.get("/api/jobs").json()
    assert listed[0]["id"] == job["id"] and listed[0]["kind"] == "pipeline"

    stages = client.get(f"/api/jobs/{job['id']}/stages").json()
    assert stages["job_id"] == job["id"] and stages["kind"] == "pipeline" and stages["state"] == "succeeded"
    assert [s["name"] for s in stages["stages"]] == ["sft", "grpo", "rs", "eval"]
    assert all(s["state"] == "succeeded" and s["step"] == 2 for s in stages["stages"])
    assert stages["final_model_path"] == stages["stages"][1]["summary"]["final_model_path"]
    assert stages["stages"][2]["summary"]["accepted"] == 3

    one = client.get(f"/api/jobs/{job['id']}/stages/1").json()
    assert one["name"] == "grpo" and one["status"]["state"] == "succeeded"

    metrics = client.get(f"/api/jobs/{job['id']}/stages/1/metrics").json()
    assert [row["step"] for row in metrics["rows"]] == [1, 2] and metrics["next"] == 2
    assert metrics["rows"][0]["reward/mean"] == 0.25
    page = client.get(f"/api/jobs/{job['id']}/stages/1/metrics", params={"since": 1, "limit": 1}).json()
    assert len(page["rows"]) == 1 and page["rows"][0]["step"] == 2 and page["next"] == 2
    rs_metrics = client.get(f"/api/jobs/{job['id']}/stages/2/metrics").json()
    assert rs_metrics["rows"][0]["phase"] == "rejection" and "rs/accept_rate" in rs_metrics["rows"][0]
    eval_metrics = client.get(f"/api/jobs/{job['id']}/stages/3/metrics").json()
    assert eval_metrics["rows"][-1]["eval/reward_mean"] == 0.75

    logs = client.get(f"/api/jobs/{job['id']}/stages/0/logs").json()
    assert any("stage sft step 2" in line for line in logs["lines"]) and logs["next"] == len(logs["lines"])
    tail = client.get(f"/api/jobs/{job['id']}/stages/0/logs", params={"tail": 1}).json()
    assert len(tail["lines"]) == 1

    traj = client.get(f"/api/jobs/{job['id']}/stages/2/trajectories").json()
    assert len(traj["rows"]) == 2 and traj["rows"][0]["response"] == "r"

    ckpt = client.get(f"/api/jobs/{job['id']}/stages/1/checkpoints").json()
    assert ckpt["checkpoints"] == [
        {"step": 2, "path": str(Path(stages["stages"][1]["run_dir"]) / "checkpoints" / "step-2")}
    ]
    assert ckpt["final"] == str(Path(stages["stages"][1]["run_dir"]) / "final")
    assert client.get(f"/api/jobs/{job['id']}/stages/3/checkpoints").json() == {
        "checkpoints": [],
        "final": None,
    }

    # job-level readers still work on the pipeline dir itself
    assert "fake pipeline stdout line" in "\n".join(client.get(f"/api/jobs/{job['id']}/logs").json()["lines"])
    assert client.get(f"/api/jobs/{job['id']}/stages/9/metrics").status_code == 404
    assert client.get("/api/jobs/nope/stages").status_code == 404


def test_bare_pipeline_payload_and_run_kind(
    client: TestClient, pipeline_dict: dict[str, Any], config_dict: dict[str, Any], terminal
) -> None:
    job = client.post("/api/jobs", json=pipeline_dict).json()
    assert job["kind"] == "pipeline"
    assert terminal(job["id"])["state"] == "succeeded"
    run = client.post("/api/jobs", json={"config": config_dict}).json()
    assert run["kind"] == "run"
    assert client.get(f"/api/jobs/{run['id']}/stages").json()["stages"] == []
    assert client.get(f"/api/jobs/{run['id']}/stages/0/metrics").status_code == 404
    assert terminal(run["id"])["state"] == "succeeded"


def test_invalid_pipeline_is_422(client: TestClient, pipeline_dict: dict[str, Any]) -> None:
    bad = dict(pipeline_dict)
    bad["stages"] = [{"type": "train", "name": "x", "data_from": "previous"}]
    r = client.post("/api/jobs", json={"pipeline": bad})
    assert r.status_code == 422
    assert any("rejection_sample" in e["msg"] for e in r.json()["errors"])
    assert client.post("/api/jobs", json={"pipeline": "nope"}).status_code == 422


def test_cancel_pipeline_job(
    client: TestClient, pipeline_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch, wait_for, terminal
) -> None:
    monkeypatch.setenv("FAKE_PIPELINE_MODE", "hang")
    job = client.post("/api/jobs", json={"pipeline": pipeline_dict}).json()

    def first_stage_done():  # noqa: ANN202
        stages = client.get(f"/api/jobs/{job['id']}/stages").json()["stages"]
        return bool(stages) and stages[0]["state"] == "succeeded"

    wait_for(first_stage_done)
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["cancel_requested_at"]
    final = terminal(job["id"])
    assert final["state"] == "cancelled" and final["step"] == 1
    stages = client.get(f"/api/jobs/{job['id']}/stages").json()
    assert stages["state"] == "cancelled"
    assert [s["state"] for s in stages["stages"]] == ["succeeded", "pending", "pending", "pending"]


def test_manager_persists_kind(settings, pipeline_dict: dict[str, Any], wait_for) -> None:  # noqa: ANN001
    first = JobManager(settings.home)
    job = first.create({"pipeline": pipeline_dict})
    assert job["kind"] == "pipeline"
    wait_for(lambda: first.get(job["id"]) if first.get(job["id"])["state"] in TERMINAL_STATES else None)
    second = JobManager(settings.home)
    assert second.get(job["id"])["kind"] == "pipeline"
    first.shutdown()


def test_pipeline_schema_and_validate(client: TestClient, pipeline_dict: dict[str, Any]) -> None:
    schema = client.get("/api/config/pipeline_schema").json()
    assert "stages" in schema["properties"] and "base" in schema["properties"]
    ok = client.post("/api/config/validate", json={"pipeline": pipeline_dict}).json()
    assert ok["ok"] is True and ok["pipeline"]["name"] == "srv-pipeline"
    bare = client.post("/api/config/validate", json=pipeline_dict).json()
    assert bare["ok"] is True and "pipeline" in bare
    bad = client.post("/api/config/validate", json={"pipeline": {**pipeline_dict, "stages": []}}).json()
    assert bad["ok"] is False and bad["errors"]


def test_recipes_carry_kind(
    client: TestClient, recipes_dir: Path, pipeline_dict: dict[str, Any], config_dict: dict[str, Any]
) -> None:
    (recipes_dir / "loop.yaml").write_text(
        "# A pipeline recipe.\n" + yaml.safe_dump(pipeline_dict, sort_keys=False)
    )
    (recipes_dir / "run.yaml").write_text("# A run recipe.\n" + yaml.safe_dump(config_dict, sort_keys=False))
    recipes = {r["name"]: r for r in client.get("/api/recipes").json()}
    assert recipes["loop"]["kind"] == "pipeline" and recipes["loop"]["valid"] is True
    assert recipes["run"]["kind"] == "run" and recipes["run"]["valid"] is True
    assert [s["name"] for s in recipes["loop"]["config"]["stages"]] == ["sft", "grpo", "rs", "eval"]


def test_bundled_pipeline_recipes_validate() -> None:
    from monorl.cli.recipes import recipes_dir

    recipes = {r["name"]: r for r in load_recipes(recipes_dir())}
    for name in ("r1_sft_grpo_loop_cpu", "r1_sft_grpo_loop_8gpu_vllm"):
        assert recipes[name]["kind"] == "pipeline"
        assert recipes[name]["valid"] is True, recipes[name]["errors"]
