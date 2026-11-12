"""Pipeline jobs through the API: creation (``{pipeline: ...}``), ``kind``, stage routes and their
SSE streams, recipes and schema; driven by ``fake_pipeline.py`` via ``MONORL_PIPELINE_CMD``."""

from __future__ import annotations

import shlex
import sys
from collections.abc import Callable, Iterator
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


# -- SSE streams -------------------------------------------------------------------------------


@pytest.fixture
def gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., None]]:
    """Park the fake pipeline before every stage; ``gate(i, ...)`` lets stages ``i, ...`` run."""
    root = tmp_path / "gate"
    root.mkdir()
    monkeypatch.setenv("FAKE_PIPELINE_GATE", str(root))

    def release(*stages: int) -> None:
        for index in stages:
            (root / str(index)).touch()

    yield release
    (root / "all").touch()  # never leave a parked runner behind


@pytest.fixture
def parked(
    client: TestClient, pipeline_dict: dict[str, Any], gate: Callable[..., None], wait_for: Callable[..., Any]
) -> str:
    """Id of a pipeline job whose manifest is written and whose four stages are all pending."""
    job = client.post("/api/jobs", json={"pipeline": pipeline_dict}).json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}/stages").json()["stages"])
    return job["id"]


async def test_stages_stream_emits_on_stage_change(
    client: TestClient, parked: str, gate: Callable[..., None], sse: Callable[..., Any]
) -> None:
    url = f"/api/jobs/{parked}/stages"
    async with sse(f"{url}/stream") as stream:
        seen = [await stream.event()]
        assert seen[0] == ("stages", client.get(url).json())
        assert [s["state"] for s in seen[0][1]["stages"]] == ["pending"] * 4
        assert await stream.idle(0.2)  # nothing changes: nothing is sent

        gate(0)
        while seen[-1][1]["stages"][0]["state"] != "succeeded":
            seen.append(await stream.event())
        assert seen[-1][1]["stages"][1]["state"] == "pending" and seen[-1][1]["state"] == "running"

        gate(1, 2, 3)
        *more, end = await stream.until_end()
    seen += more
    assert {name for name, _ in seen} == {"stages"}
    payloads = [payload for _, payload in seen]
    assert all(a != b for a, b in zip(payloads, payloads[1:], strict=False))
    assert payloads[-1] == client.get(url).json()
    assert payloads[-1]["state"] == "succeeded" and payloads[-1]["final_model_path"]
    assert end == ("end", {"state": "succeeded", "next": None, "error": None})


def _data(events: list[tuple[str, Any]], name: str) -> list[Any]:
    return [data for event, data in events if event == name]


async def test_pending_stage_events_start_with_the_stage(
    client: TestClient, parked: str, gate: Callable[..., None], sse: Callable[..., Any]
) -> None:
    base = f"/api/jobs/{parked}/stages/1"
    async with sse(f"{base}/events") as stream:
        assert await stream.idle(0.3)  # stage 1 has no run dir yet: the stream stays open and silent
        gate(0, 1)
        *events, end = await stream.until_end()

    # stage 2 is still parked, so it was the stage's own terminal state that ended the stream
    assert client.get(f"/api/jobs/{parked}").json()["state"] == "running"
    metrics = client.get(f"{base}/metrics").json()["rows"]
    lines = client.get(f"{base}/logs").json()["lines"]
    trajectories = client.get(f"{base}/trajectories").json()["rows"]
    assert {name for name, _ in events} == {"metrics", "log", "trajectory"}
    assert _data(events, "metrics") == metrics and [row["step"] for row in metrics] == [1, 2]
    assert _data(events, "log") == lines and any(line.endswith("stage grpo step 2") for line in lines)
    assert _data(events, "trajectory") == trajectories and [row["group_id"] for row in trajectories] == [
        "g1",
        "g2",
    ]
    cursors = {"metrics": 2, "logs": len(lines), "trajectories": 2}
    assert end == ("end", {"state": "succeeded", "error": None, "next": cursors})

    # every kind resumes from its own cursor
    async with sse(f"{base}/events", metrics=1, logs=len(lines) - 1, trajectories=2) as stream:
        assert await stream.until_end() == [("metrics", metrics[1]), ("log", lines[-1]), end]
    async with sse(f"{base}/events", metrics=2, logs=len(lines)) as stream:
        assert await stream.until_end() == [
            ("trajectory", trajectories[0]),
            ("trajectory", trajectories[1]),
            end,
        ]


async def test_stage_events_end_when_the_job_does(
    client: TestClient,
    parked: str,
    gate: Callable[..., None],
    sse: Callable[..., Any],
    wait_for: Callable[..., Any],
) -> None:
    gate(0)
    wait_for(lambda: client.get(f"/api/jobs/{parked}/stages/0").json()["state"] == "succeeded")
    async with sse(f"/api/jobs/{parked}/stages/2/events") as stream:
        assert await stream.idle(0.2)
        client.post(f"/api/jobs/{parked}/cancel")
        # stage 2 never ran and stays `pending` in the manifest: the job's state closes its stream
        nothing = {"metrics": 0, "logs": 0, "trajectories": 0}
        assert await stream.until_end() == [("end", {"state": "cancelled", "error": None, "next": nothing})]
    # a stage that did finish keeps reporting its own state
    logged = client.get(f"/api/jobs/{parked}/stages/0/logs").json()["next"]
    async with sse(f"/api/jobs/{parked}/stages/0/events", metrics=1, logs=logged, trajectories=2) as stream:
        events = await stream.until_end()
    assert [name for name, _ in events] == ["metrics", "end"]
    assert events[-1][1] == {
        "state": "succeeded",
        "error": None,
        "next": {"metrics": 2, "logs": logged, "trajectories": 2},
    }


def test_stage_events_404(
    client: TestClient, pipeline_dict: dict[str, Any], config_dict: dict[str, Any], terminal
) -> None:
    job = client.post("/api/jobs", json={"pipeline": pipeline_dict}).json()
    run = client.post("/api/jobs", json={"config": config_dict}).json()
    terminal(job["id"])
    terminal(run["id"])
    r = client.get(f"/api/jobs/{job['id']}/stages/4/events")
    assert r.status_code == 404 and r.json() == {"detail": f"job {job['id']!r} has no stage 4"}
    assert client.get(f"/api/jobs/{run['id']}/stages/0/events").status_code == 404
    assert client.get("/api/jobs/nope/stages/0/events").status_code == 404
    assert client.get("/api/jobs/nope/stages/stream").status_code == 404
    assert client.get("/api/jobs/nope/events").status_code == 404


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
