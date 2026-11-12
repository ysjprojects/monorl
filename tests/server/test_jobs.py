"""Job lifecycle through the API and the JobManager, driven by ``fake_trainer.py``."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from monorl.server.app import ServerSettings, create_app
from monorl.server.jobs import TERMINAL_STATES, JobManager, JobNotFound, train_command


@pytest.fixture
def terminal(client: TestClient, wait_for):  # noqa: ANN001, ANN201
    def _terminal(job_id: str, timeout: float = 15.0) -> dict[str, Any]:
        def poll() -> dict[str, Any] | None:
            job = client.get(f"/api/jobs/{job_id}").json()
            return job if job["state"] in TERMINAL_STATES else None

        return wait_for(poll, timeout=timeout)

    return _terminal


def test_train_command_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONORL_TRAIN_CMD", raising=False)
    assert train_command() == [sys.executable, "-m", "monorl.cli", "train"]
    monkeypatch.setenv("MONORL_TRAIN_CMD", "/usr/bin/env 'my trainer' --flag")
    assert train_command() == ["/usr/bin/env", "my trainer", "--flag"]


def test_create_invalid_config_is_422(client: TestClient, config_dict: dict[str, Any]) -> None:
    bad = dict(config_dict)
    bad["model"] = {"name_or_path": "x", "dtype": "int4"}
    r = client.post("/api/jobs", json={"config": bad})
    assert r.status_code == 422
    body = r.json()
    assert body["ok"] is False
    assert [e["loc"] for e in body["errors"]] == [["model", "dtype"]]
    assert client.get("/api/jobs").json() == []
    assert client.post("/api/jobs", json={"config": "nope"}).status_code == 422


def test_full_lifecycle(
    client: TestClient,
    config_dict: dict[str, Any],
    tmp_path: Path,
    home: Path,
    fake_trainer_path: Path,
    terminal,
) -> None:
    r = client.post("/api/jobs", json={"config": config_dict})
    assert r.status_code == 201, r.text
    job = r.json()
    job_id = job["id"]
    run_dir = Path(job["run_dir"])
    assert run_dir == tmp_path / "runs" / "srv-test"
    assert job["name"] == "srv-test"
    assert job["state"] == "running" and isinstance(job["pid"], int)
    assert job["created_at"] <= job["started_at"]
    assert job["finished_at"] is None and job["error"] is None
    assert job["config"]["run_name"] == "srv-test"
    assert job["command"][:2] == [sys.executable, str(fake_trainer_path)]
    assert job["command"][2:] == ["--config", str(run_dir / "config.yaml"), "--run-dir", str(run_dir)]
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["model"]["name_or_path"] == "org/tiny-model"

    # persisted immediately
    persisted = json.loads((home / "jobs.json").read_text())["jobs"]
    assert [j["id"] for j in persisted] == [job_id]

    final = terminal(job_id)
    assert final["state"] == "succeeded", final
    assert final["step"] == 3
    assert final["message"] == "done"
    assert final["finished_at"] is not None
    assert final["status"]["state"] == "succeeded"
    assert final["status"]["pid"] == job["pid"]

    listed = client.get("/api/jobs").json()
    assert [j["id"] for j in listed] == [job_id]
    assert listed[0]["state"] == "succeeded"
    assert "status" not in listed[0]

    # metrics with pagination
    m = client.get(f"/api/jobs/{job_id}/metrics").json()
    assert [row["step"] for row in m["rows"]] == [1, 2, 3]
    assert m["rows"][0]["reward/mean"] == 0.25 and m["rows"][2]["loss/policy"] == pytest.approx(1 / 3)
    assert m["next"] == 3
    m2 = client.get(f"/api/jobs/{job_id}/metrics", params={"since": 2}).json()
    assert [row["step"] for row in m2["rows"]] == [3] and m2["next"] == 3
    assert client.get(f"/api/jobs/{job_id}/metrics", params={"since": 3}).json() == {"rows": [], "next": 3}
    page = client.get(f"/api/jobs/{job_id}/metrics", params={"limit": 2}).json()
    assert [row["step"] for row in page["rows"]] == [1, 2] and page["next"] == 2
    page2 = client.get(f"/api/jobs/{job_id}/metrics", params={"since": page["next"], "limit": 2}).json()
    assert [row["step"] for row in page2["rows"]] == [3]

    # trajectories
    t = client.get(f"/api/jobs/{job_id}/trajectories").json()
    assert [row["group_id"] for row in t["rows"]] == ["g1", "g2", "g3"] and t["next"] == 3
    assert client.get(f"/api/jobs/{job_id}/trajectories", params={"since": 1}).json()["rows"][0]["step"] == 2

    # checkpoints
    c = client.get(f"/api/jobs/{job_id}/checkpoints").json()
    assert c["checkpoints"] == [
        {"step": 1, "path": str(run_dir / "checkpoints" / "step-1")},
        {"step": 3, "path": str(run_dir / "checkpoints" / "step-3")},
    ]
    assert c["final"] == str(run_dir / "final")

    # logs: subprocess stdout and reporter lines share run.log
    logs = client.get(f"/api/jobs/{job_id}/logs").json()
    assert any("fake trainer stdout line" in line for line in logs["lines"])
    assert any(line.endswith("step 3 done") for line in logs["lines"])
    assert logs["next"] == len(logs["lines"])
    tail = client.get(f"/api/jobs/{job_id}/logs", params={"tail": 1}).json()
    assert len(tail["lines"]) == 1 and tail["lines"][0].endswith("[succeeded] done")
    assert client.get(f"/api/jobs/{job_id}/logs", params={"since": logs["next"]}).json()["lines"] == []

    # delete
    assert client.delete(f"/api/jobs/{job_id}").json() == {"ok": True, "id": job_id}
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert json.loads((home / "jobs.json").read_text())["jobs"] == []
    assert run_dir.exists()


def test_run_dir_collision_gets_suffix(
    client: TestClient, config_dict: dict[str, Any], tmp_path: Path, terminal
) -> None:
    a = client.post("/api/jobs", json={"config": config_dict}).json()
    b = client.post("/api/jobs", json={"config": config_dict}).json()
    assert a["run_dir"] == str(tmp_path / "runs" / "srv-test")
    assert b["run_dir"] == str(tmp_path / "runs" / "srv-test-2")
    assert b["name"] == "srv-test-2" and b["config"]["run_name"] == "srv-test-2"
    for j in (a, b):
        assert terminal(j["id"])["state"] == "succeeded"
    assert [j["id"] for j in client.get("/api/jobs").json()] == [b["id"], a["id"]]  # newest first


def test_generated_run_name(
    client: TestClient, config_dict: dict[str, Any], tmp_path: Path, terminal
) -> None:
    cfg = dict(config_dict)
    cfg.pop("run_name")
    cfg["model"] = {"name_or_path": "HuggingFaceTB/SmolLM2-135M-Instruct"}
    job = client.post("/api/jobs", json={"config": cfg}).json()
    assert job["name"].startswith("SmolLM2-135M-Instruct-grpo-")
    assert Path(job["run_dir"]).parent == tmp_path / "runs"
    assert terminal(job["id"])["state"] == "succeeded"


def test_crash_without_terminal_status_is_failed(
    client: TestClient, config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch, terminal
) -> None:
    monkeypatch.setenv("FAKE_TRAINER_MODE", "crash")
    job = client.post("/api/jobs", json={"config": config_dict}).json()
    final = terminal(job["id"])
    assert final["state"] == "failed"
    assert final["error"] == "trainer exited with code 3 without reporting a terminal status"
    assert final["status"]["state"] == "running"  # the trainer never updated it


def test_spawn_failure(
    client: TestClient, config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MONORL_TRAIN_CMD", "/definitely/not/a/binary")
    r = client.post("/api/jobs", json={"config": config_dict})
    assert r.status_code == 201
    job = r.json()
    assert job["state"] == "failed" and job["pid"] is None
    assert job["error"].startswith("failed to spawn trainer")


def test_cancel_graceful(
    client: TestClient, config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch, wait_for, terminal
) -> None:
    monkeypatch.setenv("FAKE_TRAINER_MODE", "hang")
    job = client.post("/api/jobs", json={"config": config_dict}).json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}/metrics").json()["rows"])
    r = client.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 200
    assert r.json()["message"] == "cancel requested"
    final = terminal(job["id"])
    assert final["state"] == "cancelled"
    assert final["status"]["state"] == "cancelled"  # the trainer handled SIGTERM
    assert final["status"]["message"] == "stopped by SIGTERM"
    # cancelling a finished job is a no-op
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["state"] == "cancelled"


def test_cancel_escalates_to_sigkill(
    settings: ServerSettings, config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch, wait_for
) -> None:
    monkeypatch.setenv("FAKE_TRAINER_MODE", "stubborn")
    jobs = JobManager(settings.home, grace_seconds=0.3)
    with TestClient(create_app(settings, jobs=jobs)) as client:
        job = client.post("/api/jobs", json={"config": config_dict}).json()
        wait_for(lambda: client.get(f"/api/jobs/{job['id']}/metrics").json()["rows"])
        t0 = time.time()
        client.post(f"/api/jobs/{job['id']}/cancel")
        final = wait_for(lambda: _if_terminal(client.get(f"/api/jobs/{job['id']}").json()))
        assert final["state"] == "cancelled"
        assert final["status"]["state"] == "running"  # never got to write a terminal status
        assert time.time() - t0 < 10
        with pytest.raises(ProcessLookupError):
            os.kill(job["pid"], 0)
        assert client.delete(f"/api/jobs/{job['id']}", params={"remove_files": "true"}).status_code == 200
        assert not Path(job["run_dir"]).exists()


def _if_terminal(job: dict[str, Any]) -> dict[str, Any] | None:
    return job if job["state"] in TERMINAL_STATES else None


def test_delete_running_job_is_409(
    client: TestClient, config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch, terminal
) -> None:
    monkeypatch.setenv("FAKE_TRAINER_MODE", "hang")
    job = client.post("/api/jobs", json={"config": config_dict}).json()
    r = client.delete(f"/api/jobs/{job['id']}")
    assert r.status_code == 409 and "cancel it" in r.json()["detail"]
    client.post(f"/api/jobs/{job['id']}/cancel")
    assert terminal(job["id"])["state"] == "cancelled"


def test_manager_reload_reconciles_without_process_handle(
    settings: ServerSettings, config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch, wait_for
) -> None:
    """A second JobManager (server restart) sees the same jobs and reconciles via pid liveness."""
    monkeypatch.setenv("FAKE_TRAINER_MODE", "hang")
    first = JobManager(settings.home)
    job = first.create(config_dict)
    assert job["state"] == "running"

    second = JobManager(settings.home)
    assert [j["id"] for j in second.list()] == [job["id"]]
    assert second.is_alive(job["id"])
    second.cancel(job["id"])
    final = wait_for(lambda: _if_terminal(second.get(job["id"])))
    assert final["state"] == "cancelled"
    with pytest.raises(JobNotFound):
        second.get("missing")
    first.shutdown()


def test_manager_crash_detection_after_reload(
    settings: ServerSettings, config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch, wait_for
) -> None:
    monkeypatch.setenv("FAKE_TRAINER_MODE", "crash")
    first = JobManager(settings.home)
    job = first.create(config_dict)
    wait_for(lambda: not first.is_alive(job["id"]))
    second = JobManager(settings.home)  # no Popen handle: liveness via psutil, no exit code
    final = second.get(job["id"])
    assert final["state"] == "failed"
    assert final["error"] == "trainer process is gone without reporting a terminal status"
    first.shutdown()


def test_sse_metrics_stream(
    client: TestClient, config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_TRAINER_STEP_SLEEP", "0.3")
    job = client.post("/api/jobs", json={"config": config_dict}).json()
    events = _collect_sse(client, f"/api/jobs/{job['id']}/metrics/stream")
    names = [e for e, _ in events]
    assert names == ["metrics", "metrics", "metrics", "end"], events
    assert [json.loads(d)["step"] for e, d in events if e == "metrics"] == [1, 2, 3]
    assert json.loads(events[-1][1]) == {"state": "succeeded", "next": 3, "error": None}

    # resume from an offset on a finished job: remaining rows then end
    events = _collect_sse(client, f"/api/jobs/{job['id']}/metrics/stream", params={"since": 2})
    assert [e for e, _ in events] == ["metrics", "end"]
    assert json.loads(events[0][1])["step"] == 3


def test_sse_logs_stream(client: TestClient, config_dict: dict[str, Any]) -> None:
    job = client.post("/api/jobs", json={"config": config_dict}).json()
    events = _collect_sse(client, f"/api/jobs/{job['id']}/logs/stream")
    assert events[-1][0] == "end"
    lines = [json.loads(d) for e, d in events if e == "log"]
    assert all(isinstance(line, str) for line in lines)
    assert any(line.endswith("[succeeded] done") for line in lines)
    assert any("fake trainer stdout line" in line for line in lines)


def _collect_sse(client: TestClient, url: str, params: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    event, data = "message", []
    with client.stream("GET", url, params=params) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
            elif line == "":
                if data:
                    events.append((event, "\n".join(data)))
                event, data = "message", []
            if events and events[-1][0] == "end":
                break
    return events
