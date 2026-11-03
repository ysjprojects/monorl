"""VerlBackend: registry wiring, command line, env, run/cancel/fail against a stand-in entry module."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

from monorl.adapters.verl import backend as backend_module
from monorl.adapters.verl.backend import (
    VerlBackend,
    checkpoint_dirs,
    final_model_path,
    latest_step,
    link_checkpoints,
)
from monorl.core.errors import BackendUnavailable
from monorl.core.registry import registry
from monorl.core.reporting import RunReporter, list_checkpoints, read_status

_FAKE_ENTRY = '''
import json, os, signal, sys, time
from pathlib import Path

run_dir = Path(os.environ["MONORL_RUN_DIR"])
mode = os.environ.get("FAKE_ENTRY_MODE", "ok")
print("fake verl entry argv:", sys.argv[1:], flush=True)
print("RAY_ADDRESS=", os.environ.get("RAY_ADDRESS"), flush=True)
if mode == "fail":
    print("boom", file=sys.stderr, flush=True)
    sys.exit(3)
with open(run_dir / "metrics.jsonl", "a") as f:
    for step in (1, 2):
        f.write(json.dumps({"step": step, "time": 0.0, "phase": "train", "reward/mean": 0.25 * step, "actor/lr": 1e-6}) + "\\n")
        f.flush()
        if mode == "hang":
            time.sleep(0.2)
ckpt = run_dir / "checkpoints" / "global_step_2" / "actor" / "huggingface"
ckpt.mkdir(parents=True)
(run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").write_text("2")
if mode == "hang":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    time.sleep(60)
print("done", flush=True)
'''


@pytest.fixture
def fake_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    mod_dir = tmp_path / "fake_entry_pkg"
    mod_dir.mkdir()
    (mod_dir / "fake_verl_entry.py").write_text(_FAKE_ENTRY)
    monkeypatch.setattr(backend_module, "ENTRY_MODULE", "fake_verl_entry")
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", str(mod_dir) + (os.pathsep + existing if existing else ""))
    return mod_dir


def _backend(cfg) -> VerlBackend:
    backend = VerlBackend(cfg, python=sys.executable)
    backend.poll_seconds = 0.05
    backend.grace_seconds = 5.0
    return backend


def test_registry_and_protocol(make_config):
    factory = registry.get("pipeline_backend", "verl")
    backend = factory(make_config())
    assert isinstance(backend, VerlBackend)
    assert backend.name == "verl"
    assert backend.python == sys.executable


def test_command_and_env(make_config, tmp_path: Path):
    cfg = make_config(cluster={"ray_address": "auto"}, trainer={"backend": "verl", "extra": {"verl_python": "/opt/verl/bin/python"}})
    backend = VerlBackend(cfg)
    cmd = backend.build_command(["a=1", "b.c=x"])
    assert cmd == ["/opt/verl/bin/python", "-m", "monorl.adapters.verl.entry", "a=1", "b.c=x"]
    env = backend.build_env(cfg, tmp_path)
    assert env["RAY_ADDRESS"] == "auto"
    assert env["MONORL_RUN_DIR"] == str(tmp_path.resolve())
    assert env["PYTHONUNBUFFERED"] == "1"
    assert "RAY_ADDRESS" not in VerlBackend(make_config()).build_env(make_config(), tmp_path)


def test_prepare_stages_run_dir(make_config, tmp_path: Path):
    cfg = make_config(env={"type": "tool_calling", "params": {"tools": ["calculator"]}})
    run_dir = tmp_path / "run"
    overrides = VerlBackend(cfg).prepare(cfg, run_dir)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "data" / "train.parquet").exists()
    assert (run_dir / "monorl_reward.py").exists()
    assert (run_dir / "verl_tools.yaml").exists()
    assert (run_dir / "verl_config.yaml").exists()
    assert (run_dir / "verl_overrides.txt").read_text().splitlines() == overrides
    assert "actor_rollout_ref.rollout.multi_turn.enable=true" in overrides


def test_run_requires_verl_unless_custom_python(make_config, tmp_path: Path):
    if backend_module.is_available():
        pytest.skip("verl importable")
    with pytest.raises(BackendUnavailable):
        VerlBackend(make_config()).run(make_config(), tmp_path / "run", reporter=RunReporter(tmp_path / "run", echo=False))


def test_run_succeeds_and_collects_checkpoints(make_config, tmp_path: Path, fake_entry: Path):
    cfg = make_config()
    run_dir = tmp_path / "run"
    reporter = RunReporter(run_dir, echo=False)
    result = _backend(cfg).run(cfg, run_dir, reporter=reporter)
    assert result.state == "succeeded", result.error
    assert result.final_step == 2
    assert result.checkpoints == [str(run_dir / "checkpoints" / "global_step_2")]
    assert result.final_model_path == str(run_dir / "checkpoints" / "global_step_2" / "actor" / "huggingface")
    assert result.metrics["reward/mean"] == 0.5
    assert read_status(run_dir)["state"] == "succeeded"
    assert read_status(run_dir)["step"] == 2
    log = (run_dir / "run.log").read_text()
    assert "fake verl entry argv:" in log and "data.train_files=" in log and "done" in log
    assert list_checkpoints(run_dir) == [str(run_dir / "checkpoints" / "step-2")]
    assert (run_dir / "checkpoints" / "step-2").resolve() == (run_dir / "checkpoints" / "global_step_2").resolve()


def test_run_failure_reports_exit_code(make_config, tmp_path: Path, fake_entry: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_ENTRY_MODE", "fail")
    cfg = make_config()
    run_dir = tmp_path / "run"
    result = _backend(cfg).run(cfg, run_dir, reporter=RunReporter(run_dir, echo=False))
    assert result.state == "failed"
    assert "exited with code 3" in result.error and "boom" in result.error
    assert result.final_model_path is None
    assert read_status(run_dir)["state"] == "failed"


def test_cancel_propagates_sigterm(make_config, tmp_path: Path, fake_entry: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_ENTRY_MODE", "hang")
    cfg = make_config()
    run_dir = tmp_path / "run"
    backend = _backend(cfg)

    def cancel_when_running():
        while True:
            status = read_status(run_dir)
            if status and status.get("step", 0) >= 2:
                backend.cancel()
                return

    thread = threading.Thread(target=cancel_when_running, daemon=True)
    thread.start()
    result = backend.run(cfg, run_dir, reporter=RunReporter(run_dir, echo=False))
    thread.join(timeout=5)
    assert result.state == "cancelled"
    assert result.error is None
    assert result.final_step == 2
    assert read_status(run_dir)["state"] == "cancelled"
    assert "cancel requested" in (run_dir / "run.log").read_text()


def test_checkpoint_helpers(tmp_path: Path):
    root = tmp_path / "checkpoints"
    for n in (10, 2):
        (root / f"global_step_{n}").mkdir(parents=True)
    (root / "junk").mkdir()
    assert [s for s, _ in checkpoint_dirs(tmp_path)] == [2, 10]
    assert latest_step(tmp_path) == 10
    assert final_model_path(tmp_path) == str(root / "global_step_10")
    assert link_checkpoints(tmp_path) == [str(root / "global_step_2"), str(root / "global_step_10")]
    assert list_checkpoints(tmp_path) == [str(root / "step-2"), str(root / "step-10")]
    (root / "latest_checkpointed_iteration.txt").write_text("7")
    assert latest_step(tmp_path) == 7
    assert json.loads(json.dumps(link_checkpoints(tmp_path))) == link_checkpoints(tmp_path)  # idempotent
