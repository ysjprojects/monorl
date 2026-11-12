from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from monorl.adapters.nemo_rl import register
from monorl.adapters.nemo_rl.backend import (
    ENTRY_MODULE,
    NemoRLBackend,
    build_command,
    build_env,
    final_model_path,
    latest_step,
    list_checkpoints,
    monorl_src_root,
    nemo_rl_home,
)
from monorl.adapters.nemo_rl.data import MODULE_NAME
from monorl.adapters.nemo_rl.entry import HOME_ENV, parse_args, resolve_home
from monorl.adapters.nemo_rl.env_bridge import ACTOR_FQN
from monorl.config.schema import RunConfig
from monorl.core.errors import BackendUnavailable, ConfigError
from monorl.core.registry import Registry
from monorl.core.reporting import RunReporter, iter_metrics, read_status


def _with_extra(config: RunConfig, **extra) -> RunConfig:
    data = config.to_dict()
    data["trainer"]["extra"] = {**data["trainer"]["extra"], **extra}
    return RunConfig.from_dict(data)


def test_home_resolution(nemo_config: RunConfig, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HOME_ENV, raising=False)
    with pytest.raises(BackendUnavailable, match="NEMO_RL_HOME"):
        nemo_rl_home(nemo_config)
    with pytest.raises(BackendUnavailable, match="not a NeMo-RL checkout"):
        resolve_home(tmp_path)
    monkeypatch.setenv(HOME_ENV, str(fake_home))
    assert nemo_rl_home(nemo_config) == fake_home.resolve()
    other = tmp_path / "other"
    (other / "examples").mkdir(parents=True)
    (other / "examples" / "run_grpo.py").write_text("")
    assert nemo_rl_home(_with_extra(nemo_config, nemo_rl_home=str(other))) == other.resolve()


def test_run_raises_backend_unavailable_without_home(nemo_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HOME_ENV, raising=False)
    reporter = RunReporter(tmp_path / "run", echo=False)
    with pytest.raises(BackendUnavailable):
        NemoRLBackend().run(nemo_config, tmp_path / "run", reporter=reporter)


def test_build_command_and_env(nemo_config: RunConfig, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "run"
    yaml_path = run_dir / "nemo_rl" / "nemo_rl.yaml"
    tail = [
        "-m", ENTRY_MODULE, "--config", str(yaml_path), "--run-dir", str(run_dir), "--nemo-rl-home", str(fake_home),
        "--script", "examples/run_grpo.py",
    ]

    monkeypatch.setattr("monorl.adapters.nemo_rl.backend.shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    assert build_command(nemo_config, fake_home, yaml_path, run_dir) == ["uv", "run", "python", *tail]
    monkeypatch.setattr("monorl.adapters.nemo_rl.backend.shutil.which", lambda name: None)
    assert build_command(nemo_config, fake_home, yaml_path, run_dir) == [sys.executable, *tail]
    assert build_command(_with_extra(nemo_config, nemo_rl_launcher="uv"), fake_home, yaml_path, run_dir)[:3] == ["uv", "run", "python"]
    custom = _with_extra(nemo_config, nemo_rl_launcher="python", nemo_rl_python="/opt/py", nemo_rl_overrides=["grpo.max_num_steps=3"])
    assert build_command(custom, fake_home, yaml_path, run_dir) == ["/opt/py", *tail, "grpo.max_num_steps=3"]
    with pytest.raises(ConfigError, match="nemo_rl_launcher"):
        build_command(_with_extra(nemo_config, nemo_rl_launcher="conda"), fake_home, yaml_path, run_dir)
    with pytest.raises(ConfigError, match="nemo_rl_overrides"):
        build_command(_with_extra(nemo_config, nemo_rl_overrides="x=1"), fake_home, yaml_path, run_dir)

    env = build_env(nemo_config, fake_home, run_dir, base={"PATH": "/bin", "PYTHONPATH": "/extra"})
    assert env["MONORL_RUN_DIR"] == str(run_dir) and env["NEMO_RL_HOME"] == str(fake_home)
    assert env["MONORL_ALGORITHM"] == "grpo"
    assert env["PYTHONPATH"].split(os.pathsep) == [str(run_dir), str(monorl_src_root()), "/extra"]
    assert env["PYTHONUNBUFFERED"] == "1" and "RAY_ADDRESS" not in env and env["PATH"] == "/bin"
    ray_cfg = RunConfig.from_dict({**nemo_config.to_dict(), "cluster": {**nemo_config.cluster.model_dump(mode="json"), "ray_address": "auto"}})
    assert build_env(ray_cfg, fake_home, run_dir, base={})["RAY_ADDRESS"] == "auto"


def test_checkpoint_listing(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    for name in ("step_10", "step_2", "step-3", "tmp_step_4", "not-a-step"):
        (root / name).mkdir(parents=True)
    (root / "step_10" / "policy" / "weights").mkdir(parents=True)
    listed = list_checkpoints(tmp_path)
    assert [Path(p).name for p in listed] == ["step_2", "step-3", "step_10"]
    assert final_model_path(listed) == str(root / "step_10" / "policy" / "weights")
    assert final_model_path([str(root / "step_2")]) == str(root / "step_2")
    assert final_model_path([]) is None
    assert list_checkpoints(tmp_path / "missing") == []


def test_latest_step_reads_metrics(tmp_path: Path) -> None:
    assert latest_step(tmp_path) == 0
    (tmp_path / "metrics.jsonl").write_text('{"step": 3, "x": 1}\n{"step": 7}\n{"step": "bad"}\n')
    assert latest_step(tmp_path) == 7


def test_entry_parse_args() -> None:
    args, overrides = parse_args(["--config", "c.yaml", "--run-dir", "r", "grpo.max_num_steps=3", "++policy.x=1"])
    assert (args.config, args.run_dir, args.nemo_rl_home) == ("c.yaml", "r", None)
    assert overrides == ["grpo.max_num_steps=3", "++policy.x=1"]


def test_register_exposes_pipeline_backend(nemo_config: RunConfig) -> None:
    reg = Registry()
    register(reg)
    backend = reg.get("pipeline_backend", "nemo_rl")(nemo_config)
    assert isinstance(backend, NemoRLBackend) and backend.name == "nemo_rl"


@pytest.fixture
def python_launch(nemo_config: RunConfig, fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    monkeypatch.setenv(HOME_ENV, str(fake_home))
    monkeypatch.delenv("FAKE_NEMO_RL_SLEEP", raising=False)
    monkeypatch.delenv("FAKE_NEMO_RL_FAIL", raising=False)
    return _with_extra(nemo_config, nemo_rl_launcher="python", nemo_rl_overrides=["grpo.max_num_steps=2"])


def test_run_end_to_end_with_fake_checkout(python_launch: RunConfig, fake_home: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    reporter = RunReporter(run_dir, echo=False)
    result = NemoRLBackend(python_launch).run(python_launch, run_dir, reporter=reporter)

    assert result.state == "succeeded" and result.error is None, (run_dir / "run.log").read_text()
    assert result.final_step == 2 and result.run_dir == str(run_dir.resolve())
    assert [Path(p).name for p in result.checkpoints] == ["step_2"]
    assert result.final_model_path == str(run_dir.resolve() / "checkpoints" / "step_2" / "policy" / "weights")

    status = read_status(run_dir)
    assert status["state"] == "succeeded" and status["step"] == 2 and status["returncode"] == 0
    rows = [r for _, r in iter_metrics(run_dir)]
    assert [(r["step"], r["phase"]) for r in rows] == [(1, "train"), (1, "train"), (2, "train"), (2, "train")]
    assert rows[0]["loss/policy"] == 0.5 and rows[2]["reward/mean"] == 0.5 and rows[3]["time/step"] == 3.0

    child = json.loads((run_dir / "child_env.json").read_text())
    assert Path(child["cwd"]).resolve() == fake_home.resolve()
    assert child["env_registry"]["monorl"] == {"actor_class_fqn": ACTOR_FQN}
    assert child["actor_registry"][ACTOR_FQN] == "python"
    assert child["patched"] == "nemo_rl.logger_metrics"
    assert child["ray_address"] is None
    assert child["dataset_name"] == f"{MODULE_NAME}.MonorlDataset"
    assert child["argv"][0].endswith("examples/run_grpo.py") and child["argv"][-1] == "grpo.max_num_steps=2"

    log = (run_dir / "run.log").read_text()
    assert "training done" in log and "LOG train step=1" in log and "launching (cwd=" in log
    assert (run_dir / "nemo_rl" / "train.jsonl").exists() and (run_dir / f"{MODULE_NAME}.py").exists()
    assert (run_dir / "nemo_rl" / "nemo_rl.yaml").exists()


def test_run_reports_failure(python_launch: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_NEMO_RL_FAIL", "1")
    run_dir = tmp_path / "run"
    result = NemoRLBackend().run(python_launch, run_dir, reporter=RunReporter(run_dir, echo=False))
    assert result.state == "failed" and "code 3" in result.error
    assert result.final_step == 2 and read_status(run_dir)["state"] == "failed"
    assert "boom" in (run_dir / "run.log").read_text()


def test_cancel_propagates_to_child(python_launch: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_NEMO_RL_SLEEP", "1")
    run_dir = tmp_path / "run"
    reporter = RunReporter(run_dir, echo=False)
    backend = NemoRLBackend()
    results: list = []
    thread = threading.Thread(target=lambda: results.append(backend.run(python_launch, run_dir, reporter=reporter)))
    thread.start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if "sleeping" in ((run_dir / "run.log").read_text() if (run_dir / "run.log").exists() else ""):
            break
        time.sleep(0.05)
    else:
        pytest.fail("child never started")
    backend.cancel(grace=5)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert results[0].state == "cancelled" and backend.cancelled
    assert read_status(run_dir)["state"] == "cancelled"
    assert read_status(run_dir)["returncode"] != 0
