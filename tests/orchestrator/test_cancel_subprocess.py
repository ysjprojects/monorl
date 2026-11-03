"""SIGTERM delivered to a real subprocess running ``run()``: the current step finishes, a
checkpoint is written and ``status.json`` ends in ``cancelled``."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import fakes

from monorl.core.reporting import iter_metrics, list_checkpoints, read_status

TESTS_DIR = Path(__file__).parent
SRC_DIR = TESTS_DIR.parent.parent / "src"

SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import fakes
fakes.install_fakes()
from monorl.config.schema import RunConfig
from monorl.orchestrator import run
config = RunConfig.from_dict(json.loads(open(sys.argv[2]).read()))
result = run(config)
print("RESULT " + json.dumps({"state": result.state, "final_step": result.final_step, "checkpoints": result.checkpoints}))
"""


def _wait_for(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_sigterm_cancels_run_after_finishing_current_step(tmp_path: Path) -> None:
    cfg = fakes.fake_config_dict(
        tmp_path,
        run_name="cancel-me",
        trainer={"max_steps": 100_000, "save_every": 1_000, "extra": {"fake_sleep": 0.05}},
    )
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    run_dir = tmp_path / "runs" / "cancel-me"

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC_DIR), os.environ.get("PYTHONPATH")]))}
    proc = subprocess.Popen(
        [sys.executable, "-c", SCRIPT, str(TESTS_DIR), str(cfg_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        _wait_for(
            lambda: proc.poll() is not None
            or ((s := read_status(run_dir)) is not None and s["state"] == "running" and s["step"] >= 2),
            timeout=60,
            what="the run to reach step 2",
        )
        assert proc.poll() is None, proc.stderr.read()
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()

    assert proc.returncode == 0, stderr
    result_line = next(line for line in stdout.splitlines() if line.startswith("RESULT "))
    result = json.loads(result_line.removeprefix("RESULT "))
    assert result["state"] == "cancelled"
    final_step = result["final_step"]
    assert final_step >= 2

    status = read_status(run_dir)
    assert status is not None and status["state"] == "cancelled"
    assert status["step"] == final_step
    assert "cancelled after step" in status["message"]
    # the interrupted step was finished (metrics for it exist) and a checkpoint was written
    steps = [row["step"] for _, row in iter_metrics(run_dir)]
    assert steps == list(range(1, final_step + 1))
    assert [Path(p).name for p in list_checkpoints(run_dir)] == [f"step-{final_step}"]
    assert result["checkpoints"] == list_checkpoints(run_dir)
    assert not (run_dir / "final").exists()
    assert "received SIGTERM" in (run_dir / "run.log").read_text()
