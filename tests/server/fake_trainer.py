"""Stand-in for ``python -m monorl.cli train`` used by the server tests.

Invoked as ``fake_trainer.py --config <path> --run-dir <dir>``; behaviour is selected with the
``FAKE_TRAINER_MODE`` environment variable:

* ``success`` (default): status running -> 3 metric rows + trajectories + 2 checkpoints -> succeeded
* ``hang``: status running, then wait; SIGTERM -> status cancelled and exit 0
* ``stubborn``: like ``hang`` but ignores SIGTERM (must be SIGKILLed)
* ``crash``: status running, then exit 3 without a terminal status
* ``spawn_fail``: never reached; tests point MONORL_TRAIN_CMD at a missing binary instead
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

from monorl.core.reporting import RunReporter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    mode = os.environ.get("FAKE_TRAINER_MODE", "success")
    assert os.path.isfile(args.config), args.config

    reporter = RunReporter(args.run_dir, echo=False)
    reporter.set_status("running", "fake trainer started", step=0)
    print("fake trainer stdout line", flush=True)

    if mode == "crash":
        reporter.log("about to crash")
        return 3

    if mode in ("hang", "stubborn"):
        stop = {"flag": False}

        def on_term(signum, frame):  # noqa: ANN001
            if mode == "stubborn":
                return
            stop["flag"] = True

        signal.signal(signal.SIGTERM, on_term)
        reporter.log_metrics(1, {"reward/mean": 0.1})
        while not stop["flag"]:
            time.sleep(0.05)
        reporter.set_status("cancelled", "stopped by SIGTERM")
        return 0

    for step in range(1, 4):
        reporter.log_metrics(step, {"reward/mean": step * 0.25, "loss/policy": 1.0 / step})
        reporter.log(f"step {step} done")
        row = {"step": step, "group_id": f"g{step}", "prompt": "p", "response": "r", "reward": step * 0.25}
        with open(reporter.run_dir / "trajectories.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        if step in (1, 3):
            reporter.checkpoint_path(step).mkdir(parents=True, exist_ok=True)
        time.sleep(float(os.environ.get("FAKE_TRAINER_STEP_SLEEP", "0.05")))
    reporter.final_dir.mkdir(exist_ok=True)
    reporter.set_status("succeeded", "done", step=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
