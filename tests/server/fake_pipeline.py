"""Stand-in for ``python -m monorl.cli pipeline`` used by the server tests.

Invoked as ``fake_pipeline.py --config <pipeline.yaml> --run-dir <dir>``. Writes the pipeline
run-dir layout the real runner produces: top-level ``status.json`` (``step`` = stages finished),
``pipeline.json`` manifest and one stage dir per configured stage with metrics / trajectories /
checkpoints / run.log. ``FAKE_PIPELINE_MODE=hang`` stops after the first stage and waits for
SIGTERM (then writes ``cancelled``).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import yaml

from monorl.core.reporting import RunReporter
from monorl.pipeline.stages import DATASET_FILE, Manifest, StageRecord, stage_dir_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    mode = os.environ.get("FAKE_PIPELINE_MODE", "success")
    config = yaml.safe_load(open(args.config))
    assert "stages" in config and "base" in config, config

    reporter = RunReporter(args.run_dir, echo=False)
    stages_root = reporter.run_dir / "stages"
    records = [
        StageRecord(index=i, name=s["name"], type=s["type"], run_dir=str(stages_root / stage_dir_name(i, s["name"])))
        for i, s in enumerate(config["stages"])
    ]
    manifest = Manifest(name=config.get("name") or "fake", run_dir=str(reporter.run_dir), stages=records, state="running")
    manifest.write()
    reporter.set_status("running", "fake pipeline started", step=0)
    print("fake pipeline stdout line", flush=True)

    stop = {"flag": False}

    def on_term(signum, frame):  # noqa: ANN001
        stop["flag"] = True

    signal.signal(signal.SIGTERM, on_term)

    for record in records:
        record.state, record.started_at = "running", time.time()
        manifest.write()
        reporter.set_status("running", f"stage {record.index + 1}/{len(records)} {record.name} ({record.type})", step=record.index)
        stage = RunReporter(record.run_dir, echo=False)
        stage.set_status("running", "fake stage", step=0)
        for step in range(1, 3):
            if record.type == "rejection_sample":
                stage.log_metrics(step, {"rs/sampled": 4.0, "rs/accepted": float(step), "rs/accept_rate": step / 4, "reward/mean": 0.5}, phase="rejection")
            elif record.type == "eval":
                stage.log_metrics(step, {"eval/reward_mean": 0.75}, phase="eval")
            else:
                stage.log_metrics(step, {"reward/mean": step * 0.25, "loss/policy": 1.0 / step})
            stage.log(f"stage {record.name} step {step}")
            with open(stage.run_dir / "trajectories.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "group_id": f"g{step}", "prompt": "p", "response": "r", "reward": 0.5}) + "\n")
        if record.type == "train":
            stage.checkpoint_path(2).mkdir(parents=True, exist_ok=True)
            stage.final_dir.mkdir(exist_ok=True)
            record.summary = {"reward/mean": 0.5, "final_model_path": str(stage.final_dir)}
        elif record.type == "rejection_sample":
            (stage.run_dir / DATASET_FILE).write_text(json.dumps({"messages": [{"role": "user", "content": "p"}, {"role": "assistant", "content": "r"}], "reference": "r", "reward": 1.0, "source": "sampled"}) + "\n")
            record.summary = {"sampled": 8, "accepted": 3, "accept_rate": 0.375, "dataset": str(stage.run_dir / DATASET_FILE)}
        else:
            record.summary = {"eval/reward_mean": 0.75}
        stage.set_status("succeeded", "done", step=2)
        record.state, record.step, record.finished_at = "succeeded", 2, time.time()
        manifest.write()
        if mode == "hang":
            reporter.set_status("running", "hanging after first stage", step=1)
            while not stop["flag"]:
                time.sleep(0.05)
            manifest.state = "cancelled"
            manifest.write()
            reporter.set_status("cancelled", "stopped by SIGTERM", step=1)
            return 0
        time.sleep(float(os.environ.get("FAKE_PIPELINE_STAGE_SLEEP", "0.02")))
    manifest.state = "succeeded"
    manifest.final_model_path = next((r.summary.get("final_model_path") for r in reversed(records) if r.type == "train"), None)
    manifest.write()
    reporter.set_status("succeeded", f"finished {len(records)} stages", step=len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
