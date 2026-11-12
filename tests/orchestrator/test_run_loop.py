from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import fakes
import pytest
import torch

from monorl.core.registry import registry
from monorl.core.reporting import iter_metrics, iter_trajectories, list_checkpoints, read_status
from monorl.core.types import RunResult
from monorl.orchestrator import resolve_run_dir, run

CONTRACT_KEYS = {
    "reward/mean",
    "reward/std",
    "reward/test_exact",
    "loss/policy",
    "loss/kl",
    "loss/entropy",
    "policy/clipfrac",
    "policy/approx_kl",
    "tokens/response_mean",
    "tokens/prompt_mean",
    "time/rollout",
    "time/train",
    "time/sync",
    "lr",
}


def _rows(run_dir: Path, phase: str = "train") -> list[dict[str, Any]]:
    return [row for _, row in iter_metrics(run_dir) if row["phase"] == phase]


def _samples_by_step(run_dir: Path, steps: tuple[int, ...]) -> dict[int, list[str]]:
    rows = [row for _, row in iter_trajectories(run_dir)]
    return {s: [t["sample_id"] for t in rows if t["step"] == s] for s in steps}


def test_resolve_run_dir_uses_run_root_and_name(make_config: Any, tmp_path: Path) -> None:
    cfg = make_config()
    assert resolve_run_dir(cfg) == tmp_path / "runs" / "fake-run"
    anon = cfg.model_copy(update={"run_name": None})
    name = resolve_run_dir(anon).name
    assert name.startswith("fake-model-grpo-") and resolve_run_dir(anon).parent == tmp_path / "runs"


def test_run_writes_complete_run_dir(make_config: Any, tmp_path: Path) -> None:
    cfg = make_config()
    result = run(cfg)

    assert isinstance(result, RunResult)
    assert result.state == "succeeded" and result.final_step == 2
    run_dir = Path(result.run_dir)
    assert run_dir == (tmp_path / "runs" / "fake-run").resolve()
    assert (run_dir / "config.yaml").exists()
    assert read_status(run_dir)["state"] == "succeeded"
    assert read_status(run_dir)["step"] == 2

    rows = _rows(run_dir)
    assert [r["step"] for r in rows] == [1, 2]
    assert CONTRACT_KEYS <= set(rows[0])
    assert rows[0]["policy/staleness"] == 0.0
    assert rows[0]["tokens/prefix_mismatch"] == 0.0

    assert list_checkpoints(run_dir) == [
        str(run_dir / "checkpoints" / "step-1"),
        str(run_dir / "checkpoints" / "step-2"),
    ]
    assert result.checkpoints == list_checkpoints(run_dir)
    assert (run_dir / "final" / "config.json").exists()
    assert result.final_model_path == str(run_dir / "final")

    trajs = [row for _, row in iter_trajectories(run_dir)]
    assert {t["step"] for t in trajs} == {1, 2}
    assert len([t for t in trajs if t["step"] == 1]) == 4  # prompts_per_step * group_size
    assert {"prompt", "response", "reward", "group_id", "sample_id"} <= set(trajs[0])
    assert "run.log" in {p.name for p in run_dir.iterdir()}

    trainer = fakes.TRAINERS[-1]
    assert trainer.step == 2 and len(trainer.batches) == 2
    engine = fakes.ENGINES[-1]
    assert engine.update_count == 2  # weight sync after every step
    assert engine.weight_version == 2.0
    assert engine.shutdown_count >= 1  # batching wrapper forwards shutdown, runtime.close() repeats it
    assert fakes.BATCHING_ENGINES and fakes.BATCHING_ENGINES[0].engine is engine
    assert fakes.BATCHING_ENGINES[0].max_batch_size == cfg.sequences_per_step()


def test_rewards_and_advantages_are_broadcast_over_response_positions(make_config: Any) -> None:
    cfg = make_config(rollout={"extra": {"max_parallel_episodes": 1}}, trainer={"max_steps": 1})
    run(cfg)
    batch = fakes.TRAINERS[-1].batches[0]
    adv = fakes.ADVANTAGE_CALLS[0]
    assert adv.shape == (4,)
    assert batch.rewards.tolist() == [1.0, 0.0, 1.0, 0.0]  # alternate right/wrong, serial order
    assert batch.group_ids[0] == batch.group_ids[1] != batch.group_ids[2] == batch.group_ids[3]
    expected = adv[:, None] * batch.response_mask.float()
    assert torch.equal(batch.advantages, expected)
    assert torch.all(batch.advantages[batch.response_mask == 0] == 0)
    # old logprobs come from the engine completions
    assert batch.old_logprobs is not None
    first_span = batch.old_logprobs[0][batch.response_mask[0] == 1]
    assert first_span[0].item() == pytest.approx(-0.1) and first_span[1].item() == pytest.approx(-0.11)
    assert batch.meta["step"] == 1


def test_kl_coef_requests_reference_logprobs(make_config: Any) -> None:
    run(make_config(trainer={"max_steps": 1}))
    assert fakes.TRAINERS[-1].logprob_calls == []
    assert fakes.TRAINERS[-1].batches[0].ref_logprobs is None

    run(make_config(trainer={"max_steps": 1}, algorithm={"kl_coef": 0.1}, run_name="kl"))
    trainer = fakes.TRAINERS[-1]
    assert trainer.logprob_calls == [True]
    batch = trainer.batches[0]
    assert batch.ref_logprobs is not None
    assert torch.all(batch.ref_logprobs[batch.response_mask == 1] == -0.5)
    assert torch.all(batch.ref_logprobs[batch.response_mask == 0] == 0)


def test_missing_engine_logprobs_are_recomputed_by_trainer(make_config: Any) -> None:
    run(make_config(trainer={"max_steps": 1}, rollout={"extra": {"fake_no_logprobs": True}}))
    trainer = fakes.TRAINERS[-1]
    assert trainer.logprob_calls == [False]
    assert torch.all(trainer.batches[0].old_logprobs[trainer.batches[0].response_mask == 1] == -0.25)


def test_num_epochs_runs_multiple_train_steps_per_batch(make_config: Any) -> None:
    run(make_config(trainer={"max_steps": 2}, algorithm={"num_epochs": 3}))
    trainer = fakes.TRAINERS[-1]
    assert trainer.step == 6 and len(trainer.batches) == 6
    assert fakes.ENGINES[-1].update_count == 2


def test_checkpoint_cadence_and_final_export(make_config: Any) -> None:
    result = run(make_config(trainer={"max_steps": 5, "save_every": 2}, logging={"log_every": 2}))
    run_dir = Path(result.run_dir)
    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-2", "step-4"]
    assert (run_dir / "final" / "config.json").exists()
    assert [r["step"] for r in _rows(run_dir)] == [2, 4, 5]  # log_every=2 plus the last step
    assert read_status(run_dir)["step"] == 5


def test_resume_continues_from_latest_checkpoint(make_config: Any, tmp_path: Path) -> None:
    first = run(make_config(trainer={"max_steps": 2}))
    run_dir = Path(first.run_dir)
    fakes.reset_recorders()

    # a from-scratch 4-step run tells us which prompts steps 3 and 4 must see
    scratch = run(make_config(trainer={"max_steps": 4}, run_name="scratch"))
    scratch_samples = _samples_by_step(Path(scratch.run_dir), (3, 4))
    fakes.reset_recorders()

    resumed = run(make_config(trainer={"max_steps": 4}), resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4
    trainer = fakes.TRAINERS[-1]
    assert trainer.loaded_from == run_dir / "checkpoints" / "step-2"
    assert len(trainer.batches) == 2 and trainer.step == 4
    assert [r["step"] for r in _rows(run_dir)] == [1, 2, 3, 4]
    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-1", "step-2", "step-3", "step-4"]
    assert fakes.ENGINES[-1].update_count == 3  # restore sync + 2 training syncs
    assert _samples_by_step(run_dir, (3, 4)) == scratch_samples


def test_resume_without_checkpoint_starts_fresh(make_config: Any) -> None:
    result = run(make_config(trainer={"max_steps": 1}), resume=True)
    assert result.state == "succeeded" and result.final_step == 1
    assert fakes.TRAINERS[-1].loaded_from is None


def test_async_producer_respects_staleness_bound(make_config: Any) -> None:
    cfg = make_config(
        trainer={"max_steps": 8, "save_every": 100, "extra": {"fake_sleep": 0.03}},
        rollout={"async_steps_ahead": 2, "extra": {"fake_latency": 0.001}},
    )
    result = run(cfg)
    assert result.state == "succeeded" and result.final_step == 8
    rows = _rows(Path(result.run_dir))
    staleness = [r["policy/staleness"] for r in rows]
    assert len(staleness) == 8
    assert staleness[0] == 0.0
    assert max(staleness) <= 2.0
    assert max(staleness) >= 1.0, staleness  # the producer actually ran ahead of training
    assert fakes.ENGINES[-1].update_count == 8
    assert fakes.TRAINERS[-1].step == 8


def test_dynamic_sampling_drops_uniform_groups_and_skips_update(make_config: Any) -> None:
    cfg = make_config(
        algorithm={"dynamic_sampling": True},
        rollout={"extra": {"fake_mode": "always_right", "dynamic_sampling_max_rounds": 2}},
        trainer={"max_steps": 1},
    )
    result = run(cfg)
    assert result.state == "succeeded"
    row = _rows(Path(result.run_dir))[0]
    assert row["rollout/skipped"] == 1.0
    assert row["rollout/resample_rounds"] == 2.0
    assert row["rollout/groups_dropped"] == 2 + 2 * 2  # initial groups + two resample rounds
    assert fakes.TRAINERS[-1].batches == []


def test_dynamic_sampling_keeps_mixed_groups(make_config: Any) -> None:
    cfg = make_config(
        algorithm={"dynamic_sampling": True},
        rollout={"extra": {"max_parallel_episodes": 1}},
        trainer={"max_steps": 1},
    )
    result = run(cfg)
    row = _rows(Path(result.run_dir))[0]
    assert row["rollout/groups_dropped"] == 0.0 and row["rollout/resample_rounds"] == 0.0
    assert len(fakes.TRAINERS[-1].batches[0]) == 4


def test_overlong_penalty_applies_to_length_truncated_completions(make_config: Any) -> None:
    cfg = make_config(
        algorithm={"overlong_penalty": 0.5},
        rollout={"extra": {"fake_finish": "length", "fake_mode": "always_right"}},
        trainer={"max_steps": 1},
    )
    result = run(cfg)
    batch = fakes.TRAINERS[-1].batches[0]
    assert batch.rewards.tolist() == [0.5, 0.5, 0.5, 0.5]
    rows = [row for _, row in iter_trajectories(Path(result.run_dir))]
    assert rows[0]["reward_breakdown"] == {"test_exact": 1.0, "overlong_penalty": -0.5}
    assert _rows(Path(result.run_dir))[0]["reward/overlong_penalty"] == -0.5


def test_multi_turn_env_rewards_are_discounted_and_masked(make_config: Any) -> None:
    cfg = make_config(
        env={"type": "fake_multi", "max_turns": 3, "params": {"turn_reward": 1.0}},
        algorithm={"gamma": 0.5},
        rollout={"extra": {"fake_mode": "always_right"}},
        trainer={"max_steps": 1},
    )
    result = run(cfg)
    batch = fakes.TRAINERS[-1].batches[0]
    # reward-fn score (last answer "AGAIN2" != reference) 0 + env return 1 + 0.5 + 0.25
    assert batch.rewards.tolist() == pytest.approx([1.75] * 4)
    rows = [row for _, row in iter_trajectories(Path(result.run_dir))]
    assert rows[0]["reward_breakdown"]["env"] == pytest.approx(1.75)
    assert rows[0]["num_turns"] == 3
    metrics = _rows(Path(result.run_dir))[0]
    assert metrics["tokens/prefix_mismatch"] == 0.0
    assert metrics["reward/env"] == pytest.approx(1.75)
    # all three completion spans are loss positions: 3 × (len(answer) + EOS)
    first = batch.response_mask[0].sum().item()
    assert first == (len("ABC") + 1) + (len("AGAIN1") + 1) + (len("AGAIN2") + 1)


def test_sampled_tokens_stay_in_context_so_an_episode_is_one_sequence(make_config: Any) -> None:
    """The fake engine samples a token its text does not re-tokenise to. Token rendering keeps it
    in the next prompt, so both turns of every episode are trained as one sequence."""
    cfg = make_config(
        env={"type": "fake_multi", "max_turns": 2},
        rollout={"extra": {"fake_mode": "always_right", "fake_mismatch": True}},
        trainer={"max_steps": 1},
    )
    result = run(cfg)
    metrics = _rows(Path(result.run_dir))[0]
    assert metrics["tokens/prefix_mismatch"] == 0.0 and metrics["tokens/sequences"] == 4.0
    batch = fakes.TRAINERS[-1].batches[0]
    assert len(batch) == 4
    # both completions (+ mismatch token + EOS each) are loss positions of the one sequence
    assert batch.response_mask[0].sum().item() == (len("ABC") + 2) + (len("AGAIN1") + 2)
    assert (batch.input_ids[0] == 9999).sum().item() == 2


def test_template_rendering_trains_mismatching_turns_as_separate_sequences(make_config: Any) -> None:
    """With ``turn_rendering: template`` the re-rendered chat loses the odd token, so the second
    prompt does not extend the first turn: each turn becomes its own sequence, none is dropped."""
    cfg = make_config(
        env={"type": "fake_multi", "max_turns": 2},
        rollout={
            "turn_rendering": "template",
            "extra": {"fake_mode": "always_right", "fake_mismatch": True},
        },
        trainer={"max_steps": 1},
    )
    result = run(cfg)
    metrics = _rows(Path(result.run_dir))[0]
    assert metrics["tokens/prefix_mismatch"] == 4.0 and metrics["tokens/sequences"] == 8.0
    assert metrics["rollout/num_sequences"] == 4.0
    batch = fakes.TRAINERS[-1].batches[0]
    assert len(batch) == 8
    assert batch.response_mask[:2].sum(dim=1).tolist() == [len("ABC") + 2, len("AGAIN1") + 2]
    assert batch.group_ids[0] == batch.group_ids[1]
    assert batch.rewards[0].item() == batch.rewards[1].item()


def test_eval_every_logs_greedy_eval_metrics(make_config: Any, tmp_path: Path) -> None:
    eval_file = fakes.write_jsonl(
        tmp_path / "eval.jsonl", [{"prompt": "xyz", "answer": "XYZ"}, {"prompt": "uvw", "answer": "UVW"}]
    )
    cfg = make_config(
        data={"eval_uri": str(eval_file)},
        trainer={"max_steps": 2, "eval_every": 2},
        rollout={"extra": {"fake_mode": "always_right"}},
    )
    result = run(cfg)
    eval_rows = _rows(Path(result.run_dir), phase="eval")
    assert [r["step"] for r in eval_rows] == [2]
    assert eval_rows[0]["eval/reward_mean"] == 1.0 and eval_rows[0]["eval/num_samples"] == 2
    greedy_calls = [c for c in fakes.ENGINES[-1].calls if c["temperature"] == 0.0]
    assert len(greedy_calls) == 2


def test_failure_is_recorded_in_status_and_log(make_config: Any) -> None:
    result = run(make_config(trainer={"max_steps": 3, "extra": {"fake_fail_at": 2}}))
    assert result.state == "failed"
    assert result.final_step == 1
    assert "fake trainer failure at step 2" in (result.error or "")
    run_dir = Path(result.run_dir)
    status = read_status(run_dir)
    assert status["state"] == "failed" and "RuntimeError" in status["message"]
    log = (run_dir / "run.log").read_text()
    assert "Traceback" in log and "fake trainer failure" in log
    assert fakes.ENGINES[-1].shutdown_count >= 1


def test_sigint_mid_run_finishes_step_and_checkpoints(make_config: Any) -> None:
    import os
    import signal

    class SignalTrainer(fakes.FakeTrainer):
        def train_step(self, batch: Any) -> dict[str, float]:
            if self._step == 1:  # inside step 2: signal arrives mid-step
                os.kill(os.getpid(), signal.SIGINT)
            return super().train_step(batch)

    registry.register("trainer", "fake", SignalTrainer, overwrite=True)
    try:
        result = run(make_config(trainer={"max_steps": 10, "save_every": 5}))
    finally:
        registry.register("trainer", "fake", fakes.FakeTrainer, overwrite=True)
    assert result.state == "cancelled" and result.final_step == 2
    run_dir = Path(result.run_dir)
    assert read_status(run_dir)["state"] == "cancelled"
    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-2"]
    assert not (run_dir / "final").exists()
    assert [r["step"] for r in _rows(run_dir)] == [1, 2]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_pipeline_backend_dispatch(make_config: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    class FakePipeline:
        name = "trl"

        def __init__(self, config: Any) -> None:
            self.config = config

        def run(self, config: Any, run_dir: Path, *, reporter: Any) -> RunResult:
            calls.append((config, run_dir, reporter))
            reporter.log_metrics(1, {"reward/mean": 0.5})
            return RunResult(state="succeeded", final_step=1, run_dir=str(run_dir))

    previous = registry._items["pipeline_backend"].get("trl")
    registry.register("pipeline_backend", "trl", FakePipeline, overwrite=True)
    try:
        cfg = make_config(trainer={"backend": "trl"})
        assert cfg.is_pipeline_backend
        result = run(cfg)
    finally:
        if previous is None:
            registry._items["pipeline_backend"].pop("trl", None)
        else:
            registry.register("pipeline_backend", "trl", previous, overwrite=True)
    assert result.state == "succeeded" and result.final_step == 1
    config, run_dir, reporter = calls[0]
    assert config.trainer.backend == "trl" and run_dir == reporter.run_dir
    assert read_status(run_dir)["state"] == "succeeded"
    assert (run_dir / "config.yaml").exists()
    assert fakes.TRAINERS == []  # the component loop never ran


def test_config_yaml_records_resolved_run_name(make_config: Any) -> None:
    import yaml

    cfg = make_config(trainer={"max_steps": 1}).model_copy(update={"run_name": None})
    result = run(cfg)
    saved = yaml.safe_load((Path(result.run_dir) / "config.yaml").read_text())
    assert saved["run_name"] == Path(result.run_dir).name
    assert json.loads((Path(result.run_dir) / "status.json").read_text())["state"] == "succeeded"


def test_cancel_aborts_async_producer_generation_in_flight(make_config: Any) -> None:
    """Async rollouts: the signal arrives while the producer generates and nothing is training,
    so the run must stop without waiting out the (60 s) generation."""
    import os
    import signal

    registry.register("rollout_engine", "fake", fakes.SlowCancellableEngine, overwrite=True)

    def interrupt() -> None:
        deadline = time.monotonic() + 30
        while not fakes.ENGINES and time.monotonic() < deadline:
            time.sleep(0.005)
        if fakes.ENGINES and fakes.ENGINES[-1].generating.wait(30):
            os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt, daemon=True).start()
    t0 = time.monotonic()
    try:
        result = run(make_config(trainer={"max_steps": 3}, rollout={"async_steps_ahead": 1}))
    finally:
        registry.register("rollout_engine", "fake", fakes.FakeEngine, overwrite=True)
    assert time.monotonic() - t0 < 20.0
    assert result.state == "cancelled" and result.final_step == 0
    engine = fakes.ENGINES[-1]
    assert engine.calls == [] and engine.cancel_calls >= 1  # no generation ran to completion
    assert engine.shutdown_count >= 1
    assert read_status(Path(result.run_dir))["state"] == "cancelled"
    leftovers = ("monorl-episode", "monorl-rollout-producer")
    assert not [t.name for t in threading.enumerate() if t.name.startswith(leftovers)]
