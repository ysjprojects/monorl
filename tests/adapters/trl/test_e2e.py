"""Real CPU runs of the ``trl`` pipeline backend on the tiny Llama (single- and multi-turn)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from transformers import AutoModelForCausalLM
from trl_support import ENV_NAME, REWARD_NAME, TwoTurnEnv

from monorl.adapters.trl.rollout import EpisodeRollout, TrainerPolicyEngine
from monorl.config.schema import RunConfig
from monorl.core.registry import registry
from monorl.core.reporting import RunReporter, iter_metrics, iter_trajectories, list_checkpoints, read_status
from monorl.patching import is_applied

pytestmark = pytest.mark.slow


def _run(config: RunConfig, run_dir: Path):
    reporter = RunReporter(run_dir, echo=False)
    backend = registry.get("pipeline_backend", "trl")(config)
    assert backend.name == "trl"
    return backend.run(config, run_dir, reporter=reporter)


def test_single_turn_grpo_end_to_end(trl_config: RunConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = _run(trl_config, run_dir)

    assert result.state == "succeeded", result.error
    assert result.final_step == 2
    assert is_applied("trl.grpo.monorl_hooks", "trl")

    rows = [row for _, row in iter_metrics(run_dir)]
    train_rows = [r for r in rows if "loss/policy" in r]
    assert len(train_rows) == 2
    for r in train_rows:
        assert {"reward/mean", "reward/std", "loss/policy", "lr", "tokens/response_mean", "loss/entropy"} <= set(r)
        assert f"reward/{REWARD_NAME}" in r and r["phase"] == "train"
    assert [r["step"] for r in train_rows] == [1, 2]

    status = read_status(run_dir)
    assert status is not None and status["state"] == "succeeded" and status["step"] == 2

    assert (run_dir / "final" / "config.json").exists()
    assert result.final_model_path == str(run_dir / "final")
    AutoModelForCausalLM.from_pretrained(run_dir / "final")

    checkpoints = list_checkpoints(run_dir)
    assert [Path(p).name for p in checkpoints] == ["step-1", "step-2"]
    assert result.checkpoints == checkpoints
    assert (Path(checkpoints[0]) / "trainer_state.json").exists()

    trajs = [t for _, t in iter_trajectories(run_dir)]
    assert len(trajs) == 8  # 4 sequences/step x 2 steps, trajectories_per_step=4
    assert {t["step"] for t in trajs} == {1, 2}
    assert all(t["num_turns"] == 1 and t["num_response_tokens"] > 0 for t in trajs)
    assert all(REWARD_NAME in t["reward_breakdown"] for t in trajs)
    assert all(t["reference"] in {"4", "8", "6", "21"} for t in trajs)


def test_lora_and_resume_from_checkpoint(trl_config: RunConfig, tmp_path: Path) -> None:
    cfg = trl_config.model_copy(deep=True)
    cfg.model.lora.enabled = True
    cfg.model.lora.r = 4
    cfg.model.lora.alpha = 8
    cfg.trainer.max_steps = 1
    run_dir = tmp_path / "run-lora"
    first = _run(cfg, run_dir)
    assert first.state == "succeeded", first.error
    final_cfg = json.loads((run_dir / "final" / "config.json").read_text())
    assert "peft_type" not in final_cfg and final_cfg["architectures"] == ["LlamaForCausalLM"]  # merged export
    assert (run_dir / "checkpoints" / "step-1" / "adapter_config.json").exists()

    cfg.trainer.max_steps = 2
    second = _run(cfg, run_dir)
    assert second.state == "succeeded", second.error
    assert second.final_step == 2
    log = (run_dir / "run.log").read_text()
    assert "resuming from" in log and "checkpoint-1" in log
    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-1", "step-2"]


class _AlwaysStopEngine(TrainerPolicyEngine):
    """The random tiny model never emits EOS within 8 tokens; mark completions finished so the
    environment's second turn is actually exercised. Tokens, logprobs and prompts stay real."""

    def generate(self, prompts, params):
        out = super().generate(prompts, params)
        for group in out:
            for c in group:
                c.finish_reason = "stop"
        return out


def test_multi_turn_env_through_patch(trl_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = trl_config.model_copy(deep=True)
    cfg.env.type = ENV_NAME
    cfg.env.max_turns = 2
    cfg.rewards = []  # the environment owns the reward
    cfg.trainer.save_every = 2
    monkeypatch.setattr(
        EpisodeRollout,
        "default_engine",
        staticmethod(
            lambda trainer: _AlwaysStopEngine(
                trainer.accelerator.unwrap_model(trainer.model),
                trainer.processing_class,
                device=trainer.accelerator.device,
                eos_token_ids=trainer.eos_token_ids,
            )
        ),
    )
    TwoTurnEnv.resets.clear()
    run_dir = tmp_path / "run-mt"
    result = _run(cfg, run_dir)

    assert result.state == "succeeded", result.error
    assert len(TwoTurnEnv.resets) == 8  # one episode per sequence: 4 per step x 2 steps
    train_rows = [r for _, r in iter_metrics(run_dir) if "loss/policy" in r]
    assert len(train_rows) == 2
    for r in train_rows:
        assert r["reward/mean"] == pytest.approx(1.5)  # 0.5 + 1.0 env reward per episode
        assert r["reward/env"] == pytest.approx(1.5)
        # Policy tokens only: up to 2 turns x 8; the ~15-token env follow-up must be masked out of
        # the count (a prefix-mismatched episode trains its last turn only, hence the lower bound).
        assert 8.0 <= r["tokens/response_mean"] <= 16.0 and r["trl/completions/max_length"] <= 16.0
    trajs = [t for _, t in iter_trajectories(run_dir)]
    assert trajs and all(t["num_turns"] == 2 for t in trajs)
    assert all(t["reward_breakdown"] == {"env": 1.5} for t in trajs)
    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-2"]
    assert (run_dir / "final" / "config.json").exists()
