"""Unit tests for the data, reward, rollout and patch bridges (fakes, no training)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from transformers import AutoTokenizer
from trl_support import REWARD_NAME, VowelReward

from monorl.adapters.trl import PATCHES
from monorl.adapters.trl.callbacks import map_metrics
from monorl.adapters.trl.data import build_dataset, sample_from_row, sample_to_row
from monorl.adapters.trl.patches import HOOKS_ATTR, build_hooks_patch, verify_hooks_target
from monorl.adapters.trl.rewards import TRAJECTORY_FIELD, MonorlRewardFunc
from monorl.adapters.trl.rollout import EpisodeRollout, GRPOHooks, flatten_for_trl, sampling_params
from monorl.config.schema import AlgorithmConfig, RunConfig
from monorl.core.errors import ConfigError
from monorl.core.types import Completion, Sample, Trajectory, Turn
from monorl.patching import status


# -- data ---------------------------------------------------------------------------------------
def test_sample_row_roundtrip() -> None:
    s = Sample(id="x1", messages=[{"role": "user", "content": "hi"}], reference={"answer": 4}, metadata={"k": "v"})
    row = sample_to_row(s)
    assert row["prompt"] == [{"role": "user", "content": "hi"}]
    assert row["reference"] == '{"answer": 4}'
    back = sample_from_row(row)
    assert back == s


def test_build_dataset_from_monorl_source(trl_config: RunConfig, tiny_model_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    ds = build_dataset(trl_config.data, tokenizer=tok, seed=0)
    assert len(ds) == 4
    assert set(ds.column_names) >= {"prompt", "sample_id", "reference", "sample_json"}
    row = ds[0]
    assert row["prompt"][-1]["role"] == "user"
    assert sample_from_row(row).reference == row["reference"]


# -- rewards --------------------------------------------------------------------------------------
def test_reward_wrapper_single_turn(tiny_model_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    fn = MonorlRewardFunc(VowelReward(), tokenizer=tok, eos_token_ids=[tok.eos_token_id])
    assert fn.__name__ == "monorl"
    samples = [Sample(id=f"s{i}", messages=[{"role": "user", "content": f"q{i}"}], reference=str(i)) for i in range(2)]
    rows = [sample_to_row(s) for s in samples]
    logged: dict[str, float] = {}
    scores = fn(
        prompts=[r["prompt"] for r in rows],
        completions=[[{"role": "assistant", "content": "aeiou"}], [{"role": "assistant", "content": "xyz"}]],
        completion_ids=[[1, 2, tok.eos_token_id], [3, 4]],
        sample_json=[r["sample_json"] for r in rows],
        sample_id=[r["sample_id"] for r in rows],
        reference=[r["reference"] for r in rows],
        trainer_state=SimpleNamespace(global_step=3),
        log_metric=lambda k, v: logged.__setitem__(k, v),
    )
    assert scores == [5.0, 0.0]
    assert logged == {f"reward/{REWARD_NAME}": 2.5}
    t0, t1 = fn.last_batch
    assert t0.sample.reference == "0" and t0.turns[0].completion.finish_reason == "stop"
    assert t1.turns[0].completion.finish_reason == "length"
    assert t0.turns[0].prompt_token_ids and t0.group_id == "3-s0"
    assert t0.reward_breakdown == {REWARD_NAME: 5.0} and t0.info["reward_info"] == {"chars": 5}


def test_reward_wrapper_uses_prebuilt_trajectories_and_env_return(tiny_model_dir: Path) -> None:
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    algo = AlgorithmConfig(gamma=0.5, overlong_penalty=0.25)
    fn = MonorlRewardFunc(None, tokenizer=tok, eos_token_ids=[tok.eos_token_id], algorithm=algo)
    traj = Trajectory(
        sample=Sample(id="a", messages=[]),
        group_id="g",
        messages=[],
        turns=[
            Turn(prompt_token_ids=[1], completion=Completion(text="x", token_ids=[2]), reward=1.0),
            Turn(prompt_token_ids=[1, 2, 3], completion=Completion(text="y", token_ids=[4], finish_reason="length"), reward=2.0),
        ],
    )
    out = fn(prompts=[[{"role": "user", "content": "?"}]], completions=["xy"], completion_ids=[[2, 3, 4]], **{TRAJECTORY_FIELD: [traj]})
    assert out == [pytest.approx(1.0 + 0.5 * 2.0 - 0.25)]
    assert traj.reward_breakdown == {"env": 2.0, "overlong_penalty": -0.25}
    assert fn.last_batch[0] is traj


# -- rollout flattening ---------------------------------------------------------------------------
def _two_turn_trajectory(chain_ok: bool) -> Trajectory:
    t0 = Turn(prompt_token_ids=[10, 11], completion=Completion(text="a", token_ids=[20, 21], logprobs=[-0.1, -0.2]))
    prefix = [10, 11, 20, 21] if chain_ok else [10, 11, 99, 21]
    t1 = Turn(prompt_token_ids=prefix + [30, 31], completion=Completion(text="b", token_ids=[40], logprobs=[-0.3]))
    return Trajectory(sample=Sample(id="s", messages=[]), group_id="g", messages=[], turns=[t0, t1])


def test_flatten_marks_env_tokens_with_zero_mask() -> None:
    flat = flatten_for_trl(_two_turn_trajectory(chain_ok=True))
    assert flat["prompt_ids"] == [10, 11]
    assert flat["completion_ids"] == [20, 21, 30, 31, 40]
    assert flat["env_mask"] == [1, 1, 0, 0, 1]
    assert flat["logprobs"] == pytest.approx([-0.1, -0.2, 0.0, 0.0, -0.3])


def test_flatten_prefix_mismatch_trains_last_completion_only() -> None:
    flat = flatten_for_trl(_two_turn_trajectory(chain_ok=False))
    assert flat["completion_ids"] == [99, 21, 30, 31, 40]
    assert flat["env_mask"] == [0, 0, 0, 0, 1]


def test_sampling_params_from_grpo_args() -> None:
    args = SimpleNamespace(max_completion_length=16, temperature=0.7, top_p=0.9, top_k=0, min_p=None, repetition_penalty=1.1)
    p = sampling_params(args, stop=["</s>"])
    assert (p.n, p.max_new_tokens, p.temperature, p.top_p, p.top_k, p.min_p, p.stop) == (1, 16, 0.7, 0.9, -1, 0.0, ["</s>"])
    assert p.logprobs is True


def test_episode_rollout_requires_bound_inputs(trl_config: RunConfig) -> None:
    rollout = EpisodeRollout(trl_config, lambda: None)  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="monorl_hooks"):
        rollout([[{"role": "user", "content": "x"}]], trainer=None)


# -- patch ----------------------------------------------------------------------------------------
class _FakeTrainer:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls.append(inputs)
        return {"n": len(inputs)}


def test_hooks_patch_builder_passes_through_and_fires_hooks() -> None:
    assert verify_hooks_target(_FakeTrainer._generate_and_score_completions)
    patched = build_hooks_patch(_FakeTrainer._generate_and_score_completions)
    plain = _FakeTrainer()
    assert patched(plain, [{"prompt": "p"}]) == {"n": 1}

    events: list[tuple[str, Any]] = []
    rollout = EpisodeRollout(RunConfig.model_construct(), lambda: None)  # type: ignore[arg-type]
    hooked = _FakeTrainer()
    setattr(
        hooked,
        HOOKS_ATTR,
        GRPOHooks(rollout=rollout, on_scored=lambda tr, out: events.append(("scored", out))),
    )
    inputs = [{"prompt": "a"}, {"prompt": "b"}]
    assert patched(hooked, inputs) == {"n": 2}
    assert rollout._inputs == inputs
    assert events == [("scored", {"n": 2})]


def test_patchset_registered_and_applicable_to_installed_trl() -> None:
    assert PATCHES.framework == "trl"
    ids = {p.id for p in PATCHES.patches}
    assert ids == {"trl.grpo.monorl_hooks"}
    rows = {s.id: s for s in status("trl")}
    assert rows["trl.grpo.monorl_hooks"].applicable, rows["trl.grpo.monorl_hooks"].reason


# -- metrics --------------------------------------------------------------------------------------
def test_metric_mapping() -> None:
    logs = {
        "reward": 1.0,
        "reward_std": 0.5,
        "rewards/monorl/mean": 1.0,
        "rewards/other_fn/mean": 0.2,
        "reward/exact_match": 0.7,
        "kl": 0.01,
        "loss": 0.3,
        "completions/mean_length": 12,
        "clip_ratio/region_mean": 0.1,
        "learning_rate": 1e-6,
        "entropy": 2.0,
        "step_time": 0.4,
        "eval_reward": 0.9,
        "foo": None,
        "step": 3,
    }
    out = map_metrics(logs)
    assert out["reward/mean"] == 1.0 and out["reward/std"] == 0.5
    assert "reward/monorl" not in out and out["reward/other_fn"] == 0.2 and out["reward/exact_match"] == 0.7
    assert out["loss/kl"] == 0.01 and out["loss/policy"] == 0.3 and out["loss/entropy"] == 2.0
    assert out["tokens/response_mean"] == 12 and out["policy/clipfrac"] == 0.1 and out["lr"] == 1e-6
    assert out["trl/step_time"] == 0.4 and out["eval/reward/mean"] == 0.9
    assert "foo" not in out and "step" not in out
