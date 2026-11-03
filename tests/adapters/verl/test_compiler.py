"""Config compilation snapshots: exact verl v0.9.1 override keys/values per algorithm."""

from __future__ import annotations

from pathlib import Path

from monorl.adapters.verl import VERL_COMMIT, VERL_TAG, VERL_VERSIONS, compile_config, to_overrides
from monorl.adapters.verl.compiler import flatten, format_value


def _overrides(config, run_dir: Path) -> dict[str, str]:
    out = {}
    for item in to_overrides(compile_config(config, run_dir)):
        key, _, value = item.partition("=")
        out[key] = value
    return out


def test_targets_pinned_upstream():
    assert VERL_TAG == "v0.9.1"
    assert VERL_COMMIT == "1876b06d0a3e4e71e06230be10af14492ca8a75b"
    assert VERL_VERSIONS == ">=0.9,<0.10"


def test_grpo_snapshot(make_config, tmp_path: Path):
    run_dir = tmp_path / "run"
    ov = _overrides(make_config(), run_dir)
    expected = {
        "data.train_files": str(run_dir / "data" / "train.parquet"),
        "data.val_files": str(run_dir / "data" / "train.parquet"),
        "data.prompt_key": "prompt",
        "data.reward_fn_key": "data_source",
        "data.max_prompt_length": "512",
        "data.max_response_length": "1024",
        "data.train_batch_size": "16",
        "data.shuffle": "true",
        "data.seed": "7",
        "data.filter_overlong_prompts": "true",
        "data.truncation": "error",
        "data.return_raw_chat": "true",
        "actor_rollout_ref.hybrid_engine": "true",
        "actor_rollout_ref.model.path": "Qwen/Qwen2.5-0.5B-Instruct",
        "actor_rollout_ref.model.lora_rank": "0",
        "actor_rollout_ref.model.enable_gradient_checkpointing": "false",
        "actor_rollout_ref.actor.optim.lr": "2e-06",
        "actor_rollout_ref.actor.optim.total_training_steps": "30",
        "actor_rollout_ref.actor.ppo_mini_batch_size": "16",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "2",
        "actor_rollout_ref.actor.ppo_epochs": "1",
        "actor_rollout_ref.actor.use_kl_loss": "true",
        "actor_rollout_ref.actor.kl_loss_coef": "0.001",
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.clip_ratio": "0.2",
        "actor_rollout_ref.actor.clip_ratio_low": "0.2",
        "actor_rollout_ref.actor.clip_ratio_high": "0.2",
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.entropy_coeff": "0.0",
        "actor_rollout_ref.actor.calculate_entropy": "false",
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": "5",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": "1",
        "actor_rollout_ref.rollout.gpu_memory_utilization": "0.6",
        "actor_rollout_ref.rollout.temperature": "0.9",
        "actor_rollout_ref.rollout.top_p": "0.95",
        "actor_rollout_ref.rollout.top_k": "-1",
        "actor_rollout_ref.rollout.max_model_len": "null",
        "actor_rollout_ref.rollout.free_cache_engine": "true",
        "actor_rollout_ref.rollout.multi_turn.enable": "false",
        "actor_rollout_ref.rollout.agent.default_agent_loop": "single_turn_agent",
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": "true",
        "algorithm.use_kl_in_reward": "false",
        "algorithm.filter_groups.enable": "false",
        "reward.custom_reward_function.path": str(run_dir / "monorl_reward.py"),
        "reward.custom_reward_function.name": "compute_score",
        "trainer.nnodes": "2",
        "trainer.n_gpus_per_node": "8",
        "trainer.total_training_steps": "30",
        "trainer.save_freq": "10",
        "trainer.test_freq": "-1",
        "trainer.val_before_train": "false",
        "trainer.project_name": "monorl",
        "trainer.experiment_name": "verl-test",
        "trainer.logger": "[console]",
        "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.resume_mode": "auto",
    }
    for key, value in expected.items():
        assert ov[key] == value, key
    assert "critic.model.path" not in ov
    assert not any(k.startswith("+") for k in ov)


def test_dapo_snapshot(make_config, tmp_path: Path):
    cfg = make_config(
        algorithm={"name": "dapo", "kl_coef": 0.0, "clip_eps": 0.2, "clip_eps_high": 0.28, "dynamic_sampling": True, "overlong_penalty": 1.0},
        rollout={"backend": "vllm", "group_size": 8, "max_new_tokens": 2000},
    )
    ov = _overrides(cfg, tmp_path)
    assert ov["algorithm.adv_estimator"] == "grpo"
    assert ov["actor_rollout_ref.actor.clip_ratio_low"] == "0.2"
    assert ov["actor_rollout_ref.actor.clip_ratio_high"] == "0.28"
    assert ov["actor_rollout_ref.actor.use_kl_loss"] == "false"
    assert ov["actor_rollout_ref.actor.loss_agg_mode"] == "token-mean"
    assert ov["algorithm.filter_groups.enable"] == "true"
    assert ov["algorithm.filter_groups.metric"] == "reward"
    assert ov["algorithm.filter_groups.max_num_gen_batches"] == "10"
    assert ov["data.gen_batch_size"] == "32"
    assert ov["reward.reward_manager.name"] == "dapo"
    assert ov["+reward.reward_kwargs.overlong_buffer_cfg.enable"] == "true"
    assert ov["+reward.reward_kwargs.overlong_buffer_cfg.len"] == "400"
    assert ov["+reward.reward_kwargs.overlong_buffer_cfg.penalty_factor"] == "1.0"
    assert ov["+reward.reward_kwargs.max_resp_len"] == "2000"


def test_rloo_snapshot(make_config, tmp_path: Path):
    cfg = make_config(algorithm={"name": "rloo", "kl_coef": 0.01, "kl_estimator": "k1", "loss_agg": "sequence"})
    ov = _overrides(cfg, tmp_path)
    assert ov["algorithm.adv_estimator"] == "rloo"
    assert ov["actor_rollout_ref.actor.use_kl_loss"] == "true"
    assert ov["actor_rollout_ref.actor.kl_loss_coef"] == "0.01"
    assert ov["actor_rollout_ref.actor.kl_loss_type"] == "kl"
    assert ov["actor_rollout_ref.actor.loss_agg_mode"] == "seq-mean-token-mean"
    assert "reward.reward_manager.name" not in ov


def test_reinforce_and_ppo(make_config, tmp_path: Path):
    ov = _overrides(make_config(algorithm={"name": "reinforce", "advantage_norm": "batch"}, rollout={"group_size": 1}), tmp_path)
    assert ov["algorithm.adv_estimator"] == "reinforce_plus_plus"
    ov = _overrides(make_config(algorithm={"name": "reinforce", "advantage_norm": "group"}), tmp_path)
    assert ov["algorithm.adv_estimator"] == "reinforce_plus_plus_baseline"
    ov = _overrides(make_config(algorithm={"name": "ppo", "kl_coef": 0.05, "gamma": 0.99, "lam": 0.95}), tmp_path)
    assert ov["algorithm.adv_estimator"] == "gae"
    assert ov["algorithm.gamma"] == "0.99"
    assert ov["algorithm.lam"] == "0.95"
    assert ov["algorithm.use_kl_in_reward"] == "true"
    assert ov["algorithm.kl_ctrl.kl_coef"] == "0.05"
    assert ov["actor_rollout_ref.actor.use_kl_loss"] == "false"
    assert ov["critic.model.path"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert ov["critic.ppo_micro_batch_size_per_gpu"] == "2"


def test_tool_calling_snapshot(make_config, tmp_path: Path):
    cfg = make_config(
        env={"type": "tool_calling", "max_turns": 4, "params": {"tools": ["calculator"]}},
        rollout={"backend": "sglang", "group_size": 4, "max_new_tokens": 512},
    )
    ov = _overrides(cfg, tmp_path)
    assert ov["actor_rollout_ref.rollout.name"] == "sglang"
    assert ov["actor_rollout_ref.rollout.multi_turn.enable"] == "true"
    assert ov["actor_rollout_ref.rollout.multi_turn.tool_config_path"] == str(tmp_path / "verl_tools.yaml")
    assert ov["actor_rollout_ref.rollout.multi_turn.max_assistant_turns"] == "4"
    assert ov["actor_rollout_ref.rollout.multi_turn.max_user_turns"] == "4"
    assert ov["actor_rollout_ref.rollout.multi_turn.format"] == "hermes"
    assert ov["actor_rollout_ref.rollout.agent.default_agent_loop"] == "tool_agent"


def test_model_options(make_config, tmp_path: Path):
    cfg = make_config(
        model={
            "name_or_path": "meta-llama/Llama-3.2-1B",
            "dtype": "float16",
            "trust_remote_code": True,
            "tokenizer_name_or_path": "tok/path",
            "lora": {"enabled": True, "r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        },
        trainer={"backend": "verl", "gradient_checkpointing": True, "warmup_steps": 5, "weight_decay": 0.1},
        rollout={"backend": "hf", "group_size": 2, "tensor_parallel_size": 2, "max_model_len": 4096, "enable_sleep_mode": False},
    )
    ov = _overrides(cfg, tmp_path)
    assert ov["actor_rollout_ref.model.lora_rank"] == "8"
    assert ov["actor_rollout_ref.model.lora_alpha"] == "16"
    assert ov["actor_rollout_ref.model.target_modules"] == "[q_proj,v_proj]"
    assert ov["actor_rollout_ref.model.trust_remote_code"] == "true"
    assert ov["actor_rollout_ref.model.tokenizer_path"] == "tok/path"
    assert ov["actor_rollout_ref.model.enable_gradient_checkpointing"] == "true"
    assert ov["actor_rollout_ref.actor.optim.lr_warmup_steps"] == "5"
    assert ov["actor_rollout_ref.actor.optim.weight_decay"] == "0.1"
    assert ov["actor_rollout_ref.rollout.dtype"] == "float16"
    assert ov["actor_rollout_ref.rollout.name"] == "vllm"  # hf rollout has no verl equivalent in v0.9.1
    assert ov["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "2"
    assert ov["actor_rollout_ref.rollout.max_model_len"] == "4096"
    assert ov["actor_rollout_ref.rollout.free_cache_engine"] == "false"


def test_eval_and_logging(make_config, tmp_path: Path, data_file: Path):
    cfg = make_config(
        data={"uri": str(data_file), "prompt_field": "prompt", "answer_field": "answer", "eval_uri": str(data_file)},
        trainer={"backend": "verl", "eval_every": 5},
        logging={"wandb_project": "proj", "tensorboard": True, "save_trajectories": False},
    )
    ov = _overrides(cfg, tmp_path)
    assert ov["data.val_files"] == str(tmp_path / "data" / "val.parquet")
    assert ov["trainer.test_freq"] == "5"
    assert ov["trainer.logger"] == "[console,wandb,tensorboard]"
    assert ov["trainer.rollout_data_dir"] == "null"


def test_disaggregated_placement(make_config, tmp_path: Path):
    cfg = make_config(cluster={"num_nodes": 1, "gpus_per_node": 8, "placement": "disaggregated", "rollout_gpus": 2, "trainer_gpus": 6})
    ov = _overrides(cfg, tmp_path)
    assert ov["actor_rollout_ref.hybrid_engine"] == "false"
    assert ov["actor_rollout_ref.rollout.nnodes"] == "1"
    assert ov["actor_rollout_ref.rollout.n_gpus_per_node"] == "2"
    assert ov["trainer.n_gpus_per_node"] == "6"
    assert ov["trainer.v1.trainer_mode"] == "separate_async"


def test_extra_deep_merge_and_reserved_keys(make_config, tmp_path: Path):
    cfg = make_config(
        trainer={
            "backend": "verl",
            "extra": {
                "actor_rollout_ref": {"actor": {"use_dynamic_bsz": True, "ppo_max_token_len_per_gpu": 20000}},
                "trainer": {"logger": ["console", "mlflow"], "+custom_flag": "x"},
                "verl_python": "/opt/verl/bin/python",
            },
        }
    )
    tree = compile_config(cfg, tmp_path)
    assert tree["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is True
    assert tree["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == 16  # untouched sibling
    assert tree["trainer"]["logger"] == ["console", "mlflow"]
    assert "verl_python" not in tree["trainer"]
    ov = dict(item.partition("=")[::2] for item in to_overrides(tree))
    assert ov["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] == "20000"
    assert ov["+trainer.custom_flag"] == "x"


def test_override_rendering():
    assert format_value(None) == "null"
    assert format_value(True) == "true"
    assert format_value(1e-6) == "1e-06"
    assert format_value("all-linear") == "all-linear"
    assert format_value("/a/b.parquet") == "/a/b.parquet"
    assert format_value("hello world") == "'hello world'"
    assert format_value("it's") == "'it\\'s'"
    assert format_value("123") == "'123'"  # keep strings that look like literals quoted
    assert format_value(["a", 1]) == "[a,1]"
    assert format_value({"k": "v"}) == "{k:v}"
    assert flatten({"a": {"b": 1, "+c": {"d": 2}}, "+e": 3}) == [("a.b", 1), ("+a.c.d", 2), ("+e", 3)]
