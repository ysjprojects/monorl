from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from monorl.adapters.nemo_rl.compile import (
    CONFIG_FILENAME,
    ENV_NAME,
    colocation,
    compile_config,
    config_path,
    deep_merge,
    write_config,
)
from monorl.adapters.nemo_rl.data import MODULE_NAME, TASK_NAME
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError


def _with(config: RunConfig, **sections) -> RunConfig:
    data = config.to_dict()
    for key, update in sections.items():
        data[key] = {**data[key], **update}
    return RunConfig.from_dict(data)


def test_grpo_snapshot(nemo_config: RunConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cfg = compile_config(nemo_config, run_dir)
    tr, ro = nemo_config.trainer, nemo_config.rollout

    assert set(cfg) == {
        "grpo",
        "loss_fn",
        "checkpointing",
        "policy",
        "data",
        "env",
        "logger",
        "cluster",
        "data_plane",
        "on_policy_distillation",
    }
    policy = cfg["policy"]
    assert policy["model_name"] == nemo_config.model.name_or_path
    assert policy["tokenizer"] == {"name": nemo_config.model.name_or_path, "chat_template_kwargs": None}
    assert policy["train_global_batch_size"] == tr.prompts_per_step * ro.group_size == 4
    assert policy["train_micro_batch_size"] == tr.micro_batch_size
    assert policy["max_total_sequence_length"] == tr.max_seq_len == 256
    assert policy["precision"] == "bfloat16"
    assert policy["dtensor_cfg"]["enabled"] is True and policy["dtensor_cfg"]["_v2"] is True
    assert policy["dtensor_cfg"]["lora_cfg"] == {"enabled": False}
    assert policy["megatron_cfg"] == {"enabled": False}
    assert policy["optimizer"] == {
        "name": "torch.optim.AdamW",
        "kwargs": {"lr": tr.learning_rate, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8},
    }
    assert policy["scheduler"] == {
        "name": "torch.optim.lr_scheduler.ConstantLR",
        "kwargs": {"factor": 1.0, "total_iters": 2},
    }
    assert policy["max_grad_norm"] == tr.max_grad_norm
    gen = policy["generation"]
    assert gen["backend"] == "vllm"
    assert (gen["max_new_tokens"], gen["temperature"], gen["top_p"], gen["top_k"]) == (8, 1.0, 1.0, None)
    assert gen["stop_strings"] is None
    assert gen["colocated"] == {"enabled": True, "resources": {"gpus_per_node": None, "num_nodes": None}}
    assert gen["vllm_cfg"]["tensor_parallel_size"] == 1
    assert gen["vllm_cfg"]["gpu_memory_utilization"] == 0.6
    assert gen["vllm_cfg"]["max_model_len"] == 256
    assert gen["vllm_cfg"]["precision"] == "bfloat16"
    assert gen["vllm_cfg"]["async_engine"] is False

    grpo = cfg["grpo"]
    assert grpo["num_prompts_per_step"] == 2 and grpo["num_generations_per_prompt"] == 2
    assert grpo["max_num_steps"] == 2 and grpo["max_rollout_turns"] == 1
    assert grpo["normalize_rewards"] is True and grpo["use_leave_one_out_baseline"] is False
    assert grpo["val_period"] == 0 and grpo["val_at_end"] is False
    assert grpo["use_dynamic_sampling"] is False
    assert grpo["reward_shaping"]["enabled"] is False
    assert grpo["async_grpo"]["enabled"] is False
    assert grpo["seed"] == nemo_config.seed

    loss = cfg["loss_fn"]
    assert loss["reference_policy_kl_penalty"] == 0.0
    assert loss["reference_policy_kl_type"] == "k3"
    assert (loss["ratio_clip_min"], loss["ratio_clip_max"]) == (0.2, 0.2)
    assert loss["token_level_loss"] is True
    assert loss["disable_ppo_ratio"] is False
    assert loss["use_on_policy_kl_approximation"] is False
    assert loss["use_importance_sampling_correction"] is False

    data = cfg["data"]
    assert data["train"]["dataset_name"] == f"{MODULE_NAME}.MonorlDataset"
    assert data["train"]["data_path"] == str(run_dir.resolve() / "nemo_rl" / "train.jsonl")
    assert data["validation"] is None
    assert data["default"] == {
        "prompt_file": None,
        "system_prompt_file": None,
        "processor": TASK_NAME,
        "env_name": ENV_NAME,
    }
    assert data["max_input_seq_length"] == 128

    env = cfg["env"][ENV_NAME]
    assert env["run_config"] == nemo_config.to_dict()
    assert env["rewards"] == [{"type": "exact_match", "weight": 1.0, "params": {}}]
    assert env["run_dir"] == str(run_dir.resolve())

    ck = cfg["checkpointing"]
    assert ck["enabled"] is True
    assert ck["checkpoint_dir"] == str(run_dir.resolve() / "checkpoints")
    assert ck["save_period"] == tr.save_every == 1
    assert ck["is_peft"] is False

    lg = cfg["logger"]
    assert lg["log_dir"] == str(run_dir.resolve() / "nemo_rl" / "logs")
    assert lg["wandb_enabled"] is False and lg["tensorboard_enabled"] is False
    assert lg["mlflow_enabled"] is False and lg["swanlab_enabled"] is False and lg["monitor_gpus"] is False
    assert lg["monorl"] == {
        "enabled": True,
        "metrics_file": str(run_dir.resolve() / "metrics.jsonl"),
        "run_dir": str(run_dir.resolve()),
    }

    assert cfg["cluster"]["gpus_per_node"] == 8 and cfg["cluster"]["num_nodes"] == 1
    assert cfg["data_plane"] is None and cfg["on_policy_distillation"] is None
    yaml.safe_dump(cfg)  # yaml-serialisable (no python objects)


def test_rloo_dapo_reinforce_mapping(nemo_config: RunConfig, tmp_path: Path) -> None:
    rloo = compile_config(
        _with(nemo_config, algorithm={"name": "rloo", "advantage_std_norm": False}), tmp_path
    )
    assert rloo["grpo"]["use_leave_one_out_baseline"] is True
    assert rloo["grpo"]["normalize_rewards"] is False
    assert rloo["grpo"]["adv_estimator"]["use_leave_one_out_baseline"] is True

    dapo_cfg = _with(
        nemo_config,
        algorithm={
            "name": "dapo",
            "clip_eps": 0.2,
            "clip_eps_high": 0.28,
            "dynamic_sampling": True,
            "overlong_penalty": 0.5,
            "kl_coef": 0.04,
            "kl_estimator": "k1",
            "loss_agg": "sequence",
        },
    )
    dapo = compile_config(dapo_cfg, tmp_path)
    assert (dapo["loss_fn"]["ratio_clip_min"], dapo["loss_fn"]["ratio_clip_max"]) == (0.2, 0.28)
    assert dapo["loss_fn"]["reference_policy_kl_penalty"] == 0.04
    assert dapo["loss_fn"]["reference_policy_kl_type"] == "k1"
    assert dapo["loss_fn"]["token_level_loss"] is False
    assert dapo["grpo"]["use_dynamic_sampling"] is True
    assert dapo["grpo"]["reward_shaping"]["enabled"] is True
    assert dapo["grpo"]["reward_shaping"]["overlong_buffer_penalty"] == 0.5

    reinforce = compile_config(
        _with(nemo_config, algorithm={"name": "reinforce", "advantage_norm": "none"}), tmp_path
    )
    assert reinforce["loss_fn"]["disable_ppo_ratio"] is True
    assert reinforce["grpo"]["normalize_rewards"] is False
    assert reinforce["grpo"]["adv_estimator"]["minus_baseline"] is False


def test_unsupported_settings_raise(nemo_config: RunConfig, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="ppo"):
        compile_config(_with(nemo_config, algorithm={"name": "ppo"}), tmp_path)
    with pytest.raises(ConfigError, match="num_epochs"):
        compile_config(_with(nemo_config, algorithm={"num_epochs": 2}), tmp_path)
    with pytest.raises(ConfigError, match="entropy_coef"):
        compile_config(_with(nemo_config, algorithm={"entropy_coef": 0.01}), tmp_path)
    with pytest.raises(ConfigError, match="rollout.backend"):
        compile_config(_with(nemo_config, rollout={"backend": "sglang"}), tmp_path)
    with pytest.raises(ConfigError, match="nemo_rl"):
        compile_config(_with(nemo_config, trainer={"extra": {"nemo_rl": ["not", "a", "dict"]}}), tmp_path)
    with pytest.raises(ConfigError, match="harness"):
        harness = {"type": "harness", "params": {"command": ["agent"]}}
        compile_config(_with(nemo_config, env=harness), tmp_path)


def test_colocated_vs_disaggregated_vllm_cfg(nemo_config: RunConfig, tmp_path: Path) -> None:
    colocated = compile_config(_with(nemo_config, cluster={"placement": "colocated"}), tmp_path)
    assert colocated["policy"]["generation"]["colocated"]["enabled"] is True

    single = _with(
        nemo_config,
        cluster={"placement": "disaggregated", "rollout_gpus": 2, "trainer_gpus": 6},
        rollout={"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9, "max_model_len": 4096},
    )
    cfg = compile_config(single, tmp_path)
    gen = cfg["policy"]["generation"]
    assert gen["colocated"] == {"enabled": False, "resources": {"gpus_per_node": 2, "num_nodes": None}}
    assert gen["vllm_cfg"]["tensor_parallel_size"] == 2
    assert gen["vllm_cfg"]["gpu_memory_utilization"] == 0.9
    assert gen["vllm_cfg"]["max_model_len"] == 4096

    multi = _with(
        nemo_config,
        cluster={
            "placement": "disaggregated",
            "num_nodes": 4,
            "gpus_per_node": 8,
            "rollout_gpus": 16,
            "trainer_gpus": 16,
        },
    )
    assert colocation(multi) == {"enabled": False, "resources": {"gpus_per_node": 8, "num_nodes": 2}}
    assert compile_config(multi, tmp_path)["cluster"] == {
        "gpus_per_node": 8,
        "num_nodes": 4,
        "master_port_range_low": 25000,
        "master_port_range_high": 28000,
        "segment_size": None,
    }


def test_async_rollouts_enable_async_grpo(nemo_config: RunConfig, tmp_path: Path) -> None:
    cfg = compile_config(_with(nemo_config, rollout={"async_steps_ahead": 2}), tmp_path)
    assert cfg["grpo"]["async_grpo"] == {
        "enabled": True,
        "max_trajectory_age_steps": 2,
        "in_flight_weight_updates": False,
        "recompute_kv_cache_after_weight_updates": False,
    }
    assert cfg["loss_fn"]["use_importance_sampling_correction"] is True
    assert cfg["policy"]["generation"]["vllm_cfg"]["async_engine"] is True


def test_scheduler_warmup_lora_eval_and_logging(nemo_config: RunConfig, tmp_path: Path) -> None:
    cfg = compile_config(
        _with(
            nemo_config,
            trainer={
                "scheduler": "cosine",
                "warmup_steps": 5,
                "max_steps": 50,
                "eval_every": 10,
                "optimizer": "sgd",
                "weight_decay": 0.1,
            },
            model={"lora": {"enabled": True, "r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}},
            data={"eval_uri": str(Path(nemo_config.data.uri)), "eval_split": "test"},
            logging={"wandb_project": "proj", "tensorboard": True},
            rollout={"stop": ["</answer>"], "top_k": 50},
        ),
        tmp_path,
    )
    sched = cfg["policy"]["scheduler"]
    assert sched[0]["name"].endswith("LinearLR") and sched[0]["kwargs"]["total_iters"] == 5
    assert sched[1] == {
        "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "kwargs": {"T_max": 45, "eta_min": 0.0},
    }
    assert sched[2] == {"milestones": [5]}
    assert cfg["policy"]["optimizer"]["name"] == "torch.optim.SGD"
    assert cfg["policy"]["optimizer"]["kwargs"]["weight_decay"] == 0.1
    lora = cfg["policy"]["dtensor_cfg"]["lora_cfg"]
    assert lora["enabled"] is True and lora["dim"] == 8 and lora["alpha"] == 16
    assert lora["target_modules"] == ["q_proj", "v_proj"] and lora["match_all_linear"] is False
    assert cfg["checkpointing"]["is_peft"] is True
    assert cfg["grpo"]["val_period"] == 10 and cfg["grpo"]["val_at_end"] is True
    assert cfg["data"]["validation"]["data_path"].endswith("nemo_rl/eval.jsonl")
    assert cfg["logger"]["wandb_enabled"] is True and cfg["logger"]["wandb"]["project"] == "proj"
    assert cfg["logger"]["tensorboard_enabled"] is True
    assert cfg["policy"]["generation"]["stop_strings"] == ["</answer>"]
    assert cfg["policy"]["generation"]["top_k"] == 50


def test_megatron_from_trainer_extra(nemo_config: RunConfig, tmp_path: Path) -> None:
    cfg = compile_config(
        _with(
            nemo_config,
            trainer={
                "extra": {"megatron": {"enabled": True, "tensor_model_parallel_size": 2}},
                "learning_rate": 2e-6,
            },
        ),
        tmp_path,
    )
    mcfg = cfg["policy"]["megatron_cfg"]
    assert mcfg["enabled"] is True and mcfg["tensor_model_parallel_size"] == 2
    assert mcfg["optimizer"]["lr"] == 2e-6 and mcfg["optimizer"]["optimizer"] == "adam"
    assert mcfg["pipeline_dtype"] == "bfloat16" and mcfg["optimizer"]["bf16"] is True
    assert mcfg["scheduler"]["lr_decay_style"] == "constant"
    assert cfg["policy"]["dtensor_cfg"] == {"enabled": False}
    assert cfg["policy"]["optimizer"] is None and cfg["policy"]["scheduler"] is None

    dt = compile_config(
        _with(
            nemo_config,
            trainer={"extra": {"dtensor": {"tensor_parallel_size": 4, "sequence_parallel": True}}},
        ),
        tmp_path,
    )
    assert dt["policy"]["dtensor_cfg"]["tensor_parallel_size"] == 4
    assert dt["policy"]["dtensor_cfg"]["sequence_parallel"] is True


def test_extra_nemo_rl_deep_merges_last(nemo_config: RunConfig, tmp_path: Path) -> None:
    override = {
        "policy": {
            "sequence_packing": {
                "enabled": True,
                "train_mb_tokens": 4096,
                "algorithm": "modified_first_fit_decreasing",
            },
            "generation": {"vllm_cfg": {"enforce_eager": True}},
        },
        "grpo": {"max_val_samples": 16},
        "logger": {"monitor_gpus": True},
        "custom_section": {"a": 1},
    }
    cfg = compile_config(_with(nemo_config, trainer={"extra": {"nemo_rl": override}}), tmp_path)
    assert cfg["policy"]["sequence_packing"] == {
        "enabled": True,
        "train_mb_tokens": 4096,
        "algorithm": "modified_first_fit_decreasing",
    }
    gen = cfg["policy"]["generation"]
    assert gen["vllm_cfg"]["enforce_eager"] is True
    assert gen["vllm_cfg"]["tensor_parallel_size"] == 1  # siblings untouched
    assert gen["backend"] == "vllm"
    assert cfg["grpo"]["max_val_samples"] == 16 and cfg["grpo"]["num_prompts_per_step"] == 2
    assert cfg["logger"]["monitor_gpus"] is True
    assert cfg["custom_section"] == {"a": 1}


def test_deep_merge_copies_and_replaces_non_dicts() -> None:
    base = {"a": {"b": 1, "c": [1]}, "d": 2}
    out = deep_merge(base, {"a": {"c": [2, 3]}, "d": {"x": 1}})
    assert out == {"a": {"b": 1, "c": [2, 3]}, "d": {"x": 1}}
    assert base == {"a": {"b": 1, "c": [1]}, "d": 2}


def test_write_config_roundtrip(nemo_config: RunConfig, tmp_path: Path) -> None:
    cfg = compile_config(nemo_config, tmp_path)
    path = write_config(cfg, tmp_path)
    assert path == config_path(tmp_path) == tmp_path.resolve() / "nemo_rl" / CONFIG_FILENAME
    assert yaml.safe_load(path.read_text()) == cfg
