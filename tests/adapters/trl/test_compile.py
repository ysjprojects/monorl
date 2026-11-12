from __future__ import annotations

from pathlib import Path

import pytest
import torch

from monorl.adapters.trl.compile import compile_grpo_config, grpo_config_kwargs, split_batch
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError

CPU = torch.device("cpu")


def test_core_mapping(trl_config: RunConfig, tmp_path: Path) -> None:
    kw = grpo_config_kwargs(trl_config, tmp_path, device=CPU)
    assert kw["output_dir"] == str(tmp_path / "checkpoints")
    assert kw["num_generations"] == 2
    assert kw["per_device_train_batch_size"] == 2 and kw["gradient_accumulation_steps"] == 2
    assert kw["max_completion_length"] == 8
    assert kw["max_steps"] == 2 and kw["save_steps"] == 1 and kw["logging_steps"] == 1
    assert kw["loss_type"] == "bnpo" and kw["scale_rewards"] == "group"
    assert kw["beta"] == 0.0 and kw["epsilon"] == 0.2 and kw["epsilon_high"] is None
    assert kw["report_to"] == "none" and kw["use_cpu"] is True
    assert kw["bf16"] is False and kw["fp16"] is False
    assert kw["seed"] == 42 and kw["top_k"] == 0
    assert "use_vllm" not in kw


def test_split_batch_respects_micro_batch_and_group_divisibility(trl_config: RunConfig) -> None:
    cfg = trl_config.model_copy(deep=True)
    cfg.trainer.prompts_per_step, cfg.rollout.group_size, cfg.trainer.micro_batch_size = 3, 4, 8
    per_device, accum = split_batch(cfg)
    assert per_device * accum == 12 and per_device <= 8 and per_device == 6


def test_dapo_and_precision_on_cuda(trl_config: RunConfig, tmp_path: Path) -> None:
    cfg = trl_config.model_copy(deep=True)
    cfg.algorithm.name = "dapo"
    cfg.algorithm.clip_eps_high = 0.28
    cfg.algorithm.advantage_std_norm = False
    cfg.algorithm.kl_coef = 0.01
    cfg.algorithm.num_epochs = 2
    cfg.trainer.precision = "bf16"
    cfg.trainer.gradient_checkpointing = True
    kw = grpo_config_kwargs(cfg, tmp_path, device=torch.device("cuda"))
    assert kw["loss_type"] == "dapo" and kw["scale_rewards"] == "none"
    assert kw["epsilon_high"] == 0.28 and kw["beta"] == 0.01 and kw["num_iterations"] == 2
    assert kw["bf16"] is True and kw["use_cpu"] is False and kw["gradient_checkpointing"] is True


def test_vllm_server_and_colocate(trl_config: RunConfig, tmp_path: Path) -> None:
    cfg = trl_config.model_copy(deep=True)
    cfg.rollout.backend = "vllm"
    kw = grpo_config_kwargs(cfg, tmp_path, device=CPU)
    assert kw["use_vllm"] is True and kw["vllm_mode"] == "colocate"
    cfg.rollout.extra = {"server_host": "10.0.0.5", "server_port": 8123}
    kw = grpo_config_kwargs(cfg, tmp_path, device=CPU)
    assert (
        kw["vllm_mode"] == "server"
        and kw["vllm_server_host"] == "10.0.0.5"
        and kw["vllm_server_port"] == 8123
    )


def test_extra_passthrough_and_unknown_keys(trl_config: RunConfig, tmp_path: Path) -> None:
    cfg = trl_config.model_copy(deep=True)
    cfg.trainer.extra = {"mask_truncated_completions": True, "loss_type": "dr_grpo"}
    args = compile_grpo_config(cfg, tmp_path, device=CPU)
    assert args.mask_truncated_completions is True and args.loss_type == "dr_grpo"
    assert args.generation_batch_size == 4
    cfg.trainer.extra = {"not_a_field": 1}
    with pytest.raises(ConfigError, match="not_a_field"):
        compile_grpo_config(cfg, tmp_path, device=CPU)


def test_unsupported_algorithm_is_a_config_error(trl_config: RunConfig, tmp_path: Path) -> None:
    cfg = trl_config.model_copy(deep=True)
    cfg.algorithm.name = "rloo"
    with pytest.raises(ConfigError, match="rloo"):
        grpo_config_kwargs(cfg, tmp_path, device=CPU)
