"""``model.mtp`` in the NeMo-RL tree: vLLM ``speculative_config`` and the Megatron ``mtp_*`` keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from monorl.adapters.nemo_rl.compile import compile_config
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError

MTP_KEYS = ("mtp_num_layers", "mtp_loss_scaling_factor", "mtp_use_repeated_layer", "mtp_detach_heads")


def _config(
    base: RunConfig,
    mtp: dict[str, Any],
    *,
    megatron: dict[str, Any] | None = None,
    vllm_kwargs: dict[str, Any] | None = None,
    sft: bool = False,
) -> RunConfig:
    data = base.to_dict()
    data["model"] = {**data["model"], "mtp": mtp}
    if megatron is not None:
        data["trainer"] = {**data["trainer"], "extra": {"megatron": megatron}}
    if vllm_kwargs is not None:
        data["rollout"] = {**data["rollout"], "extra": {"vllm_kwargs": vllm_kwargs}}
    if sft:
        data["algorithm"] = {"name": "sft"}
        data["rewards"] = []
    return RunConfig.from_dict(data)


def _mtp_cfg(compiled: dict[str, Any]) -> dict[str, Any]:
    megatron = compiled["policy"]["megatron_cfg"]
    return {key: megatron[key] for key in MTP_KEYS}


def test_mtp_off_or_enabled_only_changes_nothing(nemo_config: RunConfig, tmp_path: Path) -> None:
    plain = compile_config(nemo_config, tmp_path)
    assert plain["policy"]["generation"]["vllm_kwargs"] == {}
    held = compile_config(_config(nemo_config, {"enabled": True}), tmp_path)
    # the bridge env carries the RunConfig itself; everything NeMo-RL reads is unchanged
    assert {k: v for k, v in held.items() if k != "env"} == {k: v for k, v in plain.items() if k != "env"}


@pytest.mark.parametrize(("tokens", "expected"), [(None, 1), (3, 3)])
def test_rollout_adds_the_speculative_config(
    nemo_config: RunConfig, tmp_path: Path, tokens, expected
) -> None:
    mtp = {"enabled": True, "rollout": True, "num_speculative_tokens": tokens}
    compiled = compile_config(_config(nemo_config, mtp), tmp_path)
    assert compiled["policy"]["generation"]["vllm_kwargs"] == {
        "speculative_config": {"method": "mtp", "num_speculative_tokens": expected}
    }
    assert compiled["policy"]["megatron_cfg"] == {"enabled": False}


def test_user_speculative_config_keys_win(nemo_config: RunConfig, tmp_path: Path) -> None:
    config = _config(
        nemo_config,
        {"enabled": True, "rollout": True, "num_speculative_tokens": 2},
        vllm_kwargs={
            "speculative_config": {"num_speculative_tokens": 5, "disable_padded_drafter_batch": True}
        },
    )
    assert compile_config(config, tmp_path)["policy"]["generation"]["vllm_kwargs"]["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 5,
        "disable_padded_drafter_batch": True,
    }
    # without MTP rollouts the user's kwargs pass through untouched
    passthrough = _config(nemo_config, {}, vllm_kwargs={"speculative_config": {"method": "eagle3"}})
    assert compile_config(passthrough, tmp_path)["policy"]["generation"]["vllm_kwargs"] == {
        "speculative_config": {"method": "eagle3"}
    }


@pytest.mark.parametrize("sft", [False, True])
def test_train_maps_onto_the_megatron_policy(nemo_config: RunConfig, tmp_path: Path, sft: bool) -> None:
    config = _config(
        nemo_config,
        {"enabled": True, "train": True, "loss_coef": 0.3},
        megatron={"enabled": True, "mtp_num_layers": 2},
        sft=sft,
    )
    assert _mtp_cfg(compile_config(config, tmp_path)) == {
        "mtp_num_layers": 2,
        "mtp_loss_scaling_factor": 0.3,
        "mtp_use_repeated_layer": False,
        "mtp_detach_heads": True,
    }


def test_train_through_the_trunk_and_explicit_megatron_values(nemo_config: RunConfig, tmp_path: Path) -> None:
    mtp = {"enabled": True, "train": True, "loss_coef": 0.3, "detach_trunk": False}
    through = _config(nemo_config, mtp, megatron={"enabled": True, "mtp_num_layers": 1})
    assert _mtp_cfg(compile_config(through, tmp_path))["mtp_detach_heads"] is False
    explicit = _config(
        nemo_config,
        mtp,
        megatron={
            "enabled": True,
            "mtp_num_layers": 5,
            "mtp_use_repeated_layer": True,
            "mtp_loss_scaling_factor": 0.05,
            "mtp_detach_heads": True,
        },
    )
    assert _mtp_cfg(compile_config(explicit, tmp_path)) == {
        "mtp_num_layers": 5,
        "mtp_loss_scaling_factor": 0.05,
        "mtp_use_repeated_layer": True,
        "mtp_detach_heads": True,
    }


def test_megatron_without_mtp_training_keeps_mtp_off(nemo_config: RunConfig, tmp_path: Path) -> None:
    config = _config(nemo_config, {"enabled": True, "rollout": True}, megatron={"enabled": True})
    assert _mtp_cfg(compile_config(config, tmp_path)) == {
        "mtp_num_layers": 0,
        "mtp_loss_scaling_factor": 0.0,
        "mtp_use_repeated_layer": False,
        "mtp_detach_heads": False,
    }


@pytest.mark.parametrize("sft", [False, True])
@pytest.mark.parametrize("megatron", [None, {"enabled": False, "mtp_num_layers": 1}])
def test_train_with_dtensor_is_rejected(nemo_config: RunConfig, tmp_path: Path, sft: bool, megatron) -> None:
    config = _config(nemo_config, {"enabled": True, "train": True}, megatron=megatron, sft=sft)
    with pytest.raises(ConfigError, match=r"model\.mtp\.train.*DTensor") as err:
        compile_config(config, tmp_path)
    assert "trainer.extra.megatron.enabled" in str(err.value)
    assert "trainer.extra.megatron.mtp_num_layers" in str(err.value)


@pytest.mark.parametrize("num_layers", [None, 0, -1, True, "2"])
def test_train_needs_the_number_of_mtp_layers(nemo_config: RunConfig, tmp_path: Path, num_layers) -> None:
    megatron = {"enabled": True} if num_layers is None else {"enabled": True, "mtp_num_layers": num_layers}
    config = _config(nemo_config, {"enabled": True, "train": True}, megatron=megatron)
    with pytest.raises(ConfigError, match=r"trainer\.extra\.megatron\.mtp_num_layers") as err:
        compile_config(config, tmp_path)
    assert "model.mtp.train" in str(err.value)


def test_speculative_config_must_be_a_mapping(nemo_config: RunConfig, tmp_path: Path) -> None:
    config = _config(
        nemo_config, {"enabled": True, "rollout": True}, vllm_kwargs={"speculative_config": "mtp"}
    )
    with pytest.raises(ConfigError, match=r"rollout\.extra\.vllm_kwargs\.speculative_config"):
        compile_config(config, tmp_path)
