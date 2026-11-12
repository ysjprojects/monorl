"""``algorithm.off_policy`` -> TRL 1.17 ``GRPOConfig.vllm_importance_sampling_*``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from monorl.adapters.trl.compile import grpo_config_kwargs
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError


def _kwargs(config: RunConfig, tmp_path: Path, off_policy: dict[str, Any], backend: str) -> dict[str, Any]:
    data = config.to_dict()
    data["algorithm"]["off_policy"] = off_policy
    data["rollout"]["backend"] = backend
    kwargs = grpo_config_kwargs(RunConfig.from_dict(data), tmp_path, device=torch.device("cpu"))
    return {k: v for k, v in kwargs.items() if k.startswith("vllm_importance_sampling")}


def test_defaults_keep_trls_own_correction(trl_config: RunConfig, tmp_path: Path) -> None:
    assert _kwargs(trl_config, tmp_path, {}, "vllm") == {}
    assert _kwargs(trl_config, tmp_path, {"anchor": "trainer"}, "hf") == {}


def test_vllm_rollouts_map_to_truncate_and_mask_modes(trl_config: RunConfig, tmp_path: Path) -> None:
    assert _kwargs(trl_config, tmp_path, {"anchor": "trainer"}, "vllm") == {
        "vllm_importance_sampling_correction": False
    }
    off = {"importance_weights": "token", "importance_cap": 2.5}
    assert _kwargs(trl_config, tmp_path, off, "vllm") == {
        "vllm_importance_sampling_correction": True,
        "vllm_importance_sampling_mode": "token_truncate",
        "vllm_importance_sampling_clip_min": None,
        "vllm_importance_sampling_clip_max": 2.5,
    }
    off = {"importance_weights": "sequence", "importance_cap": None, "ratio_mask": "sequence"}
    assert _kwargs(trl_config, tmp_path, off, "vllm") == {
        "vllm_importance_sampling_correction": True,
        "vllm_importance_sampling_mode": "sequence_mask",
        "vllm_importance_sampling_clip_min": 0.5,
        "vllm_importance_sampling_clip_max": 5.0,
    }


@pytest.mark.parametrize(
    ("off_policy", "backend", "match"),
    [
        ({"score_centering": True}, "hf", "score_centering"),
        ({"anchor": "behavior"}, "vllm", "anchor='behavior'"),
        ({"importance_weights": "token"}, "hf", "rollout.backend='vllm'"),
        ({"ratio_mask": "token"}, "vllm", "no TRL equivalent"),
        ({"importance_weights": "token", "ratio_mask": "sequence_mean"}, "vllm", "no TRL equivalent"),
        ({"importance_weights": "token", "importance_cap": None}, "vllm", "no TRL equivalent"),
    ],
)
def test_options_without_a_trl_equivalent_are_rejected(
    trl_config: RunConfig, tmp_path: Path, off_policy, backend, match
) -> None:
    with pytest.raises(ConfigError, match=match):
        _kwargs(trl_config, tmp_path, off_policy, backend)
