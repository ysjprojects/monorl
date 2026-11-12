"""``algorithm.off_policy`` -> verl v0.9.1 ``algorithm.rollout_correction`` overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from monorl.adapters.verl import compile_config, to_overrides
from monorl.core.errors import ConfigError

PREFIX = "algorithm.rollout_correction."


def _correction(config, run_dir: Path) -> dict[str, str]:
    items = (item.partition("=") for item in to_overrides(compile_config(config, run_dir)))
    return {key.removeprefix(PREFIX): value for key, _, value in items if key.startswith(PREFIX)}


def test_defaults_leave_verls_rollout_correction_untouched(make_config, tmp_path: Path):
    assert _correction(make_config(), tmp_path) == {}


def test_anchor_selects_bypass_or_decoupled_mode(make_config, tmp_path: Path):
    behavior = make_config(algorithm={"off_policy": {"anchor": "behavior"}})
    assert _correction(behavior, tmp_path) == {"bypass_mode": "true"}
    trainer = make_config(algorithm={"off_policy": {"anchor": "trainer"}})
    assert _correction(trainer, tmp_path) == {"bypass_mode": "false"}


def test_weights_and_ratio_mask_map_to_is_and_k1_rejection(make_config, tmp_path: Path):
    off = {
        "importance_weights": "sequence",
        "importance_cap": 4.0,
        "ratio_mask": "sequence_mean",
        "ratio_mask_low": 0.8,
        "ratio_mask_high": 1.25,
    }
    assert _correction(make_config(algorithm={"off_policy": off}), tmp_path) == {
        "bypass_mode": "false",
        "rollout_is": "sequence",
        "rollout_is_threshold": "4.0",
        "rollout_rs": "seq_mean_k1",
        # verl thresholds -log(ratio): the ratio band [0.8, 1.25] is [1/1.25, 1/0.8]
        "rollout_rs_threshold": "0.8_1.25",
    }
    off = {"ratio_mask": "token", "ratio_mask_low": 0.5, "ratio_mask_high": 5.0}
    assert _correction(make_config(algorithm={"off_policy": off}), tmp_path) == {
        "bypass_mode": "false",
        "rollout_rs": "token_k1",
        "rollout_rs_threshold": "0.2_2.0",
    }


@pytest.mark.parametrize(
    ("off_policy", "match"),
    [
        ({"score_centering": True}, "score_centering"),
        ({"importance_weights": "token", "importance_cap": None}, "importance_cap"),
    ],
)
def test_options_without_a_verl_equivalent_are_rejected(make_config, tmp_path: Path, off_policy, match):
    config = make_config(algorithm={"off_policy": off_policy}, rollout={"top_p": 1.0})
    with pytest.raises(ConfigError, match=match):
        compile_config(config, tmp_path)
