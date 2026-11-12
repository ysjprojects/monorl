"""``algorithm.off_policy`` -> NeMo-RL v0.7.0 ``loss_fn`` importance-sampling keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from monorl.adapters.nemo_rl.compile import compile_config
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError

KEYS = (
    "use_importance_sampling_correction",
    "sequence_level_importance_ratios",
    "truncated_importance_sampling_type",
    "truncated_importance_sampling_ratio_min",
    "truncated_importance_sampling_ratio",
)


def _keys(config: RunConfig, tmp_path: Path, off_policy: dict[str, Any], **rollout: Any) -> tuple:
    data = config.to_dict()
    data["algorithm"]["off_policy"] = off_policy
    data["rollout"].update(rollout)
    loss_fn = compile_config(RunConfig.from_dict(data), tmp_path)["loss_fn"]
    return tuple(loss_fn[key] for key in KEYS)


@pytest.mark.parametrize(
    ("off_policy", "expected"),
    [
        ({}, (False, False, None, None, None)),
        ({"anchor": "trainer"}, (False, False, None, None, None)),
        ({"importance_weights": "token", "importance_cap": 3.0}, (True, False, "tis", None, 3.0)),
        ({"importance_weights": "sequence", "importance_cap": None}, (True, True, None, None, None)),
        (
            {"importance_weights": "token", "importance_cap": None, "ratio_mask": "token"},
            (True, False, "icepop", 0.5, 5.0),
        ),
        (
            {"importance_weights": "sequence", "importance_cap": 8.0, "ratio_mask": "sequence"},
            (True, True, "icepop", 0.5, 5.0),
        ),
        (
            {
                "importance_weights": "token",
                "importance_cap": None,
                "ratio_mask": "sequence_mean",
                "ratio_mask_low": 0.999,
                "ratio_mask_high": 1.002,
            },
            (True, False, "seq-mask-tis", 0.999, 1.002),
        ),
    ],
)
def test_off_policy_options_map_to_loss_fn(nemo_config: RunConfig, tmp_path: Path, off_policy, expected):
    assert _keys(nemo_config, tmp_path, off_policy) == expected


@pytest.mark.parametrize(
    ("off_policy", "rollout", "match"),
    [
        ({"score_centering": True}, {}, "score_centering"),
        ({"anchor": "behavior"}, {}, "anchor='behavior'"),
        ({"ratio_mask": "token"}, {}, "without importance_weights"),
        # the cap would bind inside the accepted band, which icepop cannot express
        ({"importance_weights": "token", "importance_cap": 2.0, "ratio_mask": "token"}, {}, "icepop"),
        ({"importance_weights": "sequence", "ratio_mask": "token"}, {}, "icepop"),
        ({"anchor": "trainer"}, {"async_steps_ahead": 2}, "async"),
    ],
)
def test_options_without_a_nemo_rl_equivalent_are_rejected(
    nemo_config: RunConfig, tmp_path: Path, off_policy, rollout, match
):
    with pytest.raises(ConfigError, match=match):
        _keys(nemo_config, tmp_path, off_policy, **rollout)
