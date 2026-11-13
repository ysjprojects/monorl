"""``model.mtp`` in the verl override tree (``actor_rollout_ref.model.mtp``, hf_model.yaml v0.9.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from monorl.adapters.verl.compiler import compile_config, compile_sft_config, to_overrides
from monorl.core.errors import ConfigError

MTP = "actor_rollout_ref.model.mtp."


def _mtp_overrides(config, run_dir: Path) -> dict[str, str]:
    """The Hydra overrides that mention MTP or speculative decoding, as verl's CLI receives them."""
    out = {}
    for item in to_overrides(compile_config(config, run_dir)):
        key, _, value = item.partition("=")
        if "mtp" in key or "speculative" in key:
            out[key] = value
    return out


def test_mtp_off_emits_nothing(make_config, tmp_path: Path) -> None:
    assert _mtp_overrides(make_config(), tmp_path) == {}
    sft = make_config(algorithm={"name": "sft"}, rewards=[])
    assert "mtp" not in compile_sft_config(sft, tmp_path)["model"]


def test_enabled_only_keeps_the_layers_without_drafting(make_config, tmp_path: Path) -> None:
    config = make_config(model={"name_or_path": "m", "mtp": {"enabled": True}})
    assert _mtp_overrides(config, tmp_path) == {
        MTP + "enable": "true",
        MTP + "enable_rollout": "false",
        MTP + "enable_train": "false",
    }


@pytest.mark.parametrize(("tokens", "expected"), [(None, "1"), (3, "3")])
def test_vllm_rollout_drafts_with_the_mtp_method(make_config, tmp_path: Path, tokens, expected) -> None:
    mtp = {"enabled": True, "rollout": True, "num_speculative_tokens": tokens}
    config = make_config(model={"name_or_path": "m", "mtp": mtp})
    assert _mtp_overrides(config, tmp_path) == {
        MTP + "enable": "true",
        MTP + "enable_rollout": "true",
        MTP + "enable_train": "false",
        MTP + "method": "mtp",
        MTP + "num_speculative_tokens": expected,
    }


@pytest.mark.parametrize(("tokens", "steps", "draft"), [(None, "3", "4"), (2, "2", "3")])
def test_sglang_rollout_drafts_a_chain(make_config, tmp_path: Path, tokens, steps, draft) -> None:
    mtp = {"enabled": True, "rollout": True, "num_speculative_tokens": tokens}
    config = make_config(model={"name_or_path": "m", "mtp": mtp}, rollout={"backend": "sglang"})
    assert _mtp_overrides(config, tmp_path) == {
        MTP + "enable": "true",
        MTP + "enable_rollout": "true",
        MTP + "enable_train": "false",
        MTP + "speculative_algorithm": "EAGLE",
        MTP + "speculative_num_steps": steps,
        MTP + "speculative_eagle_topk": "1",
        MTP + "speculative_num_draft_tokens": draft,
    }


def test_trainer_extra_overrides_the_compiled_mtp_keys(make_config, tmp_path: Path) -> None:
    config = make_config(
        model={"name_or_path": "m", "mtp": {"enabled": True, "rollout": True}},
        trainer={"extra": {"actor_rollout_ref": {"model": {"mtp": {"num_speculative_tokens": 2}}}}},
    )
    overrides = _mtp_overrides(config, tmp_path)
    assert overrides[MTP + "num_speculative_tokens"] == "2"
    assert overrides[MTP + "method"] == "mtp" and overrides[MTP + "enable_rollout"] == "true"


def test_sft_keeps_the_mtp_layers_in_the_exported_model(make_config, tmp_path: Path) -> None:
    config = make_config(
        algorithm={"name": "sft"}, rewards=[], model={"name_or_path": "m", "mtp": {"enabled": True}}
    )
    assert compile_sft_config(config, tmp_path)["model"]["mtp"] == {"enable": True}
    assert "model.mtp.enable=true" in to_overrides(compile_config(config, tmp_path))


@pytest.mark.parametrize("overrides", [{}, {"algorithm": {"name": "sft"}, "rewards": []}])
def test_training_mtp_layers_is_rejected_with_the_reason(make_config, overrides: dict) -> None:
    with pytest.raises(ConfigError, match=r"model\.mtp\.train.*verl.*Megatron"):
        make_config(model={"name_or_path": "m", "mtp": {"enabled": True, "train": True}}, **overrides)
