"""``model.mtp`` rules of the RunConfig as a user meets them."""

from __future__ import annotations

import pytest

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError


def _base(mtp: dict | None = None, **overrides):
    cfg = {"model": {"name_or_path": "x"}, "data": {"uri": "d.jsonl"}, "rewards": [{"type": "exact_match"}]}
    if mtp is not None:
        cfg["model"]["mtp"] = mtp
    cfg.update(overrides)
    return cfg


def test_a_config_without_the_section_is_unchanged() -> None:
    plain = RunConfig.from_dict(_base())
    mtp = plain.model.mtp
    assert not (mtp.enabled or mtp.rollout or mtp.train)
    dumped = plain.to_dict()
    assert RunConfig.from_dict(dumped).to_dict() == dumped
    # an all-off section is the same run as no section, on every backend
    for backend in ("reference", "unsloth", "trl", "verl", "nemo_rl"):
        with_section = RunConfig.from_dict(_base({"enabled": False}, trainer={"backend": backend}))
        without = RunConfig.from_dict(_base(trainer={"backend": backend}))
        assert with_section.to_dict() == without.to_dict()


def test_mtp_settings_survive_a_round_trip() -> None:
    mtp = {"enabled": True, "rollout": True, "num_speculative_tokens": 2, "train": True, "loss_coef": 0.5}
    cfg = RunConfig.from_dict(_base(mtp))
    again = RunConfig.from_dict(cfg.to_dict())
    assert again.model.mtp == cfg.model.mtp
    assert again.model.mtp.num_speculative_tokens == 2 and again.model.mtp.loss_coef == 0.5
    assert again.model.mtp.detach_trunk is True


@pytest.mark.parametrize("mtp", [{"rollout": True}, {"train": True}, {"rollout": True, "train": True}])
def test_rollout_and_train_need_enabled(mtp: dict) -> None:
    with pytest.raises(ConfigError, match=r"model\.mtp\.enabled"):
        RunConfig.from_dict(_base(mtp))
    assert RunConfig.from_dict(_base({"enabled": True, **mtp})).model.mtp.enabled


@pytest.mark.parametrize("mtp", [{"num_speculative_tokens": 0}, {"loss_coef": -0.1}, {"draft": True}])
def test_invalid_values_and_unknown_keys_are_rejected(mtp: dict) -> None:
    with pytest.raises(ConfigError, match="mtp"):
        RunConfig.from_dict(_base({"enabled": True, **mtp}))


@pytest.mark.parametrize(
    ("backend", "reason"), [("unsloth", "Unsloth loads the model"), ("trl", "TRL neither")]
)
def test_backends_that_cannot_hold_mtp_layers(backend: str, reason: str) -> None:
    with pytest.raises(ConfigError, match=rf"model\.mtp.*trainer\.backend='{backend}'.*{reason}"):
        RunConfig.from_dict(_base({"enabled": True}, trainer={"backend": backend}))


def test_verl_drafts_but_does_not_train() -> None:
    with pytest.raises(ConfigError, match=r"model\.mtp\.train.*trainer\.backend='verl'.*Megatron"):
        RunConfig.from_dict(_base({"enabled": True, "train": True}, trainer={"backend": "verl"}))
    cfg = RunConfig.from_dict(_base({"enabled": True, "rollout": True}, trainer={"backend": "verl"}))
    assert cfg.model.mtp.rollout and not cfg.model.mtp.train


@pytest.mark.parametrize("backend", ["reference", "nemo_rl"])
def test_backends_that_accept_every_capability(backend: str) -> None:
    mtp = {"enabled": True, "rollout": True, "train": True}
    assert RunConfig.from_dict(_base(mtp, trainer={"backend": backend})).model.mtp.train


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_min_p_cannot_be_combined_with_drafting_servers(engine: str) -> None:
    rollout = {"backend": engine, "extra": {"min_p": 0.05}}
    with pytest.raises(ConfigError, match=rf"model\.mtp\.rollout.*{engine}.*min_p"):
        RunConfig.from_dict(_base({"enabled": True, "rollout": True}, rollout=rollout))
    # min_p stays available without drafting, and drafting without min_p
    assert RunConfig.from_dict(_base({"enabled": True}, rollout=rollout)).rollout.extra["min_p"] == 0.05
    no_min_p = {"backend": engine, "extra": {"min_p": 0.0}}
    assert RunConfig.from_dict(_base({"enabled": True, "rollout": True}, rollout=no_min_p)).model.mtp.rollout


def test_min_p_with_drafting_is_fine_on_the_hf_engine() -> None:
    rollout = {"backend": "hf", "extra": {"min_p": 0.05}}
    cfg = RunConfig.from_dict(_base({"enabled": True, "rollout": True}, rollout=rollout))
    assert cfg.model.mtp.rollout and cfg.rollout.extra["min_p"] == 0.05


@pytest.mark.parametrize("backend", ["verl", "nemo_rl"])
def test_min_p_with_drafting_is_refused_where_the_backend_serves_the_rollouts(backend: str) -> None:
    """verl and NeMo-RL generate with vLLM / SGLang whatever ``rollout.backend`` says."""
    rollout = {"backend": "hf", "extra": {"min_p": 0.05}}
    with pytest.raises(ConfigError, match=rf"model\.mtp\.rollout on {backend}.*min_p"):
        RunConfig.from_dict(
            _base({"enabled": True, "rollout": True}, rollout=rollout, trainer={"backend": backend})
        )
