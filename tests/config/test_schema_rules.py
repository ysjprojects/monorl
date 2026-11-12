import pytest

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError


def _base(**overrides):
    cfg = {"model": {"name_or_path": "x"}, "data": {"uri": "d.jsonl"}, "rewards": [{"type": "exact_match"}]}
    cfg.update(overrides)
    return cfg


def test_ppo_requires_a_critic_backend() -> None:
    with pytest.raises(ConfigError, match="critic"):
        RunConfig.from_dict(_base(algorithm={"name": "ppo"}, trainer={"backend": "reference"}))
    assert (
        RunConfig.from_dict(_base(algorithm={"name": "ppo"}, trainer={"backend": "verl"})).algorithm.name
        == "ppo"
    )


def test_group_algorithms_need_groups() -> None:
    with pytest.raises(ConfigError, match="group_size"):
        RunConfig.from_dict(_base(algorithm={"name": "grpo"}, rollout={"group_size": 1}))
