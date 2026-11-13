"""``RunConfig.providers``: what the schema accepts."""

from __future__ import annotations

import pytest

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError

BASE = {"model": {"name_or_path": "x"}, "data": {"uri": "d.jsonl"}, "rewards": [{"type": "exact_match"}]}


def test_providers_are_module_paths_listed_once() -> None:
    assert RunConfig.from_dict(BASE).providers == []
    cfg = RunConfig.from_dict({**BASE, "providers": ["my_rewards", "mypkg.envs.guess", "my_rewards"]})
    assert cfg.providers == ["my_rewards", "mypkg.envs.guess"]
    assert RunConfig.from_dict(cfg.to_dict()).providers == cfg.providers


@pytest.mark.parametrize("entry", ["./my_rewards.py", "my_rewards.py:register", "my-rewards", "pkg..mod", ""])
def test_a_file_path_or_a_malformed_name_is_rejected(entry: str) -> None:
    with pytest.raises(ConfigError, match="is not a module path"):
        RunConfig.from_dict({**BASE, "providers": [entry]})
