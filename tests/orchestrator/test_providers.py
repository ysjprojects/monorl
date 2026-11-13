"""A run loads the provider modules its config names and uses what they register."""

from __future__ import annotations

import sys
import textwrap
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from monorl.core.registry import registry
from monorl.core.reporting import iter_metrics, read_status
from monorl.orchestrator import run

PROVIDER = """
    from monorl.core.types import RewardResult, StepResult


    class Half:
        def __init__(self, params):
            self.name = "{reward}"

        def __call__(self, trajectories):
            return [RewardResult(score=0.5, breakdown={{self.name: 0.5}}) for _ in trajectories]


    class OneTurn:
        max_turns = 1

        def reset(self, sample):
            return list(sample.messages)

        def step(self, action):
            return StepResult([], reward=0.25, done=True)

        def close(self):
            pass


    def register(registry):
        registry.register("reward", "{reward}", lambda params, **kw: Half(params))
        registry.register("env", "{env}", lambda config, **kw: OneTurn())
"""


@pytest.fixture
def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, str]]:
    """A provider module in the working directory, which is not on ``sys.path``; whatever it adds to
    the process-wide registry is removed again."""
    tag = uuid.uuid4().hex[:10]
    names = {"module": f"provider_{tag}", "reward": f"half_{tag}", "env": f"one_turn_{tag}"}
    (tmp_path / f"{names['module']}.py").write_text(textwrap.dedent(PROVIDER).format(**names))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", ".", str(tmp_path))])
    yield names
    sys.modules.pop(names["module"], None)
    registry._loaded_providers.discard(names["module"])
    registry._items["reward"].pop(names["reward"], None)
    registry._items["env"].pop(names["env"], None)


def test_a_run_uses_the_reward_and_the_environment_of_its_providers(
    make_config: Any, provider: dict[str, str]
) -> None:
    cfg = make_config(
        providers=[provider["module"]],
        rewards=[{"type": provider["reward"]}],
        env={"type": provider["env"]},
        trainer={"max_steps": 1},
    )
    result = run(cfg)
    assert result.state == "succeeded", result.error
    row = next(row for _, row in iter_metrics(Path(result.run_dir)) if row["phase"] == "train")
    assert row[f"reward/{provider['reward']}"] == pytest.approx(0.5)
    assert row["reward/env"] == pytest.approx(0.25)  # the turn reward of the provider's environment
    assert row["reward/mean"] == pytest.approx(0.75)


def test_the_reward_and_environment_builders_load_providers_themselves(
    make_config: Any, provider: dict[str, str]
) -> None:
    """The bridges of pipeline backends build rewards and environments in processes of their own,
    where no run loop has loaded anything: the builders load the config's providers."""
    from monorl.core.types import Completion, Sample
    from monorl.envs import make_env
    from monorl.rewards import build_reward

    cfg = make_config(
        providers=[provider["module"]], rewards=[{"type": provider["reward"]}], env={"type": provider["env"]}
    )
    assert not registry.contains("reward", provider["reward"])
    reward = build_reward(cfg.rewards, config=cfg)
    assert reward([]) == []
    registry._loaded_providers.discard(provider["module"])
    registry._items["reward"].pop(provider["reward"])
    registry._items["env"].pop(provider["env"])
    sys.modules.pop(provider["module"])

    env = make_env(cfg)
    assert env.reset(Sample(id="s", messages=[{"role": "user", "content": "hi"}])) == [
        {"role": "user", "content": "hi"}
    ]
    assert env.step(Completion(text="x", token_ids=[])).reward == 0.25


def test_a_provider_that_cannot_be_imported_fails_the_run_by_name(make_config: Any, tmp_path: Path) -> None:
    missing = f"provider_{uuid.uuid4().hex[:10]}"
    result = run(make_config(providers=[missing]))
    assert result.state == "failed"
    assert f"provider {missing} failed to import" in read_status(Path(result.run_dir))["message"]
