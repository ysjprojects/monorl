"""``monorl env serve`` serves an environment type that the config's providers register."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import yaml

from monorl.cli.envcheck import run_check

PROVIDER = """
    from monorl.core.types import StepResult


    class Echo:
        def __init__(self, max_turns):
            self.max_turns = max_turns

        def reset(self, sample):
            return list(sample.messages)

        def step(self, action):
            return StepResult([{"role": "user", "content": action.text}], reward=0.5, done=False)

        def close(self):
            pass


    def register(registry):
        registry.register("env", "echo", lambda config, **kw: Echo(config.env.max_turns))
"""


def test_env_serve_finds_a_type_from_the_configs_providers(tmp_path: Path) -> None:
    (tmp_path / "echo_provider.py").write_text(textwrap.dedent(PROVIDER))
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "providers": ["echo_provider"],
                "model": {"name_or_path": "unused"},
                "data": {"uri": "unused"},
                "rewards": [],
                "env": {"type": "echo", "max_turns": 3},
            }
        )
    )
    # the installed command does not put the working directory on sys.path; the loader does
    script = Path(sys.executable).parent / "monorl"
    command = [str(script)] if script.exists() else [sys.executable, "-m", "monorl.cli"]
    report = run_check(
        command=[*command, "env", "serve", "echo", "--config", str(config)], cwd=str(tmp_path), turns=2
    )
    assert report.ok, report.error
    assert report.spec is not None and report.spec["name"] == "echo" and report.spec["max_turns"] == 3
    assert [(step.reward, step.done) for step in report.steps] == [(0.5, False), (0.5, False)]
