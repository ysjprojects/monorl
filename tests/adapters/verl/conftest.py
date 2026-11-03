"""Fixtures for the verl adapter tests.

``fake_verl`` writes a minimal on-disk ``verl`` package (plus a ``hydra`` shim for the entry
wrapper) mirroring the v0.9.1 surfaces the adapter touches, puts it on ``sys.path`` and cleans
``sys.modules``/the patch engine afterwards. Real-verl tests use ``pytest.importorskip``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from monorl.config.schema import RunConfig
from monorl.patching import revert_all

FAKE_VERL_VERSION = "0.9.1"

_VERL_INIT = '__version__ = "{version}"\n'

_TRACKING = '''
class LocalLogger:
    def __init__(self):
        self.rows = []

    def log(self, data, step):
        self.rows.append((dict(data), step))


class Tracking:
    supported_backend = ["console"]

    def __init__(self, project_name, experiment_name, default_backend="console", config=None):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.console_logger = LocalLogger()
        self.logger = {"console": self.console_logger}
        self.calls = []

    def log(self, data, step, backend=None):
        self.calls.append((data, step, backend))
        for name, logger in self.logger.items():
            if backend is None or name in backend:
                logger.log(data=data, step=step)

    def finish(self, exit_code=0):
        self.finished = exit_code
'''

_MAIN_PPO = '''
import ray

PPO_RAY_RUNTIME_ENV = {"env_vars": {"TOKENIZERS_PARALLELISM": "true"}}
CALLS = []


def get_ppo_ray_runtime_env(config=None):
    return {"env_vars": dict(PPO_RAY_RUNTIME_ENV["env_vars"])}


def run_ppo(config, task_runner_class):
    CALLS.append((config, task_runner_class))
    # Run the actor class body in-process (no cluster): mirrors ray.get(runner.run.remote(config)).
    runner = task_runner_class.__ray_actor_class__()
    runner.run(config)


@ray.remote
class TaskRunnerV1:
    RUNS = []

    def run(self, config):
        from verl.utils.tracking import Tracking

        tracking = Tracking("p", "e", "console", config=config)
        tracking.log({"critic/score/mean": 0.5, "actor/pg_loss": -0.1}, step=1)
        type(self).RUNS.append((config, tracking))
        return tracking
'''

_PPO_UTILS = '''
def need_reference_policy(config):
    return bool(config.get("actor_rollout_ref", {}).get("actor", {}).get("use_kl_loss"))


def need_critic(config):
    return config.get("algorithm", {}).get("adv_estimator") == "gae"
'''

_UTILS_CONFIG = '''
VALIDATED = []


def validate_config(config, use_reference_policy, use_critic):
    VALIDATED.append((config, use_reference_policy, use_critic))
'''

_UTILS_DEVICE = '''
def auto_set_device(config):
    config.setdefault("trainer", {})["device"] = "cpu"
'''

_TOOL_SCHEMAS = '''
from typing import Any

from pydantic import BaseModel, Field


class OpenAIFunctionPropertySchema(BaseModel):
    type: str | list[str]
    description: str | None = None
    enum: list[Any] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    required: list[str] = Field(default_factory=list)


class OpenAIFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    type: str
    function: OpenAIFunctionSchema


class ToolResponse(BaseModel):
    text: str | None = None
    image: list[Any] | None = None
    video: list[Any] | None = None
'''

_BASE_TOOL = '''
from uuid import uuid4

from .schemas import OpenAIFunctionToolSchema, ToolResponse


class BaseTool:
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        assert self.tool_schema is not None, "Tool schema is not set!"
        self.name = self.tool_schema.function.name

    def get_openai_tool_schema(self):
        return self.tool_schema

    async def create(self, instance_id=None, **kwargs):
        return (instance_id or str(uuid4())), ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        return ToolResponse(text="Updated the tool state."), 0.0, {}

    async def calc_reward(self, instance_id, **kwargs):
        return 0.0

    async def release(self, instance_id, **kwargs):
        pass
'''

_HYDRA = '''
"""Minimal hydra.main shim: parses ``key=value`` argv overrides into a plain dict config."""
import sys

LAST = {}


def _set(tree, dotted, value):
    parts = dotted.lstrip("+").split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def main(config_path=None, config_name=None, version_base=None):
    LAST.update({"config_path": config_path, "config_name": config_name})

    def decorator(fn):
        def wrapper(*args, **kwargs):
            config = {}
            for arg in sys.argv[1:]:
                key, _, value = arg.partition("=")
                _set(config, key, value)
            return fn(config)

        return wrapper

    return decorator
'''


@dataclass
class FakeVerl:
    root: Path
    version: str

    def module(self, name: str) -> Any:
        return sys.modules[name]


def write_fake_verl(root: Path, version: str = FAKE_VERL_VERSION) -> Path:
    verl = root / "verl"
    (verl / "utils").mkdir(parents=True)
    (verl / "trainer" / "ppo").mkdir(parents=True)
    (verl / "tools").mkdir(parents=True)
    (verl / "__init__.py").write_text(_VERL_INIT.format(version=version))
    (verl / "utils" / "__init__.py").write_text("")
    (verl / "utils" / "tracking.py").write_text(_TRACKING)
    (verl / "utils" / "config.py").write_text(_UTILS_CONFIG)
    (verl / "utils" / "device.py").write_text(_UTILS_DEVICE)
    (verl / "trainer" / "__init__.py").write_text("")
    (verl / "trainer" / "main_ppo.py").write_text(_MAIN_PPO)
    (verl / "trainer" / "ppo" / "__init__.py").write_text("")
    (verl / "trainer" / "ppo" / "utils.py").write_text(_PPO_UTILS)
    (verl / "tools" / "__init__.py").write_text("")
    (verl / "tools" / "schemas.py").write_text(_TOOL_SCHEMAS)
    (verl / "tools" / "base_tool.py").write_text(_BASE_TOOL)
    hydra = root / "hydra"
    hydra.mkdir()
    (hydra / "__init__.py").write_text(_HYDRA)
    return verl


def _purge(*names: str) -> None:
    for name in names:
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]


@pytest.fixture
def fake_verl(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeVerl]:
    pytest.importorskip("ray")
    if "verl" in sys.modules and not getattr(sys.modules["verl"], "__monorl_fake__", False):
        pytest.skip("real verl is importable; fake-module tests only run without it")
    root = tmp_path_factory.mktemp("fake-verl")
    write_fake_verl(root)
    _purge("verl", "hydra", "monorl.adapters.verl.entry", "monorl.adapters.verl.verl_tool")
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delenv("MONORL_DISABLE_PATCHES", raising=False)
    import verl  # noqa: F401

    sys.modules["verl"].__monorl_fake__ = True
    try:
        yield FakeVerl(root=root, version=FAKE_VERL_VERSION)
    finally:
        revert_all()
        _purge("verl", "hydra", "monorl.adapters.verl.entry", "monorl.adapters.verl.verl_tool")


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"prompt": "What is 2+2?", "answer": "4"}',
                '{"prompt": "What is 3+5?", "answer": "8"}',
                '{"prompt": "What is 10-4?", "answer": "6"}',
            ]
        )
        + "\n"
    )
    return path


@pytest.fixture
def base_config(data_file: Path, tmp_path: Path) -> dict[str, Any]:
    return {
        "run_name": "verl-test",
        "seed": 7,
        "model": {"name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"},
        "data": {"uri": str(data_file), "prompt_field": "prompt", "answer_field": "answer", "max_prompt_tokens": 512},
        "algorithm": {"name": "grpo", "kl_coef": 0.001},
        "rollout": {"backend": "vllm", "group_size": 5, "max_new_tokens": 1024, "temperature": 0.9, "top_p": 0.95},
        "trainer": {
            "backend": "verl",
            "learning_rate": 2e-6,
            "prompts_per_step": 16,
            "micro_batch_size": 2,
            "max_steps": 30,
            "save_every": 10,
        },
        "rewards": [{"type": "exact_match", "weight": 1.0}],
        "cluster": {"num_nodes": 2, "gpus_per_node": 8},
        "logging": {"run_root": str(tmp_path / "runs")},
    }


@pytest.fixture
def make_config(base_config: dict[str, Any]):
    def _make(**overrides: Any) -> RunConfig:
        payload = {**base_config}
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(payload.get(key), dict):
                payload[key] = {**payload[key], **value}
            else:
                payload[key] = value
        return RunConfig.from_dict(payload)

    return _make
