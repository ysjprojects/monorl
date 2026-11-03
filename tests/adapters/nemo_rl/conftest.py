"""Fixtures: a nemo_rl RunConfig and a fake ``nemo_rl`` package (in ``sys.modules`` or on disk)."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from monorl.config.schema import RunConfig

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from nemo_rl_fakes import FAKE_HOME_FILES, FAKE_VERSION, FakeLogger  # noqa: E402


@pytest.fixture
def nemo_config(run_config_dict: dict[str, Any]) -> RunConfig:
    cfg = dict(run_config_dict)
    cfg["trainer"] = {**cfg["trainer"], "backend": "nemo_rl", "precision": "bf16"}
    cfg["rollout"] = {**cfg["rollout"], "backend": "vllm"}
    cfg["cluster"] = {"mode": "ray", "num_nodes": 1, "gpus_per_node": 8}
    return RunConfig.from_dict(cfg)


@pytest.fixture
def fake_nemo_rl(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, types.ModuleType]]:
    """Install a fake ``nemo_rl`` tree in ``sys.modules`` (version 0.7.0) and clean up patches after."""
    from monorl.patching import revert

    FakeLogger.calls = []
    root = types.ModuleType("nemo_rl")
    root.__version__ = FAKE_VERSION
    root.__path__ = []  # mark as package
    utils = types.ModuleType("nemo_rl.utils")
    utils.__path__ = []
    logger = types.ModuleType("nemo_rl.utils.logger")
    logger.Logger = FakeLogger
    envs = types.ModuleType("nemo_rl.environments")
    envs.__path__ = []
    env_utils = types.ModuleType("nemo_rl.environments.utils")
    env_utils.ENV_REGISTRY = {"math": {"actor_class_fqn": "nemo_rl.environments.math_environment.MathEnvironment"}}

    def register_env(name: str, fqn: str) -> None:
        if name in env_utils.ENV_REGISTRY:
            raise ValueError(f"Env name {name} already registered")
        env_utils.ENV_REGISTRY[name] = {"actor_class_fqn": fqn}

    env_utils.register_env = register_env
    dist = types.ModuleType("nemo_rl.distributed")
    dist.__path__ = []
    actor_registry = types.ModuleType("nemo_rl.distributed.ray_actor_environment_registry")
    actor_registry.ACTOR_ENVIRONMENT_REGISTRY = {}
    virtual_cluster = types.ModuleType("nemo_rl.distributed.virtual_cluster")

    class PY_EXECUTABLES:
        SYSTEM = "python"

    virtual_cluster.PY_EXECUTABLES = PY_EXECUTABLES
    data = types.ModuleType("nemo_rl.data")
    data.__path__ = []
    interfaces = types.ModuleType("nemo_rl.data.interfaces")

    class TaskDataSpec:
        def __init__(self, task_name=None, prompt_file=None, system_prompt_file=None):
            self.task_name = task_name
            self.prompt = Path(prompt_file).read_text() if prompt_file else None
            self.system_prompt = Path(system_prompt_file).read_text() if system_prompt_file else None

    interfaces.TaskDataSpec = TaskDataSpec

    modules = {
        "nemo_rl": root,
        "nemo_rl.utils": utils,
        "nemo_rl.utils.logger": logger,
        "nemo_rl.environments": envs,
        "nemo_rl.environments.utils": env_utils,
        "nemo_rl.distributed": dist,
        "nemo_rl.distributed.ray_actor_environment_registry": actor_registry,
        "nemo_rl.distributed.virtual_cluster": virtual_cluster,
        "nemo_rl.data": data,
        "nemo_rl.data.interfaces": interfaces,
    }
    root.utils, utils.logger = utils, logger
    root.environments, envs.utils = envs, env_utils
    root.distributed = dist
    dist.ray_actor_environment_registry, dist.virtual_cluster = actor_registry, virtual_cluster
    root.data, data.interfaces = data, interfaces
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    try:
        yield modules
    finally:
        revert("nemo_rl")


@pytest.fixture
def fake_home(tmp_path: Path) -> Path:
    """A fake NeMo-RL checkout: ``examples/run_grpo.py`` plus the ``nemo_rl`` modules the entry wrapper touches."""
    home = tmp_path / "nemo-rl"
    for rel, source in FAKE_HOME_FILES.items():
        path = home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return home
