"""Against an installed verl (>=0.9.1): compose the real Hydra tree with monorl's overrides, check
the patch targets resolve, and run the BaseTool bridge. Skipped when verl is absent."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

verl = pytest.importorskip("verl")
hydra = pytest.importorskip("hydra")

from hydra import compose, initialize_config_module  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from monorl.adapters.verl import PATCHES, VERL_VERSIONS, compile_config, to_overrides  # noqa: E402
from monorl.adapters.verl.tools import build_tool_config  # noqa: E402
from monorl.patching import apply, register_patchset, status  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("algorithm", ["grpo", "dapo", "rloo", "ppo"])
def test_overrides_compose_with_real_config(make_config, tmp_path: Path, algorithm: str):
    cfg = make_config(
        algorithm={
            "name": algorithm,
            "clip_eps_high": 0.28,
            "dynamic_sampling": algorithm == "dapo",
            "overlong_penalty": 1.0,
        }
    )
    overrides = to_overrides(compile_config(cfg, tmp_path))
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        composed = compose(config_name="ppo_trainer", overrides=overrides)
    assert composed.data.train_batch_size == 16
    assert composed.actor_rollout_ref.rollout.n == 5
    assert composed.trainer.default_local_dir == str(tmp_path / "checkpoints")
    assert composed.reward.custom_reward_function.path == str(tmp_path / "monorl_reward.py")
    assert (
        OmegaConf.select(composed, "algorithm.adv_estimator")
        == {"grpo": "grpo", "dapo": "grpo", "rloo": "rloo", "ppo": "gae"}[algorithm]
    )


def test_tool_calling_overrides_compose(make_config, tmp_path: Path):
    cfg = make_config(env={"type": "tool_calling", "max_turns": 3, "params": {"tools": ["calculator"]}})
    overrides = to_overrides(compile_config(cfg, tmp_path))
    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        composed = compose(config_name="ppo_trainer", overrides=overrides)
    assert composed.actor_rollout_ref.rollout.multi_turn.enable is True
    assert composed.actor_rollout_ref.rollout.agent.default_agent_loop == "tool_agent"


def test_patches_apply_to_real_verl():
    register_patchset(PATCHES)
    report = apply("verl")
    assert not report.failed, report.failed
    assert {s.id for s in status("verl") if s.applicable} == {p.id for p in PATCHES.patches}, VERL_VERSIONS


def test_real_base_tool_bridge(make_config):
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.tools.tool_registry import get_tool_class

    spec = build_tool_config(make_config(env={"type": "tool_calling", "params": {"tools": ["calculator"]}}))[
        "tools"
    ][0]
    tool_cls = get_tool_class(spec["class_name"])
    tool = tool_cls(
        config=spec["config"], tool_schema=OpenAIFunctionToolSchema.model_validate(spec["tool_schema"])
    )

    async def drive():
        instance_id, _ = await tool.create()
        response, reward, _ = await tool.execute(instance_id, {"expression": "6*7"})
        await tool.release(instance_id)
        return response.text, reward

    assert asyncio.run(drive()) == ("42", 0.0)
