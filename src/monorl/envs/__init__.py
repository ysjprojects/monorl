"""Built-in environments provider.

Names: ``single_turn`` (prompt is the whole episode), ``tool_calling`` (multi-turn tool use),
``remote`` (MEP/1 client for Go/C++/any-language env servers), ``harness`` (an external agent
program that calls the policy through the capture gateway, :mod:`monorl.gateway`). ``openenv``
and ``hermes`` are adapter-owned (:mod:`monorl.adapters`) and resolved lazily by the registry.

``make_env(config)`` builds a fresh environment for one episode from ``config.env``: a stepped
:class:`~monorl.core.interfaces.Environment`, or for ``harness`` an
:class:`~monorl.core.interfaces.EpisodeDriver` that runs the whole episode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monorl.core.interfaces import Environment, EpisodeDriver
from monorl.core.registry import ensure_providers, registry
from monorl.envs.harness import HarnessEnv, make_harness_env, policy_processor
from monorl.envs.protocol import EnvSpec
from monorl.envs.remote import RemoteEnv, make_remote_env
from monorl.envs.single_turn import SingleTurnEnv
from monorl.envs.tool_calling import (
    Tool,
    ToolCallingEnv,
    load_tools,
    make_tool_calling_env,
    parse_tool_calls,
    tool,
)

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.registry import Registry


def _max_turns_override(config: RunConfig) -> int | None:
    """``env.max_turns`` only when the user set it explicitly (the schema default is 1)."""
    return config.env.max_turns if "max_turns" in config.env.model_fields_set else None


def _make_single_turn(config: RunConfig, **kwargs: Any) -> SingleTurnEnv:
    params = dict(config.env.params)
    params.setdefault("system_prompt", config.data.system_prompt)
    return SingleTurnEnv(**params)


def _make_tool_calling(config: RunConfig, **kwargs: Any) -> ToolCallingEnv:
    override = _max_turns_override(config)
    params = config.env.params
    return make_tool_calling_env(
        params,
        max_turns=override if override is not None else int(params.get("max_turns", 4)),
        system_prompt=None,
    )


def _make_remote(config: RunConfig, **kwargs: Any) -> RemoteEnv:
    return make_remote_env(config.env.params, max_turns=_max_turns_override(config), seed=kwargs.get("seed"))


def _make_harness(config: RunConfig, **kwargs: Any) -> HarnessEnv:
    return make_harness_env(
        config.env.params,
        max_turns=_max_turns_override(config),
        model=config.model.name_or_path,
        processor=lambda: policy_processor(config),
    )


def make_env(config: RunConfig, **kwargs: Any) -> Environment | EpisodeDriver:
    ensure_providers(config)
    return registry.get("env", config.env.type)(config, **kwargs)


def register(registry: Registry) -> None:
    registry.register("env", "single_turn", _make_single_turn)
    registry.register("env", "tool_calling", _make_tool_calling)
    registry.register("env", "remote", _make_remote)
    registry.register("env", "harness", _make_harness)


__all__ = [
    "EnvSpec",
    "HarnessEnv",
    "RemoteEnv",
    "SingleTurnEnv",
    "Tool",
    "ToolCallingEnv",
    "load_tools",
    "make_env",
    "parse_tool_calls",
    "register",
    "tool",
]
