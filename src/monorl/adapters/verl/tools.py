"""Tool-calling bridge: monorl tools -> verl multi-turn tool config.

verl v0.9.1 loads ``actor_rollout_ref.rollout.multi_turn.tool_config_path`` through
``verl/tools/tool_registry.py::initialize_tools_from_config``: a yaml with ``tools: [{class_name,
config: {type: native, ...}, tool_schema}]``. ``class_name`` is imported by *module name*
(``importlib.util.find_spec``), so the bridge class lives in the importable module
:mod:`monorl.adapters.verl.verl_tool` and the per-run specifics (tool name, ``env.params``) go into
each tool's ``config`` dict, which verl hands back to ``BaseTool.__init__``.

Tools come from :func:`monorl.envs.tool_calling.load_tools` (``env.params["tools"]``, built-ins
``calculator``/``python_exec``/``lookup`` plus ``@tool``-registered ones).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from monorl.config.schema import RunConfig

TOOL_CONFIG_FILE = "verl_tools.yaml"
TOOL_CLASS = "monorl.adapters.verl.verl_tool.MonorlTool"
TOOL_TYPE = "native"


def tool_config_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / TOOL_CONFIG_FILE


def _yaml_safe(value: Any) -> Any:
    """``env.params`` may hold non-yaml-native objects (e.g. a corpus mapping); JSON-normalise them."""
    return json.loads(json.dumps(value, default=str))


def build_tool_config(config: RunConfig) -> dict[str, Any]:
    """The verl tool yaml tree for ``config.env`` (requires ``env.type == "tool_calling"``)."""
    if config.env.type != "tool_calling":
        raise ValueError(f"tool config needs env.type='tool_calling', got {config.env.type!r}")
    from monorl.envs.tool_calling import load_tools

    env_params = _yaml_safe(dict(config.env.params))
    tools = []
    for tool in load_tools(config.env.params):
        tools.append(
            {
                "class_name": TOOL_CLASS,
                "config": {"type": TOOL_TYPE, "tool_name": tool.name, "env_params": env_params},
                "tool_schema": {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": _yaml_safe(tool.parameters),
                    },
                },
            }
        )
    if not tools:
        raise ValueError("tool_calling env resolved to zero tools; set env.params.tools")
    return {"tools": tools}


def write_tool_config(config: RunConfig, run_dir: str | Path) -> Path:
    path = tool_config_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(build_tool_config(config), sort_keys=False))
    return path
