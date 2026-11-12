"""``BaseTool`` subclass that runs a monorl tool inside verl's tool agent loop.

Imported by verl only (``verl/tools/tool_registry.py::get_tool_class``); the module name is what
:mod:`monorl.adapters.verl.tools` writes as ``class_name``. Contract (verl v0.9.1,
``verl/tools/base_tool.py`` + ``verl/experimental/agent_loop/tool_agent_loop.py::_call_tool``)::

    instance_id, _ = await tool.create(create_kwargs=...)
    response, reward, metrics = await tool.execute(instance_id, parameters, agent_data=...)
    await tool.release(instance_id)

monorl tools are stateless (``Tool.call(arguments) -> str``), so ``create``/``release`` are the
base-class no-ops and ``execute`` returns the tool text with a zero step reward.
"""

from __future__ import annotations

import time
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from monorl.envs.tool_calling import Tool, load_tools


class MonorlTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema) -> None:
        super().__init__(config, tool_schema)
        name = config["tool_name"]
        tools: dict[str, Tool] = {t.name: t for t in load_tools(dict(config.get("env_params") or {}))}
        if name not in tools:
            raise KeyError(f"monorl tool {name!r} not found; available: {sorted(tools)}")
        self.tool = tools[name]

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs: Any
    ) -> tuple[ToolResponse, float, dict]:
        start = time.perf_counter()
        text = self.tool.call(dict(parameters))
        return ToolResponse(text=text), 0.0, {"tool/latency_s": time.perf_counter() - start}
