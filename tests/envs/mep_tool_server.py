"""MEP/1 stdio server used by the remote-env tests: a tool-calling env that can be crashed.

Usage: ``python mep_tool_server.py [parallel_episodes]``. Stepping with the text ``CRASH`` kills
the process (exit 3) to exercise client-side restart.
"""

from __future__ import annotations

import os
import sys

from monorl.core.types import Completion, StepResult
from monorl.envs.server import serve
from monorl.envs.tool_calling import ToolCallingEnv, load_tools


class CrashableToolEnv(ToolCallingEnv):
    """Tool env for tests; 'CRASH' exits the server process."""

    def step(self, action: Completion) -> StepResult:
        if action.text.strip() == "CRASH":
            sys.stderr.write("crashing on request\n")
            sys.stderr.flush()
            os._exit(3)
        return super().step(action)


def factory() -> CrashableToolEnv:
    return CrashableToolEnv(
        tools=load_tools({"tools": ["calculator", "lookup"], "corpus": {"capital of France": "Paris"}}),
        max_turns=3,
        invalid_call_penalty=0.25,
    )


if __name__ == "__main__":
    parallel = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    serve(factory, transport="stdio", name="crashable_tools", parallel_episodes=parallel)
