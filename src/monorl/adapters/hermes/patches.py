"""Patches for hermes-agent 0.19 (framework name ``hermes``, distribution ``hermes-agent``).

:class:`~monorl.adapters.hermes.env.HermesEnv` inverts control at Hermes' executor seam: it
builds the system prompt with ``AIAgent._build_system_prompt``, parses the policy's tool calls
itself and executes them through ``AIAgent._execute_tool_calls`` — Hermes' LLM-calling loop
(``agent.conversation_loop.run_conversation``) is never entered and no provider shim is needed.
hermes-agent 0.19 therefore needs no monkeypatch to accept an external policy; this PatchSet is
registered empty so ``monorl patches list`` / ``monorl doctor`` report the framework as covered
with zero patches rather than unknown.
"""

from __future__ import annotations

from monorl.patching import PatchSet

PATCHES = PatchSet(framework="hermes", distribution="hermes-agent")

__all__ = ["PATCHES"]
