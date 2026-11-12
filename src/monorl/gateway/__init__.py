"""Capture gateway: any agent program that speaks the OpenAI chat-completions API becomes an
environment.

The gateway serves an OpenAI-compatible endpoint from the policy being trained and records every
call with its exact prompt token ids and the sampled completion (ids and logprobs), so an
episode driven by an external agent — an OpenAI-SDK script, a coding-agent CLI — becomes a
trainable :class:`~monorl.core.types.Trajectory`. ``env.type: harness``
(:mod:`monorl.envs.harness`) runs one agent process per episode against it. See
``docs/GATEWAY.md``.

* :mod:`monorl.gateway.protocol` — OpenAI wire format: validation, normalisation, bodies, errors
* :mod:`monorl.gateway.session` — :class:`Session` (one episode) and :class:`CallRecord`
* :mod:`monorl.gateway.trajectory` — :func:`build_trajectory`: calls → turns and threads
* :mod:`monorl.gateway.server` — :class:`Gateway` (HTTP server + session table), :func:`get_gateway`
* :mod:`monorl.gateway.agents` — example agents

Names are imported on first use: an agent started with ``python -m monorl.gateway.agents.<name>``
imports this package and must not pay for the HTTP server stack.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from monorl.gateway.protocol import ChatRequest, GatewayError
    from monorl.gateway.server import Gateway, get_gateway
    from monorl.gateway.session import CallRecord, Session
    from monorl.gateway.trajectory import build_trajectory

_MODULES = {
    "ChatRequest": "protocol",
    "GatewayError": "protocol",
    "Gateway": "server",
    "get_gateway": "server",
    "CallRecord": "session",
    "Session": "session",
    "build_trajectory": "trajectory",
}

__all__ = [
    "CallRecord",
    "ChatRequest",
    "Gateway",
    "GatewayError",
    "Session",
    "build_trajectory",
    "get_gateway",
]


def __getattr__(name: str) -> Any:
    module = _MODULES.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
