"""monorl environment wire protocol (MEP/1).

Environments may be implemented in any language and run out of process. The orchestrator talks
to them with JSON messages over one of two transports:

* ``stdio``: the orchestrator spawns the env binary; each request is one JSON object per line on
  stdin, each response one JSON object per line on stdout (stderr is free for logging).
* ``http``: the env is a server; requests are ``POST /<op>`` with a JSON body, responses JSON.
  ``GET /spec`` and ``GET /health`` are also served. OpenEnv servers expose the same
  ``reset``/``step`` shape; :mod:`monorl.adapters.openenv` maps their typed payloads.

Every request carries ``op`` (stdio) or is routed by path (http) and an ``episode_id`` (except
``spec``/``health``). Servers MAY serve many episodes concurrently (``spec.parallel_episodes``);
the orchestrator never sends two requests for the same episode concurrently.

Operations (request -> response):

``spec``    {}                                              -> EnvSpec
``reset``   {episode_id, sample: Sample, seed?}             -> {messages, info}
``step``    {episode_id, action: Action}                    -> {messages, reward, done, info}
``close``   {episode_id}                                    -> {}
``health``  {}                                              -> {ok: true}

Errors: ``{"error": {"type": str, "message": str}}`` with HTTP 4xx/5xx on the http transport.

``Sample``: ``{"id": str, "messages": [Message], "reference": any, "metadata": {}}``
``Message``: ``{"role": "system"|"user"|"assistant"|"tool", "content": str | [Part], "tool_calls"?: [...],
"tool_call_id"?: str, "name"?: str}``
``Part``: ``{"type": "text", "text": str}`` or ``{"type": "image", "image": str, "detail"?: str}``
(``audio``/``video`` likewise). Media parts carry only a reference — an absolute local path,
``file://``/``s3://``/``http(s)://`` URI or ``data:image/...;base64,`` URL — never raw bytes, so
every wire message stays JSON. Environments that only care about text SHOULD concatenate the
``text`` parts (``monorl.core.types.message_text``); environments that reply with media use the
same parts in their own ``messages``.
``Action``: ``{"text": str, "tool_calls"?: [{"id", "name", "arguments"}], "finish_reason": str}``

This module is torch-free and is the single source of truth for the Python side; the Go and C++
SDKs under ``envs/sdk`` implement the same shapes. ``PROTOCOL_VERSION`` must match.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from monorl.core.types import Completion, Message, Sample

PROTOCOL_VERSION = "MEP/1"

OPS: tuple[str, ...] = ("spec", "reset", "step", "close", "health")


@dataclass(slots=True)
class EnvSpec:
    name: str
    version: str = "0"
    protocol: str = PROTOCOL_VERSION
    max_turns: int = 1
    parallel_episodes: int = 1
    """How many episodes the server can drive concurrently."""
    tools: list[dict[str, Any]] = field(default_factory=list)
    """OpenAI-style tool schemas exposed to the policy, if any."""
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnvSpec:
        known = {f for f in cls.__dataclass_fields__}
        payload = {k: v for k, v in d.items() if k in known}
        extra = {k: v for k, v in d.items() if k not in known}
        spec = cls(**payload)
        if extra:
            spec.extra.update(extra)
        return spec


@dataclass(slots=True)
class ProtocolError(Exception):
    type: str
    message: str

    def __str__(self) -> str:
        return f"{self.type}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"type": self.type, "message": self.message}}


# -- encoding helpers (used by the Python client AND by Python-implemented servers) -----------


def encode_sample(sample: Sample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "messages": list(sample.messages),
        "reference": sample.reference,
        "metadata": dict(sample.metadata),
    }


def decode_sample(d: dict[str, Any]) -> Sample:
    return Sample(
        id=str(d["id"]),
        messages=list(d.get("messages", [])),
        reference=d.get("reference"),
        metadata=dict(d.get("metadata") or {}),
    )


def encode_action(completion: Completion, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    action: dict[str, Any] = {"text": completion.text, "finish_reason": completion.finish_reason}
    if tool_calls:
        action["tool_calls"] = tool_calls
    return action


def reset_request(episode_id: str, sample: Sample, seed: int | None = None) -> dict[str, Any]:
    req: dict[str, Any] = {"op": "reset", "episode_id": episode_id, "sample": encode_sample(sample)}
    if seed is not None:
        req["seed"] = seed
    return req


def step_request(episode_id: str, completion: Completion, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"op": "step", "episode_id": episode_id, "action": encode_action(completion, tool_calls)}


def close_request(episode_id: str) -> dict[str, Any]:
    return {"op": "close", "episode_id": episode_id}


def spec_request() -> dict[str, Any]:
    return {"op": "spec"}


def health_request() -> dict[str, Any]:
    return {"op": "health"}


def parse_response(payload: dict[str, Any] | str | bytes) -> dict[str, Any]:
    """Decode a response object, raising :class:`ProtocolError` for error envelopes."""
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ProtocolError("protocol", f"response must be a JSON object, got {type(payload).__name__}")
    err = payload.get("error")
    if err:
        if isinstance(err, dict):
            raise ProtocolError(str(err.get("type", "error")), str(err.get("message", "")))
        raise ProtocolError("error", str(err))
    return payload


def validate_messages(messages: Any) -> list[Message]:
    if not isinstance(messages, list):
        raise ProtocolError("protocol", "messages must be a list")
    out: list[Message] = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise ProtocolError("protocol", f"bad message: {m!r}")
        out.append(m)  # type: ignore[arg-type]
    return out


def validate_step_response(payload: dict[str, Any]) -> tuple[list[Message], float, bool, dict[str, Any]]:
    msgs = validate_messages(payload.get("messages", []))
    try:
        reward = float(payload.get("reward", 0.0))
    except (TypeError, ValueError) as e:
        raise ProtocolError("protocol", f"reward must be a number: {payload.get('reward')!r}") from e
    done = bool(payload.get("done", True))
    info = payload.get("info") or {}
    if not isinstance(info, dict):
        raise ProtocolError("protocol", "info must be an object")
    return msgs, reward, done, info


def validate_reset_response(payload: dict[str, Any]) -> tuple[list[Message], dict[str, Any]]:
    msgs = validate_messages(payload.get("messages", []))
    info = payload.get("info") or {}
    if not isinstance(info, dict):
        raise ProtocolError("protocol", "info must be an object")
    return msgs, info
