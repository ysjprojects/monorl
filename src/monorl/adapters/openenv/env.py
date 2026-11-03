"""``env: openenv`` — drive an OpenEnv (openenv-core 0.3) environment as a monorl Environment.

Three ways to reach the OpenEnv environment (``env.params``):

* ``url``: a running OpenEnv server; uses ``openenv.core.GenericEnvClient(base_url).sync()``
  (one WebSocket session per episode, ``reset``/``step`` over ``/ws``).
* ``env_class``: dotted path of an in-process ``openenv.core.env_server.Environment`` subclass
  (``env_kwargs`` are passed to its constructor; the action class is taken from ``action_class``
  or inferred from ``step``'s annotation / generic base).
* ``image``: a Docker image started with ``openenv.core.containers.runtime.LocalDockerProvider``
  (``env_vars`` forwarded); the container is stopped on ``close``.

Mapping:

* observation → text via ``observation_template`` (``str.format`` over the observation fields,
  e.g. ``"{output}"``); default: the single string field if there is exactly one, else JSON.
  The text becomes a ``user`` message; ``system_prompt`` (or ``data.system_prompt``) is prepended.
* completion → action: ``action_mode="text"`` → ``{action_field: completion.text}``
  (``action_field`` defaults to ``"text"``); ``action_mode="json"`` → the completion must contain a
  JSON object (optionally fenced) used as the action; a malformed completion costs
  ``invalid_action_penalty``, gets an explanatory ``user`` message and does not touch the env.
* reward/done from the OpenEnv step result (``reward=None`` → 0.0); ``done`` is also forced after
  ``max_turns`` policy turns.
* ``reset`` kwargs: ``episode_id=sample.id``, ``seed`` (when configured), ``params.reset_kwargs``
  and ``sample.metadata["reset_kwargs"]`` (per-sample task parameters).
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import typing
from collections.abc import Mapping
from typing import Any, Protocol

from monorl.core.errors import BackendUnavailable, EnvironmentError_
from monorl.core.types import Completion, Message, Sample, StepResult

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _require_openenv() -> Any:
    try:
        import openenv.core as core
    except ImportError as e:  # pragma: no cover - exercised only without the extra installed
        raise BackendUnavailable("env 'openenv' needs the openenv-core package (pip install monorl[openenv])") from e
    return core


def _import_dotted(path: str) -> Any:
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise ValueError(f"expected a dotted path 'module.Attr', got {path!r}")
    return getattr(importlib.import_module(module_name), attr)


class Backend(Protocol):
    """Uniform access to an OpenEnv env: dict observations in, dict actions out."""

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], float | None, bool]: ...

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float | None, bool]: ...

    def close(self) -> None: ...


class InProcessBackend:
    """An ``openenv.core.env_server.Environment`` instance driven directly."""

    def __init__(self, env: Any, action_cls: type[Any]) -> None:
        self.env = env
        self.action_cls = action_cls

    @staticmethod
    def resolve_action_class(env_class: type[Any], explicit: str | None = None) -> type[Any]:
        core = _require_openenv()
        base_action = core.env_server.Action
        if explicit:
            cls = _import_dotted(explicit)
            if not (inspect.isclass(cls) and issubclass(cls, base_action)):
                raise ValueError(f"{explicit} is not an openenv Action subclass")
            return cls
        try:
            hints = typing.get_type_hints(env_class.step)
        except Exception:
            hints = {}
        cand = hints.get("action")
        if inspect.isclass(cand) and issubclass(cand, base_action) and cand is not base_action:
            return cand
        for klass in env_class.__mro__:
            for base in getattr(klass, "__orig_bases__", ()):
                for arg in typing.get_args(base):
                    if inspect.isclass(arg) and issubclass(arg, base_action) and arg is not base_action:
                        return arg
        raise ValueError(
            f"cannot infer the Action class of {env_class.__name__}; set env.params.action_class"
        )

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> InProcessBackend:
        env_class = _import_dotted(str(params["env_class"]))
        action_cls = cls.resolve_action_class(env_class, params.get("action_class"))
        return cls(env_class(**dict(params.get("env_kwargs") or {})), action_cls)

    def _filtered(self, fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        sig = inspect.signature(fn)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    @staticmethod
    def _unpack(observation: Any) -> tuple[dict[str, Any], float | None, bool]:
        from openenv.core.env_server.serialization import serialize_observation

        payload = serialize_observation(observation)
        return payload["observation"], payload["reward"], bool(payload["done"])

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], float | None, bool]:
        return self._unpack(self.env.reset(**self._filtered(self.env.reset, kwargs)))

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float | None, bool]:
        from pydantic import ValidationError

        try:
            typed = self.action_cls.model_validate(action)
        except ValidationError as e:
            raise EnvironmentError_(f"action {action!r} rejected by {self.action_cls.__name__}: {e}") from e
        return self._unpack(self.env.step(typed))

    def close(self) -> None:
        self.env.close()


class ClientBackend:
    """A ``SyncEnvClient`` session (remote server or Docker container)."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self._connected = False

    @classmethod
    def from_url(cls, url: str, *, timeout_s: float) -> ClientBackend:
        core = _require_openenv()
        return cls(core.GenericEnvClient(base_url=url, connect_timeout_s=timeout_s, message_timeout_s=timeout_s).sync())

    @classmethod
    def from_image(cls, image: str, *, timeout_s: float, env_vars: Mapping[str, str] | None = None, port: int | None = None) -> ClientBackend:
        core = _require_openenv()
        from openenv.core.containers.runtime import LocalDockerProvider

        provider = LocalDockerProvider()
        base_url = provider.start_container(image, port=port, env_vars=dict(env_vars or {}))
        provider.wait_for_ready(base_url, timeout_s=timeout_s)
        client = core.GenericEnvClient(
            base_url=base_url, connect_timeout_s=timeout_s, message_timeout_s=timeout_s, provider=provider
        ).sync()
        return cls(client)

    def _ensure(self) -> None:
        if not self._connected:
            try:
                self.client.connect()
            except Exception as e:
                raise EnvironmentError_(f"cannot connect to OpenEnv server: {e}") from e
            self._connected = True

    @staticmethod
    def _unpack(result: Any) -> tuple[dict[str, Any], float | None, bool]:
        obs = result.observation
        if not isinstance(obs, dict):
            obs = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
        return obs, result.reward, bool(result.done)

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], float | None, bool]:
        self._ensure()
        try:
            return self._unpack(self.client.reset(**kwargs))
        except RuntimeError as e:
            raise EnvironmentError_(f"OpenEnv reset failed: {e}") from e

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float | None, bool]:
        self._ensure()
        try:
            return self._unpack(self.client.step(action))
        except RuntimeError as e:
            raise EnvironmentError_(f"OpenEnv step failed: {e}") from e

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # best effort: session may already be gone
            pass
        self._connected = False


def backend_from_params(params: Mapping[str, Any]) -> Backend:
    timeout_s = float(params.get("timeout_s", 60.0))
    if params.get("env_class"):
        return InProcessBackend.from_params(params)
    if params.get("url"):
        return ClientBackend.from_url(str(params["url"]), timeout_s=timeout_s)
    if params.get("image"):
        return ClientBackend.from_image(
            str(params["image"]), timeout_s=timeout_s, env_vars=params.get("env_vars"), port=params.get("port")
        )
    raise ValueError("env 'openenv' needs one of params.url, params.env_class or params.image")


class _Missing(dict):
    def __missing__(self, key: str) -> str:
        return ""


def observation_to_text(observation: Mapping[str, Any], template: str | None = None) -> str:
    if template:
        return template.format_map(_Missing(observation))
    strings = [v for v in observation.values() if isinstance(v, str)]
    if len(strings) == 1 and len(observation) == 1:
        return strings[0]
    return json.dumps(observation, ensure_ascii=False, sort_keys=True, default=str)


def parse_json_action(text: str) -> dict[str, Any] | None:
    candidates = _FENCE_RE.findall(text)
    if not candidates:
        m = _OBJECT_RE.search(text)
        candidates = [m.group(0)] if m else []
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


class OpenEnvEnv:
    """monorl Environment over an OpenEnv backend. One instance = one OpenEnv session."""

    tools: list[dict[str, Any]] = []

    def __init__(
        self,
        backend: Backend,
        *,
        max_turns: int = 1,
        action_mode: str = "text",
        action_field: str = "text",
        observation_template: str | None = None,
        system_prompt: str | None = None,
        invalid_action_penalty: float = 0.0,
        reset_kwargs: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        if action_mode not in ("text", "json"):
            raise ValueError(f"action_mode must be 'text' or 'json', got {action_mode!r}")
        self.backend = backend
        self.max_turns = max(1, int(max_turns))
        self.action_mode = action_mode
        self.action_field = action_field
        self.observation_template = observation_template
        self.system_prompt = system_prompt
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.reset_kwargs = dict(reset_kwargs or {})
        self.seed = seed
        self._turn = 0
        self.last_observation: dict[str, Any] = {}

    @property
    def turn(self) -> int:
        return self._turn

    def reset(self, sample: Sample) -> list[Message]:
        self._turn = 0
        kwargs: dict[str, Any] = {"episode_id": sample.id, **self.reset_kwargs}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        extra = sample.metadata.get("reset_kwargs")
        if isinstance(extra, Mapping):
            kwargs.update(extra)
        obs, _reward, _done = self.backend.reset(**kwargs)
        self.last_observation = obs
        messages: list[Message] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(sample.messages)
        text = observation_to_text(obs, self.observation_template)
        if text:
            messages.append({"role": "user", "content": text})
        return messages

    def build_action(self, completion: Completion) -> dict[str, Any] | None:
        if self.action_mode == "text":
            return {self.action_field: completion.text}
        return parse_json_action(completion.text)

    def step(self, action: Completion) -> StepResult:
        self._turn += 1
        payload = self.build_action(action)
        if payload is None:
            done = self._turn >= self.max_turns
            return StepResult(
                messages=[] if done else [{"role": "user", "content": "Invalid action: reply with a JSON object."}],
                reward=-self.invalid_action_penalty,
                done=done,
                info={"turn": self._turn, "invalid_action": True},
            )
        obs, reward, done = self.backend.step(payload)
        self.last_observation = obs
        done = done or self._turn >= self.max_turns
        text = observation_to_text(obs, self.observation_template)
        messages: list[Message] = [] if done or not text else [{"role": "user", "content": text}]
        return StepResult(
            messages=messages,
            reward=float(reward) if reward is not None else 0.0,
            done=done,
            info={"turn": self._turn, "observation": obs, "action": payload},
        )

    def close(self) -> None:
        self.backend.close()


def make_openenv_env(params: Mapping[str, Any], *, max_turns: int, system_prompt: str | None = None, seed: int | None = None) -> OpenEnvEnv:
    return OpenEnvEnv(
        backend_from_params(params),
        max_turns=max_turns,
        action_mode=str(params.get("action_mode", "text")),
        action_field=str(params.get("action_field", "text")),
        observation_template=params.get("observation_template") or params.get("observation_to_text"),
        system_prompt=params.get("system_prompt", system_prompt),
        invalid_action_penalty=float(params.get("invalid_action_penalty", 0.0)),
        reset_kwargs=params.get("reset_kwargs"),
        seed=params.get("seed", seed),
    )


__all__ = [
    "Backend",
    "ClientBackend",
    "InProcessBackend",
    "OpenEnvEnv",
    "backend_from_params",
    "make_openenv_env",
    "observation_to_text",
    "parse_json_action",
]
