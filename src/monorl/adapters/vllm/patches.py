"""Versioned monkeypatches for vLLM (``>=0.33,<0.34``).

Each patch is a plain ``build(original) -> replacement`` so it can be unit-tested against fakes
without the patch engine. ``PATCHES`` is registered with :mod:`monorl.patching` when
:mod:`monorl.adapters.vllm` is imported; :func:`monorl.patching.apply` applies it (the colocated
engine and the server do so before touching vLLM).

Targets (vLLM v0.33.0):

* ``vllm.entrypoints.llm.LLM.sleep`` / ``wake_up`` — raise inside the engine core when
  ``enable_sleep_mode`` is off (``vllm/v1/worker/gpu_worker.py`` ``sleep_mode_backend``) and
  cannot be enabled on non-CUDA/ROCm platforms (``vllm/config/model.py``). The guard turns both
  into logged no-ops in that case so colocated loops stay portable.
* ``vllm.outputs.CompletionOutput`` (``@dataclass``) gets a ``sampled_logprobs`` property that
  returns the sampled-token logprob per position as ``list[float]``.
* ``vllm.entrypoints.launchers.app.register_api_routers`` — ``build_app`` looks this name up in
  its module globals, so wrapping it lets the monorl server add its routes to vLLM's
  OpenAI-compatible app without forking the launcher.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any

from monorl.adapters.vllm.common import VLLM_VERSIONS, sampled_logprobs
from monorl.patching import Patch, PatchSet

log = logging.getLogger(__name__)


def _sleep_mode_enabled(llm: Any) -> bool:
    return bool(llm.llm_engine.vllm_config.model_config.enable_sleep_mode)


def build_sleep_guard(original: Any) -> Any:
    @functools.wraps(original)
    def sleep(self: Any, level: int = 1, mode: str = "abort") -> Any:
        if not _sleep_mode_enabled(self):
            log.warning(
                "vllm.LLM.sleep(level=%s) ignored: engine was created without enable_sleep_mode", level
            )
            return None
        return original(self, level=level, mode=mode)

    return sleep


def build_wake_up_guard(original: Any) -> Any:
    @functools.wraps(original)
    def wake_up(self: Any, tags: list[str] | None = None) -> bool:
        if not _sleep_mode_enabled(self):
            return True  # "fully awake": nothing was ever released
        return original(self, tags)

    return wake_up


def build_sampled_logprobs(original: Any) -> property:
    return property(sampled_logprobs, doc="Sampled-token logprob per generated token (monorl patch).")


def build_monorl_routes(original: Any) -> Any:
    @functools.wraps(original)
    def register_api_routers(args: Any, app: Any, supported_tasks: Any, model_config: Any = None) -> Any:
        result = original(args, app, supported_tasks, model_config)
        from monorl.adapters.vllm.server import attach_router

        attach_router(app)
        return result

    return register_api_routers


def _has_params(fn: Any, *names: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return all(n in params for n in names)


PATCHES = PatchSet(
    framework="vllm",
    patches=[
        Patch(
            id="vllm.llm.sleep_guard",
            target="vllm.entrypoints.llm.LLM.sleep",
            versions=VLLM_VERSIONS,
            mode="wrap",
            build=build_sleep_guard,
            verify=lambda orig: _has_params(orig, "self", "level", "mode"),
            description="LLM.sleep becomes a logged no-op when the engine has no sleep mode (non-CUDA or disabled).",
        ),
        Patch(
            id="vllm.llm.wake_up_guard",
            target="vllm.entrypoints.llm.LLM.wake_up",
            versions=VLLM_VERSIONS,
            mode="wrap",
            build=build_wake_up_guard,
            verify=lambda orig: _has_params(orig, "self", "tags"),
            description="LLM.wake_up returns True without touching the engine when sleep mode is off.",
        ),
        Patch(
            id="vllm.outputs.sampled_logprobs",
            target="vllm.outputs.CompletionOutput.sampled_logprobs",
            versions=VLLM_VERSIONS,
            mode="add",
            build=build_sampled_logprobs,
            description="CompletionOutput.sampled_logprobs: per-token logprob of the sampled token as floats.",
        ),
        Patch(
            id="vllm.server.monorl_routes",
            target="vllm.entrypoints.launchers.app.register_api_routers",
            versions=VLLM_VERSIONS,
            mode="wrap",
            build=build_monorl_routes,
            verify=lambda orig: _has_params(orig, "args", "app", "supported_tasks"),
            description="Mount monorl's /update_named_param and /load_weights_from_path on vLLM's API server.",
        ),
    ],
)
