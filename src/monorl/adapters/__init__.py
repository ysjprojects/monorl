"""Adapter providers: one subpackage per upstream framework.

Each subpackage exposes ``register(registry)`` and (if it patches upstream) a ``PATCHES``
:class:`~monorl.patching.PatchSet`. Importing this package imports nothing heavy; providers are
loaded lazily by the registry when a component name they own is requested.

Component names owned by each provider (``<provider>`` or ``<provider>:<variant>``)::

    vllm      rollout_engine: vllm
    sglang    rollout_engine: sglang
    trl       pipeline_backend: trl
    unsloth   trainer: unsloth
    verl      pipeline_backend: verl
    nemo_rl   pipeline_backend: nemo_rl
    openenv   env: openenv            (env.params.url or image)
    hermes    env: hermes             (Hermes agent loop as a multi-turn env)
"""

from __future__ import annotations

PROVIDERS: dict[str, str] = {
    "vllm": "monorl.adapters.vllm",
    "sglang": "monorl.adapters.sglang",
    "trl": "monorl.adapters.trl",
    "unsloth": "monorl.adapters.unsloth",
    "verl": "monorl.adapters.verl",
    "nemo_rl": "monorl.adapters.nemo_rl",
    "openenv": "monorl.adapters.openenv",
    "hermes": "monorl.adapters.hermes",
}


def provider_for_name(name: str) -> str | None:
    key = name.split(":", 1)[0]
    return PROVIDERS.get(key)


def all_providers() -> dict[str, str]:
    return dict(PROVIDERS)
