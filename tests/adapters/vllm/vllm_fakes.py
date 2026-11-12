"""Fake ``vllm`` module tree mirroring the v0.33.0 API surface the adapter touches.

Shapes are copied from the upstream files cited next to each class; nothing here imports vLLM.
``conftest.py`` installs the tree in ``sys.modules`` per test.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any

import torch

from monorl.adapters.vllm.worker import MonorlWorkerExtension


# -- vllm.logprobs.Logprob (vllm/logprobs.py:13) ---------------------------------------------
@dataclass
class Logprob:
    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


# -- vllm.outputs.CompletionOutput / RequestOutput (vllm/outputs.py:58-99, 137-176) --------------
@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None
    logprobs: list[dict[int, Logprob]] | None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


class RequestOutput:
    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: Any,
        outputs: list[CompletionOutput],
        finished: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished


# -- vllm.sampling_params.SamplingParams (vllm/sampling_params.py:271-342) ----------------------
@dataclass
class SamplingParams:
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    max_tokens: int | None = 16
    min_tokens: int = 0
    logprobs: int | None = None

    def __post_init__(self) -> None:
        # vllm/sampling_params.py:681 — "top_k must be 0 (disable), or at least 1"
        if self.top_k < -1:
            raise ValueError(f"top_k must be 0 (disable), or at least 1, got {self.top_k}.")


# -- vllm.inputs.TokensPrompt (vllm/inputs/llm.py:111) -------------------------------------------
class TokensPrompt(dict):
    def __init__(self, *, prompt_token_ids: list[int], **extra: Any) -> None:
        super().__init__(prompt_token_ids=prompt_token_ids, **extra)


# -- worker side: vllm/v1/worker/gpu_worker.py GPUWorker (methods that matter to us) ------------
class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.loaded: list[tuple[str, torch.Tensor]] = []

    def load_weights(self, weights: Any) -> set[str]:
        names: set[str] = set()
        for name, tensor in weights:
            self.loaded.append((name, tensor.clone()))
            names.add(name)
        return names


class FakeModelRunner:
    def __init__(self) -> None:
        self.model = FakeModel()


class FakeGPUWorker:
    """Stands in for ``GPUWorker``; the extension is mixed in exactly like
    ``WorkerWrapperBase.init_worker`` does (``worker_class.__bases__ += (ext,)``)."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.model_runner = FakeModelRunner()
        self.reload_calls: list[dict[str, Any]] = []

    def reload_weights(self, *args: Any, **kwargs: Any) -> None:  # gpu_worker.py:636
        self.reload_calls.append({"args": args, **kwargs})


#: Public attribute names of ``GPUWorker``/``WorkerBase`` at v0.33.0 (vllm/v1/worker/gpu_worker.py,
#: vllm/v1/worker/worker_base.py). ``init_worker`` asserts the extension defines none of them.
GPU_WORKER_ATTRS = frozenset(
    {
        "sleep_mode_backend",
        "sleep",
        "wake_up",
        "discard",
        "checkpoint_prepare",
        "checkpoint_restore",
        "compute_weight_checksums",
        "reset_weights",
        "init_device",
        "handle_ft_command",
        "load_model",
        "update_config",
        "reload_weights",
        "determine_available_memory",
        "get_kv_connector_handshake_metadata",
        "get_kv_cache_spec",
        "update_max_model_len",
        "initialize_from_config",
        "get_simple_cpu_offload_num_cpu_blocks",
        "compile_or_warm_up_model",
        "reset_mm_cache",
        "reset_encoder_cache",
        "get_model",
        "get_draft_model",
        "supports_draft_weight_updates",
        "get_supported_tasks",
        "get_compilation_match_table",
        "get_encoder_timing_stats",
        "annotate_profile",
        "sample_tokens",
        "execute_model",
        "take_draft_token_ids",
        "profile",
        "randomize_dummy_inputs",
        "execute_dummy_batch",
        "add_lora",
        "remove_lora",
        "list_loras",
        "pin_lora",
        "check_health",
        "save_sharded_state",
        "save_tensorized_model",
        "init_weight_transfer_engine",
        "shutdown_weight_transfer_engine",
        "start_weight_update",
        "start_draft_weight_update",
        "update_weights",
        "finish_weight_update",
        "shutdown",
        "elastic_ep_execute",
        "get_supported_kv_cache_layouts",
        "set_kv_cache_layout",
        "synchronize_device",
        "release_late_interaction_query_cache",
        "apply_model",
        "get_model_inspection",
        "get_cache_block_size_bytes",
        "vocab_size",
        "model_runner",
        "rank",
        "device",
        "vllm_config",
        "model_config",
        "weight_transfer_engine",
    }
)


class FakeEngineCore:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self, timeout: float | None = None) -> None:
        self.shutdown_calls += 1


@dataclass
class _ModelConfig:
    enable_sleep_mode: bool = False


@dataclass
class _VllmConfig:
    model_config: _ModelConfig = field(default_factory=_ModelConfig)


class FakeLLMEngine:
    def __init__(self, enable_sleep_mode: bool) -> None:
        self.vllm_config = _VllmConfig(_ModelConfig(enable_sleep_mode))
        self.engine_core = FakeEngineCore()


class FakeLLM:
    """``vllm.LLM`` (vllm/entrypoints/llm.py:62): records constructor kwargs and every call.

    ``generate`` is deterministic: completion ``i`` of prompt ``p`` emits ``script`` tokens
    (``[p, i, 7]``) with logprob ``-(k+1)/10`` and finish reason from ``finish_reasons``.
    """

    instances: list[FakeLLM] = []
    finish_reasons: list[str | None] = ["stop"]
    extra_logprob_entries: int = 0  # emulate logprobs=N>0 (top-N + sampled)

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs
        tp = int(kwargs.get("tensor_parallel_size", 1))
        self.workers = [_make_worker(r) for r in range(tp)]
        self.llm_engine = FakeLLMEngine(bool(kwargs.get("enable_sleep_mode", False)))
        self.calls: list[tuple[str, Any]] = []
        FakeLLM.instances.append(self)

    def generate(
        self, prompts: Any, sampling_params: Any = None, *, use_tqdm: Any = True, **kw: Any
    ) -> list[Any]:
        self.calls.append(
            ("generate", {"prompts": prompts, "sampling_params": sampling_params, "use_tqdm": use_tqdm})
        )
        outs = []
        for p, prompt in enumerate(prompts):
            comps = []
            for i in range(sampling_params.n):
                token_ids = [p, i, 7]
                lps = None
                if sampling_params.logprobs is not None:
                    lps = []
                    for k, tid in enumerate(token_ids):
                        pos = {tid: Logprob(-(k + 1) / 10, rank=1, decoded_token=f"t{tid}")}
                        for j in range(FakeLLM.extra_logprob_entries):
                            pos[1000 + j] = Logprob(-5.0 - j, rank=2 + j)
                        lps.append(pos)
                fr = FakeLLM.finish_reasons[(p * sampling_params.n + i) % len(FakeLLM.finish_reasons)]
                comps.append(CompletionOutput(i, f"text-{p}-{i}", token_ids, None, lps, fr))
            comps.reverse()  # upstream order is not guaranteed; the adapter must sort by index
            outs.append(RequestOutput(f"req-{p}", None, list(prompt["prompt_token_ids"]), None, comps, True))
        return outs

    def collective_rpc(
        self, method: str, timeout: float | None = None, args: tuple = (), kwargs: dict | None = None
    ) -> list[Any]:
        self.calls.append(("collective_rpc", {"method": method, "args": args, "kwargs": kwargs}))
        return [getattr(w, method)(*args, **(kwargs or {})) for w in self.workers]

    def sleep(self, level: int = 1, mode: str = "abort") -> None:
        self.calls.append(("sleep", {"level": level, "mode": mode}))

    def wake_up(self, tags: list[str] | None = None) -> bool:
        self.calls.append(("wake_up", {"tags": tags}))
        return True

    def reset_prefix_cache(self, reset_running_requests: bool = False, reset_connector: bool = False) -> bool:
        self.calls.append(("reset_prefix_cache", {}))
        return True

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _make_worker(rank: int) -> Any:
    cls = types.new_class(f"InjectedWorker{rank}", (FakeGPUWorker, MonorlWorkerExtension))
    return cls(rank)


class FakePlatform:
    sleep_mode_available = True

    def is_sleep_mode_available(self) -> bool:
        return type(self).sleep_mode_available


def register_api_routers(args: Any, app: Any, supported_tasks: Any, model_config: Any = None) -> None:
    """vllm/entrypoints/launchers/api_server/routers.py:12, bound by name in launchers/app.py:15."""
    app.state.registered = (args, supported_tasks, model_config)


def build_fake_vllm() -> dict[str, types.ModuleType]:
    def mod(name: str, **attrs: Any) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    vllm = mod(
        "vllm", __version__="0.33.0", LLM=FakeLLM, SamplingParams=SamplingParams, TokensPrompt=TokensPrompt
    )
    vllm.__path__ = []  # mark as package so submodule imports resolve via sys.modules
    tree = {
        "vllm": vllm,
        "vllm.entrypoints": mod("vllm.entrypoints", __path__=[]),
        "vllm.entrypoints.llm": mod("vllm.entrypoints.llm", LLM=FakeLLM),
        "vllm.entrypoints.launchers": mod("vllm.entrypoints.launchers", __path__=[]),
        "vllm.entrypoints.launchers.app": mod(
            "vllm.entrypoints.launchers.app", register_api_routers=register_api_routers
        ),
        "vllm.inputs": mod("vllm.inputs", TokensPrompt=TokensPrompt),
        "vllm.outputs": mod("vllm.outputs", CompletionOutput=CompletionOutput, RequestOutput=RequestOutput),
        "vllm.logprobs": mod("vllm.logprobs", Logprob=Logprob),
        "vllm.sampling_params": mod("vllm.sampling_params", SamplingParams=SamplingParams),
        "vllm.platforms": mod("vllm.platforms", current_platform=FakePlatform()),
    }
    for name, m in tree.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(tree[parent], child, m)
    return tree
