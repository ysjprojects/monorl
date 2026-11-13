"""Fake ``vllm`` module tree mirroring the v0.33.0 API surface the adapter touches.

Shapes are copied from the upstream files cited next to each class; nothing here imports vLLM.
``conftest.py`` installs the tree in ``sys.modules`` per test.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import itertools
import queue
import select
import socket
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
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
    include_stop_str_in_output: bool = False
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
    """The target. It records every pair it is given: an MTP checkpoint key among them means the
    adapter routed a drafter weight into the policy."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.loaded: list[tuple[str, torch.Tensor]] = []
        self.lm_head: torch.Tensor | None = None  # the output head's current weight

    def load_weights(self, weights: Any) -> set[str]:
        names: set[str] = set()
        for name, tensor in weights:
            self.loaded.append((name, tensor.clone()))
            if name == "lm_head.weight":
                self.lm_head = tensor.clone()
            names.add(name)
        return names


# -- vllm/model_executor/model_loader/mtp_validation.py ------------------------------------------
_mtp_completeness_check_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "mtp_completeness_check_enabled", default=True
)


def is_mtp_completeness_check_enabled() -> bool:
    return _mtp_completeness_check_enabled.get()


@contextlib.contextmanager
def disable_mtp_completeness_check():
    token = _mtp_completeness_check_enabled.set(False)
    try:
        yield
    finally:
        _mtp_completeness_check_enabled.reset(token)


#: Checkpoint keys of the fake drafter's MTP layer (Mellum's layout).
MTP_KEYS = frozenset({"mtp.fc.weight", "mtp.layers.0.mlp.experts.3.down_proj.weight"})


class FakeDraftModel(torch.nn.Module):
    """The MTP drafter of ``target`` (``load_eagle_model``,
    vllm/v1/worker/gpu/spec_decode/eagle/utils.py): it owns the MTP layers and uses the target's
    output head, or a quantized copy of it with ``draft_lm_head_quantization``.

    ``load_weights`` behaves like the MTP loaders (vllm/model_executor/models/deepseek_mtp.py:304):
    names it does not own are skipped, and a call that brings no MTP layer raises unless the
    completeness check is disabled. A load while the quantized copy is in place is an error: it
    would write checkpoint weights into the quantized tensor.
    """

    def __init__(self, target: FakeModel, quantization: str | None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.target = [target]  # in a list: the target is not a submodule of the drafter
        self.quantization = quantization
        self.source_head_shared = quantization is None
        self.quantized_head: torch.Tensor | None = None  # what the quantized copy was derived from
        self.loaded: list[tuple[str, torch.Tensor]] = []
        self.offered: list[str] = []

    def load_weights(self, weights: Any) -> set[str]:
        if not self.source_head_shared:
            raise RuntimeError("draft load_weights called while the quantized lm_head copy is in place")
        names: set[str] = set()
        for name, tensor in weights:
            self.offered.append(name)
            if name in MTP_KEYS:
                self.loaded.append((name, tensor.clone()))
                names.add(name)
        if not names and is_mtp_completeness_check_enabled():
            raise ValueError("MTP speculative decoding layer weights missing from checkpoint.")
        return names


# -- vllm/model_executor/layers/quantization/online/lm_head.py:157-206 ---------------------------
def refresh_quantized_lm_heads(model: Any) -> None:
    if getattr(model, "quantization", None) is not None:
        head = model.target[0].lm_head
        model.quantized_head = None if head is None else head.clone()


def share_source_lm_heads(model: Any) -> None:
    model.source_head_shared = True


def restore_quantized_lm_heads(model: Any, refresh: bool = True) -> None:
    model.source_head_shared = model.quantization is None
    if refresh:
        refresh_quantized_lm_heads(model)


# -- vllm/config/speculative.py:407 SpeculativeConfig (the fields the adapter reads or sets) -----
SPECULATIVE_CONFIG_KEYS = frozenset(
    {
        "method",
        "model",
        "num_speculative_tokens",
        "draft_lm_head_quantization",
        "rejection_sample_method",
        "draft_sample_method",
        "draft_tensor_parallel_size",
        "enforce_eager",
    }
)


@dataclasses.dataclass
class FakeSpeculativeConfig:
    method: str = "mtp"
    num_speculative_tokens: int = 1
    model: str | None = None
    draft_lm_head_quantization: str | None = None
    rejection_sample_method: str = "standard"
    draft_sample_method: str = "greedy"
    draft_tensor_parallel_size: int | None = None
    enforce_eager: bool | None = None
    draft_model_config: Any = None

    def __post_init__(self) -> None:
        if self.method.endswith("_mtp"):  # speculative.py:1246, the per-family names are aliases
            self.method = "mtp"

    def apply_draft_overrides(self, vllm_config: Any) -> Any:  # speculative.py:1917
        return vllm_config


@dataclasses.dataclass
class _ModelConfig:
    enable_sleep_mode: bool = False
    model: str = ""


@dataclasses.dataclass
class _VllmConfig:
    model_config: _ModelConfig = dataclasses.field(default_factory=_ModelConfig)
    speculative_config: FakeSpeculativeConfig | None = None
    load_config: str = "auto"


def reload_weights(
    vllm_config: Any, model: Any, weights_iterator: Any, weights_path: str | None, is_checkpoint_format: bool
) -> None:
    """vllm/model_executor/model_loader/reload/inplace.py:32 — loads the safetensors file of
    ``weights_path`` into ``model`` through its ``load_weights``."""
    from safetensors.torch import load_file

    assert weights_iterator is None and weights_path is not None and is_checkpoint_format
    vllm_config.model_config.model = weights_path
    model.load_weights(load_file(f"{weights_path}/model.safetensors").items())


class FakeModelRunner:
    def __init__(self, speculative_config: FakeSpeculativeConfig | None) -> None:
        self.model = FakeModel()
        self.draft_model: FakeDraftModel | None = None
        if speculative_config is not None:
            self.draft_model = FakeDraftModel(self.model, speculative_config.draft_lm_head_quantization)


class FakeGPUWorker:
    """Stands in for ``GPUWorker``; the extension is mixed in exactly like
    ``WorkerWrapperBase.init_worker`` does (``worker_class.__bases__ += (ext,)``)."""

    def __init__(self, rank: int, speculative_config: dict[str, Any] | None = None) -> None:
        self.rank = rank
        speculative = None
        if speculative_config is not None:
            speculative = FakeSpeculativeConfig(**speculative_config, draft_model_config=_ModelConfig())
        self.vllm_config = _VllmConfig(speculative_config=speculative)
        self.model_runner = FakeModelRunner(speculative)
        self.reload_calls: list[dict[str, Any]] = []

    def get_draft_model(self) -> FakeDraftModel | None:  # gpu_worker.py:1164
        return self.model_runner.draft_model

    def reload_weights(self, *args: Any, **kwargs: Any) -> None:  # gpu_worker.py:636, the target only
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


class FakeLLMEngine:
    """``LLMEngine`` (vllm/v1/engine/llm_engine.py:54): the request queue behind ``LLM.enqueue``
    with ``has_unfinished_requests`` (:203), ``step`` (:319) and ``abort_request`` (:224).

    ``step`` finishes one request per call, newest first (upstream promises no order). While
    ``hold`` is set it finishes none, like a step in the middle of a long generation, and sets
    ``stepping``.
    """

    def __init__(self, enable_sleep_mode: bool) -> None:
        self.vllm_config = _VllmConfig(_ModelConfig(enable_sleep_mode))
        self.engine_core = FakeEngineCore()
        self.unfinished: dict[str, RequestOutput] = {}  # internal request id -> its final output
        self.aborted: list[str] = []  # internal ids removed by abort_request
        self.hold = False
        self.stepping = threading.Event()

    def has_unfinished_requests(self) -> bool:
        return bool(self.unfinished)

    def step(self) -> list[RequestOutput]:
        if self.hold:
            self.stepping.set()
            time.sleep(0.001)
            return []
        return [self.unfinished.popitem()[1]]

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        # OutputProcessor.abort_requests (vllm/v1/engine/output_processor.py:599): ``internal``
        # ids are the ones add_request returned; otherwise they are matched as external ids.
        for request_id in request_ids:
            matched = (
                [request_id]
                if internal
                else [k for k, out in self.unfinished.items() if out.request_id == request_id]
            )
            for key in matched:
                if self.unfinished.pop(key, None) is not None:
                    self.aborted.append(key)


class FakeLLM:
    """``vllm.LLM`` (vllm/entrypoints/llm.py:62): records constructor kwargs and every call.

    ``enqueue`` (:483) is deterministic: completion ``i`` of prompt ``p`` will emit ``script``
    tokens (``[p, i, 7]``) with logprob ``-(k+1)/10`` and finish reason from ``finish_reasons``.
    Like upstream it returns the engine's internal request ids (``<counter>-<suffix>``,
    ``InputProcessor.assign_request_id``) while ``RequestOutput.request_id`` is the counter value.
    """

    instances: list[FakeLLM] = []
    finish_reasons: list[str | None] = ["stop"]
    extra_logprob_entries: int = 0  # emulate logprobs=N>0 (top-N + sampled)

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs
        tp = int(kwargs.get("tensor_parallel_size", 1))
        speculative = kwargs.get("speculative_config")
        if speculative is not None:  # EngineArgs.create_speculative_config, arg_utils.py:2126
            speculative = {key.replace("-", "_"): value for key, value in speculative.items()}
        if speculative is not None and (unknown := set(speculative) - SPECULATIVE_CONFIG_KEYS):
            raise ValueError(f"SpeculativeConfig: extra inputs are not permitted: {sorted(unknown)}")
        self.workers = [_make_worker(r, speculative) for r in range(tp)]
        self.log_stats = not kwargs.get("disable_log_stats", True)  # llm.py:225
        # SpecDecodingProm counters (vllm/v1/spec_decode/metrics.py:229), registered when drafting.
        self.spec_decode = (
            None if speculative is None else dict.fromkeys(("drafts", "draft_tokens", "accepted_tokens"), 0)
        )
        self.llm_engine = FakeLLMEngine(bool(kwargs.get("enable_sleep_mode", False)))
        self.request_counter = itertools.count()
        self.calls: list[tuple[str, Any]] = []
        FakeLLM.instances.append(self)

    def enqueue(
        self,
        prompts: Any,
        sampling_params: Any = None,
        lora_request: Any = None,
        priority: Any = None,
        use_tqdm: Any = True,
        tokenization_kwargs: Any = None,
        mm_processor_kwargs: Any = None,
    ) -> list[str]:
        self.calls.append(
            ("enqueue", {"prompts": prompts, "sampling_params": sampling_params, "use_tqdm": use_tqdm})
        )
        request_ids = []
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
                if self.spec_decode is not None:
                    # Two verification passes per completion; one draft token of each pass is accepted.
                    drafted = self.workers[0].vllm_config.speculative_config.num_speculative_tokens
                    self.spec_decode["drafts"] += 2
                    self.spec_decode["draft_tokens"] += 2 * drafted
                    self.spec_decode["accepted_tokens"] += 2
            comps.reverse()  # upstream order is not guaranteed; the adapter must sort by index
            external_id = str(next(self.request_counter))
            request_ids.append(f"{external_id}-{int(external_id):08x}")
            self.llm_engine.unfinished[request_ids[-1]] = RequestOutput(
                external_id, None, list(prompt["prompt_token_ids"]), None, comps, True
            )
        return request_ids

    def collective_rpc(
        self, method: str, timeout: float | None = None, args: tuple = (), kwargs: dict | None = None
    ) -> list[Any]:
        self.calls.append(("collective_rpc", {"method": method, "args": args, "kwargs": kwargs}))
        return [getattr(w, method)(*args, **(kwargs or {})) for w in self.workers]

    def get_metrics(self) -> list[Any]:  # llm.py:890, vllm/v1/metrics/reader.py:70
        assert self.log_stats, "Stat logging disabled"  # llm_engine.py:447
        labels = {"model_name": self.model, "engine": "0"}
        metrics: list[Any] = [Gauge("vllm:num_requests_running", labels, 0.0)]
        if self.spec_decode is not None:
            metrics += [Counter(f"vllm:spec_decode_num_{k}", labels, v) for k, v in self.spec_decode.items()]
            metrics.append(Vector("vllm:spec_decode_num_accepted_tokens_per_pos", labels, [0]))
        return metrics

    def metrics_text(self) -> str:
        """The Prometheus exposition of ``get_metrics`` (``GET /metrics`` of the API server)."""
        lines = ["# TYPE vllm:num_requests_running gauge", 'vllm:num_requests_running{engine="0"} 0.0']
        for metric in self.get_metrics():
            if isinstance(metric, Counter):
                lines.append(f"# TYPE {metric.name}_total counter")
                lines.append(
                    f'{metric.name}_total{{engine="0",model_name="{self.model}"}} {metric.value:.1f}'
                )
                lines.append(f'{metric.name}_created{{engine="0",model_name="{self.model}"}} 1.7e+09')
        return "\n".join(lines) + "\n"

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


def _make_worker(rank: int, speculative_config: dict[str, Any] | None = None) -> Any:
    cls = types.new_class(f"InjectedWorker{rank}", (FakeGPUWorker, MonorlWorkerExtension))
    return cls(rank, speculative_config)


# -- vllm/v1/metrics/reader.py:12-46 ---------------------------------------------------------------
@dataclass
class Counter:
    name: str
    labels: dict[str, str]
    value: int


@dataclass
class Vector:
    name: str
    labels: dict[str, str]
    values: list[int]


@dataclass
class Gauge:
    name: str
    labels: dict[str, str]
    value: float


class FakePlatform:
    sleep_mode_available = True

    def is_sleep_mode_available(self) -> bool:
        return type(self).sleep_mode_available


def register_api_routers(args: Any, app: Any, supported_tasks: Any, model_config: Any = None) -> None:
    """vllm/entrypoints/launchers/api_server/routers.py:12, bound by name in launchers/app.py:15."""
    app.state.registered = (args, supported_tasks, model_config)


class LoopbackServer:
    """Serves an ``httpx.MockTransport``-style ``handler`` on 127.0.0.1, to observe what a real
    server sees when the client drops a request.

    While ``stall`` is set a POST is read and left unanswered, as during a long generation: its
    path goes on ``stalled`` at once and on ``disconnects`` when the client closes the connection
    — the ``http.disconnect`` vLLM's ``with_cancellation`` aborts the generation on.
    """

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.stall = False
        self.stalled: queue.Queue[str] = queue.Queue()
        self.disconnects: queue.Queue[str] = queue.Queue()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("content-length", 0)))
                if owner.stall:
                    owner.stalled.put(self.path)
                    self.close_connection = True
                    readable, _, _ = select.select([self.connection], [], [], 10)
                    try:  # end of stream (or a reset) while no request is pending: the client left
                        gone = bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
                    except ConnectionError:
                        gone = True
                    if gone:
                        owner.disconnects.put(self.path)
                    return
                url = f"http://{self.headers['host']}{self.path}"
                response = handler(httpx.Request("POST", url, headers=dict(self.headers), content=body))
                self.send_response(response.status_code)
                self.send_header("content-type", response.headers.get("content-type", "text/plain"))
                self.send_header("content-length", str(len(response.content)))
                self.end_headers()
                self.wfile.write(response.content)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, args=(0.01,), daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._thread.join()
        self._httpd.server_close()


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
        "vllm.config": mod(
            "vllm.config",
            replace=dataclasses.replace,
            set_current_vllm_config=lambda vllm_config: contextlib.nullcontext(),
        ),
        "vllm.model_executor": mod("vllm.model_executor", __path__=[]),
        "vllm.model_executor.layers": mod("vllm.model_executor.layers", __path__=[]),
        "vllm.model_executor.layers.quantization": mod(
            "vllm.model_executor.layers.quantization", __path__=[]
        ),
        "vllm.model_executor.layers.quantization.online": mod(
            "vllm.model_executor.layers.quantization.online", __path__=[]
        ),
        "vllm.model_executor.layers.quantization.online.lm_head": mod(
            "vllm.model_executor.layers.quantization.online.lm_head",
            refresh_quantized_lm_heads=refresh_quantized_lm_heads,
            share_source_lm_heads=share_source_lm_heads,
            restore_quantized_lm_heads=restore_quantized_lm_heads,
        ),
        "vllm.model_executor.model_loader": mod("vllm.model_executor.model_loader", __path__=[]),
        "vllm.model_executor.model_loader.mtp_validation": mod(
            "vllm.model_executor.model_loader.mtp_validation",
            disable_mtp_completeness_check=disable_mtp_completeness_check,
            is_mtp_completeness_check_enabled=is_mtp_completeness_check_enabled,
        ),
        "vllm.model_executor.model_loader.utils": mod(
            "vllm.model_executor.model_loader.utils",
            get_draft_load_config=lambda vllm_config: vllm_config.load_config,
        ),
        "vllm.model_executor.model_loader.reload": mod(
            "vllm.model_executor.model_loader.reload", __path__=[]
        ),
        "vllm.model_executor.model_loader.reload.inplace": mod(
            "vllm.model_executor.model_loader.reload.inplace", reload_weights=reload_weights
        ),
        "vllm.v1": mod("vllm.v1", __path__=[]),
        "vllm.v1.worker": mod("vllm.v1.worker", __path__=[]),
        "vllm.v1.worker.gpu": mod("vllm.v1.worker.gpu", __path__=[]),
        "vllm.v1.worker.gpu.spec_decode": mod("vllm.v1.worker.gpu.spec_decode", __path__=[]),
        "vllm.v1.worker.gpu.spec_decode.utils": mod(
            "vllm.v1.worker.gpu.spec_decode.utils",
            get_pp_safe_draft_load_config=lambda load_config: load_config,
        ),
    }
    for name, m in tree.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(tree[parent], child, m)
    return tree
