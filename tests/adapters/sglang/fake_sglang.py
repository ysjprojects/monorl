"""A fake ``sglang`` module tree mirroring the SGLang v0.5.23 API surface the adapter uses.

Mirrored upstream files (tag v0.5.23):

* ``python/sglang/version.py`` — ``sglang.__version__``.
* ``python/sglang/srt/entrypoints/engine.py`` — ``Engine(**ServerArgs kwargs)``, ``generate(...)``,
  ``update_weights_from_tensor(named_tensors, load_format, flush_cache)`` (serialises once per TP
  rank with ``MultiprocessingSerializer`` — ``_serialize_tensors_per_rank``),
  ``update_weights_from_disk(model_path, load_format)``, ``release_memory_occupation(tags)``,
  ``resume_memory_occupation(tags)``, ``flush_cache()``, ``shutdown()``.
* ``python/sglang/srt/utils/common.py`` (re-exported by ``sglang/srt/utils/__init__.py``) —
  ``MultiprocessingSerializer.serialize(obj, output_str)`` / ``deserialize(data)``.
* ``python/sglang/srt/managers/tokenizer_manager.py`` — ``TokenizerManager.detokenize_logprob_tokens``
  and the ``{"text", "output_ids", "meta_info"}`` output shape of ``_handle_batch_output``.
* ``python/sglang/srt/sampling/sampling_params.py`` — ``SamplingParams`` keyword names.
"""

from __future__ import annotations

import base64
import io
import pickle
import sys
import types
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Any

import torch.multiprocessing  # noqa: F401  (registers tensor reducers, as sglang does via torch)

SGLANG_FAKE_VERSION = "0.5.23"


# -- sglang.srt.utils ---------------------------------------------------------------------------
class MultiprocessingSerializer:
    """``sglang/srt/utils/common.py::MultiprocessingSerializer`` (ForkingPickler + optional base64)."""

    @staticmethod
    def serialize(obj: Any, output_str: bool = False) -> bytes | str:
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()
        if output_str:
            return base64.b64encode(output).decode("utf-8")
        return output

    @staticmethod
    def deserialize(data: bytes | str) -> Any:
        if isinstance(data, str):
            data = base64.b64decode(data, validate=True)
        return pickle.loads(data)


# -- sglang.srt.sampling.sampling_params --------------------------------------------------------
@dataclass
class SamplingParams:
    """Keyword names of ``sglang/srt/sampling/sampling_params.py::SamplingParams`` (v0.5.23)."""

    max_new_tokens: int | None = 128
    stop: str | list[str] | None = None
    stop_token_ids: set[int] | None = None
    stop_regex: str | list[str] | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 1 << 30
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    n: int = 1
    beam_width: int | None = None
    json_schema: str | None = None
    regex: str | None = None
    ebnf: str | None = None
    structural_tag: str | None = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    no_stop_trim: bool = False
    stream_interval: int | None = None
    logit_bias: dict[str, float] | None = None
    sampling_seed: int | None = None
    custom_params: dict[str, Any] | None = None


# -- sglang.srt.managers.tokenizer_manager ---------------------------------------------------------
class TokenizerManager:
    def detokenize_logprob_tokens(self, token_logprobs_val, token_logprobs_idx, decode_to_text):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx, strict=True)
            ]
        return [
            (logprob, token_id, f"<{token_id}>")
            for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx, strict=True)
        ]


# -- sglang.srt.entrypoints.engine ----------------------------------------------------------------
@dataclass
class FakeCall:
    method: str
    kwargs: dict[str, Any]


@dataclass
class EngineScript:
    """Knobs the tests twist to shape fake outputs."""

    finish: str | None = "stop"  # type field of meta_info.finish_reason; None → unfinished
    output_len: int = 3
    update_result: tuple[bool, str] = (True, "Success.")
    disk_result: tuple[bool, str, int] = (True, "Succeeded to update model weights.", 0)
    flush_success: bool = True
    include_output_ids: bool = True
    logprob_type: type = float
    raise_on_release: Exception | None = None


class Engine:
    """Records every call; emits outputs shaped like ``TokenizerManager._handle_batch_output``."""

    instances: list[Engine] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.server_args = types.SimpleNamespace(**kwargs)
        self.tp_size = int(kwargs.get("tp_size", 1))
        self.calls: list[FakeCall] = []
        self.script = EngineScript()
        self.shut_down = False
        Engine.instances.append(self)

    # Signature mirrors Engine.generate at v0.5.23 (positional names matter for the seed patch).
    def generate(
        self,
        prompt=None,
        sampling_params=None,
        input_ids=None,
        image_data=None,
        audio_data=None,
        video_data=None,
        mm_hashes=None,
        mm_content_hashes=None,
        return_logprob=False,
        logprob_start_len=None,
        top_logprobs_num=None,
        token_ids_logprob=None,
        lora_path=None,
        custom_logit_processor=None,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        stream=False,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
        routed_dp_rank=None,
        disagg_prefill_dp_rank=None,
        kv_hints=None,
        data_parallel_rank=None,
        external_trace_header=None,
        rid=None,
        session_params=None,
        priority=None,
        session_id=None,
        *,
        cache_salt=None,
    ):
        self.calls.append(
            FakeCall(
                "generate",
                {
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "input_ids": input_ids,
                    "image_data": image_data,
                    "return_logprob": return_logprob,
                    "logprob_start_len": logprob_start_len,
                    "stream": stream,
                },
            )
        )
        batch = input_ids if input_ids is not None else prompt
        single = isinstance(batch, str) or (isinstance(batch, list) and batch and isinstance(batch[0], int))
        if single:
            batch = [batch]
        sps = sampling_params if isinstance(sampling_params, list) else [sampling_params or {}] * len(batch)
        outputs = [
            self._output(
                i, k, sp, prompt_len=len(item) if isinstance(item, list) else 1, return_logprob=return_logprob
            )
            for i, (item, sp) in enumerate(zip(batch, sps, strict=True))
            for k in range(int(sp.get("n", 1) or 1))
        ]
        if single and int(sps[0].get("n", 1) or 1) == 1:
            return outputs[0]
        return outputs

    def _output(
        self, i: int, k: int, sp: dict[str, Any], *, prompt_len: int, return_logprob: bool
    ) -> dict[str, Any]:
        s = self.script
        seed = sp.get("sampling_seed")
        base = 1000 + (seed if seed is not None else i * 10 + k)
        n_tokens = min(s.output_len, int(sp.get("max_new_tokens", s.output_len) or s.output_len))
        token_ids = [base + t for t in range(n_tokens)]
        if s.finish == "stop":
            finish: Any = {"type": "stop", "matched": token_ids[-1]}
        elif s.finish == "length":
            finish = {"type": "length", "length": n_tokens}
        elif s.finish == "abort":
            finish = {"type": "abort", "message": "aborted"}
        else:
            finish = None
        meta: dict[str, Any] = {
            "id": f"req-{i}-{k}",
            "finish_reason": finish,
            "prompt_tokens": prompt_len,
            "completion_tokens": n_tokens,
        }
        if return_logprob:
            meta["input_token_logprobs"] = [(None, 0, None)]
            meta["output_token_logprobs"] = [
                (s.logprob_type(-0.5 * (t + 1)), tok, None) for t, tok in enumerate(token_ids)
            ]
            meta["output_token_logprobs_length"] = n_tokens
        out: dict[str, Any] = {"text": f"text-{i}-{k}-{base}", "meta_info": meta}
        if s.include_output_ids:
            out["output_ids"] = list(token_ids)
        return out

    def update_weights_from_tensor(self, named_tensors, load_format=None, flush_cache=True):
        # Engine._serialize_tensors_per_rank: one MultiprocessingSerializer payload per TP rank.
        serialized = [MultiprocessingSerializer.serialize(named_tensors) for _ in range(self.tp_size)]
        self.calls.append(
            FakeCall(
                "update_weights_from_tensor",
                {
                    "named_tensors": named_tensors,
                    "serialized_named_tensors": serialized,
                    "load_format": load_format,
                    "flush_cache": flush_cache,
                },
            )
        )
        return self.script.update_result

    def update_weights_from_disk(self, model_path, load_format=None):
        self.calls.append(
            FakeCall("update_weights_from_disk", {"model_path": model_path, "load_format": load_format})
        )
        return self.script.disk_result

    def release_memory_occupation(self, tags=None):
        self.calls.append(FakeCall("release_memory_occupation", {"tags": tags}))
        if self.script.raise_on_release is not None:
            raise self.script.raise_on_release

    def resume_memory_occupation(self, tags=None):
        self.calls.append(FakeCall("resume_memory_occupation", {"tags": tags}))

    def flush_cache(self):
        self.calls.append(FakeCall("flush_cache", {}))
        return types.SimpleNamespace(
            success=self.script.flush_success, message="" if self.script.flush_success else "busy"
        )

    def shutdown(self):
        self.calls.append(FakeCall("shutdown", {}))
        self.shut_down = True

    def methods(self) -> list[str]:
        return [c.method for c in self.calls]


# -- module tree ----------------------------------------------------------------------------------
_MODULES = (
    "sglang",
    "sglang.srt",
    "sglang.srt.entrypoints",
    "sglang.srt.entrypoints.engine",
    "sglang.srt.utils",
    "sglang.srt.managers",
    "sglang.srt.managers.tokenizer_manager",
    "sglang.srt.sampling",
    "sglang.srt.sampling.sampling_params",
)


def install_fake_sglang(version: str = SGLANG_FAKE_VERSION) -> dict[str, types.ModuleType]:
    mods: dict[str, types.ModuleType] = {}
    for name in _MODULES:
        mod = types.ModuleType(name)
        mod.__path__ = []  # mark as package so importlib walks submodules via sys.modules
        mods[name] = mod
    mods["sglang"].__version__ = version
    mods["sglang.srt.entrypoints.engine"].Engine = Engine
    mods["sglang"].Engine = Engine
    mods["sglang.srt.utils"].MultiprocessingSerializer = MultiprocessingSerializer
    mods["sglang.srt.managers.tokenizer_manager"].TokenizerManager = TokenizerManager
    mods["sglang.srt.sampling.sampling_params"].SamplingParams = SamplingParams
    for name, mod in mods.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(mods[parent], child, mod)
        sys.modules[name] = mod
    return mods


def remove_fake_sglang() -> None:
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            del sys.modules[name]


def make_config(run_config_dict: dict[str, Any], **rollout: Any):
    """``run_config_dict`` with ``rollout.backend='sglang'`` and overrides."""
    from monorl.config.schema import RunConfig

    payload = dict(run_config_dict)
    payload["rollout"] = {**run_config_dict["rollout"], "backend": "sglang", "device": None, **rollout}
    return RunConfig.model_validate(payload)
