"""Worker extension injected into vLLM's GPU worker via ``worker_extension_cls``.

vLLM v0.33.0 ``vllm/v1/worker/worker_base.py`` ``WorkerWrapperBase.init_worker`` resolves the
class by qualified name, asserts that **none** of its public attributes already exist on the
worker class, then appends it to ``worker_class.__bases__``; afterwards every public method is
callable through ``LLM.collective_rpc(name, args=...)`` with the worker as ``self``.

Because ``GPUWorker`` (``vllm/v1/worker/gpu_worker.py``) already defines ``update_weights`` /
``reload_weights`` for its own weight-transfer engines, our methods carry a ``monorl_`` prefix.
Inside them ``self`` is the ``GPUWorker``: ``self.model_runner`` is the ``GPUModelRunner`` whose
``.model`` is the loaded ``nn.Module`` (``load_weights(Iterable[tuple[str, Tensor]])`` is the
standard vLLM model-loading entry point) and ``self.reload_weights(weights_path=...)`` is the
upstream in-place reload (``vllm/model_executor/model_loader/reload/inplace.py``).

This module must stay importable without vLLM: the engine only hands its dotted path to vLLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class MonorlWorkerExtension:
    def monorl_update_weights(self, named_tensors: Sequence[tuple[str, torch.Tensor]]) -> int:
        """Load ``(hf_name, tensor)`` pairs into the running model. Returns the number of
        parameters vLLM reports as loaded (``load_weights`` returns a set of names)."""
        model = self.model_runner.model  # type: ignore[attr-defined]
        device = next(model.parameters()).device

        def on_device() -> Any:
            for name, tensor in named_tensors:
                yield name, tensor.to(device, non_blocking=True)

        loaded = model.load_weights(on_device())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return len(loaded) if loaded is not None else len(named_tensors)

    def monorl_load_weights_from_path(self, path: str) -> None:
        """Reload every weight from an HF-format directory (safetensors) in place."""
        self.reload_weights(weights_path=str(path), is_checkpoint_format=True)  # type: ignore[attr-defined]

    def monorl_device_info(self) -> dict[str, Any]:
        """Small diagnostic used by the server's ``/monorl/info`` endpoint and tests."""
        model = self.model_runner.model  # type: ignore[attr-defined]
        param = next(model.parameters())
        return {"device": str(param.device), "dtype": str(param.dtype).removeprefix("torch.")}
