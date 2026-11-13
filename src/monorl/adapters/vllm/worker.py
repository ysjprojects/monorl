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

Speculative decoding (``speculative_config={"method": "mtp", ...}``): ``self.get_draft_model()``
is the drafter, a second ``nn.Module`` holding the checkpoint's MTP layers and sharing the
target's embedding and output head (``vllm/v1/worker/gpu/spec_decode/eagle/utils.py``
``load_eagle_model``). Neither ``model_runner.model.load_weights`` nor ``reload_weights`` touches
it, so both entry points here load it as well, the way ``GPUWorker.start_draft_weight_update`` /
``finish_weight_update`` do for vLLM's own weight-transfer engines:

* its ``load_weights`` takes checkpoint names and skips the ones it does not own; a partial
  stream is only accepted inside ``disable_mtp_completeness_check()``
  (``vllm/model_executor/model_loader/mtp_validation.py``);
* with ``draft_lm_head_quantization`` the drafter holds a quantized copy of the target's output
  head: ``share_source_lm_heads`` swaps the real head in for the load,
  ``restore_quantized_lm_heads`` swaps the copy back and ``refresh_quantized_lm_heads``
  re-derives it from the head's new weights
  (``vllm/model_executor/layers/quantization/online/lm_head.py``).

This module must stay importable without vLLM: the engine only hands its dotted path to vLLM.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def _load_drafter(draft_model: Any, load: Callable[[], Any]) -> Any:
    """Run ``load`` (a weight load into ``draft_model``) through the unquantized output head and
    without the completeness check. The quantized copies are put back but not re-derived: the
    caller does that once every weight has arrived."""
    from vllm.model_executor.layers.quantization.online.lm_head import (
        restore_quantized_lm_heads,
        share_source_lm_heads,
    )
    from vllm.model_executor.model_loader.mtp_validation import disable_mtp_completeness_check

    share_source_lm_heads(draft_model)
    try:
        with disable_mtp_completeness_check():
            return load()
    finally:
        restore_quantized_lm_heads(draft_model, refresh=False)


class MonorlWorkerExtension:
    def monorl_update_weights(
        self,
        named_tensors: Sequence[tuple[str, torch.Tensor]],
        mtp_tensors: Sequence[tuple[str, torch.Tensor]] = (),
    ) -> int:
        """Load ``(hf_name, tensor)`` pairs into the running model, and into the drafter when
        the engine drafts. ``mtp_tensors`` are MTP layers' weights under their checkpoint keys:
        they reach the drafter only, never the target (a worker without a drafter ignores
        them). Returns the number of parameters vLLM reports as loaded (``load_weights`` returns
        a set of names).

        After the last call of an update the caller runs :meth:`monorl_refresh_drafter`."""
        model = self.model_runner.model  # type: ignore[attr-defined]
        device = next(model.parameters()).device
        draft_model = self.get_draft_model()  # type: ignore[attr-defined]

        def on_device() -> Any:
            for name, tensor in named_tensors:
                yield name, tensor.to(device, non_blocking=True)

        count = 0
        if named_tensors:
            loaded = model.load_weights(on_device())
            count += len(loaded) if loaded is not None else len(named_tensors)
        if draft_model is not None:
            draft_device = next(draft_model.parameters()).device
            # The drafter sees the whole stream, like at start-up: it picks its own layers and,
            # where it could not share them (pipeline parallelism), the embedding and the head.
            moved = [
                (name, tensor.to(draft_device, non_blocking=True))
                for name, tensor in (*named_tensors, *mtp_tensors)
            ]
            loaded = _load_drafter(draft_model, lambda: draft_model.load_weights(moved))
            count += len(loaded) if loaded is not None else len(mtp_tensors)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return count

    def monorl_refresh_drafter(self) -> None:
        """Bring the drafter's derived state in line with the weights just pushed: its quantized
        copy of the output head (``draft_lm_head_quantization``). A no-op without a drafter."""
        draft_model = self.get_draft_model()  # type: ignore[attr-defined]
        if draft_model is None:
            return
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.quantization.online.lm_head import refresh_quantized_lm_heads

        with set_current_vllm_config(self.vllm_config):  # type: ignore[attr-defined]
            refresh_quantized_lm_heads(draft_model)

    def monorl_load_weights_from_path(self, path: str) -> None:
        """Reload every weight from an HF-format directory (safetensors) in place: the target
        with the upstream reload, then the drafter with a load pass of its own over the same
        directory. A directory without MTP weights leaves the drafter's layers as they are."""
        self.reload_weights(weights_path=str(path), is_checkpoint_format=True)  # type: ignore[attr-defined]
        draft_model = self.get_draft_model()  # type: ignore[attr-defined]
        if draft_model is None:
            return
        from vllm.config import replace, set_current_vllm_config
        from vllm.model_executor.layers.quantization.online.lm_head import refresh_quantized_lm_heads
        from vllm.model_executor.model_loader.reload.inplace import reload_weights
        from vllm.model_executor.model_loader.utils import get_draft_load_config
        from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

        vllm_config = self.vllm_config  # type: ignore[attr-defined]
        speculative_config = vllm_config.speculative_config
        # The config the drafter was built with (load_eagle_model).
        draft_config = replace(
            speculative_config.apply_draft_overrides(vllm_config),
            model_config=speculative_config.draft_model_config,
            load_config=get_pp_safe_draft_load_config(get_draft_load_config(vllm_config)),
        )
        with set_current_vllm_config(vllm_config):
            _load_drafter(
                draft_model,
                lambda: reload_weights(draft_config, draft_model, None, str(path), True),
            )
            refresh_quantized_lm_heads(draft_model)

    def monorl_device_info(self) -> dict[str, Any]:
        """Small diagnostic used by the server's ``/monorl/info`` endpoint and tests.
        ``speculative`` describes the drafting this worker does (``None``: it has no drafter);
        the server-mode client checks it against ``model.mtp``."""
        model = self.model_runner.model  # type: ignore[attr-defined]
        param = next(model.parameters())
        speculative = None
        if self.get_draft_model() is not None:  # type: ignore[attr-defined]
            config = self.vllm_config.speculative_config  # type: ignore[attr-defined]
            speculative = {
                "method": config.method,
                "num_speculative_tokens": config.num_speculative_tokens,
                "rejection_sample_method": config.rejection_sample_method,
                "draft_lm_head_quantization": config.draft_lm_head_quantization,
            }
        return {
            "device": str(param.device),
            "dtype": str(param.dtype).removeprefix("torch."),
            "speculative": speculative,
        }
