"""Versioned monkeypatches reconciling Unsloth 2026.10.x with the monorl TRL backend.

What ``import unsloth`` does to TRL (``unsloth/models/rl.py``, run from ``unsloth/models/llama.py``
at import via ``PatchFastRL(FastLanguageModel=FastLlamaModel)``):

* ``patch_trl_rl_trainers`` regenerates every ``trl.trainer.*_trainer`` class from its source with
  Unsloth's replacements and **rebinds** ``trl.GRPOTrainer`` / ``trl.trainer.GRPOTrainer`` /
  ``trl.trainer.grpo_trainer.GRPOTrainer`` (and the configs) to ``UnslothGRPOTrainer``. Any patch
  monorl applied to the previous class object is orphaned.
* ``_wrap_grpo_generate_and_score`` replaces ``GRPOTrainer._generate_and_score_completions`` with
  ``wrapped(self, *args, **kwargs)`` (restores ``for_inference`` after rollouts, optional optimizer
  parking under ``UNSLOTH_PARK_OPTIMIZER_DURING_ROLLOUT=1``). The ``*args`` signature fails
  monorl's ``trl.grpo.monorl_hooks`` verify (``(self, inputs)``).
* ``patch_trl_disable_gradient_checkpointing`` makes ``trl.models.utils.disable_gradient_checkpointing``
  a no-op so TRL's generation context cannot undo Unsloth's checkpointing wrapper.
* ``patch_trl_vllm_generation`` / ``patch_trl_openenv`` / ``patch_trl_async_grpo`` /
  ``patch_trl_tool_chat_templates``: Unsloth-owned vLLM (``fast_inference=True``) and misc.

Compatibility matrix for ``pipeline_backend: trl`` with Unsloth imported:

==============================================  ==========  ==============================================
Unsloth change                                  status      how
==============================================  ==========  ==============================================
trainer class regeneration                      compatible  monorl looks ``trl.GRPOTrainer`` up at call time;
                                                            :func:`reconcile_trl_patches` re-applies the
                                                            monorl TRL patchset to the regenerated class
``_generate_and_score_completions`` wrapper     compatible  ``unsloth.trl.grpo_generate_signature`` restores
                                                            the ``(self, inputs)`` signature so the monorl
                                                            hooks wrap it
efficient GRPO loss (hidden states, fused CE)   compatible  internal to ``compute_loss``; env flags restored
``disable_gradient_checkpointing`` no-op        compatible  monorl never toggles checkpointing mid-run
``Trainer.prediction_step`` replacement         compatible  eval path only; forces logits per eval call
optimizer parking during rollouts               off         opt-in upstream (``UNSLOTH_PARK_OPTIMIZER_DURING_ROLLOUT=1``);
                                                            :func:`reconcile_trl_patches` pins the default ``0``
Unsloth vLLM generation (``fast_inference``)    off         monorl loads with ``fast_inference=False``;
                                                            TRL vLLM server/colocate modes stay TRL's
==============================================  ==========  ==============================================

Import order: Unsloth must be imported before ``trl`` is imported or patched. monorl applies TRL
patches in ``TRLPipelineBackend.run`` and looks the trainer class up afterwards, so calling
:func:`reconcile_trl_patches` (which applies this patchset and then re-applies ``trl``) right after
``import unsloth`` yields a consistent state whichever order the imports happened in.
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Any

from monorl.patching import Patch, PatchSet, apply, is_registered, register_patchset, revert

VERSIONS = ">=2026.10,<2027"
GRPO_GENERATE_TARGET = "trl.trainer.grpo_trainer.GRPOTrainer._generate_and_score_completions"
PARK_OPTIMIZER_ENV = "UNSLOTH_PARK_OPTIMIZER_DURING_ROLLOUT"


def is_unsloth_wrapper(original: Any) -> bool:
    """True for Unsloth's ``_wrap_grpo_generate_and_score`` wrapper (marker set by ``unsloth/models/rl.py``)."""
    return bool(getattr(original, "_unsloth_restore_training_wrapped", False))


def build_generate_signature_patch(original: Any) -> Any:
    """``(self, inputs)``-signature shim over Unsloth's ``wrapped(self, *args, **kwargs)``."""

    def _generate_and_score_completions(self: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        return original(self, inputs)

    _generate_and_score_completions.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("inputs", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ]
    )
    _generate_and_score_completions.__doc__ = getattr(original, "__doc__", None)
    return _generate_and_score_completions


PATCHES = PatchSet(
    framework="unsloth",
    patches=[
        Patch(
            id="unsloth.trl.grpo_generate_signature",
            target=GRPO_GENERATE_TARGET,
            versions=VERSIONS,
            mode="wrap",
            build=build_generate_signature_patch,
            verify=is_unsloth_wrapper,
            description=(
                "Give Unsloth's regenerated GRPOTrainer._generate_and_score_completions wrapper back its "
                "(self, inputs) signature so the monorl 'trl.grpo.monorl_hooks' patch verifies and wraps it."
            ),
        ),
    ],
)

register_patchset(PATCHES)


def unsloth_imported() -> bool:
    return "unsloth" in sys.modules


def trl_trainer_rewritten() -> bool:
    """True once Unsloth replaced ``trl.trainer.grpo_trainer.GRPOTrainer`` with ``UnslothGRPOTrainer``."""
    module = sys.modules.get("trl.trainer.grpo_trainer")
    cls = getattr(module, "GRPOTrainer", None)
    return cls is not None and cls.__name__.startswith("Unsloth")


def reconcile_trl_patches() -> dict[str, Any]:
    """Apply this patchset, then re-apply monorl's ``trl`` patchset on top of the Unsloth-rewritten
    trainer classes. Safe to call repeatedly; a no-op unless Unsloth is imported.

    The TRL hooks are reverted first because they stack on the same attribute as the signature
    shim: the engine judges "already applied" by identity of the live attribute, so the shim is only
    recognised while nothing sits on top of it. Journal entries that point at the pre-rewrite class
    objects are restored harmlessly on that class.

    Returns ``{"unsloth": ApplyReport | None, "trl": ApplyReport | None}``.
    """
    if not unsloth_imported():
        return {"unsloth": None, "trl": None}
    os.environ.setdefault(PARK_OPTIMIZER_ENV, "0")
    trl_registered = is_registered("trl") and "trl" in sys.modules
    if trl_registered:
        revert("trl")
    unsloth_report = apply("unsloth")
    trl_report = apply("trl") if trl_registered else None
    return {"unsloth": unsloth_report, "trl": trl_report}
