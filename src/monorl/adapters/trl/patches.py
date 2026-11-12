"""Versioned monkeypatches for TRL 1.x ``GRPOTrainer``.

``trl.grpo.monorl_hooks`` wraps ``GRPOTrainer._generate_and_score_completions`` — the one method
that sees both the per-example dataset rows and the scored generation batch — and forwards them to
the :class:`~monorl.adapters.trl.rollout.GRPOHooks` instance a trainer carries under
``trainer.monorl_hooks`` (absent on vanilla trainers → pass-through):

* ``before_generate(trainer, inputs)`` hands the rows to the multi-turn
  :class:`~monorl.adapters.trl.rollout.EpisodeRollout` so ``env.reset`` gets real
  :class:`~monorl.core.types.Sample` objects (TRL's ``rollout_func`` only receives prompts);
* ``after_score(trainer, output)`` fires once the batch is rewarded and advantages are in
  ``trainer._logs`` — the completion-logging path the monorl reporter samples trajectories from.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any

from monorl.patching import Patch, PatchSet, register_patchset

HOOKS_ATTR = "monorl_hooks"
TARGET = "trl.trainer.grpo_trainer.GRPOTrainer._generate_and_score_completions"


def build_hooks_patch(original: Any) -> Any:
    @functools.wraps(original)
    def _generate_and_score_completions(self: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        hooks = getattr(self, HOOKS_ATTR, None)
        if hooks is None:
            return original(self, inputs)
        hooks.before_generate(self, inputs)
        output = original(self, inputs)
        hooks.after_score(self, output)
        return output

    return _generate_and_score_completions


def verify_hooks_target(original: Any) -> bool:
    params = list(inspect.signature(original).parameters)
    return params[:2] == ["self", "inputs"]


PATCHES = PatchSet(
    framework="trl",
    patches=[
        Patch(
            id="trl.grpo.monorl_hooks",
            target=TARGET,
            versions=">=1.17,<2",
            mode="wrap",
            build=build_hooks_patch,
            verify=verify_hooks_target,
            description=(
                "Route GRPOTrainer generation through monorl multi-turn environments and hand each scored "
                "generation batch to the monorl reporter (via trainer.monorl_hooks)."
            ),
        )
    ],
)

register_patchset(PATCHES)


def hooks_patch_applied() -> bool:
    """True when the installed ``GRPOTrainer`` carries the ``trl.grpo.monorl_hooks`` wrapper."""
    from trl.trainer.grpo_trainer import GRPOTrainer

    return (
        getattr(GRPOTrainer._generate_and_score_completions, "__monorl_patch__", None)
        == "trl.grpo.monorl_hooks"
    )
