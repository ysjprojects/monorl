"""Patch builders and the Unsloth ↔ monorl-TRL reconciliation.

``trl`` is faked in ``sys.modules`` the way ``unsloth/models/rl.py`` leaves it after
``patch_trl_rl_trainers``: ``trl.trainer.grpo_trainer.GRPOTrainer`` rebound to a *regenerated*
``UnslothGRPOTrainer`` (not a subclass) whose ``_generate_and_score_completions`` is the
``wrapped(self, *args, **kwargs)`` closure from ``_wrap_grpo_generate_and_score`` (marker attribute
``_unsloth_restore_training_wrapped = True``).
"""

from __future__ import annotations

import inspect
import sys
import types
from typing import Any

import pytest

from monorl.patching import is_applied, is_registered, revert, status
from tests.adapters.unsloth.conftest import _module


def _pristine_trl() -> types.ModuleType:
    class GRPOTrainer:
        def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
            return {"inputs": inputs, "via": "trl"}

    grpo = _module("trl.trainer.grpo_trainer", GRPOTrainer=GRPOTrainer)
    trainer = _module("trl.trainer", grpo_trainer=grpo, GRPOTrainer=GRPOTrainer, __path__=[])
    root = _module("trl", __version__="1.17.1", trainer=trainer, GRPOTrainer=GRPOTrainer, __path__=[])
    root._submodules = {"trl.trainer": trainer, "trl.trainer.grpo_trainer": grpo}
    return root


def _unsloth_rewrite(trl: types.ModuleType) -> type:
    """What ``_patch_trl_rl_trainers`` + ``_wrap_grpo_generate_and_score`` leave behind."""

    class UnslothGRPOTrainer:
        def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
            return {"inputs": inputs, "via": "unsloth"}

    original = UnslothGRPOTrainer._generate_and_score_completions

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original(self, *args, **kwargs)

    wrapped._unsloth_restore_training_wrapped = True
    UnslothGRPOTrainer._generate_and_score_completions = wrapped
    for mod in (trl, trl.trainer, trl.trainer.grpo_trainer):
        mod.GRPOTrainer = UnslothGRPOTrainer
    return UnslothGRPOTrainer


class _Hooks:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []

    def before_generate(self, trainer: Any, inputs: Any) -> None:
        self.events.append(("before", inputs))

    def after_score(self, trainer: Any, output: Any) -> None:
        self.events.append(("after", output))


@pytest.fixture
def fake_trl(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    import monorl.adapters.trl.patches  # noqa: F401  registers the 'trl' patchset

    # Other tests in this process (the TRL backend e2e) may have applied the real patch; restore
    # the real class and clear the journal before swapping in the fake tree.
    revert("trl")
    for name in [m for m in sys.modules if m == "trl" or m.startswith("trl.")]:
        monkeypatch.delitem(sys.modules, name)
    trl = _pristine_trl()
    monkeypatch.setitem(sys.modules, "trl", trl)
    for name, mod in trl._submodules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    yield trl
    revert("trl")


# -- patchset -------------------------------------------------------------------------------------


def test_patchset_registered_and_gated_on_install(monkeypatch: pytest.MonkeyPatch) -> None:
    from monorl.adapters.unsloth import PATCHES

    assert is_registered("unsloth")
    assert PATCHES.framework == "unsloth"
    assert [p.id for p in PATCHES.patches] == ["unsloth.trl.grpo_generate_signature"]
    patch = PATCHES.get("unsloth.trl.grpo_generate_signature")
    assert patch.mode == "wrap" and patch.versions == ">=2026.10,<2027"
    assert patch.version_applicable("2026.10.11") == (True, "")
    assert patch.version_applicable("2025.12.1")[0] is False

    for name in [m for m in sys.modules if m == "unsloth" or m.startswith("unsloth.")]:
        monkeypatch.delitem(sys.modules, name)
    (st,) = status("unsloth")
    assert st.applicable is False and st.applied is False
    assert "not installed" in st.reason


def test_generate_signature_builder() -> None:
    from monorl.adapters.trl.patches import verify_hooks_target
    from monorl.adapters.unsloth.patches import build_generate_signature_patch, is_unsloth_wrapper

    seen: list[Any] = []

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> str:
        seen.append((self, args, kwargs))
        return "ok"

    wrapped._unsloth_restore_training_wrapped = True
    assert is_unsloth_wrapper(wrapped)
    assert not is_unsloth_wrapper(lambda self, inputs: None)
    assert not verify_hooks_target(wrapped)  # the raw Unsloth wrapper fails monorl's TRL verify

    shim = build_generate_signature_patch(wrapped)
    assert list(inspect.signature(shim).parameters) == ["self", "inputs"]
    assert verify_hooks_target(shim)
    shim.__wrapped__ = wrapped  # what the patch engine does for mode="wrap"; __signature__ must still win
    assert list(inspect.signature(shim).parameters) == ["self", "inputs"]
    assert shim("trainer", [{"prompt": "x"}]) == "ok"
    assert seen == [("trainer", ([{"prompt": "x"}],), {})]


# -- reconciliation -------------------------------------------------------------------------------


def test_reconcile_is_noop_without_unsloth(monkeypatch: pytest.MonkeyPatch) -> None:
    from monorl.adapters.unsloth.patches import reconcile_trl_patches, trl_trainer_rewritten, unsloth_imported

    for name in [m for m in sys.modules if m == "unsloth" or m.startswith("unsloth.")]:
        monkeypatch.delitem(sys.modules, name)
    assert not unsloth_imported()
    assert not trl_trainer_rewritten()
    assert reconcile_trl_patches() == {"unsloth": None, "trl": None}


def test_reconcile_rehooks_regenerated_grpo_trainer(fake_trl: types.ModuleType, fake_unsloth: types.ModuleType) -> None:
    from monorl.adapters.trl.patches import HOOKS_ATTR, hooks_patch_applied
    from monorl.adapters.unsloth.patches import reconcile_trl_patches, trl_trainer_rewritten
    from monorl.patching import apply

    # 1. monorl patched TRL first (TRLPipelineBackend.run does this) ...
    report = apply("trl")
    assert report.applied == ["trl.grpo.monorl_hooks"], report
    assert hooks_patch_applied()
    pristine_cls = fake_trl.trainer.grpo_trainer.GRPOTrainer

    # 2. ... then `import unsloth` regenerated the trainer class: the hook is orphaned on the old class.
    new_cls = _unsloth_rewrite(fake_trl)
    assert trl_trainer_rewritten()
    assert not hooks_patch_applied()
    assert is_applied("trl.grpo.monorl_hooks", "trl")  # journal still believes it is in place (old owner)
    assert apply("trl").skipped == [("trl.grpo.monorl_hooks", "already applied")]

    # 3. reconcile: signature shim on the Unsloth wrapper, then the monorl hooks on top.
    reports = reconcile_trl_patches()
    assert reports["unsloth"].applied == ["unsloth.trl.grpo_generate_signature"], reports["unsloth"]
    assert reports["trl"].applied == ["trl.grpo.monorl_hooks"], reports["trl"]
    assert hooks_patch_applied()
    assert fake_trl.trainer.grpo_trainer.GRPOTrainer is new_cls
    assert pristine_cls._generate_and_score_completions.__name__ == "_generate_and_score_completions"
    assert getattr(pristine_cls._generate_and_score_completions, "__monorl_patch__", None) is None  # reverted

    # hooks fire and the call flows through shim → Unsloth wrapper → regenerated method
    trainer = new_cls()
    hooks = _Hooks()
    setattr(trainer, HOOKS_ATTR, hooks)
    out = trainer._generate_and_score_completions([{"prompt": "p"}])
    assert out == {"inputs": [{"prompt": "p"}], "via": "unsloth"}
    assert hooks.events == [("before", [{"prompt": "p"}]), ("after", out)]

    # idempotent
    again = reconcile_trl_patches()
    assert again["unsloth"].skipped == [("unsloth.trl.grpo_generate_signature", "already applied")]
    assert again["trl"].applied == ["trl.grpo.monorl_hooks"]
    assert hooks_patch_applied()


def test_reconcile_without_prior_trl_apply(fake_trl: types.ModuleType, fake_unsloth: types.ModuleType) -> None:
    """``import unsloth`` happened first (the recommended order): one call wires everything."""
    from monorl.adapters.trl.patches import hooks_patch_applied
    from monorl.adapters.unsloth.patches import reconcile_trl_patches

    _unsloth_rewrite(fake_trl)
    reports = reconcile_trl_patches()
    assert reports["unsloth"].applied == ["unsloth.trl.grpo_generate_signature"]
    assert reports["trl"].applied == ["trl.grpo.monorl_hooks"]
    assert hooks_patch_applied()


def test_signature_patch_skipped_on_pristine_trl(fake_trl: types.ModuleType, fake_unsloth: types.ModuleType) -> None:
    """Unsloth imported but TRL trainers not rewritten (e.g. UNSLOTH_ALLOW_CPU=1): nothing to shim."""
    from monorl.adapters.unsloth.patches import reconcile_trl_patches

    reports = reconcile_trl_patches()
    assert reports["unsloth"].applied == []
    assert reports["unsloth"].skipped == [("unsloth.trl.grpo_generate_signature", "verify() returned False")]
    assert reports["trl"].applied == ["trl.grpo.monorl_hooks"]
