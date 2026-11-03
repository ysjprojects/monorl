"""Import hook: apply a framework's registered patchset right after its top-level module executes.

A single :class:`_PatchingFinder` sits at the front of ``sys.meta_path``. It only reacts to the
exact top-level names of registered frameworks; for those it asks the remaining finders for the
real spec and swaps the spec's loader for a thin wrapper that runs the real ``exec_module`` and
then :func:`monorl.patching.engine.apply`. Unrelated imports never touch it beyond one string
comparison, and submodule imports triggered while patching recurse harmlessly because they are
not top-level framework names.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import sys
import threading
from collections.abc import Sequence
from types import ModuleType
from typing import Any

from . import engine
from .patch import PatchSet, patches_disabled

log = logging.getLogger("monorl.patching")

_state_lock = threading.Lock()
_finder: _PatchingFinder | None = None


def _apply_logged(framework: str) -> None:
    """Apply during import; a bad patch must never break the framework import itself."""
    try:
        report = engine.apply(framework)
    except Exception:  # pragma: no cover - defensive: apply() reports rather than raises
        log.exception("import hook: applying patches for %s crashed", framework)
        return
    if report.failed:
        log.warning("import hook: %s", report)
    else:
        log.debug("import hook: %s", report)


class _PatchingLoader(importlib.abc.Loader):
    """Delegates everything to the real loader, then applies patches after ``exec_module``."""

    def __init__(self, real: Any, framework: str) -> None:
        self._real = real
        self._framework = framework

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        create = getattr(self._real, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: ModuleType) -> None:
        self._real.exec_module(module)
        # Leave no trace on the module: tools that inspect ``__loader__`` see the real one.
        if getattr(module, "__loader__", None) is self:
            module.__loader__ = self._real
        spec = getattr(module, "__spec__", None)
        if spec is not None and spec.loader is self:
            spec.loader = self._real
        _apply_logged(self._framework)

    def __getattr__(self, name: str) -> Any:  # get_code, get_source, is_package, get_resource_reader, …
        return getattr(self._real, name)

    def __repr__(self) -> str:
        return f"<monorl patching loader for {self._framework!r} wrapping {self._real!r}>"


class _PatchingFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not engine.is_registered(fullname) or patches_disabled():
            return None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is None:
                continue
            if spec.loader is None or not hasattr(spec.loader, "exec_module"):
                return spec  # namespace package / legacy loader: nothing to run after
            spec.loader = _PatchingLoader(spec.loader, fullname)
            return spec
        return None

    def invalidate_caches(self) -> None:
        return None

    def __repr__(self) -> str:
        return "<monorl patching import hook>"


def _on_register(patchset: PatchSet) -> None:
    """Patchsets registered while the hook is active get applied at once if already imported."""
    if import_hook_active() and patchset.is_imported() and not patches_disabled():
        _apply_logged(patchset.framework)


def activate_import_hook() -> bool:
    """Install the hook (idempotent) and apply patchsets whose framework is already imported.

    Returns ``True`` when the hook is active, ``False`` when ``MONORL_DISABLE_PATCHES`` is set.
    """
    global _finder
    if patches_disabled():
        log.info("import hook not activated: MONORL_DISABLE_PATCHES is set")
        return False
    with _state_lock:
        if _finder is None:
            _finder = _PatchingFinder()
            engine._on_register.append(_on_register)
        if _finder not in sys.meta_path:
            sys.meta_path.insert(0, _finder)
    for framework in engine.registered_frameworks():
        if framework in sys.modules:
            _apply_logged(framework)
    return True


def deactivate_import_hook() -> bool:
    """Remove the hook. Applied patches stay in place (use :func:`engine.revert`). Returns whether it was active."""
    global _finder
    with _state_lock:
        finder, _finder = _finder, None
        if finder is None:
            return False
        was_active = finder in sys.meta_path
        sys.meta_path[:] = [f for f in sys.meta_path if f is not finder]
        if _on_register in engine._on_register:
            engine._on_register.remove(_on_register)
    return was_active


def import_hook_active() -> bool:
    finder = _finder
    return finder is not None and finder in sys.meta_path
