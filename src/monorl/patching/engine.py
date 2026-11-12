"""Global patchset registry, the apply journal, and ``apply`` / ``revert`` / ``status`` / ``manifest``.

Adapters call :func:`register_patchset` at import time. :func:`apply` is version-gated,
verification-gated and idempotent; every applied patch is journaled with its pristine original so
:func:`revert` can restore it. ``MONORL_DISABLE_PATCHES=1`` turns ``apply`` into a no-op that
reports every patch as skipped.
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

from monorl.core.errors import PatchError

from .patch import (
    MISSING,
    ApplyReport,
    Patch,
    PatchSet,
    PatchStatus,
    ResolvedTarget,
    detect_version,
    patches_disabled,
    resolve_target,
)

log = logging.getLogger("monorl.patching")

_lock = threading.RLock()  # guards _registry / _journal / _locks bookkeeping (short sections only)
_locks: dict[str, threading.RLock] = {}  # per-framework: serialises apply/revert without blocking imports
_registry: dict[str, list[PatchSet]] = {}
_journal: dict[str, dict[str, _Applied]] = {}
_on_register: list[Any] = []  # callbacks (patchset) -> None, used by the import hook


def _framework_lock(framework: str) -> threading.RLock:
    with _lock:
        lock = _locks.get(framework)
        if lock is None:
            lock = _locks[framework] = threading.RLock()
        return lock


@dataclass
class _Applied:
    patch: Patch
    owner: Any
    attr: str
    raw_original: Any  # vars(owner)[attr] before the first apply, or MISSING
    replacement: Any

    @property
    def in_place(self) -> bool:
        """True while the attribute still holds our replacement."""
        try:
            return vars(self.owner).get(self.attr, MISSING) is self.replacement
        except TypeError:
            return getattr(self.owner, self.attr, MISSING) is self.replacement

    def restore(self) -> None:
        if self.raw_original is MISSING:
            if self.attr in getattr(self.owner, "__dict__", {}):
                delattr(self.owner, self.attr)
        else:
            setattr(self.owner, self.attr, self.raw_original)


# --------------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------------


def register_patchset(patchset: PatchSet) -> PatchSet:
    """Register a framework's patches. Re-registering a set with overlapping ids replaces it;
    a set with disjoint ids for the same framework is added alongside (ids must stay unique)."""
    if not isinstance(patchset, PatchSet):
        raise PatchError(f"register_patchset expects a PatchSet, got {type(patchset).__name__}")
    with _lock:
        sets = _registry.setdefault(patchset.framework, [])
        new_ids = {p.id for p in patchset.patches}
        kept: list[PatchSet] = []
        for existing in sets:
            if existing is patchset or new_ids & {p.id for p in existing.patches}:
                continue  # replaced by the incoming set
            kept.append(existing)
        for other in kept:
            dup = new_ids & {p.id for p in other.patches}
            if dup:
                raise PatchError(f"patch ids {sorted(dup)} already registered for {patchset.framework!r}")
        kept.append(patchset)
        _registry[patchset.framework] = kept
        callbacks = list(_on_register)
    for cb in callbacks:
        cb(patchset)
    return patchset


def unregister_patchset(framework: str) -> list[str]:
    """Revert and forget every patchset registered for ``framework``. Returns reverted patch ids."""
    with _framework_lock(framework):
        reverted = revert(framework) if is_registered(framework) else []
        with _lock:
            _registry.pop(framework, None)
            _journal.pop(framework, None)
    return reverted


def is_registered(framework: str) -> bool:
    return framework in _registry


def registered_frameworks() -> list[str]:
    with _lock:
        return sorted(_registry)


def get_patchsets(framework: str) -> list[PatchSet]:
    with _lock:
        sets = _registry.get(framework)
    if not sets:
        raise PatchError(f"no patchset registered for framework {framework!r}")
    return list(sets)


def iter_patches(framework: str) -> list[Patch]:
    return [p for ps in get_patchsets(framework) for p in ps.patches]


def installed_version(framework: str) -> str | None:
    """Installed version for a registered framework (metadata first, ``__version__`` fallback)."""
    sets = get_patchsets(framework)
    distribution = next((ps.distribution for ps in sets if ps.distribution), None)
    return detect_version(framework, distribution)


# --------------------------------------------------------------------------------------------
# apply / revert
# --------------------------------------------------------------------------------------------


def apply(framework: str, force: bool = False) -> ApplyReport:
    """Apply every registered patch for ``framework`` that passes its gates.

    Gates, in order: ``MONORL_DISABLE_PATCHES``; framework installed; already applied (idempotent);
    version specifier; target resolvable; ``expected_signature``; ``verify``. ``force=True`` re-applies
    patches that are already in place and bypasses the version/signature/verify gates (never the
    disable switch or target resolution).
    """
    patches = iter_patches(framework)
    report = ApplyReport(framework=framework)
    if patches_disabled():
        report.skipped = [(p.id, "disabled by MONORL_DISABLE_PATCHES") for p in patches]
        return report
    try:
        report.installed_version = installed_version(framework)
    except PatchError as e:
        report.failed = [(p.id, str(e)) for p in patches]
        return report
    if report.installed_version is None:
        report.skipped = [(p.id, f"framework {framework!r} is not installed") for p in patches]
        return report
    # Import the top-level module *before* taking the framework lock: when another thread is in
    # the middle of importing it (and its import hook will call apply()), we wait on the import
    # lock here instead of holding our lock while waiting, which would deadlock.
    try:
        importlib.import_module(framework)
    except Exception as e:
        msg = f"importing {framework!r} failed: {e!r}"
        report.failed = [(p.id, msg) for p in patches]
        return report

    with _framework_lock(framework):
        with _lock:
            journal = _journal.setdefault(framework, {})
        for patch in patches:
            try:
                outcome = _apply_one(patch, journal, report.installed_version, force=force)
            except PatchError as e:
                log.warning("patch %s failed: %s", patch.id, e)
                report.failed.append((patch.id, str(e)))
                continue
            except Exception as e:  # build() or verify() blew up with a non-PatchError
                msg = f"patch {patch.id!r} on target {patch.target}: {type(e).__name__}: {e}"
                log.warning("%s", msg, exc_info=True)
                report.failed.append((patch.id, msg))
                continue
            kind, reason = outcome
            if kind == "applied":
                log.info("applied patch %s -> %s", patch.id, patch.target)
                report.applied.append(patch.id)
            else:
                log.debug("skipped patch %s: %s", patch.id, reason)
                report.skipped.append((patch.id, reason))
    return report


def _apply_one(patch: Patch, journal: dict[str, _Applied], installed: str, *, force: bool) -> tuple[str, str]:
    record = journal.get(patch.id)
    if record is not None:
        if record.in_place and not force:
            return "skipped", "already applied"
        if record.in_place:
            record.restore()  # force: start again from the pristine original
        journal.pop(patch.id, None)

    if not force:
        ok, reason = patch.version_applicable(installed)
        if not ok:
            return "skipped", reason

    resolved = resolve_target(patch.target, patch_id=patch.id)
    original = _check_mode(patch, resolved)

    if not force:
        verified, reason = patch.check_original(original)
        if not verified:
            return "skipped", reason

    try:
        replacement = patch.build(original)
    except PatchError:
        raise
    except Exception as e:
        raise PatchError(
            f"patch {patch.id!r}: build() raised for target {patch.target}: {type(e).__name__}: {e}"
        ) from e
    if patch.mode == "wrap":
        if not callable(replacement):
            raise PatchError(f"patch {patch.id!r}: wrap build() must return a callable for {patch.target}")
        if getattr(replacement, "__wrapped__", None) is None:
            _try_setattr(replacement, "__wrapped__", original)
    _try_setattr(replacement, "__monorl_patch__", patch.id)

    # Keep the earliest pristine original if a previous record exists for this target.
    raw_original = resolved.raw
    if record is not None and record.owner is resolved.owner and record.attr == resolved.attr:
        raw_original = record.raw_original
    try:
        setattr(resolved.owner, resolved.attr, replacement)
    except (AttributeError, TypeError) as e:
        raise PatchError(f"patch {patch.id!r}: cannot set {patch.target} on {resolved.owner!r}: {e}") from e
    journal[patch.id] = _Applied(
        patch=patch,
        owner=resolved.owner,
        attr=resolved.attr,
        raw_original=raw_original,
        replacement=replacement,
    )
    return "applied", ""


def _check_mode(patch: Patch, resolved: ResolvedTarget) -> Any:
    """Validate mode vs. target presence; return the ``original`` passed to ``build``."""
    if patch.mode == "add":
        if resolved.exists and not patch.overwrite:
            raise PatchError(
                f"patch {patch.id!r}: mode 'add' but target {patch.target} already exists "
                f"(pass overwrite=True to replace it)"
            )
        return None
    if not resolved.exists:
        raise PatchError(
            f"patch {patch.id!r}: target {patch.target} does not exist on {resolved.owner!r} "
            f"(mode {patch.mode!r} needs an existing attribute)"
        )
    if patch.mode == "wrap" and not callable(resolved.current):
        raise PatchError(f"patch {patch.id!r}: mode 'wrap' but target {patch.target} is not callable")
    return resolved.current


def _try_setattr(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except (AttributeError, TypeError):
        pass


def revert(framework: str) -> list[str]:
    """Restore the pristine originals of every applied patch for ``framework`` (last applied first)."""
    with _framework_lock(framework), _lock:
        journal = _journal.get(framework)
        if not journal:
            return []
        reverted: list[str] = []
        for patch_id in reversed(list(journal)):
            record = journal.pop(patch_id)
            if not record.in_place:
                log.warning(
                    "patch %s: target %s was modified after apply; restoring original anyway",
                    patch_id,
                    record.patch.target,
                )
            record.restore()
            reverted.append(patch_id)
    log.info("reverted %d patch(es) for %s", len(reverted), framework)
    return reverted


def revert_all() -> dict[str, list[str]]:
    with _lock:
        return {fw: revert(fw) for fw in list(_journal)}


def is_applied(patch_id: str, framework: str | None = None) -> bool:
    with _lock:
        journals = [_journal.get(framework, {})] if framework else list(_journal.values())
        return any(patch_id in j and j[patch_id].in_place for j in journals)


# --------------------------------------------------------------------------------------------
# status / manifest
# --------------------------------------------------------------------------------------------


def status(framework: str, *, resolve: bool = True) -> list[PatchStatus]:
    """One :class:`PatchStatus` per registered patch.

    With ``resolve=True`` (default) the targets of applicable-but-unapplied patches are resolved
    and verified, which imports the framework. ``resolve=False`` only consults metadata and the
    journal (``verified=None`` for unapplied patches) — use it where importing is unwanted.
    """
    with _lock:
        sets = list(_registry.get(framework, ()))
    if not sets:
        return []  # query, not a programming error: unknown/unpatched frameworks have no statuses
    patches = [p for ps in sets for p in ps.patches]
    disabled = patches_disabled()
    try:
        installed = installed_version(framework)
        install_error = ""
    except PatchError as e:
        installed, install_error = None, str(e)
    with _lock:
        journal = dict(_journal.get(framework, {}))
    out: list[PatchStatus] = []
    for patch in patches:
        out.append(
            _status_one(
                patch,
                installed,
                journal.get(patch.id),
                disabled=disabled,
                install_error=install_error,
                resolve=resolve,
            )
        )
    return out


def _status_one(
    patch: Patch,
    installed: str | None,
    record: _Applied | None,
    *,
    disabled: bool,
    install_error: str,
    resolve: bool,
) -> PatchStatus:
    def make(applicable: bool, applied: bool, verified: bool | None, reason: str) -> PatchStatus:
        return PatchStatus(
            id=patch.id,
            target=patch.target,
            version_spec=patch.versions,
            installed_version=installed,
            applicable=applicable,
            applied=applied,
            verified=verified,
            reason=reason,
            framework=patch.framework,
            mode=patch.mode,
            description=patch.description,
        )

    applied = record is not None and record.in_place
    if record is not None and not applied:
        return make(True, False, True, "target was modified after apply")
    if disabled:
        return make(False, applied, True if applied else None, "disabled by MONORL_DISABLE_PATCHES")
    if install_error:
        return make(False, applied, None, install_error)
    if installed is None:
        return make(
            False, False, None, f"framework {patch.framework or patch.module_name!r} is not installed"
        )
    ok, reason = patch.version_applicable(installed)
    if not ok:
        return make(False, applied, True if applied else None, reason)
    if applied:
        return make(True, True, True, "")
    if not resolve:
        return make(True, False, None, "not applied")
    try:
        resolved = resolve_target(patch.target, patch_id=patch.id)
        original = _check_mode(patch, resolved)
        verified, reason = patch.check_original(original)
    except PatchError as e:
        return make(True, False, False, str(e))
    return make(True, False, verified, reason if not verified else "not applied")


def status_all(*, resolve: bool = True) -> dict[str, list[PatchStatus]]:
    return {fw: status(fw, resolve=resolve) for fw in registered_frameworks()}


def manifest() -> dict[str, Any]:
    """JSON-serialisable listing of every registered patchset (no target resolution)."""
    out: dict[str, Any] = {}
    for fw in registered_frameworks():
        sets = get_patchsets(fw)
        try:
            version = installed_version(fw)
        except PatchError:
            version = None
        with _lock:
            journal = dict(_journal.get(fw, {}))
        out[fw] = {
            "framework": fw,
            "distribution": next((ps.distribution for ps in sets if ps.distribution), None) or fw,
            "installed_version": version,
            "imported": any(ps.is_imported() for ps in sets),
            "patches": [
                {**p.to_dict(), "applied": p.id in journal and journal[p.id].in_place}
                for ps in sets
                for p in ps.patches
            ],
        }
    return out
