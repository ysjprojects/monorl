"""Patch data model: :class:`Patch`, :class:`PatchSet`, reports, target resolution, version detection.

A *target* is a dotted path ``pkg.mod.Class.attr``. It is resolved progressively: the longest
importable module prefix is imported, the remaining components are walked with ``getattr`` and
the last component is the attribute that gets replaced.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from monorl.core.errors import PatchError

PatchMode = Literal["replace", "wrap", "add"]
_MODES: tuple[str, ...] = ("replace", "wrap", "add")

DISABLE_ENV_VAR = "MONORL_DISABLE_PATCHES"


class _Missing:
    """Sentinel for "attribute absent on the owner" (distinct from ``None``)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"


MISSING: Any = _Missing()


def patches_disabled() -> bool:
    """True when ``MONORL_DISABLE_PATCHES`` is set to a truthy value (checked at call time)."""
    return os.environ.get(DISABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Patch:
    """One versioned monkeypatch.

    ``build(original)`` returns the object that replaces the target attribute. For ``replace``
    and ``wrap`` the original attribute is passed in; for ``add`` ``original`` is ``None`` and the
    target attribute must be absent unless ``overwrite=True``.
    """

    id: str
    target: str
    build: Callable[[Any], Any]
    framework: str = ""
    versions: str = ""
    mode: PatchMode = "replace"
    description: str = ""
    verify: Callable[[Any], bool] | None = None
    expected_signature: str | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise PatchError(f"patch id must be a non-empty string, got {self.id!r}")
        if not isinstance(self.target, str) or "." not in self.target.strip("."):
            raise PatchError(
                f"patch {self.id!r}: target must be a dotted path 'pkg.mod.attr', got {self.target!r}"
            )
        if self.mode not in _MODES:
            raise PatchError(f"patch {self.id!r}: mode must be one of {_MODES}, got {self.mode!r}")
        if not callable(self.build):
            raise PatchError(f"patch {self.id!r}: build must be callable")
        if self.verify is not None and not callable(self.verify):
            raise PatchError(f"patch {self.id!r}: verify must be callable or None")
        try:
            SpecifierSet(self.versions or "")
        except InvalidSpecifier as e:
            raise PatchError(f"patch {self.id!r}: invalid version specifier {self.versions!r}: {e}") from e

    @property
    def specifier(self) -> SpecifierSet:
        return SpecifierSet(self.versions or "", prereleases=True)

    @property
    def module_name(self) -> str:
        """Top-level import name derived from the target (``vllm`` for ``vllm.LLM.sleep``)."""
        return self.target.split(".", 1)[0]

    def version_applicable(self, installed: str | None) -> tuple[bool, str]:
        """Version gate. Returns ``(applicable, reason)``; ``reason`` is empty when applicable."""
        if not self.versions:
            return True, ""
        if installed is None:
            return False, "framework not installed"
        try:
            ok = Version(installed) in self.specifier
        except InvalidVersion:
            return False, f"installed version {installed!r} is not PEP 440; cannot evaluate {self.versions!r}"
        if ok:
            return True, ""
        return False, f"installed version {installed} does not satisfy {self.versions}"

    def check_original(self, original: Any) -> tuple[bool, str]:
        """Run ``expected_signature`` and ``verify`` against the resolved original.

        Returns ``(verified, reason)``. Raises :class:`PatchError` when ``verify`` itself crashes.
        """
        if self.mode == "add":
            if self.verify is not None:
                return self._run_verify(None)
            return True, ""
        if self.expected_signature is not None:
            try:
                actual = str(inspect.signature(original))
            except (TypeError, ValueError) as e:
                return False, f"cannot inspect signature of {self.target}: {e}"
            if actual.strip() != self.expected_signature.strip():
                return False, f"signature mismatch: expected {self.expected_signature!r}, found {actual!r}"
        if self.verify is not None:
            return self._run_verify(original)
        return True, ""

    def _run_verify(self, original: Any) -> tuple[bool, str]:
        try:
            ok = bool(self.verify(original))  # type: ignore[misc]
        except Exception as e:
            raise PatchError(f"patch {self.id!r}: verify() raised for target {self.target}: {e!r}") from e
        return (True, "") if ok else (False, "verify() returned False")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "framework": self.framework,
            "versions": self.versions,
            "mode": self.mode,
            "description": self.description,
            "expected_signature": self.expected_signature,
            "overwrite": self.overwrite,
        }


@dataclass
class PatchSet:
    """All patches for one upstream framework.

    ``framework`` is the *import* name; ``distribution`` is the pip name when it differs
    (``nemo-rl`` vs ``nemo_rl``). Patches may be given up front or added with the
    :meth:`patch` decorator.
    """

    framework: str
    patches: list[Patch] = field(default_factory=list)
    distribution: str | None = None

    def __post_init__(self) -> None:
        if not self.framework or not isinstance(self.framework, str):
            raise PatchError(f"PatchSet.framework must be a non-empty string, got {self.framework!r}")
        self.patches = [self._adopt(p) for p in list(self.patches)]
        self._check_unique_ids()

    def _adopt(self, patch: Patch) -> Patch:
        if patch.framework and patch.framework != self.framework:
            raise PatchError(
                f"patch {patch.id!r} declares framework {patch.framework!r} but was added to "
                f"PatchSet({self.framework!r})"
            )
        if not patch.framework:
            patch = Patch(**{**patch.__dict__, "framework": self.framework})
        return patch

    def _check_unique_ids(self) -> None:
        seen: set[str] = set()
        for p in self.patches:
            if p.id in seen:
                raise PatchError(f"PatchSet({self.framework!r}): duplicate patch id {p.id!r}")
            seen.add(p.id)

    def add(self, patch: Patch) -> Patch:
        patch = self._adopt(patch)
        if any(p.id == patch.id for p in self.patches):
            raise PatchError(f"PatchSet({self.framework!r}): duplicate patch id {patch.id!r}")
        self.patches.append(patch)
        return patch

    def patch(
        self,
        *,
        id: str,
        target: str,
        versions: str = "",
        mode: PatchMode = "replace",
        description: str = "",
        verify: Callable[[Any], bool] | None = None,
        expected_signature: str | None = None,
        overwrite: bool = False,
    ) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
        """Decorator form: the decorated function is the patch's ``build(original)``."""

        def decorator(build: Callable[[Any], Any]) -> Callable[[Any], Any]:
            self.add(
                Patch(
                    id=id,
                    target=target,
                    build=build,
                    versions=versions,
                    mode=mode,
                    description=description or (inspect.getdoc(build) or "").split("\n", 1)[0],
                    verify=verify,
                    expected_signature=expected_signature,
                    overwrite=overwrite,
                )
            )
            return build

        return decorator

    def get(self, patch_id: str) -> Patch:
        for p in self.patches:
            if p.id == patch_id:
                return p
        raise PatchError(f"PatchSet({self.framework!r}) has no patch {patch_id!r}")

    def installed_version(self) -> str | None:
        """Installed framework version, or ``None`` when the framework is not installed."""
        return detect_version(self.framework, self.distribution)

    def is_imported(self) -> bool:
        return self.framework in sys.modules


def detect_version(module_name: str, distribution: str | None = None) -> str | None:
    """``importlib.metadata.version(distribution or module_name)`` falling back to ``module.__version__``.

    The metadata lookup never imports the framework. The ``__version__`` fallback imports it; a
    missing module yields ``None``, any other import error propagates as :class:`PatchError`.
    """
    for dist in dict.fromkeys([distribution, module_name]):
        if not dist:
            continue
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name and (module_name == e.name or module_name.startswith(e.name + ".")):
            return None
        raise PatchError(f"importing {module_name!r} to detect its version failed: {e!r}") from e
    except Exception as e:  # broken install: surface, do not pretend it is absent
        raise PatchError(f"importing {module_name!r} to detect its version failed: {e!r}") from e
    version = getattr(module, "__version__", None)
    if version is None:
        return "0"  # importable but unversioned: treat as installed with the lowest version
    return str(version)


@dataclass(frozen=True)
class ResolvedTarget:
    """Owner object plus attribute name for a dotted target, with the current values."""

    owner: Any
    attr: str
    current: Any  # what ``getattr(owner, attr)`` returns, or MISSING
    raw: Any  # ``vars(owner)[attr]`` (the undecorated descriptor), or MISSING when inherited/absent

    @property
    def exists(self) -> bool:
        return self.current is not MISSING


def resolve_target(target: str, *, patch_id: str = "") -> ResolvedTarget:
    """Resolve ``pkg.mod.Class.attr``: import the longest importable module prefix, then walk attributes."""
    label = f"patch {patch_id!r}: " if patch_id else ""
    parts = [p for p in target.split(".") if p]
    if len(parts) < 2:
        raise PatchError(f"{label}target {target!r} must contain a module and an attribute")
    module = None
    split = 0
    for i in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:i])
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as e:
            if e.name and (name == e.name or name.startswith(e.name + ".")):
                continue  # this prefix is an attribute, not a module; try a shorter prefix
            raise PatchError(f"{label}importing {name!r} for target {target!r} failed: {e!r}") from e
        except Exception as e:
            raise PatchError(f"{label}importing {name!r} for target {target!r} failed: {e!r}") from e
        split = i
        break
    if module is None:
        raise PatchError(f"{label}no importable module prefix in target {target!r}")
    owner: Any = module
    for name in parts[split:-1]:
        try:
            owner = getattr(owner, name)
        except AttributeError as e:
            raise PatchError(f"{label}target {target!r}: {owner!r} has no attribute {name!r}") from e
    attr = parts[-1]
    current = getattr(owner, attr, MISSING)
    try:
        raw = vars(owner).get(attr, MISSING)
    except TypeError:  # owner without __dict__ (slots / builtins)
        raw = current
    return ResolvedTarget(owner=owner, attr=attr, current=current, raw=raw)


@dataclass(frozen=True)
class PatchStatus:
    """Status of one patch against the installed framework version."""

    id: str
    target: str
    version_spec: str
    installed_version: str | None
    applicable: bool
    applied: bool
    verified: bool | None  # None = target not inspected (framework not installed / resolve=False)
    reason: str = ""
    framework: str = ""
    mode: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "framework": self.framework,
            "mode": self.mode,
            "description": self.description,
            "version_spec": self.version_spec,
            "installed_version": self.installed_version,
            "applicable": self.applicable,
            "applied": self.applied,
            "verified": self.verified,
            "reason": self.reason,
        }


@dataclass
class ApplyReport:
    """Outcome of :func:`monorl.patching.apply` for one framework."""

    framework: str
    installed_version: str | None = None
    applied: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "installed_version": self.installed_version,
            "applied": list(self.applied),
            "skipped": [list(item) for item in self.skipped],
            "failed": [list(item) for item in self.failed],
        }

    def __str__(self) -> str:
        return (
            f"ApplyReport({self.framework}@{self.installed_version}: "
            f"{len(self.applied)} applied, {len(self.skipped)} skipped, {len(self.failed)} failed)"
        )
