"""Patch engine: versioned monkeypatches for upstream frameworks.

Adapters declare a :class:`PatchSet` and :func:`register_patchset` it at import. The engine
applies patches version-gated and verified (:func:`apply`), journals originals so they can be
restored (:func:`revert`), reports (:func:`status`, :func:`manifest`) and can hook ``import`` so
patches land automatically once the framework is imported (:func:`activate_import_hook`).
``MONORL_DISABLE_PATCHES=1`` disables everything. This package never imports adapters.
"""

from __future__ import annotations

from monorl.core.errors import PatchError

from .engine import (
    apply,
    get_patchsets,
    installed_version,
    is_applied,
    is_registered,
    manifest,
    register_patchset,
    registered_frameworks,
    revert,
    revert_all,
    status,
    status_all,
    unregister_patchset,
)
from .hook import activate_import_hook, deactivate_import_hook, import_hook_active
from .patch import (
    DISABLE_ENV_VAR,
    ApplyReport,
    Patch,
    PatchMode,
    PatchSet,
    PatchStatus,
    ResolvedTarget,
    detect_version,
    patches_disabled,
    resolve_target,
)

__all__ = [
    "DISABLE_ENV_VAR",
    "ApplyReport",
    "Patch",
    "PatchError",
    "PatchMode",
    "PatchSet",
    "PatchStatus",
    "ResolvedTarget",
    "activate_import_hook",
    "apply",
    "deactivate_import_hook",
    "detect_version",
    "get_patchsets",
    "import_hook_active",
    "installed_version",
    "is_applied",
    "is_registered",
    "manifest",
    "patches_disabled",
    "register_patchset",
    "registered_frameworks",
    "resolve_target",
    "revert",
    "revert_all",
    "status",
    "status_all",
    "unregister_patchset",
]
