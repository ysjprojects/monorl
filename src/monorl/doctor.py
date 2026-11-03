"""System / framework / patch detection report (``monorl doctor``, ``GET /api/system``).

Everything here is best-effort: a missing framework, a broken adapter import or an absent
toolchain is *reported*, never raised. :func:`detect` is the only entry point.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from monorl.core.registry import BUILTIN_PROVIDERS, NAMESPACES, registry

# framework name -> distribution name on PyPI
FRAMEWORKS: dict[str, str] = {
    "vllm": "vllm",
    "sglang": "sglang",
    "verl": "verl",
    "trl": "trl",
    "nemo_rl": "nemo-rl",
    "unsloth": "unsloth",
    "openenv": "openenv-core",
    "hermes": "hermes-agent",
    "ray": "ray",
    "peft": "peft",
    "transformers": "transformers",
}

# toolchain name -> (binary, argv, line-selector)
TOOLCHAINS: dict[str, tuple[str, list[str]]] = {
    "go": ("go", ["version"]),
    "clang++": ("clang++", ["--version"]),
    "g++": ("g++", ["--version"]),
    "cmake": ("cmake", ["--version"]),
}


@dataclass
class SystemReport:
    python: dict[str, Any]
    platform: dict[str, Any]
    torch: dict[str, Any]
    toolchains: dict[str, dict[str, Any]]
    frameworks: dict[str, dict[str, Any]]
    components: dict[str, list[str]]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _python_info() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }


def _platform_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }


def _torch_info(errors: list[str]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": False,
    }
    try:
        torch = importlib.import_module("torch")
    except Exception as e:  # noqa: BLE001 - absent or broken install is a report, not a crash
        errors.append(f"torch: {e}")
        return info
    info["installed"] = True
    info["version"] = getattr(torch, "__version__", None)
    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["cuda_device_count"] = count
            info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(count)]
    except Exception as e:  # noqa: BLE001
        errors.append(f"torch.cuda: {e}")
    try:
        mps = getattr(torch.backends, "mps", None)
        info["mps_available"] = bool(mps is not None and mps.is_available())
    except Exception as e:  # noqa: BLE001
        errors.append(f"torch.backends.mps: {e}")
    return info


def _tool_version(binary: str, args: list[str]) -> dict[str, Any]:
    path = shutil.which(binary)
    if path is None:
        return {"installed": False, "path": None, "version": None}
    try:
        proc = subprocess.run([path, *args], capture_output=True, text=True, timeout=10, check=False)
        out = (proc.stdout or proc.stderr).strip().splitlines()
        version = out[0].strip() if out else None
    except (OSError, subprocess.SubprocessError) as e:
        return {"installed": True, "path": path, "version": None, "error": str(e)}
    return {"installed": True, "path": path, "version": version}


def _toolchains() -> dict[str, dict[str, Any]]:
    return {name: _tool_version(binary, args) for name, (binary, args) in TOOLCHAINS.items()}


def _patch_statuses(framework: str, errors: list[str]) -> list[dict[str, Any]]:
    """Import the adapter (which registers its ``PatchSet``) and ask the patch engine."""
    from monorl.adapters import PROVIDERS

    if framework not in PROVIDERS:
        return []
    try:
        patching = importlib.import_module("monorl.patching")
    except Exception as e:  # noqa: BLE001 - patch engine missing while the tree is built
        errors.append(f"monorl.patching: {e}")
        return []
    try:
        importlib.import_module(PROVIDERS[framework])
    except Exception as e:  # noqa: BLE001
        errors.append(f"adapter {framework}: {e}")
    try:
        # resolve=False: version gates are computed from package metadata without importing the
        # framework, so `doctor` (and the server's /api/system) stay fast and torch-free.
        return [dict(s.__dict__) for s in patching.status(framework, resolve=False)]
    except Exception as e:  # noqa: BLE001
        errors.append(f"patches {framework}: {e}")
        return []


def _frameworks(errors: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, dist in FRAMEWORKS.items():
        try:
            version: str | None = importlib.metadata.version(dist)
            installed = True
        except importlib.metadata.PackageNotFoundError:
            version, installed = None, False
        out[name] = {
            "installed": installed,
            "version": version,
            "distribution": dist,
            "patches": _patch_statuses(name, errors),
        }
    return out


def _components(errors: list[str]) -> dict[str, list[str]]:
    from monorl.adapters import PROVIDERS

    for module_path in (*BUILTIN_PROVIDERS, *PROVIDERS.values()):
        try:
            registry.load_provider(module_path)
        except Exception as e:  # noqa: BLE001 - RegistryError or anything the provider raised
            errors.append(f"provider {module_path}: {e}")
    out: dict[str, list[str]] = {}
    for ns in NAMESPACES:
        try:
            out[ns] = registry.names(ns, load_builtins=False)
        except Exception as e:  # noqa: BLE001
            errors.append(f"registry {ns}: {e}")
            out[ns] = []
    return out


def detect() -> SystemReport:
    errors: list[str] = []
    return SystemReport(
        python=_python_info(),
        platform=_platform_info(),
        torch=_torch_info(errors),
        toolchains=_toolchains(),
        frameworks=_frameworks(errors),
        components=_components(errors),
        errors=errors,
    )


__all__ = ["FRAMEWORKS", "TOOLCHAINS", "SystemReport", "detect"]
