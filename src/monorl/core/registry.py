"""Global component registry with lazy provider loading.

Namespaces: ``rollout_engine``, ``trainer``, ``pipeline_backend``, ``reward``, ``env``,
``data_source``, ``weight_sync``.

Providers register factories by exposing ``register(registry)``. Built-in providers (listed in
``BUILTIN_PROVIDERS``) are imported on the first lookup miss; adapter providers (see
:mod:`monorl.adapters`) are imported only when a name they own is requested, so importing
monorl never imports vllm/sglang/ray/etc. A run config names its own provider modules in
``RunConfig.providers``; :func:`ensure_providers` loads them in every process that builds
components from that config.

Usage::

    from monorl.core.registry import registry

    @registry.provides("reward", "exact_match")
    def make_exact_match(params, **kw): ...

    factory = registry.get("reward", "exact_match")
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from collections.abc import Callable, Iterable
from typing import Any

from monorl.core.errors import RegistryError

NAMESPACES: tuple[str, ...] = (
    "rollout_engine",
    "trainer",
    "pipeline_backend",
    "reward",
    "env",
    "data_source",
    "weight_sync",
)

BUILTIN_PROVIDERS: tuple[str, ...] = (
    "monorl.data",
    "monorl.rewards",
    "monorl.envs",
    "monorl.rollout",
    "monorl.trainer",
    "monorl.distributed",
)

Factory = Callable[..., Any]


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, Factory]] = {ns: {} for ns in NAMESPACES}
        self._loaded_providers: set[str] = set()
        self._lock = threading.RLock()

    # -- registration -------------------------------------------------------------------
    def register(self, namespace: str, name: str, factory: Factory, *, overwrite: bool = False) -> None:
        self._check_ns(namespace)
        with self._lock:
            if name in self._items[namespace] and not overwrite:
                existing = self._items[namespace][name]
                if existing is factory:
                    return
                raise RegistryError(f"{namespace}:{name} already registered ({existing!r})")
            self._items[namespace][name] = factory

    def provides(self, namespace: str, name: str, *, overwrite: bool = False) -> Callable[[Factory], Factory]:
        def deco(factory: Factory) -> Factory:
            self.register(namespace, name, factory, overwrite=overwrite)
            return factory

        return deco

    # -- lookup -------------------------------------------------------------------------
    def get(self, namespace: str, name: str) -> Factory:
        self._check_ns(namespace)
        with self._lock:
            factory = self._items[namespace].get(name)
            if factory is None:
                self._load_builtins()
                factory = self._items[namespace].get(name)
            if factory is None:
                self._load_adapter_for(name)
                factory = self._items[namespace].get(name)
            if factory is None:
                raise RegistryError(
                    f"No {namespace} named {name!r}. Available: {sorted(self._items[namespace])}"
                )
            return factory

    def contains(self, namespace: str, name: str) -> bool:
        try:
            self.get(namespace, name)
            return True
        except RegistryError:
            return False

    def names(self, namespace: str, *, load_builtins: bool = True) -> list[str]:
        self._check_ns(namespace)
        with self._lock:
            if load_builtins:
                self._load_builtins()
            return sorted(self._items[namespace])

    def load_provider(self, module_path: str) -> None:
        """Import ``module_path`` and call its ``register(registry)`` once. A provider that fails is
        not marked as loaded, so the next call tries it again."""
        with self._lock:
            if module_path in self._loaded_providers:
                return
            self._loaded_providers.add(module_path)
            try:
                try:
                    mod = importlib.import_module(module_path)
                except ImportError as e:
                    raise RegistryError(f"provider {module_path} failed to import: {e}") from e
                reg = getattr(mod, "register", None)
                if reg is None:
                    raise RegistryError(f"provider {module_path} has no register(registry)")
                reg(self)
            except BaseException:
                self._loaded_providers.discard(module_path)
                raise

    def load_providers(self, modules: Iterable[str]) -> None:
        """Load the provider modules of a run config (``RunConfig.providers``), each once per process.

        The working directory is put on ``sys.path`` first, as ``python -m`` does, so that a module
        next to where the run was started is found. Whatever a module raises while it is imported or
        registers comes back as a :class:`RegistryError` that names the module."""
        with self._lock:
            pending = [m for m in modules if m not in self._loaded_providers]
            if not pending:
                return
            cwd = os.getcwd()
            if cwd not in sys.path:
                sys.path.insert(0, cwd)
            for module_path in pending:
                try:
                    self.load_provider(module_path)
                except RegistryError:
                    raise
                except Exception as e:
                    raise RegistryError(f"provider {module_path} failed: {type(e).__name__}: {e}") from e

    # -- internals ----------------------------------------------------------------------
    def _check_ns(self, namespace: str) -> None:
        if namespace not in self._items:
            raise RegistryError(f"unknown namespace {namespace!r}; expected one of {NAMESPACES}")

    def _load_builtins(self) -> None:
        for mod in BUILTIN_PROVIDERS:
            if mod in self._loaded_providers:
                continue
            try:
                self.load_provider(mod)
            except RegistryError:
                # A builtin provider may be absent while the tree is under construction.
                continue

    def _load_adapter_for(self, name: str) -> None:
        from monorl.adapters import provider_for_name

        module_path = provider_for_name(name)
        if module_path is not None:
            self.load_provider(module_path)


registry = Registry()


def ensure_providers(config: Any) -> None:
    """Load ``config.providers`` into this process's registry. Idempotent and cheap: called
    wherever a process turns a run config into components (the run, the runtime, rewards,
    environments), so the framework processes of pipeline backends are covered as well."""
    modules = getattr(config, "providers", None)
    if modules:
        registry.load_providers(modules)
