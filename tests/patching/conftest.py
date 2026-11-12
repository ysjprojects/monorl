"""Fixtures for the patch engine: a synthetic versioned package created per test under tmp_path."""

from __future__ import annotations

import importlib
import itertools
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from monorl.patching import deactivate_import_hook, revert_all, unregister_patchset
from monorl.patching import engine as _engine

_counter = itertools.count()

PACKAGE_SOURCE = textwrap.dedent(
    '''
    """Synthetic upstream framework used by the patch engine tests."""
    __version__ = "{version}"

    from . import sub  # noqa: F401


    def greet(name, punctuation="!"):
        return f"hello {{name}}{{punctuation}}"


    def farewell(name):
        return f"bye {{name}}"


    class Base:
        def inherited(self):
            return "base"


    class Engine(Base):
        scale = 2

        def __init__(self, factor=1):
            self.factor = factor

        def run(self, x):
            return x * self.factor

        def stop(self):
            return "stopped"

        @classmethod
        def create(cls, factor):
            return cls(factor)

        @staticmethod
        def util(v):
            return v + 1
    '''
)

SUBMODULE_SOURCE = textwrap.dedent(
    """
    def helper(a, b=1):
        return a + b
    """
)


@dataclass
class SyntheticPackage:
    name: str
    root: Path
    version: str

    @property
    def module(self):
        return importlib.import_module(self.name)


def write_synthetic_package(root: Path, name: str, version: str = "1.2.0") -> Path:
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(PACKAGE_SOURCE.format(version=version))
    (pkg / "sub.py").write_text(SUBMODULE_SOURCE)
    return pkg


@pytest.fixture
def synthetic_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SyntheticPackage]:
    name = f"synthfw{next(_counter)}"
    write_synthetic_package(tmp_path, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield SyntheticPackage(name=name, root=tmp_path, version="1.2.0")
    unregister_patchset(name)
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]


@pytest.fixture(autouse=True)
def _clean_engine_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("MONORL_DISABLE_PATCHES", raising=False)
    yield
    deactivate_import_hook()
    revert_all()
    for fw in list(_engine._registry):
        if fw.startswith("synthfw") or fw.startswith("fakefw"):
            unregister_patchset(fw)
