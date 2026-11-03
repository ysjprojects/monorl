"""Fixtures for the sglang adapter tests. The fake module tree lives in ``fake_sglang.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fake_sglang import Engine, install_fake_sglang, make_config, remove_fake_sglang  # noqa: E402


def real_sglang_installed() -> bool:
    mod = sys.modules.get("sglang")
    if mod is not None:
        return bool(getattr(mod, "__file__", None))
    try:
        import importlib.util

        return importlib.util.find_spec("sglang") is not None
    except (ImportError, ValueError):
        return False


@pytest.fixture
def fake_sglang():
    """Install the fake tree for one test; patches applied meanwhile are reverted afterwards."""
    if real_sglang_installed():
        pytest.skip("real sglang is installed; fake-module tests do not apply")
    from monorl.patching import is_registered, revert

    Engine.instances.clear()
    mods = install_fake_sglang()
    try:
        yield mods
    finally:
        if is_registered("sglang"):
            revert("sglang")
        remove_fake_sglang()
        Engine.instances.clear()


@pytest.fixture
def no_sglang():
    """Guarantee sglang is absent from ``sys.modules`` (skips when the real package exists)."""
    if real_sglang_installed():
        pytest.skip("real sglang is installed")
    remove_fake_sglang()
    yield
    remove_fake_sglang()


@pytest.fixture
def sglang_config(run_config_dict):
    return make_config(run_config_dict)
