"""Provider modules of a run config (``RunConfig.providers``) and the registry."""

from __future__ import annotations

import sys
import textwrap
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from monorl.core.errors import RegistryError
from monorl.core.registry import Registry


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], str]]:
    """A working directory that is not on ``sys.path``; the returned function writes a module into it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", ".", str(tmp_path))])
    written: list[str] = []

    def write(body: str, name: str | None = None) -> str:
        name = name or f"provider_{uuid.uuid4().hex[:10]}"
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
        written.append(name)
        return name

    yield write
    for name in written:
        sys.modules.pop(name, None)


COUNTING = """
    CALLS = []

    def register(registry):
        CALLS.append(1)
        registry.register("reward", "from_provider", lambda params, **kw: ("reward", params))
        registry.register("env", "provider_env", lambda config, **kw: ("env", config))
"""


def test_a_module_in_the_working_directory_registers_its_components(project: Callable[..., str]) -> None:
    name = project(COUNTING)
    registry = Registry()
    registry.load_providers([name])
    assert registry.get("reward", "from_provider")({"k": 1}) == ("reward", {"k": 1})
    assert registry.get("env", "provider_env")("cfg") == ("env", "cfg")
    # every process entry asks again: the module registers once
    registry.load_providers([name])
    registry.load_providers([name, name])
    assert sys.modules[name].CALLS == [1]


def test_a_module_without_register_is_named_in_the_error(project: Callable[..., str]) -> None:
    name = project("VALUE = 1\n")
    with pytest.raises(RegistryError, match=f"provider {name} has no register"):
        Registry().load_providers([name])


def test_what_a_provider_raises_comes_back_with_its_name(project: Callable[..., str]) -> None:
    broken = project("raise RuntimeError('needs a GPU')\n")
    with pytest.raises(RegistryError, match=f"provider {broken} failed: RuntimeError: needs a GPU"):
        Registry().load_providers([broken])
    clash = project(
        """
        def register(registry):
            registry.register("reward", "twice", lambda params, **kw: 1)
            registry.register("reward", "twice", lambda params, **kw: 2)
        """
    )
    with pytest.raises(RegistryError, match="reward:twice already registered"):
        Registry().load_providers([clash])


def test_a_missing_module_can_be_loaded_once_it_exists(project: Callable[..., str]) -> None:
    name = f"provider_{uuid.uuid4().hex[:10]}"
    registry = Registry()
    with pytest.raises(RegistryError, match=f"provider {name} failed to import"):
        registry.load_providers([name])
    project(COUNTING, name=name)
    registry.load_providers([name])
    assert registry.get("reward", "from_provider")({}) == ("reward", {})
