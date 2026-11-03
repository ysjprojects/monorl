from __future__ import annotations

import importlib
import importlib.machinery
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from monorl.patching import (
    Patch,
    PatchSet,
    activate_import_hook,
    deactivate_import_hook,
    import_hook_active,
    register_patchset,
    status,
)

from .conftest import SyntheticPackage, write_synthetic_package

SRC = Path(__file__).resolve().parents[2] / "src"


def _patchset(name: str, *, target_attr: str = "greet", ident: str = "hook.greet") -> PatchSet:
    return PatchSet(
        framework=name,
        patches=[Patch(id=ident, target=f"{name}.{target_attr}", mode="wrap", build=lambda o: (lambda *a, **k: "hooked:" + o(*a, **k)))],
    )


def test_hook_applies_after_import(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    register_patchset(_patchset(pkg.name))
    assert pkg.name not in sys.modules
    assert activate_import_hook() is True and import_hook_active()
    assert sys.meta_path[0].__class__.__name__ == "_PatchingFinder"
    mod = importlib.import_module(pkg.name)
    assert mod.greet("bob") == "hooked:hello bob!"
    # the wrapper loader leaves no trace on the module
    assert isinstance(mod.__loader__, importlib.machinery.SourceFileLoader)
    assert mod.__spec__.loader is mod.__loader__
    (s,) = status(pkg.name)
    assert s.applied and s.verified


def test_hook_is_idempotent_and_deactivates(synthetic_package: SyntheticPackage, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = synthetic_package
    register_patchset(_patchset(pkg.name))
    assert activate_import_hook() and activate_import_hook()
    assert sum(type(f).__name__ == "_PatchingFinder" for f in sys.meta_path) == 1
    assert deactivate_import_hook() is True
    assert deactivate_import_hook() is False
    assert not import_hook_active()
    assert all(type(f).__name__ != "_PatchingFinder" for f in sys.meta_path)
    mod = importlib.import_module(pkg.name)
    assert mod.greet("bob") == "hello bob!"  # registered but hook off → untouched
    assert not status(pkg.name)[0].applied


def test_hook_applies_immediately_when_already_imported(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    mod = importlib.import_module(pkg.name)
    register_patchset(_patchset(pkg.name))
    assert mod.greet("a") == "hello a!"
    activate_import_hook()
    assert mod.greet("a") == "hooked:hello a!"


def test_registering_after_activation_applies_if_imported(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    activate_import_hook()
    mod = importlib.import_module(pkg.name)  # not registered yet: passes through the hook untouched
    assert mod.greet("a") == "hello a!"
    register_patchset(_patchset(pkg.name))
    assert mod.greet("a") == "hooked:hello a!"


def test_hook_leaves_unrelated_imports_alone(synthetic_package: SyntheticPackage, tmp_path: Path) -> None:
    pkg = synthetic_package
    register_patchset(_patchset(pkg.name))
    other = f"{pkg.name}_unrelated"
    write_synthetic_package(tmp_path, other, version="0.1")
    activate_import_hook()
    try:
        mod = importlib.import_module(other)
        assert mod.greet("a") == "hello a!"
        assert isinstance(mod.__loader__, importlib.machinery.SourceFileLoader)
        sub = importlib.import_module(f"{pkg.name}.sub")  # submodule import of a registered framework
        assert sub.helper(1) == 2
        assert sys.modules[pkg.name].greet("a") == "hooked:hello a!"
    finally:
        for key in [k for k in sys.modules if k == other or k.startswith(other + ".")]:
            del sys.modules[key]


def test_hook_reentrant_patch_importing_submodules(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package

    def build(original):
        sub = importlib.import_module(f"{pkg.name}.sub")  # import inside apply inside exec_module
        return lambda a, b=1: sub.helper(a, b) * 1000

    register_patchset(
        PatchSet(
            framework=pkg.name,
            patches=[
                Patch(id="re.helper", target=f"{pkg.name}.sub.helper", build=lambda o: (lambda a, b=1: o(a, b) + 1)),
                Patch(id="re.greet", target=f"{pkg.name}.greet", build=build),
            ],
        )
    )
    activate_import_hook()
    mod = importlib.import_module(pkg.name)
    assert mod.sub.helper(1, 1) == 3
    assert mod.greet(1, 1) == 3000
    assert [s.applied for s in status(pkg.name)] == [True, True]


def test_failed_patch_does_not_break_import(synthetic_package: SyntheticPackage, caplog: pytest.LogCaptureFixture) -> None:
    pkg = synthetic_package
    register_patchset(
        PatchSet(framework=pkg.name, patches=[Patch(id="bad.target", target=f"{pkg.name}.Engine.absent", build=lambda o: o)])
    )
    activate_import_hook()
    with caplog.at_level("WARNING", logger="monorl.patching"):
        mod = importlib.import_module(pkg.name)
    assert mod.greet("a") == "hello a!"
    assert any("bad.target" in r.getMessage() for r in caplog.records)
    (s,) = status(pkg.name)
    assert s.applied is False and s.verified is False and "absent" in s.reason


def test_concurrent_import_and_apply_do_not_deadlock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import time

    from monorl.patching import apply

    name = "synthfw_slow"
    pkg = write_synthetic_package(tmp_path, name)
    init = pkg / "__init__.py"
    init.write_text("import time\ntime.sleep(0.5)\n" + init.read_text())
    monkeypatch.syspath_prepend(str(tmp_path))
    register_patchset(_patchset(name))
    activate_import_hook()
    results: dict[str, object] = {}

    def importer() -> None:
        results["mod"] = importlib.import_module(name)

    def applier() -> None:
        time.sleep(0.1)  # start while the import above is still executing the module body
        results["report"] = apply(name)

    threads = [threading.Thread(target=importer), threading.Thread(target=applier)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert not any(t.is_alive() for t in threads), "deadlock between import hook and apply()"
        assert results["mod"].greet("a") == "hooked:hello a!"
        report = results["report"]
        assert report.applied + [pid for pid, _ in report.skipped] == ["hook.greet"]
        assert status(name)[0].applied
    finally:
        from monorl.patching import unregister_patchset

        unregister_patchset(name)
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]


def test_hook_disabled_by_env(synthetic_package: SyntheticPackage, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = synthetic_package
    register_patchset(_patchset(pkg.name))
    monkeypatch.setenv("MONORL_DISABLE_PATCHES", "true")
    assert activate_import_hook() is False and not import_hook_active()
    mod = importlib.import_module(pkg.name)
    assert mod.greet("a") == "hello a!"


# ----------------------------------------------------------------------------- fresh process


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import importlib.machinery, json, sys
    sys.path.insert(0, {src!r})
    sys.path.insert(0, {root!r})
    from monorl.patching import Patch, PatchSet, activate_import_hook, register_patchset, status, revert

    name = {name!r}
    register_patchset(PatchSet(framework=name, patches=[
        Patch(id="sp.greet", target=f"{{name}}.greet", versions=">=1,<2", mode="wrap",
              build=lambda o: (lambda n, punctuation="!": "sub:" + o(n, punctuation))),
        Patch(id="sp.run", target=f"{{name}}.Engine.run", mode="replace", expected_signature="(self, x)",
              build=lambda o: (lambda self, x: -o(self, x))),
        Patch(id="sp.future", target=f"{{name}}.farewell", versions=">=2", build=lambda o: o),
    ]))
    active = activate_import_hook()
    assert name not in sys.modules
    import json as _json  # unrelated import while the hook is active
    mod = importlib.import_module(name)
    out = {{
        "active": active,
        "greet": mod.greet("bob"),
        "run": mod.Engine(3).run(2),
        "loader": type(mod.__loader__).__name__,
        "status": [(s.id, s.applied, s.applicable, s.reason) for s in status(name)],
        "reverted": revert(name),
        "greet_after_revert": mod.greet("bob"),
    }}
    print(_json.dumps(out))
    """
)


def _run_fresh(tmp_path: Path, name: str, env_extra: dict[str, str] | None = None) -> dict:
    import os

    env = {k: v for k, v in os.environ.items() if k != "MONORL_DISABLE_PATCHES"}
    env.update(env_extra or {})
    script = _SUBPROCESS_SCRIPT.format(src=str(SRC), root=str(tmp_path), name=name)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_import_hook_in_fresh_subprocess(tmp_path: Path) -> None:
    name = "synthfw_subproc"
    write_synthetic_package(tmp_path, name)
    out = _run_fresh(tmp_path, name)
    assert out["active"] is True
    assert out["greet"] == "sub:hello bob!"
    assert out["run"] == -6
    assert out["loader"] == "SourceFileLoader"
    assert out["status"] == [
        ["sp.greet", True, True, ""],
        ["sp.run", True, True, ""],
        ["sp.future", False, False, "installed version 1.2.0 does not satisfy >=2"],
    ]
    assert sorted(out["reverted"]) == ["sp.greet", "sp.run"]
    assert out["greet_after_revert"] == "hello bob!"


def test_import_hook_disabled_in_fresh_subprocess(tmp_path: Path) -> None:
    name = "synthfw_subproc_disabled"
    write_synthetic_package(tmp_path, name)
    out = _run_fresh(tmp_path, name, {"MONORL_DISABLE_PATCHES": "1"})
    assert out["active"] is False
    assert out["greet"] == "hello bob!" and out["run"] == 6
    assert all(applied is False and applicable is False for _, applied, applicable, _ in out["status"])
    assert {reason for *_, reason in out["status"]} == {"disabled by MONORL_DISABLE_PATCHES"}
    assert out["reverted"] == []
