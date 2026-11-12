from __future__ import annotations

import functools
import importlib
import importlib.metadata

import pytest

from monorl.core.errors import PatchError
from monorl.patching import (
    ApplyReport,
    Patch,
    PatchSet,
    PatchStatus,
    apply,
    detect_version,
    installed_version,
    is_applied,
    manifest,
    register_patchset,
    registered_frameworks,
    resolve_target,
    revert,
    status,
    status_all,
    unregister_patchset,
)

from .conftest import SyntheticPackage


def _wrap_build(original):
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        return "wrapped:" + str(original(*args, **kwargs))

    return wrapper


def _register(pkg: SyntheticPackage, *patches: Patch, distribution: str | None = None) -> PatchSet:
    return register_patchset(PatchSet(framework=pkg.name, patches=list(patches), distribution=distribution))


# ------------------------------------------------------------------------------------------ modes


def test_replace_module_function(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(
        pkg,
        Patch(
            id="p.greet",
            target=f"{pkg.name}.greet",
            build=lambda o: lambda name, punctuation="!": "hi " + name,
        ),
    )
    report = apply(pkg.name)
    assert isinstance(report, ApplyReport)
    assert report.applied == ["p.greet"] and report.ok
    assert report.installed_version == "1.2.0"
    assert pkg.module.greet("bob") == "hi bob"
    assert pkg.module.greet.__monorl_patch__ == "p.greet"


def test_wrap_module_function_receives_original(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(pkg, Patch(id="p.wrap", target=f"{pkg.name}.farewell", mode="wrap", build=_wrap_build))
    assert apply(pkg.name).applied == ["p.wrap"]
    assert pkg.module.farewell("bob") == "wrapped:bye bob"
    assert pkg.module.farewell.__wrapped__("bob") == "bye bob"


def test_wrap_sets_wrapped_when_build_does_not(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(
        pkg,
        Patch(
            id="p.wrap", target=f"{pkg.name}.farewell", mode="wrap", build=lambda o: lambda n: o(n).upper()
        ),
    )
    apply(pkg.name)
    assert pkg.module.farewell("x") == "BYE X"
    assert pkg.module.farewell.__wrapped__.__name__ == "farewell"


def test_wrap_requires_callable_target_and_result(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(
        pkg,
        Patch(id="p.notcallable", target=f"{pkg.name}.__version__", mode="wrap", build=lambda o: o),
        Patch(id="p.badresult", target=f"{pkg.name}.greet", mode="wrap", build=lambda o: 42),
    )
    report = apply(pkg.name)
    assert report.applied == []
    failed = dict(report.failed)
    assert "not callable" in failed["p.notcallable"] and f"{pkg.name}.__version__" in failed["p.notcallable"]
    assert "must return a callable" in failed["p.badresult"]
    assert pkg.module.greet("a") == "hello a!"


def test_add_module_attribute(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    seen = []

    def build(original):
        seen.append(original)
        return lambda: "new"

    _register(pkg, Patch(id="p.add", target=f"{pkg.name}.brand_new", mode="add", build=build))
    assert apply(pkg.name).applied == ["p.add"]
    assert seen == [None]
    assert pkg.module.brand_new() == "new"
    revert(pkg.name)
    assert not hasattr(pkg.module, "brand_new")


def test_add_refuses_existing_unless_overwrite(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(
        pkg,
        Patch(id="p.add.clash", target=f"{pkg.name}.greet", mode="add", build=lambda o: lambda n: "x"),
        Patch(
            id="p.add.force",
            target=f"{pkg.name}.farewell",
            mode="add",
            overwrite=True,
            build=lambda o: lambda n: "y",
        ),
    )
    report = apply(pkg.name)
    assert report.applied == ["p.add.force"]
    ((pid, err),) = report.failed
    assert (
        pid == "p.add.clash"
        and "already exists" in err
        and f"{pkg.name}.greet" in err
        and "p.add.clash" in err
    )
    assert pkg.module.farewell("a") == "y"
    revert(pkg.name)
    assert pkg.module.farewell("a") == "bye a"


def test_replace_and_wrap_class_methods(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package

    def wrap_run(original):
        def run(self, x):
            return original(self, x) + 100

        return run

    _register(
        pkg,
        Patch(id="p.run", target=f"{pkg.name}.Engine.run", mode="wrap", build=wrap_run),
        Patch(
            id="p.stop",
            target=f"{pkg.name}.Engine.stop",
            mode="replace",
            build=lambda o: lambda self: "halted",
        ),
        Patch(
            id="p.extra",
            target=f"{pkg.name}.Engine.extra",
            mode="add",
            build=lambda o: lambda self: self.factor * 10,
        ),
    )
    report = apply(pkg.name)
    assert sorted(report.applied) == ["p.extra", "p.run", "p.stop"]
    eng = pkg.module.Engine(3)
    assert eng.run(2) == 106
    assert eng.stop() == "halted"
    assert eng.extra() == 30
    revert(pkg.name)
    assert eng.run(2) == 6 and eng.stop() == "stopped" and not hasattr(eng, "extra")


def test_replace_inherited_method_and_revert_restores_inheritance(
    synthetic_package: SyntheticPackage,
) -> None:
    pkg = synthetic_package
    _register(
        pkg, Patch(id="p.inh", target=f"{pkg.name}.Engine.inherited", build=lambda o: lambda self: "override")
    )
    apply(pkg.name)
    assert pkg.module.Engine().inherited() == "override"
    assert "inherited" in vars(pkg.module.Engine)
    revert(pkg.name)
    assert "inherited" not in vars(pkg.module.Engine)
    assert pkg.module.Engine().inherited() == "base"


def test_classmethod_and_staticmethod_targets(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package

    def build_create(original):  # original is the bound classmethod
        assert original(5).factor == 5
        return classmethod(lambda cls, factor: cls(factor * 2))

    _register(
        pkg,
        Patch(id="p.cm", target=f"{pkg.name}.Engine.create", build=build_create),
        Patch(id="p.sm", target=f"{pkg.name}.Engine.util", build=lambda o: staticmethod(lambda v: o(v) * 10)),
    )
    assert apply(pkg.name).ok
    assert pkg.module.Engine.create(4).factor == 8
    assert pkg.module.Engine.util(1) == 20
    revert(pkg.name)
    assert pkg.module.Engine.create(4).factor == 4
    assert isinstance(vars(pkg.module.Engine)["create"], classmethod)
    assert isinstance(vars(pkg.module.Engine)["util"], staticmethod)


def test_submodule_target_resolution(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(
        pkg,
        Patch(id="p.helper", target=f"{pkg.name}.sub.helper", build=lambda o: lambda a, b=1: o(a, b) * 100),
    )
    assert apply(pkg.name).applied == ["p.helper"]
    sub = importlib.import_module(f"{pkg.name}.sub")
    assert sub.helper(1, 2) == 300
    revert(pkg.name)
    assert sub.helper(1, 2) == 3


def test_resolve_target_errors_name_patch_and_target(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    with pytest.raises(
        PatchError, match=rf"patch 'x'.*{pkg.name}\.Engine\.nope\.deeper.*no attribute 'nope'"
    ):
        resolve_target(f"{pkg.name}.Engine.nope.deeper", patch_id="x")
    with pytest.raises(PatchError, match="no importable module prefix"):
        resolve_target("definitely_not_a_module_xyz.attr")
    r = resolve_target(f"{pkg.name}.Engine.run")
    assert r.owner is pkg.module.Engine and r.attr == "run" and r.exists


def test_missing_target_is_failed_with_clear_message(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(pkg, Patch(id="p.missing", target=f"{pkg.name}.Engine.nothing_here", build=lambda o: o))
    report = apply(pkg.name)
    ((pid, err),) = report.failed
    assert pid == "p.missing"
    assert "p.missing" in err and f"{pkg.name}.Engine.nothing_here" in err and "does not exist" in err


def test_build_exception_is_failed_not_raised(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package

    def boom(original):
        raise RuntimeError("kaboom")

    _register(pkg, Patch(id="p.boom", target=f"{pkg.name}.greet", build=boom))
    report = apply(pkg.name)
    assert report.failed == [
        ("p.boom", f"patch 'p.boom': build() raised for target {pkg.name}.greet: RuntimeError: kaboom")
    ]
    assert pkg.module.greet("a") == "hello a!"


# ------------------------------------------------------------------------------------- gating


def test_version_gating(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(
        pkg,
        Patch(id="p.old", target=f"{pkg.name}.greet", versions=">=2.0", build=lambda o: o),
        Patch(id="p.ok", target=f"{pkg.name}.farewell", versions=">=1.0,<2", build=lambda o: lambda n: "ok"),
        Patch(id="p.any", target=f"{pkg.name}.Engine.stop", build=lambda o: lambda self: "any"),
    )
    report = apply(pkg.name)
    assert sorted(report.applied) == ["p.any", "p.ok"]
    assert report.skipped == [("p.old", "installed version 1.2.0 does not satisfy >=2.0")]
    by_id = {s.id: s for s in status(pkg.name)}
    assert by_id["p.old"].applicable is False and by_id["p.old"].applied is False
    assert by_id["p.old"].reason == "installed version 1.2.0 does not satisfy >=2.0"
    assert by_id["p.old"].version_spec == ">=2.0" and by_id["p.old"].installed_version == "1.2.0"
    assert by_id["p.ok"].applicable and by_id["p.ok"].applied and by_id["p.ok"].verified


def test_invalid_specifier_rejected_at_construction() -> None:
    with pytest.raises(PatchError, match="invalid version specifier"):
        Patch(id="bad", target="a.b", versions="banana", build=lambda o: o)
    with pytest.raises(PatchError, match="mode must be one of"):
        Patch(id="bad", target="a.b", mode="delete", build=lambda o: o)  # type: ignore[arg-type]
    with pytest.raises(PatchError, match="dotted path"):
        Patch(id="bad", target="nodots", build=lambda o: o)


def test_prerelease_versions_match_specifier() -> None:
    p = Patch(id="pre", target="a.b", versions=">=0.9", build=lambda o: o)
    assert p.version_applicable("0.9.1rc1") == (True, "")
    assert p.version_applicable("0.9.1.dev0+gabc") == (True, "")
    assert p.version_applicable("not-a-version")[0] is False
    assert "not PEP 440" in p.version_applicable("not-a-version")[1]


def test_verify_failure_skips(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(
        pkg,
        Patch(id="p.v.no", target=f"{pkg.name}.greet", verify=lambda o: False, build=lambda o: lambda n: "x"),
        Patch(
            id="p.v.yes",
            target=f"{pkg.name}.farewell",
            verify=lambda o: o("a") == "bye a",
            build=lambda o: lambda n: "y",
        ),
    )
    report = apply(pkg.name)
    assert report.applied == ["p.v.yes"]
    assert report.skipped == [("p.v.no", "verify() returned False")]
    assert pkg.module.greet("a") == "hello a!"
    s = {x.id: x for x in status(pkg.name)}
    assert (
        s["p.v.no"].applicable
        and s["p.v.no"].verified is False
        and s["p.v.no"].reason == "verify() returned False"
    )


def test_verify_raising_is_failed(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package

    def bad_verify(original):
        raise ValueError("inspect failed")

    _register(pkg, Patch(id="p.v.raise", target=f"{pkg.name}.greet", verify=bad_verify, build=lambda o: o))
    report = apply(pkg.name)
    ((pid, err),) = report.failed
    assert pid == "p.v.raise" and "verify() raised" in err and "inspect failed" in err


def test_expected_signature_gate(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(
        pkg,
        Patch(
            id="p.sig.ok",
            target=f"{pkg.name}.greet",
            expected_signature="(name, punctuation='!')",
            build=lambda o: lambda *a: "ok",
        ),
        Patch(
            id="p.sig.bad",
            target=f"{pkg.name}.farewell",
            expected_signature="(name, extra)",
            build=lambda o: lambda *a: "bad",
        ),
        Patch(
            id="p.sig.method",
            target=f"{pkg.name}.Engine.run",
            expected_signature="(self, x)",
            build=lambda o: lambda self, x: 0,
        ),
    )
    report = apply(pkg.name)
    assert sorted(report.applied) == ["p.sig.method", "p.sig.ok"]
    ((pid, reason),) = report.skipped
    assert pid == "p.sig.bad"
    assert reason == "signature mismatch: expected '(name, extra)', found '(name)'"


def test_apply_is_idempotent_and_force_reapplies(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    calls = []

    def build(original):
        calls.append(original)
        return lambda n: f"built{len(calls)}"

    _register(pkg, Patch(id="p.idem", target=f"{pkg.name}.greet", build=build))
    original = pkg.module.greet
    assert apply(pkg.name).applied == ["p.idem"]
    second = apply(pkg.name)
    assert second.applied == [] and second.skipped == [("p.idem", "already applied")]
    assert len(calls) == 1 and pkg.module.greet("a") == "built1"

    forced = apply(pkg.name, force=True)
    assert forced.applied == ["p.idem"]
    assert calls == [
        original,
        original,
    ]  # force re-builds from the pristine original, never from the patched one
    assert pkg.module.greet("a") == "built2"
    assert revert(pkg.name) == ["p.idem"]
    assert pkg.module.greet is original


def test_force_bypasses_version_and_verify_gates(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(
        pkg,
        Patch(id="p.f.ver", target=f"{pkg.name}.greet", versions=">=9", build=lambda o: lambda n: "forced"),
        Patch(
            id="p.f.verify",
            target=f"{pkg.name}.farewell",
            verify=lambda o: False,
            build=lambda o: lambda n: "forced",
        ),
        Patch(id="p.f.missing", target=f"{pkg.name}.nothing", build=lambda o: o),
    )
    assert apply(pkg.name).applied == []
    forced = apply(pkg.name, force=True)
    assert sorted(forced.applied) == ["p.f.ver", "p.f.verify"]
    assert [pid for pid, _ in forced.failed] == ["p.f.missing"]  # force never invents targets
    assert pkg.module.greet("a") == "forced"


def test_revert_restores_originals_in_reverse_order(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    original_greet, original_run = pkg.module.greet, vars(pkg.module.Engine)["run"]
    _register(
        pkg,
        Patch(id="p.a", target=f"{pkg.name}.greet", build=lambda o: lambda n: "a"),
        Patch(id="p.b", target=f"{pkg.name}.Engine.run", build=lambda o: lambda self, x: "b"),
        Patch(id="p.c", target=f"{pkg.name}.Engine.added", mode="add", build=lambda o: lambda self: "c"),
    )
    apply(pkg.name)
    assert is_applied("p.a") and is_applied("p.b", pkg.name)
    assert revert(pkg.name) == ["p.c", "p.b", "p.a"]
    assert pkg.module.greet is original_greet
    assert vars(pkg.module.Engine)["run"] is original_run
    assert not hasattr(pkg.module.Engine, "added")
    assert revert(pkg.name) == []
    assert all(not s.applied for s in status(pkg.name))


def test_target_modified_after_apply_is_reported_and_reapplied(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    original = pkg.module.greet
    _register(pkg, Patch(id="p.mod", target=f"{pkg.name}.greet", build=lambda o: lambda n: "patched"))
    apply(pkg.name)
    pkg.module.greet = lambda n: "someone else"  # third party clobbers our patch
    (s,) = status(pkg.name)
    assert s.applied is False and s.reason == "target was modified after apply"
    assert apply(pkg.name).applied == ["p.mod"]
    assert pkg.module.greet("a") == "patched"
    revert(pkg.name)
    assert pkg.module.greet is original  # pristine original, not the clobbering lambda


# ---------------------------------------------------------------------------- registry/status


def test_status_for_unregistered_framework_is_empty_and_apply_raises() -> None:
    assert status("no_such_framework_zzz") == []
    with pytest.raises(PatchError, match="no patchset registered"):
        apply("no_such_framework_zzz")


def test_status_not_installed_framework() -> None:
    register_patchset(
        PatchSet(
            framework="fakefw_missing",
            patches=[Patch(id="p.x", target="fakefw_missing.f", build=lambda o: o)],
        )
    )
    (s,) = status("fakefw_missing")
    assert isinstance(s, PatchStatus)
    assert s.installed_version is None and s.applicable is False and s.verified is None
    assert "not installed" in s.reason
    report = apply("fakefw_missing")
    assert report.applied == [] and report.skipped == [("p.x", "framework 'fakefw_missing' is not installed")]


def test_status_fields_and_resolve_false(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(
        pkg,
        Patch(
            id="p.s1",
            target=f"{pkg.name}.greet",
            versions=">=1",
            mode="wrap",
            description="greeting",
            build=_wrap_build,
        ),
        Patch(id="p.s2", target=f"{pkg.name}.Engine.missing", build=lambda o: o),
    )
    s1, s2 = status(pkg.name, resolve=False)
    assert s1.to_dict() == {
        "id": "p.s1",
        "target": f"{pkg.name}.greet",
        "framework": pkg.name,
        "mode": "wrap",
        "description": "greeting",
        "version_spec": ">=1",
        "installed_version": "1.2.0",
        "applicable": True,
        "applied": False,
        "verified": None,
        "reason": "not applied",
    }
    s1, s2 = status(pkg.name)
    assert s1.verified is True and s1.applied is False
    assert s2.verified is False and "does not exist" in s2.reason
    apply(pkg.name)
    s1, s2 = status(pkg.name)
    assert s1.applied and s1.verified and s1.reason == ""
    assert not s2.applied


def test_status_all_and_manifest(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    _register(pkg, Patch(id="p.m", target=f"{pkg.name}.greet", description="d", build=lambda o: o))
    register_patchset(
        PatchSet(
            framework="fakefw_other",
            patches=[Patch(id="p.o", target="fakefw_other.x", build=lambda o: o)],
            distribution="fake-fw-other",
        )
    )
    assert {pkg.name, "fakefw_other"} <= set(registered_frameworks())
    everything = status_all()
    assert set(everything) >= {pkg.name, "fakefw_other"}
    apply(pkg.name)
    m = manifest()
    entry = m[pkg.name]
    assert (
        entry["installed_version"] == "1.2.0"
        and entry["imported"] is True
        and entry["distribution"] == pkg.name
    )
    assert entry["patches"] == [
        {
            "id": "p.m",
            "target": f"{pkg.name}.greet",
            "framework": pkg.name,
            "versions": "",
            "mode": "replace",
            "description": "d",
            "expected_signature": None,
            "overwrite": False,
            "applied": True,
        }
    ]
    other = m["fakefw_other"]
    assert (
        other["installed_version"] is None
        and other["distribution"] == "fake-fw-other"
        and other["patches"][0]["applied"] is False
    )
    unregister_patchset("fakefw_other")
    assert "fakefw_other" not in registered_frameworks()


def test_reregistration_replaces_overlapping_set_and_rejects_duplicate_ids(
    synthetic_package: SyntheticPackage,
) -> None:
    pkg = synthetic_package
    first = _register(pkg, Patch(id="p.one", target=f"{pkg.name}.greet", build=lambda o: lambda n: "v1"))
    again = _register(pkg, Patch(id="p.one", target=f"{pkg.name}.greet", build=lambda o: lambda n: "v2"))
    extra = _register(pkg, Patch(id="p.two", target=f"{pkg.name}.farewell", build=lambda o: lambda n: "x"))
    from monorl.patching import get_patchsets

    assert get_patchsets(pkg.name) == [again, extra] and first not in get_patchsets(pkg.name)
    with pytest.raises(PatchError, match="duplicate patch id"):
        PatchSet(
            framework=pkg.name,
            patches=[
                Patch(id="d", target="a.b", build=lambda o: o),
                Patch(id="d", target="a.c", build=lambda o: o),
            ],
        )
    with pytest.raises(PatchError, match="declares framework"):
        PatchSet(framework="x", patches=[Patch(id="d", target="a.b", framework="y", build=lambda o: o)])
    assert apply(pkg.name).applied == ["p.one", "p.two"]
    assert pkg.module.greet("a") == "v2"


def test_decorator_form(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    ps = PatchSet(framework=pkg.name)

    @ps.patch(id="p.deco", target=f"{pkg.name}.greet", versions=">=1", mode="wrap")
    def build_greet(original):
        """Shout the greeting."""

        def greet(name, punctuation="!"):
            return original(name, punctuation).upper()

        return greet

    assert callable(build_greet) and ps.get("p.deco").description == "Shout the greeting."
    assert ps.patches[0].framework == pkg.name and ps.patches[0].mode == "wrap"
    register_patchset(ps)
    assert apply(pkg.name).applied == ["p.deco"]
    assert pkg.module.greet("bob") == "HELLO BOB!"
    with pytest.raises(PatchError, match="duplicate patch id"):
        ps.patch(id="p.deco", target=f"{pkg.name}.farewell")(lambda o: o)


def test_distribution_fallback_to_dunder_version(synthetic_package: SyntheticPackage) -> None:
    pkg = synthetic_package
    ps = _register(
        pkg,
        Patch(id="p.d", target=f"{pkg.name}.greet", build=lambda o: o),
        distribution="not-a-real-distribution-name",
    )
    assert ps.installed_version() == "1.2.0"
    assert installed_version(pkg.name) == "1.2.0"


def test_detect_version_prefers_metadata_without_import() -> None:
    assert detect_version("packaging") == importlib.metadata.version("packaging")
    assert detect_version("nope_nothing_here_123") is None
    # distribution name differs from import name: metadata is consulted under the pip name.
    assert detect_version("yaml", "pyyaml") == importlib.metadata.version("pyyaml")


def test_disable_env_var(synthetic_package: SyntheticPackage, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = synthetic_package
    _register(pkg, Patch(id="p.dis", target=f"{pkg.name}.greet", build=lambda o: lambda n: "patched"))
    monkeypatch.setenv("MONORL_DISABLE_PATCHES", "1")
    report = apply(pkg.name)
    assert report.applied == [] and report.skipped == [("p.dis", "disabled by MONORL_DISABLE_PATCHES")]
    assert pkg.module.greet("a") == "hello a!"
    (s,) = status(pkg.name)
    assert s.applicable is False and s.applied is False and s.reason == "disabled by MONORL_DISABLE_PATCHES"
    monkeypatch.setenv("MONORL_DISABLE_PATCHES", "0")
    assert apply(pkg.name).applied == ["p.dis"]
