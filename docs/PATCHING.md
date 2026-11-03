# Patching upstream frameworks

monorl never forks vLLM, SGLang, verl, TRL, NeMo-RL, Unsloth, OpenEnv or Hermes. Every change
to upstream behaviour is a **versioned monkeypatch**: a `Patch` with an explicit target, a
version gate, a build function and a verification hook, grouped into a `PatchSet` per framework
and registered by that framework's adapter. The engine lives in `src/monorl/patching/`; the
adapters live in `src/monorl/adapters/<framework>/patches.py`. The engine never imports an
adapter; adapters import the engine.

Pins for every framework are in [`third_party/pins.toml`](../third_party/pins.toml);
`scripts/vendor.py --fetch <name>` clones the pinned commit into `third_party/<name>` for reading.

## Declaring a patch

```python
from monorl.patching import Patch, PatchSet, register_patchset

def _sleep_with_journal(original):
    def sleep(self, level: int = 1):
        self._monorl_sleeping = level
        return original(self, level)
    return sleep

PATCHES = PatchSet(
    framework="vllm",
    patches=[
        Patch(
            id="vllm.sleep_mode",            # unique, "<framework>.<name>"
            target="vllm.LLM.sleep",         # dotted path: module[.Class].attribute
            versions=">=0.8,<0.34",          # PEP 440 specifier; "" = any version
            mode="wrap",                     # "wrap" | "replace" | "add"
            build=_sleep_with_journal,       # original -> replacement
            description="Record the sleep level so weight sync can skip sleeping engines.",
            verify=lambda original: callable(original),      # optional sanity check on the target
            expected_signature="(self, level: int = 1)",     # optional: must match inspect.signature
        ),
    ],
)
register_patchset(PATCHES)   # at adapter import
```

Modes:

| mode | target must | `build(original)` receives |
|---|---|---|
| `wrap` | exist and be callable | the original callable; return a wrapper |
| `replace` | exist | the original attribute; return the replacement |
| `add` | **not** exist (unless `overwrite=True`) | `None`; return the new attribute |

## Gates, in order

`apply(framework)` evaluates every patch of the registered sets:

1. `MONORL_DISABLE_PATCHES=1` → nothing is applied (`apply` reports every patch as skipped).
2. Framework installed? `installed_version(framework)` reads `importlib.metadata`; absent →
   skipped with reason.
3. Already applied? Applying twice is a no-op (idempotent); `force=True` re-applies.
4. Version gate: `versions` specifier vs installed version (`packaging.specifiers`).
5. Target resolution: the dotted path must import and resolve; missing targets are reported,
   never raised.
6. `expected_signature` / `verify(original)` must pass.
7. The original attribute is journaled, the replacement installed. `revert(framework)` restores
   originals in reverse order; `revert_all()` for everything.

`ApplyReport(framework, installed_version, applied, skipped, failed)` summarises the run; `ok`
is true when nothing failed.

## Inspecting

```
monorl patches list                 # every registered patch, per framework
monorl patches status [FRAMEWORK]   # applicable / applied / verified against installed versions
monorl patches apply FRAMEWORK      # apply now (also happens automatically, see below)
monorl doctor                       # frameworks table shows applicable/applied/total counts
```

`PatchStatus` fields: `id, target, version_spec, installed_version, applicable, applied,
verified, reason, framework, mode, description`. `verified` is `None` when the target was not
inspected (framework missing or `status(..., resolve=False)`). `patches list --json` and
`GET /api/system` emit the same dicts.

## Automatic application

Adapters apply their patchset when they build a component (e.g. the vLLM engine factory calls
`apply("vllm")` before constructing `LLM`). For code paths that import a framework without going
through a monorl component, `activate_import_hook()` installs a `sys.meta_path` finder that runs
`apply(framework)` right after the upstream top-level package finishes importing.

## Writing a new patch: checklist

- Target the narrowest attribute that gives you the behaviour; prefer `wrap` over `replace`.
- Gate with the version range you actually tested (`pins.toml` is the upper bound you verified).
- Add `verify` / `expected_signature` so an upstream refactor is reported instead of silently
  mis-patching.
- Unit test the patch with a fake module (`tests/patching/` shows the pattern: register a
  patchset against a synthetic module, apply, assert, revert) and add an `importorskip`-guarded
  test against the real framework.
- Document what is patched and why in [`BACKENDS.md`](BACKENDS.md).
