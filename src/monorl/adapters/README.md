# monorl adapters

One subpackage per upstream framework. This page is for people changing an adapter, bumping a
pinned version or adding a framework: what each directory contains, the rules every adapter
follows and how adapters are tested without their framework. Options and limits of each backend
are in [docs/BACKENDS.md](../../../docs/BACKENDS.md).

## What is here

| Directory | Role | Selected by | Patches |
| --- | --- | --- | --- |
| `vllm/` | Rollout engine | `rollout.backend: vllm` | 4, `>=0.33,<0.34` |
| `sglang/` | Rollout engine | `rollout.backend: sglang` | 5, `>=0.5.23,<0.6` |
| `trl/` | Pipeline backend | `trainer.backend: trl` | 1, `>=1.17,<2` |
| `unsloth/` | Trainer | `trainer.backend: unsloth` | 1, `>=2026.10,<2027` |
| `verl/` | Pipeline backend | `trainer.backend: verl` | 3, `>=0.9,<0.10` |
| `nemo_rl/` | Pipeline backend | `trainer.backend: nemo_rl` | 1, `>=0.3,<1` |
| `openenv/` | Environment | `env.type: openenv` | none |
| `hermes/` | Environment | `env.type: hermes` | empty set |

The registry name equals the directory name (`PROVIDERS` in `__init__.py`). Every directory has an
`__init__.py` with the factory and `register(registry)`. The other files:

- `vllm/`: `engine.py` (colocated in-process engine), `client.py` (server-mode HTTP client),
  `server.py` (vLLM's OpenAI-compatible server plus weight-update endpoints), `worker.py` (worker
  extension for `worker_extension_cls`), `common.py` (helpers, version range), `patches.py`.
- `sglang/`: `engine.py` (colocated engine), `client.py` (httpx client for
  `python -m sglang.launch_server`), `common.py` (sglang-free helpers, version range), `patches.py`.
- `trl/`: `backend.py` (runs `trl.GRPOTrainer` or `trl.SFTTrainer`), `compile.py` (`RunConfig` to
  `GRPOConfig`/`SFTConfig` arguments), `data.py` (`DataSource` to `datasets.Dataset`), `rewards.py`
  (monorl reward as a TRL reward function), `rollout.py` (multi-turn environments), `sft.py`
  (label-smoothed `SFTTrainer`), `callbacks.py` (run directory reporting), `patches.py`.
- `unsloth/`: `trainer.py` (`FastLanguageModel` inside the reference trainer loop), `preload.py`
  (imports Unsloth before `trl`, `transformers` and `peft`), `patches.py` (also re-applies the TRL
  patchset on Unsloth's regenerated trainer classes).
- `verl/`: `backend.py` (stage the run directory, launch, monitor), `compiler.py` (`RunConfig` to
  the Hydra tree), `data.py` (parquet export), `reward.py` (`custom_reward_function` bridge),
  `tools.py`, `verl_tool.py` (tool-calling bridge), `entry.py` (verl with patches), `patches.py`.
- `nemo_rl/`: `backend.py` (compile, export, launch in a subprocess), `compile.py` (`RunConfig` to
  the GRPO or SFT yaml), `data.py` (samples to a response dataset and `DatumSpec` processor),
  `env_bridge.py`, `env_actor.py` (`EnvironmentInterface` over monorl environments, Ray actor),
  `entry.py` (in-environment wrapper), `patches.py`.
- `openenv/`: `env.py` (an openenv-core environment as a monorl `Environment`).
- `hermes/`: `env.py` (a Hermes Agent episode with the trained policy as its LLM), `patches.py`.

## Rules every adapter follows

- The registry imports an adapter only when a name it owns is requested, and importing
  `monorl.adapters.<name>` never imports the framework: that happens inside the factory or the
  class it builds, which raises `BackendUnavailable` (`core/errors.py`) when it is missing. `vllm`
  and `sglang` import torch at package import; the other six do not (fresh-interpreter check).
- Patches live in `<name>/patches.py` as one `PatchSet` named `PATCHES`, registered with
  `register_patchset` at import. Each `Patch` carries a `versions` specifier; a patch outside its
  range is skipped, not applied. The adapter calls `apply("<framework>")` where the framework
  starts: in the engine or backend (`vllm`, `sglang`, `trl`), the factory (`unsloth`), or
  `entry.py` inside the framework's own process (`verl`, `nemo_rl`). `monorl patches status` and
  `monorl doctor` show the gates. See [docs/PATCHING.md](../../../docs/PATCHING.md).

## Tests

Tests are in `tests/adapters/<name>/`. Frameworks absent from the dev environment are faked: the
fake mirrors the pinned release's API surface and reports its version, a fixture installs it for
one test and reverts applied patches afterwards. `vllm` (`vllm_fakes.py`, fixtures `fake_vllm` and
`no_vllm`), `sglang` (`fake_sglang.py`) and `nemo_rl` (`nemo_rl_fakes.py`) put fake modules in
`sys.modules`; `verl` writes a fake package to disk and `unsloth` adds a fake CUDA flag (both in
`conftest.py`). `trl`, `openenv` and `hermes` use the real package behind `pytest.importorskip`.

Tests against a real GPU framework are separate modules that start with `pytest.importorskip` and
set `pytestmark = pytest.mark.gpu`: `vllm/test_vllm_real.py`, `sglang/test_real.py`,
`verl/test_real_verl.py`. On this CPU machine `.venv/bin/python -m pytest -m gpu tests/adapters`
reports `3 skipped, 448 deselected`; with the frameworks and CUDA they run (not run here).

## Bumping a pinned version

1. Edit the framework's table in `third_party/pins.toml` (`version`, `tag`, `commit`).
2. `.venv/bin/python scripts/vendor.py --fetch <name>` clones the tag into `third_party/<name>/`
   (gitignored) and fails if the tag does not resolve to the pinned commit; `--list` shows pins
   and what is vendored. `<name>` is the table name, which is `hermes-agent` for `hermes/`.
3. Read the upstream diff at every patch target and every API the adapter calls. Update the code,
   each patch's `expected_signature` or `verify`, and the version constants: `VLLM_VERSIONS`
   (`vllm/common.py`), `SGLANG_VERSIONS` (`sglang/common.py`), `VERL_VERSIONS`, `VERL_TAG`,
   `VERL_COMMIT` (`verl/compiler.py`), `VERSIONS` (`patches.py` of `verl`, `nemo_rl`, `unsloth`),
   `TARGET_TAG`, `TARGET_COMMIT` (`nemo_rl/__init__.py`), `PINNED_VERSION` (`unsloth/__init__.py`),
   the literal in `trl/patches.py`.
4. Update the fake and its version constant to the new surface, and the extra in `pyproject.toml`
   if the lower bound moved. Nothing checks `pins.toml` against these constants; keep them in step.
5. Run `.venv/bin/python -m pytest tests/adapters/<name>`, then `monorl patches status <name>` and
   `pytest -m gpu tests/adapters/<name>` where the framework is installed.

## Adding an adapter

- `adapters/<name>/__init__.py` with `register(registry)` and a factory that imports lazily and
  raises `BackendUnavailable`; an entry in `PROVIDERS` (the key is the registry name).
- If it patches upstream: `patches.py` with `PATCHES`, registered at import, applied at start-up.
- `FRAMEWORKS` in `doctor.py`, an extra in `pyproject.toml`, a table in `third_party/pins.toml`.
- A pipeline backend: `PIPELINE_BACKENDS` in `config/schema.py`. An environment: the tuple in
  `envs()` in `server/routes_system.py`. Web lists: [web/README](../../../web/README.md).
- `tests/adapters/<name>/`: a fake (or `importorskip`), a provider test (registry lookup,
  `BackendUnavailable` without the framework), and a `gpu`-marked module for the real framework.
- A section in [docs/BACKENDS.md](../../../docs/BACKENDS.md).
