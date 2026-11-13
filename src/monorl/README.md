# monorl source map

A map of the `monorl` Python package for people changing it: what each subpackage owns, the rules
that hold across packages, and where a new component plugs in. The contracts themselves are in
[docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md).

## Packages and modules

| Path | Role | Main document |
| --- | --- | --- |
| `adapters/` | One subpackage per upstream framework | [README](adapters/README.md), [BACKENDS](../../docs/BACKENDS.md) |
| `algorithms/` | Pure-torch advantages, losses, KL, off-policy, SFT | [ALGORITHMS](../../docs/ALGORITHMS.md) |
| `bench/` | `monorl bench`: rollout, weight sync, env, plan | [DISTRIBUTED](../../docs/DISTRIBUTED.md) |
| `cli/` | The `monorl` command, recipe discovery, `env check` | [root README](../../README.md) |
| `config/` | `RunConfig`, `PipelineConfig`, YAML loader, `--set` | [CONFIG](../../docs/CONFIG.md) |
| `core/` | Types, protocols, registry, run directory reporter | [ARCHITECTURE](../../docs/ARCHITECTURE.md) |
| `data/` | Local, `hf://` and `s3://` rows mapped to `Sample` | [DATA](../../docs/DATA.md) |
| `distributed/` | Runtime assembly, placement, Ray, weight sync | [DISTRIBUTED](../../docs/DISTRIBUTED.md) |
| `envs/` | Built-in environments, MEP/1 client and server, sandbox | [ENVIRONMENTS](../../docs/ENVIRONMENTS.md), [ENV_SDK](../../docs/ENV_SDK.md), [SANDBOX](../../docs/SANDBOX.md) |
| `gateway/` | Capture gateway behind `env.type: harness` | [GATEWAY](../../docs/GATEWAY.md) |
| `orchestrator/` | `run(config)`, the RL and SFT loops, episodes, batching | [RUNS](../../docs/RUNS.md) |
| `patching/` | Versioned monkeypatch engine and import hook | [PATCHING](../../docs/PATCHING.md) |
| `pipeline/` | Multi-stage runner, rejection sampling, eval stages | [PIPELINES](../../docs/PIPELINES.md) |
| `rewards/` | Reward functions and the weighted composite | [REWARDS](../../docs/REWARDS.md) |
| `rollout/` | `hf` engine, batching proxy, chat rendering, media | [BACKENDS](../../docs/BACKENDS.md) |
| `server/` | FastAPI app: jobs, run directory API, static UI | [WEB_UI](../../docs/WEB_UI.md) |
| `trainer/` | `reference` trainer (torch, transformers, PEFT LoRA) | [BACKENDS](../../docs/BACKENDS.md) |
| `doctor.py` | System, framework and patch report (`monorl doctor`) | [ARCHITECTURE](../../docs/ARCHITECTURE.md) |
| `mtp.py` | Multi-token prediction layers next to the policy | [MTP](../../docs/MTP.md) |

## Rules that hold across packages

**Import without torch.** Each of these, imported in a fresh interpreter, leaves `torch` out of
`sys.modules`: `monorl`, `monorl.config`, `monorl.core` (and its `types`, `interfaces`, `registry`,
`reporting`, `errors`), `monorl.cli.main`, `monorl.server.app`, `monorl.doctor`, `monorl.patching`,
`monorl.adapters`, `monorl.data`, `monorl.rewards`, `monorl.envs` (and its `protocol`, `remote`,
`server`, `harness`, `sandbox`), `monorl.gateway`, `monorl.distributed`, `monorl.pipeline`,
`monorl.bench`, `monorl.rollout` and `monorl.trainer` (the package `__init__` only: factories
import the engine and trainer modules when called). These import torch: `monorl.core.batch`,
`monorl.algorithms`, `monorl.mtp`, `monorl.orchestrator` (any submodule, through the package
`__init__`), `monorl.adapters.vllm` and `monorl.adapters.sglang`. To check a module, run
`.venv/bin/python -c "import sys, monorl.server.app; print('torch' in sys.modules)"`.

**The orchestrator imports no framework.** `orchestrator/` and `pipeline/` contain no import of
`vllm`, `sglang`, `verl`, `trl`, `nemo_rl`, `unsloth`, `ray`, `openenv`, `transformers`, `peft` or
`monorl.adapters`. They use torch and reach backends through the registry; `orchestrator/sinks.py`
imports `wandb` and TensorBoard inside the functions that need them.

**Components are found through the registry** (`core/registry.py`). Seven namespaces:
`rollout_engine`, `trainer`, `pipeline_backend`, `reward`, `env`, `data_source`, `weight_sync`. A
provider is a module with `register(registry)`. `registry.get(namespace, name)` first imports the
built-in providers on a miss (`BUILTIN_PROVIDERS`: `monorl.data`, `monorl.rewards`, `monorl.envs`,
`monorl.rollout`, `monorl.trainer`, `monorl.distributed`), then, if the name is still unknown, the
adapter that owns the part of the name before `:` (`PROVIDERS` in `adapters/__init__.py`). That
imports the adapter package, not its framework: `registry.get("rollout_engine", "vllm")` returns a
factory from `monorl.adapters.vllm` and leaves `vllm` out of `sys.modules`. A run config names
provider modules of its own in `providers`; `ensure_providers(config)` loads them in every process
that turns the config into components (the run, the runtime and its Ray actors, `build_reward`,
`make_env`), from the working directory or the Python path.

**The run directory is the interface between processes** (`core/reporting.py`). Every loop and
pipeline backend writes `config.yaml`, `status.json`, `metrics.jsonl`, `trajectories.jsonl`,
`run.log`, `checkpoints/` and `final/` through `RunReporter`. The server starts a run as a
`python -m monorl.cli train` subprocess (`server/jobs.py`) and reads those files back. See
[RUNS](../../docs/RUNS.md).

## Extension points

Interfaces are the protocols in `core/interfaces.py`. Tests go under `tests/<package>/`. Documents
are under `../../docs/`.

| To add | Interface | Register in | Tests, document |
| --- | --- | --- | --- |
| Reward | `RewardFunction` | `rewards/__init__.py` `register` | `tests/rewards/`, REWARDS |
| Environment | `Environment` (or `EpisodeDriver`) | `envs/__init__.py` `register` | `tests/envs/`, ENVIRONMENTS |
| Data source | `DataSource` | see note below | `tests/data/`, DATA |
| Rollout engine | `RolloutEngine` | `rollout/__init__.py` or an adapter | `tests/rollout/`, BACKENDS |
| Trainer | `Trainer` | `trainer/__init__.py` or an adapter | `tests/trainer/`, BACKENDS |
| Pipeline backend | `PipelineBackend` | an adapter's `register` | `tests/adapters/<name>/`, BACKENDS |
| Weight sync strategy | `WeightSync` | `distributed/__init__.py` `register` | `tests/distributed/`, DISTRIBUTED |
| Patch | `Patch` in a `PatchSet` | the adapter's `patches.py` | `tests/adapters/<name>/`, PATCHING |

- Outside this repository: a module with `register(registry)`, named in the config's `providers`,
  adds components without touching these files ([REWARDS](../../docs/REWARDS.md),
  [ENVIRONMENTS](../../docs/ENVIRONMENTS.md)).
- Data source: the loops call `monorl.data.open_source`, which picks the row source from the URI
  (`data/uri.py`, `open_rows` and the `RowSource` classes in `data/rows.py`). The `data_source`
  registry names `local`, `hf` and `s3` wrap that function, so a new source is a new `RowSource`.
- Pipeline backend: the name must also be in `PIPELINE_BACKENDS` in `config/schema.py`. Weight
  sync: also in `SYNC_NAMES` in `distributed/placement.py`.
- Patch: `Patch` and `PatchSet` are dataclasses in `patching/patch.py`, not protocols. The engine's
  own tests are in `tests/patching/`.
- The web wizard lists backend names by hand in `web/src/types.ts`; see
  [web/README](../../web/README.md). Adapters: see [adapters/README](adapters/README.md).
