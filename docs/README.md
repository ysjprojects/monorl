# monorl documentation

The [README](../README.md) has the overview, the quickstart, the recipes and the support matrix.
The documents here go one level down: one per section of the run config, one per subsystem.
[`CONFIG.md`](CONFIG.md) lists every field of the config with its default.

## By section of the config

| Section | Read |
|---|---|
| `model`, `rollout.backend`, `trainer.backend` | [`BACKENDS.md`](BACKENDS.md): engines, trainers, pipeline backends, their options and limits |
| `model.mtp` | [`MTP.md`](MTP.md): drafting with, training and exporting multi-token prediction layers |
| `data` | [`DATA.md`](DATA.md): URIs, S3 credentials, formats, field mapping |
| `algorithm` | [`ALGORITHMS.md`](ALGORITHMS.md): what each algorithm computes and what every option does |
| `algorithm.credit_assignment`, `ppo` | [`CREDIT_ASSIGNMENT.md`](CREDIT_ASSIGNMENT.md): per-turn returns and the value baseline |
| `algorithm.off_policy`, `rollout.async_steps_ahead` | [`OFF_POLICY.md`](OFF_POLICY.md): corrections for stale rollouts |
| `rewards` | [`REWARDS.md`](REWARDS.md): the built-in reward functions and how to write one |
| `env` | [`ENVIRONMENTS.md`](ENVIRONMENTS.md): the episode model, built-in environments, writing one in Python |
| `env.type: harness` | [`GATEWAY.md`](GATEWAY.md): an external agent program as the environment |
| `env.params.sandbox` | [`SANDBOX.md`](SANDBOX.md): episodes in Docker containers, task directories |
| `env.type: remote` | [`ENV_SDK.md`](ENV_SDK.md): the MEP/1 protocol and the Go and C++ SDKs |
| `cluster` | [`DISTRIBUTED.md`](DISTRIBUTED.md): placement, Ray, weight sync, async rollouts |
| `trainer`, `logging` | [`RUNS.md`](RUNS.md): what a run does and writes, metrics, checkpoints, logging |

## Running

- [`RUNS.md`](RUNS.md): the run directory, the metrics glossary, evaluation, resuming.
- [`PIPELINES.md`](PIPELINES.md): multi-stage pipelines (SFT, RL, rejection sampling, eval).
- [`WEB_UI.md`](WEB_UI.md): the web UI and its API.

## Internals and contributing

- [`ARCHITECTURE.md`](ARCHITECTURE.md): the component protocols, the registry, the orchestrator
  contract, the server API.
- [`PATCHING.md`](PATCHING.md): declaring, gating and verifying patches of upstream frameworks.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md): setup, checks, testing rules, conventions.
- [`../src/monorl/README.md`](../src/monorl/README.md): the code map and the extension points.
- [`../src/monorl/adapters/README.md`](../src/monorl/adapters/README.md): the anatomy of an
  adapter, bumping a pin, adding an adapter.

## Elsewhere in the repository

| Directory | README |
|---|---|
| `configs/recipes/` | [conventions for recipes](../configs/recipes/README.md) |
| `envs/sdk/go/`, `envs/sdk/cpp/` | the environment SDKs: [Go](../envs/sdk/go/README.md), [C++](../envs/sdk/cpp/README.md) |
| `examples/sandbox/` | [example task directories and a shell agent](../examples/sandbox/README.md) |
| `docker/` | [the images and the compose stack](../docker/README.md) |
| `web/` | [the web UI package](../web/README.md) |
