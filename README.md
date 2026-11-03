# monorl

**One RL harness, every framework.** monorl drives reinforcement learning for language models
through a single orchestrator and a single config, and bridges the frameworks you already use —
[vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang),
[verl](https://github.com/verl-project/verl), [TRL](https://github.com/huggingface/trl),
[NeMo-RL](https://github.com/NVIDIA-NeMo/RL), [Unsloth](https://github.com/unslothai/unsloth),
[OpenEnv](https://github.com/openenv/OpenEnv) and
[Hermes Agent](https://github.com/NousResearch/hermes-agent) — with **versioned monkeypatches and
adapters instead of forks**.

- **One config.** `RunConfig` (model, data, algorithm, rollout, trainer, rewards, env, cluster)
  is the whole product: recipes serialise it, the web UI renders it, adapters compile it into
  verl Hydra trees, NeMo-RL yaml or TRL `GRPOConfig`.
- **Pluggable at the seams.** Rollout engines (`hf`, `vllm`, `sglang`), trainers (`reference`
  torch loop, `unsloth`), pipeline backends (`trl`, `verl`, `nemo_rl`), rewards, data sources
  and environments all talk through `monorl.core.interfaces`; the orchestrator never imports a
  framework.
- **Patches, not forks.** Each upstream modification is a `Patch` with a target, a version gate
  and a verification hook; `monorl patches list` shows what applies to what you have installed.
  Pins live in [`third_party/pins.toml`](third_party/pins.toml).
- **Distributed when you need it.** Colocated (engine sleeps during the optimizer step) or
  disaggregated GPU placement, Ray worker groups, in-process / CUDA-IPC / NCCL-broadcast /
  checkpoint weight sync, async rollouts with bounded staleness.
- **Environments in any language.** The MEP/1 wire protocol (JSON over stdio or HTTP) lets
  Go, C++ or Python environments plug in; the orchestrator runs many episodes in parallel against
  them. SDKs and examples ship under [`envs/sdk`](envs/sdk).
- **Low-code web UI.** Pick any Hugging Face model, a dataset from your disk, S3 (credentials
  from the environment) or the Hub, preview and map columns, launch and watch runs.

Everything that can run on a laptop is tested on a laptop (CPU/MPS, Python 3.12). GPU-only
paths are written against pinned upstream APIs, version-gated, unit-tested with fakes and skipped
— never stubbed — when the framework is absent.

## Quickstart

```bash
# install (Python 3.12, uv)
make install                      # .venv + monorl[dev,trl,distributed,rewards,openenv,hermes]
source .venv/bin/activate
monorl doctor                     # what is installed, which patches apply, registered components

# web UI
make web                          # build web/dist once (Node 24 + pnpm)
monorl serve                      # http://127.0.0.1:8000, opens a browser tab

# train from the CLI: GRPO on GSM8K with SmolLM2-135M, CPU only, ~minutes
monorl recipes list
monorl train --config configs/recipes/grpo_gsm8k_smollm_cpu.yaml --set trainer.max_steps=5
#   → ~/.monorl/runs/grpo-gsm8k-smollm-cpu/{config.yaml,status.json,metrics.jsonl,trajectories.jsonl,checkpoints/,final/}

# look at data before training
monorl data inspect hf://openai/gsm8k --config-name main --limit 3
AWS_ENDPOINT_URL=http://localhost:9000 monorl data inspect s3://my-bucket/data/train.jsonl
```

### Write an environment in Go

```go
package main

import (
    "context"
    "github.com/monorl/envsdk/mep"
)

type env struct{ episodes *mep.EpisodeStore[string] }

func (e *env) Spec() mep.Spec {
    return mep.Spec{Name: "yes-no", Version: "1", MaxTurns: 1, ParallelEpisodes: 32}
}
func (e *env) Reset(_ context.Context, id string, s mep.Sample, _ *int64) (mep.ResetResult, error) {
    e.episodes.Put(id, "yes")                       // sample.Reference holds the dataset answer
    return mep.ResetResult{Messages: s.Messages}, nil
}
func (e *env) Step(_ context.Context, id string, a mep.Action) (mep.StepResult, error) {
    want, _ := e.episodes.Get(id)
    reward := 0.0
    if a.Text == want { reward = 1 }
    return mep.StepResult{Reward: reward, Done: true}, nil
}
func (e *env) Close(id string) error { e.episodes.Delete(id); return nil }

func main() { mep.Main(&env{episodes: mep.NewEpisodeStore[string]()}) }   // --transport stdio|http --addr host:port
```

```bash
go build -o envs/bin/yes-no ./yes-no
monorl env check --command envs/bin/yes-no --turns 1         # spec → health → reset → step → close
```

then point a recipe at it:

```yaml
env:
  type: remote
  params: {transport: stdio, command: ["envs/bin/yes-no"], parallel: 32}
rewards: []            # the env's per-turn reward is used
```

`make envs` builds the bundled examples (`go-arith`, `go-wordle`, `go-gridworld`, `cpp-arith`,
`cpp-number_guess`) into `envs/bin`; `configs/recipes/grpo_remote_env_*.yaml` use them. The C++
SDK is header-only (`envs/sdk/cpp/include/mep.hpp`) and serves the same protocol.

## CLI

```
monorl serve   [--host --port --no-open --reload]        web UI + API
monorl train   --config PATH [--set k.path=v ...] [--run-dir DIR] [--resume]
monorl doctor  [--json]                                  system / frameworks / patches / components
monorl bench   rollout | env | sync | plan [--config PATH ...]   size a run: tokens/s, episodes/s, sync time, placement plan
monorl patches list | status [FRAMEWORK] | apply FRAMEWORK
monorl data    inspect URI [--limit N] [--split S] [--config-name C] [--json]
monorl env     check (--command CMD... | --url URL) [--turns N] [--json]
monorl env     serve TYPE [--transport stdio|http] [--port P]
monorl recipes list | show NAME
```

`python -m monorl.cli ...` is equivalent (the web server launches runs that way); `python -m monorl.bench`
is the same as `monorl bench`.

## Feature matrix

| | hf + reference | vLLM | SGLang | Unsloth | TRL | verl | NeMo-RL | OpenEnv | Hermes | MEP/1 (Go/C++) |
|---|---|---|---|---|---|---|---|---|---|---|
| Role | engine + trainer | engine | engine | trainer | pipeline | pipeline | pipeline | env | env | env |
| Pin | transformers 5.22 | 0.33.0 | 0.5.23 | 2026.10.11 | 1.17.1 | 0.9.1 | v0.7.0 (git) | 0.3.0 | 0.19.0 | – |
| CPU / this repo's tests | ✅ | fakes | fakes | fakes | ✅ | fakes | fakes | ✅ | ✅ | ✅ |
| GRPO / DAPO / RLOO / REINFORCE / PPO | ✅ | ✅ | ✅ | ✅ | GRPO, DAPO knobs | GRPO, DAPO, PPO | GRPO | – | – | – |
| LoRA | ✅ | n/a | n/a | ✅ | ✅ | ✅ | ✅ | – | – | – |
| Multi-turn / tools | ✅ | ✅ | ✅ (prefix cache) | ✅ | single-turn | single-turn | ✅ | ✅ | ✅ | ✅ |
| Colocated (sleep) | n/a | ✅ | ✅ | – | vLLM colocate | ✅ | ✅ | – | – | – |
| Disaggregated / Ray | ✅ | ✅ | ✅ | – | vLLM server | ✅ | ✅ | – | – | – |
| Weight sync | inprocess | cuda_ipc, broadcast, checkpoint | cuda_ipc, broadcast, checkpoint | inprocess | TRL-native | verl-native | NeMo-native | – | – | – |
| Patched | – | ✅ | ✅ | – | ✅ | ✅ | ✅ | – | – | – |

Details per backend: [`docs/BACKENDS.md`](docs/BACKENDS.md).

## Layout

```
src/monorl/        core (types, interfaces, registry, reporting) · config · patching · data · algorithms
                   rewards · envs (+ MEP/1 protocol) · rollout · trainer · orchestrator · distributed
                   adapters/{vllm,sglang,trl,unsloth,verl,nemo_rl,openenv,hermes} · server · cli · doctor
envs/sdk/{go,cpp}  environment SDKs + examples            web/        Vite + React UI
configs/recipes/   example RunConfigs                      third_party/ pins.toml (+ vendored clones)
```

## Docs

- [Architecture & contracts](docs/ARCHITECTURE.md)
- [Backends: support matrix, what is patched](docs/BACKENDS.md)
- [Patching: declaring, gating, verifying](docs/PATCHING.md)
- [Data: URIs, S3/MinIO/R2 env vars, formats, field mapping](docs/DATA.md)
- [Distributed: placement, Ray, weight sync, async rollouts](docs/DISTRIBUTED.md)
- [Web UI](docs/WEB_UI.md)

## Containers & CI

```bash
scripts/dev/bootstrap.sh          # .venv (uv, Python 3.12) + Go/C++ example envs → envs/bin + web/dist
scripts/dev/smoke.sh              # CPU end-to-end: 2-step GRPO run, then `monorl serve` + API/UI probes
docker compose up --build         # UI on :8000 + 4 `env-wordle` MEP/1 replicas (http://env-wordle:8080 inside the stack)
docker compose --profile gpu up ray-head   # Ray head on the vLLM image (NVIDIA runtime)
```

Images: `docker/Dockerfile.{cpu,vllm,sglang,verl,envs}`. `.github/workflows/ci.yml` runs ruff +
pytest (CPU), `go vet`/`go test`/example builds for `envs/sdk/go` and the C++ SDK build.

## Status

Verified on macOS (CPU, Python 3.12): the full suite passes (`pytest tests -m "not network"` →
797 passed); `grpo_gsm8k_smollm_cpu.yaml` trains end to end (3 steps ≈ 75 s); the Go wordle
environment trains through `grpo_remote_env_go_wordle.yaml`; the web UI launches a run and
streams live charts in a browser. GPU backends (vLLM, SGLang, Unsloth, verl, NeMo-RL) are
exercised against fakes only here.

## Development

```
make test-fast     # pytest -m 'not slow and not network'
make test          # everything (downloads tiny models, trains for a few steps on CPU)
make lint          # ruff
scripts/vendor.py --list | --fetch vllm|all     # clone pinned upstream sources into third_party/
```

License: Apache-2.0.
