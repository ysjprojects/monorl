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
- **Train any agent.** With `env.type: harness` an external agent program — an OpenAI- or
  Anthropic-SDK script, an agent framework, an agent CLI — is the environment: monorl serves the
  policy being trained over the OpenAI chat-completions, OpenAI Responses and Anthropic Messages
  APIs (streamed or not, image inputs included) and records every call with its exact token ids
  and logprobs ([`docs/GATEWAY.md`](docs/GATEWAY.md)).
- **Sandboxed episodes, benchmark tasks as data.** `env.params.sandbox: {backend: docker}` runs
  every episode's agent and verifier in their own container (image or build, resource limits,
  removed on every exit path), and a directory of tasks in the Harbor / Terminal-Bench format is
  a dataset: each task brings its instruction, image and verifier
  ([`docs/SANDBOX.md`](docs/SANDBOX.md)).
- **Built for multi-turn.** Prompts are continued token-in/token-out instead of re-rendering the
  chat, an episode trains as one sequence per token-contiguous chain of turns (compaction,
  sub-agents and re-rendered chats lose no tokens), `loss_agg: rollout` weighs every episode
  equally, and stale async rollouts get off-policy corrections
  ([`docs/OFF_POLICY.md`](docs/OFF_POLICY.md)).
- **Credit where it is due.** `credit_assignment: turn` credits each turn with the return from
  that turn on instead of the episode's total, and `algorithm.name: ppo` trains a value head next
  to the policy for turn-level GAE ([`docs/CREDIT_ASSIGNMENT.md`](docs/CREDIT_ASSIGNMENT.md)).
- **Multi-token prediction.** For models that ship MTP layers (DeepSeek-V3, GLM, Step, Mellum, …)
  `model.mtp` drafts with them during rollouts (speculative decoding that leaves sampled tokens
  and logprobs the policy's), trains them next to the policy so drafts keep matching it, and
  keeps them in weight sync, checkpoints and the exported model ([`docs/MTP.md`](docs/MTP.md)).
- **Survives real agents.** An episode whose agent crashes or whose environment fails is retried
  or left out of the batch instead of failing the run (`env.retries`, `env.max_failed_frac`), and
  every episode stays inside a context budget: generation is capped to the room left and an agent
  behind the gateway gets `context_length_exceeded`, so it can compact its history.
- **See what the agent did.** Every saved rollout carries its full transcript (conversations,
  per-turn rewards, tool calls and results); runs report success rate, pass@k and turns per
  episode; the web UI renders episodes turn by turn.
- **Distil what worked.** Rejection sampling keeps whole episodes — tool calls as the policy wrote
  them, tool results, the tool schemas — and SFT trains every policy turn of them
  (`algorithm.sft_turns: all`), so a pipeline can sample agent episodes, keep the solved ones and
  fine-tune on them ([`docs/PIPELINES.md`](docs/PIPELINES.md)).
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
monorl pipeline --config PATH [--set k.path=v ...] [--run-dir DIR] [--resume]   multi-stage: SFT → GRPO → rejection sampling → SFT → GRPO → eval
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
| RL algorithms | GRPO, DAPO, RLOO, REINFORCE, PPO (value head) | any (engine) | any (engine) | GRPO, DAPO, RLOO, REINFORCE, PPO (value head) | GRPO, DAPO | GRPO, DAPO, RLOO, REINFORCE, PPO (own critic) | GRPO, DAPO, RLOO, REINFORCE | – | – | – |
| Per-turn credit (`credit_assignment: turn`) | ✅ | any (engine) | any (engine) | ✅ | – | – | – | – | – | – |
| SFT (`algorithm.name: sft`) | ✅ | n/a | n/a | ✅ | ✅ `SFTTrainer` | ✅ `sft_trainer_ray` | ✅ `run_sft.py` | – | – | – |
| SFT on every assistant turn (`sft_turns: all`) | ✅ | n/a | n/a | ✅ | ✅ `assistant_only_loss` | always | ✅ | – | – | – |
| Images (VLM) | ✅ | ✅ | ✅ | text only | ✅ | ✅ | text only | text only | text only | ✅ image refs |
| LoRA | ✅ | n/a | n/a | ✅ | ✅ | ✅ | ✅ | – | – | – |
| MTP layers (`model.mtp`) | ✅ draft, train, export | draft, drafter weight sync | draft, draft weight sync | – | – | draft only | draft; train with Megatron | – | – | – |
| Multi-turn / tools | ✅ | ✅ | ✅ (prefix cache) | ✅ | ✅ monorl episodes as `rollout_func` | `tool_calling` env only | ✅ env bridge (the env's reset prompt is not forwarded) | ✅ | ✅ | ✅ |
| External agent as env (`harness` + capture gateway; Docker sandbox, task directories) | ✅ | ✅ | ✅ | ✅ | ✅ | – | – | – | – | – |
| Off-policy corrections (`algorithm.off_policy`) | ✅ decoupled PPO, truncated IS weights, ratio mask, score centering | ✅ (top-k for score centering) | ✅ (top-k for score centering) | ✅ | mapped to TRL's vLLM IS correction | mapped to `rollout_correction` | mapped to `loss_fn` IS options | – | – | – |
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
examples/sandbox/  example task directories + a shell agent for sandboxed harness runs
```

## Docs

- [Architecture & contracts](docs/ARCHITECTURE.md)
- [Backends: support matrix, what is patched](docs/BACKENDS.md)
- [Patching: declaring, gating, verifying](docs/PATCHING.md)
- [Data: URIs, S3/MinIO/R2 env vars, formats, field mapping](docs/DATA.md)
- [Distributed: placement, Ray, weight sync, async rollouts](docs/DISTRIBUTED.md)
- [Pipelines: SFT → RL → rejection sampling loops, stage manifest, API](docs/PIPELINES.md)
- [Capture gateway: training external agents (`env.type: harness`)](docs/GATEWAY.md)
- [Sandboxed episodes (Docker) and Harbor task directories](docs/SANDBOX.md)
- [Credit assignment: per-turn returns, value baseline (`ppo`)](docs/CREDIT_ASSIGNMENT.md)
- [Off-policy corrections for async rollouts](docs/OFF_POLICY.md)
- [Multi-token prediction: drafting, training and exporting MTP layers](docs/MTP.md)
- [Web UI](docs/WEB_UI.md)

## Containers & CI

```bash
scripts/dev/bootstrap.sh          # .venv (uv, Python 3.12) + Go/C++ example envs → envs/bin + web/dist
scripts/dev/smoke.sh              # CPU end-to-end: 2-step GRPO run, then `monorl serve` + API/UI probes
docker compose up --build         # UI on :8000 + 4 `env-wordle` MEP/1 replicas (`env.params.urls: [http://monorl-env-wordle-1:8080, …-4:8080]` inside the stack)
docker compose --profile gpu up ray-head   # Ray head on the vLLM image (NVIDIA runtime)
```

Images: `docker/Dockerfile.{cpu,vllm,sglang,verl,envs}`. `.github/workflows/ci.yml` runs ruff
(check + format) and pytest (CPU), `go vet`/`go test`/example builds for `envs/sdk/go`, the C++ SDK
build, the web build and lint, the MEP/1 conformance tests, hadolint, and builds the `envs` and
`cpu` images.

## Status

Verified on macOS (CPU, Python 3.12): `make test` and `make lint` pass;
`grpo_gsm8k_smollm_cpu.yaml` trains end to end (`scripts/dev/smoke.sh`); the six-stage
`r1_sft_grpo_loop_cpu.yaml` pipeline runs from the CLI, through a SIGTERM and `--resume`; the Go
wordle environment trains through `grpo_remote_env_go_wordle.yaml`, over stdio and over http
against the four replicas of the compose stack; the web UI launches a pipeline on the real server
and streams its stage table, charts, log and trajectories; `docker/Dockerfile.{cpu,envs}` build,
`docker compose up` comes up healthy and the five example environments pass `monorl env check`
from the container. With SmolLM2-135M on the `hf` engine: `tool_calling` episodes with real tool
calls train as one token-contiguous sequence each; an external tool-using agent trains through the
capture gateway (`env.type: harness`) on the native loop and on the TRL backend, including an agent
with a side conversation (two threads, two training sequences per episode); agents written with
the OpenAI SDK (Responses API, `previous_response_id`) and the Anthropic SDK (tool use, streamed
final turn) train through it with token-contiguous episodes, and an image-sending agent trains a
tiny random Qwen2-VL; a run whose agent crashes on some episodes finishes with those episodes
left out or retried; a context budget ends episodes instead of truncating them in training;
`credit_assignment: turn` and `ppo` (value head, turn-level GAE) train; the
sample-episodes → SFT-on-every-turn → eval pipeline runs end to end; harness episodes run in
real Docker containers (agent and verifier inside, reward from the verifier, containers removed)
and `grpo_harness_tasks_docker_cpu.yaml` runs its Harbor-format example tasks; async rollouts
(`rollout.async_steps_ahead: 2`) run with decoupled PPO + importance weights + ratio mask and with
score centering; the transcript viewer and the wizard's fields for all of the above were
exercised on the real server. With a tiny random Mellum that carries an MTP layer
(`grpo_mtp_tiny_cpu.yaml`): rollouts are drafted by the layer, the layer is trained and pushed
back every step (`mtp/accept_rate` 0 → 0.17 on sampled rollouts, 0.41 in greedy evaluation, over
16 steps) and the export loads with it in transformers. GPU backends (vLLM, SGLang, Unsloth,
verl, NeMo-RL) and their images are exercised against fakes only here, as are Docker Engine on
Linux (only Docker Desktop was run), MTP drafting on vLLM and SGLang, MTP models other than
Mellum, and the value head on Unsloth and on several Ray trainer ranks; `pytest -m gpu` runs the
real-framework tests on a CUDA machine.

## Development

```
make test-fast     # pytest -m 'not slow and not network'
make test          # everything (downloads tiny models, trains for a few steps on CPU)
make lint          # ruff
scripts/vendor.py --list | --fetch vllm|all     # clone pinned upstream sources into third_party/
```

License: Apache-2.0.
