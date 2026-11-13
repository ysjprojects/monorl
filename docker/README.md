# Docker images

Dockerfiles for the monorl CLI and web UI on CPU, for monorl on top of the vLLM, SGLang and verl
images, and for the example environment servers. All are built with the repository root as
context. CI builds `Dockerfile.cpu` and `Dockerfile.envs` on every push, and the compose stack has
been run on Docker Desktop. `Dockerfile.vllm`, `Dockerfile.sglang` and `Dockerfile.verl` have not
been built here: they are written against their GPU base images.

## Images

| Dockerfile | Purpose | Base image |
|---|---|---|
| `Dockerfile.cpu` | Web UI and API server, reference trainer, `hf` rollouts, TRL. No CUDA. Entrypoint `monorl serve`, port 8000 | `python:3.12-slim-bookworm`; UI built in `node:24-bookworm-slim` |
| `Dockerfile.vllm` | monorl installed into the vLLM image, its torch and vllm kept as constraints. Entrypoint `monorl`, default command `doctor` | `vllm/vllm-openai:v0.33.0` |
| `Dockerfile.sglang` | The same on the SGLang image (venv `/opt/sglang`) | `lmsysorg/sglang:v0.5.23` |
| `Dockerfile.verl` | monorl installed into verl's venv, built for one inference engine and one trainer | `verlai/verl:uv.cu130` (linux/amd64 only) |
| `Dockerfile.envs` | The Go and C++ example MEP/1 servers from `envs/sdk`, one served over HTTP on port 8080 | `debian:bookworm-slim`; built in `golang:1.27-bookworm` and `debian:bookworm-slim` |

Build arguments and their defaults:

- `Dockerfile.cpu`: `PYTHON_VERSION` (3.12), `NODE_VERSION` (24), `UV_VERSION` (0.11.16),
  `MONORL_EXTRAS` (`trl,distributed,rewards,openenv,hermes`).
- `Dockerfile.vllm`: `VLLM_VERSION` (`v0.33.0`), `MONORL_EXTRAS` (`vllm,distributed,rewards`).
- `Dockerfile.sglang`: `SGLANG_VERSION` (`v0.5.23`), `MONORL_EXTRAS`
  (`sglang,distributed,rewards`).
- `Dockerfile.verl`: `VERL_IMAGE`, `INFER_BACKEND` (`vllm`), `TRAIN_BACKEND` (`fsdp`),
  `MONORL_EXTRAS` (`distributed,rewards`).
- `Dockerfile.envs`: `GO_VERSION` (1.27), `ENV_NAME` (`go-wordle`).

Every monorl image sets `MONORL_HOME=/root/.monorl`, `MONORL_RECIPES_DIR=/app/configs/recipes`
and `HF_HOME=/root/.cache/huggingface`, and declares `/root/.monorl` and
`/root/.cache/huggingface` as volumes. `.hadolint.yaml` is the lint configuration.

## Build and run

```sh
docker build -f docker/Dockerfile.cpu -t monorl:cpu .
docker run --rm -p 8000:8000 -v ~/.monorl:/root/.monorl monorl:cpu

docker build -f docker/Dockerfile.vllm -t monorl:vllm .
docker run --rm --gpus all --shm-size 10g -v ~/.monorl:/root/.monorl \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  monorl:vllm train --config configs/recipes/grpo_gsm8k_qwen_vllm_1gpu.yaml

docker build -f docker/Dockerfile.envs --build-arg ENV_NAME=go-wordle \
  -t monorl-env:wordle .
docker run --rm -p 8080:8080 monorl-env:wordle
curl -s localhost:8080/spec
```

The vLLM commands need a GPU host and have not been run here. The headers of
`Dockerfile.sglang` and `Dockerfile.verl` hold the matching commands (`Dockerfile.verl` takes
`--build-arg INFER_BACKEND=... --build-arg TRAIN_BACKEND=...`).

## `env-entrypoint.sh`

Entrypoint of the envs image. It runs `/opt/envs/bin/$ENV_NAME --transport http --addr $ENV_ADDR`
followed by any extra arguments (for example `--idle-timeout`). `ENV_NAME` is one of `go-arith`,
`go-wordle`, `go-gridworld`, `cpp-arith`, `cpp-number_guess` and can be changed at run time with
`-e ENV_NAME=...`; `ENV_ADDR` defaults to `0.0.0.0:8080`. An unknown name prints the available
binaries and exits with status 2. The protocol is in [docs/ENV_SDK.md](../docs/ENV_SDK.md).

## Compose services

[`../docker-compose.yml`](../docker-compose.yml) (`docker compose up --build`; the `gpu` profile has
not been run here):

| Service | Image | What it runs |
|---|---|---|
| `ui` | `monorl:cpu` (`Dockerfile.cpu`) | `monorl serve` on port 8000; mounts `~/.monorl` and the Hugging Face cache; starts after `env-wordle` is healthy |
| `env-wordle` | `monorl-env:wordle` (`Dockerfile.envs`) | 4 replicas of `go-wordle`, on host ports 8081 to 8084 |
| `ray-head` | `monorl:vllm` (`Dockerfile.vllm`) | `ray start --head` (ports 6379, 8265, 10001); profile `gpu`, needs the NVIDIA runtime |

The header of the compose file shows the `env.type: remote` block that addresses the replicas
one by one.

## What CI builds

`.github/workflows/ci.yml` has two jobs for this directory:

- `docker-lint` runs hadolint over all five Dockerfiles with `docker/.hadolint.yaml`.
- `docker-build` builds `Dockerfile.envs` (then starts it and requests `/health` and `/spec`)
  and `Dockerfile.cpu` (then runs `monorl doctor` in it).

`Dockerfile.vllm`, `Dockerfile.sglang` and `Dockerfile.verl` are linted but not built by CI, and
nothing in this repository pushes any image to a registry.
