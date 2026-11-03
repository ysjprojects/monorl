# monorl architecture & contracts

monorl is a unified RL harness. One orchestrator drives pluggable components; adapters bridge
upstream frameworks through **versioned monkeypatches** instead of forks, and everything lives in
one repo (Python core, adapters, Go/C++ env SDKs, web UI, recipes).

```
            ┌──────────────── monorl.orchestrator ────────────────┐
            │  data → env episodes (parallel) → rewards → advantages → trainer → weight sync  │
            └───┬──────────┬─────────────┬───────────┬──────────────┬────────────────┘
          DataSource  Environment   RewardFunction  Trainer      RolloutEngine
          local/hf/s3 single_turn   exact/format/   reference    hf (transformers)
                      tool_calling  math/code/      unsloth      vllm (patched)
                      remote MEP/1  judge/model                  sglang (patched)
                      (Go, C++, …)
                      openenv, hermes
          ─────────────── pipeline backends (own the whole loop) ───────────────
                      trl  ·  verl  ·  nemo_rl   (RunConfig compiled to native configs,
                                                 monorl data/rewards/envs injected via patches)
```

## Principles

1. **Patches, not forks.** Every upstream modification is a `Patch` with an explicit target,
   version gate, and verification; `monorl patches list` shows what applies to the installed
   versions. Upstream pins live in `third_party/pins.toml`.
2. **Protocols at the seams.** Components talk only through `monorl.core.interfaces`. The
   orchestrator never imports a framework.
3. **Config is the product.** `RunConfig` (`src/monorl/config/schema.py`) is the single
   user-facing schema; the web UI renders it, recipes serialise it, adapters compile it.
4. **Run dir is the API between processes.** Trainers write `status.json`/`metrics.jsonl`/…
   (`monorl.core.reporting`); the server and CLI only read files, never import backends.
5. **Environments are language-agnostic.** The MEP/1 wire protocol
   (`src/monorl/envs/protocol.py`) lets Go/C++/any env servers plug in; the orchestrator runs
   many episodes in parallel against them.
6. **Everything that can run on this dev box must be proven here** (CPU/MPS, Python 3.12).
   GPU-only paths are written against pinned upstream APIs, version-gated, unit-tested with
   fakes, and skipped (not stubbed) when the framework is absent.

## Dev environment facts

- macOS arm64 (M1 Pro, 16 GB), **no CUDA**. MPS exists but tests use CPU for determinism.
- `.venv` = Python 3.12 with: torch 2.15, **transformers 5.22** (v5 API!), trl 1.17.1,
  peft 0.21, datasets 5.1, ray 2.59, fastapi, s3fs, moto, openenv-core 0.3.0, hermes-agent 0.19,
  math-verify 0.9, pytest(+asyncio,timeout,xdist), ruff. Run things with `.venv/bin/python`.
- Not installable here: vllm (0.33 pinned), sglang (0.5.23), verl (0.9.1), unsloth (2026.10),
  nemo-rl (git, NVIDIA-NeMo/RL). Adapters for these must gate on import and version.
- Go 1.27 (`/opt/homebrew/bin/go`), Apple clang++ 21, GNU make. No cmake.
- HF cache already holds `HuggingFaceTB/SmolLM-135M-Instruct` (tokenizer used by the
  `tiny_model_dir` fixture), `HuggingFaceTB/SmolLM2-135M-Instruct`, and
  `hf-internal-testing/tiny-random-LlamaForCausalLM`.
- transformers v5 gotchas: `tokenizer.apply_chat_template(..., return_tensors="pt")` returns a
  `BatchEncoding` (use `return_dict=True` and `**enc`, or `tokenize=False` then call the
  tokenizer). `torch_dtype=` is `dtype=`. Prefer `AutoModelForCausalLM.from_pretrained(path, dtype=...)`.

## Layout & ownership

```
src/monorl/
  core/            types, interfaces, registry, reporting, errors      (frozen contract)
  config/          RunConfig schema + yaml loader                      (frozen contract)
  patching/        patch engine: Patch/PatchSet, version gates, apply/revert/verify, import hook
  data/            DataSource impls: local files, hf://, s3:// (fsspec), schema sniffing, prompt formatting
  algorithms/      pure-torch losses + advantage estimators (grpo/dapo/rloo/reinforce/ppo), KL estimators
  rewards/         reward functions: exact_match, format/regex, math_verify, code_exec, length, llm_judge, reward_model, composite
  envs/            Environment impls: single_turn, tool_calling, remote (MEP/1 http+stdio), protocol
  rollout/         RolloutEngine impls: hf (transformers) + dynamic batching proxy for parallel episodes
  trainer/         Trainer impls: reference torch trainer (full / LoRA), checkpoints, logprob computation
  orchestrator/    run(): episode runner (parallel envs), batch building, sync + async pipelined loops, resume
  distributed/     runtime assembly (local/ray), placement planner, Ray worker groups, weight sync strategies
  adapters/{vllm,sglang,trl,unsloth,verl,nemo_rl,openenv,hermes}/   engine|backend|env + patches.py
  server/          FastAPI app + job manager (subprocess `python -m monorl.cli train`) + static web
  cli/             typer app: serve, train, doctor, patches, data, env
  doctor.py        system/framework/patch detection report
envs/sdk/go/       Go SDK (MEP/1 http+stdio) + example envs           envs/sdk/cpp/  C++ SDK + examples
web/               Vite + React + TS UI                                configs/recipes/  example RunConfigs
third_party/       pins.toml (+ vendor script clones, gitignored)      tests/<area>/
```

Each workstream owns the directories named in its task and **only** those. Shared files
(`core/`, `config/`, `tests/conftest.py`, `pyproject.toml`) are frozen; if you truly need a
change there, make the smallest possible additive edit and say so in your final report.

## Contracts

### Component protocols — `src/monorl/core/interfaces.py`
Read the file. Key points:
- `RolloutEngine.generate(prompts, params) -> list[list[Completion]]`: exactly `params.n`
  completions per prompt, order preserved; prompts are chats **or** raw token ids; when
  `params.logprobs` is true every completion carries per-token logprobs under the sampling policy.
- `Trainer.train_step(TrainBatch) -> metrics`, `compute_logprobs(batch, reference=False)`,
  `export_weights()` (merged, HF-named), `save_checkpoint/load_checkpoint` (resumable),
  `save_pretrained` (HF export).
- `Environment.reset(sample) -> messages`, `step(Completion) -> StepResult`, `max_turns`.
- `RewardFunction(trajectories) -> list[RewardResult]` (batched).
- `PipelineBackend.run(config, run_dir, reporter=...) -> RunResult` for trl/verl/nemo_rl.
- `TrainBatch` (`core/batch.py`): right-padded `[B,T]`, `response_mask` marks loss positions,
  `advantages` per-token, `old_logprobs`/`ref_logprobs` aligned with `input_ids`.

### Registry — `src/monorl/core/registry.py`
`registry.register(namespace, name, factory)`; factories are `factory(config: RunConfig, **kw)`
except rewards: `factory(params: dict, **kw)` and data sources: `factory(data_config: DataConfig, **kw)`.
Providers expose `register(registry)`. Built-in providers are imported lazily on lookup miss;
adapters via `monorl.adapters.PROVIDERS`.

Names: rollout_engine `hf|vllm|sglang`; trainer `reference|unsloth`; pipeline_backend
`trl|verl|nemo_rl`; env `single_turn|tool_calling|remote|openenv|hermes`; weight_sync
`inprocess|checkpoint|broadcast|cuda_ipc`; data_source `local|hf|s3`; reward names listed above.

### Orchestrator entrypoint — `monorl.orchestrator`
```python
def run(config: RunConfig, run_dir: Path | None = None, *, resume: bool = False) -> RunResult
def resolve_run_dir(config: RunConfig) -> Path   # <logging.run_root>/<run_name>
```
Dispatch: `config.is_pipeline_backend` → `registry.get("pipeline_backend", backend)(config).run(...)`;
otherwise build `monorl.distributed.build_runtime(config)` and run the component loop.
Loop per step: sample `prompts_per_step` prompts → run `group_size` episodes per prompt
through the env (parallel; generation batched across episodes) → rewards → advantages
(`monorl.algorithms`) → `TrainBatch` → `trainer.train_step` (× `algorithm.num_epochs`) →
`weight_sync.sync` → checkpoint every `save_every` → `reporter.log_metrics/log_trajectories`.
`rollout.async_steps_ahead > 0` enables a producer thread generating ahead (bounded staleness).
Honour `SIGTERM`/`SIGINT` → `status cancelled`. Resume from the latest checkpoint when `resume`.

Metric names (used by the UI): `reward/mean`, `reward/std`, `reward/<name>` per reward,
`loss/policy`, `loss/kl`, `loss/entropy`, `policy/clipfrac`, `policy/approx_kl`,
`tokens/response_mean`, `tokens/prompt_mean`, `time/rollout`, `time/train`, `time/sync`, `lr`.

### Internal helper contracts (consumed across workstreams)
```python
# monorl.data
def open_source(data_config: DataConfig, *, split: str | None = None, tokenizer=None, seed: int = 0) -> DataSource
#   resolves uri scheme → local/hf/s3 source; sniffs format; applies prompt_template/system_prompt;
#   drops samples over max_prompt_tokens when a tokenizer is given; shuffle(seed)/limit.
def sniff_format(columns: dict[str, str], sample_rows: list[dict]) -> dict   # {format, prompt_field, answer_field, messages_field}

# monorl.rewards
def build_reward(specs: Sequence[RewardSpec], *, config: RunConfig | None = None) -> RewardFunction
#   weighted composite; RewardResult.breakdown[<spec.type or params["name"]>] = raw score of each member.

# monorl.algorithms
def compute_advantages(rewards: Tensor[B], group_ids: list[str], *, algorithm: AlgorithmConfig) -> Tensor[B]
#   group/batch/none normalisation; rloo = leave-one-out baseline; reinforce = raw reward.
def policy_loss(logprobs: Tensor[B,T], old_logprobs: Tensor[B,T] | None, advantages: Tensor[B,T],
                response_mask: Tensor[B,T], *, algorithm: AlgorithmConfig,
                ref_logprobs: Tensor[B,T] | None = None, entropy: Tensor[B,T] | None = None) -> PolicyLossOutput
#   PolicyLossOutput(loss: scalar Tensor, metrics: dict[str, float]) with keys loss/policy, loss/kl, loss/entropy,
#   policy/clipfrac, policy/approx_kl, policy/ratio_mean. old_logprobs=None → ratio=1 (on-policy REINFORCE-style).
def kl_penalty(logprobs, ref_logprobs, estimator: Literal["k1","k2","k3"]) -> Tensor[B,T]

# monorl.rollout
class BatchingEngine(RolloutEngine):   # wraps any engine; concurrent generate() calls from episode
    def __init__(self, engine, *, max_batch_size: int, max_wait_ms: float)   # threads are coalesced into one engine call

# monorl.envs
def make_env(config: RunConfig) -> Environment     # registry.get("env", config.env.type)(config)
```

### Patching API — `monorl.patching`
```python
from monorl.patching import Patch, PatchSet, register_patchset, apply, revert, status, activate_import_hook
PatchSet(framework="vllm", patches=[Patch(id="vllm.sleep_mode", target="vllm.LLM.sleep",
         versions=">=0.8", mode="wrap"|"replace"|"add", build=lambda original: new_callable,
         description=..., verify=lambda original: bool)])
register_patchset(PATCHES)         # adapters do this at import
apply("vllm") -> ApplyReport        # version-gated, idempotent, records originals for revert
status("vllm") -> list[PatchStatus] # id, target, version_spec, installed_version, applicable, applied, verified, reason
activate_import_hook()             # apply registered patchsets automatically after upstream import
```
Env `MONORL_DISABLE_PATCHES=1` disables everything. Adapters import the patch engine; the
engine never imports adapters.

### Doctor — `monorl.doctor.detect() -> SystemReport` (`.to_dict()`)
`python`, `platform`, `torch` (version, cuda count, mps), `toolchains` (go, cxx), `frameworks`
(`{name: {installed, version, patches: [PatchStatus]}}` for vllm, sglang, verl, trl, nemo_rl,
unsloth, openenv, hermes, ray), `components` (registry names per namespace).

### Run directory — `monorl.core.reporting` (read the module docstring)

### Server API (FastAPI, `monorl.server.app:create_app`)
```
GET  /api/system                      doctor report
GET  /api/config/schema               RunConfig JSON schema
POST /api/config/validate  {config}   -> {ok, errors:[{loc, msg}]}
GET  /api/recipes                     [{name, description, config}] from configs/recipes/*.yaml
GET  /api/models/search?q=&limit=     HF hub text-generation models [{id, downloads, likes, pipeline_tag, tags, last_modified}]
GET  /api/models/info?id=             config summary (architecture, params if known, gated)
GET  /api/datasets/search?q=&limit=   HF hub datasets
GET  /api/datasets/browse?uri=        list entries of a local dir or s3:// prefix [{name, uri, is_dir, size}]
POST /api/datasets/preview {uri, split?, hf_config_name?, limit?} -> {columns:{name:dtype}, rows, inferred:{format, prompt_field, answer_field, messages_field}, num_rows?}
GET  /api/connectors                  {s3: {configured: bool, region, endpoint_url, vars_present:[...]}}  (never echo secrets)
POST /api/jobs {config}               -> job {id, run_dir, state,...}; spawns `python -m monorl.cli train --config <run_dir>/config.yaml --run-dir <run_dir>`
GET  /api/jobs                        [job]
GET  /api/jobs/{id}                   job + status.json
POST /api/jobs/{id}/cancel
GET  /api/jobs/{id}/metrics?since=N   {rows, next}         GET /api/jobs/{id}/metrics/stream  (SSE)
GET  /api/jobs/{id}/logs?tail=N       {lines, next}        GET /api/jobs/{id}/logs/stream     (SSE)
GET  /api/jobs/{id}/trajectories?since=N
GET  /api/jobs/{id}/checkpoints
GET  /api/envs                        registered env types + MEP example servers discovered under envs/sdk
```
Jobs persist in `$MONORL_HOME/jobs.json` (default `~/.monorl`); run dirs under
`logging.run_root`. The server serves `web/dist` at `/` when present.

### CLI (`monorl` / `python -m monorl.cli`)
`serve [--host --port --no-open]`, `train --config PATH [--set k=v ...] [--run-dir DIR] [--resume]`,
`doctor [--json]`, `patches list|status|apply <framework>`, `data inspect URI [--limit N]`,
`env check (--command ... | --url ...)` (drives a MEP/1 server through spec/reset/step/close),
`recipes list|show NAME`, `bench rollout|env|sync|plan` (`src/monorl/bench/cli.py`, also
`python -m monorl.bench`: tokens/s, episodes/s, weight-sync timing and the placement plan for a recipe).

### MEP/1 environment protocol — `src/monorl/envs/protocol.py` (read the docstring)
Python side: env type `remote` with `env.params = {"transport": "stdio", "command": [...], "cwd": ...}`
or `{"transport": "http", "url": "http://host:port", "parallel": N}`. Go SDK: `envs/sdk/go`
(module `github.com/monorl/envsdk`), C++ SDK: `envs/sdk/cpp` (header-only `mep.hpp` +
`nlohmann/json.hpp` vendored). Both serve stdio and http from the same env implementation.

## What we take from each framework

| Framework | Take | Adapter shape |
|---|---|---|
| vLLM 0.33 | throughput, `LLM`/`AsyncLLMEngine`, sleep/wake (`sleep(level)`, `wake_up`), weight updates via `collective_rpc` worker extension, `enable_sleep_mode`, TP/DP | `rollout_engine: vllm` + patches (worker extension for `update_weights`, logprob extraction, sampling param mapping) |
| SGLang 0.5 | RadixAttention prefix cache for multi-turn/agentic rollouts, `Engine.update_weights_from_tensor`, `release_memory_occupation/resume_memory_occupation`, structured outputs | `rollout_engine: sglang` + patches |
| verl 0.9 | HybridFlow single-controller Ray design, resource pools, FSDP/Megatron workers, reward manager, DAPO/GRPO tuning | `pipeline_backend: verl`: compile RunConfig → verl Hydra config, inject monorl reward manager/data via patches, launch `main_ppo` |
| TRL 1.17 | simplest API (`GRPOTrainer`, `GRPOConfig`), PEFT integration, vLLM server/colocate modes | `pipeline_backend: trl` (**runnable here on CPU**): monorl data → HF dataset, monorl rewards → TRL reward funcs, patch `GRPOTrainer` for monorl reporter + env rollouts |
| NeMo-RL | Ray virtual cluster, DTensor/Megatron policies, `EnvironmentInterface` for multi-turn | `pipeline_backend: nemo_rl`: compile config, bridge monorl envs to `EnvironmentInterface`, launch |
| Unsloth | fast kernels + memory-efficient LoRA via patched transformers | `trainer: unsloth`: `FastLanguageModel` inside the reference trainer loop |
| OpenEnv 0.3 | Gym-style `reset/step/state` HTTP envs in containers | `env: openenv`: MEP-compatible client over OpenEnv servers (typed action/observation mapping) |
| Hermes Agent 0.19 | tool-using agent loop, skills | `env: hermes`: run a Hermes agent turn loop as a multi-turn env; policy tokens come from our engine |

## Rules for every workstream

- Tests go in `tests/<area>/`; per-area fixtures in `tests/<area>/conftest.py`. Use
  `tiny_model_dir` for anything that loads a model; mark real-model tests `@pytest.mark.slow`;
  skip (never fake) when an optional framework/toolchain is absent (`pytest.importorskip`,
  `shutil.which`).
- Run only your own tests while working: `.venv/bin/python -m pytest tests/<area> -q`. Do not run
  the whole suite, formatters, or `uv sync`. Need a package? `uv pip install --python .venv/bin/python <pkg>`
  and add it to the matching extra in `pyproject.toml` (additive edit).
- No stubs, placeholders, `NotImplementedError` "for later", or fake fallbacks. If a path needs a
  GPU/framework you don't have, implement it against the pinned upstream API (research with
  web_search / the upstream repo) and cover it with a fake-based unit test plus an
  import-guarded real test.
- Keep modules torch-free where the contract says so (`core/types`, `config`, `envs/protocol`,
  `server`, `cli`).
- Final report: what exists, how it was verified (commands + observed output), known gaps.
