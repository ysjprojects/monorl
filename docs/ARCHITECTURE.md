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
- `Environment.reset(sample) -> messages`, `step(Completion) -> StepResult`, `max_turns`; or an
  `EpisodeDriver` that runs the whole episode itself (`run_episode(sample, *, group_id, engine,
  tokenizer, params, abort, turn_rendering) -> Trajectory`): an external agent behind the capture
  gateway (`env.type: harness`, [`GATEWAY.md`](GATEWAY.md)). The env factory may return either.
- `RewardFunction(trajectories) -> list[RewardResult]` (batched).
- `PipelineBackend.run(config, run_dir, reporter=...) -> RunResult` for trl/verl/nemo_rl.
- `TrainBatch` (`core/batch.py`): right-padded `[B,T]`, `response_mask` marks loss positions,
  `advantages` per-token, `old_logprobs`/`ref_logprobs` aligned with `input_ids`, optional
  `loss_weights` (per-token weights of the loss mean, set for `algorithm.loss_agg: rollout`).
- `Trajectory` (`core/types.py`): `messages` is the conversation the episode ended in (thread 0),
  `extra_threads` hold other contexts of the same episode (a history before compaction, a
  sub-agent); every `Turn` names its thread and the position of its assistant message.

### Registry — `src/monorl/core/registry.py`
`registry.register(namespace, name, factory)`; factories are `factory(config: RunConfig, **kw)`
except rewards: `factory(params: dict, **kw)` and data sources: `factory(data_config: DataConfig, **kw)`.
Providers expose `register(registry)`. Built-in providers are imported lazily on lookup miss;
adapters via `monorl.adapters.PROVIDERS`.

Names: rollout_engine `hf|vllm|sglang`; trainer `reference|unsloth`; pipeline_backend
`trl|verl|nemo_rl`; env `single_turn|tool_calling|remote|openenv|hermes|harness`; weight_sync
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
`rollout.async_steps_ahead > 0` enables a producer thread generating ahead (bounded staleness);
`algorithm.off_policy` configures how the loss accounts for it (decoupled PPO, truncated importance
weights, ratio masks, score centering: [`OFF_POLICY.md`](OFF_POLICY.md)).
Honour `SIGTERM`/`SIGINT` → `status cancelled`. Resume from the latest checkpoint when `resume`.

#### Multi-turn episodes: prompts and training sequences
`rollout.turn_rendering` decides how the prompt of turn `k+1` is built
(`monorl.rollout.render.continue_prompt_ids`, used by the episode runner and the capture gateway):

- `tokens` (default): previous prompt ids + the completion ids exactly as sampled + the template's
  end-of-turn text (skipped when the completion already ended with that token) + the rendering of
  only the new messages and the generation prompt. Nothing the policy produced is re-tokenised, so
  special tokens, whitespace the template would trim and odd tokenisations stay in the context, the
  prompt is an exact extension of the previous turn (prefix cache hits) and the episode trains as
  one sequence. This fixes tokens, never content: a turn is continued only when the template
  itself would carry the assistant's content over unchanged. A template that rewrites history
  (Qwen3 / DeepSeek-R1 dropping the reasoning of earlier turns once a new user message arrives) is
  followed, i.e. that turn is re-rendered.
- `template`: the chat template re-renders the whole conversation every turn.

`monorl.orchestrator.batching.build_train_batch` turns each trajectory into one sequence per
token-contiguous chain of turns (`trajectory_rows`): turn `k` continues turn `j` when its prompt
starts with `j`'s prompt + completion ids. Re-rendered turns, a compacted history or a sub-agent's
conversation therefore become rows of their own instead of being dropped, and a turn shared by two
branches is a loss position in exactly one row. All rows of a trajectory carry its advantage,
reward and group id; `tokens/sequences` is the row count and `tokens/prefix_mismatch` the number of
trajectories that needed more than one row. `algorithm.loss_agg`: `token` (mean over response
tokens), `sequence` (per row, then over rows) or `rollout` (per trajectory across its rows, then
over trajectories, through `TrainBatch.loss_weights`; on backends that train one sequence per
rollout it equals `sequence`). Multimodal trajectories stay a single sequence (the final turn's
chain) because the media tensors belong to the final prompt.

#### Episodes that fail, episodes that run out of context
`monorl.orchestrator.episodes.run_episodes` is the one place episodes run (native loop, loop and
pipeline evaluation, rejection sampling, the TRL rollout). Two policies apply there:

- **Failures.** An abort (`EpisodeAborted`) and an engine failure (`GenerationFailed`) always end
  the batch. Anything else an episode raises is a failure of its environment: a crashing
  `reset`/`step`, an agent that dies before calling the policy, a conversation the chat template
  rejects, a first prompt over the context budget. Such an episode is started again `env.retries`
  times with a fresh environment. What still fails ends the run by default; with
  `env.max_failed_frac > 0` up to that share of a rollout batch is left out instead
  (`rollout/failed_episodes`, `rollout/failed_frac`, `eval/failed_frac`, `rs/failed_episodes`), and
  a prompt group left with a single member is dropped when the estimator needs a group baseline
  (`rollout/groups_incomplete`). Pipeline backends take one result per rollout and reject
  `env.max_failed_frac`.
- **Context budget.** `RunConfig.context_budget` is `rollout.max_context_tokens`, by default
  `trainer.max_seq_len` (and `rollout.max_model_len` when set): no prompt plus completion of a turn
  may exceed it, so nothing an episode samples is cut off in training. A turn's generation is
  capped to the room left; a stepped episode whose next prompt leaves no room ends with the turns
  it has (`Trajectory.info["context_exhausted"]`, `done=False`); an agent behind the gateway gets
  OpenAI's `context_length_exceeded` error and may compact its history, which starts a new
  conversation thread (`info["context_exceeded_calls"]`). `episode/context_exhausted_frac` counts
  both. A budget above `trainer.max_seq_len` is a config error. Multimodal prompts are tokenised
  by the engine and are not budgeted.

Metric names (used by the UI): `reward/mean`, `reward/std`, `reward/<name>` per reward,
`loss/policy`, `loss/kl`, `loss/entropy`, `policy/clipfrac`, `policy/approx_kl`,
`policy/staleness`, `tokens/response_mean`, `tokens/prompt_mean`, `tokens/sequences`,
`tokens/prefix_mismatch`, `time/rollout`, `time/train`, `time/sync`, `lr`, `offpolicy/*` when
`algorithm.off_policy` is active, and `rollout/failed_*` when `env.max_failed_frac > 0`.

Agent metrics (`monorl.orchestrator.metrics.episode_metrics(trajectories, *, success_threshold)`,
reported per step as `reward/success_rate`, `reward/pass_at_k` and `episode/<key>`, and by every
evaluation as `eval/<key>` next to `eval/reward_mean`, `eval/reward_std`, `eval/reward/<name>`,
`eval/num_samples`, `eval/time`):

| key | meaning |
| --- | --- |
| `success_rate` | episodes with total reward >= `logging.success_threshold` |
| `pass_at_k` | `group_id` groups with at least one success; only when some group has 2+ episodes |
| `turns_mean`, `turns_max` | policy turns per episode |
| `done_frac` | episodes the environment ended (`Trajectory.done`) |
| `truncated_frac` | episodes whose last turn finished with `length` |
| `context_tokens_mean`, `context_tokens_max` | largest prompt + completion token count of an episode |
| `tool_calls_mean`, `invalid_calls_mean` | per-episode sums of `Turn.info["tool_calls"]` (length) / `Turn.info["invalid_calls"]`; only when some turn reports them |
| `threads_mean` | conversations per episode; only when some episode has `extra_threads` |

#### `trajectories.jsonl` rows — `monorl.core.reporting.trajectory_row`
At most `logging.trajectories_per_step` rows per logged step (`logging.save_trajectories`), written
by training (`phase: train`), loop and pipeline evaluation (`eval`), rejection sampling
(`rejection`) and SFT sample logging (`sft`). Rows never hold token ids or logprobs.

```
{step, phase, sample_id, group_id, prompt, response, reward, reward_breakdown, success, done,
 finish_reason, num_turns, num_response_tokens, num_context_tokens, reference, info,
 threads?: [[{role, content: str | [part], name?, tool_call_id?, tool_calls?}]],
 turns?: [{index, thread, message_index, reward, finish_reason, prompt_tokens, response_tokens, info}]}
```

- `prompt` = last user message of the sample, `response` = all completions concatenated,
  `finish_reason` = the last turn's, `num_context_tokens` = largest prompt + completion token count
  over the turns, `success` = `reward >= logging.success_threshold`, `info` = `Trajectory.info`.
- `threads` / `turns` are present when `logging.save_transcripts` is true. `threads` is
  `Trajectory.threads` (thread 0 = the conversation the episode ended in). `turns[i].thread` indexes
  `threads`; `turns[i].message_index` is the position of that turn's assistant message in its thread
  — `Turn.message_index` when recorded, else the turns of a thread map in order onto the thread's
  last N assistant messages (N = its turn count), else `null`. `turns[i].info` = `Turn.info`.
- Size guards: any string over 20000 characters is cut and ends with
  `… [<n> characters truncated]`; media parts keep paths, URLs and `data:` URLs up to 64 KiB, anything
  else becomes `{"type": "image", "image": null, "note": "<what was omitted>"}`; values that are not
  JSON data (objects, bytes, non-finite floats) become short strings.

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
                ref_logprobs: Tensor[B,T] | None = None, entropy: Tensor[B,T] | None = None,
                loss_weights: Tensor[B,T] | None = None, behavior_logprobs: Tensor[B,T] | None = None,
                top_logprobs: Tensor[B,T,K] | None = None,
                behavior_top_logprobs: Tensor[B,T,K] | None = None) -> PolicyLossOutput
#   PolicyLossOutput(loss: scalar Tensor, metrics: dict[str, float]) with keys loss/policy, loss/kl, loss/entropy,
#   policy/clipfrac, policy/approx_kl, policy/ratio_mean. old_logprobs=None → ratio=1 (on-policy REINFORCE-style).
#   loss_weights: per-token weights of the loss mean (loss_agg="rollout"); behavior_logprobs and the top-k
#   tensors feed algorithm.off_policy. The trainer passes each of the four only when the batch carries it.
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
GET  /api/jobs/{id}/metrics?since=N        {rows, next}    GET /api/jobs/{id}/metrics/stream?since=N  (SSE: `metrics` per row, then `end` {state, next, error})
GET  /api/jobs/{id}/logs?tail=N            {lines, next}   GET /api/jobs/{id}/logs/stream?since=N     (SSE: `log` per line, then `end` {state, next, error})
GET  /api/jobs/{id}/trajectories?since=N   {rows, next}
GET  /api/jobs/{id}/checkpoints
GET  /api/jobs/{id}/events?metrics=N&logs=N&trajectories=N   SSE tail of the whole run dir on one connection: `metrics` / `log` / `trajectory`
                                           per row or line from each cursor (default 0, as `since` above), then `end`
                                           {state, error, next: {metrics, logs, trajectories}} once terminal and drained
GET  /api/jobs/{id}/stages                 pipeline manifest {job_id, kind, name, state, final_model_path, error, stages:[...]} (`stages: []` for plain runs)
GET  /api/jobs/{id}/stages/stream          SSE: `stages` = that payload, now and on every change, then `end` {state, next: null, error}
GET  /api/jobs/{id}/stages/{index}         one stage + its status.json
GET  /api/jobs/{id}/stages/{index}/(metrics|logs|trajectories|checkpoints)   the stage run dir, same shapes as the job-level routes
GET  /api/jobs/{id}/stages/{index}/events?metrics=N&logs=N&trajectories=N    SSE as /api/jobs/{id}/events for the stage run dir; `end` once the stage
                                           (or the job) is terminal; a pending stage's stream waits for it to start; unknown index -> 404
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
or `{"transport": "http", "url": "http://host:port", "parallel": N}` (`"urls": [...]` for replicas; each
episode is pinned to one). Go SDK: `envs/sdk/go`
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

## Phase 2 contracts: multimodal data, SFT, multi-stage pipelines

### Multimodal messages — `monorl.core.types`
`Message.content` is `str | list[ContentPart]`; parts are `{"type":"text","text"}`,
`{"type":"image","image": <path | file:// | s3:// | http(s):// | data:image/...;base64,...>}`,
`audio`/`video` likewise. Helpers: `message_text(m)` (text only — every reward/format check/log
uses it), `message_media(m, "image")`, `messages_have_media(messages)`. `DataConfig.image_field`
(column or columns: HF `Image` feature, path/URI, bytes, base64) attaches images to the user
message as parts; `'<image>'` in `prompt_template` positions them (default: before the text).
Images materialised from bytes/PIL are written to `$MONORL_HOME/cache/media/<sha256>.<ext>` (the
bytes' own container — png/jpg/gif/webp/bmp; anything else is re-encoded as PNG) and
referenced by path so samples/trajectories/MEP/1 stay JSON. Engines and trainers load media
through the model's `AutoProcessor`; `Completion.prompt_token_ids` includes media placeholder
tokens; `TrainBatch.extra` carries processor tensors (`pixel_values`, `image_grid_thw`, ...)
built per micro-batch. Raw-token prompts (`list[int]`) stay text-only; multi-turn VLM episodes
re-encode the chat each turn.

```python
# monorl.rollout.media  (torch-free at import; shared by hf engine, trainer, batch builder, vllm/sglang/trl adapters)
def load_image(ref) -> PIL.Image.Image            # path | file:// | data: | http(s):// | s3://,hf:// via monorl.data.media.MediaResolver
def chat_images(messages) -> list[PIL.Image.Image]
def hf_content_parts(messages, *, load_media=True) -> list[dict]   # monorl parts → HF template parts (image → PIL)
def image_data_url(image, *, format="PNG") -> str
def is_vlm_model(config) -> bool                  # AutoConfig: vision_config / image_token_id / known model_type
def auto_model_class(config)                      # AutoModelForImageTextToText for VLMs, else AutoModelForCausalLM
def load_processor(config) -> AutoProcessor | None   # None for text-only models (no second tokenizer load)
def vlm_encode(processor, messages, *, add_generation_prompt=True) -> dict[str, Tensor]   # the one encoding recipe
def vlm_prompt_ids(processor, messages) -> list[int]
def split_processor_outputs(enc) -> (token_aligned, media); def collate_media(per_sequence) -> dict[str, Tensor]
```
`TrainBatch.extra` convention: tensors are ``[B, ...]`` (sliced on dim 0, e.g. ``mm_token_type_ids``),
lists are per-sequence (``extra["per_sequence"][b]`` = that sequence's ``pixel_values``/``image_grid_thw``
dict or ``None``); the trainer concatenates them per micro-batch and lets the model derive its own
(M-RoPE) position ids. ``build_train_batch(..., processor=)`` re-encodes the final-turn chat and
requires the ids to equal the engine's recorded ``prompt_token_ids``. The HF engine expands
multimodal prompts to ``n`` rows itself (``generate``'s ``num_return_sequences`` would
``repeat_interleave`` patch tensors). Multimodal episodes send the chat to the engine on every
turn (one processor pass per turn, no prefix cache) and reject environment tool schemas.

### SFT — `algorithm.name: sft`
No rollouts, no rewards: the orchestrator's SFT loop iterates the dataset, encodes
`messages + assistant response` (`data.response_field`, else `answer_field`, else the trailing
assistant turn kept by the data layer in `metadata["assistant_turn"]`), masks the response span
(token-exact prefix chaining like multi-turn; `sft_train_on_prompt` includes the prompt) and
trains with `monorl.algorithms.sft_loss(logprobs, response_mask, *, label_smoothing, loss_agg)`
→ `PolicyLossOutput` with metrics `loss/sft`, `sft/ppl`, `sft/tokens`. `trainer.prompts_per_step`
= sequences per optimizer step; `max_steps`/`save_every`/`eval_every` as for RL (eval = held-out
NLL `eval/loss`). Pipeline backends map it to TRL `SFTTrainer`, verl's engine-based
`sft_trainer_ray` (`sft_trainer_engine.yaml`; v0.9.1 dropped `fsdp_sft_trainer`), NeMo-RL `run_sft.py`.

### Pipelines — `monorl.config.pipeline.PipelineConfig` → `monorl.pipeline.run_pipeline(config, run_dir=None, *, resume=False) -> PipelineResult`
Run dir layout: `<run_dir>/pipeline.yaml`, `status.json` (same shape as a run; `step` = stages
finished), `pipeline.json` manifest `{name, stages:[{index, name, type, state, run_dir, started_at,
finished_at, step, summary:{...last metrics / accepted counts}}]}`, `stages/<02d>-<name>/` each a
standard run dir (train stages) or `{dataset.jsonl, status.json, metrics.jsonl, trajectories.jsonl,
run.log}` (rejection_sample/eval). Model hand-off = previous train stage's `final/`. Resume skips
stages whose state is `succeeded`. Cancellation propagates to the active stage.

CLI: `monorl pipeline --config PATH [--set ...] [--run-dir] [--resume]`; `monorl train` on a file
with `stages:`+`base:` dispatches to the pipeline runner. Server: `POST /api/jobs` accepts
`{config: RunConfig}` or `{pipeline: PipelineConfig}` → job `kind: "run" | "pipeline"`;
`GET /api/jobs/{id}/stages` → manifest stages; `GET /api/jobs/{id}/stages/{index}/(metrics|logs|trajectories|checkpoints)`
read that stage's run dir with the same query params/shapes as the job-level routes;
`GET /api/config/pipeline_schema`. UI: pipeline jobs show a stage table and per-stage charts.

