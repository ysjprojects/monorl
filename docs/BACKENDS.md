# Backends

monorl has two kinds of backends. **Component backends** plug into the monorl orchestrator loop
(data → parallel env episodes → rewards → advantages → trainer → weight sync): rollout engines,
trainers and environments. **Pipeline backends** own the whole loop: monorl compiles the
`RunConfig` into the framework's native config, injects monorl data sources, rewards and
environments through patches, and launches the framework's trainer, reporting into the same run
directory. Select with `rollout.backend`, `trainer.backend` and `env.type`.

Pins: [`third_party/pins.toml`](../third_party/pins.toml). Patch mechanics:
[`PATCHING.md`](PATCHING.md). `monorl doctor` shows what is installed here.

## Support matrix

| Framework | Pin | Role | `RunConfig` selector |
|---|---|---|---|
| transformers | 5.22, 5.23 | rollout engine | `rollout.backend: hf` |
| reference trainer (torch) | – | trainer (full / LoRA via peft) | `trainer.backend: reference` |
| vLLM | 0.33.0 | rollout engine | `rollout.backend: vllm` |
| SGLang | 0.5.23 | rollout engine | `rollout.backend: sglang` |
| Unsloth | 2026.10.11 (`v0.1.913-beta`) | trainer | `trainer.backend: unsloth` |
| TRL | 1.17.1 | pipeline backend | `trainer.backend: trl` |
| verl | 0.9.1 (`1876b06`) | pipeline backend | `trainer.backend: verl` |
| NeMo-RL | git v0.7.0 (`81aa43d`) | pipeline backend | `trainer.backend: nemo_rl` |
| OpenEnv | 0.3.0 | environment | `env.type: openenv` |
| Hermes Agent | 0.19.0 | environment | `env.type: hermes` |
| Go / C++ / any (MEP/1) | – | environment | `env.type: remote` |
| Ray | 2.60 | placement / worker groups | `cluster.mode: ray` |

| Framework | Runs here on CPU | Modes |
|---|---|---|
| transformers | yes | local, ray |
| reference trainer (torch) | yes | local, ray |
| vLLM | no (CUDA); `rollout.extra.mode: server` runs here against a remote `python -m monorl.adapters.vllm.server` | colocated (sleep mode), disaggregated, TP/DP, server (HTTP) |
| SGLang | no (CUDA); `rollout.extra.mode: server` runs here against a remote `sglang.launch_server` | colocated (`release/resume_memory_occupation`), disaggregated, server (HTTP) |
| Unsloth | no (CUDA only) | local |
| TRL | **yes** | local (`GRPOTrainer`), vLLM server/colocate via `trainer.extra` |
| verl | no (Ray + CUDA) | ray, FSDP / Megatron workers, colocated |
| NeMo-RL | no (Ray + CUDA; needs `NEMO_RL_HOME`) | ray virtual cluster, DTensor / Megatron |
| OpenEnv | yes (needs a running env server or in-process class) | http / in-process / docker image |
| Hermes Agent | yes (CPU, tools run locally) | multi-turn agent loop |
| Go / C++ / any (MEP/1) | yes | stdio, http |
| Ray | yes (local Ray) | colocated, disaggregated |

| Framework | Patched |
|---|---|
| transformers | – |
| reference trainer (torch) | – |
| vLLM | worker extension `monorl_update_weights` / `monorl_load_weights_from_path` via `collective_rpc` (vLLM's `GPUWorker` already owns `update_weights`), per-token logprob extraction, `SamplingParams` mapping, `sleep(level)`/`wake_up` journaling, `/monorl/update_weights` + `/load_weights_from_path` mounted on the API server |
| SGLang | `sglang.logprob_floats` (float/int coercion in `TokenizerManager.detokenize_logprob_tokens`), `sglang.seeded_parallel_sampling` (`Engine.generate` with `sampling_seed` + `n>1` gets one seed per sample), `sglang.release_memory_guard` / `sglang.resume_memory_guard` (no-op without CUDA or `enable_memory_saver`), `sglang.update_weights_named_tensors` (dict/generator inputs, contiguous detached tensors) |
| Unsloth | `FastLanguageModel` inside the reference loop; `unsloth.trl.*` patches reconcile Unsloth's TRL rewrites with monorl's TRL patchset. **Pin conflict:** declares `trl<=1.15.0`, `transformers<=5.17.0` vs monorl's trl 1.17.1 / transformers 5.23; see below |
| TRL | `GRPOTrainer` reporter hook → `metrics.jsonl`/`status.json`; env rollouts in place of plain generation |
| verl | reward manager → monorl rewards; dataset → monorl data source; `python -m monorl.adapters.verl.entry` runs `main_ppo` with the compiled Hydra tree and applies the `Tracking` metrics patch inside the Ray actor |
| NeMo-RL | `monorl.adapters.nemo_rl.env_actor.MonorlEnvironment` (`EnvironmentInterface`) bridge to monorl envs; reward bridge; config compiled to `<run_dir>/nemo_rl/nemo_rl.yaml` |
| OpenEnv | typed action/observation ↔ chat messages mapping |
| Hermes Agent | policy generation routed to the monorl rollout engine; tool execution + skills from Hermes |
| Go / C++ / any (MEP/1) | – (wire protocol, see `src/monorl/envs/protocol.py`) |
| Ray | – |

## Component backends

### Rollout engines (`rollout.backend`)

* `hf` — transformers `generate` with per-token logprobs, chat template applied by the engine,
  works for CPU/MPS/CUDA. Used by every CPU recipe and by tests. A batch is decoded by a chain of
  `generate` calls: when a row finishes, the call ends and the next one continues the others from
  the key-value cache without it, so no forward pass computes a finished row (run here on
  SmolLM2-135M, 16 prompts of up to 256 tokens: 4096 row-steps before, about 1640 after). Models
  with a processor and caches other than the default dynamic one are decoded in one call.
  During row retirement, indices are prepared once per current storage device and reused for
  cache K/V, token sequences, and masks. On ROCm this avoids per-tensor host-index promotions;
  row selection and KV-data copy volume are unchanged.
* `vllm` — `vllm.LLM` (offline) with `enable_sleep_mode` for colocated runs; TP via
  `rollout.tensor_parallel_size`, DP via `rollout.data_parallel_size` (one `LLM` per replica).
  Weight updates arrive through the patched worker extension — `monorl_update_weights` /
  `monorl_load_weights_from_path` over `collective_rpc`; the names are prefixed because vLLM 0.33's
  `GPUWorker` already defines `update_weights` — and CUDA IPC sync avoids host round-trips when
  trainer and engine share GPUs. `rollout.extra.mode: server` talks over HTTP to
  `python -m monorl.adapters.vllm.server`, which starts the OpenAI-compatible server with
  `VLLM_SERVER_DEV_MODE=1` (vLLM's own `/sleep`, `/wake_up`, `/is_sleeping`,
  `/reset_prefix_cache`) plus monorl's `/monorl/update_weights` (many tensors per request, raw
  bytes, at most `rollout.extra.weight_request_bytes` of them, default 64 MiB) and
  `/load_weights_from_path`. Image prompts are one chat request each, in flight together with
  the call's completion requests; an image is encoded to a data URL once per run
  (`rollout.extra.image_cache_bytes`).
* `sglang` — `sglang.srt.entrypoints.engine.Engine` (colocate, default) or an httpx client for
  `sglang.launch_server` (`rollout.extra.mode: server`, `url`/`urls`) using `/generate` and
  `/update_weights_from_tensor`; the tensor route only works when trainer and server share a host
  (serialised CUDA tensors), so cross-host server runs sync with `load_weights_from_path`
  (`cluster.weight_sync: checkpoint`). RadixAttention keeps shared
  prefixes cached across multi-turn episodes, which is what makes tool-calling / agent rollouts
  cheap. Prompts are always sent as raw `input_ids` (chat prompts tokenised with the HF chat
  template) so continuations stay byte-exact for the prefix cache. Weight updates go through
  `update_weights_from_tensor` in `rollout.extra.bucket_bytes`-sized buckets inside one
  weight-update session (`begin_weight_update(selector="all")` … `end_weight_update()`, which
  SGLang 0.5.23 requires for tensor updates and which updates the policy and a draft runner
  together), followed by one `flush_cache()`; checkpoints through `update_weights_from_disk`;
  `release/resume_memory_occupation`
  around training steps when colocated (`rollout.enable_sleep_mode` → `enable_memory_saver`,
  `rollout.extra.sleep_tags` optional). `ServerArgs` knobs (`schedule_policy`, `enable_metrics`,
  `random_seed`, `attention_backend`, `disable_cuda_graph`, …) pass through `rollout.extra`;
  `rollout.extra.engine_kwargs` is forwarded verbatim.

With `model.mtp.rollout` the engines draft with the model's MTP layers ([`MTP.md`](MTP.md)): `hf`
through transformers' assisted decoding with layers it holds for its lifetime (one sequence at a
time), `vllm` through `speculative_config {method: mtp}` with the worker extension updating the
drafter next to the policy, `sglang` through `speculative_algorithm: EAGLE` with the draft runner
updated in the same session. Options that would make the sampled distribution differ from the
recorded logprobs (vLLM's lossy rejection samplers, SGLang accept thresholds below 1.0, `min_p`
with drafting on either) are config errors.

Every engine implements `generate(prompts, params, *, max_new_tokens=None)` and returns
`list[list[Completion]]` with exactly `params.n` completions per prompt and per-token logprobs;
the turns of concurrent episodes reach it as batches
([ENVIRONMENTS.md](ENVIRONMENTS.md#the-episode-model)). `max_new_tokens` is a generation cap per
prompt (the room a turn has left in the context budget) that takes the place of
`params.max_new_tokens`. A row of the `hf` engine leaves the batch at its cap, vLLM gets one
`SamplingParams` per prompt and SGLang one `sampling_params` dict per request, in process and
over `POST /generate`; the vLLM server client sends one `/v1/completions` request per distinct
cap, all at once. A completion that
ends on one of `rollout.stop` keeps the stop string in its text on every engine — vLLM and
SGLang cut it by
default, so the adapters send `include_stop_str_in_output=True` / `no_stop_trim=True` — which is
what lets `tool_calling` parse a closing `</tool_call>` and keeps re-rendered chats token-exact.

All four GPU engine classes implement the optional `cancel()` (in-flight `generate` calls raise
`GenerationCancelled`, the engine stays usable). Colocated `vllm` checks between engine steps and
calls `LLMEngine.abort_request`; colocated `sglang` cancels its `Engine.async_generate` task, which
makes sglang send an abort per dispatched request. Both server clients close the HTTP request in
flight, and the server aborts on the disconnect (vLLM at once via `with_cancellation`; SGLang's
`/generate` waiter polls every `SGLANG_REQUEST_STATE_WAIT_TIMEOUT` seconds, 4 by default).

#### Prompts that rows share on HF

Rows of a batch that have the same prompt compute it once: the `n` samples of a prompt, and the
equal prompts of episodes whose calls were batched together (the first turn of a group). The HF
engine runs one forward pass over the distinct prompts without their last token, copies the
key-value cache rows to the rows that have each prompt, and lets `generate` start at the last
prompt token of every row. For a seed the tokens and finish reasons are those of a batch in which
every row computes its prompt, and so are the log-probs up to float32 rounding (the largest
difference measured below is 2.3e-5). The cache lives for the one call.

The engine does this when it saves at least 256 positions: the rows that are copies, times the
padded prompt length without the last token (`HFRolloutEngine.shared_prefill_min_positions`).
Below that the one forward pass more costs about what it saves. Every row computes its prompt
itself, as before, for prompts with images, with `model.mtp.rollout` (rows are drafted one by
one), and for a model whose prompts cannot be computed apart from their last token:

- a checkpoint with a processor, or with code of its own (`trust_remote_code`);
- layers whose cache is not the keys and values of every position: sliding-window attention,
  linear attention, state-space layers (tested with Mistral's `sliding_window`, Qwen3.5, Mamba);
- a `rope_type` other than `default`, whose frequencies can depend on the length of a forward
  pass (tested with `dynamic`);
- a `cache_implementation` in the checkpoint's `generation_config.json`;
- a model class whose `forward` takes no `past_key_values` or no `logits_to_keep`;
- a PEFT adapter of virtual tokens (LoRA adapters share), and a model in training mode.

Measured here on CPU (SmolLM2-135M, float32, 2 torch threads): 8 rows, 4 samples each of 2
prompts, 16 new tokens, sampled with a seed. Positions are rows times tokens over the forward
passes of the call; the time is the median CPU time of 7 calls, taken alternately with and
without sharing while other jobs ran on the machine. The tokens were the same in every case.

| Prompt tokens | Positions, every row its prompt | Positions, shared | CPU time | CPU time, shared |
|---|---|---|---|---|
| 32 | 362 | 176 | 0.93 s | 0.88 s |
| 64 | 632 | 254 | 1.08 s | 0.97 s |
| 128 | 1144 | 382 | 1.38 s | 1.16 s |
| 256 | 2168 | 638 | 2.10 s | 1.38 s |
| 512 | 4216 | 1150 | 3.80 s | 2.02 s |
| 931 | 7568 | 1988 | 6.84 s | 3.02 s |

The first row saves 186 positions and is below the threshold: it was measured with the threshold
lowered, and is why the threshold is where it is.

What it saves is prompt computation, so it shows where prompts are long next to what is
generated. On the quickstart recipe (8 rows over 2 prompts per step, 69 to 104 tokens wide, 128
new tokens) the model computes 3598 positions in three steps instead of 5176 and the 24 episodes
are the same ones, but the rollout takes as long as before: its 384 passes of one token per row
are what it spends its time on.

On MPS the same model and batch gave the same tokens with and without sharing, and a call (median
wall time of 5) took 0.67 s instead of 0.85 s for prompts of 300 tokens and 0.95 s instead of
1.71 s for 931. Not measured: CUDA, and half-precision checkpoints.

#### Multimodal (image) prompts

Chat prompts with `{"type": "image", ...}` parts (`data.image_field`, see `docs/ARCHITECTURE.md`
"Multimodal messages") are supported by every engine; raw-token prompts stay text-only.

| Engine / Mode | How images reach the model | Prompt ids recorded on `Completion` |
|---|---|---|
| `hf` / in-process | `AutoProcessor` (`monorl.rollout.media`) | processor chat template (placeholders included) |
| `vllm` / colocate | `TokensPrompt(prompt_token_ids=<processor ids>, multi_modal_data={"image": [PIL…]})`; `rollout.extra.limit_mm_per_prompt` (`{"image": N}` or `N`) → `EngineArgs.limit_mm_per_prompt` | processor chat template |
| `vllm` / server | `POST /v1/chat/completions` with `image_url` data-URL parts, `logprobs=true, top_logprobs=0`, `return_token_ids=true`; one request per prompt (`n` choices). The chat endpoint's logprob entries carry token *strings*, so ids come from `choices[i].token_ids` and the prompt ids from the response-level `prompt_token_ids` (rendered by the server's chat template) | server `prompt_token_ids` |
| `sglang` / colocate | `Engine.async_generate(input_ids=<processor ids>, image_data=[[PIL…] or None per request])` run on the engine's loop; sglang's multimodal processor decodes the ids, re-expands the image placeholders and returns the final ids | processor chat template |
| `sglang` / server | `POST /generate` with `image_data` = per-request lists of `data:image/png;base64,…` URLs (sglang `load_image`) | processor chat template |

In vLLM server mode all the requests of a `generate` call are in flight together: one
`/v1/completions` request per distinct cap and one chat request per image prompt. The
completions come back in the order of the prompts, and every request has the body it has when
it is sent alone. A request that fails drops the others before the call raises its error,
`cancel()` disconnects all of them, and weight updates and sleep wait for the call. The client
builds the body of every request before it sends any, so its memory grows with the batch;
httpx's pool (100 connections by default) bounds how many reach the server at a time.

Text-only models never load a processor (`load_processor` decides from the HF config); a VLM
prompt against a text-only model is a `ConfigError`. Audio/video parts are not forwarded by the
vLLM/SGLang adapters yet.

### Trainers (`trainer.backend`)

* `reference` — pure torch loop: policy loss from `monorl.algorithms` (grpo/dapo/rloo/
  reinforce, and `ppo` with a value head on the policy's hidden states and turn-level GAE,
  [`CREDIT_ASSIGNMENT.md`](CREDIT_ASSIGNMENT.md)), optional KL to a reference model, full
  fine-tuning or LoRA (`model.lora`), resumable checkpoints, HF export. With `model.mtp` it holds
  the model's MTP layers, trains them with the MTP loss (`model.mtp.train`) and writes them into
  weight sync, checkpoints and the export ([`MTP.md`](MTP.md)).
  **SFT** (`algorithm.name: sft`): `monorl.orchestrator.run` skips rollouts, rewards and
  environments and runs `monorl.orchestrator.sft.SFTLoop` — each sample is encoded as
  `messages + assistant response` (`data.response_field` → `answer_field` → trailing assistant
  turn) with the trainer tokenizer's chat template, the response span is found by token-exact
  prefix chaining (a template that re-renders the prompt differently falls back to tokenising the
  response alone, counted under `tokens/prefix_mismatch`; `sft_train_on_prompt: true` trains on
  every token), sequences are right-padded into a `TrainBatch` (`trainer.prompts_per_step`
  sequences per optimizer step, truncated to `trainer.max_seq_len`) and the trainer's unchanged
  `train_step` applies `monorl.algorithms.sft_loss` (masked NLL, `loss_agg` token/sequence,
  `sft_label_smoothing` via the vocabulary-mean logprob the trainer derives from its logits).
  Metrics: `loss/sft`, `sft/ppl`, `sft/tokens`, `tokens/response_mean`, `lr`, `time/train`;
  `eval_every` on `data.eval_uri` logs the held-out NLL as `eval/loss` (+ `eval/ppl`). Batches are
  a pure function of `(seed, step)` (epochs cycle the dataset with seeded permutations), so
  `save_every` checkpoints resume exactly; `final/` holds the HF export. Recipe:
  `configs/recipes/sft_coldstart_cpu.yaml`. The same config on `trl`/`verl`/`nemo_rl` maps to
  their native SFT trainers.
* `unsloth` — same loop, model built with `FastLanguageModel.from_pretrained` /
  `get_peft_model`; CUDA only. The adapter targets Unsloth 2026.10.11 (GitHub tag
  `v0.1.913-beta`). Unsloth must be imported before `trl`/`transformers`/`peft`:
  `python -m monorl.adapters.unsloth.preload monorl.cli train --config …` does that and then runs
  the CLI (`import monorl.adapters.unsloth.preload` as the first line of a script is equivalent);
  `monorl.adapters.unsloth.reconcile_trl_patches()` re-applies monorl's TRL patchset to the
  trainer classes Unsloth regenerates, so `trainer.backend: trl` also works with Unsloth imported.
  **Pin conflict:** unsloth 2026.10.11 declares `trl<=1.15.0` and `transformers<=5.17.0`, while
  `requirements.lock` has trl 1.17.1 / transformers 5.23.0; `pip install unsloth` into the monorl
  venv therefore downgrades (or refuses) those packages. Run Unsloth in its own environment:
  `make install-unsloth` creates `.venv-unsloth` with `monorl[unsloth,dev]` and the unsloth version
  pinned in `third_party/pins.toml` (uv resolves that to trl 1.15.0 / transformers 5.17.0 for Linux
  x86_64, Python 3.12; checked with `uv pip compile`, not installed here for lack of CUDA), then
  `.venv-unsloth/bin/python -m monorl.adapters.unsloth.preload monorl.cli train --config …`. The
  adapter's version gates are written against 2026.10.11.

### Environments (`env.type`)

[`ENVIRONMENTS.md`](ENVIRONMENTS.md) has the episode model, every parameter of the built-in
environments and how to write one; this table is the summary.

| type | what it does | key `env.params` |
|---|---|---|
| `single_turn` | one prompt → one completion, reward from `rewards` | `system_prompt` |
| `tool_calling` | multi-turn with built-in `calculator`, `python_exec`, `lookup` tools; Hermes or OpenAI call syntax | `tools`, `tool_modules`, `invalid_call_penalty`, `corpus` |
| `remote` | MEP/1 client for out-of-process envs (Go/C++/Python) | `transport: stdio` + `command`, `cwd`, `env`, `parallel`, `timeout_s` · `transport: http` + `url` or `urls` (replicas; each episode pinned to one), `parallel` (open-episode cap per server, default its `parallel_episodes`), `timeout_s` |
| `openenv` | OpenEnv Gym-style env as a multi-turn env | `url` · `env_class` + `env_kwargs` · `image` + `env_vars`; `action_mode`, `action_field`, `observation_template`, `system_prompt`, `invalid_action_penalty`, `reset_kwargs`, `timeout_s` |
| `hermes` | Hermes Agent turn loop (tools, skills, sandboxed terminal); policy tokens from our engine | `tools` (tool/toolset names, default `[file, terminal]`), `tool_modules`, `skills`, `skills_dir`, `hermes_home`, `workspace_dir` (default: temp dir per episode), `sandbox: {backend: local|docker|…, image}`, `max_tool_output_chars`, `max_calls_per_turn`, `invalid_call_penalty`, `system_prompt`, `hermes_system_prompt`, `model_name` |
| `harness` | any external agent program is the environment: one process per episode that calls the policy through the OpenAI-compatible capture gateway, which records every call with its token ids and logprobs ([`GATEWAY.md`](GATEWAY.md)) | `command` (argv or string; placeholders `{task_file}`, `{workdir}`, `{base_url}`, `{api_key}`, `{model}`, `{python}`, …), `timeout`, `env`, `verifier` + `verifier_timeout`, `workdir: {root, template}`, `model`; `env.max_turns` caps the policy calls |

Multi-turn prompts are built token-in/token-out by default (`rollout.turn_rendering: tokens`): the
sampled tokens of earlier turns stay in the context and only new messages are rendered, and each
trajectory trains as one sequence per token-contiguous chain of turns (see "Multi-turn episodes"
in [`ARCHITECTURE.md`](ARCHITECTURE.md)).

## Pipeline backends

Pipeline backends receive the full `RunConfig` and a run dir and return a `RunResult`. They are
selected by `trainer.backend` and ignore `rollout.backend`/`cluster.weight_sync` where the
framework makes its own choices; `trainer.extra` carries native overrides (dotted keys for verl
Hydra and NeMo-RL yaml, `GRPOConfig` kwargs for TRL).

* **TRL** — `monorl.data` → `datasets.Dataset` (`prompt` + `reference` columns), `rewards` →
  TRL reward callables, `GRPOConfig` from `trainer`/`rollout`/`algorithm`, patched
  `GRPOTrainer` writes `metrics.jsonl`/`status.json`/`trajectories.jsonl`. Runnable on CPU:
  `monorl train --config configs/recipes/trl_grpo_cpu.yaml`.
  Multimodal datasets add an `images` column (PIL, `datasets.Image`) and render `prompt` content
  as parts with `{"type": "image"}` placeholders that `trl.data_utils.prepare_multimodal_messages`
  fills; the trainer gets the model's `AutoProcessor` as `processing_class` and the policy is
  loaded with `AutoModelForImageTextToText`. `algorithm.name: sft` → `trl.SFTTrainer`/`SFTConfig`
  on `prompt`/`completion` conversational rows (`completion_only_loss` unless
  `algorithm.sft_train_on_prompt`, `max_length = trainer.max_seq_len`, `trainer.prompts_per_step`
  sequences per optimizer step). `algorithm.sft_label_smoothing > 0` swaps in
  `monorl.adapters.trl.sft.LabelSmoothedSFTTrainer`, whose `compute_loss` applies
  `monorl.algorithms.sft_loss` (smoothing and `loss_agg`) to the full logits at TRL's loss positions
  instead of TRL's deprecated `label_smoothing_factor`; smoothing 0 keeps the stock fused-LM-head
  path and `eval_loss` is always the unsmoothed NLL. Metrics map `loss → loss/sft`,
  `mean_token_accuracy → sft/token_accuracy`, `eval_loss → eval/loss`; checkpoints (`step-<n>`),
  resume and the merged `final/` export are shared with the GRPO path. Runnable on CPU
  (`tests/adapters/trl/test_sft_and_multimodal.py`).
* **verl** — pinned to v0.9.1 (commit `1876b06d`). Compiles `actor_rollout_ref.*`, `data.*`,
  `algorithm.*`, `trainer.*` from the RunConfig (FSDP actor, vLLM/SGLang rollout, GRPO/DAPO/PPO
  advantage estimator), registers a reward manager that calls `monorl.rewards.build_reward`, and
  launches `python -m monorl.adapters.verl.entry` on the Ray cluster — a `main_ppo` wrapper
  whose `TaskRunner` subclass applies the `verl.tracking.*` patches inside the Ray actor (the
  driver process never sees verl's `Tracking` logger), so metrics reach `metrics.jsonl`.
  Limits: `env.type` must be `single_turn` or `tool_calling` (verl's `single_turn_agent` /
  `tool_agent` loops) — any other environment is a `ConfigError`, because verl would sample
  single-turn completions and never step it; `rollout.stop` has no verl equivalent and is
  ignored; `rollout.backend: hf` falls back to vLLM (verl only has `vllm`/`sglang`); the
  per-reward breakdown is only available in
  verl's `reward_extra_info` (the scalar `score` is the reward). A secret that the config read
  from an environment variable is `${NAME}` in `monorl_reward.py` and `config.yaml`, and verl's
  reward workers resolve it themselves: set the variable on every node of the Ray cluster
  ([RUNS.md](RUNS.md#configyaml)). `trainer.extra` keys are
  deep-merged into the Hydra tree, except `verl_python`, which picks the interpreter that runs
  the entry (default: the current one).
  Multimodal: the parquet follows `examples/data_preprocess/geo3k.py` — message content keeps
  `<image>` placeholders and an `images` column holds `{"bytes": png, "path": null}` structs; the
  compiler sets `data.image_key=images`, `actor_rollout_ref.model.use_remove_padding=True` and
  `+actor_rollout_ref.rollout.engine_kwargs.<vllm|sglang>.limit_mm_per_prompt` from
  `rollout.extra.limit_mm_per_prompt`. `algorithm.name: sft` → v0.9.1's engine-based SFT trainer
  (`verl/trainer/sft_trainer_ray.py`, config `sft_trainer_engine.yaml`; `fsdp_sft_trainer` no
  longer exists at this tag): `data.{train_files,val_files,messages_key,max_length,train_batch_size,
  micro_batch_size_per_gpu}` over a `messages` parquet (`MultiTurnSFTDataset`, assistant turns
  masked), `model.path`, `optim.{lr,weight_decay,lr_warmup_steps,lr_scheduler_type,clip_grad}`,
  `trainer.{total_training_steps,total_epochs,save_freq,test_freq,default_local_dir,project_name,
  experiment_name,logger}`, `checkpoint.save_contents` + `hf_model` (so
  `global_step_<n>/huggingface` is the HF export), launched as `python -m monorl.adapters.verl.entry
  --sft <overrides>`; `train/loss → loss/sft`, `val/loss → eval/loss`.
* **NeMo-RL** — git-only (tag `v0.7.0`, commit `81aa43dd`, not on PyPI): set `NEMO_RL_HOME`
  (or `trainer.extra.nemo_rl_home`) to a NeMo-RL checkout, otherwise the backend raises
  `BackendUnavailable`. The adapter compiles `<run_dir>/nemo_rl/nemo_rl.yaml` (DTensor or
  Megatron policy, vLLM generation, cluster) and exports data to `<run_dir>/nemo_rl/train.jsonl`,
  then launches `uv run python -m monorl.adapters.nemo_rl.entry --config … --run-dir …` with
  `cwd=NEMO_RL_HOME` (NeMo-RL's own `uv` environment); the entry applies the patches, registers
  monorl envs as the `monorl` `EnvironmentInterface` actor
  (`monorl.adapters.nemo_rl.env_actor.MonorlEnvironment`) and runs `examples/run_grpo.py`.
  Metrics flow through the `nemo_rl.logger_metrics` patch (`Logger.log_metrics` →
  `metrics.jsonl`); checkpoints land in `<run_dir>/checkpoints/step_<n>/policy/weights`.
  Algorithms: grpo/dapo/rloo/reinforce/sft — `ppo` (no critic), `algorithm.num_epochs > 1` and
  `algorithm.entropy_coef > 0` are rejected at compile time; `rollout.backend` must be `vllm`.
  `algorithm.name: sft` compiles `examples/configs/sft.yaml` (`sft.{max_num_epochs,max_num_steps,
  val_period,…}`, the same `policy`/`checkpointing`/`logger`/`cluster` sections, `data.train` =
  the stock `ResponseDataset` over `<run_dir>/nemo_rl/sft_train.jsonl` rows of
  `{"messages": [...]}` with the `sft_processor`) and launches `examples/run_sft.py`
  (`--script`); the launcher exports `MONORL_ALGORITHM=sft` so the logger patch names
  `train/loss` `loss/sft` and `validation/val_loss` `eval/loss`. Images: **not supported** —
  NeMo-RL v0.7.0 trains VLMs only through its own datasets/processors (`examples/run_vlm_grpo.py`,
  `run_vlm_sft.py`, `vlm_hf_data_processor`), which the monorl text `message_log` bridge does not
  produce; `data.image_field` is a `ConfigError` (use `trl`, `verl` or `reference`).
  `trainer.extra`: `nemo_rl` (yaml overrides, deep-merged last so they win over everything the
  compiler derived), `nemo_rl_overrides` (hydra-style list), `nemo_rl_launcher` (`uv`|`python`),
  `nemo_rl_python`, `dtensor: {...}`, `megatron: {enabled: true, ...}` (Megatron policy config).

### Resume with a pipeline backend

`monorl train --resume` reaches the backend (`PipelineBackend.run(..., resume=True)`). The backend
continues from the newest checkpoint its framework can load, and first takes `metrics.jsonl` and
`trajectories.jsonl` back to the step of that checkpoint, as the loops do
([RUNS.md](RUNS.md#the-record-files-after-a-resume)): rows of later steps and an unfinished last
line are removed, because the framework writes those steps again. Without a checkpoint the run
starts at step 0 and `run.log` says so. Without `--resume` no backend continues an earlier run.

* **TRL.** `--resume` gives `Trainer.train(resume_from_checkpoint=...)` the newest
  `checkpoints/checkpoint-<n>` whose `trainer_state.json` can be read. `Trainer` writes that file
  last, so a checkpoint that a killed run left half-written is passed over. The trainer restores
  the weights, the optimizer, the scheduler, its state and the random generators of the process.
  Without `--resume` training starts at step 0, and a `checkpoint-<n>` that is there is written
  over when its step is saved.
  Run here on CPU: `trl_grpo_cpu.yaml` with `trainer.max_steps=4`, `trainer.save_every=2`,
  `data.limit=8`, `rollout.max_new_tokens=24`, `trainer.learning_rate=1e-4` and the `length`
  reward (`unit: chars, max_tokens: 100, penalty_per_token: 0.01`), once to the end, once with
  `trainer.max_steps=2` followed by `--resume` to 4, and once killed with `kill -9` when
  `metrics.jsonl` had the row of step 3 and resumed. In all three the rows of steps 1 to 4 were
  the same apart from the wall-clock fields (`time`, `trl/step_time`), the 32 trajectory rows were
  the same, and `final/model.safetensors` was identical in all 134,515,008 values. The resume of
  the killed run removed 1 line of `metrics.jsonl` and 8 of `trajectories.jsonl` (TRL samples the
  episodes of a step before it trains on them). The same finished directory started again without
  `--resume` and with another learning rate trained its 4 steps with that rate.
  What is not the uninterrupted run:
  - The row `Trainer` logs when training ends holds `trl/train_loss`, which it averages over the
    steps of the process: 0.00474 after the resume, 0.00358 for the run to the end. A run that
    ended at step 2 and is resumed with a larger `trainer.max_steps` keeps its row of that kind.
  - A checkpoint past the first pass over the dataset. `GRPOTrainer`'s sampler draws the prompt
    order of every pass from a generator of its own, seeded when the sampler is built, and a
    resumed `Trainer` starts at the pass of its checkpoint without the draws of the passes before
    it. With `data.limit=4` (2 steps are one pass) the run resumed from step 2 trained steps 3
    and 4 on the prompt order of steps 1 and 2, and its final weights differed from those of the
    run to the end in 133,501,909 values (largest difference 3.1e-4).
  - `algorithm.name: sft` on this backend was not run with `--resume`.
* **verl.** `--resume` compiles `trainer.resume_mode: auto`, a run without it `disable`. At the
  pin (`_load_checkpoint` in `verl/trainer/ppo/ray_trainer.py` and `v1/trainer_base.py`,
  `_determine_resume_path` in `verl/utils/checkpoint/checkpoint_handler.py` for SFT) `auto` loads
  the `global_step_<n>` that `checkpoints/latest_checkpointed_iteration.txt` names (actor, critic,
  dataloader state) and continues with step `n + 1`, or trains from scratch when the marker or the
  directory is missing; `disable` loads nothing. The marker is written when the checkpoint is
  complete, and the backend takes the record files back to its step. PPO saves the checkpoint of
  a step before it logs the step, so a run killed in between has no row of that step. Not run
  here (no GPU): the tests use a stand-in entry that follows these rules. How closely a resumed
  verl run repeats the run it resumes was not looked at.
* **NeMo-RL.** It has no setting for this. Every algorithm sets itself up from
  `CheckpointManager.get_latest_checkpoint_path()` (`nemo_rl/utils/checkpoint.py`, called in
  `nemo_rl/algorithms/grpo.py` and `sft.py` at the pin): the `step_<n>` with the largest `n` in
  `checkpointing.checkpoint_dir`, whenever there is one. So `--resume` is what NeMo-RL does in a
  directory with checkpoints, with the record files taken back to step `n` first. A run without
  `--resume` in such a directory is refused with a `ConfigError` before anything is launched,
  because NeMo-RL cannot start at step 0 there: pass `--resume`, or use another run directory.
  NeMo-RL renames a finished checkpoint from `tmp_step_<n>`, and writes it before the `train` row
  of step `n`, so a run killed in between has no such row. Not run here: the tests use the fake
  checkout, whose training script continues after the newest `step_<n>` as NeMo-RL does.

### Signals with a pipeline backend

[RUNS.md](RUNS.md#cancelling) says what SIGINT and SIGTERM do in a run: the first ends it after
the step in flight, with a checkpoint of that step, and a second one stops it at once. `run()`
gives the backend the event that the first signal sets (`PipelineBackend.run(..., cancel=event)`).

* **TRL** trains in the process of `monorl train` and does the same. A callback reads the event at
  the end of every step. Once it is set, `Trainer` logs that step and saves
  `checkpoints/checkpoint-<n>` (`step-<n>`), whatever `logging.log_every` and `trainer.save_every`
  are, and leaves its loop. `final/` is not written, the state is `cancelled`, the exit code 130,
  and `--resume` continues from that checkpoint. A signal in the last step leaves nothing to
  cancel and the run ends `succeeded`. A signal that arrives before step 1 has ended, also while
  the model is loading, ends the run after step 1. A second signal raises `KeyboardInterrupt`
  inside `Trainer.train`: the state is `cancelled`, and the step it interrupted has no checkpoint.
  Run here on CPU: `trl_grpo_cpu.yaml` with `trainer.max_steps=4`, `trainer.save_every=4`,
  `data.limit=8`, `rollout.max_new_tokens=32` and LoRA, SIGINT six seconds into step 1:

  ```text
  2026-11-17 02:59:30 received SIGINT: finishing the current step, then stopping
  2026-11-17 02:59:41 stopping after step 1 (cancel requested)
  2026-11-17 02:59:42 checkpoint saved: .../checkpoints/step-1
  2026-11-17 02:59:42 [cancelled] cancelled after step 1
  ```

  `--resume` then logged `resuming from .../checkpoints/checkpoint-1`, trained steps 2 to 4 and
  ended `succeeded`; `metrics.jsonl` has the row of every step from 1 to 4 once, and
  `trajectories.jsonl` the 8 episodes of each.
* **verl** and **NeMo-RL** run the framework in a child process and do not finish the step. What
  the backends do is tested against stand-ins for the two frameworks (a stand-in entry, a fake
  checkout); what the frameworks do is read from the pinned sources. Neither was run with a
  signal here. Common to both: a signal that arrives while monorl prepares the run (compiling
  the config, exporting the data) is logged with the line of any run, the preparation is
  finished, and the run ends `cancelled` without starting the child
  (`cancelled before verl was started`, `cancelled before NeMo-RL was started`); a second signal
  in that time ends the run at once. From the start of the child to its end the backend handles
  the two signals for the same event. The first one asks the child to stop and gives it 30
  seconds; `run.log` has `received SIGINT: stopping the verl driver` or `stopping NeMo-RL`. A
  second one kills the child at once, with the line `received SIGINT again: stopping now`. The
  training loops of the two frameworks handle neither signal at the pin (`verl/trainer/`;
  `nemo_rl/algorithms/grpo.py` and `sft.py`, `examples/run_grpo.py` and `run_sft.py`): the step
  in flight is lost, no checkpoint is written for the signal, and `--resume` continues from the
  checkpoint the framework saved last on its own schedule. The run ends `cancelled` with exit
  code 130.
  * **verl.** Within a second of the first signal (the backend looks once a second) the driver,
    `python -m monorl.adapters.verl.entry`, gets SIGTERM, and `run.log` the line
    `cancel requested: sent SIGTERM to verl driver`. A driver that is still running 30 seconds
    later (`VerlBackend.grace_seconds`, with the line
    `verl driver did not exit in time: sending SIGKILL`) is killed, and so is the driver at a
    second signal: SIGKILL to the driver first, then to every process that descends from it. The
    process tree is read twice for that, before the driver is killed and again before its
    descendants are, for what they started in between. Not reached: a process that is no longer
    in the driver's tree (its parent has ended, or it was started detached; whether the daemons
    Ray starts for the driver are among them was not looked at), and a process the driver starts
    between the first reading and its own SIGKILL. The driver stays in the process group of
    `monorl train`, so that Ctrl-C in a terminal reaches it as SIGINT as well, and so does what
    the server sends to the group of a job it cancels. The status message is
    `verl run cancelled`, and the step is that of `checkpoints/latest_checkpointed_iteration.txt`,
    or of the last row of `metrics.jsonl` when verl saved no checkpoint.
  * **NeMo-RL.** The child runs in a session of its own, so Ctrl-C in a terminal reaches only
    monorl. At the first signal the child's process group gets SIGTERM, and SIGKILL if the child
    is still running 30 seconds later (`CANCEL_GRACE_SECONDS`). What the child prints while it
    shuts down is copied to `run.log` like the rest of its output. At a second signal the group
    gets SIGKILL at once: the child and what it started, as far as that stayed in its group. The
    status message is `cancelled by user`, and the step is that of the last row of
    `metrics.jsonl`.

## Choosing

* Laptop / CPU / MPS: `hf` + `reference` (or `trl`).
* 1 GPU: `vllm` + `reference` colocated with sleep mode (`grpo_gsm8k_qwen_vllm_1gpu.yaml`).
* Multi-GPU single node: `vllm` TP/DP + `reference`, colocated with `cuda_ipc` sync
  (`dapo_math_vllm_8gpu_colocated.yaml`) or disaggregated with `broadcast`.
* Multi-node / Megatron scale: `verl` or `nemo_rl` pipeline backends.
* Agentic multi-turn: `sglang` + `tool_calling` / `hermes` / `remote`.
