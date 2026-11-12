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

| Framework | Pin | Role | `RunConfig` selector | Runs on this dev box (CPU) | Modes | Patched |
|---|---|---|---|---|---|---|
| transformers | 5.22 | rollout engine | `rollout.backend: hf` | yes | local, ray | – |
| reference trainer (torch) | – | trainer (full / LoRA via peft) | `trainer.backend: reference` | yes | local, ray | – |
| vLLM | 0.33.0 | rollout engine | `rollout.backend: vllm` | no (CUDA); `rollout.extra.mode: server` runs here against a remote `python -m monorl.adapters.vllm.server` | colocated (sleep mode), disaggregated, TP/DP, server (HTTP) | worker extension `monorl_update_weights` / `monorl_load_weights_from_path` via `collective_rpc` (vLLM's `GPUWorker` already owns `update_weights`), per-token logprob extraction, `SamplingParams` mapping, `sleep(level)`/`wake_up` journaling, `/update_named_param` + `/load_weights_from_path` mounted on the API server |
| SGLang | 0.5.23 | rollout engine | `rollout.backend: sglang` | no (CUDA); `rollout.extra.mode: server` runs here against a remote `sglang.launch_server` | colocated (`release/resume_memory_occupation`), disaggregated, server (HTTP) | `sglang.logprob_floats` (float/int coercion in `TokenizerManager.detokenize_logprob_tokens`), `sglang.seeded_parallel_sampling` (`Engine.generate` with `sampling_seed` + `n>1` gets one seed per sample), `sglang.release_memory_guard` / `sglang.resume_memory_guard` (no-op without CUDA or `enable_memory_saver`), `sglang.update_weights_named_tensors` (dict/generator inputs, contiguous detached tensors) |
| Unsloth | 2026.10.11 (`v0.1.913-beta`) | trainer | `trainer.backend: unsloth` | no (CUDA only) | local | `FastLanguageModel` inside the reference loop; `unsloth.trl.*` patches reconcile Unsloth's TRL rewrites with monorl's TRL patchset. **Pin conflict:** declares `trl<=1.15.0`, `transformers<=5.17.0` vs monorl's trl 1.17.1 / transformers 5.22 — see below |
| TRL | 1.17.1 | pipeline backend | `trainer.backend: trl` | **yes** | local (`GRPOTrainer`), vLLM server/colocate via `trainer.extra` | `GRPOTrainer` reporter hook → `metrics.jsonl`/`status.json`; env rollouts in place of plain generation |
| verl | 0.9.1 (`1876b06`) | pipeline backend | `trainer.backend: verl` | no (Ray + CUDA) | ray, FSDP / Megatron workers, colocated | reward manager → monorl rewards; dataset → monorl data source; `python -m monorl.adapters.verl.entry` runs `main_ppo` with the compiled Hydra tree and applies the `Tracking` metrics patch inside the Ray actor |
| NeMo-RL | git v0.7.0 (`81aa43d`) | pipeline backend | `trainer.backend: nemo_rl` | no (Ray + CUDA; needs `NEMO_RL_HOME`) | ray virtual cluster, DTensor / Megatron | `monorl.adapters.nemo_rl.env_actor.MonorlEnvironment` (`EnvironmentInterface`) bridge to monorl envs; reward bridge; config compiled to `<run_dir>/nemo_rl/nemo_rl.yaml` |
| OpenEnv | 0.3.0 | environment | `env.type: openenv` | yes (needs a running env server or in-process class) | http / in-process / docker image | typed action/observation ↔ chat messages mapping |
| Hermes Agent | 0.19.0 | environment | `env.type: hermes` | yes (CPU, tools run locally) | multi-turn agent loop | policy generation routed to the monorl rollout engine; tool execution + skills from Hermes |
| Go / C++ / any (MEP/1) | – | environment | `env.type: remote` | yes | stdio, http | – (wire protocol, see `src/monorl/envs/protocol.py`) |
| Ray | 2.59 | placement / worker groups | `cluster.mode: ray` | yes (local Ray) | colocated, disaggregated | – |

## Component backends

### Rollout engines (`rollout.backend`)

* `hf` — transformers `generate` with per-token logprobs, chat template applied by the engine,
  works for CPU/MPS/CUDA. Used by every CPU recipe and by tests.
* `vllm` — `vllm.LLM` (offline) with `enable_sleep_mode` for colocated runs; TP via
  `rollout.tensor_parallel_size`, DP via `rollout.data_parallel_size` (one `LLM` per replica).
  Weight updates arrive through the patched worker extension — `monorl_update_weights` /
  `monorl_load_weights_from_path` over `collective_rpc`; the names are prefixed because vLLM 0.33's
  `GPUWorker` already defines `update_weights` — and CUDA IPC sync avoids host round-trips when
  trainer and engine share GPUs. `rollout.extra.mode: server` talks over HTTP to
  `python -m monorl.adapters.vllm.server`, which starts the OpenAI-compatible server with
  `VLLM_SERVER_DEV_MODE=1` (vLLM's own `/sleep`, `/wake_up`, `/is_sleeping`,
  `/reset_prefix_cache`) plus monorl's `/update_named_param` (raw tensor bytes) and
  `/load_weights_from_path`.
* `sglang` — `sglang.srt.entrypoints.engine.Engine` (colocate, default) or an httpx client for
  `sglang.launch_server` (`rollout.extra.mode: server`, `url`/`urls`) using `/generate` and
  `/update_weights_from_tensor`; the tensor route only works when trainer and server share a host
  (serialised CUDA tensors), so cross-host server runs sync with `load_weights_from_path`
  (`cluster.weight_sync: checkpoint`). RadixAttention keeps shared
  prefixes cached across multi-turn episodes, which is what makes tool-calling / agent rollouts
  cheap. Prompts are always sent as raw `input_ids` (chat prompts tokenised with the HF chat
  template) so continuations stay byte-exact for the prefix cache. Weight updates go through
  `update_weights_from_tensor` in `rollout.extra.bucket_bytes`-sized buckets followed by one
  `flush_cache()`; checkpoints through `update_weights_from_disk`; `release/resume_memory_occupation`
  around training steps when colocated (`rollout.enable_sleep_mode` → `enable_memory_saver`,
  `rollout.extra.sleep_tags` optional). `ServerArgs` knobs (`schedule_policy`, `enable_metrics`,
  `random_seed`, `attention_backend`, `disable_cuda_graph`, …) pass through `rollout.extra`;
  `rollout.extra.engine_kwargs` is forwarded verbatim.

Every engine implements `generate(prompts, params) -> list[list[Completion]]` with exactly
`params.n` completions per prompt and per-token logprobs; the orchestrator's `BatchingEngine`
coalesces concurrent episode requests into one engine call.

#### Multimodal (image) prompts

Chat prompts with `{"type": "image", ...}` parts (`data.image_field`, see `docs/ARCHITECTURE.md`
"Multimodal messages") are supported by every engine; raw-token prompts stay text-only.

| Engine | Mode | How images reach the model | Prompt ids recorded on `Completion` |
|---|---|---|---|
| `hf` | in-process | `AutoProcessor` (`monorl.rollout.media`) | processor chat template (placeholders included) |
| `vllm` | colocate | `TokensPrompt(prompt_token_ids=<processor ids>, multi_modal_data={"image": [PIL…]})`; `rollout.extra.limit_mm_per_prompt` (`{"image": N}` or `N`) → `EngineArgs.limit_mm_per_prompt` | processor chat template |
| `vllm` | server | `POST /v1/chat/completions` with `image_url` data-URL parts, `logprobs=true, top_logprobs=0`, `return_token_ids=true`; one request per prompt (`n` choices). The chat endpoint's logprob entries carry token *strings*, so ids come from `choices[i].token_ids` and the prompt ids from the response-level `prompt_token_ids` (rendered by the server's chat template) | server `prompt_token_ids` |
| `sglang` | colocate | `Engine.generate(input_ids=<processor ids>, image_data=[[PIL…] or None per request])`; sglang's multimodal processor decodes the ids, re-expands the image placeholders and returns the final ids | processor chat template |
| `sglang` | server | `POST /generate` with `image_data` = per-request lists of `data:image/png;base64,…` URLs (sglang `load_image`) | processor chat template |

Text-only models never load a processor (`load_processor` decides from the HF config); a VLM
prompt against a text-only model is a `ConfigError`. Audio/video parts are not forwarded by the
vLLM/SGLang adapters yet.

### Trainers (`trainer.backend`)

* `reference` — pure torch loop: policy loss from `monorl.algorithms` (grpo/dapo/rloo/
  reinforce; `ppo` needs a critic and is only available through the verl backend), optional KL to a reference model, full fine-tuning or LoRA (`model.lora`),
  resumable checkpoints, HF export.
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
  monorl's environment pins trl 1.17.1 / transformers 5.22; `pip install unsloth` into the monorl
  venv therefore downgrades (or refuses) those packages. Run Unsloth in its own environment on
  the versions it accepts; the adapter's version gates are written against 2026.10.11.

### Environments (`env.type`)

| type | what it does | key `env.params` |
|---|---|---|
| `single_turn` | one prompt → one completion, reward from `rewards` | `system_prompt` |
| `tool_calling` | multi-turn with built-in `calculator`, `python_exec`, `lookup` tools; Hermes or OpenAI call syntax | `tools`, `tool_modules`, `invalid_call_penalty`, `corpus` |
| `remote` | MEP/1 client for out-of-process envs (Go/C++/Python) | `transport: stdio` + `command`, `cwd`, `env`, `parallel`, `timeout_s` · `transport: http` + `url` |
| `openenv` | OpenEnv Gym-style env as a multi-turn env | `url` · `env_class` + `env_kwargs` · `image` + `env_vars`; `action_mode`, `action_field`, `observation_template`, `system_prompt`, `invalid_action_penalty`, `reset_kwargs`, `timeout_s` |
| `hermes` | Hermes Agent turn loop (tools, skills, sandboxed terminal); policy tokens from our engine | `tools` (tool/toolset names, default `[file, terminal]`), `tool_modules`, `skills`, `skills_dir`, `hermes_home`, `workspace_dir` (default: temp dir per episode), `sandbox: {backend: local|docker|…, image}`, `max_tool_output_chars`, `max_calls_per_turn`, `invalid_call_penalty`, `system_prompt`, `hermes_system_prompt`, `model_name` |

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
  `algorithm.sft_train_on_prompt`, `max_length = trainer.max_seq_len`,
  `label_smoothing_factor = algorithm.sft_label_smoothing`, `trainer.prompts_per_step` sequences
  per optimizer step); metrics map `loss → loss/sft`, `mean_token_accuracy → sft/token_accuracy`,
  `eval_loss → eval/loss`; checkpoints (`step-<n>`), resume and the merged `final/` export are
  shared with the GRPO path. Runnable on CPU (`tests/adapters/trl/test_sft_and_multimodal.py`).
* **verl** — pinned to v0.9.1 (commit `1876b06d`). Compiles `actor_rollout_ref.*`, `data.*`,
  `algorithm.*`, `trainer.*` from the RunConfig (FSDP actor, vLLM/SGLang rollout, GRPO/DAPO/PPO
  advantage estimator), registers a reward manager that calls `monorl.rewards.build_reward`, and
  launches `python -m monorl.adapters.verl.entry` on the Ray cluster — a `main_ppo` wrapper
  whose `TaskRunner` subclass applies the `verl.tracking.*` patches inside the Ray actor (the
  driver process never sees verl's `Tracking` logger), so metrics reach `metrics.jsonl`.
  Limits: `rollout.stop` has no verl equivalent and is ignored; `rollout.backend: hf` falls
  back to vLLM (verl only has `vllm`/`sglang`); the per-reward breakdown is only available in
  verl's `reward_extra_info` (the scalar `score` is the reward). `trainer.extra` keys are
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
  experiment_name,logger}`, `checkpoint.save_contents` + `hf_model` (so `global_step_<n>/huggingface`
  is the HF export), launched as `python -m monorl.adapters.verl.entry --sft <overrides>`;
  `train/loss → loss/sft`, `val/loss → eval/loss`.
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

## Choosing

* Laptop / CPU / MPS: `hf` + `reference` (or `trl`).
* 1 GPU: `vllm` + `reference` colocated with sleep mode (`grpo_gsm8k_qwen_vllm_1gpu.yaml`).
* Multi-GPU single node: `vllm` TP/DP + `reference`, colocated with `cuda_ipc` sync
  (`dapo_math_vllm_8gpu_colocated.yaml`) or disaggregated with `broadcast`.
* Multi-node / Megatron scale: `verl` or `nemo_rl` pipeline backends.
* Agentic multi-turn: `sglang` + `tool_calling` / `hermes` / `remote`.
