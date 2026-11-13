# Runs

This document describes what a native monorl run does and what it writes: the per-step lifecycle,
the run directory, every metric name, evaluation, checkpoints and resume, and the optional logging
sinks. "Native" means the component RL loop (`src/monorl/orchestrator/loop.py`) and the SFT loop
(`src/monorl/orchestrator/sft.py`), both driven by `trainer.backend: reference` or `unsloth`. Read
it when you need to interpret `metrics.jsonl`, script against a run directory, or resume a run.
The examples are real output of 2 and 3 step CPU runs of the recipes
`configs/recipes/grpo_gsm8k_smollm_cpu.yaml` and `configs/recipes/sft_coldstart_cpu.yaml`.

## What a run does

`monorl.orchestrator.run.run(config, run_dir=None, *, resume=False)` creates the run directory,
writes `config.yaml`, sets the status to `running` and dispatches on the config: a pipeline backend
(`trainer.backend: trl | verl | nemo_rl`), the SFT loop (`algorithm.name: sft`) or the component
RL loop (everything else).

The component loop runs steps `1..trainer.max_steps`. Each step:

1. Prompt sampling: `trainer.prompts_per_step` samples from `data.uri`, a pure function of
   `(seed, step)`; passes after the first are reshuffled when `data.shuffle` is true.
2. Rollouts: `rollout.group_size` episodes per prompt, all with group id `s<step>-p<index>`, so a
   step trains on `prompts_per_step * group_size` episodes (`rollout/num_sequences`).
3. Rewards: the weighted sum of `rewards` ([REWARDS.md](REWARDS.md)) plus the discounted per-turn
   environment rewards, minus `algorithm.overlong_penalty` when the last completion hit the limit.
4. Train step: advantages and the token batch are built, then `trainer.train_step` runs
   `algorithm.num_epochs` times on that batch ([ALGORITHMS.md](ALGORITHMS.md)).
5. Weight sync: the trainer's weights are pushed to every rollout engine.
6. Checkpoint: when `step % trainer.save_every == 0` (default 50).
7. Report: when `step % logging.log_every == 0` (default 1) or on the last step, one `train` row
   goes to `metrics.jsonl` and up to `logging.trajectories_per_step` episodes go to
   `trajectories.jsonl`. `status.json` is updated every step.
8. Evaluation: when `trainer.eval_every` is set (default unset), `step % trainer.eval_every == 0`
   and `data.eval_uri` is set. See [Evaluation](#evaluation).

After the last step the policy is exported to `final/`. With `rollout.async_steps_ahead > 0`
rollouts for later steps are generated while earlier steps train ([DISTRIBUTED.md](DISTRIBUTED.md)).

Repeatability: Python and torch are seeded with `seed` at the start and with `seed + step` at
every step. A run with one turn per episode on the `hf` engine repeats for the same config and
seed (two such runs gave identical responses here). Multi-turn runs do not: episodes run on
threads and their generation requests are batched in arrival order
(`src/monorl/rollout/batching.py`), so the random draws a turn gets depend on timing (two runs of
a 3-turn environment with the same seed gave different episodes here).

The SFT loop has no rollouts, rewards, environments or weight sync. Each step encodes
`trainer.prompts_per_step` dataset samples with their target responses and calls `train_step`
`algorithm.num_epochs` times; checkpoints, reporting and evaluation follow the same cadence keys.

### States

| State | Written when |
|---|---|
| `pending` | only by pipeline backends, while they compile their config |
| `running` | at start, then after every step with a progress message |
| `succeeded` | all steps ran and `final/` was exported |
| `cancelled` | a signal stopped the run before `trainer.max_steps` |
| `failed` | any exception; the message is `<ExceptionType>: <text>`, the traceback is in `run.log` |

`status.json` holds one of these states. `monorl train` exits with 0 for `succeeded`, 130 for
`cancelled` and 1 otherwise.

### Cancelling

`run()` installs handlers for SIGINT (Ctrl-C) and SIGTERM when it is called on the main thread.

- First signal: a line is written to `run.log`, the step in progress finishes (train step, weight
  sync, a due checkpoint, reporting, a due evaluation), then the loop stops. If that step was not
  checkpointed already, `checkpoints/step-N/` is saved. `final/` is not written and the state is
  `cancelled`. If the step that finished was the last one, the run completes as `succeeded`.
- Second signal: `KeyboardInterrupt` is raised at once. The state is `cancelled`, no checkpoint is
  saved for the interrupted step (from the code, not run here).

A real cancel of an SFT run (`kill -INT` after step 1 was logged, exit code 130):

```text
2026-11-12 22:42:41 received SIGINT: finishing the current step, then stopping
2026-11-12 22:42:42 stopping after step 2 (cancel requested)
2026-11-12 22:42:44 saved checkpoint .../cancel/checkpoints/step-2
2026-11-12 22:42:44 [cancelled] cancelled after step 2
```

## Starting runs

```bash
.venv/bin/monorl train --config configs/recipes/grpo_gsm8k_smollm_cpu.yaml \
  --set trainer.max_steps=2 --set data.limit=4 \
  --set data.eval_uri=hf://openai/gsm8k --set trainer.eval_every=1 \
  --set trainer.save_every=1 --set logging.tensorboard=true \
  --run-dir /tmp/monorl-docs-runs/grpo
```

`--config` (`-c`) takes a `RunConfig` YAML; a file with `base:` and `stages:` is run as a
pipeline. `--set` (`-s`) is repeatable: it splits at the first `=`, walks the dotted key (creating
missing mappings) and parses the value with `yaml.safe_load`, so `10` is an int, `true` a bool,
`null` is None, `[a, b]` a list, `{type: exact_match}` a mapping, and text that is not valid YAML
stays a string. The merged config is then validated by the schema, which rejects unknown keys and
converts compatible strings (YAML reads `2e-6` as a string; the `float` field accepts it). List
items cannot be addressed (`rewards.0.weight=2` fails validation): replace the whole list.

`logging.run_root` defaults to `~/.monorl/runs`. `run_name` defaults to
`<model basename>-<algorithm.name>-<YYYYmmdd-HHMMSS>`, for example
`SmolLM2-135M-Instruct-grpo-20261112-224338`, and the default run directory is
`<logging.run_root>/<run_name>`. The shipped recipes set a fixed `run_name`, so without
`--run-dir` every run of a recipe uses the same directory. A run started in an existing directory
without `--resume` starts at step 0 and appends to `metrics.jsonl`, `trajectories.jsonl`, `run.log`.

### Resume

```bash
.venv/bin/monorl train --config configs/recipes/grpo_gsm8k_smollm_cpu.yaml \
  --set trainer.max_steps=3 --set data.limit=4 --set trainer.save_every=1 \
  --set logging.save_transcripts=false --run-dir /tmp/monorl-docs-runs/grpo --resume
```

`--resume` picks the highest-numbered non-empty `checkpoints/step-N/` and restores, through
`Trainer.load_checkpoint`: the model weights (or the LoRA adapter), optimizer, scheduler, gradient
scaler, the value head and MTP layers when configured, and the trainer's step counter. The loop
then seeds Python's and torch's RNG with `seed + N`, pushes the weights to the rollout engines and
continues at step `N + 1`; prompt batches need no state because they depend only on `(seed, step)`.

Not restored: the config (the one given on this command line is used and overwrites `config.yaml`),
`status.json` (rewritten with a new `started_at` and `pid`) and earlier rows of `metrics.jsonl`
and `trajectories.jsonl` (new rows are appended; nothing is truncated). Without a checkpoint the
run logs `resume requested but no complete checkpoint exists; starting from step 0`.

## The run directory

`src/monorl/core/reporting.py` owns the layout and has the readers the server and CLI use:
`read_status`, `iter_metrics`, `iter_trajectories` and `list_checkpoints`. Besides the files below,
`<run_dir>/tensorboard/` exists when the TensorBoard sink is active.

### `config.yaml`

The full validated `RunConfig` with every default filled in (`run_name` included), written before
anything else. Field reference: [CONFIG.md](CONFIG.md).

### `status.json`

Written atomically (temporary file, then rename). `message` is the latest progress line, `step`
the last finished step, `started_at` and `updated_at` are Unix seconds, `pid` is the process that
runs the loop. Pipeline backends add keys such as `child_pid`.

```text
{"state": "succeeded", "message": "finished 3 steps", "step": 3,
 "started_at": 1794541325.714816, "updated_at": 1794541351.5524049, "pid": 73073}
```

### `metrics.jsonl`

One JSON object per line: `step`, `time` (Unix seconds), `phase` (`train` or `eval`) and the
metrics of [Metrics](#metrics) as floats. A real `train` row and the `eval` row of the same step
(wrapped here):

```text
{"step": 1, "time": 1794541154.869489, "phase": "train", "rollout/groups_dropped": 0.0,
 "rollout/resample_rounds": 0.0, "time/rollout": 8.763545166002586, ...,
 "reward/mean": 0.0, "reward/std": 0.0, "reward/exact_match": 0.0, ...,
 "grad_norm": 0.0, "lr": 2e-06, "time/train": 7.402411415998358, ...,
 "time/sync": 0.23273037500621285, "time/step": 9.869084416000987}
{"step": 1, "time": 1794541162.3309078, "phase": "eval", "eval/reward_mean": 0.0,
 "eval/reward_std": 0.0, "eval/reward/exact_match": 0.0, "eval/success_rate": 0.0, ...,
 "eval/num_samples": 4.0, "eval/time": 6.8477982910117134}
```

### `trajectories.jsonl`

Written when `logging.save_trajectories` is true (default) and `logging.trajectories_per_step`
(default 8) is above 0: the first `trajectories_per_step` episodes of each logged step and of
each evaluation. Rows never hold token ids or log-probs.

| Field | Content |
|---|---|
| `step`, `phase` | step number; `train`, `eval` or `sft` (`rejection` in pipelines) |
| `sample_id`, `group_id` | dataset sample id; prompt group (`s1-p0`, `eval-s1-<sample id>`) |
| `prompt`, `response` | text of the last user message; the policy's response text |
| `reward`, `reward_breakdown`, `success` | total reward; its parts by name; `reward >= logging.success_threshold` |
| `done`, `finish_reason` | episode ended by the environment; finish reason of the last turn |
| `num_turns`, `num_response_tokens` | policy turns; generated tokens |
| `num_context_tokens` | largest prompt plus completion token count of any turn |
| `reference`, `info` | the sample's reference answer; episode info (reward details, env data) |

With `logging.save_transcripts` (default true) a row also has `threads` (every conversation of the
episode as a list of chat messages with `role`, `content` and, when present, `name`,
`tool_call_id`, `tool_calls`) and `turns` (one object per policy turn). Strings over 20,000
characters are cut with a marker and inline `data:` media over 64 KiB is replaced by a note.

```text
{"step": 1, "phase": "train", "sample_id": "hf://openai/gsm8k#train#1458",
 "group_id": "s1-p0", "prompt": "Stefan goes to a restaurant to eat dinner ...",
 "response": "To solve this problem, we need to divide the total amount ...",
 "reward": 0.0, "reward_breakdown": {"exact_match": 0.0}, "success": false,
 "done": true, "finish_reason": "length", "num_turns": 1, "num_response_tokens": 128,
 "num_context_tokens": 221, "reference": "108",
 "info": {..., "reward_info": {"exact_match": {"extracted": "50"}}},
 "threads": [[{"role": "system", "content": "Solve the problem step by step ..."}, ...]],
 "turns": [{"index": 0, "thread": 0, "message_index": 2, "reward": 0.0,
            "finish_reason": "length", "prompt_tokens": 93, "response_tokens": 128,
            "info": {}}]}
```

`turns[i].message_index` is the position of that turn's assistant message in
`threads[turns[i].thread]`. SFT rows (`phase: sft`) have the dataset response as `response`,
`reward` 0.0 and an empty `reward_breakdown`.

### `run.log`

One line per event, `YYYY-MM-DD HH:MM:SS <message>`; status changes appear as `[state] message`.
Lines are also echoed to stderr unless `MONORL_REPORTER_ECHO=0`.

### `checkpoints/step-N/`

Written by the reference trainer into `step-N.tmp` and renamed, so a `step-N` directory is always
complete. Old checkpoints are not deleted. Contents: `model/` (`config.json`,
`generation_config.json`, `model.safetensors`), or `adapter/` (the PEFT adapter) with
`model.lora.enabled` (not run here), and `trainer_state.pt`, a `torch.save` dict with `step`,
`optimizer`, `scheduler`, `scaler`, `value_head`, `mtp` and `rng`.
For SmolLM2-135M in fp32 with AdamW one checkpoint is 1.6 GB (514 MB weights, 1.1 GB state).

### `final/`

Written only when the run succeeds: `config.json`, `generation_config.json`, `model.safetensors`,
`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`. It loads with
`AutoModelForCausalLM.from_pretrained`. With LoRA the exported weights have the adapter merged, and
with `model.mtp.enabled` the MTP layers are included (both from the code, not run here).

## Metrics

Names are exact keys of `metrics.jsonl`. Trainer metrics are means over micro-batches and over
`algorithm.num_epochs`. "Response tokens" are the tokens the loss is computed on.

Rewards and episodes (`train` rows of the RL loop, `src/monorl/orchestrator/metrics.py`):

| Metric | Meaning |
|---|---|
| `reward/mean`, `reward/std` | mean and population standard deviation of episode rewards |
| `reward/<name>` | mean of one reward part over all episodes: a reward function's name (`reward/exact_match`), `env`, `overlong_penalty` |
| `reward/success_rate` | share of episodes with `reward >= logging.success_threshold` |
| `reward/pass_at_k` | share of prompt groups with at least one successful episode; only when a group has 2 or more episodes |
| `episode/turns_mean`, `episode/turns_max` | policy turns per episode |
| `episode/done_frac`, `episode/truncated_frac` | share of episodes the environment ended; share whose last completion has finish reason `length` |
| `episode/context_tokens_mean`, `episode/context_tokens_max` | largest prompt plus completion token count reached in an episode |
| `episode/tool_calls_mean`, `episode/invalid_calls_mean` | per-episode counts reported by the environment; only when a turn reports them |
| `episode/threads_mean` | conversations per episode; only when an episode has more than one |
| `episode/context_exhausted_frac` | share of episodes that ran out of context budget; only when above 0 |

Rollouts and batch:

| Metric | Meaning |
|---|---|
| `rollout/num_sequences` | episodes the step trains on |
| `rollout/groups_dropped`, `rollout/resample_rounds` | `algorithm.dynamic_sampling`: groups with identical rewards dropped; extra sampling rounds |
| `rollout/skipped` | 1 when no episode was left and the update was skipped |
| `rollout/failed_episodes`, `rollout/failed_frac`, `rollout/groups_incomplete` | only with `env.max_failed_frac > 0`: failed episodes (count, share); groups dropped because failures left one member |
| `policy/staleness` | weight syncs between sampling and training a batch (0 without `rollout.async_steps_ahead`) |
| `tokens/response_mean`, `tokens/response_max` | response tokens per sequence |
| `tokens/prompt_mean` | tokens before the first trained completion of a sequence |
| `tokens/seq_mean`, `tokens/total` | non-padding tokens per sequence, and in the batch (RL only) |
| `tokens/prefix_mismatch` | episodes (SFT: samples) that needed more than one sequence or a fallback tokenisation |
| `tokens/sequences`, `tokens/truncated` | rows of the batch; rows cut at `trainer.max_seq_len` |
| `tokens/truncated_turns` | SFT: trained turns that lost tokens to the cut |
| `tokens/old_logprobs_recomputed` | 1 when the trainer recomputed the old log-probs instead of using the engine's |
| `advantage/mean`, `advantage/abs_mean` | mean and mean magnitude of the advantage over response tokens |
| `advantage/turn_spread`, `advantage/turn_spread_max`, `advantage/first_turn_abs`, `advantage/last_turn_abs` | per-turn credit, see [CREDIT_ASSIGNMENT.md](CREDIT_ASSIGNMENT.md) |

Loss and optimisation (`src/monorl/algorithms/losses.py`, `src/monorl/trainer/reference.py`):

| Metric | Meaning |
|---|---|
| `loss/policy` | clipped policy-gradient surrogate |
| `loss/kl` | KL penalty to the reference policy before `algorithm.kl_coef`; 0.0 when `kl_coef` is 0 |
| `loss/entropy` | mean token entropy of the policy, before `algorithm.entropy_coef` |
| `loss/total` | the optimised loss: policy + `kl_coef` * kl - `entropy_coef` * entropy, plus the value and MTP terms when present |
| `policy/clipfrac` | share of response tokens where the clipped surrogate is the active one |
| `policy/approx_kl`, `policy/ratio_mean` | mean of `(ratio - 1) - log ratio` (k3 estimate of KL(old, new)); mean ratio new/old |
| `grad_norm` | gradient norm of the policy (and value head) before clipping to `trainer.max_grad_norm` |
| `lr` | learning rate after the scheduler step |
| `loss/sft`, `sft/ppl` | SFT loss (label smoothing included); `exp` of the unsmoothed token-mean NLL |
| `sft/tokens`, `sft/turns` | SFT response tokens and trained assistant turns in the batch |
| `data/epoch` | SFT: 0-based pass over the dataset the step draws from |
| `loss/value`, `value/clipfrac`, `value/mean`, `value/return_mean`, `value/explained_variance` | `algorithm.name: ppo`, see [CREDIT_ASSIGNMENT.md](CREDIT_ASSIGNMENT.md) |
| `loss/mtp`, `loss/mtp_<k>`, `mtp/accuracy`, `mtp/accept_rate`, `mtp/accept_length` | MTP training and drafting, see [MTP.md](MTP.md) |

Off-policy corrections, all defined in [OFF_POLICY.md](OFF_POLICY.md): `offpolicy/weight_mean`,
`offpolicy/weight_max`, `offpolicy/truncated_frac`, `offpolicy/masked_frac`, `offpolicy/ess`,
`offpolicy/kl`, `offpolicy/kl_k3`, `offpolicy/skipped`, `offpolicy/sc_head_mass`,
`offpolicy/sc_coef_l1`, `offpolicy/sc_tail_ratio`.

Timing and weight sync (seconds unless noted):

| Metric | Meaning |
|---|---|
| `time/rollout`, `time/reward` | generating the step's episodes; scoring them |
| `time/train`, `time/values` | log-prob passes plus all `train_step` calls; `ppo`: the value pass before them |
| `time/sync` | weight sync as seen by the loop |
| `time/step` | training, sync and checkpoint save (SFT: encoding too); no rollouts, rewards, evaluation |
| `weight_sync/bytes`, `weight_sync/seconds` | bytes of weights pushed; duration inside the sync |
| `weight_sync/save_seconds`, `weight_sync/export_seconds`, `weight_sync/chunks`, `weight_sync/buckets`, `weight_sync/tensors` | extra timings and unit counts of the checkpoint, Ray object-store, broadcast and CUDA IPC syncs ([DISTRIBUTED.md](DISTRIBUTED.md); not run here) |

Evaluation (`eval` rows):

| Metric | Meaning |
|---|---|
| `eval/reward_mean`, `eval/reward_std` | as `reward/mean`, `reward/std` |
| `eval/reward/<name>` | as `reward/<name>` (`eval/reward/exact_match`) |
| `eval/success_rate`, `eval/pass_at_k` | as `reward/success_rate`, `reward/pass_at_k` |
| `eval/turns_mean`, `eval/turns_max`, `eval/done_frac`, `eval/truncated_frac`, `eval/context_tokens_mean`, `eval/context_tokens_max` | as the `episode/` metrics of the same name |
| `eval/tool_calls_mean`, `eval/invalid_calls_mean`, `eval/threads_mean`, `eval/context_exhausted_frac` | as the `episode/` metrics, same conditions |
| `eval/mtp/accept_rate`, `eval/mtp/accept_length` | MTP drafting during evaluation |
| `eval/failed_frac` | share of evaluation episodes that failed; only with `env.max_failed_frac > 0` |
| `eval/num_samples`, `eval/time` | episodes scored (SFT: samples); wall time of the evaluation |
| `eval/loss`, `eval/nll_token`, `eval/ppl`, `eval/tokens` | SFT only: NLL aggregated per `algorithm.loss_agg`, token-mean NLL, its `exp`, response tokens scored |

## Evaluation

Evaluation runs every `trainer.eval_every` steps when `data.eval_uri` is set. The set is loaded
once, on first use, with the run's `data` section where `uri` is replaced by `data.eval_uri`,
`split` by `data.eval_split` (default `test`) and `shuffle` is false. Every other `data.*` key
applies unchanged, including `limit` and the field mapping (see [DATA.md](DATA.md)).

In the RL loop every evaluation sample gets one episode in the run's environment, generated by the
current policy on the training engines with temperature 0.0, `top_p` 1.0, no top-k and the run's
`rollout.max_new_tokens` and `rollout.stop`, and scored with the same `rewards`. An episode is a
success when its total reward is at least `logging.success_threshold` (default 1.0); the same
threshold defines `success` in `trajectories.jsonl` and `reward/success_rate`. pass@k is the share
of groups with at least one success: in training, `reward/pass_at_k` with k = `rollout.group_size`.
Evaluation groups have a single episode, so the native loop does not emit `eval/pass_at_k`.

In the SFT loop evaluation generates nothing: it scores the target responses of the evaluation set
with the current weights and reports `eval/loss`, `eval/nll_token`, `eval/ppl` and `eval/tokens`.

## Logging sinks

`metrics.jsonl` is always written. `src/monorl/orchestrator/sinks.py` adds two optional sinks to
the native loops; each receives the numeric metrics of every `metrics.jsonl` row, at the same step.

| Key (default) | Package | Writes |
|---|---|---|
| `logging.wandb_project` (unset) | `wandb` | a W&B run in that project, named `run_name`, with the full config (`dir` is the run directory, `resume="allow"`); metrics under their names |
| `logging.tensorboard` (false) | `tensorboard` | event files in `<run_dir>/tensorboard`; metrics under their names |

Neither package is a dependency of monorl. When the package is missing or initialisation fails the
sink is skipped and the run continues, as in this run without the package (logging to an
installed W&B or TensorBoard was not run here):

```text
... logging.tensorboard is set but `tensorboard` is not installed; skipping
```

## Pipeline backends and pipelines

With `trainer.backend: trl | verl | nemo_rl` the framework owns the loop and writes through the
same `RunReporter`, so `config.yaml`, `status.json`, `run.log` and its metrics land in the same
run directory. Steps, metric names, checkpoints, cancelling and `--resume` (which `run()` does
not forward to pipeline backends) are backend-specific: see [BACKENDS.md](BACKENDS.md).

Multi-stage pipelines write one such run directory per stage: see [PIPELINES.md](PIPELINES.md).
