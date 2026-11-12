# Multi-stage pipelines

A pipeline chains runs the DeepSeek-R1 way: cold-start SFT → GRPO → rejection sampling with the
RL policy → SFT on the accepted samples → GRPO → eval. One `PipelineConfig`
(`src/monorl/config/pipeline.py`) holds a **base** `RunConfig` plus an ordered list of **stages**;
every stage runs in its own standard run directory under `<pipeline>/stages/`, so the CLI, API and
UI can inspect it with the same readers as a plain run.

```bash
monorl pipeline --config configs/recipes/r1_sft_grpo_loop_cpu.yaml            # CPU, SmolLM2-135M, GSM8K
monorl pipeline --config configs/recipes/r1_sft_grpo_loop_8gpu_vllm.yaml      # 8 GPUs, Qwen2.5-7B, vLLM colocated
monorl pipeline --config configs/recipes/distill_tool_calling_cpu.yaml        # CPU, Qwen2.5-0.5B: distil tool-using episodes
monorl pipeline --config my_loop.yaml --set base.trainer.max_steps=5 --run-dir runs/loop --resume
monorl train    --config my_loop.yaml      # a yaml with `base:` + `stages:` is dispatched to the pipeline runner
```

## Config

```yaml
name: r1-loop                     # pipeline run name (default <model>-pipeline-<timestamp>)
base: { ...RunConfig... }         # model, data, algorithm, rollout, trainer, rewards, env, cluster, logging
stages:
  - type: train                   # a full run: base deep-merged with `overrides`
    name: cold_start_sft
    init_from: base               # base | previous (default: last train stage's final/) | <path>
    overrides: {algorithm: {name: sft}, data: {response_field: answer}, trainer: {max_steps: 20}}
  - type: train
    name: grpo_1                  # init_from: previous → starts from cold_start_sft/final
    overrides: {trainer: {max_steps: 10}}
  - type: rejection_sample
    name: rejection_sample
    num_samples: 4                # episodes per prompt with the current policy
    min_reward: 1.0               # keep episodes whose total reward >= this
    max_per_prompt: 1             # best-first cap per prompt
    require_finish: true          # drop episodes in which a turn was cut off (length) or failed
    dedupe: true                  # drop episodes whose conversation repeats a kept one of the prompt
    max_response_tokens: null     # drop episodes in which a policy turn is longer than this
    max_context_tokens: null      # drop episodes whose longest context (prompt + completion) exceeds this
    max_prompts: 64               # sample from at most this many prompts
    data: null                    # DataConfig of the prompts (default: base data)
    rollout_overrides: {temperature: 0.8}
    reward_overrides: null        # replace base rewards for scoring
    mix: [hf://openai/gsm8k]      # extra SFT datasets appended to the output
    output: null                  # default <stage dir>/dataset.jsonl
  - type: train
    name: sft_on_samples
    data_from: previous           # data.uri = the preceding rejection_sample output (chat format);
                                  # an SFT stage then trains every policy turn (algorithm.sft_turns: all)
    overrides: {algorithm: {name: sft}}
  - type: eval
    name: eval                    # episodes on `data` (default: base data.eval_uri, else data)
    max_prompts: 32
    num_samples: 1                # 1 = greedy; k > 1 = k sampled completions per prompt (pass@k)
    rollout_overrides: {}         # e.g. {temperature: 0.8} for the sampled completions
```

An eval stage with `num_samples: 1` decodes greedily. With `num_samples: k > 1` it samples `k`
completions per prompt at the rollout temperature (after `rollout_overrides`), groups them per
prompt and additionally reports `eval/pass_at_k`, the fraction of prompts with at least one
completion whose reward reaches `logging.success_threshold`; `eval/reward_mean` and the other
`eval/*` metrics are then taken over all completions and `eval/num_samples` counts completions.

Rules checked at load time: stage names are unique; `data_from: previous` needs a
`rejection_sample` stage right before it. All stage configs are also built once with placeholder
hand-offs before anything runs, so an override typo in stage 5 fails immediately.

### How a train stage's config is built

1. `overrides` are deep-merged over `base` (dicts merge, lists/scalars replace).
2. `init_from` sets `model.name_or_path`: `base` → the base model; `previous` → the previous train
   stage's `final/` export (the merged model when no train stage ran yet); a string → that path.
3. `data_from: previous` replaces `data` with the rejection output in chat format
   (`format: chat`, `messages_field: messages`, no prompt/answer/response field, no
   `prompt_template`; `shuffle` and `limit` are kept). The rows are the conversations as the policy
   had them, so nothing is rewritten on the way in: `system_prompt` is cleared (a row starts with
   the system prompt it was sampled with, which for a tool environment lists the tools) and
   `max_prompt_tokens` is raised to the context budget of the sampling stage, so no conversation
   is dropped for the length of what precedes its final turn. The data layer strips the trailing
   assistant turn into `metadata["assistant_turn"]` (the SFT target) and puts the row's `tools`
   into `metadata["tools"]`.
4. An SFT stage (`algorithm.name: sft`) with `data_from: previous` gets `algorithm.sft_turns: all`:
   every policy turn of a conversation is trained, not only its last one. It is written to the
   stage's `config.yaml`; set `algorithm.sft_turns` in the stage's `overrides` to decide otherwise.
   An SFT stage builds no environment, so an `env` it would inherit from the base (a
   `tool_calling` or `harness` pipeline) is reset to the default single-turn one unless the
   overrides set `env`.
5. `run_name = <index>-<name>`, `logging.run_root = <pipeline>/stages`.

Sampling stages (rejection/eval) use `base` with `rollout_overrides` merged into `rollout`, the
current policy as the model, optionally their own `data` and (rejection) `reward_overrides`. They
build only a rollout engine from the registry — no trainer — and shut it down when done, so the
next stage can load its own policy.

### Rejection sampling output

`dataset.jsonl`, one chat row per conversation of an accepted episode (plus mixed-in rows). A
single-turn episode is the prompt and the sampled response:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
              {"role": "assistant", "content": "<sampled response>"}],
 "reference": "42", "reward": 1.0, "source": "sampled", "sample_id": "...", "thread": 0, "num_turns": 1}
```

A multi-turn or tool-using episode is its whole conversation, in order:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "What is 17*23?"},
              {"role": "assistant", "content": "<tool_call>\n{\"name\": \"calculator\", ...}\n</tool_call>"},
              {"role": "tool", "name": "calculator", "tool_call_id": "call_3f2a", "content": "391"},
              {"role": "assistant", "content": "17*23 = 391"}],
 "tools": [{"type": "function", "function": {"name": "calculator", "description": "...", "parameters": {}}}],
 "reference": "391", "reward": 1.0, "source": "sampled", "sample_id": "...", "thread": 0, "num_turns": 2}
```

- Every policy turn is an assistant message holding exactly the text the policy sampled
  (`Turn.completion.text`). For `env.type: harness` the conversation stores the parsed form of an
  answer (`tool_calls`); the row has the text the policy wrote instead, tool-call markup included.
- Tool results keep `name` and `tool_call_id`. Messages after the last policy turn (a tool result
  nobody answered, an unanswered user turn) are dropped.
- `tools` are the schemas the episode's prompts were rendered with (`Trajectory.info["tools"]`);
  the key is absent without tools. SFT passes them to the chat template again, so the prompts it
  trains on are the prompts the policy saw.
- An episode with several conversations (`Trajectory.extra_threads`: a sub-agent, the history
  before a compaction) gives one row per conversation that holds policy turns, `thread` naming it
  (0 = the conversation the episode ended in). `num_turns` counts the policy turns of the row;
  `reward`, `reference` and `sample_id` are the episode's.
- An assistant message that is not a policy turn of its row is context and carries `"weight": 0`
  (the OpenAI fine-tuning convention; SFT with `sft_turns: all` does not train it): a turn two
  conversations share (a retry from an earlier point) is written as sampled in both and trained in
  the conversation it belongs to; an assistant message the agent supplied itself (a worked
  example) is kept as it is, with `tool_calls` in the OpenAI shape.

The filters judge an episode as a conversation: `require_finish` and `max_response_tokens` look at
every policy turn, `max_context_tokens` at the largest prompt + completion a turn reached,
`dedupe` compares the conversations (ignoring whitespace around assistant text and tool-call ids,
which are often random), and an episode without a policy turn is never kept.

`mix` entries are dataset URIs. A uri that the pipeline already describes (the base data or a train
stage's `data` overrides) reuses that field mapping — `hf_config_name`, `prompt_field`,
`response_field`, `system_prompt`, `limit` — so the cold-start set can be mixed back in verbatim;
other uris are opened with auto-sniffed columns and the base `system_prompt`. The target response
of a mixed sample is `metadata["response"]` (`response_field`), else the trailing assistant turn,
else the `response_field`/`answer_field` column, else the raw answer, else a string reference.
Mixed rows have `"reward": null, "source": "mix"` and the sample's `tools` when it has any. Total
reward = reward-function score + discounted env return, as in the RL loop. If nothing is accepted
and nothing is mixed the stage fails (lower `min_reward` or raise `num_samples`).

### SFT on agent conversations — `algorithm.sft_turns`

| value | loss on | use |
| --- | --- | --- |
| `last` (default) | the final assistant turn; earlier turns are context | prompt → response datasets |
| `all` | every assistant turn, except those a row marks `"weight": 0` | multi-turn and tool-using conversations; set by `data_from: previous` |

The rest of the conversation (system prompt, user turns, tool results, the tool schemas rendered
by the chat template) is context in both. `algorithm.sft_train_on_prompt` puts the loss on
everything either way.

A turn's loss span is taken from the chat template, token for token: the chat before the turn is
rendered with a generation prompt (the prompt a rollout engine would build), the chat up to and
including the turn without one, and the difference is the span, i.e. the turn's text and the
template's end-of-turn. All turns of a conversation are spans of **one** sequence as long as each
turn's prompt starts with the tokens of the sequence so far. Two things can break that, and both
are counted, never silent:

- The template renders earlier turns differently once more messages follow (Qwen3 / DeepSeek-R1
  drop the reasoning of earlier turns). Such a turn starts a sequence of its own, with the history
  as the template gives it as context; the earlier turns keep their loss in the previous sequence.
  Every trained turn is a loss span in exactly one sequence; the sequences of a conversation share
  its group id.
- The prompt is not a prefix of the closed turn at all. The turn's text plus EOS is tokenised on
  its own and appended to the prompt.

Metrics of an SFT step, next to `loss/sft`, `sft/ppl`, `sft/tokens`: `sft/turns` (assistant turns
that carry the loss), `tokens/sequences` (training sequences), `tokens/prefix_mismatch`
(conversations that are not one template-exact sequence, by either fallback), `tokens/truncated`
(sequences cut from the right at `trainer.max_seq_len`) and `tokens/truncated_turns` (trained turns
that lost tokens to it). The first truncation is also written to the run log, as is the number of
rows the data layer dropped for `data.max_prompt_tokens`.

Pipeline backends use their own mechanism:

| backend | `last` | `all` |
| --- | --- | --- |
| `trl` (1.17) | `prompt` / `completion` rows, `completion_only_loss` | `messages` rows, `assistant_only_loss` (the assistant mask of a chat template with `{% generation %}` markers; `SFTTrainer` swaps one in for the templates TRL knows and refuses other templates without them) |
| `verl` (0.9.1) | `MultiTurnSFTDataset` trains every assistant message and has no other mode: rows with one assistant turn only, a row with earlier assistant turns is a `ConfigError` | `MultiTurnSFTDataset` |
| `nemo_rl` (0.7.0) | `sft.only_unmask_final: true` | `sft.only_unmask_final: false` |

On NeMo-RL, SFT over conversations with several assistant turns therefore trains the last turn by
default instead of all of them, like the native loop (rows with one assistant turn are
unaffected). All three get the row's `tools` (a JSON string column for TRL, a `tools` column for
verl's `data.tools_key` and NeMo-RL's `sft_processor`). They cannot leave single assistant turns
out, so a `"weight": 0` turn under `all` is a `ConfigError` there; verl and NeMo-RL load their rows
through Arrow, so tool schemas that differ in shape between rows (Arrow would add null keys), and
for verl rows with and without tools in one dataset, are a `ConfigError` too.

## Run directory

```
<run_root>/<name>/
  pipeline.yaml              resolved PipelineConfig
  status.json                {state, message: "stage i/n <name> (<type>)", step: <stages finished>, ...}
  pipeline.json              manifest (below)
  run.log
  stages/00-cold_start_sft/  standard run dir: config.yaml status.json metrics.jsonl trajectories.jsonl checkpoints/ final/
  stages/01-grpo_1/
  stages/02-rejection_sample/  config.yaml status.json metrics.jsonl trajectories.jsonl dataset.jsonl run.log
  stages/03-sft_on_samples/
  stages/04-grpo_2/
  stages/05-eval/            config.yaml status.json metrics.jsonl trajectories.jsonl run.log
```

`pipeline.json`:

```json
{"name": "...", "run_dir": "...", "state": "running|succeeded|failed|cancelled",
 "final_model_path": ".../stages/04-grpo_2/final", "error": null, "created_at": 0, "updated_at": 0,
 "stages": [{"index": 0, "name": "cold_start_sft", "type": "train", "state": "succeeded",
             "run_dir": ".../stages/00-cold_start_sft", "started_at": 0, "finished_at": 0,
             "step": 20, "summary": {"loss/sft": 1.2, "final_model_path": "..."}, "error": null}, ...]}
```

Stage `summary`: train → the last metrics row (floats) + `final_model_path`; rejection →
`{prompts, sampled, accepted, accept_rate, mixed, rows, dataset, "reward/mean"}` (`sampled` and
`accepted` count episodes, `rows` the rows written: one per conversation of an accepted episode
plus the mixed ones); eval → the `eval/*` metrics.

Metrics written by sampling stages: rejection `rs/prompts`, `rs/sampled`, `rs/accepted`,
`rs/accept_rate`, `reward/mean`, `reward/std`, `reward/<name>`, `tokens/response_mean`,
`time/rollout` per prompt chunk (`step` = chunk index, phase `rejection`; chunk size =
`trainer.prompts_per_step` or `rollout.extra.rejection_prompts_per_step`) and a final phase
`summary` row with totals; eval `eval/reward_mean`, `eval/reward_std`, `eval/reward/<name>`,
`eval/num_samples`, `eval/time` and the agent metrics `eval/success_rate`, `eval/pass_at_k` (only
with `num_samples > 1`), `eval/turns_mean`, `eval/turns_max`, `eval/done_frac`,
`eval/truncated_frac`, `eval/context_tokens_mean`, `eval/context_tokens_max` and, when reported,
`eval/tool_calls_mean`, `eval/invalid_calls_mean`, `eval/threads_mean` (phase `eval`; see
`docs/ARCHITECTURE.md`), plus `eval/chunk_reward_mean` per prompt chunk. Both stages write their
sampled episodes to `trajectories.jsonl` (phase `rejection` / `eval`, at most
`logging.trajectories_per_step` per chunk).

## Resume and cancellation

`--resume` reads `pipeline.json`, skips stages whose state is `succeeded` (reusing their `final/` /
`dataset.jsonl`; a stage whose outputs are gone is re-run) and resumes the first unfinished train
stage from its latest checkpoint. The stage list (names and types) must match the manifest.

SIGTERM/SIGINT cancel the pipeline: a train stage finishes its current step, checkpoints and reports
`cancelled` (the orchestrator's own handling); a sampling stage aborts its episodes, does not start
the next prompt chunk and discards its partial dataset. Generation in flight is aborted, not
finished: the episodes stop waiting for it at once, and an engine implementing `cancel()` (`hf`,
`vllm` and `sglang` all do) stops generating within one step. An engine without `cancel()` still completes the batch
it is running before the stage shuts it down, with the results dropped. An environment step in
flight finishes before its episode aborts. The stage and the pipeline end in state `cancelled`
(`final_model_path` is only set for a succeeded pipeline); later stages stay `pending`. A second
signal interrupts immediately.

## API and UI

- `POST /api/jobs {pipeline: PipelineConfig}` (or a bare `{base, stages}` dict) → job
  `kind: "pipeline"`, `config` = the PipelineConfig; the server spawns
  `python -m monorl.cli pipeline --config <run_dir>/pipeline.yaml --run-dir <run_dir>`
  (`MONORL_PIPELINE_CMD` overrides the command prefix). `POST /api/jobs/{id}/cancel` SIGTERMs it.
- `GET /api/jobs/{id}/stages` → `{job_id, kind, name, state, final_model_path, error, stages: [...]}` from `pipeline.json`
  (`stages: []` for plain runs); `GET /api/jobs/{id}/stages/{index}` → one stage + its `status.json`.
- `GET /api/jobs/{id}/stages/{index}/metrics|trajectories|logs|checkpoints` read that stage's run dir
  with the same query parameters and shapes as the job-level routes.
- Server-sent events (named events, a final `end` event), for live views:
  - `GET /api/jobs/{id}/stages/stream` → a `stages` event carrying the `GET /api/jobs/{id}/stages`
    payload, sent on connect and again whenever it changes; `end` `{state, next: null, error}`
    follows the last one once the job is terminal. It is a snapshot stream: no cursor.
  - `GET /api/jobs/{id}/stages/{index}/events?metrics=N&logs=N&trajectories=N` → the stage run
    dir's three append-only files on one connection: `metrics` / `log` / `trajectory` events (one
    row or line each, file order within a kind), each kind starting at its own cursor (default 0,
    same meaning as `since` on the JSON routes). `end` `{state, error, next: {metrics, logs,
    trajectories}}` comes once that stage (or the job) is terminal and the files are drained, with
    the stage's state, or the job's when the stage never finished. A stream opened for a `pending`
    stage stays open and starts emitting when the stage starts; an index the pipeline does not
    have is a 404.
  - `GET /api/jobs/{id}/events` is the same stream for the job's own run dir (a plain run's
    metrics, log and trajectories; a pipeline's top-level log).
- The UI holds two connections on a live pipeline (the manifest and the selected stage) and one
  on a live run, and falls back to polling the JSON routes if an `EventSource` fails.
- `GET /api/config/pipeline_schema`; `POST /api/config/validate {pipeline: ...}` → `{ok, errors, pipeline}`.
- `GET /api/recipes` entries carry `kind: "run" | "pipeline"`.

Python: `monorl.pipeline.run_pipeline(config, run_dir=None, *, resume=False) -> PipelineResult(state,
run_dir, stages, final_model_path, error)`; `monorl.pipeline.read_manifest(run_dir)`.
