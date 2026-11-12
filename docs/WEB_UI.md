# Web UI

`monorl serve` starts a FastAPI server (`monorl.server.app:create_app`) on
`http://127.0.0.1:8000`, serves the built React UI from `web/dist` at `/` and opens a browser
tab (`--no-open` to skip). The UI is a low-code front end over `RunConfig`: it generates forms
from `RunConfig.model_json_schema()`, so every field, description and default comes from
`src/monorl/config/schema.py` — nothing is hard-coded twice.

```
monorl serve [--host 127.0.0.1] [--port 8000] [--no-open] [--reload]
make web          # pnpm install + build web/dist (needed once; `monorl serve` works API-only without it)
```

Development: `cd web && pnpm dev` runs Vite on `:5173` proxying `/api` to the Python server
(CORS is enabled for both origins). `pnpm dev:mock` serves a mock of the API on `:8000` instead
(`web/mock/server.ts`): seeded runs and pipelines with fake metrics, logs and rollouts, including
`tool-calling-grpo-seed`, whose trajectories cover every part of the transcript view below, and
`gsm8k-grpo-smol-seed`, whose rows and metrics predate transcripts.

## Pages

**Runs** (`/`) — every job the server launched, with kind (`run` / `pipeline`), state, step
(stages finished for pipelines), last reward and run dir.
Jobs survive server restarts: they persist in `$MONORL_HOME/jobs.json` (default `~/.monorl`),
and each run is a subprocess `python -m monorl.cli train --config <run_dir>/config.yaml
--run-dir <run_dir>` whose progress is read back from the run directory files.

**New run** (`/new`) — a wizard that produces one `RunConfig`, or — with the *Pipeline (SFT → RL
loop)* toggle at the top — a `PipelineConfig` whose `base` is the same wizard config:

1. *Recipes* — start from any `configs/recipes/*.yaml` (`GET /api/recipes`; the leading `# `
   comment is the description) or from scratch.
2. *Model* — search the Hugging Face Hub (`/api/models/search`, text-generation models), see
   architecture / parameter count / gated status (`/api/models/info`), or type a local path.
   LoRA, dtype, attention implementation, chat template override.
3. *Data* — pick a source:
   * **Local** — browse directories on the server's filesystem (`/api/datasets/browse?uri=`).
   * **S3** — browse `s3://bucket/prefix` with credentials from the server's `AWS_*` environment;
     the connector panel (`/api/connectors`) shows region/endpoint and which variables are set,
     never values. Works with MinIO/R2 through `AWS_ENDPOINT_URL` (see [`DATA.md`](DATA.md)).
   * **HF Hub** — dataset search (`/api/datasets/search`), config/subset and split.
   Then *Preview* (`POST /api/datasets/preview`): columns with dtypes, first rows, the inferred
   layout, and a column mapper writing `prompt_field` / `answer_field` / `messages_field` /
   `tools_field` (the column with a chat row's tool schemas) / `image_field` (comma list for
   several columns; auto-filled from the sniffer's `image_fields`) and, for SFT, `response_field`. Image cells render as a `[image W×H]` chip
   (decoded images) or a thumbnail (`http(s)://…png|jpg|webp`, `data:image/…` URIs).
4. *Rewards & environment* — add weighted reward functions (names and parameter hints come from
   `GET /api/system` components) and choose the env type; remote MEP/1 envs list the example
   binaries discovered under `envs/bin` (`GET /api/envs`). `harness` (an external agent program
   is the environment and calls the policy through the OpenAI-compatible gateway) has its own
   form: *Command* (required; selecting the type prefills the bundled example agent,
   `{python} -m monorl.gateway.agents.calc`), *Timeout*, *Model name*, an optional *Verifier*
   with its timeout, and the remaining params (`env`, `workdir: {root, template}`) as JSON. A
   command edited there is stored as one string, which the env splits like a shell would. The
   form lists the command placeholders (`{task_file}`, `{workdir}`, `{base_url}`, …) and the
   variables the agent gets (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MONORL_TASK_FILE`); for this
   type `env.max_turns` reads *Max policy calls*. Its *Sandbox* section picks where the agent and
   the verifier run (`env.params.sandbox`): `local`, or `docker` with an *Image* or a *Build
   context* / *Dockerfile* and the limits *Network*, *CPUs*, *Memory*, *User*. *Other params
   (JSON)* shows every other key, the whole `sandbox` mapping included, so options without a
   field (`mounts`, `gpus`, `gateway`, …) and keys the form does not know are kept as they are.
   Options left behind under the `local` backend, an image together with a build and
   `network: none` are flagged inline. Leaving the rewards empty is flagged unless the
   environment can score itself (remote / openenv / hermes, or a harness verifier or posted
   reward). Every environment type has *Retries* (`env.retries`) and *Max failed fraction*
   (`env.max_failed_frac`); the card flags `max_failed_frac > 0` with a pipeline backend
   (trl / verl / nemo_rl) and `retries > 0` with verl / nemo_rl.
5. *Training* — algorithm (grpo/dapo/rloo/reinforce/ppo/**sft**) and its knobs, rollout engine +
   sampling, trainer + optimizer, logging. Selecting `sft` hides the rollout / rewards / env
   sections (they are ignored by the SFT loop) and surfaces `sft_label_smoothing`,
   `sft_train_on_prompt`, `sft_turns` (*Turns trained*: the `last` assistant turn or `all`) and
   `trainer.eval_every`. For RL runs the *Algorithm* card also holds
   `rollout.turn_rendering` (`tokens` / `template`: how multi-turn prompts are built),
   `algorithm.loss_agg`, whose `rollout` option weighs every episode equally, the *Context
   budget* (`rollout.max_context_tokens`, next to *Max new tokens*; empty = `trainer.max_seq_len`;
   flagged when it exceeds `trainer.max_seq_len` or the trainer is verl / nemo_rl) and *Credit
   assignment* (`algorithm.credit_assignment`: `episode` / `turn`; `turn` is flagged with a
   pipeline backend and shows the discount `gamma`). `ppo` runs on every trainer and shows
   `lam`, `value_coef`, `value_clip` (empty = plain squared error) and `gamma`. The *Logging* card
   holds `logging.save_trajectories`, `trajectories_per_step`, `save_transcripts` (store the full
   conversation and per-turn details of each saved rollout; off leaves only prompt / response)
   and `success_threshold` (the total reward from which an episode counts as solved). The
   *Off-policy corrections* card (RL only) holds `rollout.async_steps_ahead` and
   `algorithm.off_policy`: the proximal `anchor`, `importance_weights` (with `importance_cap`
   once on), `ratio_mask` (with `ratio_mask_low` / `ratio_mask_high` once on) and
   `score_centering` (with `score_centering_top_k` once on). The two schema rules show inline:
   importance weights / ratio mask do not combine with `anchor: behavior`, and score centering
   does not combine with `anchor: trainer`, nor with importance weights / ratio mask under
   `auto`.
6. *Compute* — local vs Ray, GPUs, placement, weight sync — pre-filled from the doctor report.
7. *Stages* (pipeline mode only) — ordered stage list (add / remove / reorder; *R1 loop* inserts
   cold-start SFT → GRPO → rejection sampling → SFT → GRPO). Field help comes from
   `GET /api/config/pipeline_schema`. Train stages edit their `overrides` as YAML with
   `init_from` / `data_from` selectors; the `data_from: previous` and unique-name rules are
   checked inline before the server sees the document; choosing `data_from: previous` notes
   that an SFT stage then trains every policy turn of the sampled conversations
   (`algorithm.sft_turns: all`). Rejection-sampling stages count episodes: *Episodes / prompt*
   (`num_samples`), *Keep / prompt*, *Max turn tokens* (`max_response_tokens`) and *Max context
   tokens* (`max_context_tokens`). Eval stages have *Samples / prompt*
   (`num_samples`, default 1 = greedy; more are sampled at the rollout temperature and add
   `eval/pass_at_k`).
8. *Review* — the YAML (editable; a pasted `base:` + `stages:` document switches the pipeline
   fields), live validation (`POST /api/config/validate` with `{config}` or `{pipeline}` →
   `{ok, errors:[{loc,msg}]}`; a server without pipeline validation yields client-side structural
   checks, flagged as such) and **Launch** (`POST /api/jobs` with `{config}` or `{pipeline}`).

**Run detail** (`/runs/:id`) — status, config, live metric charts (`reward/mean`,
`reward/success_rate`, `episode/turns_mean`, `loss/policy`, `loss/kl`, token counts; every other
numeric metric, e.g. `reward/pass_at_k`, `episode/tool_calls_mean`,
`episode/context_exhausted_frac` (episodes that ran out of context), `rollout/failed_frac`
(episodes left out as failed, with `env.max_failed_frac > 0`), `policy/clipfrac` and the
timings, is behind *Show more metrics*), log tail, sampled trajectories, checkpoints list, and
*Cancel* (`POST /api/jobs/{id}/cancel`, sends `SIGTERM`; the run ends as `cancelled`). Charts, log
and trajectories arrive over one SSE connection, `/api/jobs/{id}/events` (events `metrics`, `log`,
`trajectory`). SFT runs chart `loss/sft`, `sft/ppl`, `eval/loss`. A chart whose metric the run
does not log (older runs, backends without agent metrics) reads *no data yet*.

*Trajectories* lists the latest 200 saved rollouts, newest first: step, prompt, response, turns,
a solved mark (`success`: ✓ / ✗, `–` on rows that do not record it), reward and response tokens.
The *step* filter narrows the list to one step; a *phase* filter and column (`train` / `eval` /
`rejection` / `sft`) appear once the rows carry more than one phase. Clicking a row expands the
episode (one row at a time; only that one is rendered):

* a summary — phase, solved, done, the last finish reason, turns, response and context tokens,
  the reward with its breakdown, the reference, group / sample ids and the episode `info`. A
  *context exhausted* badge marks an episode that ran out of its context budget
  (`info.context_exhausted`), *N calls over the context budget* an agent whose policy calls were
  refused for it (`info.context_exceeded_calls`);
* the transcript, when the row has `threads`: each thread as a chat with messages styled by role.
  Several threads become tabs: *Final conversation* is thread 0, the one the episode ended in;
  *Thread N* are its other contexts (the conversation before a compaction, a sub-agent). Image
  parts show as a thumbnail (URLs, `data:image/…`) or a chip (file paths, images the backend left
  out, with its note);
* every policy turn, highlighted on its assistant message (`turns[i].message_index`) with the turn
  number, per-turn reward, finish reason and prompt / response token counts. For tool-using
  environments the turn also lists its `info.tool_calls` (name, arguments, result), the number of
  invalid calls and whether it was the final answer; other step info is under *step info*. `tool`
  messages show the tool name and call id. A turn without a recorded message is listed after the
  messages of its thread;
* rows without `threads` (older runs, `logging.save_transcripts: false`, backends that do not
  record them) show the prompt and the response side by side instead.

Long system prompts, tool output and call results start collapsed (*Show all* / *Collapse*), and
the conversation scrolls inside a fixed height, so one long episode does not stretch the table.

Pipeline jobs (`kind: pipeline`) show the stage table (index, name, type, state, step, summary,
duration; the running stage is followed until you pick one) and a status card with stages
finished / total. The table is the pipeline manifest, pushed over SSE from
`/api/jobs/{id}/stages/stream` whenever a stage changes. Selecting a stage streams its metrics,
log and trajectories from `/api/jobs/{id}/stages/{index}/events` into the same charts / log /
trajectory components (its checkpoints come from `/stages/{index}/checkpoints`); a stage that has
not started yet shows empty charts that fill in once it does, and switching stages closes the
previous stage's stream. Rejection-sampling stages show sampled / accepted counts and the
accept-rate chart (`rs/accept_rate`); eval stages chart `eval/reward_mean`, `eval/success_rate`
and `eval/turns_mean`.

A live run view therefore holds one `EventSource` and a live pipeline view two (manifest +
selected stage), which leaves room under the browser's six-connections-per-origin limit for
several tabs. The badge next to the tabs reads *live (SSE)* while the events stream is open; if
it fails (a proxy that buffers or drops event streams, an older server) the view closes it and
polls the JSON routes (`…/metrics?since=N`, `…/logs?tail=N`, `…/trajectories?since=N`) from the
rows it had reached, and the badge reads *polling*. The manifest falls back to
`GET /api/jobs/{id}/stages` the same way. The job header and the checkpoints list are polled.

Run logs: the server spawns the trainer with `MONORL_REPORTER_ECHO=0` and redirects its
stdout/stderr into `<run_dir>/run.log`, so the reporter writes each line to `run.log` once instead
of also echoing it to the console (the default, `1`, when you run `monorl train` yourself). The
log tail and `/logs/stream` read that file.

**System** (`/system`) — the `monorl doctor` report: Python, torch (CUDA/MPS), Go/C++ toolchains,
installed frameworks with patch status, registered components per namespace.

## Files and environment

| | |
|---|---|
| `MONORL_HOME` | jobs database and default run root (`~/.monorl`) |
| `logging.run_root` | where run directories go (`<run_root>/<run_name>`) |
| `MONORL_RECIPES_DIR`, `MONORL_WEB_DIST`, `MONORL_ENVS_BIN`, `MONORL_ENVS_SDK` | override discovery paths |
| `AWS_*`, `HF_TOKEN` | data connectors (read by the server process) |

The server never imports a training backend: it writes `config.yaml`, spawns the CLI, and reads
`status.json`, `metrics.jsonl`, `trajectories.jsonl`, `run.log` and `checkpoints/`
(`src/monorl/core/reporting.py`). Anything the UI shows can therefore also be tailed from the
run directory or re-launched with `monorl train --config <run_dir>/config.yaml`.
