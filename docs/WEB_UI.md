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
(CORS is enabled for both origins).

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
   `image_field` (comma list for several columns; auto-filled from the sniffer's
   `image_fields`) and, for SFT, `response_field`. Image cells render as a `[image W×H]` chip
   (decoded images) or a thumbnail (`http(s)://…png|jpg|webp`, `data:image/…` URIs).
4. *Rewards & environment* — add weighted reward functions (names and parameter hints come from
   `GET /api/system` components) and choose the env type; remote MEP/1 envs list the example
   binaries discovered under `envs/bin` (`GET /api/envs`).
5. *Training* — algorithm (grpo/dapo/rloo/reinforce/ppo/**sft**) and its knobs, rollout engine +
   sampling, trainer + optimizer, logging. Selecting `sft` hides the rollout / rewards / env
   sections (they are ignored by the SFT loop) and surfaces `sft_label_smoothing`,
   `sft_train_on_prompt` and `trainer.eval_every`.
6. *Compute* — local vs Ray, GPUs, placement, weight sync — pre-filled from the doctor report.
7. *Stages* (pipeline mode only) — ordered stage list (add / remove / reorder; *R1 loop* inserts
   cold-start SFT → GRPO → rejection sampling → SFT → GRPO). Field help comes from
   `GET /api/config/pipeline_schema`. Train stages edit their `overrides` as YAML with
   `init_from` / `data_from` selectors; the `data_from: previous` and unique-name rules are
   checked inline before the server sees the document.
8. *Review* — the YAML (editable; a pasted `base:` + `stages:` document switches the pipeline
   fields), live validation (`POST /api/config/validate` with `{config}` or `{pipeline}` →
   `{ok, errors:[{loc,msg}]}`; a server without pipeline validation yields client-side structural
   checks, flagged as such) and **Launch** (`POST /api/jobs` with `{config}` or `{pipeline}`).

**Run detail** (`/runs/:id`) — status, config, live metric charts (`reward/mean`, `loss/policy`,
`loss/kl`, `policy/clipfrac`, token counts, timings), log tail, sampled trajectories with prompt /
response / reward breakdown, checkpoints list, and *Cancel* (`POST /api/jobs/{id}/cancel`, sends
`SIGTERM`; the run ends as `cancelled`). Charts, log and trajectories arrive over one SSE
connection, `/api/jobs/{id}/events` (events `metrics`, `log`, `trajectory`). SFT runs chart
`loss/sft`, `sft/ppl`, `eval/loss`.

Pipeline jobs (`kind: pipeline`) show the stage table (index, name, type, state, step, summary,
duration; the running stage is followed until you pick one) and a status card with stages
finished / total. The table is the pipeline manifest, pushed over SSE from
`/api/jobs/{id}/stages/stream` whenever a stage changes. Selecting a stage streams its metrics,
log and trajectories from `/api/jobs/{id}/stages/{index}/events` into the same charts / log /
trajectory components (its checkpoints come from `/stages/{index}/checkpoints`); a stage that has
not started yet shows empty charts that fill in once it does, and switching stages closes the
previous stage's stream. Rejection-sampling stages show sampled / accepted counts and the
accept-rate chart (`rs/accept_rate`); eval stages chart `eval/reward_mean`.

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
