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

**Runs** (`/`) — every job the server launched, with state, step, last reward and run dir.
Jobs survive server restarts: they persist in `$MONORL_HOME/jobs.json` (default `~/.monorl`),
and each run is a subprocess `python -m monorl.cli train --config <run_dir>/config.yaml
--run-dir <run_dir>` whose progress is read back from the run directory files.

**New run** (`/new`) — a wizard that produces one `RunConfig`:

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
   layout, and a column mapper writing `prompt_field` / `answer_field` / `messages_field`.
4. *Rewards & environment* — add weighted reward functions (names and parameter hints come from
   `GET /api/system` components) and choose the env type; remote MEP/1 envs list the example
   binaries discovered under `envs/bin` (`GET /api/envs`).
5. *Training* — algorithm (grpo/dapo/rloo/reinforce/ppo) and its knobs, rollout engine +
   sampling, trainer + optimizer, logging.
6. *Compute* — local vs Ray, GPUs, placement, weight sync — pre-filled from the doctor report.
7. *Review* — the YAML, live validation (`POST /api/config/validate` → `{ok, errors:[{loc,msg}]}`)
   and **Launch** (`POST /api/jobs`).

**Run detail** (`/runs/:id`) — status, config, live metric charts (`reward/mean`, `loss/policy`,
`loss/kl`, `policy/clipfrac`, token counts, timings; streamed over SSE from
`/api/jobs/{id}/metrics/stream`), log tail (`/logs/stream`), sampled trajectories with prompt /
response / reward breakdown, checkpoints list, and *Cancel* (`POST /api/jobs/{id}/cancel`, sends
`SIGTERM`; the run ends as `cancelled`).

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
