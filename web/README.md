# monorl web UI

The browser front end of `monorl serve`: a run list, a new-run wizard over `RunConfig` and
`PipelineConfig`, live run pages and a system report. This page is for people changing the UI;
what each page shows is in [docs/WEB_UI.md](../docs/WEB_UI.md).

Stack: React 19, React Router 7, Recharts, Tailwind CSS 4 and the `yaml` package, built with
Vite 7 and TypeScript 5.9 (versions in `package.json`). CI (`.github/workflows/ci.yml`, job `web`)
uses Node 24 and pnpm 10. `pnpm dev:mock` also needs [Bun](https://bun.sh) on `PATH`.

## Commands (run in `web/`)

| Command | What it does |
| --- | --- |
| `pnpm install --frozen-lockfile` | Installs exactly what `pnpm-lock.yaml` pins. |
| `pnpm dev` | Vite dev server on `:5173`, proxying `/api` to `MONORL_API`. |
| `pnpm dev:mock` | `bun run mock/server.ts`: a mock of the API on `PORT` (default 8000). |
| `pnpm build` | `tsc -b && vite build`: type-checks, then writes `dist/`. |
| `pnpm lint` | `eslint . && tsc -b --noEmit`. |

`MONORL_API` defaults to `http://127.0.0.1:8000`, so `pnpm dev` works against either
`monorl serve --no-open` or `pnpm dev:mock` unchanged. `make web` from the repository root runs
the install and the build.

## Build output

`pnpm build` empties and rewrites `web/dist/` (gitignored): `index.html` plus hashed files under
`assets/`. `monorl serve` serves it from the same FastAPI app as the API (`_mount_static` in
`src/monorl/server/app.py`; `MONORL_WEB_DIST` selects another directory): a path that is a file
under `dist/` returns the file, any other path outside `/api` returns `index.html`. Without
`dist/index.html` the server registers no UI route and serves the API only.

## Source layout

| Path | Contents |
| --- | --- |
| `src/App.tsx` | Routes (`/`, `/new`, `/runs/:id`, `/system`); loads the config schema once. |
| `src/pages/` | One file per route: `Runs`, `NewRun` (wizard shell), `RunDetail`, `System`. |
| `src/wizard/` | One component per wizard step, `Recipes`, `HarnessForm`; `common.ts` has the step list. |
| `src/components/` | `ui.tsx` (form fields, cards), `MetricChart`, `RunStreams`, `TranscriptView`. |
| `src/types.ts` | Hand-maintained TypeScript mirror of `RunConfig` and the server API shapes. |
| `src/defaults.ts` | `defaultRunConfig()`: the pydantic defaults, restated; merge helpers. |
| `src/pipeline.ts` | Stage defaults and structural checks mirroring `config/pipeline.py`. |
| `src/schema.ts`, `src/api.ts` | Schema lookup by dotted config path; typed fetch and SSE client. |
| `mock/server.ts` | Bun mock of the server API: seeded runs, a fake job, its own schema literal. |

## Adding a config field

Only field descriptions are generated: `Field` and `Toggle` in `src/components/ui.tsx` take a
`path` and read `description` from `GET /api/config/schema` (`RunConfig.json_schema()`), or from
`GET /api/config/pipeline_schema` for stages. Structure, defaults and the forms are written by
hand, so a field added to `src/monorl/config/schema.py` also needs:

its type in `src/types.ts`; its default in `src/defaults.ts` (`defaultRunConfig()`), or
`src/pipeline.ts` for a stage; an input in the wizard step that owns the section
(`src/wizard/*Step.tsx`), with `path` set so the description comes from the schema; and an entry in
the schema literal in `mock/server.ts`, so `pnpm dev:mock` shows the description.

Then run `pnpm build` and `pnpm lint`. `tsconfig.*.tsbuildinfo` are tracked; a build with no
source changes left them byte-identical here.
