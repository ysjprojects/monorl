# Contributing to monorl

How to set up a checkout, which checks must pass, and the conventions the code base holds itself
to. The code map is [`src/monorl/README.md`](src/monorl/README.md); the architecture and the
contracts between components are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Setup

```bash
make install                # .venv (uv, Python 3.12) with the extras CI installs
scripts/dev/bootstrap.sh    # the same, plus the example environments and the web build
```

Optional toolchains: Go for the Go SDK and its examples, clang++ for the C++ SDK, Node 24 and
pnpm for the web UI. Run Python through `.venv/bin/python`, or activate the venv.

Development happens on macOS (arm64, no CUDA) and CI runs on Linux x86, CPU only. vLLM, SGLang,
verl, NeMo-RL and Unsloth are not installed in either place, which shapes the testing rules
below.

## Checks

| What | Command |
|---|---|
| Python lint | `make lint` (ruff check and format check over `src tests scripts examples`, 110 columns) |
| Python tests | `make test-fast`; one area: `.venv/bin/python -m pytest tests/rewards -q` |
| Everything, including slow tests | `make test` (downloads a 135M model, trains a few steps on CPU) |
| Go SDK | `cd envs/sdk/go && go vet ./... && go test ./...` |
| Example environments | `make envs` (Go and C++, into `envs/bin`) |
| MEP/1 conformance of the examples | `.venv/bin/python -m pytest tests/envs_sdk -q` after `make envs` |
| Web UI | `cd web && pnpm install --frozen-lockfile && pnpm build && pnpm lint` |
| Config reference is current | `.venv/bin/python -m monorl.config.reference --check docs/CONFIG.md` |

CI (`.github/workflows/ci.yml`) runs on pushes to `main` and on pull requests: Python lint and
`pytest tests -m 'not slow and not network and not gpu'`, the Go and C++ SDK builds and tests,
the web build and lint, the MEP/1 conformance tests, hadolint over `docker/`, and builds of the
`envs` and `cpu` images. A newer push cancels the run of an older one.

## Tests

- **Layout.** `tests/<area>/` mirrors `src/monorl/<area>/`. Shared fixtures are in
  [`tests/conftest.py`](tests/conftest.py): `tiny_model_dir` (a random 2-layer Llama with the
  SmolLM tokenizer, built once per session), `smol_model_id` (a real 135M model, for `slow`
  tests) and `run_config_dict` (a minimal valid config). Area fixtures go in
  `tests/<area>/conftest.py`.
- **Markers.** `slow` downloads or trains a real model, `network` needs the internet, `gpu`
  needs CUDA. CI deselects all three; unmarked tests must be deterministic and offline (the
  tokenizer of `tiny_model_dir` comes from the Hugging Face cache, which CI fills in a step of
  its own).
- **Frameworks that are not installed.** An adapter is tested against a fake of its framework
  that reproduces the upstream behaviour the adapter relies on (`tests/adapters/vllm/vllm_fakes.py`,
  `tests/adapters/sglang/fake_sglang.py`, `tests/adapters/nemo_rl/nemo_rl_fakes.py`), plus a
  test against the real framework that is guarded with `pytest.importorskip` and marked `gpu`
  where it needs CUDA. Skip when something is absent; never stub the code under test.
- **What a test is for.** A test should fail for a bug a user of the component would see. No
  tests of wiring, of source text, or of defaults that nothing depends on.
- **Floats.** Compare float32 results that were computed over different batch shapes with a
  relative tolerance. The Linux runner and a Mac differ in the last digits (about 1e-6
  relative), and an absolute bound that holds on one fails on the other.
- **Scratch state.** Under pytest `MONORL_HOME` defaults to `/tmp/monorl-tests`; tests write
  everything else below `tmp_path`.

## Conventions

- **No stubs.** No placeholders, no `NotImplementedError` "for later", no fake fallbacks. A path
  that needs a GPU or a framework you do not have is written against the pinned upstream source
  (`scripts/vendor.py --fetch <name>` clones it into `third_party/`) and covered as described
  above.
- **Patches, not forks.** A change to upstream behaviour is a `Patch` with a target, a version
  gate and a verification ([`docs/PATCHING.md`](docs/PATCHING.md)).
- **The config is the product.** A new option is a field of
  [`src/monorl/config/schema.py`](src/monorl/config/schema.py) with a description:
  [`docs/CONFIG.md`](docs/CONFIG.md) is generated from it (regenerate with
  `.venv/bin/python -m monorl.config.reference --write docs/CONFIG.md`) and the web UI shows it
  next to the field. The wizard's form for the field is written by hand
  ([`web/README.md`](web/README.md)). Rules across fields are validators with a message that
  says what to change. A backend that cannot honour an option rejects it at validation instead
  of ignoring it.
- **Seams.** Components talk through `monorl.core.interfaces` and are found through the
  registry; the orchestrator imports no framework. Some packages must import without torch;
  [`src/monorl/README.md`](src/monorl/README.md) lists them.
- **Shared contracts.** `core/` and `config/` are used by everything. Change them with the
  smallest additive edit that does the job. When several people or agents work in parallel, each
  owns the directories of its task and nothing else.
- **Dependencies.** A new package goes into the matching extra in `pyproject.toml`.
- **transformers 5.** `tokenizer.apply_chat_template(..., return_tensors="pt")` returns a
  `BatchEncoding` (pass `return_dict=True` and unpack it, or render with `tokenize=False`), and
  `from_pretrained` takes `dtype=`, not `torch_dtype=`.

## Documentation

- A section of the config has a document under [`docs/`](docs/README.md); a directory that
  people build, run or extend has a `README.md`. Link to the document that owns a topic instead
  of restating it.
- Every command, snippet and config block in a document is run or validated as printed before
  it is committed. State what was not run.
- `tests/docs` checks that relative links resolve, that `docs/README.md` lists every document
  and that the README lists every recipe.

## Adding things

| To add | Start at |
|---|---|
| A reward, environment, data source, engine, trainer or backend | the extension points in [`src/monorl/README.md`](src/monorl/README.md) |
| An adapter for a framework, or a newer pin of one | [`src/monorl/adapters/README.md`](src/monorl/adapters/README.md) |
| A recipe | [`configs/recipes/README.md`](configs/recipes/README.md) |
| An environment in Go or C++ | [`docs/ENV_SDK.md`](docs/ENV_SDK.md) |
| A task directory for sandboxed runs | [`examples/sandbox/README.md`](examples/sandbox/README.md) |

## License

monorl is licensed under [Apache-2.0](LICENSE); section 5 of the license covers contributions.
