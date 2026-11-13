# Recipes

A recipe is a complete, validated config file that `monorl train` can run as it is. This page
states the conventions the code relies on and what to do when adding one. The list of recipes and
what each shows is in the [root README](../../README.md#recipes), or run `monorl recipes list`.

## Conventions

- One file per recipe, `<name>.yaml`. The name is the file stem: `monorl recipes show <name>`.
  Use the extension `.yaml`: the CLI and the test below look at `*.yaml` only.
- The file starts with a block of `# ` comment lines. They are the description: joined with
  spaces, up to the first line that is not a comment (`read_description` in
  `src/monorl/cli/recipes.py`). `monorl recipes list` and the web UI (`GET /api/recipes`) show
  it. Comments further down are not part of it.
- Two kinds, told apart by the top-level keys: a file with both `base` and `stages` is a
  pipeline config (`PipelineConfig`, see [docs/PIPELINES.md](../../docs/PIPELINES.md)), anything
  else is a run config (`RunConfig`, see [docs/CONFIG.md](../../docs/CONFIG.md)).
  `monorl recipes list` prints the kind. `monorl train --config` accepts both and hands a
  pipeline to the pipeline runner.
- Relative paths in a recipe (`data.uri: examples/sandbox/tasks`,
  `command: ["envs/bin/go-wordle"]`) assume the repository root as working directory. Run
  recipes from there.
- Every recipe is validated by `test_every_recipe_validates` in `tests/cli/test_cli.py`: a run
  config must load with `RunConfig.from_dict`, a pipeline config with `PipelineConfig.from_dict`.
  `test_recipes_dir_and_descriptions` in the same file requires the description and that the
  file begins with `# `, and names a set of recipes that must exist.

## Using a recipe

```sh
monorl recipes list
monorl recipes show grpo_gsm8k_smollm_cpu
monorl train --config configs/recipes/grpo_gsm8k_smollm_cpu.yaml \
  --set trainer.max_steps=1 --set data.limit=4 --run-dir /tmp/monorl-recipe/run
```

`recipes show` validates the file and prints it (`--json` prints the validated config as
JSON). `--set key.path=value` overrides one key without editing the file; the
value is parsed as YAML. For a pipeline config the keys of the shared run config start with
`base.` (the example in `monorl pipeline --help` is `base.trainer.device=cpu`).

`recipes list`, `recipes show` and the web UI read this directory. The environment variable
`MONORL_RECIPES_DIR` points them at another one:

```sh
MONORL_RECIPES_DIR=/path/to/my/recipes monorl recipes list
```

## Adding a recipe

1. Copy the closest existing recipe to `configs/recipes/<name>.yaml`.
2. Write the leading `# ` block: what the recipe shows, what it needs (GPUs, a server to start
   first, a Docker daemon), and that it must be run from the repository root if it has relative
   paths.
3. Set `run_name` (run config) or `name` (pipeline config): it names the default run directory.
4. Check it: `monorl recipes show <name>` must print the file, and
   `.venv/bin/python -m pytest tests/cli/test_cli.py -q -p no:cacheprovider -k recipe` must pass.
5. Run it, shortened with `--set` if needed, on hardware that can.
6. Add a row to the matching table in the [root README](../../README.md#recipes).
