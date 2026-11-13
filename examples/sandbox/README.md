# Sandbox examples: task directories and a shell agent

Three small tasks in the Harbor task format and one agent that can attempt them, used by the
recipe `configs/recipes/grpo_harness_tasks_docker_cpu.yaml`. Read this to see what a task
directory looks like on disk or to add one. The format, the mapping of `task.toml` to sandbox
options, the reward rules and the limits are in [docs/SANDBOX.md](../../docs/SANDBOX.md).

## Contents

| Path | What it is |
|---|---|
| `tasks/hello-file` | Create a file with one fixed line. Prebuilt image (`docker_image`), reward 0 or 1 in `reward.txt` |
| `tasks/count-lines` | Count the lines of `words.txt`, which `environment/Dockerfile` copies into the image. `reward.txt` |
| `tasks/sort-numbers` | Sort `numbers.txt`. `tests/check.py` writes `reward.json`: 1.0 sorted, 0.5 right numbers in another order, else 0.0 |
| `agents/shell_agent.py` | Tool loop with one tool, a shell. Standard library only, Python 3.8+ |

All three images are `python:3.12-slim-bookworm` or built from it, with `/app` as the working
directory.

## One task directory

`tasks/sort-numbers` as it is on disk:

```text
sort-numbers/
├── instruction.md             the task text, the only thing the agent is given
├── task.toml                  [task] name, [metadata], timeouts, cpus, memory_mb
├── environment/
│   ├── Dockerfile             the task image (python:3.12-slim-bookworm, WORKDIR /app)
│   └── numbers.txt            copied into the image by the Dockerfile
├── tests/
│   ├── test.sh                the verifier, runs after the agent has ended
│   └── check.py               called by test.sh, writes /logs/verifier/reward.json
└── solution/
    └── solve.sh               reference solution, never read by monorl
```

`hello-file` has no `environment/` directory: its `task.toml` names the image and the working
directory (`[environment] docker_image`, `workdir`). The three `task.toml` files set
`[agent] timeout_sec = 300.0`, `[verifier] timeout_sec = 60.0`, `cpus = 1`, `memory_mb = 512`.

## The agent

`agents/shell_agent.py` reads the task from `MONORL_TASK_FILE`, calls the policy at
`OPENAI_BASE_URL` (`/chat/completions` with one function tool, `shell`), runs each command with
`sh -c` and returns its exit status and output. It stops when the model answers without a tool
call or after `--max-steps` policy calls (default 4). The system prompt and two worked examples
show the call syntax; `--no-examples` leaves the examples out.
`python3 examples/sandbox/agents/shell_agent.py --help` lists the options.

## Running

From the repository root (the dataset and the agent mount are relative paths):

```sh
monorl data inspect examples/sandbox/tasks
monorl train --config configs/recipes/grpo_harness_tasks_docker_cpu.yaml
```

`data inspect` shows the three tasks as dataset rows and needs no Docker. The `train` command
needs a running Docker daemon; it has been run with Docker Desktop on macOS, not with Docker
Engine on Linux. The recipe starts one container per episode, mounts `examples/sandbox/agents`
read-only at `/opt/monorl-agents` and takes the reward from each task's verifier
(`rewards: []`).

## Adding a task

1. Create `tasks/<name>/` with `instruction.md` and `task.toml` (copy one from here; set
   `[task] name`).
2. Give it an image: `[environment] docker_image` and `workdir` in `task.toml`, or an
   `environment/Dockerfile` with the files it needs. The image needs `python3` for this agent.
3. Write `tests/test.sh`. It must write one number to `/logs/verifier/reward.txt`
   or a JSON object with a `reward` entry to `/logs/verifier/reward.json`.
4. Optionally add `solution/solve.sh` for readers.
5. Run `monorl data inspect examples/sandbox/tasks`: the new task must appear as a row.

The recipe picks up every task directory under `data.uri`; no list has to be edited.
