# Sandboxed harness episodes and task directories

`env.type: harness` ([`GATEWAY.md`](GATEWAY.md)) runs an external agent program per episode. Two
things decide where and on what:

- **`env.params.sandbox`** — where the agent command and the verifier run: as processes on the
  trainer host (default, unchanged) or in a **container started per episode**.
- **Task directories** — a dataset that is a directory of tasks in the
  [Harbor task format](https://docs.harborframework.com/tasks/overview) (the format of
  Terminal-Bench 2 and the benchmarks published for Harbor). Every task brings its instruction,
  its image, its verifier and its limits, and the harness uses them for that sample.

```
 trainer host                                         container (one per episode)
┌──────────────────────────────────────────┐        ┌──────────────────────────────────────┐
│ HarnessEnv ── docker run -d ─────────────┼───────▶│ sh -c "sleep infinity"   (under --init)│
│     │        docker exec ─ agent ────────┼───────▶│ agent ── tools, files                 │
│     │                                    │◀───────┼── OPENAI_BASE_URL=http://host.docker. │
│ capture gateway ◀── policy calls ────────┤        │        internal:<port>/s/<session>/v1 │
│     │        docker exec ─ verifier ─────┼───────▶│ verifier (same container)             │
│     ▼        docker rm -f                │        │ /monorl = episode directory (mount)   │
│ Trajectory (+ info.sandbox)              │        └──────────────────────────────────────┘
└──────────────────────────────────────────┘
```

Code: `src/monorl/envs/sandbox.py` (backends), `src/monorl/envs/harness.py` (episode),
`src/monorl/data/tasks.py` (task directories). Recipe:
`configs/recipes/grpo_harness_tasks_docker_cpu.yaml`; example tasks and agent: `examples/sandbox/`.

## 1. `env.params.sandbox`

```yaml
env:
  type: harness
  params:
    command: ["{python}", "/opt/agent/agent.py", "--task", "{task_file}"]
    verifier: ["{python}", "/opt/agent/check.py", "{reference_file}", "{stdout_file}"]
    sandbox:
      backend: docker
      image: python:3.12-slim-bookworm
      mounts: ["./my_agent:/opt/agent:ro"]
      cpus: 2
      memory: 2g
```

| Option | |
|---|---|
| `backend` | `local` (default): processes on the trainer host, exactly as without a sandbox. `docker`: a container per episode, through the Docker CLI. Every other option needs `docker`. (Same key as the Hermes environment's `sandbox.backend`.) |
| `image` | Image to start. Pulled once when it is not local. |
| `build` | Instead of `image`: a build context directory, or `{context, dockerfile, args, target}`. Built once and cached: the tag is `monorl-sandbox:<hash of the context's files, Dockerfile path, build arguments, target, platform>`, so an unchanged context is never built again, also by later runs; a changed one gets a new tag. |
| `build_timeout` | Seconds a pull or build may take (default 600). |
| `workdir` | Working directory in the container. Unset: `/monorl/work`, the episode's working directory on the host (a copy of `env.params.workdir.template`), bind-mounted. An absolute path: that directory of the container's own filesystem (created if missing; the template is copied into it). `image`: the image's `WORKDIR`. |
| `mounts` | Bind mounts, `host:container[:ro]` (Docker's `-v` syntax). Relative host paths are resolved against the directory monorl runs in; a host path that does not exist is an error. |
| `network` | Docker network of the container (default: Docker's default bridge). `host` shares the host's network. `none` is rejected: the agent could not reach the policy. |
| `cpus`, `memory`, `pids_limit`, `gpus` | Resource limits, passed to `--cpus`, `--memory` (Docker's format: `512m`, `2g`), `--pids-limit`, `--gpus`. |
| `user` | User the agent and the verifier run as (`docker exec -u`); default: the image's. |
| `platform` | `--platform` for pull, build and run (e.g. `linux/amd64` images on Apple Silicon). |
| `python` | What `{python}` means in the container (default `python3`). On the host it is the interpreter running monorl, which does not exist in an image. |
| `keep_entrypoint` | `false` (default): the image's `ENTRYPOINT` and `HEALTHCHECK` are switched off and the container just idles. `true`: the image's entrypoint runs with `sh -c "sleep infinity"` as its command, as Harbor starts task images. |
| `keep_processes` | `false` (default): when the agent has ended, the container is restarted before the verifier runs, so nothing the agent started is alive while the verifier's inputs are around; the filesystem is kept. `true`: processes the agent left running keep running during verification (for verifiers that probe a service the agent was asked to start). After a timeout the container is restarted either way. |
| `args` | Extra `docker run` arguments, placed last so they win (`["--cap-drop", "ALL"]`, `["--security-opt", "no-new-privileges"]`, `["--tmpfs", "/scratch"]`). |
| `start_timeout` | Seconds `docker run` may take (default 120). |

Requirements on the image: `sh` and `sleep` (the idle process), whatever interpreter the agent
needs, and for resource reporting `cat`. The agent itself comes from the image or from `mounts`.

### What happens in an episode

1. The image is pulled or built if needed (once per image and process; parallel episodes wait).
2. `docker run -d --init` starts the container with the episode directory mounted at `/monorl`:

   | In the container | |
   |---|---|
   | `/monorl/task.json` | the task file (`{task_file}`, `MONORL_TASK_FILE`) |
   | `/monorl/work` | the working directory unless `sandbox.workdir` says otherwise (`{workdir}`, `MONORL_WORKDIR`) |
   | `/monorl/agent.stdout`, `/monorl/agent.stderr` | the agent's captured output (`{stdout_file}`, `{stderr_file}` for the verifier) |
   | `/monorl/reference.json` | the sample with its reference (`{reference_file}`), written after the agent has ended |

   Every placeholder and variable that names a path names the container's path.
3. The agent is a `docker exec` in the working directory. It gets the variables of
   [`GATEWAY.md`](GATEWAY.md) §3 (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_BASE_URL` /
   `ANTHROPIC_API_KEY` when the gateway serves the Anthropic API, `MONORL_*`), `env.params.env`,
   and `NO_PROXY` for the gateway host. **Nothing of monorl's own environment enters the
   container** (locally the agent inherits it). stdout and stderr are captured on the host.
4. When the agent exits or its `timeout` passes, the container's CPU time and peak memory are
   read, and the container is restarted (see `keep_processes`).
5. The verifier is a `docker exec` in the same container and working directory: it sees
   everything the agent left on the filesystem, not only the working directory.
6. `docker rm -f -v` removes the container; the episode directory is deleted.

### `Trajectory.info`

`info.sandbox` for container episodes:

```
{"backend": "docker", "image": "...", "container": "monorl-harness-<id>",
 "cpus": 2.0, "memory": "2g", "pids_limit": 256,        # the configured limits
 "cpu_seconds": 3.1, "memory_peak_bytes": 48234496,     # what agent phase used (cgroup v2)
 "oom_killed": false}                                   # a process hit the memory limit
```

The usage fields are left out when the container cannot report them (no `cat`, cgroup v1 host,
kernel before 5.19 for the peak). An agent killed by the memory limit has `exit_code: 137` and
`oom_killed: true`. The rest of `info` is as for local episodes (`exit_code`, `timed_out`,
`stdout_tail`, `stderr_tail`, `verifier`, `posted_reward`); `exit_code` is `docker exec`'s, so
`126`/`127` mean the command could not be started in the container.

### Failures and cleanup

| | |
|---|---|
| no `docker` executable on `PATH` | `ConfigError` when the environment is built |
| the daemon does not answer (`docker info` fails, or hangs for 20 s) | `EnvironmentError_` naming the CLI's message |
| pull / build fails or exceeds `build_timeout`, `docker run` fails | `EnvironmentError_` with the end of the CLI output |
| the agent made no policy call | `EnvironmentError_` with exit code and stderr tail (as locally) |
| timeout | the agent is stopped, the episode counts if a call was answered (`info.timed_out`) |
| abort (Ctrl-C, cancelled job) | `EpisodeAborted`; a pull or build in flight is stopped as well |

`EnvironmentError_` episodes are subject to `env.retries` / `env.max_failed_frac`.

A container never outlives its episode: `close()` removes it on every path above. If monorl
itself dies without a chance to clean up (`SIGKILL`, a crash, the terminal closing), a small
watchdog process that monorl starts with its first container notices that its parent is gone and
removes every container labelled `monorl.harness.owner=<that process>`. To check by hand:
`docker ps -a --filter label=monorl.harness.owner`.

## 2. Reaching the gateway from a container

The capture gateway runs in the trainer process. The session id in the URL is unguessable and is
the only credential; it is handed to the container through the CLI's environment, not its
command line.

| Platform | Gateway binds | Agent is told | Why |
|---|---|---|---|
| local processes | `127.0.0.1` | `127.0.0.1` | |
| Docker Desktop (macOS, Windows, Linux), and any daemon used from macOS or Windows (colima, OrbStack, Rancher Desktop) | `127.0.0.1` | `host.docker.internal` | the daemon's VM forwards that name to the host's loopback interface, so the gateway does not have to be exposed |
| Docker Engine on Linux | `0.0.0.0` | `host.docker.internal`, with `--add-host host.docker.internal:host-gateway` | containers reach the host through the bridge gateway (`172.17.0.1`), not through loopback; the gateway has to listen there |
| Docker Engine on Linux, `network: host` | `127.0.0.1` | `127.0.0.1` | the container shares the host's loopback |

The platform is detected from `docker info` (`OperatingSystem`) and the host OS. Override any of
it with `env.params.gateway`:

```yaml
env:
  params:
    gateway:
      bind: 172.17.0.1      # Linux: listen on the Docker bridge only instead of every interface
      port: 8765            # default: any free port
      host: 172.17.0.1      # the name or address put into the URLs the agent gets
```

**Image URLs.** The gateway can download images an agent sends as `http(s)` URLs
([`GATEWAY.md`](GATEWAY.md)), i.e. the trainer host issues requests to addresses the agent
chooses. `env.params.fetch_images` controls it: default `true` for the local backend, `false`
with `sandbox.backend: docker` (such images are refused; `data:` URLs still work), so a sandboxed
agent cannot reach through the trainer host into networks its container cannot see. An explicit
`fetch_images: true` allows it again.

Notes for Docker Engine on Linux:

- Binding `0.0.0.0` makes the gateway reachable from the network the host is on. Sessions are
  protected by their id only; bind the bridge address as above, or firewall the port, on hosts
  that are not trusted networks.
- A host firewall that drops traffic from the Docker bridges (ufw's default `deny incoming`)
  also drops the agent's calls: allow the bridge interface (`ufw allow in on docker0`) or the
  gateway port.
- Rootless Docker and user-defined networks resolve `host-gateway` differently; if the agent
  cannot connect, set `gateway.host` (and `gateway.bind`) to an address of the host that the
  container can route to.
- A remote daemon (`DOCKER_HOST=ssh://...`) is not supported: the episode directory is
  bind-mounted, so the daemon has to see the trainer's filesystem.

## 3. Task directories

`data.uri` may name a task directory, a directory whose children are task directories, or a glob
of either:

```yaml
data:
  uri: examples/sandbox/tasks        # or ~/terminal-bench-2/tasks, or "tasks/sql-*"
env:
  type: harness
  params:
    command: ["{python}", "/opt/monorl-agents/shell_agent.py"]
    sandbox: {backend: docker, mounts: ["examples/sandbox/agents:/opt/monorl-agents:ro"]}
rewards: []                          # the reward is the task verifier's
```

The format is Harbor's ([overview](https://docs.harborframework.com/tasks/overview),
[`task.toml`](https://docs.harborframework.com/tasks/configuration),
[environment](https://docs.harborframework.com/tasks/environment),
[verifier](https://docs.harborframework.com/tasks/verifier),
[solution](https://docs.harborframework.com/tasks/solution)), implemented against the
`harbor-framework/harbor` sources at commit `feea734` (`schema_version` 1.4;
`src/harbor/models/task/{config,paths}.py`, `src/harbor/verifier/verifier.py`,
`src/harbor/environments/docker/`):

```
my-task/
├── instruction.md          the task, given to the agent
├── task.toml               limits, environment and verifier configuration
├── environment/Dockerfile  the task's image (or [environment].docker_image)
├── tests/test.sh           verifier: writes /logs/verifier/reward.txt or reward.json
└── solution/solve.sh       reference solution (never read by monorl)
```

Each task is one `Sample`: `id` = the directory name, `messages` = `[user: instruction]`
(`data.system_prompt`, `prompt_template`, `max_prompt_tokens`, `shuffle`, `limit` apply as for
any dataset), `reference` = `None`, and `metadata["row"]["task"]` = the task's settings.
`monorl data inspect <dir>` and the UI's dataset preview show the rows.

### Mapping

| Task | monorl |
|---|---|
| `instruction.md` | the user message; leading canary comment lines are removed as Harbor does, surrounding whitespace is trimmed |
| `[environment].docker_image` | `sandbox.image` (preferred over `environment/`, as in Harbor) |
| `environment/Dockerfile` | `sandbox.build` with the `environment/` directory as context: built once, cached under its content hash |
| `[environment].build_timeout_sec` (600) | `sandbox.build_timeout` |
| `[environment].workdir` | the working directory; unset: the image's `WORKDIR` |
| `[environment].cpus`, `memory_mb` (or the deprecated `memory = "2G"`), `gpus` | `sandbox.cpus`, `memory`, `gpus` |
| `[environment].env` | the container's environment; a value that is exactly `${VAR}` or `${VAR:-default}` is read from monorl's environment (unset without default: `ConfigError`) |
| `[environment.healthcheck]` | run before the agent with Docker `HEALTHCHECK` semantics (`command` via `bash -c`, `interval_sec`, `timeout_sec`, `start_period_sec`, `start_interval_sec`, `retries`); a failing check fails the episode |
| image `ENTRYPOINT` | kept (`keep_entrypoint`), with `sh -c "sleep infinity"` as command, as Harbor's Docker environment does |
| `[agent].timeout_sec` | the agent's timeout for this sample; unset: `env.params.timeout` |
| `[agent].user`, `[verifier].user` | the user of the agent / of `test.sh` |
| `tests/` | copied to `/tests` after the agent has ended; `/tests/test.sh` is made executable and run in the working directory |
| `[verifier].timeout_sec` (600), `[verifier].env` | the verifier's timeout and environment (`${VAR}` as above) |
| `/logs/verifier/reward.json`, `reward.txt` | the episode's reward (below) |
| `/logs/agent`, `/logs/verifier`, `/logs/artifacts` | exist and are writable; discarded with the episode |
| `[task].name` | `Trajectory.info["task"]` (default: the directory name) |
| `[metadata]` | kept in the task mapping; not shown to the agent |
| `solution/`, `[solution]` | ignored |

The other `env.params.sandbox` options (`mounts`, `network`, `pids_limit`, `platform`, `python`,
`args`, `keep_processes`) apply to every task; `env.params.command` is the agent for all of them.
`env.params.verifier` is not used for task samples. Tasks need `sandbox.backend: docker`.

**Reward.** As in Harbor, `reward.json` wins over `reward.txt`. `reward.txt` holds one number.
`reward.json` holds an object of labelled numbers (or one number): the reward is its `reward`
entry if it has one, otherwise the mean of its values (Harbor defines no scalar for several
metrics; this is monorl's choice), and all of them are in `info.verifier.rewards`. No reward
file, an unreadable one, or a verifier timeout scores `0.0` (`info.verifier.error` /
`timed_out`). The reward is added to the last turn like any verifier's.

**What the agent can see.** The task file holds the instruction only — not the task's
directory, its `[metadata]` or its settings. `tests/` is copied into the container after the
agent has ended and (by default) after the restart that kills what it left running; before the
copy, `/tests` and the reward files are deleted, so a test directory or a reward the agent
planted has no effect. `solution/` is never copied anywhere.

### Not supported

A task that needs one of these is reported by name when the dataset is opened (`DataError`);
`data.extra.skip_unsupported: true` leaves such tasks out instead (logged).

| | Why |
|---|---|
| multi-step tasks (`[[steps]]`, `steps/`) | one instruction, one verification per episode |
| `environment/docker-compose.yaml` (multi-container), other environment specs | one container per episode |
| separate verifier environments (`[verifier].environment_mode = "separate"`, `[verifier.environment]`) | the verifier runs in the agent's container (Harbor's shared mode) |
| `network_mode` other than `public` in `[environment]`, `[agent]` or `[verifier]` (and `allow_internet = false`) | the agent has to reach the capture gateway on the host, and monorl has no egress filter |
| `[environment].os = "windows"` | Linux containers only |
| `trajectory.json` (prior agent context) | not replayed |
| `schema_version` other than 1.x; Terminal-Bench 1 `task.yaml` tasks | convert with Harbor's migration tooling |

Read but not acted on: `[environment].storage_mb` (Docker enforces disk quotas only on some
storage drivers), `mcp_servers` and `skills_dir` (agent capabilities; configure your agent),
`artifacts` (nothing is collected to the host), `.harborignore` (the whole `tests/` directory is
copied).

### The agent

`env.params.command` runs inside every task's image, so the agent has to be runnable there:

- mount it (`sandbox.mounts`) — a script for an interpreter the images have, or a static
  binary — or bake it into the images;
- images without the interpreter can install it first, at the cost of the agent's time budget:
  `command: ["sh", "-c", "apt-get install -y python3 >/dev/null && exec python3 /opt/agent/agent.py"]`.

`examples/sandbox/agents/shell_agent.py` is a standard-library example (one tool, a shell; the
task from `MONORL_TASK_FILE`, the policy through `OPENAI_BASE_URL`). `examples/sandbox/tasks/`
has three small tasks: a prebuilt image with `reward.txt`, a Dockerfile with a file baked in, and
`reward.json` with partial credit. Train on them (needs a Docker daemon; run from the repository
root, the paths are relative):

```
monorl train --config configs/recipes/grpo_harness_tasks_docker_cpu.yaml
```

A dataset row of any other source can bring the same settings: a `task` column holding
`{"sandbox": {...}, "timeout": ..., "verifier": {...}}` (shape: `monorl.envs.harness`) is used
for that sample in the same way.

## 4. Security notes

- **A container is not a strong boundary.** The agent runs model-written commands as the image's
  user, by default root inside the container, with the default capabilities and with network
  access (it needs the gateway). Tighten with `user`, `pids_limit`, `memory`, `cpus` and `args`
  (`--cap-drop ALL`, `--security-opt no-new-privileges`, `--read-only` plus `--tmpfs`), or run
  the daemon rootless / with a sandboxed runtime (`args: ["--runtime", "runsc"]`).
- **The episode directory is writable from the container** (`/monorl`, for tasks also `/logs`):
  the agent can fill the host's disk through it. `mounts` are writable unless marked `:ro`. Never
  mount the Docker socket or directories with credentials.
- **Network.** The agent can reach the internet and, on Docker Engine, the host's services on
  the bridge address. There is no egress filtering; tasks that ask for it are refused.
- **The gateway.** On Docker Engine it listens on every interface (§2). The session id is the
  credential; it is closed when the agent ends, so a process that survives cannot call the
  policy or post rewards.
- **Verifier integrity.** The verifier runs in the container the agent had: files outside
  `/tests` and the reward files are the agent's to change (a replaced `python3`, say). The
  default restart removes its processes, not what it wrote. With `keep_processes: true` a
  process of the agent is alive while tests and reference are in the container. Reward functions
  on the trajectory, or a verifier that only reads the agent's output, are harder to game.
- **Requests from the trainer host.** Image URLs in a sandboxed agent's requests are not
  fetched unless `env.params.fetch_images: true` (§2).
- **Secrets.** monorl's environment is not passed into containers; what `env.params.env` and a
  task's `${VAR}` entries name is.
- With `user` set, the episode directory is made world-readable and its working directory
  world-writable on the host so that the container user can use them.

## 5. Limits

- Docker CLI only (`docker` on `PATH`); the daemon has to share the trainer's filesystem. No
  Podman, Kubernetes, Modal or Daytona backend.
- One container per episode, started for it: no pooling or snapshots. Expect a few hundred
  milliseconds of container overhead per episode plus the image's own start-up.
- Resource usage is what the container's cgroup reports for the agent phase; nothing is
  reported on cgroup v1 hosts.
- A timed-out agent is stopped by restarting the container: tmpfs contents and running services
  are gone when the verifier runs.
- The context of a `build` is hashed file by file on first use in a process; keep it small.
- Images are never removed by monorl: `docker image ls monorl-sandbox` lists the built ones.
- Backends that step environments themselves (verl, NeMo-RL) cannot use `harness`, hence neither
  sandboxes nor task directories.

## 6. Tests

`tests/envs/test_harness_sandbox.py` drives the whole lifecycle through a stand-in `docker`
executable (`tests/envs/fake_docker_cli.py`) and needs no daemon. `tests/envs/test_sandbox_docker.py`
repeats the essentials against a real daemon and is skipped when none answers:

```
.venv/bin/python -m pytest tests/envs/test_sandbox_docker.py -q
```

It uses the local image `monorl:cpu` (`MONORL_SANDBOX_TEST_IMAGE` to change; needs `python3` and
`bash`), and `python:3.12-slim-bookworm` for the example tasks (that test is skipped when the
image is not local). It removes the containers it starts and the images it builds.
