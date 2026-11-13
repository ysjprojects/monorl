# Environments

This document covers the `env:` section of a run config: how an episode runs, the six registered
environment types, the built-in `single_turn` and `tool_calling` environments, the `openenv` and
`hermes` adapters, and how to write and serve an environment in Python. Read it when a task needs
more than one prompt and one completion, or when you add an environment. Example files live in
`$DEMO` (`/tmp/monorl-docs-environments` when written); commands run from the repository root.

## The episode model

A fresh environment is built per episode by the factory registered for `env.type`. The runner
(`src/monorl/orchestrator/episodes.py`) calls `reset(sample)` for the first messages, then
repeats: the rollout engine generates one completion (a turn), appended as an `assistant`
message, and `step(completion)` returns a `StepResult`: new messages (tool results, the next
user turn), a per-turn `reward`, `done` and `info`. `close()` always runs last.

An episode ends when `step` returns `done=True`, at the turn limit, when a completion finishes
with `length` or `error`, or when the next prompt leaves no room in the context budget
(`Trajectory.done` is then `False`; otherwise it is the last `done` the environment returned).
The turn limit is the environment object's `max_turns`. `single_turn`: always 1. `tool_calling`:
`env.max_turns` if set, else `env.params.max_turns`, else 4. `openenv` and `hermes`:
`env.max_turns`, so 1 unless you set it. `remote` and `harness`: `env.max_turns` if set (`remote`
otherwise takes the server's `max_turns`). Context budget: no prompt plus completion of a turn
may exceed `rollout.max_context_tokens` (default: the smaller of `trainer.max_seq_len` and
`rollout.max_model_len`); a first prompt that does not fit fails the episode.

Rewards: the reward of an episode is the score of the `rewards:` functions
([REWARDS.md](REWARDS.md)) plus `sum(gamma**k * r_k)` over the environment's turn rewards `r_k`,
`k = 0, 1, ...`, with `gamma = algorithm.gamma` (default 1.0). A non-zero sum is stored as `env`
in the trajectory's `reward_breakdown` and reported as `reward/env` ([RUNS.md](RUNS.md#metrics)).
`rewards:` may be empty for every type except `single_turn`. See also
[CREDIT_ASSIGNMENT.md](CREDIT_ASSIGNMENT.md) (per-turn credit) and [ALGORITHMS.md](ALGORITHMS.md).

Parallel episodes: episodes run on a thread pool, at most `rollout.extra.max_parallel_episodes`
(default 64) at a time, each with its own environment instance; generation requests are batched
across them. An exception from `reset`, `step` or prompt rendering is a failure of that
episode's environment; aborts and rollout engine failures always end the batch.

| Key | Default | Meaning |
|---|---|---|
| `env.type` | `single_turn` | registered environment type |
| `env.max_turns` | `1` | turn limit, as described above |
| `env.retries` | `0` | times a failed episode is restarted with a fresh environment |
| `env.max_failed_frac` | `0.0` | share of a rollout batch that may fail for good and be left out; above it, and always at `0`, a failed episode fails the run |
| `env.params` | `{}` | parameters of the chosen type |

With `env.max_failed_frac > 0` a run reports `rollout/failed_episodes`, `rollout/failed_frac` and
`rollout/groups_incomplete`. The schema rejects `env.max_failed_frac > 0` for `trainer.backend`
`trl`, `verl` and `nemo_rl`, `env.retries > 0` or a set `rollout.max_context_tokens` for `verl`
and `nemo_rl` ([BACKENDS.md](BACKENDS.md)), and any type but `single_turn` for
`algorithm.name: sft`.

## Registered types

| Type | What it is | Covered in |
|---|---|---|
| `single_turn` | the prompt is the whole episode; rewards come from `rewards:` | below |
| `tool_calling` | multi-turn tool use with built-in and custom Python tools | below |
| `harness` | an external agent program run as the episode | [GATEWAY.md](GATEWAY.md), [SANDBOX.md](SANDBOX.md) |
| `remote` | client for an environment server speaking MEP/1 | [ENV_SDK.md](ENV_SDK.md) |
| `openenv` | an OpenEnv (`openenv-core`) environment | below |
| `hermes` | Hermes Agent tools and prompt, stepped with the monorl policy | below |

`harness` starts one agent process per episode, which calls the policy through a recording
gateway; it is not stepped but runs the whole episode itself (`EpisodeDriver.run_episode`).
`remote` drives a server over stdio or http ([Python server](#serving-and-checking-it-over-mep1)).

## `single_turn`

`reset` returns the sample's messages and `step` ends the episode with reward 0, so at least one
entry in `rewards:` is required. One parameter: `env.params.system_prompt` (default
`data.system_prompt`), inserted first when the sample has no `system` message; others are ignored.

## `tool_calling`

The system prompt lists the tools; the policy answers with tool calls or a final reply. Each call
is executed and its result appended as a `tool` message (`name`, `tool_call_id`, `content`). The
tool schemas are also passed to the chat template as `tools`.

### Built-in tools and call syntaxes

- `calculator` (argument `expression`) evaluates arithmetic on the Python AST, without `eval`:
  numbers, `+ - * / // % **`, unary signs, `abs`, `round`, `min`, `max`, `sqrt`, `log`, `exp`,
  `sin`, `cos`, `tan`, `floor`, `ceil`, `pi` and `e`. An exponent above 1000 in absolute value, or
  anything else (names, attributes, strings), returns `error: ...`.
- `python_exec` (argument `code`) runs `sys.executable -I -c <code>`: the interpreter that runs
  monorl, in isolated mode (no `PYTHON*` variables, no user site), with an environment of only
  `PATH`, `PYTHONHASHSEED=0` and `PYTHONDONTWRITEBYTECODE=1`, stdin from `/dev/null` and the
  system temp directory as working directory. It is killed after `python_timeout_s` seconds of
  wall-clock time (`error: execution timed out after 5.0s`). The result is stdout, then `stderr:`
  and `exit code: N` when present, or `(no output)`. This is not a sandbox: the code runs as your
  user with file system, network and installed packages, and has no memory or CPU limit.
- `lookup` (argument `query`) returns the `env.params.corpus` value of an exact key match,
  otherwise up to `lookup_max_results` lines `key: value` whose key or value contains the query
  (case-insensitive), otherwise `no results for '<query>'`.

A missing required argument, or one not in the schema, returns `error: ...` to the policy and is
not counted as an invalid call. Two call syntaxes are accepted:

- Hermes style, anywhere in the text, one block per call:
  `<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>`.
- OpenAI style, when the text has no `<tool_call>` block: JSON that is the whole reply or sits in
  a code fence. Accepted: `{"tool_calls": [...]}`, one `{"name": ..., "arguments": ...}` or
  `{"type": "function", "function": {...}}` object, or a list of those. `arguments` may be an
  object or a JSON string. JSON inside prose without a fence is not a call.

### Parameters, penalties and the end of an episode

| `env.params` key | Default | Meaning |
|---|---|---|
| `tools` | the three built-ins | tool names to expose; an unknown name is an error |
| `tool_modules` | `[]` | modules imported before tools are resolved |
| `system_prompt` | Hermes-style prompt | template; `{tools}` becomes one JSON schema per line. A leading `system` message of the sample is appended to it |
| `invalid_call_penalty` | `0.0` | subtracted from the reward of a turn with an invalid call |
| `max_calls_per_turn` | `8` | calls executed per turn; more are ignored and make the turn invalid |
| `max_turns` | `4` | turn limit when `env.max_turns` is not set |
| `python_timeout_s` | `5.0` | `python_exec` timeout |
| `max_tool_output_chars` | `4000` | cap on a built-in tool's result (custom tools: always 4000) |
| `corpus` | `{}` | mapping searched by `lookup` |
| `lookup_max_results` | `3` | lines returned by a `lookup` substring search |

A turn is invalid when it has malformed `<tool_call>` JSON, an unterminated `<tool_call>`, a call
without a name or with non-object arguments, an unknown tool, or more than `max_calls_per_turn`
calls. Its reward is `-invalid_call_penalty`, once per turn however many errors it has; every
other turn has reward 0. Parse errors come back as `tool` messages named `error`. The episode is
done on a reply without tool call or parse error (the final answer), or at the turn limit. The
answer is not scored: add a reward function that reads the last assistant message, as
`configs/recipes/grpo_tool_calling_sglang.yaml` does. `StepResult.info` has `turn`,
`tool_calls`, `invalid_calls` and `final_answer`.

### Custom tools

Decorate a function with `monorl.envs.tool_calling.tool` and list its module in `tool_modules`
(it must be importable where the environment runs). The function gets the call's arguments as
keywords and returns a string or a JSON-serialisable value. `$DEMO/my_tools.py`:

```python
from monorl.envs.tool_calling import tool

PROPS = {"text": {"type": "string", "description": "Text to count."}}
SCHEMA = {"type": "object", "properties": PROPS, "required": ["text"]}


@tool(name="word_count", description="Count the words in a text.", parameters=SCHEMA)
def word_count(text: str) -> dict:
    return {"words": len(text.split())}
```

`$DEMO/drive_tools.py` steps the environment with scripted `Completion` objects, without a model:

```python
from monorl.core.types import Completion, Sample
from monorl.envs.tool_calling import make_tool_calling_env

params = {"tools": ["calculator", "word_count"], "tool_modules": ["my_tools"],
          "invalid_call_penalty": 0.1}
env = make_tool_calling_env(params, max_turns=4)
messages = env.reset(Sample(id="s0", messages=[{"role": "user", "content": "17 * 23?"}]))
print([m["role"] for m in messages])
for text in [
    '<tool_call>{"name": "calculator", "arguments": {"expression": "17*23"}}</tool_call>',
    '{"name": "word_count", "arguments": {"text": "seventeen times twenty-three"}}',
    '<tool_call>{"name": "search", "arguments": {}}</tool_call>',
    "17 * 23 = 391",
]:
    result = env.step(Completion(text=text, token_ids=[]))
    print(result.reward, result.done, [m["content"] for m in result.messages])
env.close()
```

```console
$ PYTHONPATH=$DEMO .venv/bin/python $DEMO/drive_tools.py
['system', 'user']
0.0 False ['391']
0.0 False ['{"words": 3}']
-0.1 False ["error: unknown tool 'search'; available: ['calculator', 'word_count']"]
0.0 True []
```

## `openenv` and `hermes`

Both need an optional dependency (extras `openenv` and `hermes` in `pyproject.toml`), else the
factory raises `BackendUnavailable`. Neither reads a turn limit from `env.params`.

### `openenv`

One environment instance is one OpenEnv session. The observation becomes a `user` message, the
completion becomes the action, reward and `done` come from the OpenEnv step (a missing reward is
0). The target is the first of `env_class`, `url`, `image` that is set; one is required.

| `env.params` key | Default | Meaning |
|---|---|---|
| `env_class`, `env_kwargs` | none, `{}` | dotted path of an in-process `openenv.core.env_server.Environment` subclass and its constructor arguments |
| `action_class` | inferred | dotted path of its `Action` subclass; inferred from `step`'s annotation or the generic base |
| `url` | none | running OpenEnv server; one WebSocket session per episode |
| `image`, `env_vars`, `port` | none | Docker image started with `LocalDockerProvider`, stopped on `close` |
| `timeout_s` | `60.0` | connect and message timeout for `url` and `image` |
| `action_mode` | `text` | `text`: the action is `{action_field: <completion text>}`. `json`: the completion must contain a JSON object (fenced or bare), which is the action |
| `action_field` | `text` | field name used in `text` mode |
| `observation_template` | none | `str.format` template over the observation fields (missing ones are empty). Unset: the value of a lone string field, else the observation as JSON |
| `system_prompt` | `data.system_prompt` | first message |
| `invalid_action_penalty` | `0.0` | `json` mode: cost of a completion without a JSON object |
| `reset_kwargs`, `seed` | `{}`, none | extra arguments of `reset` |

`reset` gets `episode_id=<sample id>`, then `reset_kwargs`, `seed` and the sample's
`metadata["reset_kwargs"]`. Limits: no tool schemas; an invalid `json` action does not reach the
environment; an action the in-process `Action` class rejects raises, which fails the episode.
Run here: `env_class`, stepped from Python against `ReverseEnv` in
`tests/adapters/openenv/sample_envs.py`. Not run here: `url`, `image` (no server, no Docker).

### `hermes`

Builds a Hermes `AIAgent` for its tool schemas, system prompt and tool runtime but never lets it
call a model: tool calls are parsed from the policy's text with the `tool_calling` parser, then
executed by Hermes. No API key is used. Unknown keys in `env.params` are an error.

| `env.params` key | Default | Meaning |
|---|---|---|
| `tools` | `[file, terminal]` | Hermes tool or toolset names; unknown names are an error |
| `tool_modules` | `[]` | modules imported first, so tools they register with Hermes can be named |
| `skills`, `skills_dir` | `[]`, none | skills whose `SKILL.md` body is added to the system prompt; directory searched for `<name>/SKILL.md` (unset: Hermes' own lookup) |
| `hermes_home` | none | `HERMES_HOME` override |
| `workspace_dir` | none | working directory shared by all episodes. Unset: a new directory per episode under `$MONORL_HOME/hermes-workspaces` (`~/.monorl`), removed when the episode ends |
| `sandbox` | `{backend: local}` | `backend`: `local`, `docker`, `singularity`, `modal`, `daytona`, `ssh`; `image`: for all but `local` and `ssh` |
| `max_tool_output_chars` | `8000` | longer tool results go through Hermes' truncation |
| `max_calls_per_turn` | `8` | as in `tool_calling` |
| `invalid_call_penalty` | `0.0` | as in `tool_calling` |
| `system_prompt` | `data.system_prompt` | extra instructions added to the system prompt |
| `hermes_system_prompt` | `true` | `false`: a short prompt listing the tool schemas instead of Hermes' full one |
| `model_name` | `model.name_or_path` | model name given to the agent (never contacted) |

Limits: Hermes has no task scoring, so turn rewards are 0 or `-invalid_call_penalty` and a reward
function must score the result. A non-local `sandbox.backend` sets `TERMINAL_ENV` for the whole
process; with `backend: local` the `terminal` tool runs commands on the host as your user. Run
here: `sandbox: {backend: local}`, `tools: [terminal, read_file, write_file]` and
`hermes_system_prompt: false`, stepped from Python (a `terminal` call, an unknown tool, a final
answer). Not run here: the other backends, `skills`, `configs/recipes/hermes_agent_example.yaml`.

## Writing an environment in Python

An environment is any object with these members (`monorl.core.interfaces.Environment`):

| Member | Contract |
|---|---|
| `max_turns: int` | turn limit read by the runner (1 when missing) |
| `reset(sample) -> list[Message]` | start an episode for `Sample(id, messages, reference, metadata)` |
| `step(action) -> StepResult` | `action` is a `Completion` (`text`, `finish_reason`, ...); return `StepResult(messages, reward=0.0, done=True, info={})`, with no messages on the last step |
| `close() -> None` | release resources; also called after a failure |
| `tools` (optional) | OpenAI-style function schemas for the chat template; not with image inputs |

`$DEMO/guess_env.py`:

```python
import re

from monorl.core.types import Completion, Message, Sample, StepResult


class GuessEnv:
    """Guess the secret integer (the sample's reference) in at most max_turns tries."""

    def __init__(self, *, max_turns: int = 5, high: int = 20) -> None:
        self.max_turns, self.high = max_turns, high

    def reset(self, sample: Sample) -> list[Message]:
        ref = str(sample.reference).strip()
        self._secret, self._turn = (int(ref) if ref.isdigit() else 1), 0
        rules = f"Guess my number between 1 and {self.high}. Reply with a number."
        return [{"role": "system", "content": rules}, *sample.messages]

    def step(self, action: Completion) -> StepResult:
        self._turn += 1
        numbers = re.findall(r"\d+", action.text)
        guess = int(numbers[-1]) if numbers else 0
        solved = guess == self._secret
        done = solved or self._turn >= self.max_turns
        hint = "higher" if guess < self._secret else "lower"
        messages = [] if done else [{"role": "user", "content": hint}]
        reward = 1.0 if solved else 0.0 if numbers else -0.1
        return StepResult(messages, reward=reward, done=done, info={"solved": solved})

    def close(self) -> None:
        pass


def make_guess_env(config, **kwargs) -> GuessEnv:
    return GuessEnv(max_turns=config.env.max_turns, **config.env.params)


def register(registry) -> None:
    registry.register("env", "guess", make_guess_env)


if __name__ == "__main__":
    from monorl.envs.server import serve

    serve(GuessEnv, transport="stdio", name="guess", parallel_episodes=4)
```

### Registering it

A factory takes the `RunConfig` (plus keyword arguments it may ignore) and returns a fresh
environment. A module a run can load exposes `register(registry)`, as `guess_env.py` does. Name
the module in the config's `providers` (in a pipeline config: under `base`) and use the type like
a built-in one:

```yaml
providers: [guess_env]
env: {type: guess, max_turns: 3, params: {high: 20}}
rewards: []
```

The modules in `providers` are imported before a run builds its components, in every process
that builds them, from the working directory or the Python path: start the run where the module
is. Jobs launched from the web UI also find modules in the directory `monorl serve` was started
in. Run here
from `$DEMO` with `monorl train` on `configs/recipes/grpo_gsm8k_smollm_cpu.yaml`, these three
keys as `--set` overrides, `algorithm.gamma=0.9`, `trainer.max_steps=1`,
`rollout.max_new_tokens=24`, `rollout.stop=["."]` and a two-row JSONL
(`{"question": "Start guessing.", "answer": "7"}`) as `data.uri`: the run succeeded. One recorded
episode had turn rewards `[-0.1, -0.1, -0.1]` and reward
`-0.1 + 0.9 * -0.1 + 0.81 * -0.1 = -0.271`; another found the number on its third turn (`0.81`).

### Serving and checking it over MEP/1

Served over MEP/1, the same class needs no registration: a run reaches it with `env.type: remote`
and `env.params: {transport: stdio, command: [...]}` ([ENV_SDK.md](ENV_SDK.md); validated against
the schema with this server's command, not trained here). `monorl.envs.server.serve(factory,
transport=, host=, port=, name=, parallel_episodes=)` serves a zero-argument factory over `stdio`
or `http`, as the `__main__` block above does. `monorl env check` starts the server, then runs
spec, health, reset, `--turns` steps (default 1) and close, validating every response. It resets
with a fixed sample (`reference: "hello"`) and steps with the text `hello`.

```console
$ .venv/bin/monorl env check --command .venv/bin/python --command $DEMO/guess_env.py
stdio .venv/bin/python /tmp/monorl-docs-environments/guess_env.py
spec: name=guess version=0 protocol=MEP/1 max_turns=5 parallel_episodes=4
tools=0
  Guess the secret integer (the sample's reference) in at most max_turns tries.
...
step 1: reward=-0.1 done=False messages=1 info={'solved': False}
close: ok
MEP/1 conformance: OK
```

The exit status is 1 on a protocol error; `--url http://host:port` checks an http server.
`monorl env serve <type>` serves a registered type without a script, with `env.params` from
`--config <run config>` or `--param key=value`, and `--transport stdio|http` (default `stdio`),
`--host` (`127.0.0.1`), `--port` (`8080`), `--parallel` (`8` concurrent episodes). The `providers`
of the `--config` file are loaded, so it serves your types too. Checked here over http:
`monorl env serve tool_calling --transport http --port 18473 --param 'tools=[calculator]'` and
`monorl env serve guess --config serve.yaml --transport http --port 18474` (a config with the
three keys above), each followed by `monorl env check --url http://127.0.0.1:<port>`.

### When to use another language

Write the server in Go, C++ or any language that can exchange JSON over stdio or http when the
simulator already exists in that language or must not share the trainer's Python process. It
implements MEP/1 itself and the run uses `env.type: remote`: see [ENV_SDK.md](ENV_SDK.md). To
train against an existing agent program instead, use `env.type: harness` ([GATEWAY.md](GATEWAY.md)).
