# Rewards

This document covers the `rewards:` section of a run config: how entries combine into an episode's
total reward, every built-in reward function with its parameters and run-time behaviour, and how
to add one. How the total becomes an advantage is in [ALGORITHMS.md](ALGORITHMS.md).

## Configuring `rewards`

`rewards` is a list of entries (`RewardSpec` in `src/monorl/config/schema.py`) with three fields:
`type` (required, a name in the registry namespace `reward`), `weight` (default `1.0`) and `params`
(default `{}`, the dict passed to the reward's factory).

```yaml
rewards:
  - {type: exact_match, params: {extract: last_number}}
  - {type: format, weight: 0.1, params: {name: phrasing, patterns: ["answer is"]}}
```

`monorl.rewards.build_reward` turns the list into one `CompositeReward` whose score is the sum of
`weight * score` over the entries; weights are not normalised. Each entry's unweighted score is
kept in a breakdown under `params.name`, else the `type` (a repeated key gets `#2`, `#3`, ...). An
unknown `type` passes config validation and raises `RegistryError` when the loop is built.

A reward function receives whole `Trajectory` objects. Every built-in accepts `params.name`. All
except `reward_model` and `composite` accept `scope`: `last` (default) scores the text of the
final policy turn only, `all` the texts of all policy turns joined in order with no separator;
environment messages such as tool results are in neither. `reward_model` always scores the prompt
messages plus the final turn. The reference is `sample.reference` (see [DATA.md](DATA.md)), or
with `reference_key: k` (`exact_match`, `math_verify`, `code_exec`, `llm_judge`) `reference[k]` of
a dict reference holding `k`, else `sample.metadata[k]` when present, else the whole reference.

`rewards: []` is rejected at validation with `env.type: single_turn` and valid otherwise: the
environment then supplies the whole reward. Per-turn rewards returned by an environment
(`Turn.reward`, see [ENVIRONMENTS.md](ENVIRONMENTS.md)) are added in `ComponentLoop._score`
(`src/monorl/orchestrator/loop.py`): the total is the weighted sum, plus the sum over turns `k`
(from 0) of `algorithm.gamma ** k * turn.reward`, minus `algorithm.overlong_penalty` when the
final turn's finish reason is `length`. The discounted environment sum is the breakdown key `env`,
metric `reward/env`, present whenever a turn reward is non-zero; the penalty is reported as
`reward/overlong_penalty`. Checked here with `_score`: turn rewards 1, 2, 0, `gamma: 0.5` and a
matching `exact_match` give total `3.0` with `env` `2.0`. For `algorithm.credit_assignment: turn`
see [CREDIT_ASSIGNMENT.md](CREDIT_ASSIGNMENT.md).

## Built-in rewards

The examples form one `.venv/bin/python` session that starts with this helper; comments show output.

```python
from monorl.core.registry import registry
from monorl.core.types import Completion, RewardResult, Sample, Trajectory, Turn

def traj(*responses, reference=None):
    msgs = [{"role": "user", "content": "Solve it."}]
    done = [Completion(text=r, token_ids=list(range(len(r.split())))) for r in responses]
    turns = [Turn(prompt_token_ids=[1], completion=c) for c in done]
    sample = Sample(id="s0", messages=msgs, reference=reference)
    return Trajectory(sample=sample, group_id="g0", messages=list(msgs), turns=turns)
```

### `exact_match`

Returns 1.0 when the answer extracted from the response equals the reference, else 0.0. The
reference goes through the same extractor (raw when nothing is extracted) and may be a list of
acceptable answers. Two numbers (after removing `,`, `$`, a trailing `%`) are compared numerically.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `extract` | `last_number` | `last_number`, `boxed` (last `\boxed{}`), `after_hash`, `none` |
| `normalize` | `[whitespace, case]` | For string comparison: collapse whitespace, lower-case |
| `numeric_tolerance` | `1e-6` | Absolute tolerance of the numeric comparison |

```yaml
rewards:
  - {type: exact_match, params: {extract: boxed, normalize: [whitespace]}}
```

```python
fn = registry.get("reward", "exact_match")({"extract": "last_number"})
r = fn([traj("3 + 4 = 7. The answer is 7.", reference="#### 7")])[0]
print(r.score, r.breakdown, r.info)  # 1.0 {'exact_match': 1.0} {'extracted': '7'}
print(fn([traj("I think 7.", "No, it is 9.", reference="7")])[0].score)  # 0.0 (two turns)
```

### `math_verify`

Returns 1.0 when the response's final answer is mathematically equivalent to the reference per
`math-verify`, else 0.0. Without the `rewards` extra the factory raises `BackendUnavailable`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `timeout_s` | `5` | Parse/verify timeout, applied on the main thread only (`signal.alarm`) |
| `float_rounding`, `numeric_precision`, `strict` | `6`, `15`, `True` | To `math_verify.verify` |

```yaml
rewards:
  - type: math_verify
```

```python
fn = registry.get("reward", "math_verify")({})
r = fn([traj("So the result is $\\boxed{\\frac{1}{2}}$.", reference="0.5")])[0]
print(r.score, r.info)  # 1.0 {'extracted': ['1/2', '\\frac{1}{2}']}
```

### `format`

Regex check of the response: 1.0 or 0.0, or a fraction in [0, 1] with `require_all: false`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `patterns` | required | List of regexes; one string, or the key `pattern`, also works |
| `require_all` | `True` | `True`: 1.0 only if every pattern matches. `False`: fraction matched |
| `flags`, `fullmatch` | `[DOTALL]`, `False` | `re` flag names; `re.fullmatch`, not `re.search` |

```yaml
rewards:
  - {type: format, weight: 0.2, params: {patterns: ["<think>.*</think>"]}}
```

```python
params = {"patterns": ["<think>.*</think>", "<answer>.*</answer>"], "require_all": False}
r = registry.get("reward", "format")(params)([traj("<think>2+2</think> 4")])[0]
print(r.score, r.info)  # 0.5 {'matched': [True, False]}
```

### `length`

Returns `base` when the response length is inside `[min_tokens, max_tokens]` and
`max(min_reward, base - penalty_per_token * excess)` outside it: the range is `[min_reward, base]`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_tokens`, `max_tokens` | `0`, unset | Band in `unit`; no upper bound when unset |
| `penalty_per_token`, `base`, `min_reward` | `0.01`, `1.0`, `0.0` | See the formula above |
| `unit` | `tokens` | `tokens` (completion token ids), `chars`, or `words` (whitespace split) |

```yaml
rewards:
  - {type: length, weight: 0.1, params: {unit: words, min_tokens: 2, max_tokens: 4}}
```

```python
params = {"unit": "words", "min_tokens": 2, "max_tokens": 4, "penalty_per_token": 0.25}
r = registry.get("reward", "length")(params)([traj("one two three four five six")])[0]
print(r.score, r.info)  # 0.5 {'length': 6}
```

### `code_exec`

Runs the last fenced code block tagged `language` (or untagged) of the response against the test
cases in the reference. Returns the fraction of cases passed, or with `mode: all` 1.0 only if all
pass; 0.0 without a code block; without test cases, 1.0 if the code exits with status 0. The
reference is test code appended to the solution (a string), a list of such strings or of
`{"input": ..., "output": ...}` dicts (stdin, expected stdout), or one of the other layouts
accepted by `normalize_tests` in `src/monorl/rewards/code_exec.py`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `timeout_s`, `memory_mb` | `5.0`, `512` | Per test case: wall-clock limit, memory cap (`0`: off) |
| `language`, `python` | `python`, current interpreter | Fence tag; interpreter used |
| `mode`, `max_workers` | `fraction`, `min(8, cpu_count)` | `fraction` or `all`; thread count |

```yaml
rewards:
  - {type: code_exec, params: {mode: all, timeout_s: 10, memory_mb: 256}}
```

```python
fence = "`" * 3
code = f"{fence}python\ndef add(a, b):\n    return a + b\n{fence}"
fn = registry.get("reward", "code_exec")({"timeout_s": 5})
r = fn([traj(code, reference=["assert add(1, 2) == 3", "assert add(2, 2) == 5"])])[0]
print(r.score, r.info["cases"])  # 0.5 ['passed', 'error']
```

### `llm_judge`

Asks an OpenAI-compatible chat-completions endpoint for a grade and parses the last
`<score>N</score>` of the reply, clamped to `[0, max_score]`. The score is `N / max_score`, or `N`
with `normalize: false`. Not run here against an endpoint; the entry below was validated only.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `base_url`, `model` | required | Endpoint root such as `http://host:8000/v1`; judge model name |
| `rubric` | built-in | Template: `{prompt}`, `{response}`, `{reference}`, `{max_score}` |
| `api_key_env`, `system_prompt` | `OPENAI_API_KEY`, unset | Bearer token variable; system message |
| `max_score`, `normalize` | `10`, `True` | Scale of `N`; whether to divide by it |
| `temperature`, `max_tokens`, `extra_body` | `0.0`, `512`, `{}` | Request fields; extra fields |
| `timeout_s`, `max_workers`, `on_error` | `60`, `8`, `zero` | Seconds; threads; `zero` or `raise` |

```yaml
rewards:
  - type: llm_judge
    params: {base_url: "http://127.0.0.1:8000/v1", model: judge, api_key_env: JUDGE_KEY}
```

### `reward_model`

Scores the conversation with an `AutoModelForSequenceClassification` checkpoint: logit
`label_index`, unbounded, or its sigmoid in (0, 1) with `normalize: sigmoid`. Not run here (no
checkpoint was loaded); the entry below was validated only.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model`, `trust_remote_code` | required, `False` | Hugging Face id or path; to `from_pretrained` |
| `device`, `dtype` | `cpu`, `float32` | Torch device; `float32`, `bfloat16` or `float16` |
| `batch_size`, `max_length` | `8`, `4096` | Texts per forward pass; truncation length in tokens |
| `label_index`, `normalize` | `0`, `none` | Logit used as the score; `none` or `sigmoid` |

```yaml
rewards:
  - {type: reward_model, params: {model: /models/my-reward-model, normalize: sigmoid}}
```

### `composite`

A weighted sum of member rewards as one entry: `params.rewards` (required) is a list shaped like
top-level `rewards`. As a top-level entry only its total, under its `name`, is in the breakdown.

```yaml
rewards:
  - type: composite
    params: {rewards: [{type: length}, {type: format, params: {pattern: "7"}}]}
```

```python
members = [{"type": "length"}, {"type": "format", "params": {"pattern": "7"}}]
r = registry.get("reward", "composite")({"rewards": members})([traj("It is 7.")])[0]
print(r.score, r.breakdown)  # 2.0 {'length': 1.0, 'format': 1.0}
```

## Operational behaviour

`code_exec` runs every test case in a new `python -I` subprocess in its own session and process
group, with a fresh temporary directory (deleted afterwards) as working directory, `HOME` and
`TMPDIR`, and an environment of only those two, `PATH` and three `PYTHON*` settings. On timeout or
over the memory cap the process group gets `SIGKILL` (case status `timeout` or `memory`); a
non-zero exit is `error`. The cap is set in the child with `resource.setrlimit` (which only
Linux enforces) and enforced by the parent, which polls the resident set size of the process tree
every 20 ms. This is no security
boundary: same user, filesystem and network access, no container, no CPU or process-count limit.

`llm_judge` sends one `POST {base_url}/chat/completions` per trajectory, up to `max_workers` at a
time. The only environment variable read is the one named by `api_key_env`, at construction; if
unset, no `Authorization` header is sent. There are no retries. With `on_error: zero` an HTTP error,
a malformed body or a reply without a score tag scores 0.0 with `info["error"]`; `raise` propagates.

`reward_model` loads tokenizer and model once, when the loop is built, in the loop's process. Each
conversation is rendered with the tokenizer's chat template when it has one, else as
`role: content` lines, then scored in batches of `batch_size`, truncated to `max_length`.

The loop calls the composite once per step with all of that step's trajectories and stores each
result's `info` in `Trajectory.info["reward_info"]`. It does not catch exceptions from rewards: run
here with a reward that raises, the run ended `failed` at step 0 with the error in `status.json`.

## Metrics

`reward/mean`, `reward/std`, `reward/<name>` per breakdown key (including `reward/env` and
`reward/overlong_penalty`), `reward/success_rate`, `reward/pass_at_k`, `time/reward`,
`eval/reward_mean`, `eval/reward_std`, `eval/reward/<name>`: see [RUNS.md](RUNS.md#metrics).

## Pipeline backends

From the adapters' code, not run here. With `trainer.backend: trl` the composite is wrapped as one
TRL reward function (`src/monorl/adapters/trl/rewards.py`) that computes the total as above,
including `env` and `overlong_penalty`. With `verl` (`src/monorl/adapters/verl/reward.py`) at
least one entry is required: a generated `<run_dir>/monorl_reward.py` scores one response at a
time, and the breakdown goes to verl's `reward_extra_info`. With `nemo_rl` (`env_bridge.py` of
that adapter) the score is added to the final turn's reward. Under `verl` and `nemo_rl` a reward
sees the decoded text of a response and no token ids, so `length` with `unit: tokens` (its
default) is rejected at validation: use `chars` or `words`. See [BACKENDS.md](BACKENDS.md).

## Writing a reward function

The interface is the `RewardFunction` protocol in `src/monorl/core/interfaces.py`: an object with
a `name` attribute and `__call__(trajectories) -> list[RewardResult]`, one result per trajectory,
in order. What gets registered (`src/monorl/core/registry.py`, `registry.register` or the
decorator below) is a factory `factory(params: dict, **kw)`; `build_reward` passes it
`config=<RunConfig>` as a keyword. Continuing the session above:

```python
class KeywordReward:
    def __init__(self, params):
        self.name = str(params.get("name") or "keyword")
        self.keyword = str(params["keyword"]).lower()
    def __call__(self, trajs):
        scores = [float(self.keyword in t.last_response_text.lower()) for t in trajs]
        return [RewardResult(score=s, breakdown={self.name: s}) for s in scores]

@registry.provides("reward", "keyword")
def make_keyword(params, **_):
    return KeywordReward(params)

print(registry.get("reward", "keyword")({"keyword": "seven"})([traj("Seven.")])[0])
# RewardResult(score=1.0, breakdown={'keyword': 1.0}, info={})
```

The registry is a per-process global: nothing discovers reward modules outside monorl, and the
`monorl` command has no option to import one. A run finds your reward when the module that
registers it was imported in the process that starts the run. A launcher that keeps the whole
command line, with the code above saved as `my_rewards.py`:

```python
import my_rewards  # noqa: F401  registers the "keyword" reward

from monorl.cli.main import app

app()
```

`python train.py train --config my.yaml` then behaves like `monorl train` (run here for one CPU
step with `--set 'rewards=[{type: keyword, params: {keyword: the}}]'`; the step reported
`reward/keyword`). Runs started from the web UI use the stock entry point and see the built-in
names only. A reward that belongs in monorl itself is added to `REWARD_NAMES` in
`src/monorl/rewards/__init__.py`; its tests go in `tests/rewards/`, whose `conftest.py` has a
`traj` fixture that builds a trajectory from a response and a reference.
