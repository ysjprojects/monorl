# Capture gateway and the `harness` environment

The gateway makes **any program that speaks the OpenAI chat-completions API** usable as an RL
environment: an OpenAI-SDK script, an agent framework, a coding-agent CLI. Nothing in the agent
has to know about monorl.

```
            ┌──────────────── monorl (trainer process) ─────────────────┐
 episode ──▶│ HarnessEnv ──spawns──▶ agent process (own process group)  │
            │     │                      │  OPENAI_BASE_URL=http://127.0.0.1:<port>/s/<session>/v1
            │     │                      ▼
            │     │              capture gateway (FastAPI, localhost)   │
            │     │                      │  prompt ids ─▶ RolloutEngine.generate ─▶ Completion
            │     ▼                      ▼
            │  Trajectory ◀── call records (prompt ids, completion ids, logprobs)
            └────────────────────────────────────────────────────────────┘
```

Per episode monorl starts the agent, the agent calls the policy through an OpenAI-compatible
endpoint that monorl serves **from the policy being trained**, and every call is recorded with the
exact prompt token ids and the sampled completion (ids and logprobs under the sampling policy).
When the agent exits, the calls are the turns of a normal `Trajectory`: rewards, advantages,
batching and the trainer treat it like any other episode.

Code: `src/monorl/gateway/` (`protocol.py` wire format, `session.py` sessions and call records,
`trajectory.py` assembly, `server.py` HTTP server, `agents/` examples) and
`src/monorl/envs/harness.py` (`env.type: harness`). Recipe:
`configs/recipes/grpo_harness_calc_cpu.yaml`.

## 1. Protocol

One gateway per process, started with the first episode, bound to `127.0.0.1` on a free port.
Every episode gets a **session**; its id is unguessable and doubles as the API key.

| Route | |
|---|---|
| `POST /s/{session}/v1/chat/completions` | chat completion for the session (any API key) |
| `POST /v1/chat/completions` | the same, session taken from `Authorization: Bearer <session>` |
| `GET /s/{session}/v1/models`, `GET /v1/models` | the one model of the session |
| `POST /s/{session}/reward` | body `{"reward": <number>}`; adds to the episode's reward, answers `{"reward": <total so far>}` |
| `GET /health` | `{"status": "ok", "sessions": <open sessions>}` |

So an OpenAI client works with either `base_url = http://127.0.0.1:<port>/s/<session>/v1` and any
key, or `base_url = http://127.0.0.1:<port>/v1` and `api_key = <session>`.

### What a chat-completions request may ask for

| Field | Treatment |
|---|---|
| `messages` | Required. `system`/`developer` (→ `system`), `user`, `assistant` (with `tool_calls`), `tool`. Content is a string, `null`, or text parts (joined). |
| `tools` | Function schemas; passed to the model's chat template. With tools the completion is searched for tool calls (below). `tool_choice: "none"` turns that off; every other `tool_choice` behaves like `"auto"`. |
| `max_tokens` / `max_completion_tokens` | May only **lower** `rollout.max_new_tokens`. |
| `stop` | Added to `rollout.stop`. A stop string the request added is cut from the returned text, as OpenAI does. |
| `stream` | `true` answers with server-sent `chat.completion.chunk` events of the *finished* completion (role, content, tool calls, finish reason), then `data: [DONE]`. `stream_options.include_usage` adds the usage chunk. |
| `n` | Must be `1` (or absent): one call is one recorded policy turn. Send separate requests for more samples. |
| `temperature`, `top_p`, `seed`, penalties, `logit_bias`, `logprobs`, `response_format`, `model`, anything else | **Ignored.** |

The sampling fields are ignored on purpose: the trainer owns the sampling distribution. The
logprobs recorded for a turn are the logprobs under `rollout.temperature`/`top_p`/`top_k`, and the
importance ratios of the policy loss are only right if every token was sampled that way. An agent
cannot change it, by accident or otherwise.

Requests that lower `max_tokens` or add `stop` strings are batched separately from the others
(the batching proxy never mixes sampling parameters).

The response is a `chat.completion` object: one choice, `usage` with the real token counts,
`model` = the session's model name. `finish_reason` is `stop`, `length` or `tool_calls`.

### Tool calls

When the request has `tools`, the completion text is parsed with
`monorl.envs.tool_calling.parse_tool_calls` — Hermes style
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` or an OpenAI-style JSON object — and
the calls are returned as OpenAI `tool_calls` (`arguments` a JSON string, `finish_reason:
"tool_calls"`); `content` is the text around the calls, `null` if there is none. A completion
with a malformed call is returned as plain content.

How the model learns about the tools is up to its chat template: `tools` is handed to
`apply_chat_template`. Templates without tool support (SmolLM2, many small models) silently drop
it, and they drop the `tool_calls` of earlier assistant messages as well. For such models put the
call syntax into the conversation, as the example agent does: in the system prompt and in a few
worked examples whose assistant messages carry the call as text. Set
`rollout.stop: ["</tool_call>"]` to end a turn at the call.

### Errors

OpenAI's shape, `{"error": {"message", "type", "param", "code"}}`:

| Status | When |
|---|---|
| 400 | invalid request: `n > 1`, image or other non-text parts (`unsupported_content`), bad `messages`/`tools`, a chat the template cannot render (`template_error`), more calls than `env.max_turns` (`max_turns_exceeded`) |
| 401 | `/v1/...` without the session id as bearer token (`invalid_api_key`) |
| 404 | unknown session in the path (`session_not_found`), unknown route |
| 500 | the rollout engine failed; the episode fails with that error |
| 503 | the session is closed (`session_closed`) or its generation was cancelled because the run is being aborted (`generation_cancelled`) |

### Concurrency

A completion blocks until the rollout engine has produced it. The gateway runs those waits on its
own worker-thread limiter (512 in flight, `monorl.gateway.server.MAX_IN_FLIGHT`), not on the
event loop's default 40 threads, so the calls of all parallel episodes are in flight together and
the run's `BatchingEngine` coalesces them into batched engine calls exactly as it does for
stepped environments.

## 2. From calls to turns and threads

Each answered call is one `Turn`: `prompt_token_ids`, the engine's `Completion` unchanged (text
including any tool-call markup, token ids, logprobs), and `info`:

```
{"call": 3, "parent": 2, "bridged": true,
 "tool_calls": [{"id", "name", "arguments", "result"}], "invalid_calls": 0}    # last two only for requests with tools
```

`result` is the content of the `tool` message that later answered the call, `null` if the agent
never sent one.

**Token in, token out.** For each request the gateway looks for the latest earlier call of the
session whose request messages plus answer are a message-level prefix of the new request, with
identical `tools` (`parent`). If there is one, the new prompt is that call's prompt ids + its
sampled completion ids + only the new messages rendered (`monorl.rollout.render.continue_prompt_ids`):
the agent's copy of the assistant message is never re-tokenised, so the policy sees — and is
trained on — exactly the tokens it sampled (`bridged: true`). The batch builder then trains such
a chain of calls as **one** sequence. With `rollout.turn_rendering: template`, or when the chat
template would not carry the assistant content over verbatim (templates that drop earlier
reasoning), the conversation is re-rendered from its messages (`bridged: false`) and the call
trains as a sequence of its own.

Messages are compared after normalisation (text parts joined, `null` content as `""`,
`tool_calls` arguments decoded, unknown fields dropped), so the SDK's message object sent back
as-is matches. An agent that edits an earlier assistant message, or changes `tools`, starts a new
context.

**Threads.** A request that continues no earlier call — after the agent compacted or rewrote
its history, a sub-agent with its own chat, a one-off side question — starts a new conversation.
`Trajectory.messages` (thread 0) is the conversation of the **last** call: its request messages
plus its answer. Every other conversation that no later call continued is an entry of
`Trajectory.extra_threads`, in the order they ended. A turn belongs to thread 0 if the last call
descends from it, otherwise to the thread of the latest conversation that descends from it;
`Turn.message_index` is the position of its answer there. In threads, assistant tool calls are
monorl's `{id, name, arguments}`.

Reward functions see the trajectory as usual; `scope: last` rewards score the completion of the
last call.

## 3. `env.type: harness`

```yaml
env:
  type: harness
  max_turns: 20            # optional: cap on policy calls per episode
  params:
    command: ["{python}", "my_agent.py", "--task", "{task_file}"]
    timeout: 600
    env: {MY_AGENT_API_BASE: "{base_url}"}
    verifier: ["{python}", "check.py", "{reference_file}", "{stdout_file}"]
    workdir: {template: ./task_skeleton}
```

| Param | |
|---|---|
| `command` | Required. The agent: an argv list, or a string that is split like a shell would split it (no shell runs it). |
| `timeout` | Seconds the agent may run (default 600). |
| `env` | Extra environment variables (values may use placeholders). |
| `verifier` | Optional command run in the working directory after the agent ended. Reward = the last line of its stdout if that is a number, else `1.0` for exit status 0 and `0.0` otherwise. |
| `verifier_timeout` | Seconds (default 60); a verifier that takes longer scores 0. |
| `workdir` | `root`: directory the per-episode directories are created under (default: system temp dir). `template`: directory copied into every fresh working directory. |
| `model` | Model name the agent is told (default `model.name_or_path`). |

Placeholders in `command`, `env` values and `verifier`: `{task_file}`, `{workdir}`,
`{base_url}`, `{api_key}`, `{model}`, `{python}` (the interpreter running monorl), `{session}`,
`{reward_url}`, `{sample_id}`; in `verifier` also `{reference_file}`, `{stdout_file}`,
`{stderr_file}`, `{exit_code}`. Other braces are left untouched.

What happens in an episode:

1. A session is opened and a fresh directory is created: `task.json`, the captured output and
   the working directory `work/` (a copy of `workdir.template` if given).
2. The task file holds the sample's `id`, `messages` and `metadata`. It **never** holds
   `sample.reference`, nor the copies of the answer the data layer keeps in the metadata
   (`answer_raw`, `response`, `assistant_turn`, and the answer column inside `metadata.row`).
   Other dataset columns are passed through under `metadata.row`.
3. The agent starts in the working directory, in its own process group, stdin closed, stdout and
   stderr captured, with these variables on top of monorl's own environment:

   | Variable | |
   |---|---|
   | `OPENAI_BASE_URL`, `OPENAI_API_KEY` | the session's `/s/<session>/v1` URL and id — an unmodified OpenAI SDK client just works |
   | `MONORL_BASE_URL`, `MONORL_SESSION` | the same two values |
   | `MONORL_TASK_FILE`, `MONORL_WORKDIR` | paths of the task file and the working directory |
   | `MONORL_MODEL` | model name to send |
   | `MONORL_REWARD_URL` | `POST {"reward": x}` here to add `x` to the episode's reward |
   | `NO_PROXY`, `no_proxy` | extended by `127.0.0.1,localhost` |

4. The episode ends when the agent exits. After `timeout` its process group is killed; the
   episode still counts if at least one call was answered (`info.timed_out`, `done: false`).
   Anything the agent left running in its process group is killed as well.
5. With a `verifier`, the full sample — reference included — is written to `{reference_file}`
   (only now, after the agent is gone) and the verifier runs with the same variables plus
   `MONORL_REFERENCE_FILE`, `MONORL_AGENT_STDOUT`, `MONORL_AGENT_STDERR`,
   `MONORL_AGENT_EXIT_CODE`. The session is closed by then; a verifier cannot call the policy.
6. The verifier's reward and the rewards posted to `MONORL_REWARD_URL` are added to the **last**
   turn's `reward`. The loop adds per-turn rewards to the reward-function score
   (`Σ gamma^k · reward_k`, reported as `reward/env`), so `rewards: []` with a verifier, a reward
   function alone, or both, all work.
7. The session is closed, the process reaped and the directory removed — also when the episode
   fails or is aborted.

`Trajectory.info`: `exit_code`, `timed_out`, `stdout_tail`, `stderr_tail` (last 2000
characters), `posted_reward` if any, `verifier` (`reward`, `exit_code`, `timed_out`,
`stdout_tail`, `stderr_tail`) if configured, plus `finish_reason` and `num_turns`.

Failure modes:

- **No call was answered** → `EnvironmentError_` with the exit code and the stderr tail (the run
  fails: the agent is broken).
- **Non-zero exit after at least one call** → a normal trajectory with `exit_code` in `info`;
  reward it as you see fit.
- **`env.max_turns` reached** → further calls get a 400 `max_turns_exceeded`; the trajectory has
  `done: false`.
- **Abort** (Ctrl-C, cancelled job, async producer shutdown) → the agent is killed and the
  episode raises `EpisodeAborted`, also out of a generation in flight.

`harness` is an `EpisodeDriver` (`monorl.core.interfaces`): it works wherever episodes are run by
`monorl.orchestrator.episodes.run_episodes` — the native loop, pipeline sampling and eval stages,
the TRL rollout. Backends that step environments themselves (verl, NeMo-RL) cannot use it.

## 4. Plugging in an agent

An OpenAI-SDK script needs no configuration at all, because the SDK reads `OPENAI_BASE_URL` and
`OPENAI_API_KEY`:

```python
# my_agent.py
import json, os
from openai import OpenAI

client = OpenAI()
task = json.load(open(os.environ["MONORL_TASK_FILE"]))
messages = list(task["messages"])
while True:
    reply = client.chat.completions.create(model=os.environ["MONORL_MODEL"], messages=messages, tools=TOOLS)
    message = reply.choices[0].message
    messages.append(message)                      # send the SDK's object back unchanged
    if not message.tool_calls:
        break
    for call in message.tool_calls:
        messages.append({"role": "tool", "tool_call_id": call.id, "content": run_tool(call)})
print(message.content)
```

```yaml
env:
  type: harness
  max_turns: 8
  params:
    command: ["{python}", "my_agent.py"]
```

Guidelines:

- **Append, do not rewrite.** Keep sending the assistant messages back exactly as they came. Then
  the whole episode is one token-exact training sequence. Rewriting history is allowed (it starts
  a new thread) but the earlier calls then train as separate, shorter sequences.
- **Let the trainer sample.** Do not rely on `temperature`, `n` or `response_format`.
- **Agent CLIs**: pass what they need through placeholders, e.g.
  `command: ["my-agent", "--model", "{model}", "--prompt-file", "{task_file}"]` and
  `env: {OPENAI_API_BASE: "{base_url}"}` for tools that read other variable names.
- **Scoring**: if the answer is the last assistant message, a reward function
  (`exact_match` with `scope: last`) is enough. If the result is a file or the program's output,
  use a `verifier`, or let the agent's own harness `POST` to `MONORL_REWARD_URL`.

The shipped example, `python -m monorl.gateway.agents.calc`, is such a loop written with the
standard library only (a calculator tool via function calling). So that small models whose
templates ignore `tools` and `tool_calls` still learn to call the tool, its system prompt spells
the call syntax out and two worked examples precede the task (`--no-examples` leaves them out).
Each example is valid OpenAI history that reads the same with any template: the question, an
assistant message carrying the call both as text
(`<tool_call>\n{"name": "calculator", "arguments": {...}}\n</tool_call>`) and as structured
`tool_calls`, the `tool` result, and the final number. Try it:

```
monorl train --config configs/recipes/grpo_harness_calc_cpu.yaml
```

## 5. Limits

- **Chat completions only.** No `/v1/completions`, `/v1/responses`, embeddings or audio.
- **Text only.** Image/audio parts are rejected with a 400.
- **`n = 1`**, no `logprobs` in responses, no constrained decoding (`response_format`,
  forced `tool_choice`).
- **Streaming is not incremental**: the events are sent once the completion is finished.
- **Localhost.** The gateway binds to `127.0.0.1` and the session id is the only credential, so
  the agent has to run on the trainer's host (a container needs host networking).
- Tool calls are recognised in the syntaxes `parse_tool_calls` knows; the model has to be
  prompted (by its template or the system prompt) to use one of them.
- One call is one generation: a timed-out agent's generation in flight still finishes on the
  engine and is discarded.
