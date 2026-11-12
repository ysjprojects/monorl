# Capture gateway and the `harness` environment

The gateway makes **any program that speaks the OpenAI chat-completions API, the OpenAI
Responses API or the Anthropic Messages API** usable as an RL environment: an SDK script, an
agent framework, a coding-agent CLI. Nothing in the agent has to know about monorl.

```
            ┌──────────────── monorl (trainer process) ─────────────────┐
 episode ──▶│ HarnessEnv ──spawns──▶ agent process (own process group)  │
            │     │                      │  OPENAI_BASE_URL=http://127.0.0.1:<port>/s/<session>/v1
            │     │                      │  ANTHROPIC_BASE_URL=http://127.0.0.1:<port>/s/<session>
            │     │                      ▼
            │     │              capture gateway (FastAPI, localhost)   │
            │     │                      │  prompt ids ─▶ RolloutEngine.generate ─▶ Completion
            │     ▼                      ▼
            │  Trajectory ◀── call records (prompt ids, completion ids, logprobs)
            └────────────────────────────────────────────────────────────┘
```

Per episode monorl starts the agent, the agent calls the policy through an endpoint that monorl
serves **from the policy being trained**, and every call is recorded with the exact prompt token
ids and the sampled completion (ids and logprobs under the sampling policy). When the agent
exits, the calls are the turns of a normal `Trajectory`: rewards, advantages, batching and the
trainer treat it like any other episode.

Code: `src/monorl/gateway/` (`protocol.py` what the wire formats share and chat completions,
`openai_responses.py`, `anthropic_messages.py`, `images.py` image inputs, `session.py` sessions
and call records, `trajectory.py` assembly, `server.py` HTTP server, `agents/` examples) and
`src/monorl/envs/harness.py` (`env.type: harness`). Recipe:
`configs/recipes/grpo_harness_calc_cpu.yaml`.

## 1. Protocol

One gateway per process and bind address, started with the first episode, bound to `127.0.0.1`
on a free port unless configured otherwise. Every episode gets a **session**; its id is
unguessable and doubles as the API key. Every route except `/health` needs it, in the path or
as the key:

| Route (under `/s/{session}`, or bare with the session as API key) | |
|---|---|
| `POST /v1/chat/completions` | OpenAI chat completions |
| `POST /v1/responses` | OpenAI Responses |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/messages/count_tokens` | Anthropic token counting: `{"input_tokens": n}`, the prompt tokens the same request would be answered from. Not a policy call. |
| `GET /v1/models` | the one model of the session (OpenAI's list shape) |
| `POST /s/{session}/reward` | body `{"reward": <number>}`; adds to the episode's reward, answers `{"reward": <total so far>}` (path only) |
| `GET /health` | `{"status": "ok", "sessions": <open sessions>}` |

The API key is read from `Authorization: Bearer <session>` (OpenAI clients, Anthropic auth
tokens) or `x-api-key: <session>` (Anthropic API keys). So a client works with either the
session's base URL and any key, or the gateway's root and the session id as key:

| Client | Base URL | Key |
|---|---|---|
| OpenAI (chat completions, Responses) | `http://127.0.0.1:<port>/s/<session>/v1` — `gateway.base_url(session)` | any |
| | `http://127.0.0.1:<port>/v1` | the session id |
| Anthropic | `http://127.0.0.1:<port>/s/<session>` — `gateway.anthropic_base_url(session)` | any |
| | `http://127.0.0.1:<port>` | the session id |

All three APIs are views of **one session model**. A request is turned into one normalised chat
(system/user/assistant/tool messages, function-tool schemas in the form chat templates take),
answered by one sample from the policy, and recorded as one `Turn` (§2). What differs is only how
the chat is written down on the wire and how the answer is handed back.

### What every API honours, and what it ignores

| | Chat completions | Responses | Messages |
|---|---|---|---|
| The chat | `messages` | `instructions` + `input` (+ `previous_response_id`) | `system` + `messages` |
| Function tools | `tools` (`function`) | `tools` (flat `function`) | `tools` (`name`, `description`, `input_schema`) |
| Turning tool-call parsing off | `tool_choice: "none"` | `tool_choice: "none"` | `tool_choice: {"type": "none"}` |
| Shorter completion (may only **lower** `rollout.max_new_tokens`) | `max_tokens`, `max_completion_tokens` | `max_output_tokens` | `max_tokens` |
| Extra stop strings (added to `rollout.stop`) | `stop` | — | `stop_sequences` |
| Streaming | `stream` (`stream_options.include_usage`) | `stream` | `stream` |

**Ignored, in every API:** `temperature`, `top_p`, `top_k`, `seed`, penalties, `logit_bias`,
`logprobs`/`top_logprobs`, `response_format` / `text.format` / `output_config`, `reasoning` /
`thinking`, any other `tool_choice` (it behaves like "auto"), `parallel_tool_calls`, `truncation`,
`store`, `metadata`, `cache_control` markers, `service_tier`, `model`, and everything unknown.

The sampling fields are ignored on purpose: the trainer owns the sampling distribution. The
logprobs recorded for a turn are the logprobs under `rollout.temperature`/`top_p`/`top_k`, and the
importance ratios of the policy loss are only right if every token was sampled that way. An agent
cannot change it, by accident or otherwise. (A Responses object reports the run's `temperature`
and `top_p`: what was sampled with.)

Requests that lower the completion length or add stop strings are batched separately from the
others (the batching proxy never mixes sampling parameters). A stop string the request added is
cut from the returned text, as the vendors do; the recorded turn keeps it.

What the policy cannot do is **refused with a 400, not ignored**: `n > 1`, hosted or server tools
(web search, code interpreter, computer use, MCP, ...), stored conversations and prompt
templates, background responses, uploaded files, audio/document parts, reasoning items and
thinking blocks in the history, assistant prefill (a Messages conversation ending with an
assistant message).

### Chat completions

`messages`: `system`/`developer` (→ `system`), `user`, `assistant` (with `tool_calls`), `tool`.
Content is a string, `null`, or parts: `text` parts are joined, `image_url` parts are images
(below). The response is a `chat.completion` object: one choice, `usage` with the real token
counts, `model` = the session's model name. `finish_reason` is `stop`, `length` or `tool_calls`.

### Responses

- `instructions` is a leading `system` message. `input` is a string (one `user` message) or a
  list of items: messages (`role` `user` / `assistant` / `system` / `developer`; content a
  string or `input_text` / `output_text` / `input_image` parts), `function_call` items (the tool
  calls of the assistant message they follow) and `function_call_output` items (`tool`
  messages; the output is a string or `input_text` parts).
- `previous_response_id` names an earlier response **of the same session**: the chat is that
  call's chat, its answer, then `input`. As with OpenAI, the earlier `instructions` are not
  inherited: repeat them and the request continues the call token for token, change or drop them
  and it starts a new context. An unknown id is a 400 `previous_response_not_found`.
- The answer is a `response` object: `output` holds a `message` item with one `output_text`
  part (left out when the completion is nothing but tool calls) and one `function_call` item per
  tool call (`call_id`, `name`, `arguments` as a JSON string); `status` is `completed`, or
  `incomplete` with `incomplete_details.reason: "max_output_tokens"` when the length limit cut
  the completion; `usage` has the real `input_tokens` / `output_tokens`.
- Responses are kept for the session's lifetime only; there is no `GET /v1/responses/{id}`.

### Messages

- `system` (a string or text blocks) is a leading `system` message. In a `user` message, `text`
  and `image` blocks form one `user` message and every `tool_result` block is a `tool` message
  (its content a string or text blocks; `is_error` is not passed on), in the order sent. In an
  `assistant` message, `text` blocks are the content and `tool_use` blocks the tool calls.
  Several text blocks of a message are concatenated.
- The answer is a `message`: a `text` block (left out when the completion is nothing but tool
  calls) and one `tool_use` block per tool call (`input` is the arguments object). `stop_reason`
  is `tool_use`, `max_tokens`, `stop_sequence` (with `stop_sequence` naming which of the
  request's it was) or `end_turn`; `usage` has the real `input_tokens` / `output_tokens`.
- `anthropic-version` and `anthropic-beta` headers are accepted and not interpreted.

### Tool calls

When the request has tools, the completion text is parsed with
`monorl.envs.tool_calling.parse_tool_calls` — Hermes style
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` or an OpenAI-style JSON object — and
the calls are returned in the API's native form: OpenAI `tool_calls` (`finish_reason:
"tool_calls"`), Responses `function_call` items, Anthropic `tool_use` blocks (`stop_reason:
"tool_use"`). The text around the calls is the answer's text. A completion with a malformed call
is returned as plain text.

How the model learns about the tools is up to its chat template: the schemas are handed to
`apply_chat_template` as OpenAI function tools, whichever API sent them. Templates without tool
support (SmolLM2, many small models) silently drop them, and they drop the tool calls of earlier
assistant messages as well. For such models put the call syntax into the conversation, as the
example agent does: in the system prompt and in a few worked examples whose assistant messages
carry the call as text. Set `rollout.stop: ["</tool_call>"]` to end a turn at the call.

### Streaming

With `stream: true` the request is first admitted and its prompt built — everything it can be
refused for is still a plain HTTP error — and then the stream **opens at once** and stays alive
for as long as the generation takes: every 10 seconds
(`monorl.gateway.server.KEEPALIVE_S`) without news the gateway sends the API's keep-alive, so
clients and proxies with idle timeouts do not give up while a batch is being generated.

| | Opens with | Keep-alive | Then |
|---|---|---|---|
| Chat completions | the `role` chunk | SSE comment `: keep-alive` | `chat.completion.chunk` events: the content, the tool calls, the finish reason, the usage chunk if asked for, `data: [DONE]` |
| Responses | `response.created`, `response.in_progress` | SSE comment `: keep-alive` | per output item `response.output_item.added`, `response.content_part.added` / `response.output_text.delta` / `response.output_text.done` / `response.content_part.done` or `response.function_call_arguments.delta` / `.done`, `response.output_item.done`; then `response.completed` (or `response.incomplete`) with the whole response. `sequence_number` counts the events. |
| Messages | `message_start` (with `usage.input_tokens`) | `ping` events | per block `content_block_start`, `content_block_delta` (`text_delta` / `input_json_delta`), `content_block_stop`; then `message_delta` (stop reason, usage), `message_stop` |

`RolloutEngine.generate` returns whole completions, so tokens are **not** delivered one by one
while they are generated: the completion arrives as one text delta (and one arguments delta per
tool call) when it is done. The streamed answer is the same as the one a request without
`stream` gets.

A generation that fails after the stream began is reported in-band: an `{"error": ...}` data
event (chat completions), `response.failed` (Responses), an `error` event (Messages). A client
that hangs up does not stop the generation; the call is recorded like any other.

### Images

For vision-language policies (the rollout engine has a processor: `hf` with a VLM checkpoint;
pass `processor=` to `Gateway.open_session` for engines that do not expose theirs) user
messages may carry images:

| API | Image part |
|---|---|
| Chat completions | `{"type": "image_url", "image_url": {"url": <https URL or data:image/...;base64,...>}}` |
| Responses | `{"type": "input_image", "image_url": <https URL or data URL>}` |
| Messages | `{"type": "image", "source": {"type": "base64", "media_type", "data"}}` or `{"type": "url", "url"}` |

The gateway stores the bytes once in the media cache (`$MONORL_HOME/cache/media/<sha256>.<ext>`)
— a remote image is downloaded exactly once, at most 20 MB — and the chat carries a monorl
image part that refers to the file. Such a chat goes to the rollout engine as messages (token
ids cannot carry pixels); the engine encodes it with the model's processor and the recorded
prompt ids include the image placeholder tokens. The trajectory is what a stepped multimodal
episode produces: `build_train_batch(..., processor=...)` re-encodes the final conversation,
checks that the ids match and attaches the pixel tensors. `detail` is ignored.

Refused with a 400:

- images for a text-only policy;
- images together with `tools` — a chat that travels as messages cannot carry tool schemas;
- images outside `user` messages (system or assistant messages, tool results);
- anything that is not an `http(s)` URL or base64 data: local paths, `file://`, uploaded files
  (`file_id`), broken data, a URL that cannot be fetched (`invalid_image_url`,
  `invalid_base64_image`, `failed_to_download_image`, `image_too_large`);
- remote URLs when the session was opened with `fetch_images=False` (the gateway then makes no
  outbound request on an agent's behalf — worth setting for agents confined to a sandbox
  without network access);
- **a call without images after a call with images.** The batch builder trains a multimodal
  episode as the conversation it ended in, with that conversation's pixels; a later text-only
  conversation would leave the image turns without them. Keep the images in the history. (The
  other order is fine: text-only calls made before the first image are simply not part of the
  trained sequence.)

The context budget counts the image tokens: the prompt is encoded before anything is generated.

### Errors

In the shape of the API the request came in on — OpenAI's `{"error": {"message", "type",
"param", "code"}}` for chat completions and Responses, Anthropic's `{"type": "error", "error":
{"type", "message"}}` for Messages (`type`: `invalid_request_error`, `authentication_error`,
`not_found_error`, `api_error`):

| Status | When |
|---|---|
| 400 | invalid request: what the tables above list as refused (`unsupported_content` for content the policy cannot take), bad messages/items/tools, a chat the template or processor cannot render (`template_error`), an unknown `previous_response_id`, more calls than `env.max_turns` (`max_turns_exceeded`), a prompt over the context budget |
| 401 | a bare `/v1/...` route without the session id as key (`invalid_api_key`) |
| 404 | unknown session in the path (`session_not_found`), unknown route |
| 500 | the rollout engine failed; the episode fails with that error |
| 503 | the session is closed (`session_closed`) or its generation was cancelled because the run is being aborted (`generation_cancelled`) |

**The context budget.** With `rollout.max_context_tokens` set, a request whose prompt leaves no
room for a completion is refused before anything is generated, worded the way the vendor words
it so that agent frameworks which detect the condition (and compact their history) do:

| API | 400 body |
|---|---|
| Chat completions | `code: "context_length_exceeded"`, `param: "messages"`, `This model's maximum context length is <budget> tokens. However, your messages resulted in <n> tokens. Please reduce the length of the messages.` |
| Responses | `code: "context_length_exceeded"`, `param: "input"`, `Your input exceeds the context window of this model. Please adjust your input and try again.` |
| Messages | `type: "invalid_request_error"`, `prompt is too long: <n> tokens > <budget> maximum` |

A prompt that fits gets a completion capped to the room that is left.

### Concurrency

A completion blocks until the rollout engine has produced it. The gateway runs those waits on its
own worker-thread limiter (512 in flight, `monorl.gateway.server.MAX_IN_FLIGHT`), not on the
event loop's default 40 threads, so the calls of all parallel episodes are in flight together and
the run's `BatchingEngine` coalesces them into batched engine calls exactly as it does for
stepped environments.

### Reachability

By default the gateway listens on `127.0.0.1` only, so the agent has to run on the trainer's
host. For agents that run elsewhere — a container, another machine — bind it to an address
they can reach and tell them the name this machine has for them:

```python
gateway = get_gateway(host="0.0.0.0", port=8811)          # one shared gateway per bind address
gateway.base_url(session, host="host.docker.internal")    # http://host.docker.internal:8811/s/<session>/v1
gateway.anthropic_base_url(session, host="host.docker.internal")
gateway.reward_url(session, host="host.docker.internal")
```

```yaml
env:
  type: harness
  params:
    gateway: {bind: 0.0.0.0, port: 8811, host: host.docker.internal}
```

`host` only changes the URLs handed to the agent; without it they name the bind address
(loopback for a gateway bound to every interface). Beyond loopback the gateway is reachable by
everything that can reach the port: the unguessable session id (128 random bits, valid for one
episode) is the only credential, every route except `/health` requires it, and the traffic is
plain HTTP. Bind to the narrowest interface that works (a bridge address rather than
`0.0.0.0`) and do not expose the port to networks you do not trust.

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

Messages are compared in the normalised form every API is turned into (text parts and blocks
joined, `null` content as `""`, tool-call arguments decoded, images as files of the media cache,
unknown fields dropped), so what the gateway answered with — the SDK's message object, the
Responses output items, Anthropic's content blocks — matches when it is sent back as it came,
and a conversation begun over one API can be continued over another. An agent that edits an
earlier assistant message, or changes `tools`, starts a new context. A chat with images is
encoded again by the model's processor on every call (`bridged: false`); its calls still chain
into one training sequence when the re-encoded prompt starts with the earlier prompt and
completion tokens.

**Threads.** A request that continues no earlier call — after the agent compacted or rewrote
its history, a sub-agent with its own chat, a one-off side question — starts a new conversation.
`Trajectory.messages` (thread 0) is the conversation of the **last** call: its request messages
plus its answer. Every other conversation that no later call continued is an entry of
`Trajectory.extra_threads`, in the order they ended. A turn belongs to thread 0 if the last call
descends from it, otherwise to the thread of the latest conversation that descends from it;
`Turn.message_index` is the position of its answer there. In threads, assistant tool calls are
monorl's `{id, name, arguments}` and a user message with images has monorl content parts
(`{"type": "image", "image": <file>}`).

`Trajectory.info["tools"]` holds the tool schemas the prompts of thread 0 were rendered with
(chat-completions function schemas, whichever API sent them); it is absent when that
conversation had none.

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
| `gateway` | `bind`: address the gateway listens on, `port` (default: any free one), `host`: the name the agent reaches this machine under (§1, Reachability). The defaults follow where the agent runs: loopback for processes on this host. |

Placeholders in `command`, `env` values and `verifier`: `{task_file}`, `{workdir}`,
`{base_url}`, `{anthropic_base_url}`, `{api_key}`, `{model}`, `{python}` (the interpreter running
monorl), `{session}`, `{reward_url}`, `{sample_id}`; in `verifier` also `{reference_file}`,
`{stdout_file}`, `{stderr_file}`, `{exit_code}`. Other braces are left untouched.

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
   | `OPENAI_BASE_URL`, `OPENAI_API_KEY` | the session's `/s/<session>/v1` URL and id — an unmodified OpenAI SDK client (chat completions or Responses) just works |
   | `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` | the session's `/s/<session>` URL and id — an unmodified Anthropic SDK client just works |
   | `MONORL_BASE_URL`, `MONORL_SESSION` | the OpenAI base URL and the session id |
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
`OPENAI_API_KEY`, which the harness sets:

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

The same loop against the **Responses API** (same two variables; the SDK posts to
`$OPENAI_BASE_URL/responses`). Either send the history back yourself, or let
`previous_response_id` do it:

```python
client = OpenAI()
response = client.responses.create(model=MODEL, instructions=SYSTEM, input=task_text, tools=TOOLS)
while calls := [item for item in response.output if item.type == "function_call"]:
    outputs = [{"type": "function_call_output", "call_id": c.call_id, "output": run_tool(c)} for c in calls]
    response = client.responses.create(
        model=MODEL, instructions=SYSTEM, previous_response_id=response.id, input=outputs, tools=TOOLS
    )
print(response.output_text)
```

Repeat `instructions` in every request: as with OpenAI they are not inherited from the previous
response, and a request without them starts a new context (§2).

An **Anthropic-SDK** script needs no configuration either: the SDK reads `ANTHROPIC_BASE_URL`
(the session's root URL; the SDK appends `/v1/messages`) and `ANTHROPIC_API_KEY` (or
`ANTHROPIC_AUTH_TOKEN`), which the harness sets:

```python
from anthropic import Anthropic

client = Anthropic()
messages = [{"role": "user", "content": task_text}]
while True:
    reply = client.messages.create(model=MODEL, max_tokens=1024, system=SYSTEM, messages=messages, tools=TOOLS)
    messages.append({"role": "assistant", "content": reply.content})   # the blocks, unchanged
    if reply.stop_reason != "tool_use":
        break
    results = [
        {"type": "tool_result", "tool_use_id": block.id, "content": run_tool(block)}
        for block in reply.content if block.type == "tool_use"
    ]
    messages.append({"role": "user", "content": results})
```

Outside the harness, the same values come from the gateway
(`from monorl.gateway import get_gateway`):

| Client | Point it at | Key |
|---|---|---|
| OpenAI SDK (chat completions, Responses) | `OPENAI_BASE_URL` / `base_url=` ← `gateway.base_url(session)` | `OPENAI_API_KEY` / `api_key=` ← anything (or `http://host:port/v1` with the session id as key) |
| Anthropic SDK | `ANTHROPIC_BASE_URL` / `base_url=` ← `gateway.anthropic_base_url(session)` | `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` ← anything (or `http://host:port` with the session id as key) |

Guidelines:

- **Append, do not rewrite.** Keep sending the assistant messages (output items, content blocks)
  back exactly as they came. Then the whole episode is one token-exact training sequence.
  Rewriting history is allowed (it starts a new thread) but the earlier calls then train as
  separate, shorter sequences.
- **Let the trainer sample.** Do not rely on `temperature`, `n`, `response_format` or forced
  tool choices.
- **Stream long generations.** A streamed request is kept alive for as long as the batch takes;
  one that is not streamed has to fit the client's timeout.
- **React to the context budget.** With `rollout.max_context_tokens` set, a request whose prompt
  no longer fits is refused the way the vendor refuses it (§1, Errors); frameworks that compact
  on that error do so here as well, and the compacted history starts a new thread.
- **Agent CLIs**: pass what they need through placeholders, e.g.
  `command: ["my-agent", "--model", "{model}", "--prompt-file", "{task_file}"]` and
  `env: {OPENAI_API_BASE: "{base_url}"}` for tools that read other variable names
  (`{anthropic_base_url}` for the Anthropic API). Remember that every request such a CLI makes
  is a policy turn (§5).
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

- **Three APIs, their core only.** No `/v1/completions`, embeddings, audio, files, batches,
  stored conversations or retrieval of earlier responses; no hosted/server tools (web search,
  code execution, computer use, MCP): function tools that the agent runs itself only.
- **`n = 1`**, no `logprobs` in responses, no constrained decoding (`response_format`,
  `text.format`, strict schemas, forced tool choice), no reasoning/thinking blocks, no assistant
  prefill on the Messages API.
- **Streaming is not incremental.** `RolloutEngine.generate` returns whole completions, so a
  stream opens at once, is kept alive while the batch is generated, and then delivers the
  completion in one text delta (and one arguments delta per tool call). Only streams can outlive
  a client's idle timeout; a request that is not streamed has to be answered within it.
- **Images**: user messages only, not together with tools, vision-language policies only; once
  an episode has sent images, every later call must carry images too (§1, Images). No audio,
  video or documents.
- **Every request is a policy turn**, also the auxiliary calls some agent CLIs make (titles,
  summaries, connectivity probes): they are sampled from the policy, recorded and trained on.
- **The session id is the only credential**, and the gateway speaks plain HTTP. Bind it beyond
  loopback only on networks you trust (§1, Reachability).
- Tool calls are recognised in the syntaxes `parse_tool_calls` knows; the model has to be
  prompted (by its template or the system prompt) to use one of them.
- One call is one generation: a timed-out agent's generation in flight still finishes on the
  engine and is discarded.
