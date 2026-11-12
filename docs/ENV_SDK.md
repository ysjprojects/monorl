# Environment SDKs (Go, C++) and the MEP/1 wire protocol

monorl trains policies against environments written in any language. An environment is a
server that speaks **MEP/1** (`src/monorl/envs/protocol.py` is the normative spec); the
orchestrator drives it through `env: {type: remote}` and runs many episodes in parallel. This
document covers the wire format, the Go and C++ SDKs under `envs/sdk/`, the shipped examples,
and how to point a run at them.

```
envs/
  Makefile                 make go | cpp | all | clean   → envs/bin/<lang>-<example>
  sdk/go/                  module github.com/monorl/envsdk, package mep (stdlib only)
    examples/{arith,wordle,gridworld}/main.go
  sdk/cpp/                 header-only include/mep/mep.hpp + third_party/nlohmann/json.hpp (v3.12.0)
    examples/{arith,number_guess}.cpp
tests/envs_sdk/            Python conformance tests (build + drive every example over both transports)
```

## 1. Wire format

Five operations. Every request is a JSON object; every response is a JSON object.

| op | request | response |
|---|---|---|
| `spec` | `{}` | `EnvSpec` (below) |
| `health` | `{}` | `{"ok": true}` |
| `reset` | `{"episode_id", "sample": Sample, "seed"?: int}` | `{"messages": [Message], "info": {}}` |
| `step` | `{"episode_id", "action": Action}` | `{"messages": [Message], "reward": number, "done": bool, "info": {}}` |
| `close` | `{"episode_id"}` | `{}` |

Shapes:

```
Sample  = {"id": str, "messages": [Message], "reference": any, "metadata": {}}
Message = {"role": "system"|"user"|"assistant"|"tool", "content": str | [Part],
           "tool_calls"?: [{"id", "name", "arguments"}], "tool_call_id"?: str, "name"?: str}
Part    = {"type": "text", "text": str}
        | {"type": "image", "image": str, "detail"?: str}      # also "audio"/"video"
Action  = {"text": str, "tool_calls"?: [{"id", "name", "arguments"}], "finish_reason": str}
EnvSpec = {"name": str, "version": str, "protocol": "MEP/1", "max_turns": int,
           "parallel_episodes": int, "tools": [...], "description": str, "extra": {}}
```

Semantics:

- `reset` returns the **whole initial conversation** the policy sees (system/user turns). For a
  single-turn verifier that is simply `sample.messages`. The orchestrator appends the
  assistant turn itself and then calls `step`.
- `step` returns the messages to append *after* the assistant turn (tool results, the next user
  turn) and `done`. The orchestrator stops after `done` or `spec.max_turns` assistant turns.
  `reward` is the per-turn reward; the orchestrator folds the turns into one trajectory
  return (`Σ gamma^k · reward_k`, `algorithm.gamma`) reported as `reward/env`.
- Episodes are identified by `episode_id`. Servers may host many concurrently
  (`spec.parallel_episodes` tells the client how many); the client never sends two requests
  for the same episode at once. `close` is idempotent. Calling `reset` on a live id replaces
  the episode.
- Errors: `{"error": {"type": str, "message": str}}`. Types used by the SDKs:
  `invalid_request` (400), `unknown_episode` (404), `unknown_op`/`not_found` (404),
  `method_not_allowed` (405), `env_error` (500 — the environment threw/returned an error).
  Lists/objects are never `null` on the wire (`messages: []`, `info: {}`, `tools: []`).
- Multimodal messages carry `content` as a list of parts. Media parts hold only a **reference**
  (`image`: absolute local path, `file://`/`s3://`/`http(s)://` URI or `data:image/...;base64,`
  URL), never raw bytes, so every wire message is plain JSON. The data layer materialises
  dataset images to `$MONORL_HOME/cache/media/<sha256>.<ext>` before they reach an env, so a
  local path is the common case; remote URIs appear only when the dataset referenced them that
  way. Text-only environments concatenate the `text` parts (`monorl.core.types.message_text`
  in Python) and ignore the rest; environments may return the same parts in their own
  `messages` (e.g. a screenshot after a `step`).

Transports:

- **stdio** — the orchestrator spawns the binary. One request per line on stdin, one response
  per line on stdout, **in request order** (there is no request id). stdout is reserved for the
  protocol: log to stderr only. The `op` field selects the operation. EOF on stdin closes every
  episode and exits 0.
- **http** — `GET /spec`, `GET /health` (POST also accepted), `POST /reset`, `POST /step`,
  `POST /close` with JSON bodies; HTTP status mirrors the error type. Different episodes are
  served concurrently from different connections.

Both SDKs serve both transports from one `Env` implementation and accept the same flags:

```
<binary> [--transport stdio|http] [--addr host:port] [--idle-timeout ...]
```

`--transport` defaults to `stdio`. `--addr` (http only) defaults to `127.0.0.1:8000`; `:0`
binds a free port and the server prints `... listening on http://127.0.0.1:<port>` on stderr.
`--idle-timeout` (Go: a duration such as `10m`; C++: seconds) closes episodes that received no
request for that long (default 10 minutes; a negative value such as `-1s` disables it in Go,
`0` or negative in C++).

## 2. Go SDK — `envs/sdk/go` (`github.com/monorl/envsdk/mep`)

```go
package main

import (
    "context"
    "github.com/monorl/envsdk/mep"
)

type state struct{ answer string }

type myEnv struct{ episodes *mep.EpisodeStore[*state] }

func (e *myEnv) Spec() mep.Spec {
    return mep.Spec{Name: "my-env", Version: "1", MaxTurns: 1, ParallelEpisodes: 64}
}

func (e *myEnv) Reset(ctx context.Context, id string, s mep.Sample, seed *int64) (mep.ResetResult, error) {
    e.episodes.Put(id, &state{answer: s.Reference.(string)})
    return mep.ResetResult{Messages: s.Messages}, nil
}

func (e *myEnv) Step(ctx context.Context, id string, a mep.Action) (mep.StepResult, error) {
    st, ok := e.episodes.Get(id)
    if !ok {
        return mep.StepResult{}, mep.UnknownEpisode(id)
    }
    reward := 0.0
    if a.Text == st.answer {
        reward = 1
    }
    return mep.StepResult{Reward: reward, Done: true}, nil
}

func (e *myEnv) Close(id string) error { e.episodes.Delete(id); return nil }

func main() { mep.Main(&myEnv{episodes: mep.NewEpisodeStore[*state]()}) }
```

API surface (`mep` package, stdlib only):

- Types: `Sample`, `Message`, `ToolCall`, `Action`, `Spec`, `ResetResult`, `StepResult`;
  `ProtocolVersion = "MEP/1"`.
- `Env` interface: `Spec()`, `Reset(ctx, episodeID, sample, seed *int64)`, `Step(ctx,
  episodeID, action)`, `Close(episodeID)`. Return `*mep.Error` (`mep.InvalidRequest(...)`,
  `mep.UnknownEpisode(id)`, `mep.NewError(type, status, msg)`) for typed errors; any other
  error becomes `env_error`.
- `EpisodeStore[T]` — mutex-guarded id → state map for your per-episode state.
- `Handler` — transport-agnostic dispatcher with the session table: one lock per episode,
  idle reaper, `CloseAll()` on shutdown.
- `ServeStdio(ctx, handler, in, out)`, `ListenHTTP(handler, addr)` (+ `Serve`, `Shutdown`,
  `URL()`), `NewHTTPHandler(handler)` (plain `http.Handler` if you want to mount it yourself).
- `Serve(env, Options{Transport, Addr, IdleTimeout, Logger, In, Out})`, `Flags(*flag.FlagSet)`,
  `ParseFlags(args)`, `Main(env)`.

Tests: `cd envs/sdk/go && go test ./...` (stdio loop, http routes/status codes, 16 concurrent
episodes, idle reaper, flags).

## 3. C++ SDK — `envs/sdk/cpp` (`include/mep/mep.hpp`, C++17, header-only)

```cpp
#include "mep/mep.hpp"

struct State { std::string answer; };

class MyEnv : public mep::Env {
public:
    mep::Spec spec() const override {
        mep::Spec s; s.name = "my-env"; s.max_turns = 1; s.parallel_episodes = 64; return s;
    }
    mep::ResetResult reset(const std::string& id, const mep::Sample& sample, std::optional<int64_t>) override {
        episodes_.put(id, State{sample.reference.get<std::string>()});
        mep::ResetResult r; r.messages = sample.messages; return r;
    }
    mep::StepResult step(const std::string& id, const mep::Action& a) override {
        auto st = episodes_.require(id);          // throws mep::unknown_episode
        mep::StepResult r; r.reward = a.text == st->answer ? 1.0 : 0.0; r.done = true; return r;
    }
    void close(const std::string& id) override { episodes_.erase(id); }
private:
    mep::EpisodeStore<State> episodes_;
};

int main(int argc, char** argv) { MyEnv env; return mep::run_main(argc, argv, env); }
```

API surface (`namespace mep`):

- Structs mirroring the Go ones (`Sample`, `Message`, `ToolCall`, `Action`, `Spec`,
  `ResetResult`, `StepResult`) with `to_json`/`from_json`; `mep::json` is `nlohmann::json`.
- `Env` abstract class; throw `mep::Error(type, message, status)` or the helpers
  `invalid_request`, `unknown_episode`, `env_error`. Other exceptions become `env_error`.
- `EpisodeStore<T>` (`put`, `get`, `require`, `erase`, `size`) — `std::mutex`-guarded.
- `Handler` — session manager (per-episode `std::mutex`, idle reaper thread).
- `serve_stdio(env, in, out, idle)`; `serve_http(env, host, port, idle)` /
  `HttpServer(env, host, port)` — minimal HTTP/1.1 on POSIX sockets: thread per connection,
  keep-alive, `Content-Length` bodies (`Expect: 100-continue` honoured, chunked requests
  rejected with 411), GET/POST routing with the status codes above; `SIGINT`/`SIGTERM` stop it.
- `run_main(argc, argv, env)` parses `--transport`, `--addr`, `--idle-timeout` and returns the
  exit code.

Build: `make -C envs cpp` (`clang++ -std=c++17 -O2 -pthread -Wall -Wextra`). The Makefile
detects macOS Command Line Tools installs whose toolchain libc++ include dir is incomplete and
falls back to the SDK's headers automatically.

## 4. Example environments

| binary | turns | task | reward |
|---|---|---|---|
| `envs/bin/go-arith`, `envs/bin/cpp-arith` | 1 | `sample.messages` is the question, `sample.reference` the answer | 1.0 when the last number in the reply equals the reference |
| `envs/bin/go-wordle` | ≤6 | hidden 5-letter word (`sample.reference` if it is one, else chosen from a list by `seed`/sample id); feedback `G`/`Y`/`_` per letter | 1 − 0.1·(extra turns) on solve, 0 otherwise |
| `envs/bin/go-gridworld` | ≤12 | 4×4 text grid (`A` agent, `G` goal, `#` walls; `metadata.size` 2–12); commands `move north\|south\|east\|west` | +1 on the goal, −0.05 per move, −0.05 per bump/unparseable command |
| `envs/bin/cpp-number_guess` | ≤8 | hidden integer in `[1, metadata.range or 100]` (`sample.reference` if in range, else seeded); replies higher/lower | 1 − 0.1·(extra turns) on success (min 0.1), 0 otherwise |

Try one by hand:

```
make -C envs all
printf '%s\n' '{"op":"spec"}' \
  '{"op":"reset","episode_id":"e1","sample":{"id":"s","messages":[],"reference":"crane","metadata":{}}}' \
  '{"op":"step","episode_id":"e1","action":{"text":"apple","finish_reason":"stop"}}' \
  '{"op":"close","episode_id":"e1"}' | envs/bin/go-wordle
monorl env check --command envs/bin/go-wordle            # CLI driver over spec/reset/step/close
envs/bin/cpp-number_guess --transport http --addr 127.0.0.1:8081 &
curl -s localhost:8081/spec; monorl env check --url http://127.0.0.1:8081
```

## 5. Using a server from a run

Stdio — the orchestrator spawns the binary (one process per `parallel_episodes` episodes, more
processes as needed):

```yaml
env:
  type: remote
  params:
    transport: stdio
    command: [envs/bin/go-wordle]
    cwd: .                 # optional; env: {VAR: value} and timeout_s: 60 are also accepted
rollout:
  group_size: 8
  max_new_tokens: 16
```

Http — you run the server(s), the orchestrator connects:

```yaml
env:
  type: remote
  params:
    transport: http
    url: http://127.0.0.1:8081
```

Spec, sampling, rewards and advantages work exactly as with Python envs: the env's per-turn
rewards become the `env` term of the trajectory reward (discounted by `algorithm.gamma`) and are
added to the score of any `rewards:` entries you configure.

### Scaling parallel episodes

The orchestrator runs up to `rollout.extra.max_parallel_episodes` (default 64) episodes at a
time and batches their generation calls. Three knobs determine how many env episodes can be in
flight:

1. **The server's `spec.parallel_episodes`.** Both SDKs serialise calls per episode and run
   different episodes concurrently; set it to the number of episodes one process can hold
   (the examples advertise 64). Over stdio the client keeps at most that many episodes on one
   process and spawns another process beyond it (`params.parallel` overrides the advertised
   value).
2. **http for the highest fan-out.** One http server handles every episode on separate
   connections/threads, so `parallel_episodes` can be large and the server can live on another
   machine. Start it with `--transport http --addr 0.0.0.0:8081` and point `params.url` at it.
   The client keeps at most `spec.parallel_episodes` episodes open per server (`params.parallel`
   overrides that) and queues further resets until one ends, since a server at capacity refuses
   new episodes. For several replicas of one environment list them in `params.urls`: an
   episode's state lives in one process, so each episode is pinned to one replica (the least
   loaded) for all of its requests — a single load-balanced or round-robin address in `url`
   would scatter them and fail with `unknown_episode`.
3. **`rollout.extra.max_parallel_episodes`** caps the orchestrator side; raise it together with
   `parallel_episodes` and the rollout engine's batch size.

```yaml
env:
  type: remote
  params: {transport: http, url: http://envhost:8081}
rollout:
  group_size: 16
  extra:
    max_parallel_episodes: 256
```

## 6. Conformance tests

`.venv/bin/python -m pytest tests/envs_sdk -q` builds all examples through the Makefiles
(skipped per language when `go` or `clang++` is missing) and, for every example, drives a
scripted episode over stdio and http using only `monorl.envs.protocol` helpers: spec/health,
protocol version, rewards/done/messages, interleaved episodes on one stdio stream, error
envelopes and HTTP status codes for bad input, four concurrent http episodes from threads, seed
determinism, and clean exit on stdin EOF.
