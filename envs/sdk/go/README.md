# monorl Go environment SDK (MEP/1)

Server side of the monorl environment protocol for environments written in Go. The module is
`github.com/monorl/envsdk` (`go.mod`: `go 1.22`), it has no dependencies outside the standard
library, and it is not published: it is used from this checkout. The protocol, the API and how a
run talks to a server (`env.type: remote`) are in [docs/ENV_SDK.md](../../../docs/ENV_SDK.md).

## Package `mep`

| File | Contents |
|---|---|
| `mep/types.go` | Wire types (`Message`, `Sample`, `Action`, `Spec`, `ResetResult`, `StepResult`) and the `Env` interface |
| `mep/errors.go` | `Error`, the error type names, `NewError`, `InvalidRequest`, `UnknownEpisode`, `EnvError` |
| `mep/handler.go` | `Handler`: dispatches requests to an `Env`, one lock per episode, idle reaper (`DefaultIdleTimeout`, 10 minutes) |
| `mep/stdio.go` | `ServeStdio`: the JSON-lines loop, one request and one response per line |
| `mep/http.go` | `NewHTTPHandler` (`/spec`, `/health`, `/reset`, `/step`, `/close`) and `ListenHTTP` |
| `mep/serve.go` | `Options`, `Serve`, the flags `--transport`, `--addr`, `--idle-timeout`, and `Main` |
| `mep/store.go` | `EpisodeStore[T]`: concurrency-safe map from episode id to state |
| `mep/mep_test.go` | Tests of the handler and of both transports |

## Examples

Each directory under `examples/` is one `package main` that implements `mep.Env` and ends in
`mep.Main(env)`:

- `examples/arith`: single turn; reward 1.0 when the last number of the answer equals
  `sample.reference`.
- `examples/wordle`: up to 6 turns of word guessing with feedback per letter.
- `examples/gridworld`: up to 12 turns of text navigation on a grid with walls.

## Build and test

In this directory:

```sh
go vet ./...
go test ./...
```

From the repository root, build every example into `envs/bin/go-<example>` (the directory is
ignored by git) and check one against the protocol:

```sh
make -C envs go
.venv/bin/monorl env check --command envs/bin/go-arith --turns 1
```

The check ends with `MEP/1 conformance: OK`.

## Writing your own environment

Start from a copy of `examples/arith/main.go`: change `Spec`, `Reset`, `Step` and `Close`.

In this module: put the file in a new directory `examples/<name>/`. `make -C envs go` builds
every directory under `examples/`, so the binary is `envs/bin/go-<name>`.

In your own module, outside the repository: require the SDK and point it at the checkout with a
`replace` directive (`/path/to/monorl` is the root of your checkout).

```sh
mkdir myenv && cd myenv
cp /path/to/monorl/envs/sdk/go/examples/arith/main.go .
go mod init example.com/myenv
go mod edit -require github.com/monorl/envsdk@v0.0.0 \
  -replace github.com/monorl/envsdk=/path/to/monorl/envs/sdk/go
go build -o myenv .
```

`go.mod` then holds `replace github.com/monorl/envsdk => /path/to/monorl/envs/sdk/go`. Check the
binary the same way, from the monorl checkout:
`.venv/bin/monorl env check --command /path/to/myenv/myenv --turns 1`.
