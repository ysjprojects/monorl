#!/bin/sh
# Serves one MEP/1 example environment over HTTP.
#   ENV_NAME  binary under /opt/envs/bin (go-arith, go-wordle, go-gridworld, cpp-arith, cpp-number_guess)
#   ENV_ADDR  listen address (default 0.0.0.0:8080)
# Extra arguments are passed through to the server (e.g. --idle-timeout).
set -eu
: "${ENV_NAME:?ENV_NAME must name a binary in /opt/envs/bin}"
: "${ENV_ADDR:=0.0.0.0:8080}"
bin="/opt/envs/bin/${ENV_NAME}"
if [ ! -x "${bin}" ]; then
    echo "env-entrypoint: unknown ENV_NAME '${ENV_NAME}'; available:" >&2
    ls /opt/envs/bin >&2
    exit 2
fi
exec "${bin}" --transport http --addr "${ENV_ADDR}" "$@"
