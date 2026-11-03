#!/usr/bin/env bash
# Bootstraps a monorl development checkout:
#   1. .venv (Python 3.12) via uv with the CPU-friendly extras
#   2. Go + C++ example MEP/1 environments → envs/bin
#   3. web UI → web/dist (served by `monorl serve`)
#
# Usage: scripts/dev/bootstrap.sh [--check] [--no-envs] [--no-web] [--extras LIST] [--python VERSION]
#   --check     print every command that would run, without running anything
#   --no-envs   skip the Go/C++ environment build
#   --no-web    skip the web UI build
#   --extras    comma-separated pyproject extras (default: trl,distributed,dev,openenv,hermes,rewards)
#   --python    interpreter version for uv venv (default: 3.12)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CHECK=0
BUILD_ENVS=1
BUILD_WEB=1
EXTRAS="trl,distributed,dev,openenv,hermes,rewards"
PYTHON_VERSION="3.12"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK=1 ;;
        --no-envs) BUILD_ENVS=0 ;;
        --no-web) BUILD_WEB=0 ;;
        --extras) EXTRAS="$2"; shift ;;
        --extras=*) EXTRAS="${1#*=}" ;;
        --python) PYTHON_VERSION="$2"; shift ;;
        --python=*) PYTHON_VERSION="${1#*=}" ;;
        -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "bootstrap: unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
    shift
done

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
# run: echo the command; execute it unless --check.
run() {
    printf '    $ %s\n' "$*"
    if [[ ${CHECK} -eq 0 ]]; then "$@"; fi
}
have() { command -v "$1" >/dev/null 2>&1; }

if [[ ${CHECK} -eq 1 ]]; then
    log "check mode: printing the plan for ${REPO_ROOT}, nothing will be executed"
fi

# --- 1. Python environment --------------------------------------------------------------------
log "Python: .venv (${PYTHON_VERSION}) with extras [${EXTRAS}]"
if ! have uv; then
    if [[ ${CHECK} -eq 1 ]]; then
        warn "uv not found; it would be installed from https://astral.sh/uv/install.sh"
        printf '    $ %s\n' "curl -LsSf https://astral.sh/uv/install.sh | sh"
    else
        log "installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${HOME}/.local/bin:${PATH}"
        have uv || { echo "bootstrap: uv still not on PATH after install" >&2; exit 1; }
    fi
fi
if [[ -x .venv/bin/python ]]; then
    log ".venv already exists; reusing it"
else
    run uv venv .venv --python "${PYTHON_VERSION}"
fi
run uv pip install --python .venv/bin/python -e ".[${EXTRAS}]"
run .venv/bin/python -c 'import monorl, torch, transformers; print("monorl ok · torch", torch.__version__, "· transformers", transformers.__version__)'

# --- 2. Go / C++ example environments -----------------------------------------------------------
if [[ ${BUILD_ENVS} -eq 1 ]]; then
    log "MEP/1 example environments → envs/bin"
    if have go; then
        run make -C envs go
    else
        warn "go not found; skipping Go examples (brew install go / https://go.dev/dl)"
    fi
    if have clang++ || have g++; then
        run make -C envs cpp
    else
        warn "no C++ compiler (clang++/g++) found; skipping C++ examples"
    fi
else
    log "skipping environment build (--no-envs)"
fi

# --- 3. Web UI --------------------------------------------------------------------------------
if [[ ${BUILD_WEB} -eq 1 ]]; then
    log "web UI → web/dist"
    if ! have pnpm && have corepack; then
        run corepack enable
    fi
    if have pnpm; then
        run make web
    else
        warn "pnpm not found; skipping web build (npm i -g pnpm or corepack enable)"
    fi
else
    log "skipping web build (--no-web)"
fi

log "done. Next: .venv/bin/monorl doctor · .venv/bin/monorl serve · scripts/dev/smoke.sh"
