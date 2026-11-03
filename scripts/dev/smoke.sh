#!/usr/bin/env bash
# End-to-end smoke test on CPU:
#   1. `monorl train` on the CPU GRPO recipe for a couple of steps (SmolLM2-135M, GSM8K subset)
#   2. `monorl serve` on a free port; curl the system / recipes / jobs endpoints and the UI index
#
# Usage: scripts/dev/smoke.sh [--steps N] [--skip-train] [--skip-serve] [--run-dir DIR]
# Env:   MONORL_PY (default .venv/bin/python), MONORL_SMOKE_RECIPE (default configs/recipes/grpo_gsm8k_smollm_cpu.yaml)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PY="${MONORL_PY:-.venv/bin/python}"
RECIPE="${MONORL_SMOKE_RECIPE:-configs/recipes/grpo_gsm8k_smollm_cpu.yaml}"
STEPS=2
DO_TRAIN=1
DO_SERVE=1
RUN_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift ;;
        --steps=*) STEPS="${1#*=}" ;;
        --skip-train) DO_TRAIN=0 ;;
        --skip-serve) DO_SERVE=0 ;;
        --run-dir) RUN_DIR="$2"; shift ;;
        --run-dir=*) RUN_DIR="${1#*=}" ;;
        -h|--help) sed -n '2,7p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "smoke: unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
    shift
done

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31mFAIL:\033[0m %s\n' "$*" >&2; exit 1; }

[[ -x "${PY}" ]] || fail "python not found at ${PY} (run scripts/dev/bootstrap.sh or set MONORL_PY)"
command -v curl >/dev/null || fail "curl is required"
[[ -f "${RECIPE}" ]] || fail "recipe not found: ${RECIPE}"

SERVER_PID=""
SERVER_LOG=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    [[ -n "${SERVER_LOG}" ]] && rm -f "${SERVER_LOG}"
}
trap cleanup EXIT

# --- 1. train -----------------------------------------------------------------------------------
if [[ ${DO_TRAIN} -eq 1 ]]; then
    if [[ -z "${RUN_DIR}" ]]; then
        RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/monorl-smoke.XXXXXX")"
    fi
    log "train: ${RECIPE} for ${STEPS} step(s) → ${RUN_DIR}"
    "${PY}" -m monorl.cli train --config "${RECIPE}" --run-dir "${RUN_DIR}" \
        --set "trainer.max_steps=${STEPS}" \
        --set "trainer.save_every=${STEPS}" \
        --set "data.limit=16" \
        --set "logging.trajectories_per_step=2"
    for f in config.yaml status.json metrics.jsonl; do
        [[ -s "${RUN_DIR}/${f}" ]] || fail "run dir is missing ${f}"
    done
    "${PY}" - "${RUN_DIR}" "${STEPS}" <<'EOF'
import json, sys
from pathlib import Path
run_dir, steps = Path(sys.argv[1]), int(sys.argv[2])
status = json.loads((run_dir / "status.json").read_text())
assert status.get("state") == "succeeded", f"unexpected final state: {status}"
rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
assert rows, "metrics.jsonl is empty"
assert max(r["step"] for r in rows) >= steps, f"expected step >= {steps}, got {[r['step'] for r in rows]}"
print(f"train ok: state={status['state']} metrics_rows={len(rows)} last_step={max(r['step'] for r in rows)}")
EOF
fi

# --- 2. serve + curl -----------------------------------------------------------------------------
if [[ ${DO_SERVE} -eq 1 ]]; then
    PORT="$("${PY}" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
    SERVER_LOG="$(mktemp "${TMPDIR:-/tmp}/monorl-smoke-server.XXXXXX")"
    BASE="http://127.0.0.1:${PORT}"
    log "serve: ${BASE} (log: ${SERVER_LOG})"
    "${PY}" -m monorl.cli serve --host 127.0.0.1 --port "${PORT}" --no-open >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    for _ in $(seq 1 60); do
        if curl -fsS "${BASE}/api/system" >/dev/null 2>&1; then break; fi
        kill -0 "${SERVER_PID}" 2>/dev/null || { cat "${SERVER_LOG}"; fail "server exited early"; }
        sleep 0.5
    done
    curl -fsS "${BASE}/api/system" >/dev/null || { cat "${SERVER_LOG}"; fail "server did not become ready on ${BASE}"; }

    check() {  # check <path> <python expression over the parsed JSON body>
        local body
        body="$(curl -fsS "${BASE}$1")" || fail "GET $1 failed"
        printf '%s' "${body}" | "${PY}" -c "import json,sys; d=json.load(sys.stdin); assert ($2), 'GET $1: unexpected body'; print('GET $1 ok')"
    }
    check /api/system "'python' in d and 'frameworks' in d"
    check /api/config/schema "d.get('type') == 'object' or 'properties' in d"
    check /api/recipes "isinstance(d, list) or isinstance(d, dict)"
    check /api/jobs "isinstance(d, list) or isinstance(d, dict)"
    if [[ -f web/dist/index.html ]]; then
        curl -fsS "${BASE}/" | grep -qi '<div id="root"' && echo "GET / ok (web UI index served)"
    else
        log "web/dist missing; skipping UI index check (run make web)"
    fi
fi

log "smoke ok"
