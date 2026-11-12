from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from monorl.bench.env import (
    EnvTarget,
    StdioClient,
    canned_sample,
    clients_for,
    measure_parallel,
    run_env_benchmark,
    run_episode,
)
from monorl.core.errors import EnvironmentError_
from monorl.envs.protocol import EnvSpec, spec_request


def test_env_target_validation() -> None:
    with pytest.raises(ValueError):
        EnvTarget()
    with pytest.raises(ValueError):
        EnvTarget(command=["x"], url="http://h")
    assert EnvTarget(command=["a", "b"]).transport == "stdio"
    assert EnvTarget(url="http://h:1").transport == "http"
    assert EnvTarget(url="http://h:1").label == "http://h:1"


def test_stdio_client_episode_roundtrip(mep_command) -> None:
    client = StdioClient(mep_command, timeout_s=20)
    try:
        spec = EnvSpec.from_dict(client.request(spec_request()))
        assert spec.name == "bench-echo" and spec.parallel_episodes == 2 and spec.max_turns == 3
        reset_s, steps = run_episode(client, canned_sample(0), turns=5)
        assert reset_s > 0 and len(steps) == 3  # server reports done after 3 turns
    finally:
        client.close()
    assert client.proc.poll() is not None


def test_stdio_client_reports_dead_process() -> None:
    client = StdioClient([sys.executable, "-c", "import sys; sys.exit(0)"], timeout_s=5)
    with pytest.raises(EnvironmentError_):
        client.request(spec_request())
    client.close()


def test_clients_for_sizes_stdio_processes(mep_command) -> None:
    target = EnvTarget(command=mep_command)
    clients = clients_for(target, 4, 2)
    try:
        assert len(clients) == 2
    finally:
        for c in clients:
            c.close()
    clients = clients_for(target, 3, 2)
    try:
        assert len(clients) == 2
    finally:
        for c in clients:
            c.close()


def test_measure_parallel_runs_all_episodes(mep_command) -> None:
    target = EnvTarget(command=mep_command, timeout_s=20)
    r = measure_parallel(target, 4, episodes=8, turns=2, parallel_episodes=2)
    assert r.parallel == 4 and r.processes == 2
    assert r.episodes == 8 and r.steps == 16 and r.errors == 0
    assert r.step_latency.count == 16 and r.step_latency.p50_s >= 0.002
    assert r.reset_latency.count == 8 and r.episode_latency.count == 8
    assert r.episodes_per_s == pytest.approx(8 / r.seconds)
    assert r.steps_per_s == pytest.approx(16 / r.seconds)


def test_run_env_benchmark_stdio_parallel_1_and_4(mep_command) -> None:
    bench = run_env_benchmark(
        EnvTarget(command=mep_command, timeout_s=20), parallel_levels=[4, 1], episodes=12, turns=10
    )
    assert [r.parallel for r in bench.results] == [1, 4]
    assert bench.turns == 3  # clamped to spec.max_turns
    assert bench.spec["name"] == "bench-echo"
    for r in bench.results:
        assert r.episodes == 12 and r.steps == 36 and r.errors == 0
    assert bench.results[0].processes == 1 and bench.results[1].processes == 2
    # two processes serve 4 workers: wall time must drop versus the serial run
    assert bench.results[1].seconds < bench.results[0].seconds
    assert bench.best_parallel == 4
    assert any("max_parallel_episodes=4" in n for n in bench.recommendations)
    assert any("clamped" in n for n in bench.recommendations)
    d = bench.to_dict()
    assert d["kind"] == "env" and d["transport"] == "stdio" and d["target"]["command"] == mep_command
    json.loads(json.dumps(d))


def test_run_env_benchmark_counts_failures() -> None:
    bad = [
        sys.executable,
        "-c",
        'import sys; print(\'{"name":"x","parallel_episodes":1}\'); sys.stdout.flush(); sys.exit(0)',
    ]
    bench = run_env_benchmark(EnvTarget(command=bad, timeout_s=5), parallel_levels=[1], episodes=2)
    assert bench.results[0].episodes == 0 and bench.results[0].errors == 2
    assert bench.best_parallel is None
    assert bench.results[0].first_error and "EnvironmentError_" in bench.results[0].first_error


class _Handler(BaseHTTPRequestHandler):
    turns: dict[str, int] = {}
    lock = threading.Lock()

    def log_message(self, *a) -> None:
        pass

    def _send(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/spec":
            self._send({"name": "http-echo", "max_turns": 2, "parallel_episodes": 8})
        elif self.path == "/health":
            self._send({"ok": True})
        else:
            self._send({"error": {"type": "not_found", "message": self.path}}, 404)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}"))
        ep = body.get("episode_id")
        if self.path == "/reset":
            with self.lock:
                self.turns[ep] = 0
            self._send({"messages": body["sample"]["messages"], "info": {}})
        elif self.path == "/step":
            with self.lock:
                self.turns[ep] = self.turns.get(ep, 0) + 1
                t = self.turns[ep]
            self._send({"messages": [], "reward": 0.5, "done": t >= 2, "info": {}})
        elif self.path == "/close":
            with self.lock:
                self.turns.pop(ep, None)
            self._send({})
        else:
            self._send({"error": {"type": "bad_op", "message": self.path}}, 400)


@pytest.fixture
def http_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_run_env_benchmark_http(http_url) -> None:
    pytest.importorskip("httpx")
    bench = run_env_benchmark(
        EnvTarget(url=http_url, timeout_s=10), parallel_levels=[1, 4], episodes=6, turns=2
    )
    assert bench.spec["name"] == "http-echo"
    for r in bench.results:
        assert r.episodes == 6 and r.steps == 12 and r.errors == 0 and r.processes == 0
    assert any("http:" in n for n in bench.recommendations)
    assert bench.to_dict()["transport"] == "http"
