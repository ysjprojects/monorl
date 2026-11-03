"""MEP/1 conformance of the Go and C++ example servers over both transports."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

import pytest

from monorl.envs import protocol

from .conftest import Example, HttpClient, StdioClient, completion, make_sample

Client = StdioClient | HttpClient


# --- scripted episodes per example ---------------------------------------------------------


def play_arith(c: Client, ep: str) -> float:
    sample = make_sample(f"{ep}-s", "What is 12*3?", "36")
    msgs, info = c.reset(ep, sample)
    assert msgs == sample.messages
    assert info["reference"] == "36"
    msgs, reward, done, info = c.step(ep, "12*3 = 36.")
    assert done and msgs == [] and reward == 1.0 and info["correct"] is True
    assert c.close(ep) == {}
    # wrong answer scores 0; reference given as a number also works
    c.reset(ep, make_sample(f"{ep}-s2", "What is 1+1?", 2))
    _, reward, done, info = c.step(ep, "I think it is 3")
    assert done and reward == 0.0 and info["correct"] is False
    assert c.close(ep) == {}
    return 1.0


def play_wordle(c: Client, ep: str) -> float:
    msgs, info = c.reset(ep, make_sample(f"{ep}-s", "play", "crane"))
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert info["max_turns"] == 6
    msgs, reward, done, info = c.step(ep, "apple")
    assert not done and reward == 0.0
    assert info["feedback"] == "Y___G" and info["guess"] == "apple"
    assert len(msgs) == 1 and msgs[0]["role"] == "user" and "Y___G" in msgs[0]["content"]
    msgs, reward, done, info = c.step(ep, "My guess: CRANE")
    assert done and msgs == [] and info["solved"] is True
    assert abs(reward - 0.9) < 1e-9
    assert c.close(ep) == {}
    return reward


def _parse_grid(text: str) -> list[str]:
    rows = [line for line in text.splitlines() if line and set(line) <= set("AG#.")]
    assert rows and all(len(r) == len(rows) for r in rows), text
    return rows


def _shortest_path(grid: list[str]) -> list[str]:
    n = len(grid)
    start = next((r, c) for r in range(n) for c in range(n) if grid[r][c] == "A")
    goal = next((r, c) for r in range(n) for c in range(n) if grid[r][c] == "G")
    moves = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
    prev: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if cur == goal:
            break
        for name, (dr, dc) in moves.items():
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < n and 0 <= nxt[1] < n and grid[nxt[0]][nxt[1]] != "#" and nxt not in prev:
                prev[nxt] = (cur, name)
                queue.append(nxt)
    assert goal in prev, "goal unreachable"
    path: list[str] = []
    cur = goal
    while prev[cur] is not None:
        cur, name = prev[cur]  # type: ignore[misc]
        path.append(name)
    return path[::-1]


def play_gridworld(c: Client, ep: str) -> float:
    msgs, info = c.reset(ep, make_sample(f"{ep}-s", "go", None), seed=11)
    assert [m["role"] for m in msgs] == ["system", "user"]
    grid = _parse_grid(msgs[1]["content"])
    assert info["size"] == len(grid) == 4
    path = _shortest_path(grid)
    assert 1 <= len(path) <= 12
    # an unparseable command costs a step + bump penalty and does not move
    msgs, reward, done, info = c.step(ep, "jump")
    assert not done and reward == 0.0 and info["move"] is None
    assert "did not understand" in msgs[0]["content"]
    for i, move in enumerate(path):
        msgs, reward, done, info = c.step(ep, f"move {move}")
        assert info["move"] == move
        if i < len(path) - 1:
            assert not done and reward == 0.0 and _parse_grid(msgs[0]["content"])
    assert done and info["reached_goal"] is True and msgs == []
    expected = 1.0 - 0.05 * (len(path) + 1) - 0.05
    assert abs(reward - expected) < 1e-9, (reward, expected, path)
    assert c.close(ep) == {}
    return reward


def play_number_guess(c: Client, ep: str) -> float:
    msgs, info = c.reset(ep, make_sample(f"{ep}-s", "play", 42))
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert info["range"] == 100 and info["max_turns"] == 8
    msgs, reward, done, info = c.step(ep, "50")
    assert not done and reward == 0.0 and info["guess"] == 50
    assert msgs[0]["role"] == "user" and msgs[0]["content"].startswith("Lower")
    msgs, reward, done, info = c.step(ep, "I guess 42")
    assert done and msgs == [] and info["solved"] is True and info["target"] == 42
    assert abs(reward - 0.9) < 1e-9
    assert c.close(ep) == {}
    return reward


SCENARIOS: dict[str, Callable[[Client, str], float]] = {
    "arith": play_arith,
    "wordle": play_wordle,
    "gridworld": play_gridworld,
    "number_guess": play_number_guess,
}

EXPECTED_MAX_TURNS = {"arith": 1, "wordle": 6, "gridworld": 12, "number_guess": 8}


def reveal_hidden(c: Client, ep: str, name: str, seed: int) -> Any:
    """Returns the env's seeded hidden state (running the episode to exhaustion when needed)."""
    _, info = c.reset(ep, make_sample(f"{ep}-s", "play", None), seed=seed)
    if name == "gridworld":
        c.close(ep)
        return (info["agent"], info["goal"])
    text, key = {"wordle": ("zzzzz", "word"), "number_guess": ("0", "target")}[name]
    for _ in range(EXPECTED_MAX_TURNS[name]):
        _, _, done, info = c.step(ep, text)
    assert done
    c.close(ep)
    return info[key]


# --- stdio ----------------------------------------------------------------------------------


def test_stdio_spec_and_health(stdio: StdioClient, example: Example) -> None:
    spec = stdio.spec()
    assert spec.protocol == protocol.PROTOCOL_VERSION
    assert spec.name == example.name
    assert spec.max_turns == EXPECTED_MAX_TURNS[example.name]
    assert spec.parallel_episodes >= 1
    assert isinstance(spec.tools, list)
    assert stdio.health()["ok"] is True


def test_stdio_episode(stdio: StdioClient, example: Example) -> None:
    SCENARIOS[example.name](stdio, "ep-1")
    assert stdio.health()["ok"] is True


def test_stdio_interleaved_episodes(stdio: StdioClient, example: Example) -> None:
    """Two episodes on one stdio stream, requests interleaved, each keeps its own state."""
    play = SCENARIOS[example.name]
    if example.name == "arith":
        a, b = make_sample("a", "What is 2+3?", "5"), make_sample("b", "What is 9-1?", "8")
        stdio.reset("A", a)
        stdio.reset("B", b)
        assert stdio.step("B", "8")[1] == 1.0
        assert stdio.step("A", "8")[1] == 0.0
        stdio.close("A")
        stdio.close("B")
        return
    if example.name == "number_guess":
        stdio.reset("A", make_sample("a", "p", 10))
        stdio.reset("B", make_sample("b", "p", 90))
        assert stdio.step("A", "50")[0][0]["content"].startswith("Lower")
        assert stdio.step("B", "50")[0][0]["content"].startswith("Higher")
        assert stdio.step("B", "90")[2] is True
        assert stdio.step("A", "10")[2] is True
        stdio.close("A")
        stdio.close("B")
        return
    play(stdio, "A")
    play(stdio, "B")


def test_stdio_error_envelopes(stdio: StdioClient) -> None:
    with pytest.raises(protocol.ProtocolError) as e:
        stdio.step("ghost", "hello")
    assert e.value.type == "unknown_episode"
    with pytest.raises(protocol.ProtocolError) as e:
        protocol.parse_response(stdio.raw("this is not json"))
    assert e.value.type == "invalid_request"
    with pytest.raises(protocol.ProtocolError) as e:
        stdio.request({"op": "frobnicate"})
    assert e.value.type == "unknown_op"
    with pytest.raises(protocol.ProtocolError) as e:
        stdio.request({"op": "reset", "episode_id": "x"})
    assert e.value.type == "invalid_request"
    with pytest.raises(protocol.ProtocolError) as e:
        stdio.request({"op": "step", "episode_id": "x"})
    assert e.value.type == "invalid_request"
    with pytest.raises(protocol.ProtocolError):
        stdio.request({"op": "reset", "sample": {"id": "no-episode"}})
    # closing an unknown episode is idempotent, and the server is still healthy afterwards
    assert stdio.close("ghost") == {}
    assert stdio.health()["ok"] is True


def test_stdio_exit_on_eof(example: Example) -> None:
    client = StdioClient(example.binary)
    assert client.health()["ok"] is True
    assert client.shutdown() == 0


def test_seed_determinism(stdio: StdioClient, example: Example) -> None:
    if example.name == "arith":
        pytest.skip("arith has no hidden state")
    first = reveal_hidden(stdio, "d1", example.name, seed=1234)
    again = reveal_hidden(stdio, "d2", example.name, seed=1234)
    other = reveal_hidden(stdio, "d3", example.name, seed=4321)
    assert first == again
    assert first != other or example.name == "gridworld"  # grid layouts may collide; words/targets must not


# --- http -----------------------------------------------------------------------------------


def test_http_spec_health_and_statuses(http_server, example: Example) -> None:
    client, _ = http_server
    spec = client.spec()
    assert spec.protocol == protocol.PROTOCOL_VERSION and spec.name == example.name
    assert client.health()["ok"] is True
    h = client.http
    assert h.post("/spec").status_code == 200
    r = h.get("/reset")
    assert r.status_code == 405 and r.json()["error"]["type"] == "method_not_allowed"
    r = h.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found"
    r = h.post("/step", json={"episode_id": "ghost", "action": {"text": "x", "finish_reason": "stop"}})
    assert r.status_code == 404 and r.json()["error"]["type"] == "unknown_episode"
    r = h.post("/reset", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request"
    r = h.post("/reset", json={"episode_id": "x"})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request"
    r = h.post("/close", json={"episode_id": "ghost"})
    assert r.status_code == 200 and r.json() == {}
    assert r.headers["content-type"].startswith("application/json")


def test_http_episode(http_server, example: Example) -> None:
    client, _ = http_server
    SCENARIOS[example.name](client, "h-1")


def test_http_concurrent_episodes(http_server, example: Example) -> None:
    client, _ = http_server
    play = SCENARIOS[example.name]
    results: dict[str, float] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run(i: int) -> None:
        ep = f"par-{i}"
        try:
            # each thread uses its own connection to actually hit the server concurrently
            own = HttpClient(client.base_url)
            try:
                r = play(own, ep)
            finally:
                own.http.close()
            with lock:
                results[ep] = r
        except BaseException as e:  # noqa: BLE001 - collected and re-raised below
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert len(results) == 4
    assert client.health()["ok"] is True


def test_http_stderr_announces_url(http_server) -> None:
    client, proc = http_server
    assert client.base_url.startswith("http://127.0.0.1:")
    assert proc.poll() is None


def test_wire_shapes_match_python_helpers(stdio: StdioClient, example: Example) -> None:
    """Responses decode through the exact Python helpers the orchestrator uses, and the raw JSON
    carries list/object (never null) for messages/info/tools."""
    raw_spec = stdio.raw(json.dumps(protocol.spec_request()))
    assert raw_spec["tools"] == [] and isinstance(raw_spec["extra"], dict)
    sample = make_sample("w", "What is 1+1?", "2" if example.name == "arith" else 2)
    raw_reset = stdio.raw(json.dumps(protocol.reset_request("w", sample, seed=3)))
    assert isinstance(raw_reset["messages"], list) and isinstance(raw_reset["info"], dict)
    protocol.validate_reset_response(protocol.parse_response(raw_reset))
    raw_step = stdio.raw(json.dumps(protocol.step_request("w", completion("2"))))
    assert isinstance(raw_step["messages"], list) and isinstance(raw_step["info"], dict)
    assert isinstance(raw_step["reward"], (int, float)) and isinstance(raw_step["done"], bool)
    protocol.validate_step_response(protocol.parse_response(raw_step))
    assert stdio.raw(json.dumps(protocol.close_request("w"))) == {}
