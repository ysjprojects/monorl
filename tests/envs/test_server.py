from __future__ import annotations

import io
import json

import httpx

from monorl.core.types import Sample
from monorl.envs.protocol import PROTOCOL_VERSION, encode_sample
from monorl.envs.server import MEPServer, create_app, serve_stdio, start_http_server
from monorl.envs.tool_calling import make_tool_calling_env


def _factory():
    return make_tool_calling_env({"tools": ["calculator"]}, max_turns=3)


def _sample(i: str = "s") -> dict:
    return encode_sample(Sample(id=i, messages=[{"role": "user", "content": "2+2?"}], reference="4"))


def test_handle_spec_health_and_episode_lifecycle():
    srv = MEPServer(_factory, name="tools", parallel_episodes=2)
    spec = srv.handle({"op": "spec"})
    assert spec["protocol"] == PROTOCOL_VERSION and spec["name"] == "tools"
    assert spec["max_turns"] == 3 and spec["parallel_episodes"] == 2
    assert spec["tools"][0]["function"]["name"] == "calculator"
    assert srv.handle({"op": "health"}) == {"ok": True}

    r = srv.handle({"op": "reset", "episode_id": "e1", "sample": _sample()})
    assert r["messages"][0]["role"] == "system" and r["info"]["episode_id"] == "e1"
    assert srv.active_episodes == 1

    s = srv.handle(
        {
            "op": "step",
            "episode_id": "e1",
            "action": {
                "text": '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>',
                "finish_reason": "stop",
            },
        }
    )
    assert s["done"] is False and s["messages"][0]["content"] == "4" and s["reward"] == 0.0
    s = srv.handle({"op": "step", "episode_id": "e1", "action": {"text": "4"}})
    assert s["done"] is True and s["messages"] == []
    assert srv.active_episodes == 0  # done episodes are closed server-side

    assert (
        srv.handle({"op": "step", "episode_id": "e1", "action": {"text": "x"}})["error"]["type"]
        == "unknown_episode"
    )
    srv.handle({"op": "reset", "episode_id": "e2", "sample": _sample()})
    assert srv.handle({"op": "close", "episode_id": "e2"}) == {}
    assert srv.active_episodes == 0
    assert srv.handle({"op": "close", "episode_id": "never"}) == {}


def test_handle_errors_and_capacity():
    srv = MEPServer(_factory, parallel_episodes=1)
    assert srv.handle({"op": "bogus"})["error"]["type"] == "protocol"
    assert srv.handle({"op": "reset"})["error"]["type"] == "protocol"
    assert srv.handle({"op": "reset", "episode_id": "a", "sample": {}})["error"]["type"] == "protocol"
    assert srv.handle({"op": "step", "episode_id": "a", "action": {}})["error"]["type"] == "protocol"
    srv.handle({"op": "reset", "episode_id": "a", "sample": _sample()})
    assert srv.handle({"op": "reset", "episode_id": "b", "sample": _sample()})["error"]["type"] == "capacity"
    # re-reset of an active episode replaces it without consuming capacity
    assert "messages" in srv.handle({"op": "reset", "episode_id": "a", "sample": _sample()})
    srv.close_all()
    assert srv.active_episodes == 0


def test_env_exception_becomes_error_envelope():
    class Boom:
        max_turns = 1

        def reset(self, sample):
            raise RuntimeError("kaboom")

        def step(self, action):
            raise AssertionError

        def close(self):
            pass

    srv = MEPServer(Boom)
    err = srv.handle({"op": "reset", "episode_id": "x", "sample": _sample()})["error"]
    assert err == {"type": "RuntimeError", "message": "kaboom"}


def test_serve_stdio_roundtrip():
    srv = MEPServer(_factory, parallel_episodes=4)
    requests = [
        {"op": "spec"},
        "not json",
        "",
        {"op": "reset", "episode_id": "e", "sample": _sample()},
        {"op": "step", "episode_id": "e", "action": {"text": "done"}},
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in requests) + "\n")
    stdout = io.StringIO()
    serve_stdio(srv, stdin=stdin, stdout=stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(lines) == 4  # blank line ignored
    assert lines[0]["protocol"] == PROTOCOL_VERSION
    assert lines[1]["error"]["type"] == "protocol"
    assert lines[2]["messages"][-1]["content"] == "2+2?"
    assert lines[3]["done"] is True


def test_http_app_routes_and_status_codes():
    from fastapi.testclient import TestClient

    app = create_app(MEPServer(_factory, parallel_episodes=1))
    with TestClient(app) as c:
        assert c.get("/spec").json()["protocol"] == PROTOCOL_VERSION
        assert c.get("/health").json() == {"ok": True}
        r = c.post("/reset", json={"episode_id": "e", "sample": _sample()})
        assert r.status_code == 200 and r.json()["messages"]
        assert c.post("/reset", json={"episode_id": "f", "sample": _sample()}).status_code == 503
        assert c.post("/step", json={"episode_id": "nope", "action": {"text": "x"}}).status_code == 400
        assert c.post("/bogus", json={}).status_code == 400
        assert c.post("/step", content=b"{bad").status_code == 400
        assert c.post("/close", json={"episode_id": "e"}).json() == {}


def test_start_http_server_thread():
    handle = start_http_server(MEPServer(_factory), port=0)
    try:
        assert handle.port != 0
        assert httpx.get(f"{handle.url}/health").json() == {"ok": True}
    finally:
        handle.stop()
