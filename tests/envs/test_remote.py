from __future__ import annotations

import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from monorl.core.errors import EnvironmentError_
from monorl.core.interfaces import Environment
from monorl.core.types import Completion, Sample
from monorl.envs.remote import (
    HttpTransport,
    RemoteEnv,
    StdioPool,
    StdioTransport,
    shutdown_pools,
    slot_source_for,
)
from monorl.envs.server import MEPServer, start_http_server
from monorl.envs.tool_calling import make_tool_calling_env

SERVER_SCRIPT = Path(__file__).with_name("mep_tool_server.py")
CALC = '<tool_call>{"name": "calculator", "arguments": {"expression": "%s"}}</tool_call>'


def _sample(i: str = "s") -> Sample:
    return Sample(id=i, messages=[{"role": "user", "content": "What is 2+2?"}], reference="4")


def _c(text: str) -> Completion:
    return Completion(text=text, token_ids=[])


@pytest.fixture(autouse=True)
def _clean_pools():
    shutdown_pools()
    yield
    shutdown_pools()


@pytest.fixture
def stdio_params() -> dict:
    return {"transport": "stdio", "command": [sys.executable, str(SERVER_SCRIPT), "2"], "timeout_s": 30}


@pytest.fixture(scope="module")
def http_server():
    handle = start_http_server(
        MEPServer(
            lambda: make_tool_calling_env({"tools": ["calculator"]}, max_turns=3),
            name="http_tools",
            parallel_episodes=8,
        ),
        port=0,
    )
    yield handle
    handle.stop()


@pytest.fixture(params=["stdio", "http"])
def params(request, stdio_params, http_server) -> dict:
    if request.param == "stdio":
        return stdio_params
    return {"transport": "http", "url": http_server.url, "timeout_s": 30}


# -- shared behaviour over both transports ---------------------------------------------------


def test_spec_and_episode(params):
    env = RemoteEnv.from_params(params)
    assert isinstance(env, Environment)
    assert env.spec.protocol == "MEP/1" and env.max_turns == 3
    assert env.tools and env.tools[0]["function"]["name"] == "calculator"

    msgs = env.reset(_sample())
    assert msgs[0]["role"] == "system" and msgs[-1]["content"] == "What is 2+2?"
    assert env.episode_id and env.episode_id.startswith("s-")

    r = env.step(_c(CALC % "2+2"))
    assert r.done is False and r.messages[0]["role"] == "tool" and r.messages[0]["content"] == "4"
    assert r.info["tool_calls"][0]["name"] == "calculator"
    r = env.step(_c("4"))
    assert r.done is True and r.messages == []
    assert env.episode_id is None  # slot released on done
    env.close()  # idempotent after done


def test_close_mid_episode_and_reuse(params):
    env = RemoteEnv.from_params(params)
    env.reset(_sample("a"))
    env.step(_c(CALC % "1+1"))
    env.close()
    assert env.episode_id is None
    with pytest.raises(EnvironmentError_, match="before reset"):
        env.step(_c("x"))
    msgs = env.reset(_sample("b"))
    assert msgs[-1]["content"] == "What is 2+2?"
    assert env.step(_c("done")).done


def test_max_turns_override_truncates(params):
    env = RemoteEnv.from_params(params, max_turns=1)
    assert env.max_turns == 1
    env.reset(_sample())
    r = env.step(_c(CALC % "3*3"))
    assert r.done is True and r.info.get("truncated") is True
    assert r.messages[0]["content"] == "9"
    assert env.episode_id is None


def test_four_concurrent_episodes(params):
    envs = [RemoteEnv.from_params(params) for _ in range(4)]
    barrier = threading.Barrier(4)

    def run(i: int) -> list[str]:
        env = envs[i]
        env.reset(_sample(f"ep{i}"))
        barrier.wait(timeout=20)
        out = []
        for k in range(2):
            r = env.step(_c(CALC % f"{i}*10+{k}"))
            out.append(r.messages[0]["content"])
            assert not r.done
        final = env.step(_c("done"))
        assert final.done
        return out

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, range(4)))
    assert results == [[str(i * 10), str(i * 10 + 1)] for i in range(4)]
    if params["transport"] == "stdio":
        pool_ = slot_source_for(params)
        assert isinstance(pool_, StdioPool)
        assert len(pool_.processes) == 2  # 4 episodes / parallel_episodes=2
        assert all(p.alive for p in pool_.processes)


# -- stdio specifics -------------------------------------------------------------------------


def test_stdio_crash_restarts_process(stdio_params, caplog):
    caplog.set_level(logging.INFO, logger="monorl.envs.remote")
    env = RemoteEnv.from_params(stdio_params)
    env.reset(_sample())
    transport = slot_source_for(stdio_params).processes[0]
    pid = transport.pid
    with pytest.raises(EnvironmentError_, match="restarted"):
        env.step(_c("CRASH"))
    assert env.episode_id is None
    assert transport.restarts == 1 and transport.alive and transport.pid != pid
    assert any("crashing on request" in rec.getMessage() for rec in caplog.records)  # stderr forwarded

    # a sibling episode that was live on the crashed process fails fast without a round trip
    other = RemoteEnv.from_params(stdio_params)
    # fresh episode works on the restarted process
    msgs = other.reset(_sample("after"))
    assert msgs[-1]["content"] == "What is 2+2?"
    assert other.step(_c(CALC % "5+5")).messages[0]["content"] == "10"
    other.close()
    assert len(slot_source_for(stdio_params).processes) == 1


def test_stdio_generation_check_detects_lost_episode(stdio_params):
    a = RemoteEnv.from_params(stdio_params)
    b = RemoteEnv.from_params(stdio_params)
    a.reset(_sample("a"))
    b.reset(_sample("b"))  # same process (parallel=2)
    with pytest.raises(EnvironmentError_):
        a.step(_c("CRASH"))
    with pytest.raises(EnvironmentError_, match="lost"):
        b.step(_c("1"))
    assert b.episode_id is None


def test_stdio_timeout_is_a_crash(tmp_path):
    script = tmp_path / "hang.py"
    script.write_text("import sys, time\nsys.stdin.readline()\ntime.sleep(30)\n")
    t = StdioTransport([sys.executable, str(script)], timeout_s=0.5, start_timeout_s=0.5)
    with pytest.raises(EnvironmentError_, match="no response"):
        t.spec()
    assert t.restarts == 1 and t.alive
    t.close()
    assert not t.alive


def test_stdio_bad_command():
    t = StdioTransport(["/definitely/not/a/binary"], timeout_s=1)
    with pytest.raises(EnvironmentError_, match="cannot spawn"):
        t.spec()


def test_stdio_parallel_override_spawns_per_episode(stdio_params):
    params = {**stdio_params, "parallel": 1}
    a, b = RemoteEnv.from_params(params), RemoteEnv.from_params(params)
    a.reset(_sample("a"))
    b.reset(_sample("b"))
    assert len(slot_source_for(params).processes) == 2
    a.close()
    b.close()


# -- http specifics --------------------------------------------------------------------------


def test_http_spec_cached_and_errors(http_server):
    t = HttpTransport(http_server.url)
    s1 = t.spec()
    assert s1.name == "http_tools" and t.spec() is s1
    assert t.health()
    with pytest.raises(Exception, match="unknown_episode"):
        t.request({"op": "step", "episode_id": "nope", "action": {"text": "x"}})
    t.close()


def test_http_unreachable():
    env_params = {"transport": "http", "url": "http://127.0.0.1:9", "timeout_s": 1}
    with pytest.raises(EnvironmentError_):
        RemoteEnv.from_params(env_params)


def test_bad_params():
    with pytest.raises(ValueError):
        slot_source_for({"transport": "carrier-pigeon"})
    with pytest.raises(ValueError):
        slot_source_for({"transport": "http"})
    with pytest.raises(ValueError):
        slot_source_for({"transport": "stdio"})
