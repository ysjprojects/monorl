from __future__ import annotations

import time

import pytest

from monorl.core.registry import registry
from monorl.rewards.code_exec import CodeCase, Sandbox, normalize_tests

ADD = "```python\ndef add(a, b):\n    return a + b\n```"
STDIO = "```python\nimport sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n```"


def test_normalize_tests_formats():
    assert normalize_tests("assert add(1, 2) == 3") == [CodeCase(test_code="assert add(1, 2) == 3")]
    assert normalize_tests({"inputs": ["1 2"], "outputs": ["3"]}) == [
        CodeCase(stdin="1 2", expected_stdout="3")
    ]
    assert normalize_tests([{"input": "1 2\n", "output": "3\n"}, "assert True", {"test": "assert 1"}]) == [
        CodeCase(stdin="1 2\n", expected_stdout="3\n"),
        CodeCase(test_code="assert True"),
        CodeCase(test_code="assert 1"),
    ]
    assert normalize_tests({"tests": ["assert 1"]}) == [CodeCase(test_code="assert 1")]
    assert normalize_tests(None) == []
    with pytest.raises(ValueError):
        normalize_tests({"inputs": [1], "outputs": []})
    with pytest.raises(ValueError):
        normalize_tests({"weird": 1})


def test_assert_tests_fraction_and_all(traj):
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(1, 1) == 3"]
    frac = registry.get("reward", "code_exec")({"timeout_s": 10})
    [res] = frac([traj(f"Here you go:\n{ADD}\nEnjoy.", tests)])
    assert res.score == pytest.approx(2 / 3)
    assert res.info["passed"] == 2 and res.info["total"] == 3
    assert res.info["cases"] == ["passed", "passed", "error"]
    strict = registry.get("reward", "code_exec")({"mode": "all", "timeout_s": 10})
    assert strict([traj(ADD, tests)])[0].score == 0.0
    assert strict([traj(ADD, tests[:2])])[0].score == 1.0


def test_stdin_stdout_tests_and_parallel_trajectories(traj):
    fn = registry.get("reward", "code_exec")({"timeout_s": 10, "max_workers": 4})
    ref = {"inputs": ["1 2", "10 -4\n"], "outputs": ["3\n", "6"]}
    results = fn([traj(STDIO, ref), traj("```python\nprint('wrong')\n```", ref), traj("no code here", ref)])
    assert [r.score for r in results] == [1.0, 0.0, 0.0]
    assert results[2].info["status"] == "no_code"


def test_no_tests_rewards_runnable_code(traj):
    fn = registry.get("reward", "code_exec")({"timeout_s": 10})
    assert fn([traj("```python\nx = 1\n```", None)])[0].score == 1.0
    assert fn([traj("```python\nraise SystemExit(3)\n```", None)])[0].score == 0.0


def test_timeout_kills_infinite_loop(traj):
    fn = registry.get("reward", "code_exec")({"timeout_s": 1.0})
    start = time.monotonic()
    [res] = fn([traj("```python\nwhile True:\n    pass\n```", "assert True")])
    assert time.monotonic() - start < 8
    assert res.score == 0.0
    assert res.info["cases"] == ["timeout"]


def test_memory_limit_enforced():
    sandbox = Sandbox(timeout_s=10.0, memory_mb=64)
    status, _, _ = sandbox.run(
        "x = bytearray(400 * 1024 * 1024)\nfor i in range(0, len(x), 4096):\n    x[i] = 1\nprint('done')"
    )
    assert status in ("memory", "error")  # rlimit (Linux) → MemoryError exit; RSS watchdog (macOS) → killed
    ok_status, _, out = sandbox.run("print('fine')")
    assert (ok_status, out.strip()) == ("ok", "fine")


def test_sandbox_isolation():
    sandbox = Sandbox(timeout_s=10.0, memory_mb=256)
    status, _, out = sandbox.run(
        "import os, sys\nprint(os.getcwd() != os.path.expanduser('~'), 'MONORL_SECRET' in os.environ)"
    )
    assert status == "ok"
    assert out.strip() == "True False"
