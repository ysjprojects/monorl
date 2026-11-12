"""``code_exec``: extract a fenced code block from the response and run it against test cases.

Each test case runs in a fresh ``python -I`` subprocess (own process group, temp cwd, minimal
environment) with a wall-clock timeout and a memory limit. The limit is applied twice: inside
the child via ``resource.setrlimit`` (``RLIMIT_AS``/``RLIMIT_DATA``, enforced by Linux) and from
the parent by polling the child's RSS with ``psutil`` (works on macOS, where the kernel rejects
address-space limits).

Params:
    mode: ``fraction`` (default; reward = passed / total) or ``all`` (1.0 only if every case passes).
    timeout_s: per-case wall clock (default 5.0).   memory_mb: per-case memory cap (default 512).
    language: fence tag to look for (default ``python``; untagged fences also count).
    reference_key: where the tests live (default: ``sample.reference`` itself, else ``reference["tests"]``).
    max_workers: trajectories scored concurrently (default ``min(8, cpu_count)``).
    python: interpreter for the sandbox (default: the current one).
    scope: ``last`` / ``all`` response text.

Test case formats (``sample.reference``):
    * ``"assert solve(2) == 4"`` — a string of test code appended to the solution (HumanEval ``check(f)`` style).
    * ``[{"input": "1 2\\n", "output": "3\\n"}, ...]`` — stdin → expected stdout (whitespace-trimmed per line).
    * ``{"inputs": [...], "outputs": [...]}`` — APPS/LiveCodeBench style, zipped into stdin/stdout cases.
    * ``[{"test": "assert ..."}, "assert ...", ...]`` — mixed lists; ``{"tests": [...]}`` wraps a list.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import psutil

from monorl.core.types import RewardResult, Trajectory
from monorl.rewards.base import (
    check_trajectories,
    extract_code_block,
    reference_value,
    response_text,
    reward_name,
)

# Runs inside the child: apply the rlimit when the platform allows it, then execute the file as
# ``__main__``. Kept as a -c string so the sandbox never needs monorl importable.
_RUNNER = """
import resource, runpy, sys
mb = int(sys.argv[1])
if mb > 0:
    limit = mb * 1024 * 1024
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        rl = getattr(resource, name, None)
        if rl is None:
            continue
        try:
            resource.setrlimit(rl, (limit, limit))
            break
        except (ValueError, OSError):
            pass
path = sys.argv[2]
sys.argv = [path]
sys.setrecursionlimit(10000)
runpy.run_path(path, run_name="__main__")
"""


@dataclass(slots=True)
class CodeCase:
    stdin: str | None = None
    expected_stdout: str | None = None
    test_code: str | None = None


@dataclass(slots=True)
class CaseResult:
    passed: bool
    status: str  # passed | failed | timeout | memory | error | no_code
    detail: str = ""


def normalize_tests(reference: Any) -> list[CodeCase]:
    """Coerce any supported reference layout into a list of :class:`CodeCase`."""
    if reference is None:
        return []
    if isinstance(reference, str):
        return [CodeCase(test_code=reference)]
    if isinstance(reference, dict):
        if "tests" in reference:
            return normalize_tests(reference["tests"])
        if "inputs" in reference and "outputs" in reference:
            ins, outs = reference["inputs"], reference["outputs"]
            if len(ins) != len(outs):
                raise ValueError("code_exec: inputs and outputs have different lengths")
            return [CodeCase(stdin=str(i), expected_stdout=str(o)) for i, o in zip(ins, outs, strict=True)]
        if "input" in reference or "output" in reference:
            return [
                CodeCase(
                    stdin=str(reference.get("input", "")), expected_stdout=str(reference.get("output", ""))
                )
            ]
        for key in ("test", "code", "assert"):
            if key in reference:
                return [CodeCase(test_code=str(reference[key]))]
        raise ValueError(f"code_exec: unrecognised test case dict with keys {sorted(reference)}")
    if isinstance(reference, (list, tuple)):
        cases: list[CodeCase] = []
        for item in reference:
            cases.extend(normalize_tests(item))
        return cases
    raise ValueError(f"code_exec: unsupported reference type {type(reference).__name__}")


def _outputs_match(actual: str, expected: str) -> bool:
    a = [line.rstrip() for line in actual.strip().splitlines()]
    e = [line.rstrip() for line in expected.strip().splitlines()]
    return a == e


class Sandbox:
    """Runs one program per call in an isolated subprocess with time and memory limits."""

    def __init__(self, *, timeout_s: float, memory_mb: int, python: str | None = None) -> None:
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.python = python or sys.executable
        self.memory_bytes = memory_mb * 1024 * 1024 if memory_mb > 0 else None

    def run(self, program: str, stdin: str | None = None) -> tuple[str, int | None, str]:
        """Execute ``program``; returns ``(status, returncode, stdout)``. ``status`` is one of
        ``ok``, ``timeout``, ``memory``, ``error`` (non-zero exit, stderr tail in stdout slot)."""
        workdir = tempfile.mkdtemp(prefix="monorl-exec-")
        try:
            path = os.path.join(workdir, "main.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(program)
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": workdir,
                "TMPDIR": workdir,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONHASHSEED": "0",
            }
            with (
                open(os.path.join(workdir, "stdin"), "w+", encoding="utf-8") as fin,
                open(os.path.join(workdir, "stdout"), "w+b") as fout,
                open(os.path.join(workdir, "stderr"), "w+b") as ferr,
            ):
                fin.write(stdin or "")
                fin.flush()
                fin.seek(0)
                proc = subprocess.Popen(
                    [self.python, "-I", "-c", _RUNNER, str(self.memory_mb), path],
                    cwd=workdir,
                    env=env,
                    stdin=fin,
                    stdout=fout,
                    stderr=ferr,
                    start_new_session=True,
                )
                status = self._wait(proc)
                fout.seek(0)
                ferr.seek(0)
                out = fout.read().decode("utf-8", errors="replace")
                err = ferr.read().decode("utf-8", errors="replace")
            if status != "ok":
                return status, proc.returncode, err[-2000:]
            if proc.returncode != 0:
                return "error", proc.returncode, err[-2000:]
            return "ok", 0, out
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _wait(self, proc: subprocess.Popen[bytes]) -> str:
        deadline = time.monotonic() + self.timeout_s
        ps = None
        if self.memory_bytes is not None:
            try:
                ps = psutil.Process(proc.pid)
            except psutil.Error:
                ps = None
        while True:
            try:
                proc.wait(timeout=0.02)
                return "ok"
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() >= deadline:
                self._kill(proc)
                return "timeout"
            if ps is not None:
                try:
                    rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
                except psutil.Error:
                    rss = 0
                if rss > self.memory_bytes:
                    self._kill(proc)
                    return "memory"

    @staticmethod
    def _kill(proc: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        except PermissionError:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


class CodeExecReward:
    def __init__(self, params: dict[str, Any]) -> None:
        self.name = reward_name(params, "code_exec")
        self.mode = str(params.get("mode", "fraction"))
        if self.mode not in ("fraction", "all"):
            raise ValueError(f"code_exec: mode must be 'fraction' or 'all', got {self.mode!r}")
        self.language = str(params.get("language", "python"))
        self.reference_key = params.get("reference_key")
        self.scope = str(params.get("scope", "last"))
        self.max_workers = int(params.get("max_workers", min(8, os.cpu_count() or 1)))
        self.sandbox = Sandbox(
            timeout_s=float(params.get("timeout_s", 5.0)),
            memory_mb=int(params.get("memory_mb", 512)),
            python=params.get("python"),
        )

    def run_case(self, code: str, case: CodeCase) -> CaseResult:
        program = code if case.test_code is None else f"{code}\n\n{case.test_code}\n"
        status, rc, out = self.sandbox.run(program, stdin=case.stdin)
        if status != "ok":
            return CaseResult(False, status, out)
        if case.expected_stdout is not None and not _outputs_match(out, case.expected_stdout):
            return CaseResult(False, "failed", out[-2000:])
        return CaseResult(True, "passed")

    def score_text(self, text: str, reference: Any) -> tuple[float, dict[str, Any]]:
        cases = normalize_tests(reference)
        code = extract_code_block(text, self.language)
        if code is None:
            return 0.0, {"status": "no_code", "passed": 0, "total": len(cases)}
        if not cases:
            # No tests: reward compiles-and-runs.
            result = self.run_case(code, CodeCase())
            return (1.0 if result.passed else 0.0), {
                "status": result.status,
                "passed": int(result.passed),
                "total": 1,
            }
        results = [self.run_case(code, case) for case in cases]
        passed = sum(r.passed for r in results)
        score = passed / len(results) if self.mode == "fraction" else float(passed == len(results))
        info = {
            "status": "passed" if passed == len(results) else "failed",
            "passed": passed,
            "total": len(results),
            "cases": [r.status for r in results],
        }
        return score, info

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        check_trajectories(trajectories)
        jobs = [(response_text(t, self.scope), self._tests_for(t)) for t in trajectories]
        if len(jobs) <= 1 or self.max_workers <= 1:
            scored = [self.score_text(text, ref) for text, ref in jobs]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                scored = list(pool.map(lambda job: self.score_text(*job), jobs))
        return [RewardResult(score=s, breakdown={self.name: s}, info=info) for s, info in scored]

    def _tests_for(self, trajectory: Trajectory) -> Any:
        if self.reference_key is not None:
            return reference_value(trajectory, self.reference_key)
        ref = trajectory.sample.reference
        if isinstance(ref, dict) and "tests" in ref:
            return ref["tests"]
        return ref


def make_code_exec(params: dict[str, Any], **_: Any) -> CodeExecReward:
    return CodeExecReward(params)
