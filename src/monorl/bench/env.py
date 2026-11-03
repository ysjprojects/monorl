"""MEP/1 environment throughput: episodes/s and step latency at several parallelism levels.

Drives any MEP/1 server with a canned policy (every action is the same completion) using the raw
:mod:`monorl.envs.protocol` encoders over JSON-lines subprocesses (``stdio``) or ``httpx``
(``http``). Each parallelism level ``P`` runs ``P`` threads; with stdio, ``ceil(P /
spec.parallel_episodes)`` server processes are spawned, mirroring what the orchestrator's
``remote`` env does, so the numbers size ``env.params.parallel`` / ``rollout.extra.max_parallel_episodes``.
Torch-free.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from monorl.bench.stats import HostInfo, LatencyStats, rate
from monorl.core.errors import EnvironmentError_
from monorl.core.types import Completion, Sample
from monorl.envs.protocol import (
    EnvSpec,
    ProtocolError,
    close_request,
    parse_response,
    reset_request,
    spec_request,
    step_request,
    validate_reset_response,
    validate_step_response,
)

DEFAULT_PARALLEL: tuple[int, ...] = (1, 4, 16)
CANNED_ACTION = Completion(text="The answer is 4.", token_ids=[], finish_reason="stop")

_EOF = object()


def canned_sample(i: int) -> Sample:
    return Sample(
        id=f"bench-{i}",
        messages=[{"role": "user", "content": f"Benchmark episode {i}: what is 2+2?"}],
        reference="4",
        metadata={"bench": True},
    )


class MepClient(Protocol):
    def request(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


# -- transports ------------------------------------------------------------------------------


class StdioClient:
    """One MEP/1 server process; one request in flight at a time (requests are serialised)."""

    def __init__(self, command: Sequence[str], *, cwd: str | None = None, timeout_s: float = 60.0) -> None:
        if not command:
            raise ValueError("stdio client needs a non-empty command")
        self.command = [str(c) for c in command]
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._lines: queue.Queue[Any] = queue.Queue()
        try:
            self.proc = subprocess.Popen(
                self.command,
                cwd=cwd,
                env=dict(os.environ),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            raise EnvironmentError_(f"cannot spawn env process {self.command!r}: {e}") from e
        threading.Thread(target=self._pump, daemon=True, name="bench-mep-stdout").start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(_EOF)

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.proc.poll() is not None:
                raise EnvironmentError_(f"env process exited with code {self.proc.returncode}")
            assert self.proc.stdin is not None
            try:
                self.proc.stdin.write(json.dumps(payload) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                raise EnvironmentError_(f"env process closed stdin: {e}") from e
            try:
                line = self._lines.get(timeout=self.timeout_s)
            except queue.Empty as e:
                raise EnvironmentError_(f"env process did not answer {payload.get('op')!r} within {self.timeout_s}s") from e
            if line is _EOF:
                raise EnvironmentError_(f"env process exited (code {self.proc.poll()}) while answering {payload.get('op')!r}")
        return parse_response(line)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait(timeout=5)


class HttpClient:
    """Shared ``httpx.Client``; safe to use from many threads."""

    def __init__(self, url: str, *, timeout_s: float = 60.0) -> None:
        import httpx

        self.url = url.rstrip("/")
        self._client = httpx.Client(base_url=self.url, timeout=float(timeout_s))

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        body = dict(payload)
        op = body.pop("op")
        try:
            resp = self._client.get(f"/{op}") if op in ("spec", "health") else self._client.post(f"/{op}", json=body)
        except httpx.HTTPError as e:
            raise EnvironmentError_(f"MEP http request {op} to {self.url} failed: {e}") from e
        try:
            out = resp.json()
        except ValueError as e:
            raise ProtocolError("protocol", f"non-JSON response (HTTP {resp.status_code}): {resp.text[:200]!r}") from e
        if resp.status_code >= 400 and not (isinstance(out, dict) and out.get("error")):
            raise ProtocolError("http", f"HTTP {resp.status_code}: {resp.text[:200]}")
        return parse_response(out)

    def close(self) -> None:
        self._client.close()


@dataclass(slots=True)
class EnvTarget:
    """What to benchmark: a stdio command or an http URL."""

    command: list[str] | None = None
    url: str | None = None
    cwd: str | None = None
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if bool(self.command) == bool(self.url):
            raise ValueError("pass exactly one of command or url")

    @property
    def transport(self) -> str:
        return "stdio" if self.command else "http"

    @property
    def label(self) -> str:
        return " ".join(self.command) if self.command else str(self.url)

    def open(self) -> MepClient:
        if self.command:
            return StdioClient(self.command, cwd=self.cwd, timeout_s=self.timeout_s)
        return HttpClient(str(self.url), timeout_s=self.timeout_s)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -- results ---------------------------------------------------------------------------------


@dataclass(slots=True)
class EnvParallelResult:
    parallel: int
    processes: int
    episodes: int
    steps: int
    seconds: float
    episodes_per_s: float
    steps_per_s: float
    reset_latency: LatencyStats
    step_latency: LatencyStats
    episode_latency: LatencyStats
    errors: int
    first_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EnvBenchmark:
    target: EnvTarget
    spec: dict[str, Any]
    turns: int
    results: list[EnvParallelResult]
    best_parallel: int | None
    recommendations: list[str]
    host: HostInfo
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "env",
            "target": self.target.to_dict(),
            "transport": self.target.transport,
            "spec": dict(self.spec),
            "turns": self.turns,
            "results": [r.to_dict() for r in self.results],
            "best_parallel": self.best_parallel,
            "recommendations": list(self.recommendations),
            "host": self.host.to_dict(),
            "extra": dict(self.extra),
        }


# -- driver ----------------------------------------------------------------------------------


@dataclass(slots=True)
class _Tally:
    resets: list[float] = field(default_factory=list)
    steps: list[float] = field(default_factory=list)
    episodes: list[float] = field(default_factory=list)
    done_episodes: int = 0
    errors: int = 0
    first_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def run_episode(client: MepClient, sample: Sample, *, turns: int, action: Completion = CANNED_ACTION) -> tuple[float, list[float]]:
    """reset → step×turns (stops early on ``done``) → close. Returns (reset latency, step latencies)."""
    episode_id = f"bench-{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()
    validate_reset_response(client.request(reset_request(episode_id, sample)))
    reset_s = time.perf_counter() - t0
    step_s: list[float] = []
    try:
        for _ in range(turns):
            t1 = time.perf_counter()
            _, _, done, _ = validate_step_response(client.request(step_request(episode_id, action)))
            step_s.append(time.perf_counter() - t1)
            if done:
                break
    finally:
        client.request(close_request(episode_id))
    return reset_s, step_s


def clients_for(target: EnvTarget, parallel: int, parallel_episodes: int) -> list[MepClient]:
    """stdio: ``ceil(parallel / parallel_episodes)`` processes; http: one shared client."""
    if target.transport == "http":
        return [target.open()]
    per_proc = max(1, int(parallel_episodes))
    return [target.open() for _ in range(-(-parallel // per_proc))]


def measure_parallel(
    target: EnvTarget,
    parallel: int,
    *,
    episodes: int,
    turns: int,
    parallel_episodes: int,
    action: Completion = CANNED_ACTION,
    sample_factory: Callable[[int], Sample] = canned_sample,
) -> EnvParallelResult:
    clients = clients_for(target, parallel, parallel_episodes)
    tally = _Tally()
    counter = iter(range(episodes))
    counter_lock = threading.Lock()

    def worker(idx: int) -> None:
        client = clients[idx % len(clients)]
        while True:
            with counter_lock:
                i = next(counter, None)
            if i is None:
                return
            t0 = time.perf_counter()
            try:
                reset_s, step_s = run_episode(client, sample_factory(i), turns=turns, action=action)
            except (EnvironmentError_, ProtocolError, OSError) as e:
                with tally.lock:
                    tally.errors += 1
                    if tally.first_error is None:
                        tally.first_error = f"{type(e).__name__}: {e}"
                continue
            dt = time.perf_counter() - t0
            with tally.lock:
                tally.resets.append(reset_s)
                tally.steps.extend(step_s)
                tally.episodes.append(dt)
                tally.done_episodes += 1

    t_start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="bench-env") as pool:
            list(pool.map(worker, range(parallel)))
    finally:
        for c in clients:
            c.close()
    seconds = time.perf_counter() - t_start
    return EnvParallelResult(
        parallel=parallel,
        processes=len(clients) if target.transport == "stdio" else 0,
        episodes=tally.done_episodes,
        steps=len(tally.steps),
        seconds=seconds,
        episodes_per_s=rate(tally.done_episodes, seconds),
        steps_per_s=rate(len(tally.steps), seconds),
        reset_latency=LatencyStats.from_samples(tally.resets),
        step_latency=LatencyStats.from_samples(tally.steps),
        episode_latency=LatencyStats.from_samples(tally.episodes),
        errors=tally.errors,
        first_error=tally.first_error,
    )


def recommend(results: Sequence[EnvParallelResult], spec: EnvSpec, transport: str) -> tuple[int | None, list[str]]:
    valid = [r for r in results if r.episodes > 0]
    if not valid:
        return None, ["no episode completed; check the server with `monorl env check`"]
    best = max(valid, key=lambda r: r.episodes_per_s)
    notes = [
        f"best {best.episodes_per_s:,.1f} episodes/s ({best.steps_per_s:,.1f} steps/s) at parallel={best.parallel}; "
        f"step p50 {best.step_latency.p50_s * 1e3:.1f} ms, p95 {best.step_latency.p95_s * 1e3:.1f} ms"
    ]
    if transport == "stdio":
        notes.append(
            f"stdio: that level used {best.processes} server process(es) at spec.parallel_episodes={spec.parallel_episodes}; "
            f"set env.params.parallel={spec.parallel_episodes} (episodes per process) and "
            f"rollout.extra.max_parallel_episodes={best.parallel}"
        )
    else:
        notes.append(f"http: set env.params.parallel={best.parallel} and rollout.extra.max_parallel_episodes>={best.parallel}")
    lowest = min(valid, key=lambda r: r.parallel)
    if best is not lowest:
        speedup = best.episodes_per_s / lowest.episodes_per_s if lowest.episodes_per_s else 0.0
        notes.append(f"{speedup:.1f}x over parallel={lowest.parallel}")
    highest = max(valid, key=lambda r: r.parallel)
    if highest is not best and highest.episodes_per_s < 0.9 * best.episodes_per_s:
        notes.append(
            f"throughput falls at parallel={highest.parallel} ({highest.episodes_per_s:,.1f} episodes/s): "
            "the server is saturated; more parallel episodes only add latency"
        )
    errors = sum(r.errors for r in results)
    if errors:
        first = next((r.first_error for r in results if r.first_error), None)
        notes.append(f"{errors} episode(s) failed; first error: {first}")
    return best.parallel, notes


def run_env_benchmark(
    target: EnvTarget,
    *,
    parallel_levels: Sequence[int] = DEFAULT_PARALLEL,
    episodes: int = 32,
    turns: int = 1,
    action: Completion = CANNED_ACTION,
    sample_factory: Callable[[int], Sample] = canned_sample,
    parallel_episodes: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> EnvBenchmark:
    """Benchmark ``target`` at each parallelism level. ``parallel_episodes`` overrides the server's
    ``spec.parallel_episodes`` when sizing stdio processes."""
    levels = sorted({int(p) for p in parallel_levels if int(p) >= 1})
    if not levels:
        raise ValueError("parallel_levels must contain at least one positive level")
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    probe = target.open()
    try:
        spec = EnvSpec.from_dict(probe.request(spec_request()))
    finally:
        probe.close()
    per_proc = parallel_episodes if parallel_episodes is not None else max(1, spec.parallel_episodes)
    eff_turns = max(1, min(turns, spec.max_turns)) if spec.max_turns else max(1, turns)
    results = [
        measure_parallel(
            target,
            p,
            episodes=episodes,
            turns=eff_turns,
            parallel_episodes=per_proc,
            action=action,
            sample_factory=sample_factory,
        )
        for p in levels
    ]
    best, notes = recommend(results, spec, target.transport)
    if eff_turns != turns:
        notes.append(f"turns clamped to spec.max_turns={spec.max_turns}")
    return EnvBenchmark(
        target=target,
        spec=spec.to_dict(),
        turns=eff_turns,
        results=results,
        best_parallel=best,
        recommendations=notes,
        host=HostInfo.detect(),
        extra=dict(extra or {}),
    )


__all__ = [
    "CANNED_ACTION",
    "DEFAULT_PARALLEL",
    "EnvBenchmark",
    "EnvParallelResult",
    "EnvTarget",
    "HttpClient",
    "MepClient",
    "StdioClient",
    "canned_sample",
    "clients_for",
    "measure_parallel",
    "recommend",
    "run_env_benchmark",
    "run_episode",
]
