"""Fixtures for the Go / C++ SDK conformance tests.

Binaries are built once per session with the real Makefiles (``make -C envs go|cpp``); tests
skip when the toolchain is missing. Clients here use only ``monorl.envs.protocol`` helpers plus
``subprocess``/``httpx`` so the SDKs are exercised exactly like the orchestrator would.
"""

from __future__ import annotations

import fcntl
import json
import shutil
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from monorl.core.types import Completion, Sample
from monorl.envs import protocol

REPO = Path(__file__).resolve().parents[2]
ENVS = REPO / "envs"
BIN = ENVS / "bin"

GO_EXAMPLES = ("arith", "wordle", "gridworld")
CPP_EXAMPLES = ("arith", "number_guess")


@dataclass(frozen=True)
class Example:
    lang: str
    name: str

    @property
    def binary(self) -> Path:
        return BIN / f"{self.lang}-{self.name}"

    def __str__(self) -> str:
        return f"{self.lang}-{self.name}"


EXAMPLES = [Example("go", n) for n in GO_EXAMPLES] + [Example("cpp", n) for n in CPP_EXAMPLES]

_TOOLCHAIN = {"go": "go", "cpp": "clang++"}


def _build(lang: str) -> None:
    """Runs ``make -C envs <lang>``; a file lock keeps pytest-xdist workers from building concurrently."""
    BIN.mkdir(parents=True, exist_ok=True)
    with open(BIN / ".build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        proc = subprocess.run(
            ["make", "-C", str(ENVS), lang], capture_output=True, text=True, timeout=600, check=False
        )
    if proc.returncode != 0:
        raise RuntimeError(f"make -C envs {lang} failed:\n{proc.stdout}\n{proc.stderr}")


@pytest.fixture(scope="session")
def built(request: pytest.FixtureRequest) -> dict[str, bool]:
    """Builds each language's examples once; maps lang -> built?"""
    out: dict[str, bool] = {}
    for lang, tool in _TOOLCHAIN.items():
        if shutil.which(tool) is None or shutil.which("make") is None:
            out[lang] = False
            continue
        _build(lang)
        out[lang] = True
    return out


@pytest.fixture(params=EXAMPLES, ids=str)
def example(request: pytest.FixtureRequest, built: dict[str, bool]) -> Example:
    ex: Example = request.param
    if not built[ex.lang]:
        pytest.skip(f"{_TOOLCHAIN[ex.lang]} not available; cannot build {ex}")
    assert ex.binary.exists(), f"{ex.binary} missing after build"
    return ex


def make_sample(
    sid: str = "s1",
    question: str = "What is 2+2?",
    reference: Any = "4",
    metadata: dict[str, Any] | None = None,
) -> Sample:
    return Sample(
        id=sid, messages=[{"role": "user", "content": question}], reference=reference, metadata=metadata or {}
    )


def completion(text: str) -> Completion:
    return Completion(text=text, token_ids=[], finish_reason="stop")


class StdioClient:
    """Drives a MEP/1 server over stdio JSON lines, one request at a time."""

    def __init__(self, binary: Path) -> None:
        self.proc = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=REPO,
        )
        self._lock = threading.Lock()

    def raw(self, line: str) -> dict[str, Any]:
        assert self.proc.stdin and self.proc.stdout
        with self._lock:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            out = self.proc.stdout.readline()
        if not out:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"server closed stdout (exit={self.proc.poll()}): {err}")
        return json.loads(out)

    def request(self, req: dict[str, Any]) -> dict[str, Any]:
        return protocol.parse_response(self.raw(json.dumps(req)))

    def spec(self) -> protocol.EnvSpec:
        return protocol.EnvSpec.from_dict(self.request(protocol.spec_request()))

    def health(self) -> dict[str, Any]:
        return self.request(protocol.health_request())

    def reset(self, episode_id: str, sample: Sample, seed: int | None = None):
        return protocol.validate_reset_response(
            self.request(protocol.reset_request(episode_id, sample, seed))
        )

    def step(self, episode_id: str, text: str):
        return protocol.validate_step_response(
            self.request(protocol.step_request(episode_id, completion(text)))
        )

    def close(self, episode_id: str) -> dict[str, Any]:
        return self.request(protocol.close_request(episode_id))

    def shutdown(self) -> int:
        assert self.proc.stdin
        self.proc.stdin.close()
        try:
            return self.proc.wait(timeout=10)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()


class HttpClient:
    """Drives a MEP/1 server over http; safe to share across threads (httpx client is)."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.http = httpx.Client(base_url=base_url, timeout=30)

    def _post(self, op: str, body: dict[str, Any] | None) -> dict[str, Any]:
        r = self.http.post(f"/{op}", json=body)
        payload = r.json()
        if r.status_code != 200:
            assert "error" in payload, f"{op}: status {r.status_code} without error envelope: {payload}"
        return protocol.parse_response(payload)

    def spec(self) -> protocol.EnvSpec:
        return protocol.EnvSpec.from_dict(protocol.parse_response(self.http.get("/spec").json()))

    def health(self) -> dict[str, Any]:
        return protocol.parse_response(self.http.get("/health").json())

    def reset(self, episode_id: str, sample: Sample, seed: int | None = None):
        req = protocol.reset_request(episode_id, sample, seed)
        return protocol.validate_reset_response(self._post("reset", req))

    def step(self, episode_id: str, text: str):
        return protocol.validate_step_response(
            self._post("step", protocol.step_request(episode_id, completion(text)))
        )

    def close(self, episode_id: str) -> dict[str, Any]:
        return self._post("close", protocol.close_request(episode_id))


@pytest.fixture
def stdio(example: Example) -> Iterator[StdioClient]:
    client = StdioClient(example.binary)
    try:
        yield client
    finally:
        client.shutdown()


@pytest.fixture
def http_server(example: Example) -> Iterator[tuple[HttpClient, subprocess.Popen[str]]]:
    proc = subprocess.Popen(
        [str(example.binary), "--transport", "http", "--addr", "127.0.0.1:0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO,
    )
    assert proc.stderr
    url = None
    for line in proc.stderr:
        if "listening on" in line:
            url = line.rsplit("listening on", 1)[1].strip()
            break
    if url is None:
        proc.kill()
        raise RuntimeError(f"{example} never announced its http address (exit={proc.wait()})")
    client = HttpClient(url)
    try:
        yield client, proc
    finally:
        client.http.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
