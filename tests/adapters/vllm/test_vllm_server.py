from __future__ import annotations

from typing import Any

import httpx
import pytest
import torch
from fastapi import FastAPI

from monorl.adapters.vllm.common import WORKER_EXTENSION_CLS, tensor_to_bytes
from monorl.adapters.vllm.server import attach_router


class FakeEngineClient:
    """``AsyncLLM.collective_rpc(method, timeout, args, kwargs)`` (vllm/v1/engine/async_llm.py:1249)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def collective_rpc(self, method: str, timeout: float | None = None, args: tuple = (), kwargs: Any = None):
        self.calls.append({"method": method, "args": args, "kwargs": kwargs})
        if method == "monorl_update_weights":
            return [len(args[0])]
        if method == "monorl_device_info":
            return [{"device": "cuda:0", "dtype": "bfloat16"}]
        return [None]


@pytest.fixture
def app_client() -> tuple[httpx.AsyncClient, FakeEngineClient]:
    app = FastAPI()
    attach_router(app)
    engine = FakeEngineClient()
    app.state.engine_client = engine
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://server")
    return client, engine


async def test_update_named_param_rebuilds_tensor(app_client) -> None:
    client, engine = app_client
    t = torch.linspace(-1, 1, 6, dtype=torch.bfloat16).reshape(3, 2)
    r = await client.post(
        "/update_named_param",
        params={"name": "lm_head.weight", "dtype": "bfloat16", "shape": "3,2"},
        content=tensor_to_bytes(t),
    )
    assert r.status_code == 200 and r.json() == {"name": "lm_head.weight", "loaded": [1]}
    call = engine.calls[-1]
    assert call["method"] == "monorl_update_weights"
    (name, tensor), = call["args"][0]
    assert name == "lm_head.weight" and tensor.dtype == torch.bfloat16 and torch.equal(tensor, t)


async def test_update_named_param_scalar_and_bad_payload(app_client) -> None:
    client, engine = app_client
    r = await client.post("/update_named_param", params={"name": "s", "dtype": "float32", "shape": ""}, content=b"\x00" * 4)
    assert r.status_code == 200 and engine.calls[-1]["args"][0][0][1].shape == ()
    r = await client.post("/update_named_param", params={"name": "s", "dtype": "float32", "shape": "2"}, content=b"\x00" * 4)
    assert r.status_code == 400 and "expected 8" in r.json()["detail"]
    r = await client.post("/update_named_param", params={"name": "s", "dtype": "complex128", "shape": ""}, content=b"")
    assert r.status_code == 400
    r = await client.post("/update_named_param", params={"name": "s", "dtype": "float32", "shape": "a,b"}, content=b"")
    assert r.status_code == 400


async def test_load_weights_from_path(app_client, tmp_path) -> None:
    client, engine = app_client
    r = await client.post("/load_weights_from_path", json={"path": str(tmp_path)})
    assert r.status_code == 200
    assert engine.calls[-1] == {"method": "monorl_load_weights_from_path", "args": (str(tmp_path),), "kwargs": None}
    r = await client.post("/load_weights_from_path", json={"path": str(tmp_path / "missing")})
    assert r.status_code == 400
    r = await client.post("/load_weights_from_path", json={})
    assert r.status_code == 400


async def test_info(app_client) -> None:
    client, _ = app_client
    r = await client.get("/monorl/info")
    assert r.json() == {"workers": [{"device": "cuda:0", "dtype": "bfloat16"}], "worker_extension_cls": WORKER_EXTENSION_CLS}


async def test_engine_not_ready() -> None:
    app = FastAPI()
    attach_router(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://s") as c:
        r = await c.get("/monorl/info")
    assert r.status_code == 503
