"""monorl vLLM server: vLLM's OpenAI-compatible API server plus weight-update endpoints.

Run with the same arguments as ``vllm serve``::

    python -m monorl.adapters.vllm.server <model> --tensor-parallel-size 2 --enable-sleep-mode

What it adds on top of vLLM v0.33.0's launcher (``vllm/entrypoints/launchers/api_server/entry.py``):

* ``VLLM_SERVER_DEV_MODE=1`` so vLLM mounts its own ``/sleep``, ``/wake_up``, ``/is_sleeping``,
  ``/reset_prefix_cache``, ``/pause``, ``/resume`` (``vllm/entrypoints/serve/dev/*`` and
  ``vllm/entrypoints/rl/online/api_router.py``).
* ``--worker-extension-cls monorl.adapters.vllm.worker.MonorlWorkerExtension`` by default.
* The ``vllm.server.monorl_routes`` patch, which mounts :func:`router` on the app:

  - ``POST /update_named_param?name=&dtype=&shape=`` with the raw tensor bytes as body
    (:func:`monorl.adapters.vllm.common.tensor_to_bytes`) → ``collective_rpc("monorl_update_weights")``
  - ``POST /load_weights_from_path`` ``{"path": ...}`` → ``collective_rpc("monorl_load_weights_from_path")``
  - ``GET /monorl/info`` → per-worker device/dtype

:func:`router` / :func:`attach_router` import neither vLLM nor torch-heavy modules beyond
``torch`` itself, so the routes are unit-testable with a fake ``app.state.engine_client``
(``AsyncLLM.collective_rpc(method, timeout, args, kwargs)`` is awaited).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from monorl.adapters.vllm.common import WORKER_EXTENSION_CLS, tensor_from_bytes

router = APIRouter(tags=["monorl"])


def _engine_client(request: Request) -> Any:
    client = getattr(request.app.state, "engine_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="engine not initialised yet")
    return client


def _parse_shape(shape: str) -> list[int]:
    if shape.strip() == "":
        return []
    try:
        return [int(s) for s in shape.split(",")]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"bad shape {shape!r}: {e}") from e


@router.post("/update_named_param")
async def update_named_param(
    request: Request,
    name: str = Query(..., description="HF parameter name"),
    dtype: str = Query(..., description="torch dtype name, e.g. bfloat16"),
    shape: str = Query("", description="comma-separated dims; empty for a scalar"),
) -> JSONResponse:
    body = await request.body()
    try:
        tensor = tensor_from_bytes(body, dtype, _parse_shape(shape))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    results = await _engine_client(request).collective_rpc("monorl_update_weights", args=([(name, tensor)],))
    return JSONResponse({"name": name, "loaded": [int(r) for r in results]})


@router.post("/load_weights_from_path")
async def load_weights_from_path(request: Request) -> JSONResponse:
    body = await request.json()
    path = body.get("path") if isinstance(body, dict) else None
    if not path:
        raise HTTPException(status_code=400, detail="body must be {'path': <hf model dir>}")
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"{path} is not a directory on the server host")
    await _engine_client(request).collective_rpc("monorl_load_weights_from_path", args=(str(path),))
    return JSONResponse({"path": str(path), "status": "loaded"})


@router.get("/monorl/info")
async def info(request: Request) -> JSONResponse:
    results = await _engine_client(request).collective_rpc("monorl_device_info")
    return JSONResponse({"workers": list(results), "worker_extension_cls": WORKER_EXTENSION_CLS})


def attach_router(app: FastAPI) -> None:
    app.include_router(router)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point: ``python -m monorl.adapters.vllm.server <vllm serve args>``."""
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    from monorl.adapters.vllm.engine import import_vllm

    import_vllm()
    import uvloop
    from vllm.entrypoints.common.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.common.startup import cli_env_setup
    from vllm.entrypoints.launchers.api_server.entry import run_server
    from vllm.logger import configure_logging_from_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from monorl.patching import apply

    report = apply("vllm")
    if "vllm.server.monorl_routes" not in report.applied and not any(
        pid == "vllm.server.monorl_routes" and reason == "already applied" for pid, reason in report.skipped
    ):
        raise SystemExit(
            f"could not install monorl routes on the vLLM server: {report.failed or report.skipped}"
        )

    cli_env_setup()
    parser = FlexibleArgumentParser(description="monorl vLLM server (vllm serve + weight-update endpoints).")
    parser = make_arg_parser(parser)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if not getattr(args, "worker_extension_cls", None):
        args.worker_extension_cls = WORKER_EXTENSION_CLS
    configure_logging_from_args(args)
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
