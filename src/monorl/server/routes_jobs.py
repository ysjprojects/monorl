"""Job CRUD + run-directory readers (metrics, logs, trajectories, checkpoints) with SSE tails."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from monorl.core.reporting import LOG_FILE, iter_metrics, iter_trajectories, list_checkpoints, read_status
from monorl.server.jobs import TERMINAL_STATES, JobManager

router = APIRouter()

POLL_INTERVAL = 0.5
MAX_ROWS = 5000


def _jobs(request: Request) -> JobManager:
    return request.app.state.jobs


def _rows(iterator: Any, since: int, limit: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    nxt = since
    for i, row in iterator:
        rows.append(row)
        nxt = i + 1
        if len(rows) >= limit:
            break
    return {"rows": rows, "next": nxt}


def read_log_lines(run_dir: str | os.PathLike[str], *, since: int = 0, tail: int | None = None) -> dict[str, Any]:
    """``{lines, next}`` where ``next`` is the line offset to pass as ``since`` next time."""
    p = Path(run_dir) / LOG_FILE
    if not p.exists():
        return {"lines": [], "next": 0}
    with open(p, errors="replace") as f:
        lines = f.read().splitlines()
    total = len(lines)
    if tail is not None:
        start = max(total - tail, 0)
        return {"lines": lines[start:], "next": total}
    return {"lines": lines[since:], "next": total}


# -- CRUD ------------------------------------------------------------------------------------


@router.post("/jobs", status_code=201)
def create_job(body: dict[str, Any], request: Request) -> dict[str, Any]:
    payload = body.get("config", body) if isinstance(body, dict) and "config" in body else body
    if not isinstance(payload, dict):
        return JSONResponse(  # type: ignore[return-value]
            {"ok": False, "errors": [{"loc": ["config"], "msg": "config must be an object", "type": "type_error"}]},
            status_code=422,
        )
    return _jobs(request).create(payload)


@router.get("/jobs")
def list_jobs(request: Request) -> list[dict[str, Any]]:
    return _jobs(request).list()


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict[str, Any]:
    job = _jobs(request).get(job_id)
    job["status"] = read_status(job["run_dir"])
    return job


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    return _jobs(request).cancel(job_id)


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str, request: Request, remove_files: bool = Query(False)) -> dict[str, Any]:
    _jobs(request).delete(job_id, remove_files=remove_files)
    return {"ok": True, "id": job_id}


# -- run directory readers -----------------------------------------------------------------


@router.get("/jobs/{job_id}/metrics")
def job_metrics(
    job_id: str, request: Request, since: int = Query(0, ge=0), limit: int = Query(MAX_ROWS, ge=1, le=MAX_ROWS)
) -> dict[str, Any]:
    job = _jobs(request).get(job_id)
    return _rows(iter_metrics(job["run_dir"], start_line=since), since, limit)


@router.get("/jobs/{job_id}/trajectories")
def job_trajectories(
    job_id: str, request: Request, since: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=MAX_ROWS)
) -> dict[str, Any]:
    job = _jobs(request).get(job_id)
    return _rows(iter_trajectories(job["run_dir"], start_line=since), since, limit)


@router.get("/jobs/{job_id}/logs")
def job_logs(
    job_id: str,
    request: Request,
    tail: int | None = Query(None, ge=0),
    since: int = Query(0, ge=0),
) -> dict[str, Any]:
    job = _jobs(request).get(job_id)
    return read_log_lines(job["run_dir"], since=since, tail=tail)


@router.get("/jobs/{job_id}/checkpoints")
def job_checkpoints(job_id: str, request: Request) -> dict[str, Any]:
    job = _jobs(request).get(job_id)
    paths = list_checkpoints(job["run_dir"])
    final = Path(job["run_dir"]) / "final"
    return {
        "checkpoints": [{"step": int(Path(p).name[5:]), "path": p} for p in paths],
        "final": str(final) if final.is_dir() else None,
    }


# -- SSE streams --------------------------------------------------------------------------------


async def _poll_stream(
    request: Request, job_id: str, *, since: int, read_new: Any, event: str
) -> AsyncIterator[dict[str, str]]:
    """Generic tail: ``read_new(cursor) -> (items, cursor)``; emits ``event`` per item, then
    ``end`` once the job is terminal and the file has been fully drained."""
    jobs = _jobs(request)
    cursor = since
    loop = asyncio.get_running_loop()
    while True:
        if await request.is_disconnected():
            return
        job = await loop.run_in_executor(None, jobs.get, job_id)
        terminal = job["state"] in TERMINAL_STATES
        items, cursor = await loop.run_in_executor(None, read_new, job["run_dir"], cursor)
        for item in items:
            yield {"event": event, "data": json.dumps(item)}
        if terminal:
            yield {"event": "end", "data": json.dumps({"state": job["state"], "next": cursor, "error": job.get("error")})}
            return
        await asyncio.sleep(POLL_INTERVAL)


def _new_metrics(run_dir: str, cursor: int) -> tuple[list[dict[str, Any]], int]:
    out = _rows(iter_metrics(run_dir, start_line=cursor), cursor, MAX_ROWS)
    return out["rows"], out["next"]


def _new_logs(run_dir: str, cursor: int) -> tuple[list[str], int]:
    out = read_log_lines(run_dir, since=cursor)
    return out["lines"], out["next"]


@router.get("/jobs/{job_id}/metrics/stream")
async def stream_metrics(job_id: str, request: Request, since: int = Query(0, ge=0)) -> EventSourceResponse:
    _jobs(request).get(job_id)  # 404 before the stream opens
    return EventSourceResponse(_poll_stream(request, job_id, since=since, read_new=_new_metrics, event="metrics"))


@router.get("/jobs/{job_id}/logs/stream")
async def stream_logs(job_id: str, request: Request, since: int = Query(0, ge=0)) -> EventSourceResponse:
    _jobs(request).get(job_id)
    return EventSourceResponse(_poll_stream(request, job_id, since=since, read_new=_new_logs, event="log"))
