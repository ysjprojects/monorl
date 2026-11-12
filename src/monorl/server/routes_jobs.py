"""Job CRUD + run-directory readers (metrics, logs, trajectories, checkpoints) with SSE tails,
for a job's run dir and for each stage run dir of a pipeline job."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from monorl.core.reporting import LOG_FILE, iter_metrics, iter_trajectories, list_checkpoints, read_status
from monorl.pipeline.stages import read_manifest
from monorl.server.jobs import TERMINAL_STATES, JobManager, JobNotFound

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


def read_log_lines(
    run_dir: str | os.PathLike[str], *, since: int = 0, tail: int | None = None
) -> dict[str, Any]:
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
    """``{config: RunConfig}`` or ``{pipeline: PipelineConfig}`` (bare dicts of either are accepted)."""
    if isinstance(body, dict) and "pipeline" in body:
        payload: Any = body["pipeline"]
        key = "pipeline"
    else:
        payload = body.get("config", body) if isinstance(body, dict) and "config" in body else body
        key = "config"
    if not isinstance(payload, dict):
        return JSONResponse(  # type: ignore[return-value]
            {
                "ok": False,
                "errors": [{"loc": [key], "msg": f"{key} must be an object", "type": "type_error"}],
            },
            status_code=422,
        )
    return _jobs(request).create({"pipeline": payload} if key == "pipeline" else payload)


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
    job_id: str,
    request: Request,
    since: int = Query(0, ge=0),
    limit: int = Query(MAX_ROWS, ge=1, le=MAX_ROWS),
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
    return _checkpoints(job["run_dir"])


# -- pipeline stages -------------------------------------------------------------------------


def _checkpoints(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    paths = list_checkpoints(run_dir)
    final = Path(run_dir) / "final"
    return {
        "checkpoints": [{"step": int(Path(p).name[5:]), "path": p} for p in paths],
        "final": str(final) if final.is_dir() else None,
    }


def _stages(job: dict[str, Any]) -> dict[str, Any]:
    manifest = read_manifest(job["run_dir"]) or {}
    return {
        "job_id": job["id"],
        "kind": job.get("kind", "run"),
        "name": manifest.get("name", job["name"]),
        "state": manifest.get("state", job["state"]),
        "final_model_path": manifest.get("final_model_path"),
        "error": manifest.get("error"),
        "stages": list(manifest.get("stages", [])),
    }


def _find_stage(job: dict[str, Any], index: int) -> dict[str, Any] | None:
    for stage in _stages(job)["stages"]:
        if isinstance(stage, dict) and stage.get("index") == index:
            return stage
    return None


def _stage(job: dict[str, Any], index: int) -> dict[str, Any]:
    stage = _find_stage(job, index)
    if stage is None:
        raise JobNotFound(f"job {job['id']!r} has no stage {index}")
    return stage


def _require_stage(job: dict[str, Any], index: int) -> None:
    """404 unless the pipeline has a stage ``index``: one listed in the manifest or, before the
    runner wrote ``pipeline.json``, one of the job's configured stages."""
    if _find_stage(job, index) is not None:
        return
    configured = job["config"].get("stages") if job.get("kind") == "pipeline" else None
    if not isinstance(configured, list) or not 0 <= index < len(configured):
        raise JobNotFound(f"job {job['id']!r} has no stage {index}")


@router.get("/jobs/{job_id}/stages")
def job_stages(job_id: str, request: Request) -> dict[str, Any]:
    """Manifest stages of a pipeline job (``stages: []`` for plain runs or before the runner
    wrote ``pipeline.json``)."""
    return _stages(_jobs(request).get(job_id))


@router.get("/jobs/{job_id}/stages/stream")  # registered before /stages/{index}
async def stream_stages(job_id: str, request: Request) -> EventSourceResponse:
    """SSE: a ``stages`` event with the ``GET /jobs/{id}/stages`` payload now and whenever it
    changes, then ``end`` once the job is terminal. A snapshot stream: there is no ``since``
    cursor and ``end`` carries ``next: null``."""
    _jobs(request).get(job_id)  # 404 before the stream opens
    return EventSourceResponse(_stages_stream(request, job_id))


@router.get("/jobs/{job_id}/stages/{index}")
def job_stage(job_id: str, index: int, request: Request) -> dict[str, Any]:
    job = _jobs(request).get(job_id)
    stage = _stage(job, index)
    stage["status"] = read_status(stage["run_dir"]) if stage.get("run_dir") else None
    return stage


@router.get("/jobs/{job_id}/stages/{index}/metrics")
def stage_metrics(
    job_id: str,
    index: int,
    request: Request,
    since: int = Query(0, ge=0),
    limit: int = Query(MAX_ROWS, ge=1, le=MAX_ROWS),
) -> dict[str, Any]:
    stage = _stage(_jobs(request).get(job_id), index)
    return _rows(iter_metrics(stage["run_dir"], start_line=since), since, limit)


@router.get("/jobs/{job_id}/stages/{index}/trajectories")
def stage_trajectories(
    job_id: str,
    index: int,
    request: Request,
    since: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=MAX_ROWS),
) -> dict[str, Any]:
    stage = _stage(_jobs(request).get(job_id), index)
    return _rows(iter_trajectories(stage["run_dir"], start_line=since), since, limit)


@router.get("/jobs/{job_id}/stages/{index}/logs")
def stage_logs(
    job_id: str,
    index: int,
    request: Request,
    tail: int | None = Query(None, ge=0),
    since: int = Query(0, ge=0),
) -> dict[str, Any]:
    stage = _stage(_jobs(request).get(job_id), index)
    return read_log_lines(stage["run_dir"], since=since, tail=tail)


@router.get("/jobs/{job_id}/stages/{index}/checkpoints")
def stage_checkpoints(job_id: str, index: int, request: Request) -> dict[str, Any]:
    stage = _stage(_jobs(request).get(job_id), index)
    return _checkpoints(stage["run_dir"])


# -- SSE streams --------------------------------------------------------------------------------


class _Tail(NamedTuple):
    """What a tail stream follows: the run dir to read (``None`` until the pipeline manifest
    lists the stage) and the state that decides when the stream ends."""

    run_dir: str | None
    state: str
    error: str | None


def _job_tail(jobs: JobManager, job_id: str) -> _Tail:
    job = jobs.get(job_id)
    return _Tail(job["run_dir"], job["state"], job.get("error"))


def _stage_tail(jobs: JobManager, job_id: str, index: int) -> _Tail:
    """A stage is over when it reports a terminal state or when its job does (a cancelled or
    crashed pipeline leaves stages ``pending`` / ``running`` in the manifest)."""
    job = jobs.get(job_id)  # read before the manifest: a terminal job implies a final manifest
    stage = _find_stage(job, index) or {}
    run_dir = stage.get("run_dir") or None
    state = stage.get("state", "pending")
    if state in TERMINAL_STATES:
        return _Tail(run_dir, state, stage.get("error"))
    if job["state"] in TERMINAL_STATES:
        return _Tail(run_dir, job["state"], job.get("error"))
    return _Tail(run_dir, state, None)


def _new_metrics(run_dir: str, cursor: int) -> tuple[list[dict[str, Any]], int]:
    out = _rows(iter_metrics(run_dir, start_line=cursor), cursor, MAX_ROWS)
    return out["rows"], out["next"]


def _new_logs(run_dir: str, cursor: int) -> tuple[list[str], int]:
    out = read_log_lines(run_dir, since=cursor)
    return out["lines"], out["next"]


def _new_trajectories(run_dir: str, cursor: int) -> tuple[list[dict[str, Any]], int]:
    out = _rows(iter_trajectories(run_dir, start_line=cursor), cursor, MAX_ROWS)
    return out["rows"], out["next"]


# Tailable run-dir files by cursor name: (SSE event, ``read_new(run_dir, cursor) -> (items, cursor)``).
_TAILS: dict[str, tuple[str, Callable[[str, int], tuple[list[Any], int]]]] = {
    "metrics": ("metrics", _new_metrics),
    "logs": ("log", _new_logs),
    "trajectories": ("trajectory", _new_trajectories),
}


async def _tail_stream(
    request: Request,
    tail: Callable[[], _Tail],
    cursors: dict[str, int],
    end_next: Callable[[dict[str, int]], Any],
) -> AsyncIterator[dict[str, str]]:
    """Tail the files named in ``cursors`` (``{kind: first line to send}``, kinds of ``_TAILS``):
    one event per new row / line, in file order within a kind, then ``end`` with
    ``next = end_next(cursors)`` once ``tail()`` is terminal and every file has been drained. A
    run dir that does not exist yet (pending stage) simply has nothing to read."""
    cursors = dict(cursors)
    loop = asyncio.get_running_loop()
    while True:
        if await request.is_disconnected():
            return
        target = await loop.run_in_executor(None, tail)
        for kind in cursors if target.run_dir is not None else ():
            event, read_new = _TAILS[kind]
            while True:
                items, cursors[kind] = await loop.run_in_executor(
                    None, read_new, target.run_dir, cursors[kind]
                )
                for item in items:
                    yield {"event": event, "data": json.dumps(item)}
                if len(items) < MAX_ROWS:  # a full page: more rows may be waiting
                    break
        if target.state in TERMINAL_STATES:
            end = {"state": target.state, "next": end_next(cursors), "error": target.error}
            yield {"event": "end", "data": json.dumps(end)}
            return
        await asyncio.sleep(POLL_INTERVAL)


async def _stages_stream(request: Request, job_id: str) -> AsyncIterator[dict[str, str]]:
    jobs = _jobs(request)
    loop = asyncio.get_running_loop()
    sent: str | None = None
    while True:
        if await request.is_disconnected():
            return
        job = await loop.run_in_executor(None, jobs.get, job_id)
        data = json.dumps(await loop.run_in_executor(None, _stages, job))
        if data != sent:
            yield {"event": "stages", "data": data}
            sent = data
        if job["state"] in TERMINAL_STATES:
            yield {
                "event": "end",
                "data": json.dumps({"state": job["state"], "next": None, "error": job.get("error")}),
            }
            return
        await asyncio.sleep(POLL_INTERVAL)


def _job_stream(request: Request, job_id: str, kind: str, since: int) -> EventSourceResponse:
    """Single-file tail of a job's run dir; ``end`` carries that file's cursor as ``next``."""
    jobs = _jobs(request)
    jobs.get(job_id)  # 404 before the stream opens
    return EventSourceResponse(
        _tail_stream(request, partial(_job_tail, jobs, job_id), {kind: since}, itemgetter(kind))
    )


@router.get("/jobs/{job_id}/metrics/stream")
async def stream_metrics(job_id: str, request: Request, since: int = Query(0, ge=0)) -> EventSourceResponse:
    return _job_stream(request, job_id, "metrics", since)


@router.get("/jobs/{job_id}/logs/stream")
async def stream_logs(job_id: str, request: Request, since: int = Query(0, ge=0)) -> EventSourceResponse:
    return _job_stream(request, job_id, "logs", since)


@router.get("/jobs/{job_id}/events")
async def stream_events(
    job_id: str,
    request: Request,
    metrics: int = Query(0, ge=0),
    logs: int = Query(0, ge=0),
    trajectories: int = Query(0, ge=0),
) -> EventSourceResponse:
    """SSE: the run dir's metrics, log and trajectories on one connection. Events ``metrics`` /
    ``log`` / ``trajectory`` (one row or line each), each kind starting at its own cursor (same
    meaning as ``since`` on the JSON routes), then ``end`` ``{state, error, next: {metrics, logs,
    trajectories}}`` once the job is terminal and the three files are drained."""
    jobs = _jobs(request)
    jobs.get(job_id)  # 404 before the stream opens
    cursors = {"metrics": metrics, "logs": logs, "trajectories": trajectories}
    return EventSourceResponse(_tail_stream(request, partial(_job_tail, jobs, job_id), cursors, dict))


@router.get("/jobs/{job_id}/stages/{index}/events")
async def stream_stage_events(
    job_id: str,
    index: int,
    request: Request,
    metrics: int = Query(0, ge=0),
    logs: int = Query(0, ge=0),
    trajectories: int = Query(0, ge=0),
) -> EventSourceResponse:
    """``/jobs/{id}/events`` for one stage run dir; ``end`` once that stage (or the job) is
    terminal. A stage that has not started keeps the stream open (its run dir appears when it
    does); an index the pipeline does not have is a 404."""
    jobs = _jobs(request)
    _require_stage(jobs.get(job_id), index)
    cursors = {"metrics": metrics, "logs": logs, "trajectories": trajectories}
    return EventSourceResponse(
        _tail_stream(request, partial(_stage_tail, jobs, job_id, index), cursors, dict)
    )
