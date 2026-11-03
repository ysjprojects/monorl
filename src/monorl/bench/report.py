"""Rich tables + JSON export for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from monorl.bench.env import EnvBenchmark
from monorl.bench.plan import PlanReport
from monorl.bench.rollout import RolloutBenchmark
from monorl.bench.stats import fmt_bytes, fmt_params, fmt_seconds
from monorl.bench.sync import SyncBenchmark


def write_json(result: Any, path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2, default=str) + "\n")
    return out


def _notes(console: Console, notes: list[str]) -> None:
    for n in notes:
        console.print(f"  [cyan]•[/cyan] {n}")


def render_rollout(bench: RolloutBenchmark, console: Console) -> None:
    table = Table(title=f"rollout throughput — {bench.backend} / {bench.model}")
    for col in ("batch", "calls", "reqs", "tok/s", "tot tok/s", "req/s", "p50", "p95", "p50/req", "tok/req", "RSS"):
        table.add_column(col, justify="right", no_wrap=True)
    for r in bench.results:
        table.add_row(
            str(r.batch_size),
            str(r.calls),
            str(r.requests),
            f"{r.tokens_per_s:,.1f}",
            f"{r.total_tokens_per_s:,.1f}",
            f"{r.requests_per_s:,.2f}",
            fmt_seconds(r.latency.p50_s),
            fmt_seconds(r.latency.p95_s),
            fmt_seconds(r.request_latency.p50_s),
            f"{r.mean_completion_tokens:.1f}",
            fmt_bytes(r.peak_rss_bytes) + (" *" if r.truncated_by_budget else ""),
            style="bold green" if r.batch_size == bench.best_batch_size else None,
        )
    console.print(table)
    if any(r.truncated_by_budget for r in bench.results):
        console.print("  [dim]* stopped early by --seconds[/dim]")
    console.print(f"host: {bench.host.platform}, {bench.host.cpu_count} cpus, {fmt_bytes(bench.host.memory_bytes)} RAM, device {bench.host.device}")
    _notes(console, bench.recommendations)


def render_env(bench: EnvBenchmark, console: Console) -> None:
    spec = bench.spec
    table = Table(title=f"env throughput — {bench.target.transport} {bench.target.label}")
    for col in ("par", "procs", "eps", "steps", "eps/s", "steps/s", "reset p50", "step p50", "step p95", "ep p95", "err"):
        table.add_column(col, justify="right", no_wrap=True)
    for r in bench.results:
        table.add_row(
            str(r.parallel),
            str(r.processes) if r.processes else "-",
            str(r.episodes),
            str(r.steps),
            f"{r.episodes_per_s:,.2f}",
            f"{r.steps_per_s:,.2f}",
            fmt_seconds(r.reset_latency.p50_s),
            fmt_seconds(r.step_latency.p50_s),
            fmt_seconds(r.step_latency.p95_s),
            fmt_seconds(r.episode_latency.p95_s),
            str(r.errors),
            style="bold green" if r.parallel == bench.best_parallel else None,
        )
    console.print(table)
    console.print(
        f"spec: name={spec.get('name')} version={spec.get('version')} max_turns={spec.get('max_turns')} "
        f"parallel_episodes={spec.get('parallel_episodes')}; turns/episode={bench.turns}"
    )
    _notes(console, bench.recommendations)


def render_sync(bench: SyncBenchmark, console: Console) -> None:
    title = f"weight sync — {bench.source}" + (f" ({bench.num_params:,} params)" if bench.num_params else "")
    table = Table(title=title)
    for col in ("strategy", "repeats", "bytes", "p50", "p95", "max", "GB/s", "status"):
        table.add_column(col, justify="right", no_wrap=True)
    for r in bench.results:
        table.add_row(
            r.strategy,
            str(r.repeats),
            fmt_bytes(r.bytes) if r.bytes else "-",
            fmt_seconds(r.seconds.p50_s) if r.repeats else "-",
            fmt_seconds(r.seconds.p95_s) if r.repeats else "-",
            fmt_seconds(r.seconds.max_s) if r.repeats else "-",
            f"{r.gb_per_s:.2f}" if r.repeats else "-",
            "[yellow]skipped[/yellow]" if r.skipped else "[green]ok[/green]",
            style="bold green" if r.strategy == bench.fastest else None,
        )
    console.print(table)
    console.print(f"engines: {bench.engines}; peak RSS {fmt_bytes(bench.peak_rss_bytes)}")
    _notes(console, bench.recommendations)


def render_plan(report: PlanReport, console: Console) -> None:
    plan = report.plan
    cluster = report.cluster
    table = Table(title=f"placement plan — {report.config_summary['model']}")
    table.add_column("setting")
    table.add_column("value")
    rows = [
        ("cluster", f"{cluster.get('num_nodes')} node(s) × {plan.get('gpus_per_node')} GPU(s) = {plan.get('total_gpus')}"),
        ("gpu memory", fmt_bytes(cluster["cuda_devices"][0]["total_memory_bytes"]) if cluster.get("cuda_devices") else "-"),
        ("mode / placement", f"{plan.get('mode')} / {plan.get('placement')}"),
        ("rollout", f"{plan.get('engine_replicas')} replica(s) × tp{plan.get('engine_tensor_parallel')} on {plan.get('rollout_devices')}"),
        ("trainer", f"world size {plan.get('trainer_world_size')} on {plan.get('trainer_devices')}"),
        ("engine sleep", str(plan.get("engine_sleep"))),
        ("weight sync", str(plan.get("weight_sync"))),
    ]
    est = plan.get("model_estimate")
    if est:
        rows += [
            ("model params", f"{fmt_params(est['params'])} ({fmt_bytes(est['weight_bytes'])} weights)"),
            ("engine footprint", f"{fmt_bytes(est['engine_bytes'])} (kv {fmt_bytes(est['kv_bytes'])})"),
            ("trainer footprint", fmt_bytes(est["trainer_bytes"])),
        ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)
    if report.estimate:
        e = report.estimate
        t = Table(title=f"step-time estimate (assumed {e.hardware.name}: {e.hardware.tflops:.0f} TFLOP/s, {e.hardware.bandwidth_gbs:.0f} GB/s)")
        t.add_column("phase")
        t.add_column("seconds", justify="right")
        t.add_column("share", justify="right")
        for name, s in (("generation", e.generation_s), ("logprobs", e.logprob_s), ("training", e.train_s), ("weight sync", e.sync_s)):
            t.add_row(name, fmt_seconds(s), f"{100 * s / e.total_s:.0f}%" if e.total_s else "-")
        t.add_row("[bold]total[/bold]", f"[bold]{fmt_seconds(e.total_s)}[/bold]", "")
        console.print(t)
        console.print(
            f"{e.sequences_per_step} sequences/step, {e.step_tokens:,} tokens/step ({e.generated_tokens:,} generated): "
            f"~{e.tokens_per_s:,.0f} generated tok/s, ~{e.steps_per_hour:,.1f} steps/hour"
        )
        _notes(console, e.notes)
    _notes(console, report.notes)


__all__ = ["render_env", "render_plan", "render_rollout", "render_sync", "write_json"]
