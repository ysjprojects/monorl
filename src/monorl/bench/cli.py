"""``python -m monorl.bench`` / ``monorl bench``: throughput tooling.

Subcommands: ``rollout`` (engine tokens/s per batch size), ``env`` (MEP/1 episodes/s per
parallelism), ``sync`` (weight-sync seconds per strategy), ``plan`` (placement + step-time estimate).
Heavy modules (torch, engines, data sources, runtime) are imported inside the commands.
"""

# ruff: noqa: B008  (typer.Option in defaults is the typer idiom)
from __future__ import annotations

import sys
from itertools import islice
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from monorl.bench.report import render_env, render_plan, render_rollout, render_sync, write_json
from monorl.config.loader import apply_overrides, load_yaml
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, MonorlError

app = typer.Typer(
    name="bench",
    help="Measure rollout, environment and weight-sync throughput to size an RL run for this hardware.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[bold red]error:[/bold red] {message}")
    raise typer.Exit(code)


def _int_list(text: str, flag: str) -> list[int]:
    try:
        values = [int(v) for v in text.replace(" ", "").split(",") if v]
    except ValueError:
        _fail(f"{flag} expects comma-separated integers, got {text!r}", code=2)
    if not values or any(v < 1 for v in values):
        _fail(f"{flag} needs positive integers, got {text!r}", code=2)
    return values


def _load_config(path: Path, overrides: list[str] | None) -> RunConfig:
    try:
        return RunConfig.from_dict(apply_overrides(load_yaml(path), overrides))
    except ConfigError as e:
        _fail(str(e))
        raise AssertionError("unreachable") from e


def _emit(result: Any, json_path: Path | None, render: Any) -> None:
    render(result, console)
    if json_path is not None:
        out = write_json(result, json_path)
        console.print(f"json: {out}")


# -- rollout ---------------------------------------------------------------------------------


@app.command()
def rollout(
    config: Path = typer.Option(
        ..., "--config", "-c", help="RunConfig YAML; rollout.backend picks the engine."
    ),
    set_: list[str] | None = typer.Option(
        None, "--set", "-s", help="Override: key.path=value (YAML-coerced)."
    ),
    prompts: int = typer.Option(64, "--prompts", min=1, help="Prompts taken from the configured dataset."),
    seconds: float | None = typer.Option(
        30.0, "--seconds", help="Wall-time budget per batch size (0 = unlimited)."
    ),
    batch_sizes: str = typer.Option(
        "1,4,16,64", "--batch-sizes", help="Comma-separated prompts per generate() call."
    ),
    n: int | None = typer.Option(
        None, "--n", min=1, help="Completions per prompt (default rollout.group_size)."
    ),
    max_new_tokens: int | None = typer.Option(
        None, "--max-new-tokens", min=1, help="Default rollout.max_new_tokens."
    ),
    warmup: bool = typer.Option(
        True, "--warmup/--no-warmup", help="One untimed generate() call before measuring."
    ),
    json_path: Path | None = typer.Option(None, "--json", help="Write the full result as JSON to this path."),
) -> None:
    """Generate from the recipe's dataset at several batch sizes: tokens/s, requests/s, latency, RSS."""
    from monorl.core.registry import registry
    from monorl.core.types import SamplingParams

    cfg = _load_config(config, set_)
    sizes = _int_list(batch_sizes, "--batch-sizes")
    budget = seconds if seconds and seconds > 0 else None
    try:
        engine = registry.get("rollout_engine", cfg.rollout.backend)(cfg)
    except MonorlError as e:
        _fail(str(e))
        return
    try:
        from monorl.data import open_source

        source = open_source(cfg.data, tokenizer=engine.tokenizer, seed=cfg.seed)
        prompt_list = [s.messages for s in islice(iter(source), prompts)]
        if not prompt_list:
            _fail(f"no usable samples in {cfg.data.uri}")
        params = SamplingParams(
            n=n or cfg.rollout.group_size,
            max_new_tokens=max_new_tokens or cfg.rollout.max_new_tokens,
            temperature=cfg.rollout.temperature,
            top_p=cfg.rollout.top_p,
            top_k=cfg.rollout.top_k,
            stop=list(cfg.rollout.stop),
            seed=cfg.seed,
            logprobs=True,
        )
        console.print(
            f"benchmarking {cfg.rollout.backend} on {cfg.model.name_or_path}: {len(prompt_list)} prompts × n={params.n}, "
            f"max_new_tokens={params.max_new_tokens}, batch sizes {sizes}"
        )
        from monorl.bench.rollout import run_rollout_benchmark

        result = run_rollout_benchmark(
            engine,
            prompt_list,
            params,
            batch_sizes=sizes,
            seconds=budget,
            warmup=warmup,
            backend=cfg.rollout.backend,
            model=cfg.model.name_or_path,
            group_size=cfg.rollout.group_size,
            extra={"config": str(config), "data_uri": cfg.data.uri, "seconds_budget": budget},
        )
    except MonorlError as e:
        _fail(str(e))
        return
    finally:
        engine.shutdown()
    _emit(result, json_path, render_rollout)


# -- env -------------------------------------------------------------------------------------


@app.command()
def env(
    command: list[str] | None = typer.Option(
        None, "--command", "-c", help="Server argv for stdio (repeatable)."
    ),
    url: str | None = typer.Option(None, "--url", help="Base URL of an http MEP/1 server."),
    parallel: str = typer.Option("1,4,16", "--parallel", help="Comma-separated concurrent-episode levels."),
    episodes: int = typer.Option(32, "--episodes", min=1, help="Episodes per level."),
    turns: int = typer.Option(1, "--turns", min=1, help="Steps per episode (clamped to spec.max_turns)."),
    cwd: Path | None = typer.Option(None, "--cwd", help="Working directory for the stdio server."),
    timeout: float = typer.Option(60.0, "--timeout", help="Seconds to wait for each response."),
    parallel_episodes: int | None = typer.Option(
        None,
        "--parallel-episodes",
        min=1,
        help="Episodes per stdio process (default spec.parallel_episodes).",
    ),
    json_path: Path | None = typer.Option(None, "--json", help="Write the full result as JSON to this path."),
) -> None:
    """Drive a MEP/1 server with a canned policy at several parallelism levels: episodes/s, step latency."""
    from monorl.bench.env import EnvTarget, run_env_benchmark

    if bool(command) == bool(url):
        _fail("pass exactly one of --command or --url", code=2)
    levels = _int_list(parallel, "--parallel")
    target = EnvTarget(
        command=list(command) if command else None, url=url, cwd=str(cwd) if cwd else None, timeout_s=timeout
    )
    try:
        result = run_env_benchmark(
            target,
            parallel_levels=levels,
            episodes=episodes,
            turns=turns,
            parallel_episodes=parallel_episodes,
        )
    except MonorlError as e:
        _fail(str(e))
        return
    _emit(result, json_path, render_env)
    if all(r.episodes == 0 for r in result.results):
        raise typer.Exit(1)


# -- sync ------------------------------------------------------------------------------------


@app.command()
def sync(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="RunConfig YAML; builds the real runtime."
    ),
    set_: list[str] | None = typer.Option(
        None, "--set", "-s", help="Override: key.path=value (YAML-coerced)."
    ),
    fake_params: int | None = typer.Option(
        None,
        "--fake-params",
        min=1,
        help="Use N synthetic fp32 parameters instead of the configured trainer/engine.",
    ),
    engines: int = typer.Option(
        1, "--engines", min=1, help="Synthetic engines to sync into (--fake-params only)."
    ),
    repeats: int = typer.Option(3, "--repeats", min=1, help="Sync calls per strategy."),
    all_strategies: bool = typer.Option(
        False,
        "--all",
        help="Try every registered strategy, not only the configured one (default with --fake-params).",
    ),
    strategy: list[str] | None = typer.Option(
        None, "--strategy", help="Strategy name(s) to measure (repeatable)."
    ),
    json_path: Path | None = typer.Option(None, "--json", help="Write the full result as JSON to this path."),
) -> None:
    """Time weight sync (trainer → engines) for the configured strategy or every strategy that works here."""
    from monorl.bench.sync import (
        StrategyCandidate,
        SyntheticEngine,
        SyntheticWeightSource,
        resolve_strategies,
        run_sync_benchmark,
        synthetic_config,
    )

    if config is None and fake_params is None:
        _fail("pass --config RECIPE.yaml and/or --fake-params N", code=2)
    cfg = _load_config(config, set_) if config is not None else synthetic_config()
    names = list(strategy) if strategy else None
    try:
        if fake_params is not None:
            source = SyntheticWeightSource(fake_params)
            engine_list: list[Any] = [SyntheticEngine(source) for _ in range(engines)]
            if (
                names is None
                and not all_strategies
                and config is not None
                and cfg.cluster.weight_sync != "auto"
            ):
                names = [cfg.cluster.weight_sync]
            candidates = resolve_strategies(cfg, names)
            console.print(
                f"synthetic source: {source.num_params:,} params ({source.num_bytes / 1e9:.3f} GB) → {engines} engine(s)"
            )
            result = run_sync_benchmark(
                source,
                engine_list,
                candidates,
                repeats=repeats,
                source="synthetic",
                num_params=source.num_params,
            )
        else:
            from monorl.distributed import build_runtime

            runtime = build_runtime(cfg)
            try:
                configured = StrategyCandidate(runtime.weight_sync.name, runtime.weight_sync)
                if names is not None:
                    candidates = resolve_strategies(cfg, names)
                elif all_strategies:
                    others = [c for c in resolve_strategies(cfg) if c.name != configured.name]
                    candidates = [configured, *others]
                else:
                    candidates = [configured]
                console.print(
                    f"runtime: trainer={cfg.trainer.backend} engines={len(runtime.engines)}×{cfg.rollout.backend} "
                    f"placement={runtime.placement.get('placement')} configured sync={configured.name}"
                )
                result = run_sync_benchmark(
                    runtime.trainer,
                    runtime.engines,
                    candidates,
                    repeats=repeats,
                    source="runtime",
                    extra={"config": str(config), "placement": runtime.placement},
                )
            finally:
                runtime.close()
    except MonorlError as e:
        _fail(str(e))
        return
    _emit(result, json_path, render_sync)
    if result.fastest is None:
        raise typer.Exit(1)


# -- plan ------------------------------------------------------------------------------------


@app.command()
def plan(
    config: Path = typer.Option(..., "--config", "-c", help="RunConfig YAML."),
    set_: list[str] | None = typer.Option(
        None, "--set", "-s", help="Override: key.path=value (YAML-coerced)."
    ),
    gpus: int | None = typer.Option(None, "--gpus", min=0, help="GPUs to plan for (default: detected)."),
    gpu_mem_gb: float | None = typer.Option(
        None, "--gpu-mem-gb", min=0.0, help="Memory per GPU in GiB (default: detected or 80)."
    ),
    preset: str | None = typer.Option(
        None, "--preset", help="Throughput preset: h100 (default with GPUs), a100, l40s, mps, cpu."
    ),
    tflops: float | None = typer.Option(None, "--gpu-tflops", help="Override dense bf16 TFLOP/s per device."),
    bandwidth: float | None = typer.Option(
        None, "--gpu-bandwidth-gbs", help="Override memory bandwidth per device (GB/s)."
    ),
    prompt_tokens: int = typer.Option(
        256, "--prompt-tokens", min=1, help="Assumed prompt length for the estimate."
    ),
    json_path: Path | None = typer.Option(None, "--json", help="Write the full result as JSON to this path."),
) -> None:
    """Print the placement plan for the recipe on G×M GB GPUs and a back-of-envelope step-time estimate."""
    from monorl.bench.plan import run_plan

    cfg = _load_config(config, set_)
    try:
        result = run_plan(
            cfg,
            gpus=gpus,
            gpu_mem_gb=gpu_mem_gb,
            preset=preset,
            tflops=tflops,
            bandwidth_gbs=bandwidth,
            prompt_tokens=prompt_tokens,
        )
    except (MonorlError, ValueError) as e:
        _fail(str(e))
        return
    _emit(result, json_path, render_plan)


def main(argv: list[str] | None = None) -> None:
    app(args=argv if argv is not None else sys.argv[1:], prog_name="python -m monorl.bench")


__all__ = ["app", "main"]
