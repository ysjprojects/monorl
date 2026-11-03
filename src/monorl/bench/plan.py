"""Placement plan + back-of-envelope step-time estimate for a recipe on hypothetical hardware.

``plan_placement`` (``monorl.distributed``) is pure: config + :class:`ClusterInfo` in, plan out.
Here the cluster is synthesised from ``--gpus/--gpu-mem-gb`` (or detected when omitted) so users
can ask "what would this recipe do on 8×80 GB?" from a laptop.

Step-time model (per optimizer step, ``S = prompts_per_step * group_size`` sequences):

* generation — decode is weight-bandwidth bound per engine: every decode step streams the
  replica's weight shard once, ``max_new_tokens * weight_bytes / tp / bandwidth``; compute bound
  is ``2 * params * generated_tokens / (engine_gpus * tflops * decode_mfu)``; take the max.
* logprob passes — ``forward_passes * 2 * params * step_tokens / (trainer_gpus * tflops * train_mfu)``
  (old-policy logprobs always, reference logprobs when ``kl_coef > 0``).
* training — ``6 * params`` FLOPs per token (``4`` with LoRA, no weight grads for the base) times
  ``step_tokens * num_epochs``.
* weight sync — ``weight_bytes / sync_bandwidth[strategy]`` plus a fixed latency.

Assumed throughput constants live in :data:`GPU_PRESETS` / :data:`SYNC_BANDWIDTH` and are
reported alongside the estimate; override with ``--gpu-tflops/--gpu-bandwidth-gbs``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from monorl.bench.stats import HostInfo

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.distributed.cluster import ClusterInfo
    from monorl.distributed.placement import PlacementPlan


@dataclass(frozen=True, slots=True)
class HardwareAssumptions:
    name: str
    tflops: float
    """Dense bf16 TFLOP/s per device."""
    bandwidth_gbs: float
    """HBM/unified-memory bandwidth per device in GB/s."""
    train_mfu: float = 0.40
    decode_mfu: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GPU_PRESETS: dict[str, HardwareAssumptions] = {
    "h100": HardwareAssumptions("h100", tflops=989.0, bandwidth_gbs=3350.0),
    "a100": HardwareAssumptions("a100", tflops=312.0, bandwidth_gbs=2039.0),
    "l40s": HardwareAssumptions("l40s", tflops=362.0, bandwidth_gbs=864.0),
    "mps": HardwareAssumptions("mps", tflops=10.0, bandwidth_gbs=200.0, train_mfu=0.25, decode_mfu=0.15),
    "cpu": HardwareAssumptions("cpu", tflops=1.0, bandwidth_gbs=80.0, train_mfu=0.20, decode_mfu=0.10),
}
DEFAULT_GPU_PRESET = "h100"

SYNC_BANDWIDTH: dict[str, tuple[float, float]] = {
    # strategy -> (GB/s, fixed seconds)
    "inprocess": (25.0, 0.05),
    "cuda_ipc": (200.0, 0.2),
    "broadcast": (40.0, 0.5),
    "checkpoint": (1.5, 2.0),
}

ASSUMED_PROMPT_TOKENS = 256


@dataclass(slots=True)
class StepEstimate:
    sequences_per_step: int
    prompt_tokens: int
    generated_tokens: int
    step_tokens: int
    generation_s: float
    logprob_s: float
    train_s: float
    sync_s: float
    total_s: float
    tokens_per_s: float
    steps_per_hour: float
    hardware: HardwareAssumptions
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hardware"] = self.hardware.to_dict()
        return d


@dataclass(slots=True)
class PlanReport:
    config_summary: dict[str, Any]
    cluster: dict[str, Any]
    plan: dict[str, Any]
    estimate: StepEstimate | None
    notes: list[str]
    host: HostInfo

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "plan",
            "config": dict(self.config_summary),
            "cluster": dict(self.cluster),
            "plan": dict(self.plan),
            "estimate": self.estimate.to_dict() if self.estimate else None,
            "notes": list(self.notes),
            "host": self.host.to_dict(),
        }


def synthetic_cluster(gpus: int | None, gpu_mem_gb: float | None) -> ClusterInfo:
    """Detected cluster when both are ``None``; otherwise ``gpus`` identical CUDA devices of
    ``gpu_mem_gb`` (each ``None`` falls back to the detected value, memory to 80 GiB)."""
    from monorl.distributed.cluster import ClusterInfo, GpuInfo, detect_cluster

    detected = detect_cluster(use_ray=False)
    if gpus is None and gpu_mem_gb is None:
        return detected
    count = detected.num_gpus if gpus is None else int(gpus)
    mem = int(gpu_mem_gb * (1 << 30)) if gpu_mem_gb is not None else (detected.gpu_memory_bytes or 80 << 30)
    devices = tuple(GpuInfo(index=i, name="assumed", total_memory_bytes=mem) for i in range(count))
    return ClusterInfo(cpu_count=detected.cpu_count, cuda_devices=devices, mps_available=detected.mps_available and not devices)


def estimate_step(
    config: RunConfig,
    plan: PlacementPlan,
    *,
    hardware: HardwareAssumptions,
    prompt_tokens: int = ASSUMED_PROMPT_TOKENS,
) -> StepEstimate | None:
    est = plan.model_estimate
    if est is None:
        return None
    notes: list[str] = []
    seqs = config.sequences_per_step()
    gen_tokens = seqs * config.rollout.max_new_tokens
    step_tokens = seqs * (prompt_tokens + config.rollout.max_new_tokens)
    tflops = hardware.tflops * 1e12
    bandwidth = hardware.bandwidth_gbs * 1e9
    engine_gpus = max(1, plan.engine_gpus)
    trainer_gpus = max(1, plan.trainer_gpus)
    tp = max(1, plan.engine_tensor_parallel)

    gen_bandwidth_s = config.rollout.max_new_tokens * (est.weight_bytes / tp) / bandwidth
    gen_compute_s = 2.0 * est.params * gen_tokens / (engine_gpus * tflops * hardware.decode_mfu)
    generation_s = max(gen_bandwidth_s, gen_compute_s)
    notes.append(
        "generation is "
        + ("weight-bandwidth bound (small batch); bigger batches or more replicas are nearly free" if gen_bandwidth_s >= gen_compute_s
           else "compute bound; batch is saturating the engine")
    )

    forward_passes = 1 + (1 if config.algorithm.kl_coef > 0 else 0)
    train_flops_per_token = 4.0 if config.model.lora.enabled else 6.0
    train_rate = trainer_gpus * tflops * hardware.train_mfu
    logprob_s = forward_passes * 2.0 * est.params * step_tokens / train_rate
    train_s = train_flops_per_token * est.params * step_tokens * config.algorithm.num_epochs / train_rate

    gbs, fixed = SYNC_BANDWIDTH.get(plan.weight_sync, SYNC_BANDWIDTH["checkpoint"])
    sync_s = fixed + est.weight_bytes / (gbs * 1e9)
    if plan.weight_sync == "inprocess" and plan.engine_sleep:
        sync_s += 2 * fixed
        notes.append("colocated with sleep mode: add engine sleep/wake per step")

    total = generation_s + logprob_s + train_s + sync_s
    if plan.placement == "disaggregated" and config.rollout.async_steps_ahead > 0:
        overlapped = max(generation_s, logprob_s + train_s) + sync_s
        notes.append(f"async_steps_ahead={config.rollout.async_steps_ahead}: generation overlaps training, ~{overlapped:.1f}s/step")
        total = overlapped
    share = max(generation_s, logprob_s + train_s)
    notes.append(
        "generation dominates: raise rollout.data_parallel_size or batch size"
        if share == generation_s
        else "training dominates: raise trainer.micro_batch_size / use LoRA or more trainer GPUs"
    )
    return StepEstimate(
        sequences_per_step=seqs,
        prompt_tokens=prompt_tokens,
        generated_tokens=gen_tokens,
        step_tokens=step_tokens,
        generation_s=generation_s,
        logprob_s=logprob_s,
        train_s=train_s,
        sync_s=sync_s,
        total_s=total,
        tokens_per_s=gen_tokens / total if total > 0 else 0.0,
        steps_per_hour=3600.0 / total if total > 0 else 0.0,
        hardware=hardware,
        notes=notes,
    )


def pick_hardware(cluster: ClusterInfo, preset: str | None, tflops: float | None, bandwidth_gbs: float | None) -> HardwareAssumptions:
    if preset is None:
        preset = DEFAULT_GPU_PRESET if cluster.num_gpus > 0 else ("mps" if cluster.mps_available else "cpu")
    try:
        base = GPU_PRESETS[preset]
    except KeyError as e:
        raise ValueError(f"unknown hardware preset {preset!r}; choose from {sorted(GPU_PRESETS)}") from e
    if tflops is None and bandwidth_gbs is None:
        return base
    return HardwareAssumptions(
        name=f"{base.name}*",
        tflops=tflops if tflops is not None else base.tflops,
        bandwidth_gbs=bandwidth_gbs if bandwidth_gbs is not None else base.bandwidth_gbs,
        train_mfu=base.train_mfu,
        decode_mfu=base.decode_mfu,
    )


def run_plan(
    config: RunConfig,
    *,
    gpus: int | None = None,
    gpu_mem_gb: float | None = None,
    preset: str | None = None,
    tflops: float | None = None,
    bandwidth_gbs: float | None = None,
    prompt_tokens: int = ASSUMED_PROMPT_TOKENS,
    plan_fn: Callable[[RunConfig, ClusterInfo], PlacementPlan] | None = None,
    cluster: ClusterInfo | None = None,
) -> PlanReport:
    """``plan_fn`` defaults to :func:`monorl.distributed.plan_placement` (imported lazily)."""
    if plan_fn is None:
        from monorl.distributed import plan_placement as plan_fn
    if cluster is None:
        cluster = synthetic_cluster(gpus, gpu_mem_gb)
    hardware = pick_hardware(cluster, preset, tflops, bandwidth_gbs)
    plan = plan_fn(config, cluster)
    estimate = estimate_step(config, plan, hardware=hardware, prompt_tokens=prompt_tokens)
    notes = list(plan.notes)
    if estimate is None:
        notes.append(f"model size unknown for {config.model.name_or_path}: no step-time estimate (config not readable)")
    return PlanReport(
        config_summary={
            "model": config.model.name_or_path,
            "lora": config.model.lora.enabled,
            "algorithm": config.algorithm.name,
            "rollout_backend": config.rollout.backend,
            "trainer_backend": config.trainer.backend,
            "group_size": config.rollout.group_size,
            "prompts_per_step": config.trainer.prompts_per_step,
            "max_new_tokens": config.rollout.max_new_tokens,
            "cluster_mode": config.cluster.mode,
        },
        cluster=cluster.to_dict(),
        plan=plan.to_dict(),
        estimate=estimate,
        notes=notes,
        host=HostInfo.detect(),
    )


__all__ = [
    "ASSUMED_PROMPT_TOKENS",
    "DEFAULT_GPU_PRESET",
    "GPU_PRESETS",
    "SYNC_BANDWIDTH",
    "HardwareAssumptions",
    "PlanReport",
    "StepEstimate",
    "estimate_step",
    "pick_hardware",
    "run_plan",
    "synthetic_cluster",
]
