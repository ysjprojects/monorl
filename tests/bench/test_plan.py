from __future__ import annotations

import json

import pytest

from monorl.bench.plan import (
    GPU_PRESETS,
    HardwareAssumptions,
    estimate_step,
    pick_hardware,
    run_plan,
    synthetic_cluster,
)
from monorl.config.schema import RunConfig
from monorl.distributed.cluster import ClusterInfo, GpuInfo
from monorl.distributed.placement import ModelEstimate, PlacementPlan


def _config(**overrides) -> RunConfig:
    data = {
        "model": {"name_or_path": "fake/model"},
        "data": {"uri": "x.jsonl"},
        "rewards": [{"type": "exact_match"}],
        "rollout": {"group_size": 4, "max_new_tokens": 512, "data_parallel_size": 2},
        "trainer": {"prompts_per_step": 16},
        "cluster": {"num_nodes": 1, "gpus_per_node": 0},
    }
    data.update(overrides)
    return RunConfig.from_dict(data)


def _fake_plan(config: RunConfig, cluster: ClusterInfo, *, with_estimate: bool = True) -> PlacementPlan:
    gpus = cluster.num_gpus
    params = 7_000_000_000
    est = (
        ModelEstimate(
            params=params,
            dtype_bytes=2,
            weight_bytes=params * 2,
            kv_bytes=1 << 30,
            engine_bytes=params * 2 + (1 << 30),
            trainer_bytes=params * 16,
        )
        if with_estimate
        else None
    )
    return PlacementPlan(
        mode="local",
        placement="disaggregated" if gpus >= 4 else "colocated",
        num_nodes=1,
        gpus_per_node=gpus,
        total_gpus=gpus,
        rollout_gpus=2 if gpus >= 4 else gpus,
        trainer_gpus=gpus - 2 if gpus >= 4 else gpus,
        engine_replicas=config.rollout.data_parallel_size,
        engine_tensor_parallel=1,
        trainer_world_size=max(1, gpus - 2 if gpus >= 4 else gpus),
        engine_sleep=gpus < 4,
        weight_sync="inprocess",
        device="cuda" if gpus else "cpu",
        rollout_devices=[f"cuda:{i}" for i in range(2)] if gpus else ["cpu"],
        trainer_devices=[f"cuda:{i}" for i in range(2, gpus)] if gpus >= 4 else ["cpu"],
        model_estimate=est,
        notes=["fake plan"],
    )


def test_synthetic_cluster_builds_requested_gpus() -> None:
    c = synthetic_cluster(8, 80)
    assert c.num_gpus == 8 and c.gpu_memory_bytes == 80 << 30
    assert all(g.name == "assumed" for g in c.cuda_devices)
    assert synthetic_cluster(0, None).num_gpus == 0
    detected = synthetic_cluster(None, None)
    assert isinstance(detected, ClusterInfo)


def test_pick_hardware_defaults_and_overrides() -> None:
    gpu = ClusterInfo(cpu_count=4, cuda_devices=(GpuInfo(0, "x", 1 << 30),))
    assert pick_hardware(gpu, None, None, None) is GPU_PRESETS["h100"]
    cpu = ClusterInfo(cpu_count=4)
    assert pick_hardware(cpu, None, None, None).name == "cpu"
    custom = pick_hardware(gpu, "a100", 100.0, None)
    assert custom.tflops == 100.0 and custom.bandwidth_gbs == GPU_PRESETS["a100"].bandwidth_gbs
    with pytest.raises(ValueError):
        pick_hardware(gpu, "tpu", None, None)


def test_estimate_step_math() -> None:
    cfg = _config()
    cluster = synthetic_cluster(8, 80)
    plan = _fake_plan(cfg, cluster)
    hw = HardwareAssumptions("t", tflops=1000.0, bandwidth_gbs=1000.0, train_mfu=0.5, decode_mfu=0.5)
    est = estimate_step(cfg, plan, hardware=hw, prompt_tokens=100)
    assert est is not None
    assert est.sequences_per_step == 64
    assert est.generated_tokens == 64 * 512
    assert est.step_tokens == 64 * 612
    params = plan.model_estimate.params
    gen_bw = 512 * (params * 2) / 1e12
    gen_compute = 2 * params * est.generated_tokens / (2 * 1e15 * 0.5)
    assert est.generation_s == pytest.approx(max(gen_bw, gen_compute))
    train_rate = 6 * 1e15 * 0.5
    assert est.logprob_s == pytest.approx(2 * params * est.step_tokens / train_rate)
    assert est.train_s == pytest.approx(6 * params * est.step_tokens / train_rate)
    assert est.sync_s == pytest.approx(0.05 + params * 2 / 25e9)
    assert est.total_s == pytest.approx(est.generation_s + est.logprob_s + est.train_s + est.sync_s)
    assert est.steps_per_hour == pytest.approx(3600 / est.total_s)
    assert est.tokens_per_s == pytest.approx(est.generated_tokens / est.total_s)
    assert any("dominates" in n for n in est.notes)


def test_estimate_step_lora_kl_and_async_overlap() -> None:
    cfg = _config(
        model={"name_or_path": "fake/model", "lora": {"enabled": True}},
        algorithm={"name": "grpo", "kl_coef": 0.1},
        rollout={"group_size": 4, "max_new_tokens": 512, "data_parallel_size": 2, "async_steps_ahead": 1},
    )
    plan = _fake_plan(cfg, synthetic_cluster(8, 80))
    hw = GPU_PRESETS["h100"]
    est = estimate_step(cfg, plan, hardware=hw)
    base = estimate_step(_config(), plan, hardware=hw)
    assert est is not None and base is not None
    assert est.logprob_s == pytest.approx(2 * base.logprob_s)  # reference pass added
    assert est.train_s == pytest.approx(base.train_s * 4 / 6)  # LoRA
    assert est.total_s == pytest.approx(max(est.generation_s, est.logprob_s + est.train_s) + est.sync_s)
    assert any("async_steps_ahead" in n for n in est.notes)
    assert (
        estimate_step(cfg, _fake_plan(cfg, synthetic_cluster(8, 80), with_estimate=False), hardware=hw)
        is None
    )


def test_run_plan_with_fake_plan_placement() -> None:
    cfg = _config()
    seen: dict[str, object] = {}

    def fake(config: RunConfig, cluster: ClusterInfo) -> PlacementPlan:
        seen["cluster"] = cluster
        return _fake_plan(config, cluster)

    report = run_plan(cfg, gpus=8, gpu_mem_gb=80, plan_fn=fake)
    assert seen["cluster"].num_gpus == 8
    assert report.plan["placement"] == "disaggregated" and report.plan["engine_gpus"] == 2
    assert report.plan["model_estimate"]["params"] == 7_000_000_000
    assert report.estimate is not None and report.estimate.hardware.name == "h100"
    assert "fake plan" in report.notes
    d = report.to_dict()
    assert d["kind"] == "plan" and d["config"]["model"] == "fake/model" and d["cluster"]["num_gpus"] == 8
    assert d["estimate"]["hardware"]["tflops"] == 989.0
    json.loads(json.dumps(d))

    cpu_report = run_plan(cfg, gpus=0, plan_fn=lambda c, cl: _fake_plan(c, cl, with_estimate=False))
    assert cpu_report.estimate is None
    assert any("no step-time estimate" in n for n in cpu_report.notes)
    assert cpu_report.plan["placement"] == "colocated"
