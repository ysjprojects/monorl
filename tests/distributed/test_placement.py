from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import gpu_cluster
from monorl.core.errors import ConfigError
from monorl.distributed import ClusterInfo, plan_placement
from monorl.distributed.placement import estimate_model

LLAMA_7B = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "vocab_size": 32000,
    "tie_word_embeddings": False,
}


@pytest.fixture
def llama_7b_dir(tmp_path: Path) -> Path:
    d = tmp_path / "llama7b"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(LLAMA_7B))
    return d


# -- placement -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "gpus", "expected"),
    [
        ("local", 0, "colocated"),
        ("local", 1, "colocated"),
        ("ray", 1, "colocated"),
        ("ray", 0, "colocated"),
    ],
)
def test_auto_small_clusters_are_colocated(make_config, mode, gpus, expected):
    plan = plan_placement(make_config(cluster={"mode": mode}), gpu_cluster(gpus))
    assert plan.placement == expected
    assert plan.total_gpus == gpus
    assert plan.engine_replicas == 1
    assert plan.trainer_world_size == 1
    assert plan.engine_sleep is (gpus > 0)
    assert plan.rollout_devices == (["cuda:0"] if gpus else ["cpu"])


def test_auto_with_explicit_split_is_disaggregated(make_config):
    cfg = make_config(cluster={"mode": "ray", "rollout_gpus": 1, "trainer_gpus": 3}, trainer={"prompts_per_step": 4})
    plan = plan_placement(cfg, gpu_cluster(4))
    assert plan.placement == "disaggregated"
    assert (plan.rollout_gpus, plan.trainer_gpus) == (1, 3)
    assert plan.trainer_world_size == 3
    assert plan.rollout_devices == ["cuda:0"]
    assert plan.trainer_devices == ["cuda:1", "cuda:2", "cuda:3"]
    assert plan.engine_sleep is False


def test_auto_multi_gpu_tiny_model_stays_colocated(make_config):
    plan = plan_placement(make_config(cluster={"mode": "ray"}, trainer={"prompts_per_step": 4}), gpu_cluster(4))
    assert plan.placement == "colocated"
    assert plan.trainer_world_size == 4
    assert plan.trainer_devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert plan.model_estimate is not None and plan.model_estimate.params > 0


def test_auto_heuristic_splits_when_colocated_does_not_fit(make_config, llama_7b_dir):
    cfg = make_config(
        model={"name_or_path": str(llama_7b_dir), "dtype": "bfloat16"},
        rollout={"enable_sleep_mode": False},
        cluster={"mode": "ray"},
        trainer={"prompts_per_step": 4},
    )
    plan = plan_placement(cfg, gpu_cluster(4, memory_gib=40))
    assert plan.placement == "disaggregated"
    assert (plan.rollout_gpus, plan.trainer_gpus) == (1, 3)
    assert plan.weight_sync == "broadcast"
    assert any("splitting" in n for n in plan.notes)
    est = plan.model_estimate
    assert est is not None
    assert 6_000_000_000 < est.params < 7_500_000_000
    assert est.dtype_bytes == 2
    assert est.trainer_bytes == est.params * 16


def test_auto_heuristic_lora_fits_colocated(make_config, llama_7b_dir):
    cfg = make_config(
        model={"name_or_path": str(llama_7b_dir), "dtype": "bfloat16", "lora": {"enabled": True}},
        rollout={"enable_sleep_mode": False},
        cluster={"mode": "ray"},
        trainer={"prompts_per_step": 4},
    )
    plan = plan_placement(cfg, gpu_cluster(4, memory_gib=40))
    assert plan.placement == "colocated"
    assert plan.model_estimate is not None
    assert plan.model_estimate.trainer_bytes < plan.model_estimate.weight_bytes * 1.2


def test_auto_heuristic_unknown_model_defaults_colocated(make_config):
    cfg = make_config(model={"name_or_path": "/nonexistent/model-dir"}, cluster={"mode": "ray"}, trainer={"prompts_per_step": 4})
    plan = plan_placement(cfg, gpu_cluster(4))
    assert plan.placement == "colocated"
    assert plan.model_estimate is None
    assert any("unknown" in n for n in plan.notes)


def test_explicit_disaggregated_reports_idle_gpus(make_config):
    cfg = make_config(
        cluster={"mode": "ray", "placement": "disaggregated", "gpus_per_node": 4, "rollout_gpus": 2, "trainer_gpus": 1}
    )
    plan = plan_placement(cfg, gpu_cluster(4))
    assert plan.placement == "disaggregated"
    assert any("unused" in n for n in plan.notes)
    assert any("idle" in n for n in plan.notes)


def test_tensor_parallel_replicas(make_config):
    cfg = make_config(
        rollout={"tensor_parallel_size": 2, "data_parallel_size": 2},
        cluster={"mode": "ray", "placement": "colocated"},
        trainer={"prompts_per_step": 4},
    )
    plan = plan_placement(cfg, gpu_cluster(4))
    assert plan.engine_gpus == 4
    assert plan.rollout_devices == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    assert plan.engine_replicas == 2 and plan.engine_tensor_parallel == 2


def test_detected_gpus_divided_across_configured_nodes(make_config):
    cfg = make_config(cluster={"mode": "ray", "num_nodes": 2}, trainer={"prompts_per_step": 4})
    plan = plan_placement(cfg, gpu_cluster(4, ray_nodes=2))
    assert (plan.num_nodes, plan.gpus_per_node, plan.total_gpus) == (2, 2, 4)


# -- weight sync auto rules ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "gpus", "placement", "nodes", "expected"),
    [
        ("local", 0, "auto", 1, "inprocess"),
        ("local", 1, "auto", 1, "inprocess"),
        ("ray", 0, "auto", 1, "inprocess"),
        ("ray", 1, "auto", 1, "cuda_ipc"),
        ("ray", 4, "colocated", 1, "cuda_ipc"),
        ("ray", 4, "disaggregated", 1, "broadcast"),
        ("ray", 4, "colocated", 2, "checkpoint"),
        ("ray", 0, "disaggregated", 1, "inprocess"),
    ],
)
def test_weight_sync_auto_rules(make_config, mode, gpus, placement, nodes, expected):
    cluster_cfg = {"mode": mode, "placement": placement, "num_nodes": nodes}
    if placement == "disaggregated" and gpus:
        cluster_cfg.update(gpus_per_node=gpus // nodes, rollout_gpus=1, trainer_gpus=gpus - 1)
    cfg = make_config(cluster=cluster_cfg, trainer={"prompts_per_step": 4})
    plan = plan_placement(cfg, gpu_cluster(gpus, ray_nodes=nodes if mode == "ray" else 0))
    assert plan.weight_sync == expected


@pytest.mark.parametrize("name", ["checkpoint", "inprocess"])
def test_explicit_sync_names_pass_through(make_config, name):
    plan = plan_placement(make_config(cluster={"weight_sync": name}), gpu_cluster(0))
    assert plan.weight_sync == name


# -- errors --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "gpus", "match"),
    [
        ({"rollout": {"tensor_parallel_size": 2, "data_parallel_size": 2}}, 2, "only 2 available"),
        ({"rollout": {"tensor_parallel_size": 2}}, 0, "needs GPUs"),
        ({"rollout": {"tensor_parallel_size": 2}, "cluster": {"num_nodes": 2}}, 2, "cannot span nodes"),
        ({"cluster": {"placement": "disaggregated"}}, 0, "cluster.mode='ray'"),
        ({"cluster": {"placement": "disaggregated"}}, 2, "needs rollout_gpus and trainer_gpus"),
        ({"cluster": {"placement": "disaggregated", "rollout_gpus": 2, "trainer_gpus": 2}}, 2, "exceeds 2 GPUs"),
        ({"cluster": {"placement": "disaggregated", "rollout_gpus": 1, "trainer_gpus": 0}}, 2, "trainer_gpus >= 1"),
        ({"cluster": {"placement": "colocated", "rollout_gpus": 2}}, 2, "must equal tp\\*dp"),
        ({"cluster": {"weight_sync": "broadcast"}}, 0, "cluster.mode='ray'"),
        ({"cluster": {"weight_sync": "cuda_ipc"}}, 1, "cluster.mode='ray'"),
        ({"cluster": {"mode": "ray", "weight_sync": "cuda_ipc"}}, 0, "needs CUDA"),
        ({"cluster": {"mode": "ray", "weight_sync": "cuda_ipc", "num_nodes": 2}}, 2, "single node"),
        ({"cluster": {"weight_sync": "zmq"}}, 0, "unknown cluster.weight_sync"),
        ({"cluster": {"mode": "ray", "num_nodes": 3}}, 2, "exceeds the 2 detected"),
        ({"cluster": {"mode": "ray"}, "trainer": {"prompts_per_step": 1}}, 4, "exceeds sequences per step"),
    ],
)
def test_config_errors(make_config, overrides, gpus, match):
    cfg = make_config(**overrides)
    nodes = cfg.cluster.num_nodes if cfg.cluster.mode == "ray" else 0
    with pytest.raises(ConfigError, match=match):
        plan_placement(cfg, gpu_cluster(gpus, ray_nodes=nodes))


def test_rollout_gpus_below_tp_dp_rejected(make_config):
    cfg = make_config(
        rollout={"data_parallel_size": 2},
        cluster={"mode": "ray", "placement": "disaggregated", "rollout_gpus": 1, "trainer_gpus": 1},
        trainer={"prompts_per_step": 4},
    )
    with pytest.raises(ConfigError, match="rollout_gpus=1 <"):
        plan_placement(cfg, gpu_cluster(4, ray_nodes=1))


# -- cluster info / serialisation ----------------------------------------------------------


def test_cluster_num_gpus_ignores_non_cuda_ray_gpu_on_single_node():
    info = ClusterInfo(cpu_count=8, ray_connected=True, ray_resources={"GPU": 1.0}, ray_nodes=1)
    assert info.num_gpus == 0
    multi = ClusterInfo(cpu_count=8, ray_connected=True, ray_resources={"GPU": 16.0}, ray_nodes=2)
    assert multi.num_gpus == 16 and multi.num_nodes == 2
    assert gpu_cluster(2).gpu_memory_bytes == 80 * 2**30
    assert json.dumps(info.to_dict())


def test_plan_to_dict_is_json_serialisable(make_config):
    plan = plan_placement(make_config(), gpu_cluster(0))
    d = json.loads(json.dumps(plan.to_dict()))
    assert d["weight_sync"] == "inprocess"
    assert d["engine_gpus"] == 0
    assert d["model_estimate"]["params"] > 0


def test_estimate_model_reads_gqa_fields(make_config, llama_7b_dir):
    cfg = make_config(model={"name_or_path": str(llama_7b_dir), "dtype": "float16"})
    est = estimate_model(cfg, total_gpus=0)
    assert est is not None
    assert est.kv_bytes == 2 * 32 * 32 * 128 * 2 * cfg.trainer.max_seq_len * cfg.sequences_per_step()
    assert est.engine_bytes == est.weight_bytes + est.kv_bytes
