"""Placement planner: resolves ``cluster.placement/weight_sync = "auto"`` into a concrete plan.

:func:`plan_placement` is pure (config + :class:`ClusterInfo` in, :class:`PlacementPlan` out) so
every rule below is table-testable. Hard violations raise :class:`ConfigError`; heuristic
mismatches only add entries to ``plan.notes``.

Resource resolution
-------------------
* ``num_nodes = cluster.num_nodes`` (config). ``gpus_per_node`` is the configured value, or when it
  is ``0`` the detected count: ``ClusterInfo.num_gpus // num_nodes`` (local mode only sees this
  machine's CUDA devices). ``total_gpus = num_nodes * gpus_per_node``.
* The rollout engines need ``tensor_parallel_size * data_parallel_size`` GPUs when any GPU is
  present. ``tensor_parallel_size > 1`` requires GPUs and must not exceed ``gpus_per_node`` (TP never
  spans nodes).

Placement
---------
* ``total_gpus == 0``: ``colocated`` on cpu/mps — engines and trainer share the host, no sleep/wake,
  trainer world size 1. Explicit ``disaggregated`` without GPUs is only valid in ``ray`` mode (separate
  CPU actors); in ``local`` mode it is an error.
* ``colocated`` (explicit or chosen): engines take ``tp*dp`` GPUs starting at device 0; the trainer
  shares **all** GPUs (``trainer_world_size = total_gpus``); ``engine_sleep = rollout.enable_sleep_mode``
  so the engine releases memory while the trainer steps. ``rollout_gpus``/``trainer_gpus`` must be
  unset or consistent (``rollout_gpus == tp*dp``, ``trainer_gpus == total_gpus``).
* ``disaggregated`` (explicit or chosen): ``rollout_gpus >= tp*dp`` and ``trainer_gpus >= 1`` with
  ``rollout_gpus + trainer_gpus <= total_gpus``; engines use ``[0, rollout_gpus)``, the trainer
  ``[rollout_gpus, rollout_gpus + trainer_gpus)``, ``trainer_world_size = trainer_gpus``; no sleep.
  Unused GPUs or ``rollout_gpus`` not a multiple of ``tp`` are reported in ``notes``.
* ``auto``:
  1. no GPUs or a single GPU → ``colocated``;
  2. both ``rollout_gpus`` and ``trainer_gpus`` given → ``disaggregated`` with that split;
  3. otherwise use the model-size heuristic (below). If the colocated footprint fits per GPU, or the
     model size is unknown, → ``colocated``. Else try ``disaggregated`` with ``rollout_gpus = tp*dp``
     and ``trainer_gpus = total_gpus - rollout_gpus``; chosen when ``trainer_gpus >= 1`` and the
     trainer footprint fits, otherwise fall back to ``colocated`` with a note.

Model-size heuristic (``plan.model_estimate``)
----------------------------------------------
Read with ``transformers.AutoConfig`` (``text_config`` for multimodal wrappers). If the config
cannot be read or lacks the usual fields the estimate is ``None`` and heuristics are skipped.
``params ≈ vocab*hidden*(1 if tied else 2) + layers * (hidden*heads*head_dim*2 +
hidden*kv_heads*head_dim*2 + experts*3*hidden*intermediate)``. ``dtype_bytes`` follows
``model.dtype`` (``auto`` = 2 with GPUs, 4 without).
``engine_bytes = params*dtype_bytes + kv_bytes`` where ``kv_bytes = 2*layers*kv_heads*head_dim*
dtype_bytes * seq_len * sequences_per_step`` (``seq_len = rollout.max_model_len or
trainer.max_seq_len``). ``trainer_bytes = params*16`` for full fine-tuning (weights, grads, fp32
master + Adam moments) or ``params*dtype_bytes*1.1`` with LoRA. Per-GPU footprints divide by the
GPUs each role spans; colocated with sleep needs ``max(engine, trainer)`` per GPU, without sleep
their sum. "Fits" means ``<= 0.9 * ClusterInfo.gpu_memory_bytes``; when device memory is unknown
(remote-only GPUs) nothing is checked.

Weight sync
-----------
Explicit names are validated: ``inprocess`` works in local mode and in ray mode (where the Ray
runtime realises it through the object store); ``broadcast`` needs ``ray`` mode (one process cannot
hold two ranks); ``cuda_ipc`` needs ``ray`` mode, GPUs, ``colocated`` placement and a single node;
``checkpoint`` always works. ``auto``:
* ``local`` → ``inprocess``;
* ``ray`` with no GPUs → ``inprocess`` (object store);
* ``ray`` + ``colocated`` + single node → ``cuda_ipc``;
* ``ray`` + ``disaggregated`` → ``broadcast``;
* anything else (``ray`` + ``colocated`` across nodes) → ``checkpoint``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from monorl.core.errors import ConfigError
from monorl.distributed.cluster import ClusterInfo

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig

log = logging.getLogger(__name__)

SYNC_NAMES: tuple[str, ...] = ("inprocess", "checkpoint", "broadcast", "cuda_ipc")
MEMORY_SAFETY_FRACTION = 0.9
FULL_FINETUNE_BYTES_PER_PARAM = 16
LORA_OVERHEAD = 1.1


@dataclass(frozen=True)
class ModelEstimate:
    params: int
    dtype_bytes: int
    weight_bytes: int
    kv_bytes: int
    engine_bytes: int
    trainer_bytes: int


@dataclass
class PlacementPlan:
    mode: Literal["local", "ray"]
    placement: Literal["colocated", "disaggregated"]
    num_nodes: int
    gpus_per_node: int
    total_gpus: int
    rollout_gpus: int
    trainer_gpus: int
    engine_replicas: int
    engine_tensor_parallel: int
    trainer_world_size: int
    engine_sleep: bool
    weight_sync: str
    device: str
    rollout_devices: list[str]
    trainer_devices: list[str]
    model_estimate: ModelEstimate | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def engine_gpus(self) -> int:
        return self.engine_replicas * self.engine_tensor_parallel if self.total_gpus else 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["engine_gpus"] = self.engine_gpus
        return d


# -- model size ---------------------------------------------------------------------------


def _dtype_bytes(config: RunConfig, total_gpus: int) -> int:
    if config.model.dtype == "float32":
        return 4
    if config.model.dtype in ("bfloat16", "float16"):
        return 2
    return 2 if total_gpus > 0 else 4


def estimate_model(config: RunConfig, *, total_gpus: int, allow_network: bool = False) -> ModelEstimate | None:
    """Approximate parameter count and memory footprints from the HF config (see module doc).

    Reads local paths / the HF cache first; the hub is contacted only with ``allow_network``
    (the planner sets it when the heuristic can actually change the outcome: ``auto`` placement
    over more than one GPU).
    """
    try:
        from transformers import AutoConfig

        kwargs = {"revision": config.model.revision, "trust_remote_code": config.model.trust_remote_code}
        try:
            hf = AutoConfig.from_pretrained(config.model.name_or_path, local_files_only=True, **kwargs)
        except Exception:
            if not allow_network:
                raise
            hf = AutoConfig.from_pretrained(config.model.name_or_path, **kwargs)
    except Exception as e:  # noqa: BLE001 - heuristic input only; the plan records why it is missing
        log.info("model size unknown for %s: %s", config.model.name_or_path, e)
        return None
    hf = getattr(hf, "text_config", hf)
    hidden = getattr(hf, "hidden_size", None)
    layers = getattr(hf, "num_hidden_layers", None)
    vocab = getattr(hf, "vocab_size", None)
    heads = getattr(hf, "num_attention_heads", None)
    if not all(isinstance(v, int) and v > 0 for v in (hidden, layers, vocab, heads)):
        return None
    inter = getattr(hf, "intermediate_size", None) or 4 * hidden
    kv_heads = getattr(hf, "num_key_value_heads", None) or heads
    head_dim = getattr(hf, "head_dim", None) or hidden // heads
    experts = getattr(hf, "num_local_experts", None) or getattr(hf, "num_experts", None) or 1
    tied = bool(getattr(hf, "tie_word_embeddings", False))
    params = vocab * hidden * (1 if tied else 2) + layers * (
        2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim + experts * 3 * hidden * inter
    )
    dtype_bytes = _dtype_bytes(config, total_gpus)
    seq_len = config.rollout.max_model_len or config.trainer.max_seq_len
    kv_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes
    kv_bytes = kv_per_token * seq_len * config.sequences_per_step()
    weight_bytes = params * dtype_bytes
    if config.model.lora.enabled:
        trainer_bytes = int(weight_bytes * LORA_OVERHEAD)
    else:
        trainer_bytes = params * FULL_FINETUNE_BYTES_PER_PARAM
    return ModelEstimate(
        params=params,
        dtype_bytes=dtype_bytes,
        weight_bytes=weight_bytes,
        kv_bytes=kv_bytes,
        engine_bytes=weight_bytes + kv_bytes,
        trainer_bytes=trainer_bytes,
    )


def _fits(need_bytes: float, budget: int | None) -> bool | None:
    if budget is None:
        return None
    return need_bytes <= MEMORY_SAFETY_FRACTION * budget


# -- planning -----------------------------------------------------------------------------


def _resolve_gpus(config: RunConfig, cluster: ClusterInfo) -> tuple[int, int, int]:
    nodes = config.cluster.num_nodes
    gpn = config.cluster.gpus_per_node
    if gpn == 0 and cluster.num_gpus:
        gpn = cluster.num_gpus // nodes
        if gpn == 0:
            raise ConfigError(
                f"cluster.num_nodes={nodes} exceeds the {cluster.num_gpus} detected GPU(s); set cluster.gpus_per_node"
            )
    return nodes, gpn, nodes * gpn


def _device_strings(kind: str, start: int, count: int) -> list[str]:
    if kind != "cuda":
        return [kind] * max(1, count)
    return [f"cuda:{i}" for i in range(start, start + count)]


def _auto_placement(
    config: RunConfig, cluster: ClusterInfo, total_gpus: int, engine_gpus: int, est: ModelEstimate | None, notes: list[str]
) -> tuple[str, int, int]:
    c = config.cluster
    if total_gpus <= 1:
        return "colocated", engine_gpus, total_gpus
    if c.rollout_gpus is not None and c.trainer_gpus is not None:
        return "disaggregated", c.rollout_gpus, c.trainer_gpus
    if est is None:
        notes.append("model size unknown; defaulting to colocated placement")
        return "colocated", engine_gpus, total_gpus
    budget = cluster.gpu_memory_bytes
    engine_per_gpu = est.weight_bytes / config.rollout.tensor_parallel_size + est.kv_bytes / engine_gpus
    trainer_per_gpu = est.trainer_bytes / total_gpus
    colocated_need = (
        max(engine_per_gpu, trainer_per_gpu) if config.rollout.enable_sleep_mode else engine_per_gpu + trainer_per_gpu
    )
    fits = _fits(colocated_need, budget)
    if fits is None:
        notes.append("GPU memory unknown on this driver; defaulting to colocated placement")
        return "colocated", engine_gpus, total_gpus
    if fits:
        return "colocated", engine_gpus, total_gpus
    trainer_gpus = total_gpus - engine_gpus
    if trainer_gpus >= 1:
        disagg_trainer = est.trainer_bytes / trainer_gpus
        disagg_engine = est.weight_bytes / config.rollout.tensor_parallel_size + est.kv_bytes / engine_gpus
        if _fits(disagg_trainer, budget) and _fits(disagg_engine, budget):
            notes.append(
                f"colocated footprint {colocated_need / 2**30:.1f} GiB/GPU exceeds budget; "
                f"splitting {engine_gpus} rollout / {trainer_gpus} trainer GPUs"
            )
            return "disaggregated", engine_gpus, trainer_gpus
    notes.append(
        f"estimated footprint ({colocated_need / 2**30:.1f} GiB/GPU colocated) exceeds "
        f"{MEMORY_SAFETY_FRACTION:.0%} of device memory; proceeding colocated anyway"
    )
    return "colocated", engine_gpus, total_gpus


def _resolve_weight_sync(name: str, mode: str, placement: str, total_gpus: int, nodes: int) -> str:
    if name == "auto":
        if mode == "local" or total_gpus == 0:
            return "inprocess"
        if placement == "colocated" and nodes == 1:
            return "cuda_ipc"
        if placement == "disaggregated":
            return "broadcast"
        return "checkpoint"
    if name not in SYNC_NAMES:
        raise ConfigError(f"unknown cluster.weight_sync {name!r}; expected one of {('auto', *SYNC_NAMES)}")
    if name == "broadcast" and mode != "ray":
        raise ConfigError("weight_sync='broadcast' needs cluster.mode='ray' (one process cannot hold two ranks)")
    if name == "cuda_ipc":
        if mode != "ray":
            raise ConfigError("weight_sync='cuda_ipc' needs cluster.mode='ray' (engines in separate processes)")
        if total_gpus == 0:
            raise ConfigError("weight_sync='cuda_ipc' needs CUDA GPUs")
        if placement != "colocated" or nodes != 1:
            raise ConfigError("weight_sync='cuda_ipc' needs colocated placement on a single node")
    return name


def plan_placement(config: RunConfig, cluster: ClusterInfo) -> PlacementPlan:
    """Resolve placement, GPU split, replicas, trainer world size and weight-sync strategy."""
    c = config.cluster
    mode = c.mode
    tp = config.rollout.tensor_parallel_size
    dp = config.rollout.data_parallel_size
    nodes, gpus_per_node, total_gpus = _resolve_gpus(config, cluster)
    notes: list[str] = []

    if tp > 1 and total_gpus == 0:
        raise ConfigError(f"rollout.tensor_parallel_size={tp} needs GPUs; none configured or detected")
    if tp > gpus_per_node and total_gpus > 0:
        raise ConfigError(f"rollout.tensor_parallel_size={tp} exceeds gpus_per_node={gpus_per_node}; TP cannot span nodes")
    engine_gpus = tp * dp if total_gpus > 0 else 0
    if total_gpus and engine_gpus > total_gpus:
        raise ConfigError(
            f"rollout needs tensor_parallel_size*data_parallel_size = {engine_gpus} GPUs but only {total_gpus} available"
        )
    if c.placement == "disaggregated" and total_gpus == 0 and mode == "local":
        raise ConfigError("disaggregated placement without GPUs needs cluster.mode='ray' (separate CPU actors)")

    est = estimate_model(config, total_gpus=total_gpus, allow_network=c.placement == "auto" and total_gpus > 1)

    if c.placement == "colocated":
        placement, rollout_gpus, trainer_gpus = "colocated", engine_gpus, total_gpus
        if c.rollout_gpus is not None and c.rollout_gpus != engine_gpus:
            raise ConfigError(f"colocated placement: rollout_gpus={c.rollout_gpus} must equal tp*dp={engine_gpus}")
        if c.trainer_gpus is not None and c.trainer_gpus != total_gpus:
            raise ConfigError(f"colocated placement: trainer_gpus={c.trainer_gpus} must equal total GPUs={total_gpus}")
    elif c.placement == "disaggregated":
        if total_gpus == 0:
            placement, rollout_gpus, trainer_gpus = "disaggregated", 0, 0
        else:
            if c.rollout_gpus is None or c.trainer_gpus is None:
                raise ConfigError("disaggregated placement needs rollout_gpus and trainer_gpus")
            placement, rollout_gpus, trainer_gpus = "disaggregated", c.rollout_gpus, c.trainer_gpus
    else:
        placement, rollout_gpus, trainer_gpus = _auto_placement(config, cluster, total_gpus, engine_gpus, est, notes)

    if placement == "disaggregated" and total_gpus > 0:
        if rollout_gpus < engine_gpus:
            raise ConfigError(f"rollout_gpus={rollout_gpus} < tensor_parallel_size*data_parallel_size={engine_gpus}")
        if trainer_gpus < 1:
            raise ConfigError("disaggregated placement needs trainer_gpus >= 1")
        if rollout_gpus + trainer_gpus > total_gpus:
            raise ConfigError(f"rollout_gpus + trainer_gpus = {rollout_gpus + trainer_gpus} exceeds {total_gpus} GPUs")
        if rollout_gpus > engine_gpus:
            notes.append(f"{rollout_gpus - engine_gpus} rollout GPU(s) unused by tp={tp} x dp={dp}")
        if rollout_gpus + trainer_gpus < total_gpus:
            notes.append(f"{total_gpus - rollout_gpus - trainer_gpus} GPU(s) idle")

    if total_gpus == 0:
        device = config.rollout.device or ("mps" if cluster.mps_available else "cpu")
        trainer_device = config.trainer.device or device
        rollout_devices = [device] * dp
        trainer_devices = [trainer_device]
        trainer_world_size = 1
        engine_sleep = False
    else:
        device = "cuda"
        rollout_devices = _device_strings("cuda", 0, engine_gpus)
        trainer_start = 0 if placement == "colocated" else rollout_gpus
        trainer_devices = _device_strings("cuda", trainer_start, trainer_gpus)
        trainer_world_size = trainer_gpus
        engine_sleep = placement == "colocated" and config.rollout.enable_sleep_mode
        if trainer_world_size > config.sequences_per_step():
            raise ConfigError(
                f"trainer world size {trainer_world_size} exceeds sequences per step "
                f"{config.sequences_per_step()}; raise prompts_per_step/group_size or shrink trainer_gpus"
            )

    weight_sync = _resolve_weight_sync(c.weight_sync, mode, placement, total_gpus, nodes)
    if est is not None and total_gpus and cluster.gpu_memory_bytes is not None:
        per_gpu = est.trainer_bytes / max(1, trainer_gpus)
        if _fits(per_gpu, cluster.gpu_memory_bytes) is False:
            notes.append(f"trainer footprint {per_gpu / 2**30:.1f} GiB/GPU exceeds device memory")

    return PlacementPlan(
        mode=mode,
        placement=placement,  # type: ignore[arg-type]
        num_nodes=nodes,
        gpus_per_node=gpus_per_node,
        total_gpus=total_gpus,
        rollout_gpus=rollout_gpus,
        trainer_gpus=trainer_gpus,
        engine_replicas=dp,
        engine_tensor_parallel=tp,
        trainer_world_size=trainer_world_size,
        engine_sleep=engine_sleep,
        weight_sync=weight_sync,
        device=device,
        rollout_devices=rollout_devices,
        trainer_devices=trainer_devices,
        model_estimate=est,
        notes=notes,
    )
