"""Weight-sync strategies beyond the in-process hand-off (:class:`~monorl.distributed.runtime.InProcessWeightSync`).

* :class:`CheckpointWeightSync` — ``trainer.save_pretrained(dir)`` then
  ``engine.load_weights_from_path(dir)``; works everywhere a shared filesystem exists.
* :class:`BroadcastWeightSync` — a dedicated ``torch.distributed`` process group (gloo on CPU, NCCL
  on CUDA) with the trainer at rank 0 and one rank per engine (``1 + replica * tp``). Weights are
  streamed as buckets: a pickled header (names/shapes/dtypes) precedes every flat bucket, and an
  empty header terminates the stream, so nothing is materialised twice. The group is built lazily
  on first use and kept for the run. Engines in other processes take part through
  ``begin_receive_broadcast(spec, rank)`` (see :class:`BroadcastCapableEngine`); the receiving side
  calls :meth:`BroadcastWeightSync.receive` with its local engine.
* :class:`CudaIpcWeightSync` — same-node, cross-process zero-copy hand-off via CUDA IPC handles
  (``torch.multiprocessing.reductions.reduce_tensor``). Exported tensors are kept alive until every
  engine has consumed them. Engines in other processes implement ``update_weights_from_ipc(handles)``
  (see :class:`IpcCapableEngine`); an engine living in the exporting process gets the tensors
  directly, because CUDA refuses to open an IPC handle in the process that created it.
"""

from __future__ import annotations

import logging
import pickle
import shutil
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from monorl.core.errors import WeightSyncError
from monorl.core.interfaces import RolloutEngine, Trainer

log = logging.getLogger(__name__)


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


# -- checkpoint -------------------------------------------------------------------------------


class CheckpointWeightSync:
    """Export the trainer to ``<dir>/weights`` (atomically replaced) and reload it in every engine."""

    name = "checkpoint"

    def __init__(self, dir: str | Path) -> None:
        self.dir = Path(dir).expanduser()

    @property
    def weights_dir(self) -> Path:
        return self.dir / "weights"

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        t0 = time.perf_counter()
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".weights-{trainer.step}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        trainer.save_pretrained(tmp)
        if self.weights_dir.exists():
            shutil.rmtree(self.weights_dir)
        tmp.rename(self.weights_dir)
        total = sum(p.stat().st_size for p in self.weights_dir.rglob("*") if p.is_file())
        t_saved = time.perf_counter()
        for engine in engines:
            engine.load_weights_from_path(self.weights_dir)
        return {
            "weight_sync/bytes": float(total),
            "weight_sync/seconds": time.perf_counter() - t0,
            "weight_sync/save_seconds": t_saved - t0,
        }


# -- broadcast --------------------------------------------------------------------------------


@dataclass(frozen=True)
class BroadcastGroupSpec:
    """Rendezvous + transport parameters shared by every rank of a broadcast group."""

    master_addr: str
    master_port: int
    world_size: int
    backend: Literal["gloo", "nccl"] = "gloo"
    timeout_s: float = 600.0
    bucket_bytes: int = 256 << 20


class BroadcastReceiveHandle(Protocol):
    def result(self) -> dict[str, float]: ...


@runtime_checkable
class BroadcastCapableEngine(Protocol):
    """An engine living in another process that can join a broadcast group as ``rank``."""

    def begin_receive_broadcast(self, spec: BroadcastGroupSpec, rank: int) -> BroadcastReceiveHandle: ...


@dataclass(frozen=True)
class _BucketEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str


def _dtype_from_str(s: str) -> torch.dtype:
    dt = getattr(torch, s.removeprefix("torch."), None)
    if not isinstance(dt, torch.dtype):
        raise WeightSyncError(f"unknown dtype in broadcast header: {s!r}")
    return dt


class BroadcastWeightSync:
    """Rank 0 (trainer side) sends; ranks ``>= 1`` (engine side) receive. See module docstring."""

    name = "broadcast"

    def __init__(self, spec: BroadcastGroupSpec, *, rank: int, device: str | torch.device | None = None) -> None:
        if not 0 <= rank < spec.world_size:
            raise WeightSyncError(f"rank {rank} outside world_size {spec.world_size}")
        if spec.backend == "nccl" and not torch.cuda.is_available():
            raise WeightSyncError("broadcast backend 'nccl' needs CUDA")
        self.spec = spec
        self.rank = rank
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if spec.backend == "nccl" else torch.device("cpu")
        self.device = torch.device(device)
        self._store: Any = None
        self._group: Any = None

    # -- group lifecycle ----------------------------------------------------------------
    def _ensure_group(self) -> Any:
        if self._group is not None:
            return self._group
        import torch.distributed as dist

        timeout = timedelta(seconds=self.spec.timeout_s)
        self._store = dist.TCPStore(
            self.spec.master_addr,
            self.spec.master_port,
            self.spec.world_size,
            is_master=self.rank == 0,
            timeout=timeout,
            wait_for_workers=False,
        )
        if self.spec.backend == "gloo":
            self._group = dist.ProcessGroupGloo(self._store, self.rank, self.spec.world_size, timeout)
        else:
            opts = dist.ProcessGroupNCCL.Options()
            opts.is_high_priority_stream = False
            opts._timeout = timeout
            self._group = dist.ProcessGroupNCCL(self._store, self.rank, self.spec.world_size, opts)
        log.info("broadcast group ready: rank %d/%d via %s", self.rank, self.spec.world_size, self.spec.backend)
        return self._group

    def close(self) -> None:
        self._group = None
        self._store = None

    def _broadcast(self, tensor: torch.Tensor) -> None:
        import torch.distributed as dist

        opts = dist.BroadcastOptions()
        opts.rootRank = 0
        opts.rootTensor = 0
        self._ensure_group().broadcast([tensor], opts).wait()

    def _send_header(self, entries: list[_BucketEntry]) -> int:
        payload = pickle.dumps([(e.name, e.shape, e.dtype) for e in entries])
        self._broadcast(torch.tensor([len(payload)], dtype=torch.int64, device=self.device))
        if payload:
            self._broadcast(torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(self.device))
        return 8 + len(payload)

    def _recv_header(self) -> list[_BucketEntry]:
        n = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._broadcast(n)
        size = int(n.item())
        if size == 0:
            return []
        buf = torch.empty(size, dtype=torch.uint8, device=self.device)
        self._broadcast(buf)
        return [_BucketEntry(name, tuple(shape), dtype) for name, shape, dtype in pickle.loads(buf.cpu().numpy().tobytes())]

    # -- sender -----------------------------------------------------------------------------
    def send(self, weights: Iterable[tuple[str, torch.Tensor]]) -> dict[str, float]:
        """Stream ``weights`` to every other rank. Blocks until all ranks consumed each bucket."""
        if self.rank != 0:
            raise WeightSyncError("only rank 0 sends")
        t0 = time.perf_counter()
        total = 0
        buckets = 0
        entries: list[_BucketEntry] = []
        parts: list[torch.Tensor] = []
        pending_bytes = 0

        def flush() -> None:
            nonlocal total, buckets, entries, parts, pending_bytes
            if not parts:
                return
            total += self._send_header(entries)
            flat = torch.cat(parts) if len(parts) > 1 else parts[0]
            self._broadcast(flat)
            total += _tensor_bytes(flat)
            buckets += 1
            entries, parts, pending_bytes = [], [], 0

        for name, tensor in weights:
            t = tensor.detach().to(self.device, non_blocking=False).contiguous().reshape(-1)
            if parts and (t.dtype != parts[0].dtype or pending_bytes + _tensor_bytes(t) > self.spec.bucket_bytes):
                flush()
            entries.append(_BucketEntry(name, tuple(tensor.shape), str(tensor.dtype)))
            parts.append(t)
            pending_bytes += _tensor_bytes(t)
        flush()
        total += self._send_header([])
        return {
            "weight_sync/bytes": float(total),
            "weight_sync/seconds": time.perf_counter() - t0,
            "weight_sync/buckets": float(buckets),
        }

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        """Rank-0 entry point: start every engine's receiver, stream the trainer weights, join."""
        if self.rank != 0:
            raise WeightSyncError("BroadcastWeightSync.sync must run on rank 0 (the trainer side)")
        if not engines:
            raise WeightSyncError("BroadcastWeightSync.sync needs at least one engine")
        if (self.spec.world_size - 1) % len(engines):
            raise WeightSyncError(
                f"world_size {self.spec.world_size} is not 1 + k * {len(engines)} engines (k = ranks per engine)"
            )
        ranks_per_engine = (self.spec.world_size - 1) // len(engines)  # == tp when every TP worker joins
        handles: list[BroadcastReceiveHandle] = []
        for i, engine in enumerate(engines):
            if not isinstance(engine, BroadcastCapableEngine):
                raise WeightSyncError(
                    f"engine {type(engine).__name__} cannot receive broadcasts; it must live in another "
                    "process and implement begin_receive_broadcast(spec, rank)"
                )
            handles.append(engine.begin_receive_broadcast(self.spec, 1 + i * ranks_per_engine))
        metrics = self.send(trainer.export_weights())
        for h in handles:
            h.result()
        return metrics

    # -- receiver ---------------------------------------------------------------------------
    def _recv_stream(self, counter: list[int]) -> Iterator[tuple[str, torch.Tensor]]:
        while True:
            entries = self._recv_header()
            if not entries:
                return
            dtype = _dtype_from_str(entries[0].dtype)
            numel = sum(int(torch.Size(e.shape).numel()) for e in entries)
            flat = torch.empty(numel, dtype=dtype, device=self.device)
            self._broadcast(flat)
            counter[0] += _tensor_bytes(flat)
            counter[1] += 1
            offset = 0
            for e in entries:
                n = int(torch.Size(e.shape).numel())
                yield e.name, flat[offset : offset + n].view(e.shape)
                offset += n

    def receive(self, engine: RolloutEngine) -> dict[str, float]:
        """Rank ``>= 1`` entry point: consume one full weight stream into ``engine.update_weights``."""
        if self.rank == 0:
            raise WeightSyncError("rank 0 sends; call send()/sync()")
        t0 = time.perf_counter()
        counter = [0, 0]
        engine.update_weights(self._recv_stream(counter))
        return {
            "weight_sync/bytes": float(counter[0]),
            "weight_sync/seconds": time.perf_counter() - t0,
            "weight_sync/buckets": float(counter[1]),
        }


# -- cuda ipc ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class IpcHandle:
    """Picklable CUDA IPC descriptor of one exported tensor (``rebuild(*args)`` opens it)."""

    name: str
    rebuild: Callable[..., torch.Tensor]
    args: tuple[Any, ...]

    def open(self) -> torch.Tensor:
        return self.rebuild(*self.args)


@runtime_checkable
class IpcCapableEngine(Protocol):
    """An engine in another process on the same node that can open CUDA IPC handles."""

    def update_weights_from_ipc(self, handles: Sequence[IpcHandle]) -> None: ...


def open_ipc_handles(handles: Iterable[IpcHandle]) -> Iterator[tuple[str, torch.Tensor]]:
    """Consumer side: map IPC handles to live CUDA tensors (views of the exporter's memory)."""
    for h in handles:
        yield h.name, h.open()


class CudaIpcWeightSync:
    """Same-node zero-copy sync through CUDA IPC. See module docstring."""

    name = "cuda_ipc"

    def __init__(self, device: str | torch.device | None = None) -> None:
        if not torch.cuda.is_available():
            raise WeightSyncError("weight_sync='cuda_ipc' needs CUDA; use 'inprocess', 'broadcast' or 'checkpoint'")
        self.device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
        if self.device.type != "cuda":
            raise WeightSyncError(f"cuda_ipc device must be a CUDA device, got {self.device}")

    def export_handles(self, weights: Iterable[tuple[str, torch.Tensor]]) -> tuple[list[IpcHandle], list[torch.Tensor]]:
        """Pin every tensor on ``self.device`` and produce IPC handles.

        Returns ``(handles, keepalive)``; the caller must hold ``keepalive`` until consumers finished.
        """
        from torch.multiprocessing.reductions import reduce_tensor

        handles: list[IpcHandle] = []
        keepalive: list[torch.Tensor] = []
        for name, tensor in weights:
            t = tensor.detach()
            if t.device != self.device or not t.is_contiguous():
                t = t.to(self.device).contiguous()
            rebuild, args = reduce_tensor(t)
            handles.append(IpcHandle(name=name, rebuild=rebuild, args=tuple(args)))
            keepalive.append(t)
        torch.cuda.synchronize(self.device)
        return handles, keepalive

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        t0 = time.perf_counter()
        handles, keepalive = self.export_handles(trainer.export_weights())
        total = sum(_tensor_bytes(t) for t in keepalive)
        try:
            for engine in engines:
                if isinstance(engine, IpcCapableEngine):
                    engine.update_weights_from_ipc(handles)
                else:
                    engine.update_weights((h.name, t) for h, t in zip(handles, keepalive, strict=True))
            torch.cuda.synchronize(self.device)
        finally:
            del keepalive
        return {
            "weight_sync/bytes": float(total),
            "weight_sync/seconds": time.perf_counter() - t0,
            "weight_sync/tensors": float(len(handles)),
        }
