"""Fake rollout engine / trainer used by the distributed tests.

Importable as a registry provider (``register(registry)``) so Ray actors can load it via
``cluster.extra.providers = ["distributed_fakes"]``. Weights are two small tensors; the engine's
``generate`` output encodes the weight sum so a weight sync is observable end to end.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from monorl.core.batch import TrainBatch
from monorl.core.types import Completion, PromptInput, SamplingParams

SHAPES = {"model.embed.weight": (8, 4), "model.layers.0.proj.weight": (4, 4)}


def make_weights(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {name: torch.randn(shape, generator=g) for name, shape in SHAPES.items()}


def weight_sum(weights: dict[str, torch.Tensor]) -> float:
    return float(sum(t.double().sum().item() for t in weights.values()))


class FakeEngine:
    def __init__(self, seed: int = 0, *, replica: int = 0, **kw: Any) -> None:
        self.weights = make_weights(seed)
        self.replica = replica
        self.kwargs = kw
        self.slept = False
        self.shut_down = False
        self.updates = 0

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        tag = f"w={weight_sum(self.weights):.6f}"
        return [
            [
                Completion(
                    text=f"{tag} p={i} n={k}",
                    token_ids=[i, k],
                    logprobs=[0.0, 0.0] if params.logprobs else None,
                )
                for k in range(params.n)
            ]
            for i, _ in enumerate(prompts)
        ]

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in weights:
            self.weights[name].copy_(tensor.to(self.weights[name].device, copy=True))
        self.updates += 1

    def load_weights_from_path(self, path: str | Path) -> None:
        state = torch.load(Path(path) / "weights.pt")
        self.update_weights(state.items())

    def sleep(self) -> None:
        self.slept = True

    def wake_up(self) -> None:
        self.slept = False

    def shutdown(self) -> None:
        self.shut_down = True

    @property
    def tokenizer(self) -> Any:
        return {"name": "fake-tokenizer"}


class FakeTrainer:
    def __init__(self, seed: int = 1, *, rank: int = 0, world_size: int = 1, **kw: Any) -> None:
        self.weights = make_weights(seed)
        self.rank = rank
        self.world_size = world_size
        self.kwargs = kw
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def train_step(self, batch: TrainBatch) -> dict[str, float]:
        self._step += 1
        for t in self.weights.values():
            t.add_(0.5)
        return {"loss/policy": float(batch.advantages.sum().item()), "batch/size": float(len(batch))}

    def compute_logprobs(self, batch: TrainBatch, *, reference: bool = False) -> torch.Tensor:
        return torch.full(batch.input_ids.shape, -1.0 if reference else -0.5)

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        yield from self.weights.items()

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": self.weights, "step": self._step}, path / "ckpt.pt")

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path / "ckpt.pt")
        self.weights = state["weights"]
        self._step = int(state["step"])

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.weights, path / "weights.pt")


def register(registry: Any) -> None:
    registry.register("rollout_engine", "fake", lambda config, **kw: FakeEngine(0, **kw))
    registry.register("trainer", "fake", lambda config, **kw: FakeTrainer(1, **kw))


# -- helpers for multi-process weight-sync tests (importable by spawn) ------------------------


def make_broadcast_weights(seed: int) -> dict[str, torch.Tensor]:
    """Mixed dtypes and sizes so the stream needs several buckets."""
    g = torch.Generator().manual_seed(seed)
    return {
        "a.weight": torch.randn(16, 4, generator=g),
        "b.weight": torch.randn(4, 4, generator=g),
        "c.ids": torch.arange(10, dtype=torch.int64) * seed,
        "d.half": torch.randn(3, generator=g).to(torch.float16),
        "e.weight": torch.randn(32, 8, generator=g),
    }


class CollectorEngine(FakeEngine):
    """Records every (name, tensor) it is given, cloning so views into buckets stay valid."""

    def __init__(self) -> None:
        super().__init__(0)
        self.received: dict[str, torch.Tensor] = {}

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in weights:
            self.received[name] = tensor.clone()
        self.updates += 1


def broadcast_worker(rank: int, spec: Any, seed: int, out_path: str) -> None:
    from monorl.distributed.weight_sync import BroadcastWeightSync

    sync = BroadcastWeightSync(spec, rank=rank)
    if rank == 0:
        metrics = sync.send(make_broadcast_weights(seed).items())
    else:
        engine = CollectorEngine()
        metrics = sync.receive(engine)
        torch.save(engine.received, out_path)
    sync.close()
    Path(out_path + f".rank{rank}.json").write_text(repr(metrics))


def ipc_consumer(handles: list[Any], out_queue: Any) -> None:
    from monorl.distributed.weight_sync import open_ipc_handles

    out_queue.put({name: float(t.double().sum().item()) for name, t in open_ipc_handles(handles)})
