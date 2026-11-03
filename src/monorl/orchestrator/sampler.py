"""Deterministic prompt batches as a pure function of ``(seed, step)``.

Resume only needs the step counter: step ``s`` (1-based) draws the window
``[(s-1)*P, s*P)`` of an infinite stream that walks the dataset epoch by epoch. Epoch 0 keeps the
source order (``open_source`` already shuffled with ``seed``); later epochs are permutations seeded
by ``seed`` and the epoch index. Resample rounds (DAPO dynamic sampling) draw random indices seeded
by ``(seed, step, round)`` so they never shift the main stream.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from monorl.core.types import Sample


class PromptSampler:
    def __init__(self, samples: Sequence[Sample], *, prompts_per_step: int, seed: int, shuffle: bool = True) -> None:
        if not samples:
            raise ValueError("PromptSampler needs at least one sample")
        self._samples = list(samples)
        self._per_step = int(prompts_per_step)
        self._seed = int(seed)
        self._shuffle = shuffle
        self._perm_cache: dict[int, list[int]] = {}

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def prompts_per_step(self) -> int:
        return self._per_step

    def epoch_of_step(self, step: int) -> int:
        return ((step - 1) * self._per_step) // len(self._samples)

    def batch(self, step: int, round: int = 0) -> list[Sample]:
        """Prompts for 1-based ``step``; ``round > 0`` gives an independent resample draw."""
        if step < 1:
            raise ValueError("steps are 1-based")
        n = len(self._samples)
        if round == 0:
            start = (step - 1) * self._per_step
            return [self._samples[self._index(start + i)] for i in range(self._per_step)]
        rng = random.Random(f"{self._seed}:{step}:{round}")
        return [self._samples[i] for i in rng.sample(range(n), min(self._per_step, n))]

    def _index(self, position: int) -> int:
        n = len(self._samples)
        epoch, offset = divmod(position, n)
        return self._permutation(epoch)[offset]

    def _permutation(self, epoch: int) -> list[int]:
        perm = self._perm_cache.get(epoch)
        if perm is None:
            perm = list(range(len(self._samples)))
            if self._shuffle and epoch > 0:
                random.Random(self._seed * 1_000_003 + epoch).shuffle(perm)
            if len(self._perm_cache) >= 2:
                self._perm_cache.clear()
            self._perm_cache[epoch] = perm
        return perm
