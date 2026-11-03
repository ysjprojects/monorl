"""Token-level tensor batch consumed by trainers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainBatch:
    """Right-padded ``[B, T]`` tensors. ``response_mask`` marks positions whose token was produced
    by the policy (loss positions); prompt and padding positions are 0.

    ``advantages`` is per-token (``[B, T]``); builders broadcast a per-sequence advantage across
    the response positions. ``old_logprobs``/``ref_logprobs`` are ``[B, T]`` aligned with
    ``input_ids`` (logprob of ``input_ids[:, t]`` given ``input_ids[:, :t]``), zero outside the
    response mask.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    group_ids: list[str]
    old_logprobs: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def to(self, device: torch.device | str) -> TrainBatch:
        def mv(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device)

        return TrainBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            response_mask=self.response_mask.to(device),
            advantages=self.advantages.to(device),
            rewards=self.rewards.to(device),
            group_ids=list(self.group_ids),
            old_logprobs=mv(self.old_logprobs),
            ref_logprobs=mv(self.ref_logprobs),
            position_ids=mv(self.position_ids),
            meta=dict(self.meta),
        )

    def slice(self, start: int, stop: int) -> TrainBatch:
        """Micro-batch view along the batch dimension."""

        def sl(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t[start:stop]

        return TrainBatch(
            input_ids=self.input_ids[start:stop],
            attention_mask=self.attention_mask[start:stop],
            response_mask=self.response_mask[start:stop],
            advantages=self.advantages[start:stop],
            rewards=self.rewards[start:stop],
            group_ids=self.group_ids[start:stop],
            old_logprobs=sl(self.old_logprobs),
            ref_logprobs=sl(self.ref_logprobs),
            position_ids=sl(self.position_ids),
            meta=dict(self.meta),
        )

    @property
    def num_response_tokens(self) -> int:
        return int(self.response_mask.sum().item())
