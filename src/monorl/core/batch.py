"""Token-level tensor batch consumed by trainers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TrainBatch:
    """Right-padded ``[B, T]`` tensors. ``response_mask`` marks positions whose token was produced
    by the policy (loss positions); prompt and padding positions are 0.

    ``advantages`` is per-token (``[B, T]``); builders broadcast a per-trajectory advantage across
    the response positions. ``old_logprobs``/``ref_logprobs`` are ``[B, T]`` aligned with
    ``input_ids`` (logprob of ``input_ids[:, t]`` given ``input_ids[:, :t]``), zero outside the
    response mask.

    ``old_logprobs`` is the *proximal* policy of the clipped ratio. ``behavior_logprobs`` (same
    layout) is set only when that differs from the policy that sampled the tokens: it then holds
    the sampling log-probs, from which ``algorithm.off_policy`` importance weights and ratio masks
    are computed. ``behavior_top_ids``/``behavior_top_logprobs`` (``[B, T, K]``) are the sampler's
    ``K`` most likely tokens for position ``t`` and their log-probs, independent of the token that
    was sampled (score centering); entries beyond what the engine returned and positions outside
    the response mask hold id 0 and ``-inf``.

    ``loss_weights`` (``[B, T]``, zero outside the response mask) are per-token weights of the
    loss mean. Builders set them for ``algorithm.loss_agg="rollout"``, where one trajectory may
    span several rows: each of its tokens weighs one over its total number of response tokens, so
    every trajectory counts equally. ``None`` means the plain token or per-sequence mean.

    ``values``/``returns``/``value_mask`` (``[B, T]``) carry the learned value baseline of
    ``algorithm.name="ppo"``. ``value_mask`` marks the positions whose hidden state is the state of
    a credited step: the last prompt token before a turn's completion. There ``values`` holds the
    value head's estimate when the batch was built (the anchor of the clipped value loss) and
    ``returns`` its regression target; all three are zero elsewhere and ``None`` without a critic.

    ``extra`` carries additional model inputs (multimodal processor tensors). Every value is
    batch-aligned so :meth:`slice` stays correct: a tensor is indexed along dim 0 (e.g.
    ``mm_token_type_ids`` ``[B, T]``), a list is indexed like ``group_ids`` (e.g.
    ``extra["per_sequence"]``: one ``dict[str, Tensor] | None`` per sequence holding tensors whose
    first dim is *not* the batch — ``pixel_values``, ``image_grid_thw`` — which the trainer
    concatenates per micro-batch with :func:`monorl.rollout.media.collate_media`); anything else
    is shared and passed through as-is.
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
    loss_weights: torch.Tensor | None = None
    behavior_logprobs: torch.Tensor | None = None
    behavior_top_ids: torch.Tensor | None = None
    behavior_top_logprobs: torch.Tensor | None = None
    values: torch.Tensor | None = None
    returns: torch.Tensor | None = None
    value_mask: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

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
            loss_weights=mv(self.loss_weights),
            behavior_logprobs=mv(self.behavior_logprobs),
            behavior_top_ids=mv(self.behavior_top_ids),
            behavior_top_logprobs=mv(self.behavior_top_logprobs),
            values=mv(self.values),
            returns=mv(self.returns),
            value_mask=mv(self.value_mask),
            meta=dict(self.meta),
            extra={k: _move_extra(v, device) for k, v in self.extra.items()},
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
            loss_weights=sl(self.loss_weights),
            behavior_logprobs=sl(self.behavior_logprobs),
            behavior_top_ids=sl(self.behavior_top_ids),
            behavior_top_logprobs=sl(self.behavior_top_logprobs),
            values=sl(self.values),
            returns=sl(self.returns),
            value_mask=sl(self.value_mask),
            meta=dict(self.meta),
            extra={k: _slice_extra(v, start, stop) for k, v in self.extra.items()},
        )

    @property
    def num_response_tokens(self) -> int:
        return int(self.response_mask.sum().item())


def _slice_extra(value: Any, start: int, stop: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[start:stop]
    if isinstance(value, list):
        return value[start:stop]
    return value


def _move_extra(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_extra(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _move_extra(v, device) for k, v in value.items()}
    return value
