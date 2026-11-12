"""Token-exact :class:`~monorl.core.batch.TrainBatch` construction from trajectories.

A multi-turn trajectory becomes one sequence: the final turn's prompt ids followed by the final
completion ids. When each turn's prompt is a token-exact extension of the previous turn's prompt
plus completion, every completion span is a loss position; otherwise (chat template re-rendered
the earlier turns differently) only the final completion is trained on and the sequence is
counted under ``tokens/prefix_mismatch``.

Multimodal trajectories (image parts in the chat, ``processor`` given) are re-encoded through the
model's processor with the engine's recipe (:func:`monorl.rollout.media.vlm_encode`); the
resulting ``input_ids`` must equal the final turn's recorded prompt ids, which proves the pixel
tensors line up with the placeholder tokens. Media tensors go to ``TrainBatch.extra``
(``per_sequence`` + token-aligned ``[B, T]`` tensors such as ``mm_token_type_ids``) and
``position_ids`` is left ``None`` so the model derives its multimodal positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from monorl.core.batch import TrainBatch
from monorl.core.types import Message, Trajectory, messages_have_media


@dataclass(slots=True)
class FlatSequence:
    input_ids: list[int]
    response_mask: list[int]
    old_logprobs: list[float] | None
    """``None`` when any trained completion span lacks (or has misaligned) sampling logprobs."""
    prompt_len: int
    prefix_mismatch: bool

    def truncate(self, max_len: int) -> bool:
        if len(self.input_ids) <= max_len:
            return False
        self.input_ids = self.input_ids[:max_len]
        self.response_mask = self.response_mask[:max_len]
        if self.old_logprobs is not None:
            self.old_logprobs = self.old_logprobs[:max_len]
        self.prompt_len = min(self.prompt_len, max_len)
        return True


def prefix_chain_ok(trajectory: Trajectory) -> bool:
    """True when every turn's prompt starts with the previous turn's prompt + completion ids."""
    turns = trajectory.turns
    for k in range(1, len(turns)):
        prev = list(turns[k - 1].prompt_token_ids) + list(turns[k - 1].completion.token_ids)
        cur = turns[k].prompt_token_ids
        if len(cur) < len(prev) or list(cur[: len(prev)]) != prev:
            return False
    return True


def flatten_trajectory(trajectory: Trajectory) -> FlatSequence:
    turns = trajectory.turns
    if not turns:
        raise ValueError(f"trajectory {trajectory.group_id} has no turns")
    chain_ok = prefix_chain_ok(trajectory)
    last = turns[-1]
    input_ids = [int(t) for t in last.prompt_token_ids] + [int(t) for t in last.completion.token_ids]
    mask = [0] * len(input_ids)
    logprobs: list[float] = [0.0] * len(input_ids)
    have_logprobs = True
    for turn in turns if chain_ok else [last]:
        start = len(turn.prompt_token_ids)
        end = start + len(turn.completion.token_ids)
        mask[start:end] = [1] * (end - start)
        lp = turn.completion.logprobs
        if lp is None or len(lp) != len(turn.completion.token_ids):
            have_logprobs = False
        elif have_logprobs:
            logprobs[start:end] = [float(x) for x in lp]
    return FlatSequence(
        input_ids=input_ids,
        response_mask=mask,
        old_logprobs=logprobs if have_logprobs else None,
        prompt_len=len(turns[0].prompt_token_ids),
        prefix_mismatch=not chain_ok,
    )


def final_prompt_messages(trajectory: Trajectory) -> list[Message]:
    """The chat the final turn was prompted with: every message before the last assistant turn
    (environment messages appended after the final step are excluded)."""
    messages = list(trajectory.messages)
    last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=len(messages))
    return messages[:last_assistant]


@dataclass(slots=True)
class MediaEncoding:
    token_aligned: dict[str, torch.Tensor]
    """``[1, prompt_len]`` tensors aligned with the prompt ids (e.g. ``mm_token_type_ids``)."""
    media: dict[str, torch.Tensor]
    """Per-sequence media tensors (``pixel_values``, ``image_grid_thw``, ...)."""


def encode_trajectory_media(trajectory: Trajectory, processor: Any) -> MediaEncoding | None:
    """Processor tensors for a multimodal trajectory, or ``None`` for a text-only chat. Raises when
    the re-encoded prompt ids differ from the ids the engine conditioned on."""
    if not trajectory.turns or not messages_have_media(trajectory.messages):
        return None
    from monorl.rollout.media import split_processor_outputs, vlm_encode

    enc = vlm_encode(processor, final_prompt_messages(trajectory))
    ids = [int(t) for t in enc["input_ids"][0].tolist()]
    recorded = [int(t) for t in trajectory.turns[-1].prompt_token_ids]
    if ids != recorded:
        raise ValueError(
            f"trajectory {trajectory.group_id}: processor re-encoding of the final prompt ({len(ids)} ids) differs from "
            f"the engine's recorded prompt ids ({len(recorded)}); media tensors cannot be aligned with the sequence"
        )
    aligned, media = split_processor_outputs(enc)
    return MediaEncoding(token_aligned=aligned, media=media)


def build_train_batch(
    trajectories: Sequence[Trajectory],
    advantages: torch.Tensor | Sequence[float],
    *,
    pad_token_id: int,
    max_seq_len: int,
    meta: dict[str, Any] | None = None,
    processor: Any | None = None,
) -> tuple[TrainBatch, dict[str, float]]:
    """Right-padded batch with per-sequence advantages broadcast over response positions.

    ``old_logprobs`` is ``None`` when any trajectory lacks sampling logprobs (the caller then
    decides whether to recompute them). ``ref_logprobs`` is left for the caller. ``processor``
    (the model's ``AutoProcessor``) enables media tensors for multimodal trajectories (module
    docstring). Returns the batch and token statistics (``tokens/*``).
    """
    if not trajectories:
        raise ValueError("cannot build a batch from zero trajectories")
    adv = torch.as_tensor(advantages, dtype=torch.float32).reshape(-1)
    if adv.shape[0] != len(trajectories):
        raise ValueError(f"{adv.shape[0]} advantages for {len(trajectories)} trajectories")

    flats = [flatten_trajectory(t) for t in trajectories]
    truncated = sum(1 for f in flats if f.truncate(int(max_seq_len)))
    batch_size = len(flats)
    width = max(len(f.input_ids) for f in flats)

    input_ids = torch.full((batch_size, width), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    response_mask = torch.zeros((batch_size, width), dtype=torch.long)
    have_logprobs = all(f.old_logprobs is not None for f in flats)
    old_logprobs = torch.zeros((batch_size, width), dtype=torch.float32) if have_logprobs else None
    for b, f in enumerate(flats):
        n = len(f.input_ids)
        input_ids[b, :n] = torch.tensor(f.input_ids, dtype=torch.long)
        attention_mask[b, :n] = 1
        response_mask[b, :n] = torch.tensor(f.response_mask, dtype=torch.long)
        if old_logprobs is not None:
            old_logprobs[b, :n] = torch.tensor(f.old_logprobs, dtype=torch.float32)
    advantages_bt = adv[:, None] * response_mask.to(torch.float32)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

    extra: dict[str, Any] = {}
    media = [encode_trajectory_media(t, processor) for t in trajectories] if processor is not None else []
    if any(m is not None for m in media):
        extra["per_sequence"] = [None if m is None else m.media for m in media]
        for b, m in enumerate(media):
            for key, value in (m.token_aligned if m is not None else {}).items():
                if key not in extra:
                    extra[key] = torch.zeros((batch_size, width), dtype=value.dtype)
                n = min(int(value.shape[1]), len(flats[b].input_ids))  # prompt may have been truncated
                extra[key][b, :n] = value[0, :n]
        position_ids = None  # multimodal models derive their own (M-RoPE) positions

    batch = TrainBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        advantages=advantages_bt,
        rewards=torch.tensor([float(t.reward) for t in trajectories], dtype=torch.float32),
        group_ids=[t.group_id for t in trajectories],
        old_logprobs=old_logprobs,
        ref_logprobs=None,
        position_ids=position_ids,
        meta=dict(meta or {}),
        extra=extra,
    )
    response_lengths = response_mask.sum(dim=-1).to(torch.float32)
    stats = {
        "tokens/response_mean": float(response_lengths.mean().item()),
        "tokens/response_max": float(response_lengths.max().item()),
        "tokens/prompt_mean": sum(f.prompt_len for f in flats) / batch_size,
        "tokens/seq_mean": float(attention_mask.sum(dim=-1).to(torch.float32).mean().item()),
        "tokens/prefix_mismatch": float(sum(1 for f in flats if f.prefix_mismatch)),
        "tokens/truncated": float(truncated),
        "tokens/total": float(attention_mask.sum().item()),
    }
    return batch, stats
