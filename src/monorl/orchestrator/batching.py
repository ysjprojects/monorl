"""Token-exact :class:`~monorl.core.batch.TrainBatch` construction from trajectories.

A trajectory becomes one training sequence per *token-contiguous chain* of turns. Turn ``k``
continues turn ``j`` when its prompt starts with turn ``j``'s prompt followed by turn ``j``'s
completion, token for token. The turns of an episode then form a forest, and every leaf yields one
sequence — the leaf's prompt plus completion — whose loss positions are the completion spans of the
turns on its root-to-leaf path. An episode whose turns all continue each other (the default,
``rollout.turn_rendering: tokens``) is therefore a single sequence. Contexts that do not continue
each other — a chat re-rendered with different tokens for earlier turns, a compacted history, a
sub-agent's own conversation — become separate sequences, so every sampled token is trained on
exactly once: a turn shared by several branches is a loss position in one of them and plain context
in the others. ``tokens/prefix_mismatch`` counts the trajectories that needed more than one
sequence.

Every sequence of a trajectory carries that trajectory's reward and group id, and every response
token the advantage of its trajectory or, with per-turn advantages
(``algorithm.credit_assignment: turn``), of its own turn, whichever sequence that turn is a loss
position in. :class:`TurnLayout` records where the turns sit: the response tokens of each and its
*state*, the last prompt token before its completion, where the value head of ``algorithm.name:
ppo`` is read. With ``algorithm.loss_agg: rollout`` the batch also gets ``loss_weights`` under
which each trajectory counts equally, however many sequences and response tokens it has.

Multimodal trajectories (image parts in the chat, ``processor`` given) are re-encoded through the
model's processor with the engine's recipe (:func:`monorl.rollout.media.vlm_encode`); the
resulting ``input_ids`` must equal the final turn's recorded prompt ids, which proves the pixel
tensors line up with the placeholder tokens. Those tensors belong to the final prompt only, so a
multimodal trajectory is always the single sequence of its final turn's chain. Media tensors go to
``TrainBatch.extra`` (``per_sequence`` + token-aligned ``[B, T]`` tensors such as
``mm_token_type_ids``) and ``position_ids`` is left ``None`` so the model derives its multimodal
positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from monorl.core.batch import TrainBatch
from monorl.core.types import Message, TopLogprobs, Trajectory, Turn, messages_have_media


@dataclass(slots=True)
class FlatSequence:
    input_ids: list[int]
    response_mask: list[int]
    old_logprobs: list[float] | None
    """``None`` when any trained completion span lacks (or has misaligned) sampling logprobs."""
    prompt_len: int
    """Length of the prompt of the first turn in this sequence's chain."""
    prefix_mismatch: bool
    """The trajectory has turns that are not part of this sequence's chain."""
    top_logprobs: list[tuple[int, TopLogprobs]] | None = None
    """``(start position, top-k)`` of every trained completion span; ``None`` when a span lacks
    the sampler's top-k."""
    spans: list[tuple[int, int, int]] = field(default_factory=list)
    """``(turn index, start, end)`` of every trained completion span (positions before truncation)."""

    def truncate(self, max_len: int) -> bool:
        if len(self.input_ids) <= max_len:
            return False
        self.input_ids = self.input_ids[:max_len]
        self.response_mask = self.response_mask[:max_len]
        if self.old_logprobs is not None:
            self.old_logprobs = self.old_logprobs[:max_len]
        self.prompt_len = min(self.prompt_len, max_len)
        return True


def continuation_parents(trajectory: Trajectory) -> list[int | None]:
    """For every turn, the latest earlier turn it continues token for token: its prompt starts with
    that turn's prompt ids followed by that turn's completion ids. ``None`` for a turn that starts
    a new context."""
    turns = trajectory.turns
    ends = [[*map(int, t.prompt_token_ids), *map(int, t.completion.token_ids)] for t in turns]
    parents: list[int | None] = []
    for k, turn in enumerate(turns):
        prompt = turn.prompt_token_ids
        parent = None
        for j in range(k - 1, -1, -1):
            n = len(ends[j])
            if n <= len(prompt) and list(prompt[:n]) == ends[j]:
                parent = j
                break
        parents.append(parent)
    return parents


def prefix_chain_ok(trajectory: Trajectory) -> bool:
    """True when every turn's prompt starts with the previous turn's prompt + completion ids."""
    return all(parent == k for k, parent in enumerate(continuation_parents(trajectory)[1:]))


def _chain(parents: Sequence[int | None], leaf: int) -> list[int]:
    """Turn indices from the root of ``leaf``'s context down to ``leaf``."""
    chain = [leaf]
    while (parent := parents[chain[-1]]) is not None:
        chain.append(parent)
    chain.reverse()
    return chain


def _flatten(
    turns: Sequence[Turn], chain: Sequence[int], trained: Sequence[int], *, prefix_mismatch: bool
) -> FlatSequence:
    """The sequence ending at ``chain[-1]`` with the completion spans of ``trained`` (a subset of
    ``chain``) as loss positions."""
    leaf = turns[chain[-1]]
    input_ids = [int(t) for t in leaf.prompt_token_ids] + [int(t) for t in leaf.completion.token_ids]
    mask = [0] * len(input_ids)
    logprobs: list[float] = [0.0] * len(input_ids)
    have_logprobs = True
    top: list[tuple[int, TopLogprobs]] | None = []
    spans: list[tuple[int, int, int]] = []
    for k in trained:
        turn = turns[k]
        start = len(turn.prompt_token_ids)
        end = start + len(turn.completion.token_ids)
        mask[start:end] = [1] * (end - start)
        spans.append((k, start, end))
        lp = turn.completion.logprobs
        if lp is None or len(lp) != end - start:
            have_logprobs = False
        elif have_logprobs:
            logprobs[start:end] = [float(x) for x in lp]
        tl = turn.completion.top_logprobs
        if end > start and (tl is None or len(tl) != end - start):
            top = None
        elif top is not None and tl is not None:
            top.append((start, tl))
    return FlatSequence(
        input_ids=input_ids,
        response_mask=mask,
        old_logprobs=logprobs if have_logprobs else None,
        prompt_len=len(turns[chain[0]].prompt_token_ids),
        prefix_mismatch=prefix_mismatch,
        top_logprobs=top,
        spans=spans,
    )


def trajectory_rows(trajectory: Trajectory) -> list[FlatSequence]:
    """One sequence per token-contiguous chain of turns (module docstring), in the order the
    chains end. Every turn is a loss position in exactly one sequence."""
    turns = trajectory.turns
    if not turns:
        raise ValueError(f"trajectory {trajectory.group_id} has no turns")
    parents = continuation_parents(trajectory)
    continued = {p for p in parents if p is not None}
    leaves = [k for k in range(len(turns)) if k not in continued]
    trained: set[int] = set()
    rows: list[FlatSequence] = []
    for leaf in leaves:
        chain = _chain(parents, leaf)
        rows.append(
            _flatten(turns, chain, [k for k in chain if k not in trained], prefix_mismatch=len(leaves) > 1)
        )
        trained.update(chain)
    return rows


def flatten_trajectory(trajectory: Trajectory) -> FlatSequence:
    """The single sequence of the final turn's chain, for consumers that take one sequence per
    trajectory; turns outside that chain are not trained on (``prefix_mismatch``)."""
    turns = trajectory.turns
    if not turns:
        raise ValueError(f"trajectory {trajectory.group_id} has no turns")
    chain = _chain(continuation_parents(trajectory), len(turns) - 1)
    return _flatten(turns, chain, chain, prefix_mismatch=len(chain) != len(turns))


def final_prompt_messages(trajectory: Trajectory) -> list[Message]:
    """The chat the final turn was prompted with: every message before the last assistant turn
    (environment messages appended after the final step are excluded)."""
    messages = list(trajectory.messages)
    last_assistant = max(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=len(messages)
    )
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


@dataclass(slots=True)
class TurnLayout:
    """Where the turns of a batch's trajectories sit in its ``[B, T]`` rows. Per-turn quantities
    are ``[N, K]`` tables (:mod:`monorl.algorithms.credit`): one row per trajectory, one column
    per turn, ``K`` the largest number of turns."""

    owner: torch.Tensor
    """``[B]``: the trajectory every row of the batch belongs to."""
    turn_mask: torch.Tensor
    """``[N, K]`` bool: trajectory ``i`` has a turn ``k``."""
    token_turn: torch.Tensor
    """``[B, T]``: ``i * K + k`` for a response token of turn ``k`` of trajectory ``i``, ``-1``
    elsewhere."""
    state_index: torch.Tensor
    """``[N, K]``: flat position ``b * T + t`` of a turn's state, the last prompt token before
    its completion in the row the turn is a loss position in; ``-1`` for a turn without one (an
    empty prompt, a turn outside the batch's rows or cut off by ``max_seq_len``)."""

    def per_token(self, per_turn: torch.Tensor) -> torch.Tensor:
        """``[B, T]``: every response token gets the ``[N, K]`` entry of its own turn, the rest 0."""
        flat = per_turn.to(torch.float32).reshape(-1)
        return flat[self.token_turn.clamp(min=0)] * (self.token_turn >= 0)

    def state_values(self, values: torch.Tensor) -> torch.Tensor:
        """``[N, K]``: the ``[B, T]`` entry at every turn's state, 0 for turns without one."""
        found = self.state_index >= 0
        return values.to(torch.float32).reshape(-1)[self.state_index.clamp(min=0)] * found

    def per_state(self, per_turn: torch.Tensor, select: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, T]``: the ``[N, K]`` entry of every turn at that turn's state, the rest 0.
        ``select`` (``[N, K]`` bool) restricts it to some turns."""
        chosen = self.state_index >= 0 if select is None else (self.state_index >= 0) & select.to(torch.bool)
        out = torch.zeros(self.token_turn.numel(), dtype=torch.float32)
        out[self.state_index[chosen]] = per_turn.to(torch.float32)[chosen]
        return out.view_as(self.token_turn)

    def state_mask(self, select: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, T]`` 0/1: the positions that are the state of a turn (of the ``select`` -ed turns)."""
        return self.per_state(torch.ones(self.state_index.shape), select).to(torch.long)


def build_train_batch(
    trajectories: Sequence[Trajectory],
    advantages: torch.Tensor | Sequence[float],
    *,
    pad_token_id: int,
    max_seq_len: int,
    meta: dict[str, Any] | None = None,
    processor: Any | None = None,
    loss_agg: str = "token",
) -> tuple[TrainBatch, dict[str, float]]:
    """Right-padded batch with the advantages spread over response positions.

    ``advantages`` has one value per trajectory (``[N]``), which all its response tokens carry, or
    one per turn (an ``[N, K]`` table, :mod:`monorl.algorithms.credit`), which the tokens of that
    turn carry. A trajectory may contribute several sequences (module docstring), so the batch can
    have more rows than there are trajectories.
    ``old_logprobs`` is ``None`` when any trajectory lacks sampling logprobs (the caller then
    decides whether to recompute them). ``ref_logprobs`` is left for the caller. ``processor``
    (the model's ``AutoProcessor``) enables media tensors for multimodal trajectories.
    ``loss_agg="rollout"`` sets ``loss_weights``: each response token weighs one over the number
    of response tokens its trajectory has in the batch. Returns the batch and token statistics
    (``tokens/*``).
    """
    adv = torch.as_tensor(advantages, dtype=torch.float32)
    per_turn = adv.ndim == 2
    if not per_turn:
        adv = adv.reshape(-1)
    if adv.shape[0] != len(trajectories):
        raise ValueError(f"{adv.shape[0]} advantages for {len(trajectories)} trajectories")
    batch, stats, layout = build_turn_batch(
        trajectories,
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
        meta=meta,
        processor=processor,
        loss_agg=loss_agg,
    )
    if per_turn:
        if adv.shape != layout.turn_mask.shape:
            raise ValueError(
                f"per-turn advantages have shape {tuple(adv.shape)}, the trajectories need "
                f"{tuple(layout.turn_mask.shape)}"
            )
        batch.advantages = layout.per_token(adv)
    else:
        batch.advantages = adv[layout.owner][:, None] * batch.response_mask.to(torch.float32)
    return batch, stats


def build_turn_batch(
    trajectories: Sequence[Trajectory],
    *,
    pad_token_id: int,
    max_seq_len: int,
    meta: dict[str, Any] | None = None,
    processor: Any | None = None,
    loss_agg: str = "token",
) -> tuple[TrainBatch, dict[str, float], TurnLayout]:
    """:func:`build_train_batch` before the advantages are known: the batch with zero
    ``advantages``, the token statistics and the :class:`TurnLayout` through which the caller
    assigns per-turn advantages and reads or places the value estimates of every turn's state."""
    if not trajectories:
        raise ValueError("cannot build a batch from zero trajectories")

    flats: list[FlatSequence] = []
    owners: list[int] = []  # sequence -> index of its trajectory
    media: list[MediaEncoding | None] = []
    split = 0
    for i, trajectory in enumerate(trajectories):
        encoding = encode_trajectory_media(trajectory, processor) if processor is not None else None
        rows = [flatten_trajectory(trajectory)] if encoding is not None else trajectory_rows(trajectory)
        split += int(rows[0].prefix_mismatch)
        flats.extend(rows)
        owners.extend([i] * len(rows))
        media.extend([encoding, *[None] * (len(rows) - 1)])
    truncated = sum(1 for f in flats if f.truncate(int(max_seq_len)))
    batch_size = len(flats)
    width = max(len(f.input_ids) for f in flats)
    owner = torch.tensor(owners, dtype=torch.long)

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
    top_k = (
        max((top.k for f in flats for _, top in f.top_logprobs), default=0)
        if all(f.top_logprobs is not None for f in flats)
        else 0
    )
    behavior_top_ids = behavior_top_logprobs = None
    if top_k > 0:
        behavior_top_ids = torch.zeros((batch_size, width, top_k), dtype=torch.long)
        behavior_top_logprobs = torch.full((batch_size, width, top_k), float("-inf"))
        for b, f in enumerate(flats):
            for start, top in f.top_logprobs:
                n = min(len(top), len(f.input_ids) - start)  # the sequence may have been truncated
                if n > 0:
                    ids = torch.frombuffer(top.token_ids, dtype=torch.int32).view(-1, top.k)
                    values = torch.frombuffer(top.logprobs, dtype=torch.float32).view(-1, top.k)
                    behavior_top_ids[b, start : start + n, : top.k] = ids[:n]
                    behavior_top_logprobs[b, start : start + n, : top.k] = values[:n]
    response = response_mask.to(torch.float32)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)

    max_turns = max(len(t.turns) for t in trajectories)
    turn_mask = torch.zeros((len(trajectories), max_turns), dtype=torch.bool)
    for i, trajectory in enumerate(trajectories):
        turn_mask[i, : len(trajectory.turns)] = True
    token_turn = torch.full((batch_size, width), -1, dtype=torch.long)
    state_index = torch.full((len(trajectories), max_turns), -1, dtype=torch.long)
    for b, f in enumerate(flats):
        n = len(f.input_ids)  # the sequence may have been truncated
        for k, start, end in f.spans:
            token_turn[b, start : min(end, n)] = owners[b] * max_turns + k
            if 1 <= start <= n:
                state_index[owners[b], k] = b * width + start - 1
    layout = TurnLayout(owner=owner, turn_mask=turn_mask, token_turn=token_turn, state_index=state_index)

    loss_weights = None
    if loss_agg == "rollout":
        per_trajectory = torch.zeros(len(trajectories)).index_add_(0, owner, response.sum(dim=-1))
        loss_weights = response / per_trajectory.clamp(min=1.0)[owner][:, None]

    extra: dict[str, Any] = {}
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
        advantages=torch.zeros((batch_size, width), dtype=torch.float32),
        rewards=torch.tensor([float(trajectories[i].reward) for i in owners], dtype=torch.float32),
        group_ids=[trajectories[i].group_id for i in owners],
        old_logprobs=old_logprobs,
        ref_logprobs=None,
        position_ids=position_ids,
        loss_weights=loss_weights,
        behavior_top_ids=behavior_top_ids,
        behavior_top_logprobs=behavior_top_logprobs,
        meta=dict(meta or {}),
        extra=extra,
    )
    response_lengths = response.sum(dim=-1)
    stats = {
        "tokens/response_mean": float(response_lengths.mean().item()),
        "tokens/response_max": float(response_lengths.max().item()),
        "tokens/prompt_mean": sum(f.prompt_len for f in flats) / batch_size,
        "tokens/seq_mean": float(attention_mask.sum(dim=-1).to(torch.float32).mean().item()),
        "tokens/sequences": float(batch_size),
        "tokens/prefix_mismatch": float(split),
        "tokens/truncated": float(truncated),
        "tokens/total": float(attention_mask.sum().item()),
    }
    return batch, stats, layout
