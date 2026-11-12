"""Supervised fine-tuning loop (``algorithm.name: sft``): dataset responses → token-exact
``TrainBatch`` → trainer → checkpoints. No rollout engine, rewards or environments are built.

Every sample is encoded as ``messages + {"role": "assistant", "content": response}`` with the
trainer tokenizer's chat template (and the row's tool schemas, ``metadata["tools"]``, when it has
any). ``algorithm.sft_turns`` picks the assistant turns that carry the loss: ``last`` (the final
one) or ``all`` (every assistant message not marked ``"weight": 0``); everything else is context.

A turn's loss span is found by token-exact prefix chaining: the ``add_generation_prompt=True``
rendering of the chat before the turn must be a prefix of the rendering that ends with the turn,
and the remainder is the span. Several turns share one sequence as long as each turn's prompt
starts with the sequence built so far, i.e. the template renders the earlier turns the same way
once more messages follow. Where it does not (a template that rewrites history), the turn starts
a sequence of its own, with the earlier turns as context only; where the prompt is not a prefix of
the closed turn at all, the turn's text is tokenised on its own and appended. Either way every
trained turn is a loss span in exactly one sequence, and the conversation counts under
``tokens/prefix_mismatch``. Batches are deterministic in ``(seed, step)`` through
:class:`~monorl.orchestrator.sampler.PromptSampler`, so resume only needs the step counter.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from monorl.core.batch import TrainBatch
from monorl.core.errors import ConfigError, DataError
from monorl.core.registry import registry
from monorl.core.reporting import list_checkpoints
from monorl.core.types import Completion, Message, RunResult, Sample, Trajectory, Turn, message_text
from monorl.orchestrator.loop import _checkpoint_complete
from monorl.orchestrator.sampler import PromptSampler
from monorl.orchestrator.sinks import MetricSinks
from monorl.rollout.render import render_chat

if TYPE_CHECKING:
    from monorl.config.schema import DataConfig, RunConfig
    from monorl.core.interfaces import Trainer
    from monorl.core.reporting import RunReporter


# -- sample → target -------------------------------------------------------------------------


def sft_response(sample: Sample, response_field: str | None = None) -> str:
    """Target text for ``sample``: ``metadata["response"]`` (set by the data layer from
    ``data.response_field`` / ``answer_field`` / the trailing assistant turn), else the raw
    ``response_field`` column kept in ``metadata["row"]``, else the trailing assistant turn's
    text, else ``sample.reference`` when it is a string. Raises :class:`DataError` otherwise."""
    meta = sample.metadata
    response = meta.get("response")
    if isinstance(response, str):
        return response
    row = meta.get("row")
    if response_field and isinstance(row, dict) and isinstance(row.get(response_field), str):
        return row[response_field]
    turn = meta.get("assistant_turn")
    if isinstance(turn, dict):
        text = message_text(turn)
        if text:
            return text
    if isinstance(sample.reference, str) and sample.reference:
        return sample.reference
    raise DataError(
        f"sample {sample.id!r} has no SFT target: set data.response_field, data.answer_field or "
        "include a trailing assistant turn in the messages"
    )


def sft_messages(sample: Sample, response_field: str | None = None) -> list[Message]:
    """Prompt messages followed by the assistant turn holding :func:`sft_response`."""
    return [*sample.messages, {"role": "assistant", "content": sft_response(sample, response_field)}]


def sft_tools(sample: Sample) -> list[dict[str, Any]] | None:
    """Tool schemas the sample's prompts are rendered with (``metadata["tools"]``, set by the data
    layer from the row's tools column), or ``None``."""
    return sample.metadata.get("tools") or None


def is_context_turn(message: Message) -> bool:
    """An assistant message a row excludes from training with ``"weight": 0`` (the OpenAI
    fine-tuning convention): it is context under ``algorithm.sft_turns: all``."""
    weight = message.get("weight")  # type: ignore[typeddict-item]
    return message.get("role") == "assistant" and weight is not None and float(weight) == 0.0


def trained_turns(messages: Sequence[Message], turns: str = "last") -> list[int]:
    """Indices of the assistant messages of ``messages`` that carry the loss: the final message
    (``last``), or every assistant message not marked ``"weight": 0`` plus the final one (``all``)."""
    if turns not in ("last", "all"):
        raise ValueError(f"unknown sft_turns {turns!r}; expected 'last' or 'all'")
    if not messages or messages[-1].get("role") != "assistant":
        raise DataError("SFT examples must end with an assistant message")
    last = len(messages) - 1
    if turns == "last":
        return [last]
    return [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "assistant" and (i == last or not is_context_turn(m))
    ]


def require_every_turn_trained(messages: Sequence[Message], *, backend: str, sample_id: str) -> None:
    """For pipeline backends whose own mechanism puts the loss on every assistant message of a
    conversation: fail on a conversation that marks one as context."""
    if any(is_context_turn(m) for m in messages[:-1]):
        raise ConfigError(
            f"sample {sample_id!r} marks an assistant turn as context ('weight': 0), which trainer.backend="
            f"{backend!r} cannot honour: its SFT loss covers every assistant turn of a conversation; use "
            "trainer.backend='reference' or drop the weights"
        )


def tools_survive_arrow(tools: Sequence[Any]) -> bool:
    """Whether per-row tool schemas (``None`` for a row without) read back unchanged from an Arrow
    column. Arrow gives all rows one struct type and fills the keys a row's schemas lack with
    nulls, which would reach the chat template; pipeline backends that load their SFT rows through
    Arrow check this before exporting a ``tools`` column."""
    import pyarrow as pa

    values = list(tools)
    try:
        return pa.array(values).to_pylist() == values
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return False


# -- encoding ---------------------------------------------------------------------------------


@dataclass(slots=True)
class LossSpan:
    """Token range ``[start, end)`` of one trained assistant turn in a sequence (clipped to the
    sequence when ``max_seq_len`` cut it; ``start == end`` when nothing of the turn is left)."""

    message_index: int
    start: int
    end: int


@dataclass(slots=True)
class EncodedExample:
    """One training sequence of a conversation."""

    input_ids: list[int]
    response_mask: list[int]
    prompt_len: int
    """Tokens before the first loss span (before truncation)."""
    prefix_mismatch: bool
    """A turn of this sequence was tokenised on its own instead of taken from the template."""
    truncated: bool
    spans: list[LossSpan] = field(default_factory=list)
    truncated_turns: int = 0
    """Trained turns that lost tokens to ``max_seq_len``."""

    @property
    def response_ids(self) -> list[int]:
        return [t for t, m in zip(self.input_ids, self.response_mask, strict=True) if m]


def _render(
    tokenizer: Any, messages: Sequence[Message], *, add_generation_prompt: bool, tools: Any = None
) -> str:
    if getattr(tokenizer, "chat_template", None):
        return render_chat(tokenizer, messages, tools, add_generation_prompt=add_generation_prompt)
    text = "\n".join(message_text(m) for m in messages)
    return text + "\n" if add_generation_prompt else text


def _ids(tokenizer: Any, text: str) -> list[int]:
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


def encode_sft_conversation(
    tokenizer: Any,
    messages: Sequence[Message],
    *,
    turns: str = "last",
    tools: Sequence[dict[str, Any]] | None = None,
    train_on_prompt: bool = False,
    max_seq_len: int | None = None,
) -> list[EncodedExample]:
    """Training sequences for a chat whose last message is an assistant turn (module docstring).

    For every trained turn (:func:`trained_turns`) the chat before it is rendered with
    ``add_generation_prompt=True`` exactly like a rollout prompt, and the chat up to and including
    it without one; ``tools`` go through the chat template in both. When the prompt ids are a
    token-exact prefix of the closed ids the remainder is the turn's loss span; otherwise the
    turn's text (plus EOS) is tokenised alone and appended to the prompt ids
    (``prefix_mismatch=True``). A turn extends the current sequence when its prompt starts with
    it and opens a new one otherwise, so the result is one sequence wherever the template allows.
    ``train_on_prompt`` marks every position that has a prediction (all but the first token);
    ``max_seq_len`` truncates from the right.
    """
    tools = list(tools) if tools else None
    sequences: list[tuple[list[int], list[LossSpan], bool]] = []
    ids: list[int] | None = None
    spans: list[LossSpan] = []
    mismatch = False
    for index in trained_turns(messages, turns):
        prompt_ids = _ids(
            tokenizer, _render(tokenizer, messages[:index], add_generation_prompt=True, tools=tools)
        )
        closed_ids = _ids(
            tokenizer, _render(tokenizer, messages[: index + 1], add_generation_prompt=False, tools=tools)
        )
        exact = closed_ids[: len(prompt_ids)] == prompt_ids and len(closed_ids) > len(prompt_ids)
        if not exact:
            response_ids = _ids(tokenizer, message_text(messages[index]))
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and (not response_ids or response_ids[-1] != int(eos)):
                response_ids.append(int(eos))
            closed_ids = prompt_ids + response_ids
        if ids is None or prompt_ids[: len(ids)] != ids:
            if ids is not None:
                sequences.append((ids, spans, mismatch))
            spans, mismatch = [], False
        ids = closed_ids
        spans.append(LossSpan(message_index=index, start=len(prompt_ids), end=len(closed_ids)))
        mismatch = mismatch or not exact
    assert ids is not None  # trained_turns never returns an empty list
    sequences.append((ids, spans, mismatch))

    examples: list[EncodedExample] = []
    for full_ids, full_spans, fallback in sequences:
        if train_on_prompt:
            mask = [0] + [1] * (len(full_ids) - 1)
        else:
            mask = [0] * len(full_ids)
            for span in full_spans:
                mask[span.start : span.end] = [1] * (span.end - span.start)
        limit = len(full_ids)
        truncated = max_seq_len is not None and limit > max_seq_len
        if truncated:
            limit = int(max_seq_len)  # type: ignore[arg-type]
            full_ids, mask = full_ids[:limit], mask[:limit]
        examples.append(
            EncodedExample(
                input_ids=full_ids,
                response_mask=mask,
                prompt_len=full_spans[0].start,
                prefix_mismatch=fallback,
                truncated=truncated,
                spans=[
                    LossSpan(message_index=s.message_index, start=min(s.start, limit), end=min(s.end, limit))
                    for s in full_spans
                ],
                truncated_turns=sum(1 for s in full_spans if s.end > limit),
            )
        )
    return examples


def encode_sft_example(
    tokenizer: Any,
    messages: Sequence[Message],
    *,
    tools: Sequence[dict[str, Any]] | None = None,
    train_on_prompt: bool = False,
    max_seq_len: int | None = None,
) -> EncodedExample:
    """Token ids and loss mask for a chat whose last message is the assistant target
    (``sft_turns: last``: one sequence, the final turn is the response span)."""
    (example,) = encode_sft_conversation(
        tokenizer,
        messages,
        turns="last",
        tools=tools,
        train_on_prompt=train_on_prompt,
        max_seq_len=max_seq_len,
    )
    return example


def build_sft_batch(
    examples: Sequence[EncodedExample],
    *,
    pad_token_id: int,
    group_ids: Sequence[str],
    meta: dict[str, Any] | None = None,
) -> tuple[TrainBatch, dict[str, float]]:
    """Right-padded :class:`TrainBatch` (zero advantages/rewards) plus ``tokens/*`` statistics.

    ``tokens/sequences`` is the row count, ``sft/turns`` the assistant turns that carry the loss,
    ``tokens/truncated`` the sequences cut at ``max_seq_len`` and ``tokens/truncated_turns`` the
    trained turns that lost tokens to it."""
    if not examples:
        raise ValueError("cannot build a batch from zero examples")
    batch_size = len(examples)
    width = max(len(e.input_ids) for e in examples)
    input_ids = torch.full((batch_size, width), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    response_mask = torch.zeros((batch_size, width), dtype=torch.long)
    for b, e in enumerate(examples):
        n = len(e.input_ids)
        input_ids[b, :n] = torch.tensor(e.input_ids, dtype=torch.long)
        attention_mask[b, :n] = 1
        response_mask[b, :n] = torch.tensor(e.response_mask, dtype=torch.long)
    batch = TrainBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        advantages=torch.zeros((batch_size, width), dtype=torch.float32),
        rewards=torch.zeros(batch_size, dtype=torch.float32),
        group_ids=list(group_ids),
        position_ids=(attention_mask.cumsum(dim=-1) - 1).clamp(min=0),
        meta=dict(meta or {}),
    )
    response_lengths = response_mask.sum(dim=-1).to(torch.float32)
    stats = {
        "tokens/response_mean": float(response_lengths.mean().item()),
        "tokens/response_max": float(response_lengths.max().item()),
        "tokens/prompt_mean": sum(e.prompt_len for e in examples) / batch_size,
        "tokens/seq_mean": float(attention_mask.sum(dim=-1).to(torch.float32).mean().item()),
        "tokens/prefix_mismatch": float(sum(1 for e in examples if e.prefix_mismatch)),
        "tokens/truncated": float(sum(1 for e in examples if e.truncated)),
        "tokens/truncated_turns": float(sum(e.truncated_turns for e in examples)),
        "tokens/sequences": float(batch_size),
        "sft/turns": float(sum(len(e.spans) for e in examples)),
    }
    return batch, stats


# -- loop -------------------------------------------------------------------------------------


class SFTLoop:
    def __init__(
        self, config: RunConfig, reporter: RunReporter, *, cancel: threading.Event, resume: bool
    ) -> None:
        self.config = config
        self.reporter = reporter
        self.cancel = cancel
        self.resume = resume
        self._last_checkpoint_step = 0
        self._eval_samples: list[Sample] | None = None
        self._sinks: MetricSinks | None = None
        self.trainer: Trainer | None = None
        self._truncation_logged = False

    # -- lifecycle ------------------------------------------------------------------------
    def run(self) -> RunResult:
        reporter = self.reporter
        try:
            self._setup()
            start_step = self._maybe_resume()
            last_step, last_metrics, cancelled = self._train_loop(start_step)
            final_path: str | None = None
            if cancelled:
                if last_step > self._last_checkpoint_step and last_step > 0:
                    self._save_checkpoint(last_step)
                state = "cancelled"
            else:
                reporter.log(f"exporting final policy to {reporter.final_dir}")
                self.trainer.save_pretrained(reporter.final_dir)
                final_path = str(reporter.final_dir)
                state = "succeeded"
            return RunResult(
                state=state,
                final_step=last_step,
                run_dir=str(reporter.run_dir),
                checkpoints=list_checkpoints(reporter.run_dir),
                final_model_path=final_path,
                metrics=dict(last_metrics),
            )
        finally:
            self._teardown()

    def _setup(self) -> None:
        from monorl.algorithms import select_loss_fn

        config = self.config
        reporter = self.reporter
        seed = int(config.seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.trainer = registry.get("trainer", config.trainer.backend)(
            config, loss_fn=select_loss_fn(config.algorithm)
        )
        tokenizer = getattr(self.trainer, "tokenizer", None)
        if tokenizer is None:
            from monorl.rollout.hf_engine import load_tokenizer

            tokenizer = load_tokenizer(config)
        self.tokenizer = tokenizer
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(tokenizer, "eos_token_id", None)
        self.pad_token_id = int(pad) if pad is not None else 0
        self.max_seq_len = int(getattr(self.trainer, "max_seq_len", None) or config.trainer.max_seq_len)
        reporter.log(f"sft: trainer backend {config.trainer.backend!r}, max_seq_len={self.max_seq_len}")

        source = _open_source()(config.data, tokenizer=tokenizer, seed=seed)
        self.samples = list(source)
        dropped = int(getattr(getattr(source, "mapper", None), "dropped", 0) or 0)
        if dropped:
            reporter.log(
                f"data: dropped {dropped} samples whose prompt exceeds data.max_prompt_tokens="
                f"{config.data.max_prompt_tokens}"
            )
        if not self.samples:
            raise DataError(f"no usable samples in {config.data.uri}")
        self.sampler = PromptSampler(
            self.samples,
            prompts_per_step=config.trainer.prompts_per_step,
            seed=seed,
            shuffle=config.data.shuffle,
        )
        reporter.log(
            f"data: {len(self.samples)} samples from {config.data.uri} "
            f"(loss on {config.algorithm.sft_turns} assistant turn(s))"
        )
        self._sinks = MetricSinks(config, reporter.run_dir, reporter)

    def _teardown(self) -> None:
        if self._sinks is not None:
            self._sinks.close()
        shutdown = getattr(self.trainer, "shutdown", None)
        if shutdown is not None:
            shutdown()

    def _maybe_resume(self) -> int:
        if not self.resume:
            return 0
        checkpoints = [p for p in list_checkpoints(self.reporter.run_dir) if _checkpoint_complete(Path(p))]
        if not checkpoints:
            self.reporter.log("resume requested but no complete checkpoint exists; starting from step 0")
            return 0
        latest = Path(checkpoints[-1])
        step = int(latest.name.removeprefix("step-"))
        self.trainer.load_checkpoint(latest)
        self._last_checkpoint_step = step
        random.seed(self.config.seed + step)
        torch.manual_seed(self.config.seed + step)
        self.reporter.log(f"resumed from {latest} at step {step}")
        return step

    # -- main loop ------------------------------------------------------------------------
    def _train_loop(self, start_step: int) -> tuple[int, dict[str, float], bool]:
        config = self.config
        max_steps = config.trainer.max_steps
        last_step = start_step
        last_metrics: dict[str, float] = {}
        self.reporter.set_status("running", f"sft steps {start_step + 1}..{max_steps}", step=start_step)
        for step in range(start_step + 1, max_steps + 1):
            if self.cancel.is_set():
                break
            t_step = time.perf_counter()
            samples = self.sampler.batch(step)
            batch, metrics, trajectories = self._encode(samples, step)
            metrics["data/epoch"] = float(self.sampler.epoch_of_step(step))
            t0 = time.perf_counter()
            train_metrics: dict[str, list[float]] = {}
            for _ in range(config.algorithm.num_epochs):
                out = self.trainer.train_step(batch)
                for k, v in out.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        train_metrics.setdefault(k, []).append(float(v))
            metrics.update({k: sum(v) / len(v) for k, v in train_metrics.items()})
            metrics["sft/tokens"] = float(batch.num_response_tokens)  # trainer averages it per micro-batch
            metrics["time/train"] = time.perf_counter() - t0
            metrics.setdefault(
                "lr", float(getattr(self.trainer, "learning_rate", config.trainer.learning_rate))
            )
            if step % config.trainer.save_every == 0:
                self._save_checkpoint(step)
            metrics["time/step"] = time.perf_counter() - t_step
            self._report(step, metrics, trajectories)
            if config.trainer.eval_every and step % config.trainer.eval_every == 0 and config.data.eval_uri:
                self._evaluate(step)
            last_step, last_metrics = step, metrics
            if self.cancel.is_set():
                break
        cancelled = self.cancel.is_set() and last_step < max_steps
        if cancelled:
            self.reporter.log(f"stopping after step {last_step} (cancel requested)")
        return last_step, last_metrics, cancelled

    def _encode(
        self, samples: Sequence[Sample], step: int
    ) -> tuple[TrainBatch, dict[str, float], list[Trajectory]]:
        """One or more training sequences per sample (:func:`encode_sft_conversation`); the rows
        of a sample share its group id. ``tokens/prefix_mismatch`` counts the samples that are not
        one token-exact sequence."""
        config = self.config
        examples: list[EncodedExample] = []
        group_ids: list[str] = []
        sample_ids: list[str] = []
        trajectories: list[Trajectory] = []
        mismatched = 0
        for i, sample in enumerate(samples):
            messages = sft_messages(sample, config.data.response_field)
            encoded = encode_sft_conversation(
                self.tokenizer,
                messages,
                turns=config.algorithm.sft_turns,
                tools=sft_tools(sample),
                train_on_prompt=config.algorithm.sft_train_on_prompt,
                max_seq_len=self.max_seq_len,
            )
            group_id = f"s{step}-p{i}"
            examples.extend(encoded)
            group_ids.extend([group_id] * len(encoded))
            sample_ids.extend([sample.id] * len(encoded))
            mismatched += len(encoded) > 1 or any(e.prefix_mismatch for e in encoded)
            trajectories.append(
                Trajectory(
                    sample=sample,
                    group_id=group_id,
                    messages=messages,
                    turns=[
                        Turn(
                            prompt_token_ids=example.input_ids[: span.start],
                            completion=Completion(
                                text=message_text(messages[span.message_index]),
                                token_ids=example.input_ids[span.start : span.end],
                                finish_reason="stop",
                            ),
                            message_index=span.message_index,
                        )
                        for example in encoded
                        for span in example.spans
                    ],
                )
            )
        batch, stats = build_sft_batch(
            examples,
            pad_token_id=self.pad_token_id,
            group_ids=group_ids,
            meta={"step": step, "sample_ids": sample_ids},
        )
        stats["tokens/prefix_mismatch"] = float(mismatched)
        if batch.num_response_tokens == 0:
            raise DataError(
                f"step {step}: no response tokens left after truncation to max_seq_len={self.max_seq_len}; "
                "raise trainer.max_seq_len or data.max_prompt_tokens"
            )
        if stats["tokens/truncated"] and not self._truncation_logged:
            self._truncation_logged = True
            self.reporter.log(
                f"step {step}: {int(stats['tokens/truncated'])} sequence(s) exceed trainer.max_seq_len="
                f"{self.max_seq_len} and were cut from the right ({int(stats['tokens/truncated_turns'])} "
                "assistant turn(s) lost loss tokens); see tokens/truncated and tokens/truncated_turns"
            )
        return batch, stats, trajectories

    def _save_checkpoint(self, step: int) -> None:
        path = self.reporter.checkpoint_path(step)
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        self.trainer.save_checkpoint(tmp)
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)  # a crash mid-save leaves only step-N.tmp, never a half-written step-N
        self._last_checkpoint_step = step
        self.reporter.log(f"saved checkpoint {path}")

    def _report(self, step: int, metrics: dict[str, float], trajectories: Sequence[Trajectory]) -> None:
        config = self.config
        log_cfg = config.logging
        is_last = step >= config.trainer.max_steps or self.cancel.is_set()
        if step % log_cfg.log_every == 0 or is_last:
            self.reporter.log_metrics(step, metrics)
            if self._sinks is not None:
                self._sinks.log(step, metrics)
            if log_cfg.save_trajectories and log_cfg.trajectories_per_step > 0 and trajectories:
                self.reporter.log_trajectories(
                    step,
                    trajectories,
                    max_items=log_cfg.trajectories_per_step,
                    phase="sft",
                    save_transcripts=log_cfg.save_transcripts,
                    success_threshold=log_cfg.success_threshold,
                )
            loss = metrics.get("loss/sft", float("nan"))
            ppl = metrics.get("sft/ppl", float("nan"))
            self.reporter.set_status(
                "running", f"step {step}/{config.trainer.max_steps} loss={loss:.4f} ppl={ppl:.2f}", step=step
            )
        else:
            self.reporter.set_status("running", step=step)

    # -- evaluation ----------------------------------------------------------------------
    def _evaluate(self, step: int) -> None:
        config = self.config
        if self._eval_samples is None:
            eval_cfg: DataConfig = config.data.model_copy(
                update={
                    "uri": config.data.eval_uri,
                    "split": config.data.eval_split,
                    "shuffle": False,
                    "eval_uri": None,
                }
            )
            self._eval_samples = list(_open_source()(eval_cfg, tokenizer=self.tokenizer, seed=config.seed))
            self.reporter.log(f"eval data: {len(self._eval_samples)} samples from {eval_cfg.uri}")
        if not self._eval_samples:
            return
        t0 = time.perf_counter()
        nll_sum, tokens, seq_nll_sum, seqs = 0.0, 0, 0.0, 0
        per_step = config.trainer.prompts_per_step
        for start in range(0, len(self._eval_samples), per_step):
            chunk = self._eval_samples[start : start + per_step]
            batch, _, _ = self._encode(chunk, step)
            logprobs = self.trainer.compute_logprobs(batch, reference=False)
            mask = batch.response_mask.to(logprobs.dtype)
            per_seq_tokens = mask.sum(dim=1)
            per_seq_nll = -(logprobs * mask).sum(dim=1)
            nll_sum += float(per_seq_nll.sum().item())
            tokens += int(per_seq_tokens.sum().item())
            scored = per_seq_tokens > 0
            seq_nll_sum += float((per_seq_nll[scored] / per_seq_tokens[scored]).sum().item())
            seqs += int(scored.sum().item())
        token_nll = nll_sum / max(tokens, 1)
        metrics = {
            "eval/loss": token_nll if config.algorithm.loss_agg == "token" else seq_nll_sum / max(seqs, 1),
            "eval/nll_token": token_nll,
            "eval/ppl": math.exp(min(token_nll, 80.0)),
            "eval/tokens": float(tokens),
            "eval/num_samples": float(len(self._eval_samples)),
            "eval/time": time.perf_counter() - t0,
        }
        self.reporter.log_metrics(step, metrics, phase="eval")
        if self._sinks is not None:
            self._sinks.log(step, metrics, phase="eval")
        self.reporter.log(f"eval step {step}: loss={metrics['eval/loss']:.4f} ppl={metrics['eval/ppl']:.2f}")


def _open_source() -> Any:
    from monorl.data import open_source

    return open_source
