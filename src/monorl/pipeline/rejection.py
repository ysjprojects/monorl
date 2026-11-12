"""Sampling stages: rejection sampling (policy → scored completions → SFT dataset) and greedy eval.

Both build only a rollout engine (no trainer) from the registry, run episodes through the
orchestrator's parallel episode runner against the configured environment, and score them with the
configured rewards. The engine is shut down when the stage ends so the next stage can load its own
policy.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monorl.config.pipeline import EvalStage, RejectionSampleStage
from monorl.config.schema import DataConfig, RunConfig
from monorl.core.errors import DataError
from monorl.core.registry import registry
from monorl.core.reporting import CONFIG_FILE, RunReporter
from monorl.core.types import Message, RewardResult, Sample, SamplingParams, Trajectory
from monorl.orchestrator import loop as _loop
from monorl.orchestrator.episodes import EpisodeAborted, EpisodeSpec, run_episodes
from monorl.pipeline.stages import DATASET_FILE

ROW_SOURCE_SAMPLED = "sampled"
ROW_SOURCE_MIX = "mix"


class PolicySampler:
    """One rollout engine + env factory + reward function for a sampling stage.

    Component lookups go through the orchestrator's lazy resolvers so the same fakes that drive
    the orchestrator tests drive pipeline tests.
    """

    def __init__(self, config: RunConfig, *, reward_specs: Sequence[Any] | None = None) -> None:
        self.config = config
        engine_factory = registry.get("rollout_engine", config.rollout.backend)
        self.raw_engine = engine_factory(config, replica=0)
        extra = config.rollout.extra
        batching_cls = _loop._resolve_batching_engine()
        self.engine = batching_cls(
            self.raw_engine,
            max_batch_size=int(extra.get("batch_max_size", config.sequences_per_step())),
            max_wait_ms=float(extra.get("batch_max_wait_ms", 10.0)),
        )
        self.tokenizer = self.raw_engine.tokenizer
        self.max_parallel = int(extra.get("max_parallel_episodes", 64))
        specs = list(config.rewards) if reward_specs is None else list(reward_specs)
        self.reward_fn = _loop._resolve_build_reward()(specs, config=config) if specs else None
        make_env = _loop._resolve_make_env()
        self.env_factory: Callable[[], Any] = lambda: make_env(config)

    # -- data ---------------------------------------------------------------------------------
    def load_samples(self, data: DataConfig, *, limit: int | None = None) -> list[Sample]:
        samples = list(_loop._resolve_open_source()(data, tokenizer=self.tokenizer, seed=self.config.seed))
        if limit is not None:
            samples = samples[:limit]
        return samples

    # -- generation ---------------------------------------------------------------------------
    def episodes(
        self,
        samples: Sequence[Sample],
        *,
        n: int,
        params: SamplingParams,
        group_prefix: str,
        abort: threading.Event | None = None,
    ) -> list[Trajectory]:
        specs = [
            EpisodeSpec(sample=s, group_id=f"{group_prefix}-{s.id}", engine=self.engine)
            for s in samples
            for _ in range(n)
        ]
        return run_episodes(
            specs,
            env_factory=self.env_factory,
            tokenizer=self.tokenizer,
            params=params,
            max_parallel=self.max_parallel,
            abort=abort,
        )

    def score(self, trajectories: Sequence[Trajectory]) -> None:
        """Total reward = reward-function score + (discounted) env return, as in the RL loop."""
        if not trajectories:
            return
        gamma = self.config.algorithm.gamma
        if self.reward_fn is not None:
            results = list(self.reward_fn(trajectories))
            if len(results) != len(trajectories):
                raise DataError(f"reward returned {len(results)} results for {len(trajectories)} trajectories")
        else:
            results = [RewardResult(score=0.0) for _ in trajectories]
        for traj, res in zip(trajectories, results, strict=True):
            env_return = sum((gamma**k) * float(turn.reward) for k, turn in enumerate(traj.turns))
            breakdown = {k: float(v) for k, v in res.breakdown.items()}
            if any(turn.reward != 0.0 for turn in traj.turns):
                breakdown["env"] = env_return
            traj.reward = float(res.score) + env_return
            traj.reward_breakdown = breakdown
            if res.info:
                traj.info["reward_info"] = dict(res.info)

    def shutdown(self) -> None:
        for engine in (self.engine, self.raw_engine):
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is None:
                continue
            try:
                shutdown()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass


def sampling_params(config: RunConfig, *, greedy: bool) -> SamplingParams:
    r = config.rollout
    extra = r.extra
    if greedy:
        return SamplingParams(
            n=1, max_new_tokens=r.max_new_tokens, temperature=0.0, top_p=1.0, top_k=-1, stop=list(r.stop), logprobs=False
        )
    return SamplingParams(
        n=1,
        max_new_tokens=r.max_new_tokens,
        temperature=r.temperature,
        top_p=r.top_p,
        top_k=r.top_k,
        min_p=float(extra.get("min_p", 0.0)),
        repetition_penalty=float(extra.get("repetition_penalty", 1.0)),
        stop=list(r.stop),
        logprobs=False,
    )


def reward_stats(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    rewards = [float(t.reward) for t in trajectories]
    if not rewards:
        return {"reward/mean": 0.0, "reward/std": 0.0}
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    out = {"reward/mean": mean, "reward/std": math.sqrt(var)}
    names: dict[str, list[float]] = {}
    for t in trajectories:
        for k, v in t.reward_breakdown.items():
            names.setdefault(k, []).append(float(v))
    for k, vals in names.items():
        out[f"reward/{k}"] = sum(vals) / len(rewards)
    return out


def _chunks(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


# -- rejection sampling ---------------------------------------------------------------------


@dataclass
class RejectionOutcome:
    dataset: str
    prompts: int = 0
    sampled: int = 0
    accepted: int = 0
    mixed: int = 0
    rows: int = 0
    reward_mean: float = 0.0
    steps: int = 0
    cancelled: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "prompts": self.prompts,
            "sampled": self.sampled,
            "accepted": self.accepted,
            "accept_rate": (self.accepted / self.sampled) if self.sampled else 0.0,
            "mixed": self.mixed,
            "rows": self.rows,
            "reward/mean": self.reward_mean,
        }


@dataclass
class _Accepted:
    rows: list[dict[str, Any]] = field(default_factory=list)
    reward_sum: float = 0.0


def prompt_messages(trajectory: Trajectory) -> list[Message]:
    """The conversation up to the policy's first turn (what ``env.reset`` produced)."""
    for i, m in enumerate(trajectory.messages):
        if m.get("role") == "assistant":
            return list(trajectory.messages[:i])
    return list(trajectory.sample.messages)


def chat_row(messages: Sequence[Message], response: str, *, reference: Any, reward: float | None, source: str) -> dict[str, Any]:
    return {
        "messages": [*messages, {"role": "assistant", "content": response}],
        "reference": reference,
        "reward": reward,
        "source": source,
    }


def select_accepted(stage: RejectionSampleStage, trajectories: Sequence[Trajectory]) -> list[Trajectory]:
    """Apply the stage filters per prompt: reward threshold, finish, length, dedupe, cap."""
    by_sample: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        by_sample.setdefault(t.sample.id, []).append(t)
    kept: list[Trajectory] = []
    for group in by_sample.values():
        seen: set[str] = set()
        taken = 0
        for t in sorted(group, key=lambda x: -float(x.reward)):
            if taken >= stage.max_per_prompt:
                break
            if float(t.reward) < stage.min_reward:
                continue
            finish = t.turns[-1].completion.finish_reason if t.turns else "error"
            if stage.require_finish and finish in ("length", "error"):
                continue
            if stage.max_response_tokens is not None and t.num_response_tokens > stage.max_response_tokens:
                continue
            key = t.response_text.strip()
            if stage.dedupe:
                if key in seen:
                    continue
                seen.add(key)
            kept.append(t)
            taken += 1
    return kept


def sample_response(sample: Sample, data: DataConfig) -> str | None:
    """Target response of an SFT-style sample: ``metadata["response"]`` (response_field),
    the trailing assistant turn kept by the data layer, the raw answer column, else a string
    reference."""
    meta = sample.metadata
    value = meta.get("response")
    if isinstance(value, str) and value:
        return value
    turn = meta.get("assistant_turn")
    if isinstance(turn, dict):
        content = turn.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            text = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
            if text:
                return text
    row = meta.get("row")
    if isinstance(row, dict):
        for key in (data.response_field, data.answer_field):
            if key and isinstance(row.get(key), str) and row[key]:
                return row[key]
    raw = meta.get("answer_raw")
    if isinstance(raw, str) and raw:
        return raw
    if isinstance(sample.reference, str) and sample.reference:
        return sample.reference
    return None


def mix_data_config(base: DataConfig, uri: str, known: Sequence[DataConfig] = ()) -> DataConfig:
    """Mixed-in datasets reuse the field mapping of a data config the pipeline already describes
    for that uri (the sampling stage's own data first, then ``known``: base + train-stage
    overrides); otherwise only the prompt policy (system prompt, token limit) and auto-sniffed
    columns."""
    for candidate in (base, *known):
        if candidate.uri == uri:
            return candidate.model_copy(update={"shuffle": False, "eval_uri": None})
    return DataConfig(
        uri=uri,
        system_prompt=base.system_prompt,
        max_prompt_tokens=base.max_prompt_tokens,
        shuffle=False,
    )


def run_rejection_sampling(
    stage: RejectionSampleStage,
    config: RunConfig,
    *,
    run_dir: Path,
    cancel: threading.Event | None = None,
    output: Path | None = None,
    known_data: Sequence[DataConfig] = (),
) -> RejectionOutcome:
    """Sample ``num_samples`` completions per prompt, keep the ones passing the filters, append
    ``mix`` datasets and write ``dataset.jsonl`` (chat rows). Writes ``status.json``,
    ``metrics.jsonl`` (one row per prompt chunk) and ``trajectories.jsonl`` in ``run_dir``."""
    reporter = RunReporter(run_dir)
    config.to_yaml(reporter.run_dir / CONFIG_FILE)
    target = Path(output) if output is not None else reporter.run_dir / DATASET_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    outcome = RejectionOutcome(dataset=str(target))
    reporter.set_status("running", "loading policy", step=0)
    sampler = PolicySampler(config, reward_specs=None)
    try:
        data_cfg = stage.data or config.data
        samples = sampler.load_samples(data_cfg, limit=stage.max_prompts)
        if not samples:
            raise DataError(f"no usable prompts in {data_cfg.uri}")
        outcome.prompts = len(samples)
        reporter.log(f"rejection sampling: {len(samples)} prompts x {stage.num_samples} samples from {data_cfg.uri}")
        params = sampling_params(config, greedy=False)
        chunk_size = int(config.rollout.extra.get("rejection_prompts_per_step", config.trainer.prompts_per_step))
        chunks = _chunks(samples, chunk_size)
        accepted = _Accepted()
        tmp = target.with_name(target.name + ".tmp")
        with open(tmp, "w") as out:
            for step, chunk in enumerate(chunks, start=1):
                if cancel is not None and cancel.is_set():
                    outcome.cancelled = True
                    break
                reporter.set_status("running", f"sampling chunk {step}/{len(chunks)}", step=step - 1)
                t0 = time.perf_counter()
                try:
                    trajectories = sampler.episodes(
                        chunk, n=stage.num_samples, params=params, group_prefix=f"rs{step}", abort=cancel
                    )
                except EpisodeAborted:
                    outcome.cancelled = True
                    break
                sampler.score(trajectories)
                kept = select_accepted(stage, trajectories)
                for t in kept:
                    row = chat_row(
                        prompt_messages(t),
                        t.response_text,
                        reference=t.sample.reference,
                        reward=float(t.reward),
                        source=ROW_SOURCE_SAMPLED,
                    )
                    out.write(json.dumps(row, default=str) + "\n")
                    accepted.rows.append(row)
                    accepted.reward_sum += float(t.reward)
                outcome.sampled += len(trajectories)
                outcome.accepted += len(kept)
                outcome.steps = step
                stats = reward_stats(trajectories)
                metrics = {
                    "rs/prompts": float(len(chunk)),
                    "rs/sampled": float(len(trajectories)),
                    "rs/accepted": float(len(kept)),
                    "rs/accept_rate": len(kept) / len(trajectories) if trajectories else 0.0,
                    **stats,
                    "tokens/response_mean": (
                        sum(t.num_response_tokens for t in trajectories) / len(trajectories) if trajectories else 0.0
                    ),
                    "time/rollout": time.perf_counter() - t0,
                }
                reporter.log_metrics(step, metrics, phase="rejection")
                per_step = config.logging.trajectories_per_step
                if config.logging.save_trajectories and per_step > 0:
                    reporter.log_trajectories(step, trajectories, max_items=per_step)
                reporter.log(
                    f"chunk {step}/{len(chunks)}: accepted {len(kept)}/{len(trajectories)} "
                    f"(reward mean {stats['reward/mean']:.4f})"
                )
            if not outcome.cancelled:
                for uri in stage.mix:
                    added = _append_mix(out, sampler, mix_data_config(data_cfg, uri, known_data))
                    outcome.mixed += added
                    reporter.log(f"mixed in {added} rows from {uri}")
        if outcome.cancelled:
            tmp.unlink(missing_ok=True)
            reporter.set_status("cancelled", f"cancelled after chunk {outcome.steps}", step=outcome.steps)
            return outcome
        os.replace(tmp, target)
        outcome.rows = outcome.accepted + outcome.mixed
        outcome.reward_mean = accepted.reward_sum / outcome.accepted if outcome.accepted else 0.0
        if outcome.rows == 0:
            raise DataError(
                f"rejection sampling accepted 0 of {outcome.sampled} completions and no mix datasets were given; "
                f"lower min_reward or raise num_samples"
            )
        reporter.log_metrics(
            outcome.steps,
            {
                "rs/total_sampled": float(outcome.sampled),
                "rs/total_accepted": float(outcome.accepted),
                "rs/total_accept_rate": outcome.accepted / outcome.sampled if outcome.sampled else 0.0,
                "rs/mixed": float(outcome.mixed),
                "rs/rows": float(outcome.rows),
            },
            phase="summary",
        )
        reporter.set_status(
            "succeeded",
            f"accepted {outcome.accepted}/{outcome.sampled} completions (+{outcome.mixed} mixed) -> {target}",
            step=outcome.steps,
        )
        return outcome
    except BaseException as e:
        state = "cancelled" if isinstance(e, KeyboardInterrupt) else "failed"
        reporter.set_status(state, f"{type(e).__name__}: {e}", step=outcome.steps)
        raise
    finally:
        sampler.shutdown()


def _append_mix(out: Any, sampler: PolicySampler, data: DataConfig) -> int:
    added = 0
    for sample in sampler.load_samples(data):
        response = sample_response(sample, data)
        if response is None:
            continue
        row = chat_row(sample.messages, response, reference=sample.reference, reward=None, source=ROW_SOURCE_MIX)
        out.write(json.dumps(row, default=str) + "\n")
        added += 1
    return added


# -- eval -------------------------------------------------------------------------------------


@dataclass
class EvalOutcome:
    metrics: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    steps: int = 0
    cancelled: bool = False

    def summary(self) -> dict[str, Any]:
        return dict(self.metrics)


def eval_data_config(stage: EvalStage, config: RunConfig) -> DataConfig:
    if stage.data is not None:
        return stage.data
    data = config.data
    if data.eval_uri:
        return data.model_copy(update={"uri": data.eval_uri, "split": data.eval_split, "shuffle": False, "eval_uri": None})
    return data.model_copy(update={"shuffle": False})


def run_eval(
    stage: EvalStage, config: RunConfig, *, run_dir: Path, cancel: threading.Event | None = None
) -> EvalOutcome:
    """Greedy episodes over the eval dataset; logs ``eval/*`` metrics (phase ``eval``) and a final
    aggregate row."""
    reporter = RunReporter(run_dir)
    config.to_yaml(reporter.run_dir / CONFIG_FILE)
    outcome = EvalOutcome()
    reporter.set_status("running", "loading policy", step=0)
    sampler = PolicySampler(config)
    try:
        data_cfg = eval_data_config(stage, config)
        samples = sampler.load_samples(data_cfg, limit=stage.max_prompts)
        if not samples:
            raise DataError(f"no usable prompts in {data_cfg.uri}")
        reporter.log(f"eval: {len(samples)} prompts from {data_cfg.uri}")
        params = sampling_params(config, greedy=True)
        chunks = _chunks(samples, int(config.rollout.extra.get("eval_prompts_per_step", config.trainer.prompts_per_step)))
        scored: list[Trajectory] = []
        t_start = time.perf_counter()
        for step, chunk in enumerate(chunks, start=1):
            if cancel is not None and cancel.is_set():
                outcome.cancelled = True
                break
            reporter.set_status("running", f"evaluating chunk {step}/{len(chunks)}", step=step - 1)
            try:
                trajectories = sampler.episodes(chunk, n=1, params=params, group_prefix=f"eval{step}", abort=cancel)
            except EpisodeAborted:
                outcome.cancelled = True
                break
            sampler.score(trajectories)
            scored.extend(trajectories)
            outcome.steps = step
            per_step = config.logging.trajectories_per_step
            if config.logging.save_trajectories and per_step > 0:
                reporter.log_trajectories(step, trajectories, max_items=per_step)
            stats = reward_stats(trajectories)
            reporter.log_metrics(step, {"eval/chunk_reward_mean": stats["reward/mean"]}, phase="eval")
        if outcome.cancelled:
            reporter.set_status("cancelled", f"cancelled after chunk {outcome.steps}", step=outcome.steps)
            return outcome
        stats = reward_stats(scored)
        metrics = {
            "eval/reward_mean": stats.pop("reward/mean"),
            "eval/reward_std": stats.pop("reward/std"),
            "eval/num_samples": float(len(scored)),
            "eval/time": time.perf_counter() - t_start,
            **{f"eval/{k}": v for k, v in stats.items()},
        }
        outcome.metrics = metrics
        outcome.num_samples = len(scored)
        reporter.log_metrics(outcome.steps, metrics, phase="eval")
        reporter.set_status("succeeded", f"eval reward_mean={metrics['eval/reward_mean']:.4f}", step=outcome.steps)
        return outcome
    except BaseException as e:
        state = "cancelled" if isinstance(e, KeyboardInterrupt) else "failed"
        reporter.set_status(state, f"{type(e).__name__}: {e}", step=outcome.steps)
        raise
    finally:
        sampler.shutdown()


__all__ = [
    "EvalOutcome",
    "PolicySampler",
    "RejectionOutcome",
    "chat_row",
    "eval_data_config",
    "mix_data_config",
    "prompt_messages",
    "run_eval",
    "run_rejection_sampling",
    "sample_response",
    "sampling_params",
    "select_accepted",
]
