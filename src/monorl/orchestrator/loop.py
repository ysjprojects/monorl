"""Component-level RL loop: data → parallel episodes → rewards → advantages → TrainBatch →
trainer → weight sync → checkpoints, in synchronous or pipelined (async) form.

Sibling packages are imported lazily through the ``_resolve_*`` helpers at the bottom so the
orchestrator module itself stays import-light and tests can substitute fakes.
"""

from __future__ import annotations

import math
import os
import queue
import random
import shutil
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from monorl.core.errors import DataError
from monorl.core.registry import registry
from monorl.core.reporting import list_checkpoints
from monorl.core.types import RewardResult, RunResult, Sample, SamplingParams, Trajectory
from monorl.orchestrator.batching import build_train_batch
from monorl.orchestrator.episodes import EpisodeAborted, EpisodeSpec, run_episodes
from monorl.orchestrator.sampler import PromptSampler
from monorl.orchestrator.sinks import MetricSinks

if TYPE_CHECKING:
    from monorl.config.schema import DataConfig, RunConfig
    from monorl.core.interfaces import Environment, RewardFunction, RolloutEngine, Trainer, WeightSync
    from monorl.core.reporting import RunReporter


@dataclass(slots=True)
class RolloutBatch:
    """Scored trajectories for one step plus the policy version they were sampled from."""

    step: int
    trajectories: list[Trajectory]
    policy_version: int
    metrics: dict[str, float] = field(default_factory=dict)
    error: BaseException | None = None


_END = object()


class ComponentLoop:
    def __init__(self, config: RunConfig, reporter: RunReporter, *, cancel: threading.Event, resume: bool) -> None:
        self.config = config
        self.reporter = reporter
        self.cancel = cancel
        self.resume = resume
        self.policy_version = 0
        self.engine_lock = threading.Lock()
        self._last_checkpoint_step = 0
        self._eval_samples: list[Sample] | None = None
        self._runtime: Any = None
        self._sinks: MetricSinks | None = None

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
        config = self.config
        reporter = self.reporter
        seed = int(config.seed)
        random.seed(seed)
        torch.manual_seed(seed)

        runtime = _resolve_build_runtime()(config)
        self._runtime = runtime
        self.trainer: Trainer = runtime.trainer
        self.raw_engines: list[RolloutEngine] = list(runtime.engines)
        if not self.raw_engines:
            raise DataError("runtime produced no rollout engines")
        self.weight_sync: WeightSync = runtime.weight_sync
        reporter.log(f"runtime placement: {runtime.placement}")

        self.tokenizer = self.raw_engines[0].tokenizer
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", None)
        self.pad_token_id = int(pad) if pad is not None else 0
        # VLM checkpoints: the engine's processor if it has one (HF engine), else our own copy;
        # None for text-only models, which keeps the text batch path untouched.
        self.processor = getattr(self.raw_engines[0], "processor", None)
        if self.processor is None:
            from monorl.rollout.media import load_processor

            self.processor = load_processor(config)
        self.max_seq_len = int(getattr(self.trainer, "max_seq_len", None) or config.trainer.max_seq_len)

        extra = config.rollout.extra
        batching_cls = _resolve_batching_engine()
        self.engines: list[RolloutEngine] = [
            batching_cls(
                engine,
                max_batch_size=int(extra.get("batch_max_size", config.sequences_per_step())),
                max_wait_ms=float(extra.get("batch_max_wait_ms", 10.0)),
            )
            for engine in self.raw_engines
        ]
        self.max_parallel = int(extra.get("max_parallel_episodes", 64))

        self.samples = list(_resolve_open_source()(config.data, tokenizer=self.tokenizer, seed=seed))
        if not self.samples:
            raise DataError(f"no usable samples in {config.data.uri}")
        self.sampler = PromptSampler(
            self.samples, prompts_per_step=config.trainer.prompts_per_step, seed=seed, shuffle=config.data.shuffle
        )
        reporter.log(f"data: {len(self.samples)} samples from {config.data.uri}")

        self.reward_fn: RewardFunction | None = (
            _resolve_build_reward()(config.rewards, config=config) if config.rewards else None
        )
        make_env = _resolve_make_env()
        self.env_factory: Callable[[], Environment] = lambda: make_env(config)

        r = config.rollout
        self.sampling = SamplingParams(
            n=1,
            max_new_tokens=r.max_new_tokens,
            temperature=r.temperature,
            top_p=r.top_p,
            top_k=r.top_k,
            min_p=float(extra.get("min_p", 0.0)),
            repetition_penalty=float(extra.get("repetition_penalty", 1.0)),
            stop=list(r.stop),
            logprobs=True,
        )
        self.eval_sampling = SamplingParams(
            n=1, max_new_tokens=r.max_new_tokens, temperature=0.0, top_p=1.0, top_k=-1, stop=list(r.stop), logprobs=False
        )
        self._sinks = MetricSinks(config, reporter.run_dir, reporter)

    def _teardown(self) -> None:
        if self._sinks is not None:
            self._sinks.close()
        for engine in getattr(self, "engines", []):
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()  # stops the batching worker; forwards to the wrapped engine
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
        if self._runtime is not None:
            self._runtime.close()

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
        with self.engine_lock:
            self.weight_sync.sync(self.trainer, self.raw_engines)
            self.policy_version += 1
        self.reporter.log(f"resumed from {latest} at step {step}")
        return step

    # -- main loop ------------------------------------------------------------------------
    def _train_loop(self, start_step: int) -> tuple[int, dict[str, float], bool]:
        config = self.config
        max_steps = config.trainer.max_steps
        steps = range(start_step + 1, max_steps + 1)
        last_step = start_step
        last_metrics: dict[str, float] = {}
        self.reporter.set_status("running", f"training steps {steps.start}..{max_steps}", step=start_step)
        for item in self._rollouts(steps):
            if item.error is not None:
                raise item.error
            step = item.step
            t_step = time.perf_counter()
            metrics = self._train_on(item)
            if step % config.trainer.save_every == 0:
                self._save_checkpoint(step)
            metrics["time/step"] = time.perf_counter() - t_step
            self._report(step, metrics, item.trajectories)
            if config.trainer.eval_every and step % config.trainer.eval_every == 0 and config.data.eval_uri:
                self._evaluate(step)
            last_step, last_metrics = step, metrics
            if self.cancel.is_set():
                break
        cancelled = self.cancel.is_set() and last_step < max_steps
        if cancelled:
            self.reporter.log(f"stopping after step {last_step} (cancel requested)")
        return last_step, last_metrics, cancelled

    def _rollouts(self, steps: range) -> Iterator[RolloutBatch]:
        ahead = self.config.rollout.async_steps_ahead
        if ahead <= 0:
            for step in steps:
                if self.cancel.is_set():
                    return
                with self.engine_lock:
                    version = self.policy_version
                    rollout = self._rollout_step(step)
                rollout.policy_version = version
                yield rollout
            return

        # The producer may hold at most ``ahead + 1`` generated-but-untrained steps: the one the
        # consumer is training plus ``ahead`` queued. A permit is returned only after that step's
        # train + weight sync, so a batch generated at policy version v is trained at most
        # ``ahead`` syncs later (``policy/staleness`` <= ahead).
        permits = threading.Semaphore(ahead + 1)
        q: queue.Queue[Any] = queue.Queue(maxsize=ahead + 1)
        stop = threading.Event()

        def produce() -> None:
            try:
                for step in steps:
                    if stop.is_set() or self.cancel.is_set():
                        break
                    while not permits.acquire(timeout=0.1):
                        if stop.is_set() or self.cancel.is_set():
                            return
                    with self.engine_lock:
                        version = self.policy_version
                        try:
                            rollout = self._rollout_step(step, abort=stop)
                        except EpisodeAborted:
                            break
                    rollout.policy_version = version
                    if not _put_until(q, rollout, stop):
                        break
            except BaseException as e:  # noqa: BLE001 - surfaced to the consumer
                _put_until(q, RolloutBatch(step=-1, trajectories=[], policy_version=-1, error=e), stop)
            finally:
                _put_until(q, _END, stop)

        producer = threading.Thread(target=produce, name="monorl-rollout-producer", daemon=True)
        producer.start()
        try:
            while True:
                try:
                    item = q.get(timeout=0.2)
                except queue.Empty:
                    if self.cancel.is_set():
                        return
                    continue
                if item is _END:
                    return
                yield item
                permits.release()
        finally:
            stop.set()
            producer.join()

    # -- one step --------------------------------------------------------------------------
    def _rollout_step(self, step: int, *, abort: threading.Event | None = None) -> RolloutBatch:
        config = self.config
        t0 = time.perf_counter()
        trajectories = self._episodes(self.sampler.batch(step), step, round=0, abort=abort)
        time_rollout = time.perf_counter() - t0
        t1 = time.perf_counter()
        self._score(trajectories)
        time_reward = time.perf_counter() - t1
        metrics = {"rollout/groups_dropped": 0.0, "rollout/resample_rounds": 0.0}
        if config.algorithm.dynamic_sampling:
            kept, dropped = _drop_uniform_groups(trajectories)
            metrics["rollout/groups_dropped"] += dropped
            max_rounds = int(config.rollout.extra.get("dynamic_sampling_max_rounds", 3))
            rounds = 0
            while _num_groups(kept) < config.trainer.prompts_per_step and rounds < max_rounds:
                rounds += 1
                t0 = time.perf_counter()
                extra = self._episodes(self.sampler.batch(step, round=rounds), step, round=rounds, abort=abort)
                time_rollout += time.perf_counter() - t0
                t1 = time.perf_counter()
                self._score(extra)
                time_reward += time.perf_counter() - t1
                extra_kept, dropped = _drop_uniform_groups(extra)
                metrics["rollout/groups_dropped"] += dropped
                kept.extend(_take_groups(extra_kept, config.trainer.prompts_per_step - _num_groups(kept)))
            metrics["rollout/resample_rounds"] = float(rounds)
            trajectories = kept
        metrics["time/rollout"] = time_rollout
        metrics["time/reward"] = time_reward
        return RolloutBatch(step=step, trajectories=trajectories, policy_version=self.policy_version, metrics=metrics)

    def _episodes(
        self, prompts: Sequence[Sample], step: int, *, round: int, abort: threading.Event | None
    ) -> list[Trajectory]:
        group_size = self.config.rollout.group_size
        specs: list[EpisodeSpec] = []
        for p, sample in enumerate(prompts):
            group_id = f"s{step}-p{p}" if round == 0 else f"s{step}-r{round}-p{p}"
            for _ in range(group_size):
                engine = self.engines[len(specs) % len(self.engines)]
                specs.append(EpisodeSpec(sample=sample, group_id=group_id, engine=engine))
        return run_episodes(
            specs,
            env_factory=self.env_factory,
            tokenizer=self.tokenizer,
            params=self.sampling,
            max_parallel=self.max_parallel,
            abort=abort,
        )

    def _score(self, trajectories: Sequence[Trajectory]) -> None:
        if not trajectories:
            return
        algo = self.config.algorithm
        if self.reward_fn is not None:
            results = list(self.reward_fn(trajectories))
            if len(results) != len(trajectories):
                raise DataError(f"reward returned {len(results)} results for {len(trajectories)} trajectories")
        else:
            results = [RewardResult(score=0.0) for _ in trajectories]
        for traj, res in zip(trajectories, results, strict=True):
            env_return = sum((algo.gamma**k) * float(turn.reward) for k, turn in enumerate(traj.turns))
            breakdown = {k: float(v) for k, v in res.breakdown.items()}
            reward = float(res.score) + env_return
            if any(turn.reward != 0.0 for turn in traj.turns):
                breakdown["env"] = env_return
            if algo.overlong_penalty > 0 and traj.turns and traj.turns[-1].completion.finish_reason == "length":
                reward -= algo.overlong_penalty
                breakdown["overlong_penalty"] = -algo.overlong_penalty
            traj.reward = reward
            traj.reward_breakdown = breakdown
            if res.info:
                traj.info["reward_info"] = dict(res.info)

    def _train_on(self, item: RolloutBatch) -> dict[str, float]:
        config = self.config
        algo = config.algorithm
        step = item.step
        trajectories = item.trajectories
        metrics: dict[str, float] = dict(item.metrics)
        metrics["policy/staleness"] = float(self.policy_version - item.policy_version)
        metrics["rollout/num_sequences"] = float(len(trajectories))
        if not trajectories:
            self.reporter.log(f"step {step}: no trajectories left after dynamic sampling; skipping update")
            metrics.update({"rollout/skipped": 1.0, "time/train": 0.0, "time/sync": 0.0})
            metrics.setdefault("lr", float(getattr(self.trainer, "learning_rate", config.trainer.learning_rate)))
            return metrics
        metrics.update(_reward_metrics(trajectories))

        rewards = torch.tensor([t.reward for t in trajectories], dtype=torch.float32)
        advantages = _resolve_compute_advantages()(rewards, [t.group_id for t in trajectories], algorithm=algo)
        batch, token_stats = build_train_batch(
            trajectories,
            advantages,
            pad_token_id=self.pad_token_id,
            max_seq_len=self.max_seq_len,
            meta={"step": step, "policy_version": item.policy_version, "sample_ids": [t.sample.id for t in trajectories]},
            processor=self.processor,
        )
        metrics.update(token_stats)
        metrics["advantage/mean"] = float(batch.advantages.sum().item() / max(1, batch.num_response_tokens))
        metrics["advantage/abs_mean"] = float(batch.advantages.abs().sum().item() / max(1, batch.num_response_tokens))

        t0 = time.perf_counter()
        if batch.old_logprobs is None or not self._engine_logprobs_match_trainer():
            # Engines that sample through top-k/top-p/min-p/repetition-penalty processors may report
            # logprobs of the *truncated* distribution (vLLM processed_logprobs, SGLang), while the
            # trainer scores the full temperature-scaled softmax. Mixing them biases the importance
            # ratio below 1 even on-policy, so recompute old_logprobs with the trainer instead.
            batch.old_logprobs = self.trainer.compute_logprobs(batch, reference=False) * batch.response_mask
            metrics["tokens/old_logprobs_recomputed"] = 1.0
        if algo.kl_coef > 0:
            batch.ref_logprobs = self.trainer.compute_logprobs(batch, reference=True) * batch.response_mask
        train_metrics: dict[str, list[float]] = {}
        for _ in range(algo.num_epochs):
            out = self.trainer.train_step(batch)
            for k, v in out.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    train_metrics.setdefault(k, []).append(float(v))
        metrics.update({k: sum(v) / len(v) for k, v in train_metrics.items()})
        metrics["time/train"] = time.perf_counter() - t0
        metrics.setdefault("lr", float(getattr(self.trainer, "learning_rate", config.trainer.learning_rate)))

        t0 = time.perf_counter()
        with self.engine_lock:
            sync_metrics = self.weight_sync.sync(self.trainer, self.raw_engines) or {}
            self.policy_version += 1
        metrics.update({k: float(v) for k, v in sync_metrics.items() if isinstance(v, (int, float))})
        metrics["time/sync"] = time.perf_counter() - t0
        return metrics

    def _engine_logprobs_match_trainer(self) -> bool:
        ro = self.config.rollout
        if ro.backend == "hf":
            return True  # scores raw logits / temperature regardless of the sampling processors
        if ro.extra.get("trust_engine_logprobs") is not None:
            return bool(ro.extra["trust_engine_logprobs"])
        return ro.top_k <= 0 and ro.top_p >= 1.0 and not ro.extra.get("min_p") and not ro.extra.get("repetition_penalty")

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
                self.reporter.log_trajectories(step, trajectories, max_items=log_cfg.trajectories_per_step)
            reward = metrics.get("reward/mean", float("nan"))
            loss = metrics.get("loss/policy", float("nan"))
            self.reporter.set_status(
                "running",
                f"step {step}/{config.trainer.max_steps} reward={reward:.4f} loss={loss:.4f}",
                step=step,
            )
        else:
            self.reporter.set_status("running", step=step)

    # -- evaluation ----------------------------------------------------------------------
    def _evaluate(self, step: int) -> None:
        config = self.config
        if self._eval_samples is None:
            eval_cfg: DataConfig = config.data.model_copy(
                update={"uri": config.data.eval_uri, "split": config.data.eval_split, "shuffle": False, "eval_uri": None}
            )
            self._eval_samples = list(_resolve_open_source()(eval_cfg, tokenizer=self.tokenizer, seed=config.seed))
            self.reporter.log(f"eval data: {len(self._eval_samples)} samples from {eval_cfg.uri}")
        if not self._eval_samples:
            return
        t0 = time.perf_counter()
        with self.engine_lock:
            specs = [
                EpisodeSpec(sample=s, group_id=f"eval-s{step}-{s.id}", engine=self.engines[i % len(self.engines)])
                for i, s in enumerate(self._eval_samples)
            ]
            trajectories = run_episodes(
                specs,
                env_factory=self.env_factory,
                tokenizer=self.tokenizer,
                params=self.eval_sampling,
                max_parallel=self.max_parallel,
            )
        self._score(trajectories)
        reward_metrics = _reward_metrics(trajectories)
        metrics = {
            "eval/reward_mean": reward_metrics.pop("reward/mean"),
            "eval/reward_std": reward_metrics.pop("reward/std"),
        }
        metrics.update({f"eval/{k}": v for k, v in reward_metrics.items()})
        metrics["eval/num_samples"] = float(len(trajectories))
        metrics["eval/time"] = time.perf_counter() - t0
        self.reporter.log_metrics(step, metrics, phase="eval")
        if self._sinks is not None:
            self._sinks.log(step, metrics, phase="eval")
        self.reporter.log(f"eval step {step}: reward_mean={metrics['eval/reward_mean']:.4f}")


# -- helpers -------------------------------------------------------------------------------


def _reward_metrics(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    rewards = [float(t.reward) for t in trajectories]
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    metrics = {"reward/mean": mean, "reward/std": math.sqrt(var)}
    names: dict[str, list[float]] = {}
    for t in trajectories:
        for k, v in t.reward_breakdown.items():
            names.setdefault(k, []).append(float(v))
    for k, vals in names.items():
        # Keys present on a subset of trajectories (env return, overlong penalty) average over all
        # trajectories, counting absent entries as 0.
        metrics[f"reward/{k}"] = sum(vals) / n
    return metrics


def _groups(trajectories: Sequence[Trajectory]) -> dict[str, list[Trajectory]]:
    groups: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        groups.setdefault(t.group_id, []).append(t)
    return groups


def _checkpoint_complete(path: Path) -> bool:
    """Checkpoints are written to ``step-N.tmp`` and renamed, so any non-empty ``step-N`` is complete
    (a crash mid-save leaves only the ``.tmp`` directory, which ``list_checkpoints`` ignores)."""
    return path.is_dir() and any(path.iterdir())


def _num_groups(trajectories: Sequence[Trajectory]) -> int:
    return len({t.group_id for t in trajectories})


def _drop_uniform_groups(trajectories: Sequence[Trajectory]) -> tuple[list[Trajectory], int]:
    """DAPO dynamic sampling: drop groups whose rewards are all identical (zero gradient)."""
    kept: list[Trajectory] = []
    dropped = 0
    for members in _groups(trajectories).values():
        rewards = {round(m.reward, 12) for m in members}
        if len(members) > 1 and len(rewards) <= 1:
            dropped += 1
            continue
        kept.extend(members)
    return kept, dropped


def _take_groups(trajectories: Sequence[Trajectory], n: int) -> list[Trajectory]:
    out: list[Trajectory] = []
    for i, members in enumerate(_groups(trajectories).values()):
        if i >= n:
            break
        out.extend(members)
    return out


def _put_until(q: queue.Queue[Any], item: Any, stop: threading.Event) -> bool:
    """``q.put`` that gives up once ``stop`` is set (the consumer went away)."""
    while True:
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            if stop.is_set():
                return False


def format_exception(e: BaseException) -> str:
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))


# -- lazy sibling-package resolution (tests substitute fakes here) ---------------------------


def _resolve_build_runtime() -> Callable[..., Any]:
    from monorl.distributed import build_runtime

    return build_runtime


def _resolve_open_source() -> Callable[..., Any]:
    from monorl.data import open_source

    return open_source


def _resolve_build_reward() -> Callable[..., Any]:
    from monorl.rewards import build_reward

    return build_reward


def _resolve_compute_advantages() -> Callable[..., Any]:
    from monorl.algorithms import compute_advantages

    return compute_advantages


def _resolve_batching_engine() -> Callable[..., Any]:
    from monorl.rollout import BatchingEngine

    return BatchingEngine


def _resolve_make_env() -> Callable[..., Any]:
    try:
        from monorl.envs import make_env
    except ImportError:
        make_env = None
    if make_env is not None:
        return make_env
    return lambda config, **kw: registry.get("env", config.env.type)(config, **kw)
