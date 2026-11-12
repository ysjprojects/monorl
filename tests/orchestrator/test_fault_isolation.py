"""Failed episodes: retries, leaving them out of a batch, and what is never tolerated."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import fakes
import pytest

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, GenerationFailed
from monorl.core.registry import registry
from monorl.core.reporting import iter_metrics
from monorl.core.types import Completion, Sample, SamplingParams, StepResult
from monorl.orchestrator import run
from monorl.orchestrator.episodes import EpisodeAborted, EpisodeFailure, EpisodeSpec, run_episodes


class FlakyEnv(fakes.FakeMultiTurnEnv):
    """``env.params.broken``: the numbers (1-based, in creation order) of the environment instances
    whose ``step`` fails. Every attempt at an episode creates a new instance."""

    created = 0
    lock = threading.Lock()

    def __init__(self, config: RunConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        with FlakyEnv.lock:
            FlakyEnv.created += 1
            self.broken = FlakyEnv.created in set(config.env.params.get("broken", ()))

    def step(self, action: Completion) -> StepResult:
        if self.broken:
            raise RuntimeError("tool backend unreachable")
        return super().step(action)


@pytest.fixture
def flaky(fake_components: Any) -> type[FlakyEnv]:
    registry.register("env", "flaky", FlakyEnv, overwrite=True)
    FlakyEnv.created = 0
    return FlakyEnv


def _specs(engine: Any, n: int) -> list[EpisodeSpec]:
    return [
        EpisodeSpec(
            sample=Sample(id=str(i), messages=[{"role": "user", "content": f"q{i}"}], reference=f"Q{i}"),
            group_id=f"g{i}",
            engine=engine,
        )
        for i in range(n)
    ]


def _run(cfg: RunConfig, engine: Any, n: int = 3, **kwargs: Any) -> Any:
    return run_episodes(
        _specs(engine, n),
        env_factory=lambda: FlakyEnv(cfg),
        tokenizer=engine.tokenizer,
        params=SamplingParams(n=1),
        max_parallel=1,
        **kwargs,
    )


def _engine(cfg: RunConfig) -> fakes.FakeEngine:
    return fakes.FakeEngine(cfg)


def test_a_failed_episode_is_recorded_and_the_others_are_returned(make_config: Any, flaky: Any) -> None:
    cfg = make_config(env={"type": "flaky", "params": {"broken": [2]}})
    failures: list[EpisodeFailure] = []
    trajectories = _run(cfg, _engine(cfg), failures=failures)
    assert [t.group_id for t in trajectories] == ["g0", "g2"]
    assert [(f.spec.group_id, type(f.error), f.attempts) for f in failures] == [("g1", RuntimeError, 1)]


def test_without_a_failures_list_the_failure_is_raised(make_config: Any, flaky: Any) -> None:
    cfg = make_config(env={"type": "flaky", "params": {"broken": [2]}})
    with pytest.raises(RuntimeError, match="tool backend unreachable"):
        _run(cfg, _engine(cfg))


def test_a_retry_starts_the_episode_again_with_a_fresh_environment(make_config: Any, flaky: Any) -> None:
    cfg = make_config(env={"type": "flaky", "params": {"broken": [2]}})
    failures: list[EpisodeFailure] = []
    trajectories = _run(cfg, _engine(cfg), retries=1, failures=failures)
    assert [t.group_id for t in trajectories] == ["g0", "g1", "g2"] and not failures
    assert flaky.created == 4  # the second episode needed a second environment

    flaky.created = 0
    cfg = make_config(env={"type": "flaky", "params": {"broken": [2, 3]}})
    trajectories = _run(cfg, _engine(cfg), retries=1, failures=failures)
    assert [t.group_id for t in trajectories] == ["g0", "g2"]
    assert [(f.spec.group_id, f.attempts) for f in failures] == [("g1", 2)]


def test_an_engine_failure_is_never_tolerated(make_config: Any, flaky: Any) -> None:
    class BrokenEngine(fakes.FakeEngine):
        def generate(self, prompts: Any, params: SamplingParams) -> Any:
            raise RuntimeError("CUDA out of memory")

    cfg = make_config(env={"type": "flaky"})
    failures: list[EpisodeFailure] = []
    with pytest.raises(GenerationFailed, match="CUDA out of memory") as raised:
        _run(cfg, BrokenEngine(cfg), retries=3, failures=failures)
    assert isinstance(raised.value.__cause__, RuntimeError) and not failures


def test_an_abort_is_not_a_failure(make_config: Any, flaky: Any) -> None:
    cfg = make_config(env={"type": "flaky"})
    abort = threading.Event()
    abort.set()
    failures: list[EpisodeFailure] = []
    with pytest.raises(EpisodeAborted):
        _run(cfg, _engine(cfg), retries=3, failures=failures, abort=abort)
    assert not failures


def _train_rows(run_dir: str) -> list[dict[str, Any]]:
    return [row for _, row in iter_metrics(Path(run_dir)) if row["phase"] == "train"]


def test_the_loop_leaves_failed_episodes_out_up_to_the_threshold(make_config: Any, flaky: Any) -> None:
    """One of the four episodes of the step fails: its prompt group is left with one member, which
    has no group baseline, so the step trains on the other group alone."""
    cfg = make_config(
        env={"type": "flaky", "max_failed_frac": 0.25, "params": {"broken": [2]}},
        trainer={"max_steps": 1},
    )
    result = run(cfg)
    assert result.state == "succeeded", result.error
    row = _train_rows(result.run_dir)[0]
    assert (row["rollout/failed_episodes"], row["rollout/failed_frac"]) == (1.0, 0.25)
    assert row["rollout/groups_incomplete"] == 1.0
    batch = fakes.TRAINERS[-1].batches[0]
    assert len(batch) == 2 and len(set(batch.group_ids)) == 1
    assert "1 of 4 episodes failed" in (Path(result.run_dir) / "run.log").read_text()


def test_the_loop_fails_when_more_episodes_fail_than_tolerated(make_config: Any, flaky: Any) -> None:
    cfg = make_config(
        env={"type": "flaky", "max_failed_frac": 0.25, "params": {"broken": [1, 3]}},
        trainer={"max_steps": 1},
    )
    result = run(cfg)
    assert result.state == "failed"
    assert "2 of 4 episodes failed" in (result.error or "") and "max_failed_frac" in (result.error or "")
    assert "tool backend unreachable" in (result.error or "")


def test_by_default_one_failed_episode_fails_the_run(make_config: Any, flaky: Any) -> None:
    result = run(make_config(env={"type": "flaky", "params": {"broken": [2]}}, trainer={"max_steps": 1}))
    assert result.state == "failed" and "tool backend unreachable" in (result.error or "")


def test_retries_keep_the_batch_whole(make_config: Any, flaky: Any) -> None:
    cfg = make_config(
        env={"type": "flaky", "retries": 1, "params": {"broken": [1]}}, trainer={"max_steps": 1}
    )
    result = run(cfg)
    assert result.state == "succeeded", result.error
    assert len(fakes.TRAINERS[-1].batches[0]) == 4 and flaky.created == 5


@pytest.mark.parametrize("backend", ["trl", "verl", "nemo_rl"])
def test_pipeline_backends_cannot_leave_episodes_out(make_config: Any, backend: str) -> None:
    with pytest.raises(ConfigError, match="max_failed_frac"):
        make_config(trainer={"backend": backend}, env={"max_failed_frac": 0.5})
