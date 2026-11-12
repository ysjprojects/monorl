"""Component protocols. Every backend/adapter implements one or more of these.

Two integration levels exist:

* **Component-level** (driven by :mod:`monorl.orchestrator`): :class:`RolloutEngine`,
  :class:`Trainer`, :class:`Environment`, :class:`RewardFunction`, :class:`DataSource`,
  :class:`WeightSync`. vLLM/SGLang/HF are rollout engines; the reference torch trainer and
  Unsloth-patched models are trainers.
* **Pipeline-level** (frameworks that own their whole RL loop: verl, NeMo-RL, TRL-native):
  :class:`PipelineBackend` receives the full :class:`~monorl.config.RunConfig` and must bridge
  monorl data/rewards/envs into the framework (typically via patches).

All protocols are ``runtime_checkable`` so the registry can validate factories cheaply.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from monorl.core.types import (
    Completion,
    Message,
    PromptInput,
    RewardResult,
    RunResult,
    Sample,
    SamplingParams,
    StepResult,
    Trajectory,
)

if TYPE_CHECKING:
    import torch

    from monorl.config.schema import RunConfig
    from monorl.core.batch import TrainBatch
    from monorl.core.reporting import RunReporter


@runtime_checkable
class DataSource(Protocol):
    """Iterable of :class:`Sample`. Implementations: local files, HF hub, S3 (fsspec)."""

    name: str

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Sample]: ...

    def schema(self) -> dict[str, str]:
        """Column name -> dtype string, for UI preview and field mapping."""
        ...


@runtime_checkable
class RolloutEngine(Protocol):
    """Text generation under the current policy weights.

    ``generate`` must return exactly ``params.n`` completions per prompt, preserving order.
    When a prompt is ``list[int]`` the engine conditions on those exact token ids; when it is a
    chat (``list[Message]``) the engine applies the tokenizer chat template with
    ``add_generation_prompt=True`` and records the resulting ids in ``Completion.prompt_token_ids``.
    A completion ended by one of ``params.stop`` keeps that stop string at the end of
    ``Completion.text`` (``finish_reason == "stop"``): environments parse closing tags such as
    ``</tool_call>`` from it, and a chat re-rendered from the text has to match the sampled tokens.

    Optional capability (duck-typed, not a protocol member): ``cancel() -> None`` aborts the
    ``generate`` calls that are in flight — they raise
    :class:`~monorl.core.errors.GenerationCancelled` instead of returning — without shutting the
    engine down. It is safe to call from any thread, a no-op when nothing is generating, and
    ``generate`` calls made afterwards run normally.
    """

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]: ...

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Hot-swap policy weights (HF parameter names, merged LoRA)."""
        ...

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        """Reload weights from an HF-format directory (checkpoint-based sync)."""
        ...

    def sleep(self) -> None:
        """Release accelerator memory for colocated training. No-op if unsupported."""
        ...

    def wake_up(self) -> None: ...

    def shutdown(self) -> None: ...

    @property
    def tokenizer(self) -> Any:
        """``transformers.PreTrainedTokenizerBase`` used for prompt formatting."""
        ...


@runtime_checkable
class Trainer(Protocol):
    """Owns the policy parameters and optimizer. Loss functions come from :mod:`monorl.algorithms`."""

    @property
    def step(self) -> int: ...

    def train_step(self, batch: TrainBatch) -> dict[str, float]:
        """One optimizer step over ``batch`` (trainer handles micro-batching). Returns metrics."""
        ...

    def compute_logprobs(self, batch: TrainBatch, *, reference: bool = False) -> torch.Tensor:
        """``[B, T]`` logprobs of ``input_ids`` under the current policy (or the frozen reference
        when ``reference=True``). No gradient."""
        ...

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Merged, HF-named parameters for :meth:`RolloutEngine.update_weights`."""
        ...

    def save_checkpoint(self, path: Path) -> None:
        """Resumable state (weights/adapters + optimizer + step)."""
        ...

    def load_checkpoint(self, path: Path) -> None: ...

    def save_pretrained(self, path: Path) -> None:
        """HF-format export (merged LoRA) loadable by any rollout engine."""
        ...


@runtime_checkable
class RewardFunction(Protocol):
    """Batched scoring. Return one :class:`RewardResult` per trajectory, in order."""

    name: str

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]: ...


@runtime_checkable
class Environment(Protocol):
    """Episode-level environment driven by the orchestrator::

        messages = env.reset(sample)
        for turn in range(env.max_turns):
            completion = engine.generate([messages], params)[0][0]
            messages.append({"role": "assistant", "content": completion.text})
            result = env.step(completion)
            messages.extend(result.messages)
            if result.done: break

    Environments are stateful per episode; the orchestrator creates one via the factory per
    episode (or calls ``reset`` again on a reusable instance).
    """

    max_turns: int

    def reset(self, sample: Sample) -> list[Message]: ...

    def step(self, action: Completion) -> StepResult: ...

    def close(self) -> None: ...


@runtime_checkable
class EpisodeDriver(Protocol):
    """An environment that runs a whole episode itself instead of being stepped turn by turn: an
    external agent harness that calls the policy through the capture gateway
    (:mod:`monorl.gateway`). The environment factory may return either kind; the episode runner
    calls ``run_episode`` when the object has it and ``reset``/``step`` otherwise.

    ``run_episode`` returns the finished :class:`Trajectory` (unscored: rewards are applied by the
    caller, per-turn environment rewards go in ``Turn.reward``). It must honour ``abort`` by
    raising :class:`~monorl.core.errors.EpisodeAborted` and report an engine failure as
    :class:`~monorl.core.errors.GenerationFailed`; any other exception counts as a failure of the
    episode's environment, which a run may tolerate (``env.max_failed_frac``). ``turn_rendering``
    is ``rollout.turn_rendering`` (``"tokens"`` or ``"template"``) and ``max_context_tokens`` the
    episode's context budget in tokens (``None``: unlimited): no prompt plus completion may exceed
    it.
    """

    def run_episode(
        self,
        sample: Sample,
        *,
        group_id: str,
        engine: RolloutEngine,
        tokenizer: Any,
        params: SamplingParams,
        abort: threading.Event | None = None,
        turn_rendering: str = "tokens",
        max_context_tokens: int | None = None,
    ) -> Trajectory: ...

    def close(self) -> None: ...


@runtime_checkable
class WeightSync(Protocol):
    """Moves trainer weights into rollout engines."""

    name: str

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        """Returns metrics (e.g. ``bytes``, ``seconds``)."""
        ...


@runtime_checkable
class PipelineBackend(Protocol):
    """A framework that runs the entire RL loop itself (verl, NeMo-RL, TRL-native)."""

    name: str

    def run(self, config: RunConfig, run_dir: Path, *, reporter: RunReporter) -> RunResult: ...


class RolloutEngineFactory(Protocol):
    def __call__(self, config: RunConfig, **kwargs: Any) -> RolloutEngine: ...


class TrainerFactory(Protocol):
    def __call__(self, config: RunConfig, **kwargs: Any) -> Trainer: ...


class EnvironmentFactory(Protocol):
    def __call__(self, config: RunConfig, **kwargs: Any) -> Environment: ...


class RewardFactory(Protocol):
    def __call__(self, params: dict[str, Any], **kwargs: Any) -> RewardFunction: ...
