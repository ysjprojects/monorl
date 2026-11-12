"""Multi-turn environments inside ``GRPOTrainer``.

:class:`TrainerPolicyEngine` is the HF rollout engine's sampler pointed at the TRL policy model
(no second copy of the weights). :class:`EpisodeRollout` is a TRL ``rollout_func``: it runs one
monorl episode per prompt through the configured :class:`~monorl.core.interfaces.Environment`
(``monorl.orchestrator.episodes``, parallel, generation batched) and flattens each trajectory into
TRL's ``prompt_ids`` / ``completion_ids`` / ``logprobs`` / ``env_mask`` layout, where
``env_mask=0`` marks tokens injected by the environment (tool results, next user turns) so TRL
excludes them from the loss. The trajectories travel to the reward function as the
``monorl_trajectory`` extra field.

``rollout_func`` only receives prompts; the per-example dataset rows (sample ids, references,
metadata) needed by ``env.reset`` arrive through :class:`GRPOHooks`, fed by the
``trl.grpo.monorl_hooks`` patch in :mod:`monorl.adapters.trl.patches`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

import torch

from monorl.adapters.trl.data import sample_from_row
from monorl.adapters.trl.rewards import TRAJECTORY_FIELD
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.interfaces import Environment, RolloutEngine
from monorl.core.types import SamplingParams, Trajectory
from monorl.orchestrator.batching import flatten_trajectory
from monorl.orchestrator.episodes import EpisodeSpec, run_episodes
from monorl.rollout.batching import BatchingEngine
from monorl.rollout.hf_engine import HFRolloutEngine

HOOKS_ATTR = "monorl_hooks"


class TrainerPolicyEngine(HFRolloutEngine):
    """:class:`HFRolloutEngine` sampling from an already-built (possibly PEFT-wrapped) model.

    ``processing_class`` is TRL's tokenizer or, for VLM runs, the ``AutoProcessor`` — in which
    case image prompts take the processor path of :class:`HFRolloutEngine`.
    """

    def __init__(
        self, model: Any, processing_class: Any, *, device: torch.device, eos_token_ids: Sequence[int]
    ) -> None:  # noqa: D107
        from monorl.rollout.media import is_vlm_processor

        self.config = None
        self.replica = 0
        self.model = model
        self.device = torch.device(device)
        self.dtype = next(model.parameters()).dtype
        self._processor = processing_class if is_vlm_processor(processing_class) else None
        self._tokenizer = processing_class.tokenizer if self._processor is not None else processing_class
        self._lock = threading.Lock()
        self._asleep = False
        self._eos_ids = [int(t) for t in eos_token_ids]
        self._pad_id = int(self._tokenizer.pad_token_id)

    def update_weights(self, weights: Any) -> None:  # the trainer owns the weights
        raise RuntimeError("TrainerPolicyEngine shares the trainer's parameters; nothing to update")

    def shutdown(self) -> None:
        pass


class GRPOHooks:
    """Per-trainer hook bundle read by the ``trl.grpo.monorl_hooks`` patch."""

    def __init__(
        self,
        *,
        rollout: EpisodeRollout | None = None,
        on_scored: Callable[[Any, dict[str, Any]], None] | None = None,
    ) -> None:
        self.rollout = rollout
        self.on_scored = on_scored

    def before_generate(self, trainer: Any, inputs: list[dict[str, Any]]) -> None:
        if self.rollout is not None:
            self.rollout.bind_inputs(inputs)

    def after_score(self, trainer: Any, output: dict[str, Any]) -> None:
        if self.on_scored is not None:
            self.on_scored(trainer, output)


def sampling_params(args: Any, *, stop: Sequence[str] = ()) -> SamplingParams:
    """TRL generation settings as monorl :class:`SamplingParams` (one completion per episode)."""
    return SamplingParams(
        n=1,
        max_new_tokens=int(args.max_completion_length),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k) if args.top_k else -1,
        min_p=float(args.min_p or 0.0),
        repetition_penalty=float(args.repetition_penalty),
        stop=list(stop),
        logprobs=True,
    )


def flatten_for_trl(trajectory: Trajectory) -> dict[str, Any]:
    """One trajectory → TRL rollout fields (``prompt_ids``, ``completion_ids``, ``logprobs``, ``env_mask``)."""
    flat = flatten_trajectory(trajectory)
    p = flat.prompt_len
    completion_ids = flat.input_ids[p:]
    if not completion_ids:
        raise ValueError(f"trajectory {trajectory.group_id} produced no completion tokens")
    logprobs = flat.old_logprobs[p:] if flat.old_logprobs is not None else [0.0] * len(completion_ids)
    return {
        "prompt_ids": flat.input_ids[:p],
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": flat.response_mask[p:],
    }


class EpisodeRollout:
    """TRL ``rollout_func`` running monorl episodes against ``env_factory``."""

    def __init__(
        self,
        config: RunConfig,
        env_factory: Callable[[], Environment],
        *,
        engine_factory: Callable[[Any], RolloutEngine] | None = None,
        max_parallel: int | None = None,
        batch_wait_ms: float = 5.0,
    ) -> None:
        self.config = config
        self.env_factory = env_factory
        self.engine_factory = engine_factory or self.default_engine
        self.max_parallel = max_parallel
        self.batch_wait_ms = batch_wait_ms
        self._inputs: list[dict[str, Any]] | None = None
        self.num_calls = 0

    # -- wiring ------------------------------------------------------------------------------
    def bind_inputs(self, inputs: list[dict[str, Any]]) -> None:
        self._inputs = list(inputs)

    @staticmethod
    def default_engine(trainer: Any) -> RolloutEngine:
        model = trainer.accelerator.unwrap_model(trainer.model)
        return TrainerPolicyEngine(
            model,
            trainer.processing_class,
            device=trainer.accelerator.device,
            eos_token_ids=trainer.eos_token_ids,
        )

    @staticmethod
    def engine_tokenizer(trainer: Any) -> Any:
        """Plain tokenizer for the episode runner (``processing_class.tokenizer`` for VLM processors)."""
        pc = trainer.processing_class
        return (
            pc.tokenizer
            if getattr(pc, "image_processor", None) is not None and hasattr(pc, "tokenizer")
            else pc
        )

    # -- rollout_func --------------------------------------------------------------------------
    def __call__(self, prompts: list[Any], trainer: Any) -> dict[str, Any]:
        inputs = self._inputs
        self._inputs = None
        if inputs is None or len(inputs) != len(prompts):
            raise ConfigError(
                "multi-turn TRL rollout needs the 'trl.grpo.monorl_hooks' patch applied "
                "(MONORL_DISABLE_PATCHES unset) so dataset rows reach the environment"
            )
        trajectories = self.run(prompts, inputs, trainer)
        self.num_calls += 1
        flats = [flatten_for_trl(t) for t in trajectories]
        return {
            "prompt_ids": [f["prompt_ids"] for f in flats],
            "completion_ids": [f["completion_ids"] for f in flats],
            "logprobs": [f["logprobs"] for f in flats],
            "env_mask": [f["env_mask"] for f in flats],
            TRAJECTORY_FIELD: trajectories,
        }

    def run(self, prompts: list[Any], inputs: list[dict[str, Any]], trainer: Any) -> list[Trajectory]:
        from trl.models import unwrap_model_for_generation

        group = max(
            1, int(trainer.num_generations if trainer.model.training else trainer.num_generations_eval)
        )
        step = int(trainer.state.global_step)
        samples = [sample_from_row(row) for row in inputs]
        params = sampling_params(trainer.args, stop=self.config.rollout.stop)
        parallel = self.max_parallel or len(prompts)
        with (
            unwrap_model_for_generation(
                trainer.model_wrapped,
                trainer.accelerator,
                gradient_checkpointing_kwargs=trainer.args.gradient_checkpointing_kwargs,
            ),
            torch.no_grad(),
        ):
            engine = BatchingEngine(
                self.engine_factory(trainer), max_batch_size=len(prompts), max_wait_ms=self.batch_wait_ms
            )
            try:
                specs = [
                    EpisodeSpec(sample=s, group_id=f"{step}-{i // group}-{s.id}", engine=engine)
                    for i, s in enumerate(samples)
                ]
                return run_episodes(
                    specs,
                    env_factory=self.env_factory,
                    tokenizer=self.engine_tokenizer(trainer),
                    params=params,
                    max_parallel=parallel,
                )
            finally:
                engine.shutdown()
