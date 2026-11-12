"""``pipeline_backend: trl`` — run a :class:`~monorl.config.RunConfig` with ``trl.GRPOTrainer``
(``grpo``/``dapo``) or ``trl.SFTTrainer`` (``algorithm.name: sft``; label smoothing through the
:mod:`.sft` subclass).

monorl data → HF dataset (:mod:`.data`; ``images`` column + ``AutoProcessor`` for VLMs), monorl
rewards → TRL reward function (:mod:`.rewards`), monorl environments → TRL ``rollout_func``
(:mod:`.rollout`, multi-turn only; single-turn runs on TRL's native generation path), monorl
reporter ← ``TrainerCallback`` + the ``trl.grpo.monorl_hooks`` patch (:mod:`.callbacks`,
:mod:`.patches`). Both trainers share checkpointing (``checkpoints/step-<n>``), resume and the
merged ``final/`` export.
"""

from __future__ import annotations

import signal
import sys
import threading
import traceback
import warnings
from pathlib import Path
from typing import Any

import torch

from monorl.adapters.trl.callbacks import ReporterCallback, link_checkpoint, trajectories_with_advantages
from monorl.adapters.trl.compile import (
    compile_grpo_config,
    compile_sft_config,
    resolve_device,
    sft_label_smoothing,
)
from monorl.adapters.trl.data import build_train_and_eval
from monorl.adapters.trl.patches import HOOKS_ATTR, hooks_patch_applied
from monorl.adapters.trl.rewards import MonorlRewardFunc
from monorl.adapters.trl.rollout import EpisodeRollout, GRPOHooks
from monorl.config.schema import RunConfig
from monorl.core.errors import BackendUnavailable, ConfigError
from monorl.core.reporting import FINAL_DIR, RunReporter, list_checkpoints
from monorl.core.types import RunResult
from monorl.patching import apply
from monorl.rollout.hf_engine import load_tokenizer, resolve_dtype


def _import_trl() -> Any:
    try:
        import trl
    except ImportError as e:  # pragma: no cover - trl is installed in the dev env
        raise BackendUnavailable("pipeline_backend 'trl' needs `pip install monorl[trl]`") from e
    return trl


def load_policy_model(config: RunConfig, device: torch.device, *, vlm: bool = False) -> Any:
    """``AutoModelForCausalLM`` for text models, ``AutoModelForImageTextToText`` for VLMs (the
    class ``trl`` itself resolves for image datasets)."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    cls = AutoModelForImageTextToText if vlm else AutoModelForCausalLM
    return cls.from_pretrained(
        config.model.name_or_path,
        dtype=resolve_dtype(config.model.dtype, device),
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        attn_implementation=config.model.attn_implementation,
    )


def load_processing_class(config: RunConfig) -> tuple[Any, Any]:
    """``(tokenizer, processing_class)``: the ``AutoProcessor`` for vision-language checkpoints
    (``GRPOTrainer``/``SFTTrainer`` need it to turn ``images`` into pixel inputs), else the
    tokenizer. ``load_processor`` returns ``None`` for text-only models without loading anything."""
    from monorl.rollout.media import load_processor

    tokenizer = load_tokenizer(config)
    tokenizer.padding_side = "left"
    processor = load_processor(config)
    if processor is None:
        return tokenizer, tokenizer
    processor.tokenizer.padding_side = "left"
    return tokenizer, processor


def peft_config_for(config: RunConfig) -> Any | None:
    lora = config.model.lora
    if not lora.enabled:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=lora.target_modules,
        task_type="CAUSAL_LM",
    )


class TRLPipelineBackend:
    name = "trl"

    def __init__(self, config: RunConfig, **_: Any) -> None:
        self.config = config
        self.trainer: Any = None

    # -- PipelineBackend ---------------------------------------------------------------------
    def run(self, config: RunConfig, run_dir: Path, *, reporter: RunReporter) -> RunResult:
        run_dir = Path(run_dir)
        trl = _import_trl()
        device = resolve_device(config)
        tokenizer, processing_class = load_processing_class(config)
        vlm = processing_class is not tokenizer
        model = load_policy_model(config, device, vlm=vlm)
        train_ds, eval_ds = build_train_and_eval(config, tokenizer=tokenizer)
        reporter.log(
            f"dataset: {len(train_ds)} train samples"
            + (f", {len(eval_ds)} eval" if eval_ds else "")
            + (" (multimodal)" if "images" in train_ds.column_names else "")
        )
        if config.is_sft:
            args = compile_sft_config(config, run_dir, device=device)
            callback = ReporterCallback(reporter, sft=True, label="SFTTrainer")
            trainer_cls, loss_kwargs = trl.SFTTrainer, {}
            smoothing = sft_label_smoothing(config)
            if smoothing > 0:
                from monorl.adapters.trl.sft import LabelSmoothedSFTTrainer

                trainer_cls = LabelSmoothedSFTTrainer
                loss_kwargs = {"label_smoothing": smoothing, "loss_agg": config.algorithm.loss_agg}
            trainer = trainer_cls(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                processing_class=processing_class,
                callbacks=[callback],
                peft_config=peft_config_for(config),
                **loss_kwargs,
            )
        else:
            args = compile_grpo_config(config, run_dir, device=device)
            callback = ReporterCallback(reporter)
            trainer = self._build_grpo_trainer(
                trl, config, args, model, tokenizer, processing_class, train_ds, eval_ds, callback, reporter
            )
        return self._train(trainer, args, tokenizer, callback, run_dir, reporter)

    # -- internals ---------------------------------------------------------------------------
    def _build_grpo_trainer(
        self,
        trl: Any,
        config: RunConfig,
        args: Any,
        model: Any,
        tokenizer: Any,
        processing_class: Any,
        train_ds: Any,
        eval_ds: Any,
        callback: ReporterCallback,
        reporter: RunReporter,
    ) -> Any:
        report = apply("trl")
        if "unsloth" in sys.modules:
            # Unsloth regenerates GRPOTrainer on import; its adapter re-applies our hooks on the new class.
            from monorl.adapters.unsloth.patches import reconcile_trl_patches

            report = reconcile_trl_patches()["trl"] or report
        reporter.log(
            f"trl {report.installed_version}: patches applied={report.applied} skipped={report.skipped}"
        )
        if report.failed:
            raise ConfigError(f"TRL patches failed: {report.failed}")

        reward_fn = None
        if config.rewards:
            from monorl.rewards import build_reward

            reward_fn = build_reward(config.rewards, config=config)
        eos_ids = model.generation_config.eos_token_id
        eos_ids = (
            [tokenizer.eos_token_id]
            if eos_ids is None
            else ([eos_ids] if isinstance(eos_ids, int) else list(eos_ids))
        )
        reward_func = MonorlRewardFunc(
            reward_fn, tokenizer=tokenizer, eos_token_ids=eos_ids, algorithm=config.algorithm
        )

        rollout = None
        if config.env.type != "single_turn":
            if not hooks_patch_applied():
                raise ConfigError(
                    f"env.type={config.env.type!r} with trainer.backend='trl' needs the 'trl.grpo.monorl_hooks' patch "
                    f"(status: applied={report.applied}, skipped={report.skipped})"
                )
            from monorl.envs import make_env

            rollout = EpisodeRollout(config, lambda: make_env(config))
        hooks = GRPOHooks(rollout=rollout, on_scored=self._on_scored(reporter, reward_func))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*rollout_func.*experimental.*", category=UserWarning)
            trainer = trl.GRPOTrainer(
                model=model,
                reward_funcs=[reward_func],
                args=args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                processing_class=processing_class,
                callbacks=[callback],
                peft_config=peft_config_for(config),
                rollout_func=rollout,
            )
        setattr(trainer, HOOKS_ATTR, hooks)
        return trainer

    def _train(
        self,
        trainer: Any,
        args: Any,
        tokenizer: Any,
        callback: ReporterCallback,
        run_dir: Path,
        reporter: RunReporter,
    ) -> RunResult:
        from transformers import PrinterCallback
        from transformers.trainer_utils import get_last_checkpoint

        trainer.remove_callback(PrinterCallback)  # metrics go to metrics.jsonl, not stdout
        self.trainer = trainer

        resume_from = get_last_checkpoint(args.output_dir) if Path(args.output_dir).is_dir() else None
        if resume_from:
            reporter.log(f"resuming from {resume_from}")

        state = "succeeded"
        error: str | None = None
        final_path: str | None = None
        with _sigterm_as_interrupt():
            try:
                trainer.train(resume_from_checkpoint=resume_from)
                final_path = str(self._save_final(trainer, tokenizer, run_dir / FINAL_DIR))
            except KeyboardInterrupt:
                state, error = "cancelled", "interrupted"
            except Exception as e:  # noqa: BLE001 - must land in status.json
                state, error = "failed", f"{type(e).__name__}: {e}"
                reporter.log(traceback.format_exc())
        step = int(trainer.state.global_step)
        link_checkpoint(Path(args.output_dir), step)
        message = {"succeeded": f"finished {step} steps", "cancelled": f"cancelled after step {step}"}.get(
            state, error
        )
        reporter.set_status(state, message, step=step)
        return RunResult(
            state=state,
            final_step=step,
            run_dir=str(run_dir),
            checkpoints=list_checkpoints(run_dir),
            final_model_path=final_path,
            error=error,
            metrics=dict(callback.last_metrics),
        )

    def _on_scored(self, reporter: RunReporter, reward_func: MonorlRewardFunc) -> Any:
        config = self.config
        seen = {"batches": 0}

        def on_scored(trainer: Any, output: dict[str, Any]) -> None:
            if not config.logging.save_trajectories or not trainer.model.training:
                return
            if reward_func.num_batches == seen["batches"]:
                return
            seen["batches"] = reward_func.num_batches
            trajectories = trajectories_with_advantages(
                reward_func.last_batch, list(trainer._logs["advantages"])
            )
            # Generation happens before the optimizer step that consumes it: align with that metrics row.
            reporter.log_trajectories(
                int(trainer.state.global_step) + 1,
                trajectories,
                max_items=config.logging.trajectories_per_step,
                phase="train",
                save_transcripts=config.logging.save_transcripts,
                success_threshold=config.logging.success_threshold,
            )

        return on_scored

    @staticmethod
    def _save_final(trainer: Any, tokenizer: Any, final_dir: Path) -> Path:
        model = trainer.accelerator.unwrap_model(trainer.model)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)
        return final_dir


class _sigterm_as_interrupt:
    """Turn SIGTERM into ``KeyboardInterrupt`` when nobody else (e.g. the orchestrator) handles it."""

    def __enter__(self) -> None:
        self._previous = None
        if threading.current_thread() is not threading.main_thread():
            return
        if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):
            self._previous = signal.signal(signal.SIGTERM, _raise_interrupt)

    def __exit__(self, *exc: Any) -> None:
        if self._previous is not None or (
            threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGTERM) is _raise_interrupt
        ):
            signal.signal(signal.SIGTERM, self._previous or signal.SIG_DFL)


def _raise_interrupt(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(signal.Signals(signum).name)
