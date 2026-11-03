"""``pipeline_backend: trl`` — run a :class:`~monorl.config.RunConfig` with ``trl.GRPOTrainer``.

monorl data → HF dataset (:mod:`.data`), monorl rewards → TRL reward function (:mod:`.rewards`),
monorl environments → TRL ``rollout_func`` (:mod:`.rollout`, multi-turn only; single-turn runs on
TRL's native generation path), monorl reporter ← ``TrainerCallback`` + the
``trl.grpo.monorl_hooks`` patch (:mod:`.callbacks`, :mod:`.patches`).
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
from monorl.adapters.trl.compile import compile_grpo_config, resolve_device
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


def load_policy_model(config: RunConfig, device: torch.device) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        dtype=resolve_dtype(config.model.dtype, device),
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        attn_implementation=config.model.attn_implementation,
    )


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
        report = apply("trl")
        if "unsloth" in sys.modules:
            # Unsloth regenerates GRPOTrainer on import; its adapter re-applies our hooks on the new class.
            from monorl.adapters.unsloth.patches import reconcile_trl_patches

            report = reconcile_trl_patches()["trl"] or report
        reporter.log(f"trl {report.installed_version}: patches applied={report.applied} skipped={report.skipped}")
        if report.failed:
            raise ConfigError(f"TRL patches failed: {report.failed}")

        device = resolve_device(config)
        args = compile_grpo_config(config, run_dir, device=device)
        tokenizer = load_tokenizer(config)
        tokenizer.padding_side = "left"
        model = load_policy_model(config, device)
        train_ds, eval_ds = build_train_and_eval(config, tokenizer=tokenizer)
        reporter.log(f"dataset: {len(train_ds)} train samples" + (f", {len(eval_ds)} eval" if eval_ds else ""))

        reward_fn = None
        if config.rewards:
            from monorl.rewards import build_reward

            reward_fn = build_reward(config.rewards, config=config)
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [tokenizer.eos_token_id] if eos_ids is None else ([eos_ids] if isinstance(eos_ids, int) else list(eos_ids))
        reward_func = MonorlRewardFunc(reward_fn, tokenizer=tokenizer, eos_token_ids=eos_ids, algorithm=config.algorithm)

        rollout = None
        if config.env.type != "single_turn":
            if not hooks_patch_applied():
                raise ConfigError(
                    f"env.type={config.env.type!r} with trainer.backend='trl' needs the 'trl.grpo.monorl_hooks' patch "
                    f"(status: applied={report.applied}, skipped={report.skipped})"
                )
            from monorl.envs import make_env

            rollout = EpisodeRollout(config, lambda: make_env(config))
        callback = ReporterCallback(reporter)
        hooks = GRPOHooks(rollout=rollout, on_scored=self._on_scored(reporter, reward_func))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*rollout_func.*experimental.*", category=UserWarning)
            trainer = trl.GRPOTrainer(
                model=model,
                reward_funcs=[reward_func],
                args=args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                processing_class=tokenizer,
                callbacks=[callback],
                peft_config=peft_config_for(config),
                rollout_func=rollout,
            )
        from transformers import PrinterCallback
        from transformers.trainer_utils import get_last_checkpoint

        trainer.remove_callback(PrinterCallback)  # metrics go to metrics.jsonl, not stdout
        setattr(trainer, HOOKS_ATTR, hooks)
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
        message = {"succeeded": f"finished {step} steps", "cancelled": f"cancelled after step {step}"}.get(state, error)
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

    # -- internals ---------------------------------------------------------------------------
    def _on_scored(self, reporter: RunReporter, reward_func: MonorlRewardFunc) -> Any:
        config = self.config
        seen = {"batches": 0}

        def on_scored(trainer: Any, output: dict[str, Any]) -> None:
            if not config.logging.save_trajectories or not trainer.model.training:
                return
            if reward_func.num_batches == seen["batches"]:
                return
            seen["batches"] = reward_func.num_batches
            trajectories = trajectories_with_advantages(reward_func.last_batch, list(trainer._logs["advantages"]))
            # Generation happens before the optimizer step that consumes it: align with that metrics row.
            reporter.log_trajectories(
                int(trainer.state.global_step) + 1, trajectories, max_items=config.logging.trajectories_per_step
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
