"""Optional metric sinks (Weights & Biases, TensorBoard). Neither package is required: a missing
import is logged once and the sink is skipped; ``metrics.jsonl`` stays the source of truth."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.reporting import RunReporter


class MetricSinks:
    def __init__(self, config: RunConfig, run_dir: Path, reporter: RunReporter) -> None:
        self._wandb: Any = None
        self._tb: Any = None
        logging = config.logging
        if logging.wandb_project:
            try:
                import wandb

                self._wandb = wandb.init(
                    project=logging.wandb_project,
                    name=config.run_name,
                    dir=str(run_dir),
                    config=config.to_dict(),
                    resume="allow",
                )
            except ImportError:
                reporter.log("logging.wandb_project is set but `wandb` is not installed; skipping wandb")
            except Exception as e:  # noqa: BLE001 - never let telemetry kill a run
                reporter.log(f"wandb init failed ({type(e).__name__}: {e}); skipping wandb")
        if logging.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
            except ImportError:
                reporter.log("logging.tensorboard is set but `tensorboard` is not installed; skipping")
            except Exception as e:  # noqa: BLE001
                reporter.log(f"tensorboard init failed ({type(e).__name__}: {e}); skipping")

    @property
    def active(self) -> bool:
        return self._wandb is not None or self._tb is not None

    def log(self, step: int, metrics: dict[str, float], *, phase: str = "train") -> None:
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if not numeric:
            return
        if self._wandb is not None:
            try:
                self._wandb.log(numeric, step=int(step))
            except Exception:  # noqa: BLE001
                pass
        if self._tb is not None:
            for k, v in numeric.items():
                # evaluation metrics are named eval/... already
                tag = k if phase == "train" or k.startswith(f"{phase}/") else f"{phase}/{k}"
                self._tb.add_scalar(tag, v, int(step))

    def close(self) -> None:
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception:  # noqa: BLE001
                pass
            self._tb = None
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:  # noqa: BLE001
                pass
            self._wandb = None
