"""Per-token loss weights (``algorithm.loss_agg: rollout``) through the reference trainer."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from monorl.trainer.reference import ReferenceTrainer
from tests.trainer.conftest import make_batch, make_config


def _step(tiny_model_dir: Path, batch, *, loss_agg: str, micro_batch_size: int):
    torch.manual_seed(0)
    config = make_config(tiny_model_dir, micro_batch_size=micro_batch_size)
    config.algorithm.loss_agg = loss_agg
    trainer = ReferenceTrainer(config)
    metrics = trainer.train_step(batch)
    return metrics, trainer.compute_logprobs(batch)


def test_rollout_weighted_step_does_not_depend_on_micro_batching(tiny_model_dir: Path, tokenizer) -> None:
    """Rows 0 and 1 are two sequences of one rollout, rows 2 and 3 are rollouts of their own.
    Micro-batches of one row cut through the first rollout; the update must be the same as with
    the whole batch in one pass, and differ from per-sequence aggregation."""
    batch = make_batch(tokenizer)
    mask = batch.response_mask.float()
    tokens = mask.sum(dim=1)
    per_rollout = torch.stack([tokens[0] + tokens[1], tokens[0] + tokens[1], tokens[2], tokens[3]])
    batch.loss_weights = mask / per_rollout[:, None]

    split, split_logprobs = _step(tiny_model_dir, batch, loss_agg="rollout", micro_batch_size=1)
    whole, whole_logprobs = _step(tiny_model_dir, batch, loss_agg="rollout", micro_batch_size=4)
    assert split["grad_norm"] == pytest.approx(whole["grad_norm"], rel=1e-4)
    torch.testing.assert_close(split_logprobs, whole_logprobs, atol=1e-5, rtol=1e-5)  # float32 sums

    batch.loss_weights = None
    per_sequence, _ = _step(tiny_model_dir, batch, loss_agg="sequence", micro_batch_size=4)
    assert whole["grad_norm"] != pytest.approx(per_sequence["grad_norm"], rel=1e-3)
