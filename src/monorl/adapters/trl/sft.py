"""``trl.SFTTrainer`` with monorl's label-smoothed SFT loss (``algorithm.sft_label_smoothing > 0``).

TRL 1.17 deprecates both stock ways of smoothing (``SFTConfig.label_smoothing_factor`` and
``compute_loss_func``, removed in TRL 2.0) in favour of a ``compute_loss`` override. This one applies
:func:`monorl.algorithms.sft_loss`, so the TRL backend optimises the same objective as the reference
trainer. Importing this module imports ``trl``.
"""

from __future__ import annotations

from typing import Any

import torch
from trl import SFTTrainer

from monorl.algorithms import sft_loss


class LabelSmoothedSFTTrainer(SFTTrainer):
    """``SFTTrainer`` whose training loss is :func:`~monorl.algorithms.sft_loss` with label smoothing.

    Loss positions are the stock ones: ``labels != -100`` after the causal shift, i.e. whatever TRL's
    collator left unmasked (``completion_only_loss`` / ``assistant_only_loss``). ``loss_agg="token"``
    keeps the stock normalisation (the token count of the whole accumulated batch when ``Trainer``
    passes ``num_items_in_batch``); ``"sequence"`` averages per-sequence means, each micro-batch
    weighted equally across the accumulation window.

    Smoothing needs the vocabulary-wide log-softmax, which TRL's fused LM head never builds, so a
    training step runs a full-logits forward for the loss and the stock forward without gradients for
    TRL's own metrics (``entropy``, ``mean_token_accuracy``, ``num_tokens``) — the same two passes
    TRL's deprecated smoothing took. As there, a MoE router auxiliary loss is logged but not trained.
    Evaluation is the untouched stock path: ``eval_loss`` is the unsmoothed held-out NLL, like the
    ``eval/loss`` of monorl's own SFT loop.
    """

    def __init__(self, *args: Any, label_smoothing: float, loss_agg: str = "token", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.loss_agg = loss_agg

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> Any:
        if not self.model.training:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        # Metric bookkeeping only; the unsmoothed loss it returns is not the objective.
        with torch.no_grad():
            super().compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        forward_inputs["use_cache"] = False
        outputs = model(**forward_inputs)
        logits = outputs.logits[:, :-1]
        targets = inputs["labels"][:, 1:].to(logits.device)
        mask = targets != -100
        # Only the loss positions pay for the vocabulary-wide log-softmax.
        log_probs = torch.log_softmax(logits[mask].float(), dim=-1)
        blank = log_probs.new_zeros(mask.shape)
        loss = sft_loss(
            blank.masked_scatter(mask, log_probs.gather(-1, targets[mask].unsqueeze(-1)).squeeze(-1)),
            mask,
            label_smoothing=self.label_smoothing,
            loss_agg=self.loss_agg,
            mean_logprob_all_tokens=blank.masked_scatter(mask, log_probs.mean(dim=-1)),
        ).loss
        if num_items_in_batch is not None:
            # As in the stock trainer: `Trainer` now leaves gradient accumulation to the loss.
            if self.loss_agg == "token":
                total = torch.as_tensor(num_items_in_batch, device=loss.device).clamp(min=1)
                loss = loss * (mask.sum() / total)
                if self.args.average_tokens_across_devices:
                    loss = loss * (self.accelerator.num_processes // self.get_tp_size())
            else:
                loss = loss / self.current_gradient_accumulation_steps
        return (loss, outputs) if return_outputs else loss
