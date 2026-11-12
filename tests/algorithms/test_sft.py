"""``monorl.algorithms.sft``: masked NLL, aggregation modes, label smoothing, trainer adapter."""

from __future__ import annotations

import math

import pytest
import torch

from monorl.algorithms import (
    SFT_METRIC_KEYS,
    PolicyLossOutput,
    mean_logprob_all_tokens,
    policy_loss,
    select_loss_fn,
    sft_loss,
    sft_policy_loss,
)
from monorl.config.schema import AlgorithmConfig

B, T, V = 3, 6, 11


def _inputs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, T, V, generator=g, requires_grad=True)
    input_ids = torch.randint(0, V, (B, T), generator=g)
    mask = torch.zeros(B, T, dtype=torch.long)
    mask[0, 2:] = 1  # 4 tokens
    mask[1, 4:] = 1  # 2 tokens
    mask[2, :] = 0  # empty sequence
    return logits, input_ids, mask


def _token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    logp_all = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    lp = logp_all.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    return torch.nn.functional.pad(lp, (1, 0))


def test_token_nll_matches_masked_mean() -> None:
    logits, ids, mask = _inputs()
    logprobs = _token_logprobs(logits, ids)
    out = sft_loss(logprobs, mask)
    assert isinstance(out, PolicyLossOutput)
    expected = -logprobs[mask.bool()].mean()
    assert out.loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert set(SFT_METRIC_KEYS) <= set(out.metrics)
    assert out.metrics["loss/sft"] == pytest.approx(expected.item(), rel=1e-6)
    assert out.metrics["sft/tokens"] == 6.0
    assert out.metrics["sft/ppl"] == pytest.approx(math.exp(expected.item()), rel=1e-5)
    out.loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # Masked-out positions get no gradient through the gather.
    assert logits.grad[2].abs().sum().item() == 0.0


def test_sequence_aggregation() -> None:
    logits, ids, mask = _inputs()
    logprobs = _token_logprobs(logits, ids).detach()
    out = sft_loss(logprobs, mask, loss_agg="sequence")
    seq0 = -logprobs[0, 2:].mean()
    seq1 = -logprobs[1, 4:].mean()
    expected = (seq0 + seq1) / 2  # the empty third sequence is skipped
    assert out.loss.item() == pytest.approx(expected.item(), rel=1e-6)
    token = sft_loss(logprobs, mask, loss_agg="token").loss.item()
    assert token != pytest.approx(expected.item())
    with pytest.raises(ValueError, match="loss_agg"):
        sft_loss(logprobs, mask, loss_agg="mean")


def test_label_smoothing_matches_hf_definition() -> None:
    logits, ids, mask = _inputs()
    logprobs = _token_logprobs(logits, ids).detach()
    eps = 0.1
    plain = sft_loss(logprobs, mask).loss.item()
    smoothed = sft_loss(logprobs, mask, label_smoothing=eps, logits=logits.detach())
    logp_all = torch.log_softmax(logits.detach()[:, :-1], dim=-1)
    uniform = torch.nn.functional.pad(logp_all.mean(-1), (1, 0))
    expected_tokens = (1 - eps) * (-logprobs) + eps * (-uniform)
    expected = expected_tokens[mask.bool()].mean().item()
    assert smoothed.loss.item() == pytest.approx(expected, rel=1e-6)
    assert smoothed.loss.item() != pytest.approx(plain)
    # Uniform-smoothing term is the vocabulary-mean NLL, which is >= the target NLL on average
    # only for peaked distributions; here we just check the mixing direction is linear in eps.
    half = sft_loss(logprobs, mask, label_smoothing=eps / 2, logits=logits.detach()).loss.item()
    assert half == pytest.approx((plain + expected) / 2, rel=1e-6)
    # ppl reports the unsmoothed NLL.
    assert smoothed.metrics["sft/ppl"] == pytest.approx(math.exp(plain), rel=1e-5)
    # The cheap [B, T] variant gives the same loss as passing full logits.
    via_mean = sft_loss(
        logprobs, mask, label_smoothing=eps, mean_logprob_all_tokens=mean_logprob_all_tokens(logits.detach())
    )
    assert via_mean.loss.item() == pytest.approx(smoothed.loss.item(), rel=1e-6)


def test_mean_logprob_all_tokens_matches_log_softmax() -> None:
    logits, _, _ = _inputs()
    got = mean_logprob_all_tokens(logits.detach(), 2.0)
    expected = torch.nn.functional.pad(
        torch.log_softmax(logits.detach()[:, :-1] / 2.0, dim=-1).mean(-1), (1, 0)
    )
    assert got.shape == (B, T)
    assert torch.allclose(got, expected, atol=1e-6)
    assert (got[:, 0] == 0).all()


def test_label_smoothing_needs_vocab_statistics() -> None:
    logits, ids, mask = _inputs()
    logprobs = _token_logprobs(logits, ids).detach()
    with pytest.raises(ValueError, match="mean_logprob_all_tokens"):
        sft_loss(logprobs, mask, label_smoothing=0.1)
    with pytest.raises(ValueError, match="match logprobs"):
        sft_loss(logprobs, mask, label_smoothing=0.1, mean_logprob_all_tokens=torch.zeros(B, T + 1))
    with pytest.raises(ValueError, match="label_smoothing"):
        sft_loss(logprobs, mask, label_smoothing=1.0, logits=logits.detach())
    with pytest.raises(ValueError, match=r"\[B, T\]"):
        sft_loss(logprobs[0], mask[0])


def test_empty_mask_is_zero_loss() -> None:
    logits, ids, mask = _inputs()
    logprobs = _token_logprobs(logits, ids)
    out = sft_loss(logprobs, torch.zeros_like(mask))
    assert out.loss.item() == 0.0 and out.metrics["sft/tokens"] == 0.0 and out.metrics["sft/ppl"] == 1.0
    out.loss.backward()  # still participates in autograd


def test_trainer_adapter_ignores_rl_inputs_and_dispatch() -> None:
    logits, ids, mask = _inputs()
    logprobs = _token_logprobs(logits, ids).detach()
    algo = AlgorithmConfig(name="sft", loss_agg="sequence", sft_label_smoothing=0.0)
    direct = sft_loss(logprobs, mask, loss_agg="sequence")
    adapted = sft_policy_loss(
        logprobs,
        torch.randn(B, T),
        torch.randn(B, T),
        mask,
        algorithm=algo,
        ref_logprobs=torch.randn(B, T),
        entropy=torch.rand(B, T),
    )
    assert adapted.loss.item() == pytest.approx(direct.loss.item())
    assert adapted.metrics == direct.metrics

    smooth_algo = AlgorithmConfig(name="sft", sft_label_smoothing=0.2)
    smooth = sft_policy_loss(
        logprobs,
        None,
        None,
        mask,
        algorithm=smooth_algo,
        mean_logprob_all_tokens=mean_logprob_all_tokens(logits.detach()),
    )
    assert smooth.loss.item() == pytest.approx(
        sft_loss(logprobs, mask, label_smoothing=0.2, logits=logits.detach()).loss.item(), rel=1e-6
    )

    assert select_loss_fn(AlgorithmConfig(name="sft")) is sft_policy_loss
    assert select_loss_fn(AlgorithmConfig(name="grpo")) is policy_loss
