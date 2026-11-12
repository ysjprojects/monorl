from __future__ import annotations

import pytest
import torch

from monorl.algorithms import METRIC_KEYS, gae, kl_penalty, policy_loss, value_loss
from monorl.config.schema import AlgorithmConfig

B, T = 3, 5


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logprobs = (-torch.rand(B, T, generator=g) * 2).requires_grad_(True)
    old = -torch.rand(B, T, generator=g) * 2
    ref = -torch.rand(B, T, generator=g) * 2
    adv = torch.randn(B, T, generator=g)
    mask = torch.ones(B, T)
    mask[0, 3:] = 0  # sequence 0 has 3 response tokens
    mask[1, 1:] = 0  # sequence 1 has 1 response token
    return logprobs, old, ref, adv, mask


def test_metric_keys_present_and_loss_differentiable():
    logprobs, old, ref, adv, mask = _batch()
    out = policy_loss(
        logprobs,
        old,
        adv,
        mask,
        algorithm=AlgorithmConfig(kl_coef=0.1),
        ref_logprobs=ref,
        entropy=torch.ones(B, T),
    )
    assert set(out.metrics) == set(METRIC_KEYS)
    assert out.loss.requires_grad
    out.loss.backward()
    assert logprobs.grad is not None
    # padded positions get no gradient
    assert torch.all(logprobs.grad[mask == 0] == 0)


def test_old_logprobs_none_gives_unit_ratio_with_gradient():
    logprobs, _, _, adv, mask = _batch()
    out = policy_loss(logprobs, None, adv, mask, algorithm=AlgorithmConfig())
    assert out.metrics["policy/ratio_mean"] == pytest.approx(1.0)
    assert out.metrics["policy/clipfrac"] == 0.0
    assert out.metrics["policy/approx_kl"] == pytest.approx(0.0)
    out.loss.backward()
    # REINFORCE gradient: -adv / n_tokens on masked positions
    expected = -(adv * mask) / mask.sum()
    torch.testing.assert_close(logprobs.grad, expected)


def test_clipping_zeroes_gradient_where_ratio_outside_range():
    cfg = AlgorithmConfig(clip_eps=0.2)
    mask = torch.ones(1, 4)
    old = torch.zeros(1, 4)
    # ratios: 1.5 (clipped, adv>0), 0.5 (clipped, adv<0), 1.5 (adv<0: unclipped is min), 1.0
    logprobs = torch.log(torch.tensor([[1.5, 0.5, 1.5, 1.0]])).requires_grad_(True)
    adv = torch.tensor([[1.0, -1.0, -1.0, 1.0]])
    out = policy_loss(logprobs, old, adv, mask, algorithm=cfg)
    out.loss.backward()
    grad = logprobs.grad[0]
    assert grad[0] == 0.0 and grad[1] == 0.0
    assert grad[2] != 0.0 and grad[3] != 0.0
    assert out.metrics["policy/clipfrac"] == pytest.approx(0.5)


def test_clip_higher_allows_larger_upward_ratio():
    mask = torch.ones(1, 2)
    old = torch.zeros(1, 2)
    logprobs = torch.log(torch.tensor([[1.25, 0.9]])).requires_grad_(True)
    adv = torch.tensor([[1.0, -1.0]])
    sym = policy_loss(logprobs, old, adv, mask, algorithm=AlgorithmConfig(clip_eps=0.2))
    assert sym.metrics["policy/clipfrac"] == pytest.approx(0.5)  # ratio 1.25 > 1.2 clipped
    logprobs2 = logprobs.detach().clone().requires_grad_(True)
    high = policy_loss(logprobs2, old, adv, mask, algorithm=AlgorithmConfig(clip_eps=0.2, clip_eps_high=0.28))
    assert high.metrics["policy/clipfrac"] == 0.0
    high.loss.backward()
    assert high.metrics["loss/policy"] < sym.metrics["loss/policy"]
    assert logprobs2.grad[0, 0] != 0.0


def test_kl_estimators_k3_nonnegative_and_k1_mean_matches():
    g = torch.Generator().manual_seed(1)
    lp = -torch.rand(4, 7, generator=g) * 3
    ref = -torch.rand(4, 7, generator=g) * 3
    k1, k2, k3 = (kl_penalty(lp, ref, e) for e in ("k1", "k2", "k3"))
    assert torch.all(k2 >= 0)
    assert torch.all(k3 >= 0)
    torch.testing.assert_close(k1, lp - ref)
    assert torch.all(kl_penalty(lp, lp, "k3") == 0)
    with pytest.raises(ValueError):
        kl_penalty(lp, ref, "k9")  # type: ignore[arg-type]


def test_kl_penalty_enters_loss_only_with_coef():
    logprobs, old, ref, adv, mask = _batch()
    base = policy_loss(logprobs, old, adv, mask, algorithm=AlgorithmConfig(kl_coef=0.0), ref_logprobs=ref)
    with_kl = policy_loss(logprobs, old, adv, mask, algorithm=AlgorithmConfig(kl_coef=0.5), ref_logprobs=ref)
    assert base.metrics["loss/kl"] == pytest.approx(with_kl.metrics["loss/kl"])
    assert with_kl.metrics["loss/kl"] > 0.0
    assert with_kl.loss.item() == pytest.approx(base.loss.item() + 0.5 * base.metrics["loss/kl"], rel=1e-5)
    no_ref = policy_loss(logprobs, old, adv, mask, algorithm=AlgorithmConfig(kl_coef=0.5))
    assert no_ref.metrics["loss/kl"] == 0.0


def test_entropy_bonus_lowers_loss():
    logprobs, old, _, adv, mask = _batch()
    ent = torch.full((B, T), 2.0)
    base = policy_loss(logprobs, old, adv, mask, algorithm=AlgorithmConfig(entropy_coef=0.0), entropy=ent)
    bonus = policy_loss(logprobs, old, adv, mask, algorithm=AlgorithmConfig(entropy_coef=0.1), entropy=ent)
    assert base.metrics["loss/entropy"] == pytest.approx(2.0)
    assert bonus.loss.item() == pytest.approx(base.loss.item() - 0.2, rel=1e-5)


def test_sequence_vs_token_aggregation_differ():
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    adv = torch.tensor([[1.0, 1.0, 1.0, 1.0], [-3.0, 0.0, 0.0, 0.0]])
    logprobs = torch.zeros(2, 4, requires_grad=True)
    token = policy_loss(logprobs, None, adv, mask, algorithm=AlgorithmConfig(loss_agg="token"))
    seq = policy_loss(logprobs, None, adv, mask, algorithm=AlgorithmConfig(loss_agg="sequence"))
    # token: -(4*1 - 3)/5 = -0.2 ; sequence: mean(-1, +3) = 1.0
    assert token.metrics["loss/policy"] == pytest.approx(-0.2)
    assert seq.metrics["loss/policy"] == pytest.approx(1.0)


def test_per_sequence_advantages_are_broadcast():
    logprobs = torch.zeros(2, 3, requires_grad=True)
    mask = torch.ones(2, 3)
    out = policy_loss(logprobs, None, torch.tensor([1.0, -1.0]), mask, algorithm=AlgorithmConfig())
    assert out.metrics["loss/policy"] == pytest.approx(0.0)


def test_empty_mask_is_numerically_safe():
    logprobs = torch.randn(2, 3, requires_grad=True)
    zero_mask = torch.zeros(2, 3)
    for agg in ("token", "sequence"):
        out = policy_loss(
            logprobs,
            torch.randn(2, 3),
            torch.randn(2, 3),
            zero_mask,
            algorithm=AlgorithmConfig(loss_agg=agg, kl_coef=0.1, entropy_coef=0.1),
            ref_logprobs=torch.randn(2, 3),
            entropy=torch.randn(2, 3),
        )
        assert out.loss.item() == 0.0
        assert all(v == 0.0 for v in out.metrics.values())
        out.loss.backward()


def test_gae_matches_manual_rollout():
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    values = torch.tensor([[0.5, 0.25, 0.75, 9.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    adv, ret = gae(rewards, values, mask, gamma=0.9, lam=0.8)
    # t=2 terminal: delta = 1 - 0.75 = 0.25 ; A2 = 0.25
    # t=1: delta = 0 + 0.9*0.75 - 0.25 = 0.425 ; A1 = 0.425 + 0.72*0.25 = 0.605
    # t=0: delta = 0 + 0.9*0.25 - 0.5 = -0.275 ; A0 = -0.275 + 0.72*0.605 = 0.1606
    torch.testing.assert_close(adv, torch.tensor([[0.1606, 0.605, 0.25, 0.0]]))
    torch.testing.assert_close(ret, adv + values * mask)


def test_gae_lambda_one_gamma_one_equals_return_minus_value():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 0.5, 0.5]])
    mask = torch.ones(1, 3)
    adv, ret = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    torch.testing.assert_close(ret, torch.tensor([[6.0, 5.0, 3.0]]))
    torch.testing.assert_close(adv, ret - values)


def test_value_loss_clipping():
    values = torch.tensor([[2.0, 0.0]], requires_grad=True)
    returns = torch.tensor([[1.0, 1.0]])
    old = torch.tensor([[1.0, 0.0]])
    out = value_loss(values, returns, old, clip=0.5)
    # pos0: v=2, clipped v=1.5 → err 1.0 vs 0.25 → max 1.0 ; pos1: err 1.0 both
    assert out.loss.item() == pytest.approx(0.5 * (1.0 + 1.0) / 2)
    assert out.metrics["value/clipfrac"] == 0.0
    out.loss.backward()
    assert values.grad is not None
    unclipped = value_loss(values.detach(), returns, None, clip=0.5)
    assert unclipped.loss.item() == pytest.approx(0.5)
    masked = value_loss(values.detach(), returns, old, clip=0.5, mask=torch.tensor([[0.0, 1.0]]))
    assert masked.loss.item() == pytest.approx(0.5)
