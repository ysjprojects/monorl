from __future__ import annotations

import math

import pytest
import torch

from monorl.algorithms import METRIC_KEYS, policy_loss
from monorl.algorithms.offpolicy import importance_correction, offpolicy_metrics
from monorl.config.schema import AlgorithmConfig, OffPolicyConfig

# Two sequences with 3 and 2 response tokens. The current policy equals the proximal one, so the
# clipped ratio is 1 and every token's policy loss is -advantage * coefficient.
MASK = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
ADV = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
RHO = torch.tensor([[2.0, 0.5, 4.0], [1.0, 3.0, 1.0]])  # proximal / behaviour probability ratio
PROXIMAL = torch.tensor([[-1.0, -2.0, -0.5], [-0.3, -1.5, 0.0]])
BEHAVIOR = (PROXIMAL - RHO.log()) * MASK


def _algo(loss_agg: str = "token", **off_policy) -> AlgorithmConfig:
    return AlgorithmConfig(loss_agg=loss_agg, off_policy=OffPolicyConfig(**off_policy))


def _loss(algorithm: AlgorithmConfig, *, behavior: torch.Tensor | None = BEHAVIOR, **kwargs):
    logprobs = PROXIMAL.clone().requires_grad_(True)
    extra = {} if behavior is None else {"behavior_logprobs": behavior}
    out = policy_loss(logprobs, PROXIMAL, ADV, MASK, algorithm=algorithm, **extra, **kwargs)
    out.loss.backward()
    return out, logprobs.grad


def test_defaults_reproduce_the_uncorrected_loss_exactly():
    plain, plain_grad = _loss(AlgorithmConfig(), behavior=None)
    assert plain.loss.item() == pytest.approx(-3.0)  # -(1+2+3+4+5)/5
    with_behavior, grad = _loss(AlgorithmConfig())
    assert with_behavior.loss.item() == plain.loss.item()
    assert torch.equal(grad, plain_grad)
    assert set(with_behavior.metrics) == set(METRIC_KEYS)
    # the trainer anchor alone (diagnostics only) does not touch the loss either
    anchored, grad = _loss(_algo(anchor="trainer"))
    assert anchored.loss.item() == plain.loss.item()
    assert torch.equal(grad, plain_grad)


@pytest.mark.parametrize(
    ("off_policy", "coefficients"),
    [
        # min(rho, 2.5) per token
        ({"importance_weights": "token", "importance_cap": 2.5}, [[2.0, 0.5, 2.5], [1.0, 2.5, 0.0]]),
        ({"importance_weights": "token", "importance_cap": None}, [[2.0, 0.5, 4.0], [1.0, 3.0, 0.0]]),
        # min(prod rho, 3) per sequence: products are 4 and 3
        ({"importance_weights": "sequence", "importance_cap": 3.0}, [[3.0, 3.0, 3.0], [3.0, 3.0, 0.0]]),
        ({"importance_weights": "sequence", "importance_cap": 10.0}, [[4.0, 4.0, 4.0], [3.0, 3.0, 0.0]]),
        # tokens with rho outside [0.6, 3.5] are dropped
        (
            {"ratio_mask": "token", "ratio_mask_low": 0.6, "ratio_mask_high": 3.5},
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ),
        # sequences whose product (4, 3) is outside [0.5, 3.5]
        (
            {"ratio_mask": "sequence", "ratio_mask_low": 0.5, "ratio_mask_high": 3.5},
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ),
        # sequences whose geometric mean (4^(1/3) = 1.587, sqrt(3) = 1.732) is outside [0.5, 1.6]
        (
            {"ratio_mask": "sequence_mean", "ratio_mask_low": 0.5, "ratio_mask_high": 1.6},
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        ),
        # weights and mask compose: capped token weight where the token ratio is accepted
        (
            {
                "importance_weights": "token",
                "importance_cap": 2.5,
                "ratio_mask": "token",
                "ratio_mask_low": 0.6,
                "ratio_mask_high": 3.5,
            },
            [[2.0, 0.0, 0.0], [1.0, 2.5, 0.0]],
        ),
    ],
)
def test_each_correction_scales_the_token_losses(off_policy, coefficients):
    coef = torch.tensor(coefficients)
    out, grad = _loss(_algo(**off_policy))
    # the mean keeps all 5 response tokens in the denominator, dropped ones contribute zero
    assert out.loss.item() == pytest.approx(-(ADV * coef).sum().item() / 5.0)
    torch.testing.assert_close(grad, -ADV * coef / 5.0)
    torch.testing.assert_close(
        importance_correction(PROXIMAL, BEHAVIOR, MASK, OffPolicyConfig(**off_policy)).weights, coef
    )


def test_corrections_compose_with_every_loss_aggregation():
    off = {"importance_weights": "token", "importance_cap": 2.5}
    token, _ = _loss(_algo("token", **off))
    assert token.loss.item() == pytest.approx(-(2.0 + 1.0 + 7.5 + 4.0 + 12.5) / 5.0)
    sequence, _ = _loss(_algo("sequence", **off))
    assert sequence.loss.item() == pytest.approx(-((2.0 + 1.0 + 7.5) / 3.0 + (4.0 + 12.5) / 2.0) / 2.0)
    loss_weights = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.0]])
    rollout, grad = _loss(_algo("rollout", **off), loss_weights=loss_weights)
    # sum(loss_weight * importance weight * -advantage) / sum(loss_weight)
    assert rollout.loss.item() == pytest.approx(-(0.2 + 0.2 + 2.25 + 1.6 + 6.25) / 1.5)
    coef = torch.tensor([[2.0, 0.5, 2.5], [1.0, 2.5, 0.0]])
    torch.testing.assert_close(grad, -ADV * coef * loss_weights / 1.5)


def test_weights_multiply_the_clipped_surrogate_not_the_ratio():
    # current / proximal = 1.5 is clipped at 1.2 for a positive advantage: no gradient, and the
    # importance weight scales the clipped value.
    algorithm = _algo(importance_weights="token", importance_cap=5.0)
    proximal = torch.zeros(1, 2)
    behavior = -torch.tensor([[3.0, 0.5]]).log()  # rho = 3, 0.5
    logprobs = torch.tensor([[1.5, 1.0]]).log().requires_grad_(True)
    out = policy_loss(
        logprobs,
        proximal,
        torch.ones(1, 2),
        torch.ones(1, 2),
        algorithm=algorithm,
        behavior_logprobs=behavior,
    )
    assert out.loss.item() == pytest.approx(-(3.0 * 1.2 + 0.5 * 1.0) / 2.0)
    out.loss.backward()
    torch.testing.assert_close(logprobs.grad, torch.tensor([[0.0, -0.25]]))


def test_identical_behaviour_and_proximal_policy_is_a_no_op():
    generator = torch.Generator().manual_seed(0)
    logprobs = -torch.rand(4, 6, generator=generator) * 5
    mask = (torch.rand(4, 6, generator=generator) > 0.3).float()
    for off in (
        OffPolicyConfig(importance_weights="token", ratio_mask="token"),
        OffPolicyConfig(importance_weights="sequence", ratio_mask="sequence", ratio_mask_high=1.0),
        OffPolicyConfig(importance_weights="token", ratio_mask="sequence_mean", ratio_mask_low=1.0),
    ):
        correction = importance_correction(logprobs, logprobs, mask, off)
        assert torch.equal(correction.weights, mask)
        assert not correction.masked.any() and not correction.truncated.any()
        metrics = offpolicy_metrics(logprobs, logprobs, mask, off)
        assert metrics["offpolicy/weight_mean"] == 1.0 and metrics["offpolicy/weight_max"] == 1.0
        assert metrics["offpolicy/ess"] == pytest.approx(1.0)
        assert metrics["offpolicy/kl"] == 0.0 and metrics["offpolicy/kl_k3"] == 0.0
        assert metrics["offpolicy/truncated_frac"] == 0.0 and metrics["offpolicy/masked_frac"] == 0.0


def test_diagnostics_match_hand_computed_values():
    off = OffPolicyConfig(
        importance_weights="token",
        importance_cap=2.5,
        ratio_mask="token",
        ratio_mask_low=0.6,
        ratio_mask_high=3.5,
    )
    metrics = offpolicy_metrics(PROXIMAL, BEHAVIOR, MASK, off)
    # ratios 2, 0.5, 4, 1, 3 -> applied coefficients 2, 0, 0, 1, 2.5
    assert metrics["offpolicy/weight_mean"] == pytest.approx(5.5 / 5)
    assert metrics["offpolicy/weight_max"] == pytest.approx(2.5)
    assert metrics["offpolicy/truncated_frac"] == pytest.approx(2 / 5)  # 4 and 3 exceed the cap
    assert metrics["offpolicy/masked_frac"] == pytest.approx(2 / 5)  # 0.5 and 4 leave the band
    assert metrics["offpolicy/ess"] == pytest.approx(5.5**2 / (5 * (4 + 1 + 6.25)))
    # KL(behaviour || proximal) from behaviour samples: mean(-log rho) and mean(rho - 1 - log rho)
    assert metrics["offpolicy/kl"] == pytest.approx(-math.log(12.0) / 5, abs=1e-6)
    assert metrics["offpolicy/kl_k3"] == pytest.approx((10.5 - 5 - math.log(12.0)) / 5, abs=1e-6)


# -- score centering ---------------------------------------------------------------------------
# One position over a 3-token vocabulary; the sampler recorded its two most likely tokens, so the
# modelled tail is the single remaining token and the centering is exact.
Q = torch.tensor([0.5, 0.3, 0.2])  # sampler
P = torch.tensor([0.6, 0.3, 0.1])  # trainer
HEAD = [0, 1]


def _centered_grad(sampled: int, advantage: float, *, clip_eps: float | None, q: torch.Tensor = Q, pad=False):
    """Gradient of the loss w.r.t. the trainer's logits for one sampled token."""
    logits = P.log().clone().requires_grad_(True)
    logp = torch.log_softmax(logits, dim=-1)
    behavior_top = q[HEAD].log()
    top = logp[HEAD]
    if pad:  # an absent third head entry: id 0, -inf
        behavior_top = torch.cat([behavior_top, torch.tensor([float("-inf")])])
        top = torch.cat([top, logp[:1]])
    out = policy_loss(
        logp[sampled].reshape(1, 1),
        None if clip_eps is None else q[sampled].log().reshape(1, 1),
        torch.tensor([[advantage]]),
        torch.ones(1, 1),
        algorithm=AlgorithmConfig(clip_eps=clip_eps or 0.2),
        top_logprobs=top.reshape(1, 1, -1),
        behavior_top_logprobs=behavior_top.reshape(1, 1, -1),
    )
    out.loss.backward()
    return logits.grad, out


def test_score_centering_subtracts_the_expected_score_under_the_sampler():
    grad, out = _centered_grad(0, 2.0, clip_eps=None)
    # -A * (e_y - p - sum_v q_v (e_v - p)) = -A * (e_y - q)
    torch.testing.assert_close(grad, torch.tensor([-1.0, 0.6, 0.4]), atol=1e-6, rtol=0)
    # head coefficients q_v - rho p_v with tail ratio rho = 0.2 / 0.1: -0.7 and -0.3
    assert out.metrics["offpolicy/sc_tail_ratio"] == pytest.approx(2.0, rel=1e-5)
    assert out.metrics["offpolicy/sc_coef_l1"] == pytest.approx(1.0, rel=1e-5)
    padded, _ = _centered_grad(0, 2.0, clip_eps=None, pad=True)
    torch.testing.assert_close(padded, grad, atol=1e-6, rtol=0)


def test_score_centering_cancels_the_drift_of_a_constant_reward():
    # Under a constant advantage the expected update over the sampler's tokens must vanish even
    # though the trainer's distribution differs from the sampler's.
    expected = sum(Q[y] * _centered_grad(y, 1.0, clip_eps=None)[0] for y in range(3))
    torch.testing.assert_close(expected, torch.zeros(3), atol=1e-6, rtol=0)
    logits = P.log().clone().requires_grad_(True)
    uncentered = -(Q * torch.log_softmax(logits, dim=-1)).sum()
    uncentered.backward()
    assert logits.grad.abs().max() > 0.05  # the plain estimator drifts toward the sampler


def test_score_centering_composes_with_the_clipped_ratio():
    # Ratios p/q are 1.2, 1.0 and 0.5 (tail). With clip 0.3 nothing is clipped for A > 0, the
    # surrogate is exact importance sampling and there is nothing left to center.
    grad, out = _centered_grad(0, 2.0, clip_eps=0.3)
    torch.testing.assert_close(grad, -2.0 * 1.2 * torch.tensor([0.4, -0.3, -0.1]), atol=1e-6, rtol=0)
    assert out.metrics["offpolicy/sc_coef_l1"] == pytest.approx(0.0, abs=1e-6)
    # With clip 0.1 the sampled token (ratio 1.2) is clipped: its own score has weight 0 but the
    # expected weighted score 0.3 s_1 + 0.2 * 0.5 s_2 is still subtracted.
    grad, _ = _centered_grad(0, 2.0, clip_eps=0.1)
    torch.testing.assert_close(grad, torch.tensor([-0.48, 0.36, 0.12]), atol=1e-6, rtol=0)


def test_score_centering_is_a_no_op_on_policy():
    grad, out = _centered_grad(1, 2.0, clip_eps=0.2, q=P)
    torch.testing.assert_close(grad, -2.0 * (torch.tensor([0.0, 1.0, 0.0]) - P), atol=1e-6, rtol=0)
    assert out.metrics["offpolicy/sc_coef_l1"] == pytest.approx(0.0, abs=1e-6)


def test_score_centering_rejects_a_separate_proximal_policy():
    with pytest.raises(ValueError, match="behaviour policy"):
        policy_loss(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1),
            algorithm=AlgorithmConfig(),
            behavior_logprobs=torch.zeros(1, 1),
            top_logprobs=torch.zeros(1, 1, 2),
            behavior_top_logprobs=torch.zeros(1, 1, 2),
        )
