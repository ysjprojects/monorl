"""Turn-level credit assignment: per-turn returns, group baselines for unequal turn counts and
turn-level GAE, against hand-computed numbers."""

from __future__ import annotations

import pytest
import torch

from monorl.algorithms import (
    compute_advantages,
    explained_variance,
    gae,
    turn_advantages,
    turn_metrics,
    turn_returns,
    turn_table,
    value_metrics,
)
from monorl.config.schema import AlgorithmConfig

# Three members of prompt group "g" with 3, 2 and 1 turns, and a two-turn member of group "h".
TURN_REWARDS = [[1.0, 0.0, 2.0], [0.0, 3.0], [4.0], [1.0, 1.0]]
GROUPS = ["g", "g", "g", "h"]
# gamma = 1 returns from every turn on; the last column of each row is that member's final turn.
RETURNS = [[3.0, 2.0, 2.0], [3.0, 3.0, 0.0], [4.0, 0.0, 0.0], [2.0, 1.0, 0.0]]

ESTIMATORS = [
    {"name": "grpo"},
    {"name": "grpo", "advantage_std_norm": False},
    {"name": "dapo", "advantage_norm": "batch"},
    {"name": "grpo", "advantage_norm": "none"},
    {"name": "rloo"},
    {"name": "reinforce"},
]


def _table(rows=TURN_REWARDS, gamma: float = 1.0):
    rewards, mask = turn_table(rows)
    return turn_returns(rewards, mask, gamma), mask


def test_turn_table_pads_and_masks() -> None:
    rewards, mask = turn_table(TURN_REWARDS)
    assert rewards.tolist() == [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    assert mask.tolist() == [
        [True, True, True],
        [True, True, False],
        [True, False, False],
        [True, True, False],
    ]


def test_returns_count_only_the_rewards_from_a_turn_on() -> None:
    returns, _ = _table()
    assert returns.tolist() == RETURNS
    discounted, _ = _table(gamma=0.5)
    # [1, 0, 2]: G2 = 2, G1 = 0 + 0.5 * 2, G0 = 1 + 0.5 * 1;  [0, 3]: G1 = 3, G0 = 1.5
    torch.testing.assert_close(
        discounted, torch.tensor([[1.5, 1.0, 2.0], [1.5, 3.0, 0.0], [4.0, 0.0, 0.0], [1.5, 1.0, 0.0]])
    )


def test_rloo_compares_a_turn_with_the_same_turn_of_the_other_members() -> None:
    returns, mask = _table()
    adv = turn_advantages(returns, mask, GROUPS, algorithm=AlgorithmConfig(name="rloo"))
    # Group g, aligned returns per turn index (a member that ended stands in with its last turn):
    #   turn 0: [3, 3, 4]   turn 1: [2, 3, 4*]   turn 2: [2, 3*, 4*]
    # leave-one-out baselines: turn 0: 3.5, 3.5, 3;  turn 1: 3.5 (a), 3 (b);  turn 2: 3.5 (a).
    # Group h has one member: no baseline.
    expected = torch.tensor([[-0.5, -1.5, -1.5], [-0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    torch.testing.assert_close(adv, expected)


def test_group_mean_baseline_shrinks_every_turn_by_the_same_factor() -> None:
    returns, mask = _table()
    cfg = AlgorithmConfig(name="grpo", advantage_std_norm=False)
    adv = turn_advantages(returns, mask, GROUPS, algorithm=cfg)
    # group means per turn index: 10/3, 3, 3 over all three members of g, whatever their length
    expected = torch.tensor([[-1 / 3, -1.0, -1.0], [-1 / 3, 0.0, 0.0], [2 / 3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(adv, expected, atol=1e-6, rtol=0)
    # ... which is the leave-one-out advantage times (1 - 1/n) with n = 3 at every turn, the factor
    # the episode-level estimator has: no turn is shrunk more because fewer members reached it.
    loo = turn_advantages(returns, mask, GROUPS, algorithm=AlgorithmConfig(name="rloo"))
    torch.testing.assert_close(adv[:3], loo[:3] * (2 / 3), atol=1e-6, rtol=0)


def test_group_std_normalisation_per_turn_index() -> None:
    returns, mask = _table()
    adv = turn_advantages(returns, mask, GROUPS, algorithm=AlgorithmConfig(name="grpo"))
    s0 = (1 / 3) ** 0.5  # unbiased std of [3, 3, 4]; [2, 3, 4] has std 1
    expected = torch.tensor(
        [[-1 / 3 / s0, -1.0, -1.0], [-1 / 3 / s0, 0.0, 0.0], [2 / 3 / s0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    torch.testing.assert_close(adv, expected, atol=1e-4, rtol=0)


def test_batch_normalisation_per_turn_index() -> None:
    returns, mask = _table()
    cfg = AlgorithmConfig(name="dapo", advantage_norm="batch")
    adv = turn_advantages(returns, mask, GROUPS, algorithm=cfg)
    # turn 0: [3, 3, 4, 2] mean 3, std sqrt(2/3);  turns 1 and 2: [2, 3, 4, 1] mean 2.5, std sqrt(5/3)
    s0, s1 = (2 / 3) ** 0.5, (5 / 3) ** 0.5
    expected = torch.tensor(
        [
            [0.0, -0.5 / s1, -0.5 / s1],
            [0.0, 0.5 / s1, 0.0],
            [1.0 / s0, 0.0, 0.0],
            [-1.0 / s0, -1.5 / s1, 0.0],
        ]
    )
    torch.testing.assert_close(adv, expected, atol=1e-4, rtol=0)


def test_reinforce_and_unnormalised_grpo_are_the_returns() -> None:
    returns, mask = _table()
    for cfg in (AlgorithmConfig(name="reinforce"), AlgorithmConfig(name="grpo", advantage_norm="none")):
        torch.testing.assert_close(turn_advantages(returns, mask, GROUPS, algorithm=cfg), returns)


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda e: "-".join(map(str, e.values())))
def test_rewards_of_earlier_turns_do_not_reach_a_turn(estimator: dict) -> None:
    cfg = AlgorithmConfig(**estimator)
    returns, mask = _table()
    base = turn_advantages(returns, mask, GROUPS, algorithm=cfg)
    # other first-turn rewards for every member that has later turns
    changed_rows = [[100.0, 0.0, 2.0], [-7.0, 3.0], [4.0], [50.0, 1.0]]
    returns, mask = _table(changed_rows)
    changed = turn_advantages(returns, mask, GROUPS, algorithm=cfg)
    torch.testing.assert_close(changed[:, 1:], base[:, 1:])
    assert not torch.allclose(changed[:, 0], base[:, 0])


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda e: "-".join(map(str, e.values())))
def test_rewards_at_the_final_turn_only_give_the_episode_level_advantages(estimator: dict) -> None:
    """Without per-turn rewards (and undiscounted) there is nothing to reassign: every turn of an
    episode gets exactly the episode-level advantage, whatever the turn counts."""
    cfg = AlgorithmConfig(**estimator)
    episode_rewards = torch.tensor([1.0, 0.0, 0.5, 2.0, 3.0, 3.0, -1.0])
    groups = ["a", "a", "a", "b", "b", "b", "c"]
    counts = [3, 1, 2, 2, 4, 1, 2]
    rows = [[0.0] * (n - 1) + [float(r)] for n, r in zip(counts, episode_rewards, strict=True)]
    returns, mask = _table(rows)
    adv = turn_advantages(returns, mask, groups, algorithm=cfg)
    expected = compute_advantages(episode_rewards, groups, algorithm=cfg)
    assert torch.equal(adv, expected[:, None] * mask)


@pytest.mark.parametrize("estimator", ESTIMATORS, ids=lambda e: "-".join(map(str, e.values())))
def test_single_turn_episodes_are_the_episode_level_estimator(estimator: dict) -> None:
    cfg = AlgorithmConfig(**estimator)
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 5.0])
    groups = ["a", "a", "a", "b", "b", "b", "c"]
    returns, mask = _table([[float(r)] for r in rewards], gamma=0.3)
    adv = turn_advantages(returns, mask, groups, algorithm=cfg)
    assert torch.equal(adv[:, 0], compute_advantages(rewards, groups, algorithm=cfg))


def test_turn_advantages_validates_shapes() -> None:
    returns, mask = _table()
    cfg = AlgorithmConfig(name="rloo")
    with pytest.raises(ValueError, match="group_ids"):
        turn_advantages(returns, mask, GROUPS[:-1], algorithm=cfg)
    with pytest.raises(ValueError, match="shape"):
        turn_advantages(returns[:, :2], mask, GROUPS, algorithm=cfg)
    with pytest.raises(ValueError, match="at least one turn"):
        turn_advantages(returns, torch.zeros_like(mask), GROUPS, algorithm=cfg)
    empty = turn_advantages(torch.zeros(0, 0), torch.zeros(0, 0, dtype=torch.bool), [], algorithm=cfg)
    assert empty.shape == (0, 0)


def test_turn_level_gae_matches_a_manual_computation() -> None:
    rewards, mask = turn_table([[1.0, 0.0, 2.0], [0.0, 3.0]])
    values = torch.tensor([[0.5, 1.0, -1.0], [2.0, 0.5, 9.9]])  # 9.9 sits on a turn that does not exist
    adv, returns = gae(rewards, values, mask, gamma=0.9, lam=0.8)
    # row 0: d2 = 2 + 1 = 3;  d1 = 0 + 0.9 * (-1) - 1 = -1.9;  d0 = 1 + 0.9 * 1 - 0.5 = 1.4
    #        A2 = 3;  A1 = -1.9 + 0.72 * 3 = 0.26;  A0 = 1.4 + 0.72 * 0.26 = 1.5872
    # row 1: d1 = 3 - 0.5 = 2.5 (the episode ended);  d0 = 0 + 0.9 * 0.5 - 2 = -1.55
    #        A1 = 2.5;  A0 = -1.55 + 0.72 * 2.5 = 0.25
    torch.testing.assert_close(adv, torch.tensor([[1.5872, 0.26, 3.0], [0.25, 2.5, 0.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        returns, torch.tensor([[2.0872, 1.26, 2.0], [2.25, 3.0, 0.0]]), atol=1e-6, rtol=0
    )


def test_gae_without_a_critic_is_the_return_and_lambda_one_subtracts_the_value() -> None:
    rewards, mask = turn_table(TURN_REWARDS)
    returns = turn_returns(rewards, mask, 0.5)
    values = torch.tensor([[0.3, -0.2, 1.0], [0.1, 0.4, 0.0], [2.0, 0.0, 0.0], [0.0, 0.7, 0.0]])
    adv, targets = gae(rewards, values, mask, gamma=0.5, lam=1.0)
    torch.testing.assert_close(adv, (returns - values) * mask, atol=1e-6, rtol=0)
    torch.testing.assert_close(targets, returns, atol=1e-6, rtol=0)
    # lam = 0: the one-step TD error; a turn's advantage sees only its own reward and two values
    td, _ = gae(rewards, values, mask, gamma=0.5, lam=0.0)
    assert td[0].tolist() == pytest.approx([1.0 + 0.5 * -0.2 - 0.3, 0.0 + 0.5 * 1.0 + 0.2, 2.0 - 1.0])


def test_explained_variance_and_value_metrics() -> None:
    returns = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(returns, returns) == pytest.approx(1.0)
    assert explained_variance(torch.full((4,), 2.5), returns) == pytest.approx(0.0)
    # residuals [0, 0, 0, -1]: variance 3/16 against 5/4 of the returns
    assert explained_variance(torch.tensor([1.0, 2.0, 3.0, 5.0]), returns) == pytest.approx(0.85)
    assert explained_variance(-returns, returns) < 0
    mask = torch.tensor([True, True, False, False])
    assert explained_variance(torch.tensor([1.0, 2.0, 9.0, 9.0]), returns, mask) == pytest.approx(1.0)
    assert explained_variance(returns[:1], returns[:1]) != explained_variance(returns[:1], returns[:1])  # nan

    metrics = value_metrics(torch.tensor([[1.0, 2.0, 3.0, 5.0]]), returns[None], torch.ones(1, 4))
    assert metrics == pytest.approx(
        {"value/mean": 2.75, "value/return_mean": 2.5, "value/explained_variance": 0.85}
    )
    constant = value_metrics(torch.zeros(1, 2), torch.ones(1, 2), torch.ones(1, 2))
    assert constant == {"value/mean": 0.0, "value/return_mean": 1.0}  # nothing to explain: no nan logged
    assert value_metrics(torch.zeros(1, 2), torch.ones(1, 2), torch.zeros(1, 2)) == {}


def test_turn_metrics_measure_the_spread_inside_an_episode() -> None:
    adv, mask = turn_table([[1.0, -1.0, 3.0], [2.0, 2.0]])
    assert turn_metrics(adv, mask) == pytest.approx(
        {
            "advantage/turn_spread": 2.0,  # (4 + 0) / 2
            "advantage/turn_spread_max": 4.0,
            "advantage/first_turn_abs": 1.5,
            "advantage/last_turn_abs": 2.5,
        }
    )
