from __future__ import annotations

import pytest

from monorl.config.schema import RewardSpec
from monorl.core.interfaces import RewardFunction
from monorl.core.registry import registry
from monorl.rewards import build_reward
from monorl.rewards.base import extract_boxed, extract_code_block, extract_last_number

# -- extraction helpers --------------------------------------------------------------------


def test_extractors():
    assert extract_last_number("costs $1,250.50 so total 3") == "3"
    assert extract_last_number("answer is -4.5e2") == "-4.5e2"
    assert extract_last_number("no digits") is None
    assert extract_boxed(r"so \boxed{\frac{1}{2}} and \boxed{ 42 }") == " 42 "
    assert extract_boxed(r"\boxed{unbalanced") is None
    assert extract_code_block("```python\nprint(1)\n```") == "print(1)\n"
    assert extract_code_block("```\nx=1\n```\n```python\nx=2\n```") == "x=2\n"
    assert extract_code_block("```go\nfmt.Println()\n```") is None


# -- exact_match --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params,response,reference,expected",
    [
        ({}, "The answer is 42.", "42", 1.0),
        ({}, "The answer is 42.", "43", 0.0),
        ({}, "Total: 1,000 dollars", "1000", 1.0),
        ({"extract": "boxed"}, r"so \boxed{Paris}", "paris", 1.0),
        ({"extract": "boxed", "normalize": ["whitespace"]}, r"\boxed{Paris}", "paris", 0.0),
        ({"extract": "after_hash"}, "reasoning...\n#### 17", "The sum is 10 + 7\n#### 17", 1.0),
        ({"extract": "none"}, "  Hello   World ", "hello world", 1.0),
        ({"extract": "none"}, "0.3333", "1/3", 0.0),
        ({"numeric_tolerance": 1e-3}, "pi is 3.1416", "3.14159", 1.0),
        ({}, "answers 5", ["4", "5"], 1.0),
        ({}, "answers 5", {"answer": "5"}, 1.0),
        ({}, "nothing numeric", "5", 0.0),
    ],
)
def test_exact_match(traj, params, response, reference, expected):
    fn = registry.get("reward", "exact_match")(params)
    assert isinstance(fn, RewardFunction)
    [res] = fn([traj(response, reference)])
    assert res.score == expected
    assert res.breakdown == {"exact_match": expected}


def test_exact_match_scope_all_and_reference_key(traj):
    fn = registry.get("reward", "exact_match")({"scope": "all", "reference_key": "final"})
    t = traj("done", {"final": "7", "other": "9"}, earlier_turns=["the value is 7"])
    assert fn([t])[0].score == 1.0
    assert fn([traj("the value is 7", {"final": "7"})])[0].score == 1.0
    assert fn([traj("the value is 9", {"final": "7"})])[0].score == 0.0


def test_exact_match_rejects_bad_params():
    with pytest.raises(ValueError):
        registry.get("reward", "exact_match")({"extract": "magic"})
    with pytest.raises(ValueError):
        registry.get("reward", "exact_match")({"normalize": ["unicode"]})


# -- format -------------------------------------------------------------------------------


def test_format_require_all_and_fraction(traj):
    patterns = [r"<think>.*</think>", r"<answer>.*</answer>"]
    strict = registry.get("reward", "format")({"patterns": patterns})
    partial = registry.get("reward", "format")({"patterns": patterns, "require_all": False, "name": "fmt"})
    good = traj("<think>\nhmm\n</think><answer>4</answer>")
    half = traj("<think>hmm</think> 4")
    results = strict([good, half])
    assert [r.score for r in results] == [1.0, 0.0]
    assert results[0].info["matched"] == [True, True]
    results = partial([good, half])
    assert [r.score for r in results] == [1.0, 0.5]
    assert results[1].breakdown == {"fmt": 0.5}


def test_format_flags_and_fullmatch(traj):
    fn = registry.get("reward", "format")({"pattern": r"yes", "flags": ["IGNORECASE"], "fullmatch": True})
    assert fn([traj("YES")])[0].score == 1.0
    assert fn([traj("yes!")])[0].score == 0.0
    with pytest.raises(ValueError):
        registry.get("reward", "format")({})


# -- length -------------------------------------------------------------------------------


def test_length_reward_band_and_penalty(traj):
    fn = registry.get("reward", "length")({"min_tokens": 2, "max_tokens": 4, "penalty_per_token": 0.25})
    scores = [
        r.score for r in fn([traj("a b c"), traj("a b c d e f"), traj("a"), traj("a b c d e f g h i j")])
    ]
    assert scores == [1.0, 0.5, 0.75, 0.0]
    chars = registry.get("reward", "length")(
        {"max_tokens": 3, "unit": "chars", "penalty_per_token": 0.1, "base": 0.0, "min_reward": -1.0}
    )
    assert chars([traj("abcdef")])[0].score == pytest.approx(-0.3)
    assert chars([traj("abcdef")])[0].info["length"] == 6
    with pytest.raises(ValueError):
        registry.get("reward", "length")({"min_tokens": 5, "max_tokens": 2})


# -- math_verify --------------------------------------------------------------------------


def test_math_verify(traj):
    pytest.importorskip("math_verify")
    fn = registry.get("reward", "math_verify")({})
    cases = [
        (r"The answer is $\boxed{\frac{1}{2}}$", "0.5", 1.0),
        (r"so $x = 2$", r"$\boxed{2}$", 1.0),
        (r"Final answer: 42", "42", 1.0),
        (r"$\boxed{3}$", "4", 0.0),
        ("I have no idea", "4", 0.0),
        (r"$\boxed{(1, 2)}$", ["(2, 1)", "(1, 2)"], 1.0),
    ]
    results = fn([traj(resp, ref) for resp, ref, _ in cases])
    assert [r.score for r in results] == [c[2] for c in cases]


def test_math_verify_runs_off_main_thread(traj):
    pytest.importorskip("math_verify")
    from concurrent.futures import ThreadPoolExecutor

    fn = registry.get("reward", "math_verify")({})
    with ThreadPoolExecutor(1) as pool:
        [res] = pool.submit(fn, [traj(r"$\boxed{10}$", "10")]).result()
    assert res.score == 1.0


# -- composite / build_reward -------------------------------------------------------------


def test_build_reward_weighted_sum_with_breakdown(traj):
    specs = [
        RewardSpec(type="exact_match", weight=2.0),
        RewardSpec(type="format", weight=0.5, params={"patterns": [r"\d"], "name": "has_digit"}),
        RewardSpec(type="format", weight=0.25, params={"patterns": [r"^#"]}),
        RewardSpec(type="format", weight=0.25, params={"patterns": [r"!$"]}),
    ]
    fn = build_reward(specs)
    assert isinstance(fn, RewardFunction)
    results = fn([traj("# answer 4!", "4"), traj("nope", "4")])
    assert results[0].score == pytest.approx(2.0 + 0.5 + 0.25 + 0.25)
    assert results[0].breakdown == {"exact_match": 1.0, "has_digit": 1.0, "format": 1.0, "format#2": 1.0}
    assert results[1].score == 0.0
    assert results[1].breakdown == {"exact_match": 0.0, "has_digit": 0.0, "format": 0.0, "format#2": 0.0}
    assert results[0].info["exact_match"]["extracted"] == "4"


def test_build_reward_accepts_dicts_and_registered_composite(traj):
    fn = build_reward(
        [{"type": "exact_match"}, {"type": "length", "weight": 0.1, "params": {"max_tokens": 1}}]
    )
    [res] = fn([traj("4 4", "4")])
    assert res.score == pytest.approx(1.0 + 0.1 * 0.99)
    nested = registry.get("reward", "composite")(
        {"name": "outer", "rewards": [{"type": "exact_match", "weight": 3.0}]}
    )
    assert nested.name == "outer"
    assert nested([traj("4", "4")])[0].score == 3.0
    with pytest.raises(ValueError):
        build_reward([])


def test_rewards_reject_non_trajectories():
    fn = registry.get("reward", "exact_match")({})
    with pytest.raises(TypeError):
        fn(["just a string"])  # type: ignore[list-item]
