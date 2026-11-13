"""Rules of the RunConfig about rewards a backend cannot compute."""

from __future__ import annotations

import pytest

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError


def _config(rewards: list[dict], backend: str, **overrides) -> dict:
    cfg = {
        "model": {"name_or_path": "x"},
        "data": {"uri": "d.jsonl"},
        "rewards": rewards,
        "trainer": {"backend": backend},
    }
    cfg.update(overrides)
    return cfg


@pytest.mark.parametrize("backend", ["verl", "nemo_rl"])
def test_token_lengths_are_rejected_where_the_backend_scores_text_only(backend: str) -> None:
    """verl and NeMo-RL hand rewards the decoded response: a token count would always be 0."""
    nested = {"type": "composite", "params": {"rewards": [{"type": "length", "params": {"unit": "tokens"}}]}}
    for rewards in ([{"type": "length"}], [{"type": "exact_match"}, nested]):
        with pytest.raises(ConfigError, match="unit 'tokens'.*chars"):
            RunConfig.from_dict(_config(rewards, backend))
    for unit in ("chars", "words"):
        RunConfig.from_dict(_config([{"type": "length", "params": {"unit": unit}}], backend))
    # supervised runs score nothing
    RunConfig.from_dict(_config([{"type": "length"}], backend, algorithm={"name": "sft"}))


@pytest.mark.parametrize("backend", ["reference", "trl"])
def test_token_lengths_stay_valid_where_token_ids_reach_the_reward(backend: str) -> None:
    RunConfig.from_dict(_config([{"type": "length"}], backend))
