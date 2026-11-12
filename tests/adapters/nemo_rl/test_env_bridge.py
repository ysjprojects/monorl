from __future__ import annotations

from typing import Any

import pytest
import torch

from monorl.adapters.nemo_rl.compile import build_env_config
from monorl.adapters.nemo_rl.data import monorl_data_processor, sample_to_row
from monorl.adapters.nemo_rl.env_bridge import (
    ACTOR_FQN,
    BREAKDOWN_KEY,
    EPISODE_KEY,
    EnvironmentReturn,
    MonorlEnvironmentBridge,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.registry import registry
from monorl.core.types import Completion, Message, Sample, StepResult


class _Tok:
    """Minimal tokenizer so the processor's ``extra_env_info`` can seed the env metadata."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        return {"input_ids": torch.tensor([[ord(c) % 256 for c in text]])}


def _metadata(sample: Sample) -> dict[str, Any]:
    return monorl_data_processor(sample_to_row(sample), None, _Tok(), 4096, 0)["extra_env_info"]


def _log(sample: Sample, response: str) -> list[dict[str, str]]:
    return [
        *[{"role": m["role"], "content": m["content"]} for m in sample.messages],
        {"role": "assistant", "content": response},
    ]


def test_single_turn_scores_with_monorl_rewards(nemo_config: RunConfig) -> None:
    bridge = MonorlEnvironmentBridge(build_env_config(nemo_config, "/tmp/run"))
    s1 = Sample(id="a", messages=[{"role": "user", "content": "What is 2+2?"}], reference="4")
    s2 = Sample(id="b", messages=[{"role": "user", "content": "What is 3+5?"}], reference="8")
    batch = [_log(s1, "The answer is 4"), _log(s1, "I think 5"), _log(s2, "8")]
    meta = [_metadata(s1), _metadata(s1), _metadata(s2)]

    out = bridge.step(batch, meta)
    assert isinstance(out, EnvironmentReturn)
    observations, metadata, next_stop_strings, rewards, terminateds, answers = out
    assert rewards.tolist() == [1.0, 0.0, 1.0]
    assert terminateds.tolist() == [1.0, 1.0, 1.0]
    assert rewards.dtype == torch.float32
    assert observations == [{"role": "environment", "content": ""}] * 3
    assert next_stop_strings == [None, None, None] and answers is None
    assert all(EPISODE_KEY in m for m in metadata)
    assert len({m[EPISODE_KEY] for m in metadata}) == 3  # one episode per batch item
    assert metadata[0][BREAKDOWN_KEY] == {"exact_match": 1.0}
    assert metadata[1][BREAKDOWN_KEY] == {"exact_match": 0.0}
    assert bridge.num_live_episodes == 0  # finished episodes are closed


class _TwoTurnEnv:
    """Inline multi-turn env: first reply gets a follow-up user turn, second ends with a turn reward."""

    max_turns = 2
    instances: list[_TwoTurnEnv] = []

    def __init__(self) -> None:
        self.turns = 0
        self.closed = False
        type(self).instances.append(self)

    def reset(self, sample: Sample) -> list[Message]:
        self.sample = sample
        return list(sample.messages)

    def step(self, action: Completion) -> StepResult:
        self.turns += 1
        if self.turns == 1:
            return StepResult(
                messages=[{"role": "user", "content": f"you said {action.text!r}; say DONE"}],
                reward=0.0,
                done=False,
            )
        return StepResult(
            messages=[], reward=0.5 if "DONE" in action.text else -0.5, done=True, info={"turns": self.turns}
        )

    def close(self) -> None:
        self.closed = True


def test_multi_turn_keeps_env_state_across_steps(nemo_config: RunConfig) -> None:
    registry.register("env", "two_turn_test", lambda config, **kw: _TwoTurnEnv(), overwrite=True)
    _TwoTurnEnv.instances.clear()
    config = RunConfig.from_dict(
        {**nemo_config.to_dict(), "env": {"type": "two_turn_test", "max_turns": 2}, "rewards": []}
    )
    bridge = MonorlEnvironmentBridge(build_env_config(config))
    assert bridge.reward_fn is None
    sample = Sample(id="s", messages=[{"role": "user", "content": "hi"}], reference=None)

    first = bridge.step([_log(sample, "hello")], [_metadata(sample)])
    assert first.terminateds.tolist() == [0.0] and first.rewards.tolist() == [0.0]
    assert first.observations == [{"role": "user", "content": "you said 'hello'; say DONE"}]
    assert bridge.num_live_episodes == 1 and len(_TwoTurnEnv.instances) == 1

    log = _log(sample, "hello") + [first.observations[0], {"role": "assistant", "content": "DONE"}]
    second = bridge.step([log], [first.metadata[0]])
    assert second.terminateds.tolist() == [1.0] and second.rewards.tolist() == [0.5]
    assert second.observations == [{"role": "environment", "content": ""}]
    assert len(_TwoTurnEnv.instances) == 1 and _TwoTurnEnv.instances[0].closed
    assert bridge.num_live_episodes == 0
    assert second.metadata[0][EPISODE_KEY] == first.metadata[0][EPISODE_KEY]


class _PartsEnv:
    """Never-ending env whose observation ``content`` is a list of parts instead of a string."""

    max_turns = 2
    parts: list[dict[str, Any]] = []

    def reset(self, sample: Sample) -> list[Message]:
        return list(sample.messages)

    def step(self, action: Completion) -> StepResult:
        return StepResult(messages=[{"role": "user", "content": list(self.parts)}], reward=0.0, done=False)

    def close(self) -> None:
        pass


def _parts_bridge(nemo_config: RunConfig, parts: list[dict[str, Any]]) -> MonorlEnvironmentBridge:
    _PartsEnv.parts = parts
    registry.register("env", "parts_test", lambda config, **kw: _PartsEnv(), overwrite=True)
    config = RunConfig.from_dict(
        {**nemo_config.to_dict(), "env": {"type": "parts_test", "max_turns": 2}, "rewards": []}
    )
    return MonorlEnvironmentBridge(build_env_config(config))


def test_text_parts_observation_reaches_nemo_as_plain_text(nemo_config: RunConfig) -> None:
    bridge = _parts_bridge(
        nemo_config, [{"type": "text", "text": "tool says "}, {"type": "text", "text": "42"}]
    )
    sample = Sample(id="s", messages=[{"role": "user", "content": "hi"}], reference=None)
    out = bridge.step([_log(sample, "hello")], [_metadata(sample)])
    assert out.observations == [{"role": "user", "content": "tool says 42"}]
    bridge.shutdown()


def test_media_observation_is_rejected(nemo_config: RunConfig) -> None:
    bridge = _parts_bridge(
        nemo_config, [{"type": "image", "image": "/tmp/x.png"}, {"type": "text", "text": "look"}]
    )
    sample = Sample(id="s", messages=[{"role": "user", "content": "hi"}], reference=None)
    with pytest.raises(ConfigError, match="text observations only"):
        bridge.step([_log(sample, "hello")], [_metadata(sample)])
    bridge.shutdown()


def test_rewards_added_on_final_turn_with_env_rewards(nemo_config: RunConfig) -> None:
    registry.register("env", "two_turn_test", lambda config, **kw: _TwoTurnEnv(), overwrite=True)
    config = RunConfig.from_dict({**nemo_config.to_dict(), "env": {"type": "two_turn_test", "max_turns": 2}})
    bridge = MonorlEnvironmentBridge(build_env_config(config))
    sample = Sample(id="s", messages=[{"role": "user", "content": "What is 2+2?"}], reference="4")
    first = bridge.step([_log(sample, "4")], [_metadata(sample)])
    log = _log(sample, "4") + [first.observations[0], {"role": "assistant", "content": "DONE 4"}]
    second = bridge.step([log], [first.metadata[0]])
    assert second.rewards.tolist() == [1.5]  # 0.5 env turn reward + exact_match 1.0 on the full trajectory


def test_global_post_process_and_shutdown(nemo_config: RunConfig) -> None:
    registry.register("env", "two_turn_test", lambda config, **kw: _TwoTurnEnv(), overwrite=True)
    _TwoTurnEnv.instances.clear()
    bridge = MonorlEnvironmentBridge(build_env_config(nemo_config))
    batch, metrics = bridge.global_post_process_and_metrics(
        {"total_reward": torch.tensor([1.0, 0.0, 1.0, 0.0])}
    )
    assert torch.equal(batch["total_reward"], torch.tensor([1.0, 0.0, 1.0, 0.0]))

    _, multi = bridge.global_post_process_and_metrics({"rewards": torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
    assert multi["reward_mean"] == 2.0  # first reward component

    _, empty = bridge.global_post_process_and_metrics({})
    assert "reward_mean" not in empty

    config = RunConfig.from_dict({**nemo_config.to_dict(), "env": {"type": "two_turn_test", "max_turns": 2}})
    live = MonorlEnvironmentBridge(build_env_config(config))
    sample = Sample(id="s", messages=[{"role": "user", "content": "hi"}])
    live.step([_log(sample, "x")], [_metadata(sample)])
    assert live.num_live_episodes == 1
    live.shutdown()
    assert live.num_live_episodes == 0 and _TwoTurnEnv.instances[-1].closed


def test_actor_fqn_points_at_ray_module() -> None:
    assert ACTOR_FQN == "monorl.adapters.nemo_rl.env_actor.MonorlEnvironment"
