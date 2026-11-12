"""Multi-turn prompts built by token continuation, with the real ``hf`` engine and the reference
trainer on the tiny random Llama, under a chat template that trims message content (as the
Llama 3 and Gemma templates do). A completion that starts or ends with whitespace is then not what
the template renders back: the case where re-rendering the chat breaks the token sequence."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.types import Sample, SamplingParams, Trajectory, Turn
from monorl.orchestrator.batching import build_train_batch, trajectory_rows
from monorl.rollout.hf_engine import HFRolloutEngine
from monorl.rollout.render import continue_prompt_ids
from monorl.trainer.reference import ReferenceTrainer

TRIMMING = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' + m['role'] + '\n' + (m['content'] | trim) + '<|im_end|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)
MESSAGES = [{"role": "user", "content": "Say hello."}]
FOLLOW_UPS = [[{"role": "user", "content": "And again?"}], [{"role": "user", "content": "Once more."}]]


@pytest.fixture(scope="module")
def trimming_engine(tiny_config: RunConfig) -> Iterator[HFRolloutEngine]:
    engine = HFRolloutEngine(tiny_config)
    engine.tokenizer.chat_template = TRIMMING
    yield engine
    engine.shutdown()


def _params(seed: int) -> SamplingParams:
    return SamplingParams(n=1, max_new_tokens=12, temperature=1.0, seed=seed)


@pytest.fixture(scope="module")
def seed(trimming_engine: HFRolloutEngine) -> int:
    """A seed whose first completion has surrounding whitespace (most do: about half of the BPE
    vocabulary starts with a space), so that the template would not render it back as sampled."""
    for candidate in range(64):
        text = trimming_engine.generate([MESSAGES], _params(candidate))[0][0].text
        if text and text != text.strip():
            return candidate
    raise AssertionError("no seed in 0..63 gave a completion with surrounding whitespace")


def _episode(engine: HFRolloutEngine, mode: str, seed: int) -> Trajectory:
    """Three turns: the chat, then two follow-ups whose prompts come from ``continue_prompt_ids``."""
    messages = list(MESSAGES)
    completion = engine.generate([messages], _params(seed))[0][0]
    turns = [Turn(prompt_token_ids=list(completion.prompt_token_ids), completion=completion)]
    for new in FOLLOW_UPS:
        last = turns[-1]
        prompt = continue_prompt_ids(
            engine.tokenizer,
            previous_messages=messages,
            previous_prompt_ids=last.prompt_token_ids,
            completion=last.completion,
            new_messages=new,
            mode=mode,
        ).ids
        messages = [*messages, {"role": "assistant", "content": last.completion.text}, *new]
        completion = engine.generate([prompt], _params(seed))[0][0]
        turns.append(Turn(prompt_token_ids=prompt, completion=completion))
    return Trajectory(sample=Sample(id="s", messages=MESSAGES), group_id="g", messages=messages, turns=turns)


def _batch_and_logprobs(trajectory: Trajectory, engine: HFRolloutEngine, config: RunConfig):
    batch, stats = build_train_batch(
        [trajectory], [1.0], pad_token_id=engine.tokenizer.pad_token_id, max_seq_len=512
    )
    return batch, stats, ReferenceTrainer(config).compute_logprobs(batch)


def test_token_continuation_trains_a_multi_turn_episode_as_one_sequence(
    trimming_engine: HFRolloutEngine, tiny_config: RunConfig, seed: int
) -> None:
    trajectory = _episode(trimming_engine, "tokens", seed)
    assert len(trajectory_rows(trajectory)) == 1
    batch, stats, logprobs = _batch_and_logprobs(trajectory, trimming_engine, tiny_config)
    assert stats["tokens/prefix_mismatch"] == 0 and len(batch) == 1
    assert batch.num_response_tokens == sum(len(t.completion.token_ids) for t in trajectory.turns)
    # one forward pass over the whole episode reproduces the logprobs every turn was sampled with
    torch.testing.assert_close(logprobs, batch.old_logprobs, atol=1e-3, rtol=0)


def test_template_rendering_splits_the_same_episode_without_losing_tokens(
    trimming_engine: HFRolloutEngine, tiny_config: RunConfig, seed: int
) -> None:
    trajectory = _episode(trimming_engine, "template", seed)
    assert len(trajectory_rows(trajectory)) > 1
    batch, stats, logprobs = _batch_and_logprobs(trajectory, trimming_engine, tiny_config)
    assert stats["tokens/prefix_mismatch"] == 1 and len(batch) > 1
    assert batch.num_response_tokens == sum(len(t.completion.token_ids) for t in trajectory.turns)
    torch.testing.assert_close(logprobs, batch.old_logprobs, atol=1e-3, rtol=0)
