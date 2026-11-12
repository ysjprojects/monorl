"""``algorithm.sft_turns``: which assistant turns of a conversation carry the SFT loss, the
token-exact single sequence, the counted fallbacks and the reported truncation."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch
from monorl.core.reporting import RunReporter, iter_metrics, iter_trajectories
from monorl.orchestrator import run
from monorl.orchestrator.sft import (
    SFTLoop,
    build_sft_batch,
    encode_sft_conversation,
    encode_sft_example,
    trained_turns,
)
from monorl.trainer.reference import ReferenceTrainer

CALL = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call>'
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an expression.",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
        },
    }
]


def _agent_chat(answer: str = "It is 4.") -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "Use the calculator."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": CALL},
        {"role": "tool", "name": "calculator", "tool_call_id": "call_1", "content": "4"},
        {"role": "assistant", "content": answer},
    ]


@pytest.fixture(scope="module")
def tokenizer(tiny_model_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model_dir)


def _ids(tokenizer: Any, text: str) -> list[int]:
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


def _render(tokenizer: Any, messages: Sequence[dict[str, Any]], *, generation_prompt: bool) -> list[int]:
    text = tokenizer.apply_chat_template(
        list(messages), add_generation_prompt=generation_prompt, tokenize=False
    )
    return _ids(tokenizer, text)


def _final_turn_encoding(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    train_on_prompt: bool = False,
    max_seq_len: int | None = None,
) -> tuple[list[int], list[int], int, bool, bool]:
    """The encoding SFT had before ``sft_turns`` existed, spelled out as the reference for
    ``last``: prompt = chat before the final turn + generation prompt, response = the rest of the
    full rendering, or the response text + EOS when the prompt is not a prefix of it."""
    prompt_ids = _render(tokenizer, messages[:-1], generation_prompt=True)
    full_ids = _render(tokenizer, messages, generation_prompt=False)
    mismatch = full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids)
    if mismatch:
        response_ids = _ids(tokenizer, messages[-1]["content"])
        if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
            response_ids.append(tokenizer.eos_token_id)
        full_ids = prompt_ids + response_ids
    if train_on_prompt:
        mask = [0] + [1] * (len(full_ids) - 1)
    else:
        mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    truncated = max_seq_len is not None and len(full_ids) > max_seq_len
    if truncated:
        full_ids, mask = full_ids[:max_seq_len], mask[:max_seq_len]
    return full_ids, mask, len(prompt_ids), mismatch, truncated


class _ReasoningDroppingTokenizer:
    """A chat template that rewrites history the way Qwen3 / DeepSeek-R1 do: the reasoning of an
    assistant turn is rendered only while that turn is the last message."""

    chat_template = "drops-earlier-reasoning"

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def apply_chat_template(
        self, messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool, tokenize: bool
    ) -> str:
        text = ""
        for i, message in enumerate(messages):
            content = message["content"]
            if message["role"] == "assistant" and i < len(messages) - 1:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
            text += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text: str, add_special_tokens: bool = False) -> Any:
        return self.tokenizer(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids)


class _GenerationPromptRewritingTokenizer(_ReasoningDroppingTokenizer):
    """A template whose generation prompt does not survive in the rendering of the closed turn."""

    chat_template = "rewrites-the-generation-prompt"

    def apply_chat_template(
        self, messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool, tokenize: bool
    ) -> str:
        if add_generation_prompt:
            return "USER: " + messages[-1]["content"] + " ASSISTANT:"
        return "<<" + " ".join(m["content"] for m in messages) + ">>"


# -- last ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("train_on_prompt", [False, True])
@pytest.mark.parametrize("max_seq_len", [None, 40, 12])
def test_last_is_the_final_turn_encoding_as_it_always_was(
    tokenizer: Any, train_on_prompt: bool, max_seq_len: int | None
) -> None:
    chats = [
        [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}],
        _agent_chat(),  # earlier assistant and tool turns are prompt
        [{"role": "user", "content": " spaced  out "}, {"role": "assistant", "content": "  ok \n"}],
    ]
    options = {"train_on_prompt": train_on_prompt, "max_seq_len": max_seq_len}
    for tok in (tokenizer, _GenerationPromptRewritingTokenizer(tokenizer)):
        for chat in chats:
            ids, mask, prompt_len, mismatch, truncated = _final_turn_encoding(tok, chat, **options)
            (as_conversation,) = encode_sft_conversation(tok, chat, turns="last", **options)
            for example in (encode_sft_example(tok, chat, **options), as_conversation):
                assert example.input_ids == ids and example.response_mask == mask
                assert example.prompt_len == prompt_len
                assert (example.prefix_mismatch, example.truncated) == (mismatch, truncated)
    assert _final_turn_encoding(tokenizer, chats[0])[3] is False  # the real template chains exactly
    assert _final_turn_encoding(_GenerationPromptRewritingTokenizer(tokenizer), chats[0])[3] is True


# -- all ----------------------------------------------------------------------------------------


def test_all_trains_every_assistant_turn_in_one_sequence(tokenizer: Any) -> None:
    chat = _agent_chat()
    (example,) = encode_sft_conversation(tokenizer, chat, turns="all")
    assert not example.prefix_mismatch and not example.truncated and example.truncated_turns == 0
    # one sequence: the conversation as the template renders it, nothing re-tokenised
    assert example.input_ids == _render(tokenizer, chat, generation_prompt=False)
    first, second = example.spans
    assert (first.message_index, second.message_index) == (2, 4)
    # each span starts where the model would start generating that turn ...
    assert example.input_ids[: first.start] == _render(tokenizer, chat[:2], generation_prompt=True)
    assert example.input_ids[: second.start] == _render(tokenizer, chat[:4], generation_prompt=True)
    # ... and holds the turn's text and its end-of-turn token
    assert tokenizer.decode(example.input_ids[first.start : first.end]) == CALL + "<|im_end|>\n"
    assert tokenizer.decode(example.input_ids[second.start : second.end]) == "It is 4.<|im_end|>\n"
    expected_mask = [
        int(first.start <= i < first.end or second.start <= i < second.end)
        for i in range(len(example.input_ids))
    ]
    assert example.response_mask == expected_mask and example.prompt_len == first.start

    (last,) = encode_sft_conversation(tokenizer, chat, turns="last")
    assert last.input_ids == example.input_ids
    assert last.response_mask == [int(second.start <= i < second.end) for i in range(len(last.input_ids))]
    # one assistant turn: nothing to choose
    single = chat[:2] + [{"role": "assistant", "content": "4"}]
    assert encode_sft_conversation(tokenizer, single, turns="all") == encode_sft_conversation(
        tokenizer, single, turns="last"
    )


def test_weight_zero_turns_are_context(tokenizer: Any) -> None:
    chat = _agent_chat()
    chat[2] = {**chat[2], "weight": 0}
    assert trained_turns(chat, "all") == [4]
    (example,) = encode_sft_conversation(tokenizer, chat, turns="all")
    plain = _agent_chat()
    (last,) = encode_sft_conversation(tokenizer, plain, turns="last")
    assert example.input_ids == last.input_ids and example.response_mask == last.response_mask
    # the final message is the target whatever it says about itself
    chat[4] = {**chat[4], "weight": 0}
    assert trained_turns(chat, "all") == [4]


def test_truncation_is_counted_per_sequence_and_per_turn(tokenizer: Any) -> None:
    chat = _agent_chat()
    (whole,) = encode_sft_conversation(tokenizer, chat, turns="all")
    first, second = whole.spans
    cases = {
        second.start + 2: (1, [first.end - first.start, 2]),  # inside the second turn
        second.start - 1: (1, [first.end - first.start, 0]),  # the second turn is gone
        first.start + 3: (2, [3, 0]),  # inside the first turn
    }
    for max_seq_len, (lost, span_lengths) in cases.items():
        (cut,) = encode_sft_conversation(tokenizer, chat, turns="all", max_seq_len=max_seq_len)
        assert cut.truncated and cut.input_ids == whole.input_ids[:max_seq_len]
        assert cut.truncated_turns == lost and [s.end - s.start for s in cut.spans] == span_lengths
        assert sum(cut.response_mask) == sum(span_lengths)
    (cut,) = encode_sft_conversation(tokenizer, chat, turns="all", max_seq_len=second.start + 2)
    _, stats = build_sft_batch([whole, cut], pad_token_id=0, group_ids=["a", "b"])
    assert stats["tokens/truncated"] == 1.0 and stats["tokens/truncated_turns"] == 1.0
    assert stats["tokens/sequences"] == 2.0 and stats["sft/turns"] == 4.0


def test_template_that_rewrites_history_gives_each_turn_its_own_sequence(tokenizer: Any) -> None:
    tok = _ReasoningDroppingTokenizer(tokenizer)
    chat = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>two plus two</think>4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "<think>three plus three</think>6"},
    ]
    first, second = encode_sft_conversation(tok, chat, turns="all")
    # every turn is trained exactly once, as the policy would produce it: with its reasoning
    assert tok.decode(first.response_ids) == "<think>two plus two</think>4<|im_end|>\n"
    assert tok.decode(second.response_ids) == "<think>three plus three</think>6<|im_end|>\n"
    assert [s.message_index for s in first.spans] == [1] and [s.message_index for s in second.spans] == [3]
    # the second sequence has the context the template gives the second turn: reasoning dropped
    context = tok.decode(second.input_ids[: second.prompt_len])
    assert "<think>" not in context and "<|im_start|>assistant\n4<|im_end|>" in context
    assert not first.prefix_mismatch and not second.prefix_mismatch  # both spans come from the template

    # a template whose generation prompt is not a prefix of the closed turn: text + EOS appended
    broken = _GenerationPromptRewritingTokenizer(tokenizer)
    one, two = encode_sft_conversation(broken, chat, turns="all")
    assert one.prefix_mismatch and two.prefix_mismatch
    assert two.response_ids == [*_ids(tokenizer, chat[3]["content"]), tokenizer.eos_token_id]


# -- through the loop ---------------------------------------------------------------------------


@pytest.fixture
def agent_sft_config(run_config_dict: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    data = tmp_path / "agent.jsonl"
    rows = [{"messages": _agent_chat(f"It is {n}."), "tools": TOOLS, "num_turns": 2} for n in (4, 5, 6, 7)]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {
        **run_config_dict,
        "run_name": "sft-agent",
        "data": {"uri": str(data), "format": "chat", "max_prompt_tokens": 256},
        "algorithm": {"name": "sft", "sft_turns": "all"},
        "rewards": [],
        "trainer": {
            **run_config_dict["trainer"],
            "max_steps": 5,
            "save_every": 10,
            "learning_rate": 1e-3,
            "max_seq_len": 256,
        },
    }


def test_sft_run_on_all_turns_of_tool_conversations(
    agent_sft_config: dict[str, Any], tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches: list[TrainBatch] = []
    original = ReferenceTrainer.train_step

    def recording(self: ReferenceTrainer, batch: TrainBatch) -> dict[str, float]:
        batches.append(batch)
        return original(self, batch)

    monkeypatch.setattr(ReferenceTrainer, "train_step", recording)
    result = run(RunConfig.from_dict(agent_sft_config))
    assert result.state == "succeeded", result.error

    rows = [row for _, row in iter_metrics(Path(result.run_dir)) if row["phase"] == "train"]
    assert [r["sft/turns"] for r in rows] == [4.0] * 5  # two conversations a step, two turns each
    assert all(r["tokens/sequences"] == 2.0 and r["tokens/prefix_mismatch"] == 0.0 for r in rows)
    assert all(r["tokens/truncated"] == 0.0 for r in rows)
    assert rows[-1]["loss/sft"] < rows[0]["loss/sft"]

    batch = batches[0]
    for b in range(batch.input_ids.shape[0]):
        length = int(batch.attention_mask[b].sum())
        ids, mask = batch.input_ids[b, :length].tolist(), batch.response_mask[b, :length].bool().tolist()
        trained = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if m])
        assert re.fullmatch(re.escape(CALL) + r"<\|im_end\|>\nIt is \d\.<\|im_end\|>\n", trained)
        context = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if not m])
        assert "What is 2+2?" in context and "<|im_start|>tool\n4<|im_end|>" in context
    logged = [row for _, row in iter_trajectories(Path(result.run_dir))]
    assert logged and all(t["num_turns"] == 2 and t["phase"] == "sft" for t in logged)


def test_loop_counts_conversations_that_are_not_one_sequence_and_logs_truncation(
    agent_sft_config: dict[str, Any], tokenizer: Any, tmp_path: Path
) -> None:
    data = tmp_path / "reasoning.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"What is {n}+{n}?"},
                {"role": "assistant", "content": f"<think>double {n}</think>{2 * n}"},
                {"role": "user", "content": "Sure?"},
                {"role": "assistant", "content": "<think>yes</think>Yes."},
            ]
        }
        for n in (1, 2)
    ]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = RunConfig.from_dict({**agent_sft_config, "data": {"uri": str(data), "format": "chat"}})
    reporter = RunReporter(Path(config.logging.run_root) / "fallbacks")
    loop = SFTLoop(config, reporter, cancel=threading.Event(), resume=False)
    loop._setup()
    try:
        samples = loop.sampler.batch(1)
        batch, stats, trajectories = loop._encode(samples, 1)
        assert stats["tokens/sequences"] == 2.0 and stats["tokens/prefix_mismatch"] == 0.0

        loop.tokenizer = _ReasoningDroppingTokenizer(tokenizer)
        batch, stats, trajectories = loop._encode(samples, 1)
        # two conversations, each needing a sequence per turn
        assert stats["tokens/prefix_mismatch"] == 2.0 and stats["tokens/sequences"] == 4.0
        assert stats["sft/turns"] == 4.0 and batch.input_ids.shape[0] == 4
        assert batch.group_ids[0] == batch.group_ids[1] != batch.group_ids[2] == batch.group_ids[3]
        assert batch.meta["sample_ids"] == [samples[0].id] * 2 + [samples[1].id] * 2
        for trajectory in trajectories:
            assert [t.completion.text for t in trajectory.turns] == [
                trajectory.messages[1]["content"],
                "<think>yes</think>Yes.",
            ]
            for turn in trajectory.turns:
                assert tokenizer.decode(turn.completion.token_ids) == turn.completion.text + "<|im_end|>\n"

        assert "cut from the right" not in (reporter.run_dir / "run.log").read_text()
        loop.max_seq_len = 20
        _, stats, _ = loop._encode(samples, 2)
        assert stats["tokens/truncated"] >= 1.0 and stats["tokens/truncated_turns"] >= 1.0
        loop._encode(samples, 3)
        log = (reporter.run_dir / "run.log").read_text()
        assert log.count("cut from the right") == 1 and "trainer.max_seq_len=20" in log
    finally:
        loop._teardown()
