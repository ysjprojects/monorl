"""Distilling agent episodes: accepted episodes → conversation rows → the data layer → SFT on every
policy turn, and the pipeline hand-off that connects the two stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fakes
import pytest
import yaml

from monorl.cli.recipes import recipes_dir
from monorl.config.pipeline import PipelineConfig, RejectionSampleStage
from monorl.config.schema import DataConfig, RunConfig
from monorl.core.errors import DataError
from monorl.core.registry import registry
from monorl.core.reporting import iter_metrics
from monorl.core.types import Completion, Sample, SamplingParams, Trajectory, Turn
from monorl.data import open_source
from monorl.envs.tool_calling import make_tool_calling_env
from monorl.orchestrator.episodes import EpisodeSpec, run_episodes
from monorl.orchestrator.sft import encode_sft_conversation, sft_messages, sft_tools, trained_turns
from monorl.pipeline import run_pipeline
from monorl.pipeline.rejection import conversation_rows, select_accepted
from monorl.pipeline.stages import build_train_config, validate_stage_configs


class ChatTokenizer(fakes.FakeTokenizer):
    """The fake tokenizer as the SFT loop and the data layer see a chat model's: with a template."""

    chat_template = "fake"


class ChatFakeTrainer(fakes.FakeTrainer):
    tokenizer = ChatTokenizer()


CALL = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call>'
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }
]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _spans(example: Any) -> list[list[int]]:
    return [example.input_ids[s.start : s.end] for s in example.spans]


# -- a stepped tool-calling episode, end to end ------------------------------------------------


def test_tool_calling_episode_round_trips_into_sft_on_both_turns(
    tmp_path: Path, fake_components: Any
) -> None:
    config = RunConfig.from_dict(
        fakes.fake_config_dict(
            tmp_path, rollout={"extra": {"fake_mode": "script", "fake_script": [CALL, "The answer is 4."]}}
        )
    )
    tokenizer = ChatTokenizer()
    sample = Sample(id="s1", messages=[{"role": "user", "content": "What is 2+2?"}], reference="4")
    (trajectory,) = run_episodes(
        [EpisodeSpec(sample=sample, group_id="g", engine=fakes.FakeEngine(config))],
        env_factory=lambda: make_tool_calling_env({"tools": ["calculator"]}, max_turns=4),
        tokenizer=tokenizer,
        params=SamplingParams(max_new_tokens=64, logprobs=False),
        max_parallel=1,
    )
    first, second = trajectory.turns
    trajectory.reward = 1.0

    (row,) = conversation_rows(trajectory)
    system, user, call, result, answer = row["messages"]
    assert system["role"] == "system" and "calculator" in system["content"]
    assert user == {"role": "user", "content": "What is 2+2?"}
    assert call == {"role": "assistant", "content": CALL}
    assert result["role"] == "tool" and result["name"] == "calculator" and result["content"] == "4"
    assert result["tool_call_id"] == first.info["tool_calls"][0]["id"]
    assert answer == {"role": "assistant", "content": "The answer is 4."}
    assert row["tools"] == make_tool_calling_env({"tools": ["calculator"]}, max_turns=4).tools
    assert (row["sample_id"], row["thread"], row["num_turns"]) == ("s1", 0, 2)
    assert (row["reward"], row["reference"], row["source"]) == (1.0, "4", "sampled")

    # ... written, read back by the data layer and encoded for SFT on every turn
    dataset = _write_rows(tmp_path / "dataset.jsonl", [row])
    data = DataConfig(uri=str(dataset), shuffle=False, max_prompt_tokens=8192)
    (read,) = list(open_source(data, tokenizer=tokenizer))
    assert read.messages == row["messages"][:-1] and read.metadata["tools"] == row["tools"]
    assert read.metadata["assistant_turn"] == answer
    assert read.metadata["row"]["num_turns"] == 2 and "tools" not in read.metadata["row"]

    (example,) = encode_sft_conversation(
        tokenizer, sft_messages(read), turns="all", tools=sft_tools(read), max_seq_len=4096
    )
    assert not example.prefix_mismatch and not example.truncated
    # the loss spans are the tokens the policy sampled in its two turns, in the context it had
    assert _spans(example) == [first.completion.token_ids, second.completion.token_ids]
    assert example.input_ids == second.prompt_token_ids + second.completion.token_ids
    assert example.input_ids[: example.spans[0].start] == first.prompt_token_ids
    assert sum(example.response_mask) == len(first.completion.token_ids) + len(second.completion.token_ids)
    assert example.response_ids == first.completion.token_ids + second.completion.token_ids

    # `last` trains only the final answer of the same row
    (last,) = encode_sft_conversation(tokenizer, sft_messages(read), turns="last", tools=sft_tools(read))
    assert last.input_ids == example.input_ids and _spans(last) == [second.completion.token_ids]
    # without the row's tool schemas the prompt is not the one the policy saw
    (bare,) = encode_sft_conversation(tokenizer, sft_messages(read), turns="all")
    assert bare.input_ids != example.input_ids


# -- harness-style episodes --------------------------------------------------------------------


def _turn(
    text: str, *, thread: int = 0, index: int | None = None, finish: str = "stop", tokens: int = 3
) -> Turn:
    completion = Completion(text=text, token_ids=list(range(tokens)), finish_reason=finish)  # type: ignore[arg-type]
    return Turn(prompt_token_ids=[1, 2, 3, 4], completion=completion, thread=thread, message_index=index)


MAIN: list[dict[str, Any]] = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "What is 2+2?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1", "name": "calculator", "arguments": {"expression": "2+2"}}],
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "calculator", "content": "4"},
    {"role": "assistant", "content": "It is 4."},
]


def _harness_episode(threads: list[list[dict[str, Any]]], turns: list[Turn]) -> Trajectory:
    sample = Sample(id="task-7", messages=[{"role": "user", "content": "What is 2+2?"}], reference="4")
    return Trajectory(
        sample=sample,
        group_id="g-task-7",
        messages=threads[0],  # type: ignore[arg-type]
        extra_threads=threads[1:],  # type: ignore[arg-type]
        turns=turns,
        reward=1.0,
        info={"tools": TOOLS},
    )


def test_harness_episode_gives_one_row_per_conversation_with_the_sampled_text() -> None:
    sub = [
        {"role": "system", "content": "You check sums."},
        {"role": "user", "content": "Is 2+2=4?"},
        {"role": "assistant", "content": "yes"},
    ]
    main = [*MAIN, {"role": "user", "content": "a follow-up the policy never answered"}]
    episode = _harness_episode(
        [main, sub],
        # in the order the calls were answered: the sub-agent ran between the two main calls
        [_turn(CALL, index=2), _turn(" yes\n", thread=1, index=2), _turn("It is 4.", index=4)],
    )
    main_row, sub_row = conversation_rows(episode)

    assert main_row["messages"] == [
        MAIN[0],
        MAIN[1],
        {"role": "assistant", "content": CALL},  # what the policy wrote, not the parsed tool_calls
        MAIN[3],  # the tool result with its name and id
        {"role": "assistant", "content": "It is 4."},
    ]
    assert (main_row["thread"], main_row["num_turns"], main_row["tools"]) == (0, 2, TOOLS)
    assert sub_row["messages"] == [sub[0], sub[1], {"role": "assistant", "content": " yes\n"}]
    assert (sub_row["thread"], sub_row["num_turns"], sub_row["sample_id"]) == (1, 1, "task-7")
    assert sub_row["reward"] == main_row["reward"] == 1.0 and sub_row["reference"] == "4"
    assert trained_turns(main_row["messages"], "all") == [2, 4]
    json.dumps([main_row, sub_row])


def test_shared_and_agent_supplied_assistant_messages_are_context() -> None:
    shot = [
        {"role": "user", "content": "What is 1+1?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c0", "name": "calculator", "arguments": {"expression": "1+1"}}],
        },
        {"role": "tool", "tool_call_id": "c0", "name": "calculator", "content": "2"},
    ]
    main = [MAIN[0], *shot, *MAIN[1:]]  # the agent's own worked example precedes the task
    retry = [
        *main[:-1],
        {"role": "assistant", "content": "Four."},
    ]  # the agent asked again from the tool result
    episode = _harness_episode(
        [main, retry], [_turn(CALL, index=5), _turn("It is 4.", index=7), _turn("Four!", thread=1, index=7)]
    )
    main_row, retry_row = conversation_rows(episode)

    example_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c0",
                "type": "function",
                "function": {"name": "calculator", "arguments": {"expression": "1+1"}},
            }
        ],
        "weight": 0,
    }
    assert main_row["messages"][2] == retry_row["messages"][2] == example_call
    assert main_row["messages"][5] == {"role": "assistant", "content": CALL}
    # the call both conversations share is trained once, in the conversation that owns it
    assert retry_row["messages"][5] == {"role": "assistant", "content": CALL, "weight": 0}
    assert retry_row["messages"][7] == {"role": "assistant", "content": "Four!"}
    assert trained_turns(main_row["messages"], "all") == [5, 7]
    assert trained_turns(retry_row["messages"], "all") == [7]
    assert (main_row["num_turns"], retry_row["num_turns"]) == (2, 1)

    tokenizer = ChatTokenizer()
    (encoded,) = encode_sft_conversation(tokenizer, retry_row["messages"], turns="all", tools=TOOLS)
    assert tokenizer.decode(encoded.response_ids) == "Four!<|end|>"
    assert CALL in tokenizer.decode(encoded.input_ids[: encoded.prompt_len])


def test_a_turn_without_its_assistant_message_is_an_error() -> None:
    episode = _harness_episode([MAIN], [_turn(CALL, index=2), _turn("It is 4.", index=3)])
    with pytest.raises(DataError, match="turn 1"):
        conversation_rows(episode)


# -- filters -----------------------------------------------------------------------------------


def _episode(
    texts: list[str],
    *,
    finishes: list[str] | None = None,
    tokens: int = 3,
    call_id: str = "c",
    reward: float = 1.0,
) -> Trajectory:
    sample = Sample(id="p", messages=[{"role": "user", "content": "q"}], reference="r")
    messages: list[dict[str, Any]] = list(sample.messages)  # type: ignore[arg-type]
    turns = []
    for i, text in enumerate(texts):
        if i:
            messages.append({"role": "tool", "name": "t", "tool_call_id": f"{call_id}{i}", "content": "ok"})
        turns.append(
            _turn(text, index=len(messages), finish=(finishes or ["stop"] * len(texts))[i], tokens=tokens)
        )
        messages.append({"role": "assistant", "content": text})
    return Trajectory(sample=sample, group_id="g", messages=messages, turns=turns, reward=reward)  # type: ignore[arg-type]


def test_filters_judge_the_conversation_not_the_concatenated_text() -> None:
    stage = RejectionSampleStage(name="rs", max_per_prompt=10, max_response_tokens=5)
    split_one, split_two = _episode(["ab", "c"]), _episode(["a", "bc"])  # both concatenate to "abc"
    same_but_for_ids = _episode(["ab", "c"], call_id="other")  # tool call ids are not content
    padded = _episode(["ab ", " c"])  # whitespace around a turn is not a new conversation
    cut_mid_episode = _episode(["x", "y"], finishes=["length", "stop"])
    two_short_turns = _episode(["long", "turns"], tokens=4)  # 8 tokens in total, 4 per turn
    one_long_turn = _episode(["m", "n"], tokens=6)
    kept = select_accepted(
        stage,
        [split_one, split_two, same_but_for_ids, padded, cut_mid_episode, two_short_turns, one_long_turn],
    )
    assert [[t.completion.text for t in e.turns] for e in kept] == [
        ["ab", "c"],
        ["a", "bc"],
        ["long", "turns"],
    ]

    unfiltered = RejectionSampleStage(name="rs", max_per_prompt=10, require_finish=False, dedupe=False)
    assert len(select_accepted(unfiltered, [split_one, same_but_for_ids, cut_mid_episode])) == 3
    # the context a turn reached: 4 prompt tokens + its completion
    by_context = RejectionSampleStage(name="rs", max_per_prompt=10, max_context_tokens=9)
    assert select_accepted(by_context, [two_short_turns, one_long_turn]) == [two_short_turns]


# -- hand-off ----------------------------------------------------------------------------------


def _handoff_pipeline(tmp_path: Path, sft_overrides: dict[str, Any]) -> PipelineConfig:
    base = fakes.fake_config_dict(
        tmp_path,
        data={"system_prompt": "base system prompt", "max_prompt_tokens": 100},
        trainer={"max_seq_len": 512},
        env={"type": "tool_calling", "max_turns": 3, "params": {"tools": ["calculator"]}},
    )
    base.pop("run_name")
    stages = [
        {"type": "rejection_sample", "name": "rs"},
        {"type": "train", "name": "sft", "data_from": "previous", "overrides": sft_overrides},
    ]
    return PipelineConfig.from_dict({"name": "p", "base": base, "stages": stages})


def _built(config: PipelineConfig, tmp_path: Path) -> RunConfig:
    return build_train_config(
        config, 1, stages_root=tmp_path / "stages", previous_model=None, previous_dataset="/rs/dataset.jsonl"
    )


def test_sft_stage_on_the_previous_dataset_trains_all_turns_of_the_rows_as_written(tmp_path: Path) -> None:
    sft = _built(_handoff_pipeline(tmp_path, {"algorithm": {"name": "sft"}}), tmp_path)
    assert sft.algorithm.sft_turns == "all"
    assert (
        yaml.safe_load(sft.to_yaml())["algorithm"]["sft_turns"] == "all"
    )  # visible in the stage's config.yaml
    assert sft.data.uri == "/rs/dataset.jsonl" and sft.data.format == "chat"
    # the rows hold the conversation the policy had: its system prompt is not replaced, and no row is
    # dropped for a context the sampling stage allowed
    assert sft.data.system_prompt is None and sft.data.max_prompt_tokens == 512
    assert sft.env.type == "single_turn"  # an SFT stage runs no environment

    explicit = _handoff_pipeline(tmp_path, {"algorithm": {"name": "sft", "sft_turns": "last"}})
    assert _built(explicit, tmp_path).algorithm.sft_turns == "last"
    rl = _built(_handoff_pipeline(tmp_path, {}), tmp_path)
    assert rl.algorithm.sft_turns == "last" and rl.env.type == "tool_calling"


def test_distillation_recipe_is_wired_for_agent_episodes(tmp_path: Path) -> None:
    recipe = PipelineConfig.from_yaml(recipes_dir() / "distill_tool_calling_cpu.yaml")
    validate_stage_configs(recipe, tmp_path)
    assert [s.type for s in recipe.stages] == ["rejection_sample", "train", "eval"]
    assert recipe.base.env.type == "tool_calling" and recipe.stages[0].min_reward == 1.0
    sft = _built(recipe, tmp_path)
    assert sft.is_sft and sft.algorithm.sft_turns == "all" and sft.data.system_prompt is None
    assert sft.data.max_prompt_tokens == recipe.base.context_budget


# -- the two stages together --------------------------------------------------------------------


def test_pipeline_distils_multi_turn_episodes_end_to_end(make_pipeline: Any) -> None:
    registry.register("trainer", "fake_chat", ChatFakeTrainer, overwrite=True)
    stages = [
        {
            "type": "rejection_sample",
            "name": "rs",
            "num_samples": 2,
            "min_reward": 1.0,
            "max_prompts": 3,
            "rollout_overrides": {"extra": {"fake_mode": "always_right", "max_parallel_episodes": 1}},
        },
        {
            "type": "train",
            "name": "sft",
            "init_from": "base",
            "data_from": "previous",
            "overrides": {
                "algorithm": {"name": "sft"},
                "trainer": {"backend": "fake_chat", "max_steps": 2, "prompts_per_step": 2},
            },
        },
    ]
    config = make_pipeline(
        stages,
        # two turns per episode, 0.5 reward each: every episode is "solved"
        env={"type": "fake_multi", "max_turns": 2, "params": {"turn_reward": 0.5, "tools": TOOLS}},
        trainer={"max_seq_len": 1024},
    )
    result = run_pipeline(config)
    assert result.state == "succeeded", result.error
    rs, sft = result.stages

    rows = [json.loads(line) for line in Path(rs["summary"]["dataset"]).read_text().splitlines()]
    assert len(rows) == rs["summary"]["rows"] == rs["summary"]["accepted"] == 3  # duplicates dropped
    conversations = {
        prompt: [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prompt.upper()},
            {"role": "user", "content": "again1"},
            {"role": "assistant", "content": "AGAIN1"},
        ]
        for prompt in ("abc", "def", "ghi")
    }
    assert sorted(r["messages"][0]["content"] for r in rows) == sorted(conversations)
    for row in rows:
        assert row["messages"] == conversations[row["messages"][0]["content"]]
        assert row["tools"] == TOOLS and row["num_turns"] == 2 and row["reward"] == 1.0

    stage_config = yaml.safe_load((Path(sft["run_dir"]) / "config.yaml").read_text())
    assert stage_config["algorithm"]["sft_turns"] == "all" and stage_config["env"]["type"] == "single_turn"
    assert stage_config["data"]["uri"] == rs["summary"]["dataset"]

    trainer = fakes.TRAINERS[-1]
    tokenizer = ChatTokenizer()
    assert len(trainer.batches) == 2
    for batch in trainer.batches:
        assert batch.input_ids.shape[0] == 2  # one sequence per conversation
        for b in range(2):
            length = int(batch.attention_mask[b].sum())
            ids = batch.input_ids[b, :length].tolist()
            mask = batch.response_mask[b, :length].bool().tolist()
            text = tokenizer.decode(ids)
            prompt = fakes.last_user_text(text.split("<|assistant|>")[0])
            expected = tokenizer.apply_chat_template(conversations[prompt], tools=TOOLS)
            assert text == expected  # the tool schemas and both turns, exactly as sampled
            trained = tokenizer.decode([t for t, m in zip(ids, mask, strict=True) if m])
            assert trained == f"{prompt.upper()}<|end|>AGAIN1<|end|>"
    metrics = [row for _, row in iter_metrics(sft["run_dir"]) if row["phase"] == "train"]
    assert [m["sft/turns"] for m in metrics] == [4.0, 4.0]
    assert all(m["tokens/sequences"] == 2.0 and m["tokens/prefix_mismatch"] == 0.0 for m in metrics)
    assert all(m["tokens/truncated"] == 0.0 and m["tokens/truncated_turns"] == 0.0 for m in metrics)
