"""``algorithm.sft_turns`` on the pipeline backends: each backend's own loss-mask mechanism, the rows
it is fed, and a ``ConfigError`` where a backend cannot do what the option says."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.types import Sample

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
CONVERSATION: list[dict[str, Any]] = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "<tool_call>calculator(2+2)</tool_call>"},
    {"role": "tool", "name": "calculator", "tool_call_id": "call_1", "content": "4"},
    {"role": "assistant", "content": "It is 4."},
]


def _agent_sample(*, tools: list[dict[str, Any]] | None = TOOLS, context_turn: bool = False) -> Sample:
    """A two-turn tool conversation as the data layer hands it to SFT."""
    prompt = [dict(m) for m in CONVERSATION[:-1]]
    if context_turn:
        prompt[1]["weight"] = 0
    metadata: dict[str, Any] = {"response": "It is 4.", "assistant_turn": CONVERSATION[-1]}
    if tools:
        metadata["tools"] = tools
    return Sample(id="agent-1", messages=prompt, reference="4", metadata=metadata)  # type: ignore[arg-type]


def _single_turn_sample() -> Sample:
    return Sample(id="qa-1", messages=[{"role": "user", "content": "2+2?"}], metadata={"response": "4"})


@pytest.fixture
def make_sft_config(tiny_model_dir: Path, tmp_path: Path):
    data = tmp_path / "agent.jsonl"
    data.write_text(json.dumps({"messages": CONVERSATION, "tools": TOOLS}) + "\n")

    def _make(backend: str, **algorithm: Any) -> RunConfig:
        return RunConfig.from_dict(
            {
                "model": {"name_or_path": str(tiny_model_dir), "dtype": "float32"},
                "data": {"uri": str(data), "format": "chat", "max_prompt_tokens": 512},
                "algorithm": {"name": "sft", **algorithm},
                "trainer": {
                    "backend": backend,
                    "prompts_per_step": 2,
                    "micro_batch_size": 2,
                    "max_steps": 2,
                    "precision": "bf16",
                    "max_seq_len": 256,
                },
                "cluster": {"mode": "ray", "num_nodes": 1, "gpus_per_node": 8},
                "logging": {"run_root": str(tmp_path / "runs")},
            }
        )

    return _make


# -- TRL ---------------------------------------------------------------------------------------


def test_trl_maps_turns_to_its_loss_masks(make_sft_config: Any, tmp_path: Path) -> None:
    from monorl.adapters.trl.compile import sft_config_kwargs

    cpu = torch.device("cpu")
    last = sft_config_kwargs(make_sft_config("trl"), tmp_path, device=cpu)
    assert (last["completion_only_loss"], last["assistant_only_loss"]) == (True, False)
    every = sft_config_kwargs(make_sft_config("trl", sft_turns="all"), tmp_path, device=cpu)
    assert (every["completion_only_loss"], every["assistant_only_loss"]) == (False, True)
    full = sft_config_kwargs(
        make_sft_config("trl", sft_turns="all", sft_train_on_prompt=True), tmp_path, device=cpu
    )
    assert (full["completion_only_loss"], full["assistant_only_loss"]) == (False, False)


def test_trl_rows_hold_the_conversation_and_its_tools(make_sft_config: Any) -> None:
    from monorl.adapters.trl.data import build_train_and_eval, sft_sample_to_row

    row = sft_sample_to_row(_agent_sample(), response_field=None, turns="all")
    assert row == {"messages": CONVERSATION, "tools": json.dumps(TOOLS)}
    last = sft_sample_to_row(_agent_sample(), response_field=None)
    assert [m["role"] for m in last["prompt"]] == ["user", "assistant", "tool"]
    assert last["completion"] == [CONVERSATION[-1]] and json.loads(last["tools"]) == TOOLS
    assert sft_sample_to_row(_single_turn_sample(), response_field=None, turns="all") == {
        "messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
    }
    with pytest.raises(ConfigError, match="weight"):
        sft_sample_to_row(_agent_sample(context_turn=True), response_field=None, turns="all")

    # through the data layer into the dataset SFTTrainer tokenises: `messages` + JSON `tools`
    dataset, _ = build_train_and_eval(make_sft_config("trl", sft_turns="all"))
    assert set(dataset.column_names) == {"messages", "tools"}
    assert [m["role"] for m in dataset[0]["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert dataset[0]["messages"][2]["tool_call_id"] == "call_1"
    assert json.loads(dataset[0]["tools"]) == TOOLS


CHATML_WITH_GENERATION_MARKERS = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] + '<|im_end|>' }}{% endgeneration %}"
    "{{ '\n' }}"
    "{% else %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


@pytest.mark.slow
def test_trl_run_puts_the_loss_on_both_assistant_turns(make_sft_config: Any, tmp_path: Path) -> None:
    """The mapping on TRL's side: a real ``SFTTrainer`` built from the exported rows masks the
    labels down to the two assistant turns of the conversation."""
    pytest.importorskip("trl")
    from monorl.core.registry import registry
    from monorl.core.reporting import RunReporter

    data = make_sft_config("trl", sft_turns="all").to_dict()
    # TRL takes the assistant mask from the template's generation markers
    data["model"]["chat_template"] = CHATML_WITH_GENERATION_MARKERS
    data["trainer"].update(
        {"precision": "fp32", "device": "cpu", "prompts_per_step": 1, "micro_batch_size": 1}
    )
    data["trainer"]["max_steps"] = 1
    data["cluster"] = {"mode": "local"}
    config = RunConfig.from_dict(data)
    run_dir = tmp_path / "trl-run"
    backend = registry.get("pipeline_backend", "trl")(config)
    result = backend.run(config, run_dir, reporter=RunReporter(run_dir, echo=False))
    assert result.state == "succeeded", result.error

    args, example = backend.trainer.args, backend.trainer.train_dataset[0]
    assert args.assistant_only_loss is True and args.completion_only_loss is False
    tokenizer = backend.trainer.processing_class
    trained = tokenizer.decode([t for t in example["labels"] if t != -100])
    assert trained == "<tool_call>calculator(2+2)</tool_call><|im_end|>It is 4.<|im_end|>"
    context = tokenizer.decode(example["input_ids"])
    assert "What is 2+2?" in context and "<|im_start|>tool\n4<|im_end|>" in context


# -- verl --------------------------------------------------------------------------------------


def test_verl_trains_every_assistant_message_and_refuses_last_on_multi_turn_rows(
    make_sft_config: Any, tmp_path: Path
) -> None:
    from monorl.adapters.verl.compiler import compile_sft_config
    from monorl.adapters.verl.data import export_datasets, sft_sample_to_row, write_sft_parquet

    assert (
        compile_sft_config(make_sft_config("verl", sft_turns="all"), tmp_path)["data"]["tools_key"] == "tools"
    )
    row = sft_sample_to_row(_agent_sample(), response_field=None, turns="all")
    assert [(m["role"], m["content"]) for m in row["messages"]] == [
        (m["role"], m["content"]) for m in CONVERSATION
    ]
    assert row["tools"] == TOOLS
    # one assistant turn: `last` and `all` are the same thing for MultiTurnSFTDataset
    single = sft_sample_to_row(_single_turn_sample(), response_field=None)
    assert single == sft_sample_to_row(_single_turn_sample(), response_field=None, turns="all")
    with pytest.raises(ConfigError, match="sft_turns='all'"):
        sft_sample_to_row(_agent_sample(), response_field=None, turns="last")
    with pytest.raises(ConfigError, match="weight"):
        sft_sample_to_row(_agent_sample(context_turn=True), response_field=None, turns="all")

    paths = export_datasets(make_sft_config("verl", sft_turns="all"), tmp_path / "run")
    (exported,) = pq.read_table(paths["train"]).to_pylist()
    assert exported["tools"] == TOOLS and len(exported["messages"]) == 4
    with pytest.raises(ConfigError, match="sft_turns='all'"):
        export_datasets(make_sft_config("verl"), tmp_path / "run-last")

    # the tools column must reach the chat template as the schemas were written
    with pytest.raises(ConfigError, match="without"):
        write_sft_parquet(
            [_agent_sample(), _single_turn_sample()],
            tmp_path / "mixed.parquet",
            response_field=None,
            turns="all",
        )
    other = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object", "required": ["q"]}},
        }
    ]
    with pytest.raises(ConfigError, match="differ in shape"):
        write_sft_parquet(
            [_agent_sample(), _agent_sample(tools=other)],
            tmp_path / "shapes.parquet",
            response_field=None,
            turns="all",
        )


# -- NeMo-RL -----------------------------------------------------------------------------------


def test_nemo_rl_maps_turns_to_only_unmask_final(make_sft_config: Any, tmp_path: Path) -> None:
    from monorl.adapters.nemo_rl.compile import compile_sft_config
    from monorl.adapters.nemo_rl.data import export_dataset, sft_sample_to_row

    assert compile_sft_config(make_sft_config("nemo_rl"), tmp_path)["sft"]["only_unmask_final"] is True
    every = make_sft_config("nemo_rl", sft_turns="all")
    assert compile_sft_config(every, tmp_path)["sft"]["only_unmask_final"] is False

    row = sft_sample_to_row(_agent_sample(), response_field=None, turns="all")
    assert row["tools"] == TOOLS and [m["role"] for m in row["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    # `last` needs no restriction: NeMo-RL then unmasks only the final message
    assert sft_sample_to_row(_agent_sample(), response_field=None)["messages"] == row["messages"]
    with pytest.raises(ConfigError, match="weight"):
        sft_sample_to_row(_agent_sample(context_turn=True), response_field=None, turns="all")

    exported = export_dataset(every, tmp_path / "run")
    (written,) = [json.loads(line) for line in exported.paths.sft_train_jsonl.read_text().splitlines()]
    assert written["tools"] == TOOLS and written["messages"][-1] == CONVERSATION[-1]
