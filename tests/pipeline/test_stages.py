"""Pure helpers: config merging/hand-off, manifest round-trip, rejection filters and mix targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from monorl.config.pipeline import PipelineConfig, RejectionSampleStage
from monorl.config.schema import DataConfig
from monorl.core.types import Completion, Sample, Trajectory, Turn
from monorl.pipeline.rejection import (
    chat_row,
    eval_data_config,
    mix_data_config,
    prompt_messages,
    sample_response,
    select_accepted,
)
from monorl.pipeline.stages import (
    Manifest,
    build_sampling_config,
    build_train_config,
    deep_merge,
    known_data_configs,
    new_manifest,
    previous_dataset_config,
    read_manifest,
    stage_dir_name,
    validate_stage_configs,
)


def _base(tmp_path: Path) -> dict[str, Any]:
    data = tmp_path / "d.jsonl"
    data.write_text('{"prompt": "q", "answer": "a"}\n')
    return {
        "model": {"name_or_path": "org/base"},
        "data": {
            "uri": str(data),
            "prompt_field": "prompt",
            "answer_field": "answer",
            "system_prompt": "sys",
            "limit": 10,
        },
        "rollout": {"group_size": 2, "temperature": 1.0, "extra": {"k": 1}},
        "trainer": {"max_steps": 5},
        "rewards": [{"type": "exact_match"}],
        "logging": {"run_root": str(tmp_path / "runs")},
    }


def _pipeline(tmp_path: Path, stages: list[dict[str, Any]]) -> PipelineConfig:
    return PipelineConfig.from_dict({"name": "p", "base": _base(tmp_path), "stages": stages})


def test_deep_merge_and_stage_dir_name() -> None:
    merged = deep_merge({"a": {"b": 1, "c": [1]}, "d": 2}, {"a": {"c": [2], "e": 3}})
    assert merged == {"a": {"b": 1, "c": [2], "e": 3}, "d": 2}
    assert stage_dir_name(3, "my stage/x") == "03-my-stage-x"


def test_build_train_config_overrides_and_handoff(tmp_path: Path) -> None:
    cfg = _pipeline(
        tmp_path,
        [
            {
                "type": "train",
                "name": "sft",
                "init_from": "base",
                "overrides": {"algorithm": {"name": "sft"}, "trainer": {"max_steps": 2}},
            },
            {"type": "rejection_sample", "name": "rs"},
            {"type": "train", "name": "sft2", "data_from": "previous", "overrides": {"data": {"limit": 3}}},
            {"type": "train", "name": "explicit", "init_from": "/models/x"},
        ],
    )
    root = tmp_path / "stages"
    first = build_train_config(cfg, 0, stages_root=root, previous_model=None, previous_dataset=None)
    assert (
        first.algorithm.name == "sft"
        and first.trainer.max_steps == 2
        and first.model.name_or_path == "org/base"
    )
    assert first.run_name == "00-sft" and first.logging.run_root == str(root)
    assert first.rollout.extra == {"k": 1}  # untouched base sections survive

    third = build_train_config(
        cfg, 2, stages_root=root, previous_model="/prev/final", previous_dataset="/rs/dataset.jsonl"
    )
    assert third.model.name_or_path == "/prev/final"
    assert (
        third.data.uri == "/rs/dataset.jsonl"
        and third.data.format == "chat"
        and third.data.messages_field == "messages"
    )
    assert (
        third.data.prompt_field is None
        and third.data.answer_field is None
        and third.data.response_field is None
    )
    assert third.data.system_prompt == "sys" and third.data.limit == 3  # overrides apply before the swap
    assert third.trainer.max_steps == 5

    fourth = build_train_config(cfg, 3, stages_root=root, previous_model="/prev/final", previous_dataset=None)
    assert fourth.model.name_or_path == "/models/x"
    with pytest.raises(ValueError, match="data_from"):
        build_train_config(cfg, 2, stages_root=root, previous_model=None, previous_dataset=None)
    with pytest.raises(TypeError):
        build_train_config(cfg, 1, stages_root=root, previous_model=None, previous_dataset=None)


def test_init_from_previous_without_previous_keeps_merged_model(tmp_path: Path) -> None:
    cfg = _pipeline(
        tmp_path, [{"type": "train", "name": "a", "overrides": {"model": {"name_or_path": "org/other"}}}]
    )
    built = build_train_config(cfg, 0, stages_root=tmp_path, previous_model=None, previous_dataset=None)
    assert built.model.name_or_path == "org/other"


def test_build_sampling_config(tmp_path: Path) -> None:
    other = tmp_path / "other.jsonl"
    other.write_text('{"prompt": "x", "answer": "y"}\n')
    cfg = _pipeline(
        tmp_path,
        [
            {
                "type": "rejection_sample",
                "name": "rs",
                "rollout_overrides": {"temperature": 0.3, "extra": {"j": 2}},
                "reward_overrides": [{"type": "length", "params": {"max_tokens": 4}}],
                "data": {"uri": str(other), "prompt_field": "prompt"},
            },
            {"type": "eval", "name": "ev"},
        ],
    )
    rs = build_sampling_config(cfg, 0, previous_model="/m/final")
    assert rs.model.name_or_path == "/m/final" and rs.rollout.temperature == 0.3
    assert rs.rollout.extra == {"k": 1, "j": 2} and rs.rollout.group_size == 2
    assert [r.type for r in rs.rewards] == ["length"] and rs.data.uri == str(other)
    ev = build_sampling_config(cfg, 1, previous_model=None)
    assert ev.model.name_or_path == "org/base" and [r.type for r in ev.rewards] == ["exact_match"]
    validate_stage_configs(cfg, tmp_path)


def test_known_data_configs_and_mix_resolution(tmp_path: Path) -> None:
    cfg = _pipeline(
        tmp_path,
        [
            {
                "type": "train",
                "name": "sft",
                "overrides": {"data": {"uri": "hf://x/y", "hf_config_name": "c", "response_field": "sol"}},
            },
            {"type": "rejection_sample", "name": "rs", "mix": ["hf://x/y", "hf://z/w"]},
        ],
    )
    known = known_data_configs(cfg)
    assert [d.uri for d in known] == [cfg.base.data.uri, "hf://x/y"]
    assert (
        known[1].response_field == "sol"
        and known[1].hf_config_name == "c"
        and known[1].prompt_field == "prompt"
    )
    base = cfg.base.data
    same = mix_data_config(base, base.uri, known)
    assert same.prompt_field == "prompt" and same.shuffle is False and same.limit == 10
    cold = mix_data_config(base, "hf://x/y", known)
    assert cold.response_field == "sol" and cold.hf_config_name == "c"
    unknown = mix_data_config(base, "hf://z/w", known)
    assert unknown.format == "auto" and unknown.prompt_field is None and unknown.system_prompt == "sys"


def test_previous_dataset_and_eval_data_config(tmp_path: Path) -> None:
    data = DataConfig(
        uri="hf://a/b",
        hf_config_name="main",
        prompt_field="q",
        answer_field="a",
        prompt_template="{prompt}!",
        eval_uri="hf://a/b",
        eval_split="test",
    )
    prev = previous_dataset_config(data, "/x/dataset.jsonl")
    assert prev.uri == "/x/dataset.jsonl" and prev.format == "chat" and prev.hf_config_name is None
    assert prev.prompt_template is None and prev.eval_uri == "hf://a/b"
    from monorl.config.pipeline import EvalStage
    from monorl.config.schema import RunConfig

    run = RunConfig.from_dict(
        {"model": {"name_or_path": "m"}, "data": data.model_dump(), "rewards": [{"type": "exact_match"}]}
    )
    ev = eval_data_config(EvalStage(name="e"), run)
    assert ev.uri == "hf://a/b" and ev.split == "test" and ev.shuffle is False and ev.eval_uri is None
    own = eval_data_config(EvalStage(name="e", data=DataConfig(uri="/own.jsonl")), run)
    assert own.uri == "/own.jsonl"


def test_manifest_round_trip(tmp_path: Path) -> None:
    cfg = _pipeline(tmp_path, [{"type": "train", "name": "a"}, {"type": "eval", "name": "b"}])
    manifest = new_manifest(cfg, "p", tmp_path)
    manifest.stages[0].state = "succeeded"
    manifest.stages[0].summary = {"x": 1.0}
    manifest.write()
    data = read_manifest(tmp_path)
    assert data["name"] == "p" and [s["run_dir"] for s in data["stages"]] == [
        str(tmp_path / "stages" / "00-a"),
        str(tmp_path / "stages" / "01-b"),
    ]
    back = Manifest.from_dict(data)
    assert (
        back.stages[0].state == "succeeded"
        and back.stages[0].summary == {"x": 1.0}
        and back.stages[1].state == "pending"
    )
    assert read_manifest(tmp_path / "nope") is None
    (tmp_path / "pipeline.json").write_text("{not json")
    assert read_manifest(tmp_path) is None


# -- rejection filters ------------------------------------------------------------------------


def _traj(
    sample_id: str, text: str, reward: float, *, finish: str = "stop", tokens: int = 3, reference: str = "A"
) -> Trajectory:
    sample = Sample(id=sample_id, messages=[{"role": "user", "content": "q"}], reference=reference)
    completion = Completion(text=text, token_ids=list(range(tokens)), finish_reason=finish)  # type: ignore[arg-type]
    return Trajectory(
        sample=sample,
        group_id=f"g-{sample_id}",
        messages=[*sample.messages, {"role": "assistant", "content": text}],
        turns=[Turn(prompt_token_ids=[1, 2], completion=completion)],
        reward=reward,
    )


def test_select_accepted_filters() -> None:
    stage = RejectionSampleStage(
        name="rs", min_reward=1.0, max_per_prompt=2, max_response_tokens=5, require_finish=True, dedupe=True
    )
    trajectories = [
        _traj("p1", "A", 1.0),
        _traj("p1", "A", 1.0),  # duplicate
        _traj("p1", "A ", 1.0),  # duplicate after strip
        _traj("p1", "B", 2.0),  # best, kept first
        _traj("p1", "C", 1.0, finish="length"),  # truncated
        _traj("p1", "D", 1.0, tokens=9),  # too long
        _traj("p1", "E", 0.5),  # below threshold
        _traj("p2", "F", 1.0),
        _traj("p2", "G", 1.5),
        _traj("p2", "H", 1.2),  # third best → capped out
    ]
    kept = select_accepted(stage, trajectories)
    assert [(t.sample.id, t.response_text) for t in kept] == [
        ("p1", "B"),
        ("p1", "A"),
        ("p2", "G"),
        ("p2", "H"),
    ]

    loose = RejectionSampleStage(
        name="rs", min_reward=0.0, max_per_prompt=10, require_finish=False, dedupe=False
    )
    assert len(select_accepted(loose, trajectories)) == len(trajectories)


def test_prompt_messages_and_chat_row() -> None:
    t = _traj("p", "resp", 1.0, reference="42")
    t.messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "resp"},
    ]
    assert prompt_messages(t) == t.messages[:2]
    row = chat_row(
        prompt_messages(t), t.response_text, reference=t.sample.reference, reward=t.reward, source="sampled"
    )
    assert row == {
        "messages": [*t.messages[:2], {"role": "assistant", "content": "resp"}],
        "reference": "42",
        "reward": 1.0,
        "source": "sampled",
    }
    json.dumps(row)


def test_sample_response_precedence() -> None:
    data = DataConfig(uri="x", response_field="sol", answer_field="ans")
    s = Sample(id="1", messages=[], reference="ref", metadata={"response": "from-response-field"})
    assert sample_response(s, data) == "from-response-field"
    s = Sample(
        id="2",
        messages=[],
        reference="ref",
        metadata={"assistant_turn": {"role": "assistant", "content": "turn"}},
    )
    assert sample_response(s, data) == "turn"
    s = Sample(
        id="3",
        messages=[],
        reference="ref",
        metadata={"assistant_turn": {"role": "assistant", "content": [{"type": "text", "text": "parts"}]}},
    )
    assert sample_response(s, data) == "parts"
    s = Sample(id="4", messages=[], reference="ref", metadata={"row": {"sol": "solution", "ans": "answer"}})
    assert sample_response(s, data) == "solution"
    s = Sample(id="5", messages=[], reference="ref", metadata={"row": {"ans": "answer"}})
    assert sample_response(s, data) == "answer"
    s = Sample(id="6", messages=[], reference="42", metadata={"answer_raw": "... #### 42"})
    assert sample_response(s, data) == "... #### 42"
    s = Sample(id="7", messages=[], reference="42", metadata={})
    assert sample_response(s, data) == "42"
    s = Sample(id="8", messages=[], reference=None, metadata={})
    assert sample_response(s, data) is None
