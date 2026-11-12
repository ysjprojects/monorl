"""``RunReporter.log_trajectories`` rows as the transcript viewer reads them back."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monorl.core.reporting import (
    MAX_DATA_URL_BYTES,
    MAX_TEXT_CHARS,
    TRAJECTORIES_FILE,
    RunReporter,
    iter_trajectories,
)
from monorl.core.types import Completion, Message, Sample, Trajectory, Turn

TOKEN = 987654
LOGPROB = -0.123456


def _turn(text: str, *, prompt_len: int, finish: str = "stop", **kw: Any) -> Turn:
    completion = Completion(
        text=text, token_ids=[TOKEN] * len(text), logprobs=[LOGPROB] * len(text), finish_reason=finish
    )
    return Turn(prompt_token_ids=[TOKEN] * prompt_len, completion=completion, **kw)


def _log(tmp_path: Path, trajectory: Trajectory, **kw: Any) -> dict[str, Any]:
    reporter = RunReporter(tmp_path / "run", echo=False)
    reporter.log_trajectories(3, [trajectory], **kw)
    rows = [row for _, row in iter_trajectories(reporter.run_dir)]
    assert len(rows) == 1
    return rows[0]


def _tool_trajectory() -> Trajectory:
    call = '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 2}}</tool_call>'
    few_shot: list[Message] = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "example"},
        {"role": "assistant", "content": "example answer"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    messages: list[Message] = [
        *few_shot,
        {
            "role": "assistant",
            "content": call,
            "tool_calls": [{"id": "call_0", "name": "add", "arguments": {"a": 2, "b": 2}}],
        },
        {"role": "tool", "name": "add", "tool_call_id": "call_0", "content": "4"},
        {"role": "assistant", "content": "4"},
    ]
    executed = [{"id": "call_0", "name": "add", "arguments": {"a": 2, "b": 2}, "result": "4"}]
    turns = [
        _turn(call, prompt_len=40, finish="tool_call", reward=-0.5, info={"tool_calls": executed}),
        _turn("4", prompt_len=120, info={"tool_calls": [], "invalid_calls": 0}),
    ]
    return Trajectory(
        sample=Sample(id="s1", messages=few_shot, reference="4"),
        group_id="g1",
        messages=messages,
        turns=turns,
        reward=1.0,
        reward_breakdown={"exact": 1.0},
        info={"num_turns": 2},
    )


def test_tool_calling_trajectory_round_trips_with_transcript(tmp_path: Path) -> None:
    trajectory = _tool_trajectory()
    row = _log(tmp_path, trajectory, phase="eval", success_threshold=1.0)

    assert row["step"] == 3 and row["phase"] == "eval"
    assert row["sample_id"] == "s1" and row["group_id"] == "g1" and row["reference"] == "4"
    assert row["prompt"] == "What is 2+2?" and row["response"].endswith("</tool_call>4")
    assert row["reward"] == 1.0 and row["reward_breakdown"] == {"exact": 1.0}
    assert row["success"] is True and row["done"] is True and row["finish_reason"] == "stop"
    assert row["num_turns"] == 2 and row["num_response_tokens"] == len(row["response"])
    assert row["num_context_tokens"] == 121  # the last turn: 120 prompt + 1 completion token
    assert row["info"] == {"num_turns": 2}

    assert row["threads"] == [trajectory.messages]
    first, second = row["turns"]
    # the few-shot assistant message is not a turn: turns map onto the last two assistant messages
    assert (first["index"], first["thread"], first["message_index"]) == (0, 0, 4)
    assert (second["index"], second["thread"], second["message_index"]) == (1, 0, 6)
    assert first["reward"] == -0.5 and first["finish_reason"] == "tool_call"
    assert first["prompt_tokens"] == 40 and first["response_tokens"] == len(
        trajectory.turns[0].completion.text
    )
    assert first["info"]["tool_calls"][0]["result"] == "4"
    assert second["info"] == {"tool_calls": [], "invalid_calls": 0}

    raw = (tmp_path / "run" / TRAJECTORIES_FILE).read_text()
    assert str(TOKEN) not in raw and str(LOGPROB) not in raw


def test_success_follows_the_threshold(tmp_path: Path) -> None:
    trajectory = _tool_trajectory()
    trajectory.reward = 0.6
    assert _log(tmp_path / "a", trajectory, success_threshold=0.5)["success"] is True
    assert _log(tmp_path / "b", trajectory, success_threshold=1.0)["success"] is False


def test_save_transcripts_false_omits_threads_and_turns(tmp_path: Path) -> None:
    row = _log(tmp_path, _tool_trajectory(), phase="rejection", save_transcripts=False)
    assert "threads" not in row and "turns" not in row
    assert row["phase"] == "rejection" and row["num_turns"] == 2 and row["num_context_tokens"] == 121
    assert row["response"].endswith("4") and row["done"] is True


def test_two_thread_trajectory_keeps_both_conversations(tmp_path: Path) -> None:
    before: list[Message] = [
        {"role": "user", "content": "long task"},
        {"role": "assistant", "content": "step one"},
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "step two"},
    ]
    after: list[Message] = [
        {"role": "user", "content": "summary of the work so far"},
        {"role": "assistant", "content": "done"},
    ]
    trajectory = Trajectory(
        sample=Sample(id="s", messages=before[:1]),
        group_id="g",
        messages=after,
        extra_threads=[before],
        turns=[
            _turn("step one", prompt_len=5, thread=1, message_index=1),
            _turn("step two", prompt_len=20, thread=1),
            _turn("done", prompt_len=9, thread=0),
        ],
        done=False,
    )
    row = _log(tmp_path, trajectory)
    assert row["threads"] == [after, before]
    assert [(t["thread"], t["message_index"]) for t in row["turns"]] == [(1, 1), (1, 3), (0, 1)]
    assert row["done"] is False and row["num_context_tokens"] == 28


def test_message_index_is_null_when_the_thread_lacks_the_assistant_messages(tmp_path: Path) -> None:
    trajectory = Trajectory(
        sample=Sample(id="s", messages=[{"role": "user", "content": "q"}]),
        group_id="g",
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "b"}],
        turns=[_turn("a", prompt_len=3), _turn("b", prompt_len=6), _turn("c", prompt_len=4, thread=7)],
    )
    row = _log(tmp_path, trajectory)
    assert [t["message_index"] for t in row["turns"]] == [None, None, None]


class _Unprintable:
    def __repr__(self) -> str:
        raise RuntimeError("no repr")


def test_oversized_and_unserialisable_content_is_guarded(tmp_path: Path) -> None:
    small_url = "data:image/png;base64," + "A" * 64
    big_url = "data:image/png;base64," + "A" * (MAX_DATA_URL_BYTES + 1)
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    messages: list[Any] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "y" * (MAX_TEXT_CHARS + 7)},
                {"type": "image", "image": "/data/cat.png", "detail": "low"},
                {"type": "image", "image": "https://example.com/cat.png"},
                {"type": "image", "image": small_url},
                {"type": "image", "image": big_url},
                {"type": "image", "image": b"\x89PNG" * 1000},
            ],
        },
        {"role": "assistant", "content": "x" * (MAX_TEXT_CHARS + 500), "tool_calls": [{"arguments": {1, 2}}]},
        {"role": "tool", "content": object()},
    ]
    trajectory = Trajectory(
        sample=Sample(id="s", messages=[{"role": "user", "content": "look"}]),
        group_id="g",
        messages=messages,
        turns=[
            _turn(
                "x",
                prompt_len=2,
                info={"obj": object(), "bad": _Unprintable(), "nan": float("nan"), 3: b"raw", "loop": cyclic},
            )
        ],
        info={"env": object(), "path": Path("/tmp/x"), "big": "z" * (MAX_TEXT_CHARS + 1)},
    )
    reporter = RunReporter(tmp_path / "run", echo=False)
    reporter.log_trajectories(1, [trajectory])
    line = (reporter.run_dir / TRAJECTORIES_FILE).read_text().strip()
    assert len(line) < 4 * MAX_TEXT_CHARS + 20_000  # neither the inline image nor the long texts survive
    row = json.loads(line, parse_constant=lambda name: (_ for _ in ()).throw(AssertionError(name)))

    text, path, url, small, big, raw = row["threads"][0][0]["content"]
    assert text["text"].startswith("y" * MAX_TEXT_CHARS) and "7 characters truncated" in text["text"]
    assert path == {"type": "image", "image": "/data/cat.png", "detail": "low"}
    assert url["image"] == "https://example.com/cat.png"
    assert small == {"type": "image", "image": small_url}
    assert big["type"] == "image" and big["image"] is None and str(len(big_url)) in big["note"]
    assert raw["type"] == "image" and raw["image"] is None and "bytes" in raw["note"]

    assistant, tool = row["threads"][0][1:]
    assert assistant["content"].startswith("x" * MAX_TEXT_CHARS)
    assert "500 characters truncated" in assistant["content"]
    assert sorted(assistant["tool_calls"][0]["arguments"]) == [1, 2]
    assert isinstance(tool["content"], str) and len(tool["content"]) < 300

    info = row["turns"][0]["info"]
    assert set(info) == {"obj", "bad", "nan", "3", "loop"}
    assert all(isinstance(info[k], str) for k in ("obj", "bad", "nan", "3"))
    assert isinstance(row["info"]["env"], str) and "x" in row["info"]["path"]
    assert "1 characters truncated" in row["info"]["big"]
