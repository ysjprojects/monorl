from __future__ import annotations

import json
import sys
import types

import pytest

from monorl.core.types import Completion, Sample
from monorl.envs import tool_calling as tc
from monorl.envs.tool_calling import (
    ToolCallingEnv,
    calculator,
    load_tools,
    lookup,
    make_tool_calling_env,
    parse_tool_calls,
    python_exec,
    tool,
)


def _c(text: str) -> Completion:
    return Completion(text=text, token_ids=[])


def _sample() -> Sample:
    return Sample(id="s1", messages=[{"role": "user", "content": "What is 6*7?"}], reference="42")


# -- parsing ---------------------------------------------------------------------------------


def test_parse_hermes_blocks_multiple():
    text = (
        'thinking...\n<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>\n'
        '<tool_call>{"name": "lookup", "arguments": "{\\"query\\": \\"x\\"}"}</tool_call>'
    )
    parsed = parse_tool_calls(text)
    assert not parsed.errors
    assert [c["name"] for c in parsed.calls] == ["calculator", "lookup"]
    assert parsed.calls[1]["arguments"] == {"query": "x"}
    assert all(c["id"] for c in parsed.calls)


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": "{\\"expression\\": \\"2*2\\"}"}}]}\n```',
        '{"name": "calculator", "arguments": {"expression": "2*2"}}',
        '[{"function": {"name": "calculator", "arguments": {"expression": "2*2"}}}]',
    ],
)
def test_parse_openai_shapes(text):
    parsed = parse_tool_calls(text)
    assert not parsed.errors
    assert len(parsed.calls) == 1
    assert parsed.calls[0]["name"] == "calculator"
    assert parsed.calls[0]["arguments"] == {"expression": "2*2"}


def test_parse_openai_keeps_call_id():
    parsed = parse_tool_calls('{"tool_calls": [{"id": "abc", "function": {"name": "x", "arguments": {}}}]}')
    assert parsed.calls[0]["id"] == "abc"


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("<tool_call>{not json}</tool_call>", "malformed"),
        ('<tool_call>{"arguments": {}}</tool_call>', "no 'name'"),
        ('<tool_call>{"name": "x", "arguments": "{bad"}</tool_call>', "not valid JSON"),
        ('<tool_call>{"name": "x", "arguments": [1]}</tool_call>', "must be an object"),
        ('<tool_call>{"name": "x"}', "unterminated"),
        ("<tool_call>[1, 2]</tool_call>", "not a tool call object"),
    ],
)
def test_parse_malformed(text, fragment):
    parsed = parse_tool_calls(text)
    assert not parsed.calls
    assert any(fragment in e for e in parsed.errors)


def test_parse_plain_answer_has_no_calls_or_errors():
    parsed = parse_tool_calls("The answer is {42}. Not JSON: {foo: bar}")
    assert not parsed.calls and not parsed.errors


# -- built-in tools ------------------------------------------------------------------------


def test_calculator():
    assert calculator("2 * (3 + 4)") == "14"
    assert calculator("2 ** 10 + sqrt(16)") == "1028"
    assert calculator("7 / 2") == "3.5"
    assert calculator("-pi").startswith("-3.14")
    assert calculator("1/0").startswith("error")
    assert calculator("__import__('os').system('ls')").startswith("error")
    assert calculator("9 ** 99999").startswith("error")
    assert calculator("x + 1").startswith("error")


def test_python_exec_runs_and_times_out():
    assert python_exec("print(sum(range(10)))") == "45\n"
    out = python_exec("import sys; print('e', file=sys.stderr); sys.exit(2)")
    assert "stderr:\ne" in out and "exit code: 2" in out
    assert "timed out" in python_exec("import time; time.sleep(5)", timeout_s=0.5)


def test_lookup():
    corpus = {"Paris": "capital of France", "Berlin": "capital of Germany"}
    assert lookup("Paris", corpus) == "capital of France"
    assert "Berlin" in lookup("germany", corpus)
    assert lookup("Rome", corpus).startswith("no results")


def test_tool_call_validates_arguments():
    calc = load_tools({"tools": ["calculator"]})[0]
    assert calc.name == "calculator" and calc.schema()["function"]["parameters"]["required"] == ["expression"]
    assert calc.call({}).startswith("error: missing")
    assert calc.call({"expression": "1+1", "extra": 1}).startswith("error: unknown argument")
    assert calc.call({"expression": "1+1"}) == "2"


def test_tool_output_truncation():
    tools = load_tools({"tools": ["python_exec"], "max_tool_output_chars": 20})
    out = tools[0].call({"code": "print('x' * 100)"})
    assert out.startswith("x" * 20) and "truncated" in out


def test_load_tools_unknown_and_registered(monkeypatch):
    with pytest.raises(KeyError):
        load_tools({"tools": ["nope"]})

    mod = types.ModuleType("_monorl_test_tools")

    def _define():
        @tool(
            "shout",
            "Upper-case the input.",
            {"type": "object", "properties": {"s": {"type": "string"}}, "required": ["s"]},
        )
        def shout(s: str) -> str:
            return s.upper()

    mod.define = _define
    _define()
    monkeypatch.setitem(sys.modules, "_monorl_test_tools", mod)
    try:
        tools = load_tools({"tools": ["shout", "calculator"], "tool_modules": ["_monorl_test_tools"]})
        assert [t.name for t in tools] == ["shout", "calculator"]
        assert tools[0].call({"s": "hi"}) == "HI"
    finally:
        tc._REGISTRY.pop("shout", None)


# -- environment ------------------------------------------------------------------------------


def test_reset_prepends_tool_system_prompt_and_merges_existing_system():
    env = make_tool_calling_env({"tools": ["calculator"]}, max_turns=3)
    msgs = env.reset(_sample())
    assert msgs[0]["role"] == "system" and "<tools>" in msgs[0]["content"]
    assert (
        json.loads(msgs[0]["content"].split("<tools>\n")[1].split("\n</tools>")[0])["function"]["name"]
        == "calculator"
    )
    assert msgs[1] == {"role": "user", "content": "What is 6*7?"}
    assert env.tools[0]["function"]["name"] == "calculator"

    env2 = make_tool_calling_env({"tools": ["calculator"], "system_prompt": "Tools: {tools}"}, max_turns=3)
    msgs = env2.reset(
        Sample(
            id="s", messages=[{"role": "system", "content": "Be terse."}, {"role": "user", "content": "q"}]
        )
    )
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("Tools: ") and msgs[0]["content"].endswith("Be terse.")
    assert len(msgs) == 2


def test_step_executes_tools_and_finishes_on_plain_answer():
    env = make_tool_calling_env(
        {"tools": ["calculator", "lookup"], "corpus": {"Paris": "capital of France"}}, max_turns=4
    )
    env.reset(_sample())
    r = env.step(
        _c(
            '<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>'
            '<tool_call>{"name": "lookup", "arguments": {"query": "Paris"}}</tool_call>'
        )
    )
    assert not r.done and r.reward == 0.0
    assert [m["role"] for m in r.messages] == ["tool", "tool"]
    assert r.messages[0]["name"] == "calculator" and r.messages[0]["content"] == "42"
    assert r.messages[1]["content"] == "capital of France"
    assert r.messages[0]["tool_call_id"] == r.info["tool_calls"][0]["id"]
    assert r.info["turn"] == 1 and r.info["invalid_calls"] == 0

    final = env.step(_c("The answer is 42."))
    assert final.done and final.messages == [] and final.info["final_answer"] is True
    assert env.turn == 2


def test_invalid_calls_are_penalised_once_per_turn():
    env = make_tool_calling_env({"tools": ["calculator"], "invalid_call_penalty": 0.3}, max_turns=5)
    env.reset(_sample())
    r = env.step(
        _c('<tool_call>{oops}</tool_call><tool_call>{"name": "missing", "arguments": {}}</tool_call>')
    )
    assert not r.done
    assert r.reward == pytest.approx(-0.3)
    assert r.info["invalid_calls"] == 2
    roles = [(m["role"], m["name"]) for m in r.messages]
    assert ("tool", "missing") in roles and ("tool", "error") in roles
    assert any("unknown tool" in m["content"] for m in r.messages)

    ok = env.step(_c('<tool_call>{"name": "calculator", "arguments": {"expression": "1"}}</tool_call>'))
    assert ok.reward == 0.0


def test_done_at_max_turns_even_with_tool_calls():
    env = make_tool_calling_env({"tools": ["calculator"]}, max_turns=2)
    env.reset(_sample())
    call = '<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'
    assert env.step(_c(call)).done is False
    last = env.step(_c(call))
    assert last.done is True and last.messages == [] and last.info["final_answer"] is False


def test_max_calls_per_turn():
    env = make_tool_calling_env(
        {"tools": ["calculator"], "max_calls_per_turn": 1, "invalid_call_penalty": 1.0}, max_turns=3
    )
    env.reset(_sample())
    call = '<tool_call>{"name": "calculator", "arguments": {"expression": "1"}}</tool_call>'
    r = env.step(_c(call * 3))
    assert len(r.info["tool_calls"]) == 1 and r.reward == -1.0


def test_reset_reuses_instance():
    env = ToolCallingEnv(tools=load_tools({"tools": ["calculator"]}), max_turns=2)
    env.reset(_sample())
    env.step(_c("done"))
    env.reset(_sample())
    assert env.turn == 0 and env.history == []
    env.close()
