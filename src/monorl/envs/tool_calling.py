"""Multi-turn tool-calling environment.

The policy sees a system prompt listing the available tools (OpenAI-style JSON schemas) and
answers either with a final reply or with one or more tool calls. Two call syntaxes are accepted:

* Hermes style: ``<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>``
* OpenAI style: a JSON object (optionally in a ``json`` code fence) of the form
  ``{"tool_calls": [{"type": "function", "function": {"name": ..., "arguments": ...}}]}``, a flat
  ``{"name": ..., "arguments": ...}`` object, or a list of either.

Each call is executed and its result appended as a ``tool`` message; the episode ends when the
assistant answers without tool calls or the turn budget is exhausted. Malformed or unknown calls
cost ``invalid_call_penalty`` once per turn. Final-answer scoring is left to reward functions
(the final assistant message is the last ``assistant`` turn of the trajectory).

Built-in tools: ``calculator`` (safe AST arithmetic), ``python_exec`` (isolated subprocess with a
timeout) and ``lookup`` (in-memory corpus from ``params["corpus"]``). User tools register with
:func:`tool`; list the modules that define them in ``params["tool_modules"]`` so they are imported.

Python-level interface used by other backends (verl/nemo bridges)::

    load_tools(params) -> list[Tool]      # Tool.name/.description/.parameters/.call(arguments)->str
    parse_tool_calls(text) -> ParsedToolCalls
"""

from __future__ import annotations

import ast
import importlib
import json
import math
import operator
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from monorl.core.types import Completion, Message, Sample, StepResult, ToolCall

ToolFn = Callable[..., Any]

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

DEFAULT_SYSTEM_PROMPT = (
    "You are a function-calling AI model. You are provided with function signatures within "
    "<tools></tools> XML tags. You may call one or more functions to assist with the user query. "
    "Don't make assumptions about what values to plug into functions.\n"
    "<tools>\n{tools}\n</tools>\n"
    "For each function call return a JSON object with function name and arguments within "
    "<tool_call></tool_call> XML tags as follows:\n"
    '<tool_call>\n{{"name": <function-name>, "arguments": <args-dict>}}\n</tool_call>\n'
    "When you have the final answer, reply with it directly without any tool call."
)


# -- tools --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn
    max_output_chars: int = 4000

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def call(self, arguments: Mapping[str, Any]) -> str:
        """Run the tool; argument errors are returned as text (never raised)."""
        required = self.parameters.get("required", [])
        missing = [k for k in required if k not in arguments]
        if missing:
            return f"error: missing required argument(s): {', '.join(missing)}"
        props = self.parameters.get("properties") or {}
        if props and not self.parameters.get("additionalProperties", False):
            unknown = [k for k in arguments if k not in props]
            if unknown:
                return f"error: unknown argument(s): {', '.join(unknown)}"
        try:
            result = self.fn(**arguments)
        except TypeError as e:
            return f"error: bad arguments: {e}"
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + f"... [truncated {len(text) - self.max_output_chars} chars]"
        return text


_REGISTRY: dict[str, Tool] = {}


def tool(name: str, description: str, parameters: dict[str, Any]) -> Callable[[ToolFn], ToolFn]:
    """Register ``fn(**arguments) -> str | json-able`` as a tool available to ``load_tools``."""

    def deco(fn: ToolFn) -> ToolFn:
        _REGISTRY[name] = Tool(name=name, description=description, parameters=parameters, fn=fn)
        return fn

    return deco


def registered_tools() -> dict[str, Tool]:
    return dict(_REGISTRY)


# -- built-ins -------------------------------------------------------------------------------

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*(_safe_eval(a) for a in node.args))
    raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression without ``eval``."""
    try:
        tree = ast.parse(str(expression).strip(), mode="eval")
        value = _safe_eval(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        return f"error: {e}"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return str(value)


def python_exec(code: str, *, timeout_s: float = 5.0, max_output_chars: int = 4000) -> str:
    """Run ``code`` in a fresh isolated interpreter (``-I``) with a wall-clock timeout."""
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", str(code)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=tempfile.gettempdir(),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return f"error: execution timed out after {timeout_s}s"
    out = proc.stdout
    if proc.stderr:
        out += ("\n" if out else "") + "stderr:\n" + proc.stderr
    if proc.returncode != 0:
        out += ("\n" if out else "") + f"exit code: {proc.returncode}"
    out = out or "(no output)"
    if len(out) > max_output_chars:
        out = out[:max_output_chars] + "... [truncated]"
    return out


def lookup(query: str, corpus: Mapping[str, str], *, max_results: int = 3) -> str:
    """Exact key match first, then case-insensitive substring search over keys and values."""
    q = str(query).strip()
    if q in corpus:
        return str(corpus[q])
    ql = q.lower()
    hits = [(k, v) for k, v in corpus.items() if ql and (ql in k.lower() or ql in str(v).lower())]
    if not hits:
        return f"no results for {q!r}"
    return "\n".join(f"{k}: {v}" for k, v in hits[:max_results])


def builtin_tools(params: Mapping[str, Any]) -> dict[str, Tool]:
    timeout_s = float(params.get("python_timeout_s", 5.0))
    max_out = int(params.get("max_tool_output_chars", 4000))
    corpus = {str(k): str(v) for k, v in dict(params.get("corpus") or {}).items()}
    return {
        "calculator": Tool(
            name="calculator",
            description="Evaluate an arithmetic expression (+ - * / // % **, abs, round, sqrt, log, exp, sin, cos, pi, e).",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Expression to evaluate, e.g. '2*(3+4)'."}},
                "required": ["expression"],
            },
            fn=calculator,
            max_output_chars=max_out,
        ),
        "python_exec": Tool(
            name="python_exec",
            description="Execute a Python snippet in an isolated interpreter and return its stdout/stderr.",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to run; print() what you need."}},
                "required": ["code"],
            },
            fn=lambda code: python_exec(code, timeout_s=timeout_s, max_output_chars=max_out),
            max_output_chars=max_out,
        ),
        "lookup": Tool(
            name="lookup",
            description="Look up a term in the reference corpus; returns matching entries.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Term or phrase to search for."}},
                "required": ["query"],
            },
            fn=lambda query: lookup(query, corpus, max_results=int(params.get("lookup_max_results", 3))),
            max_output_chars=max_out,
        ),
    }


def load_tools(params: Mapping[str, Any]) -> list[Tool]:
    """Resolve ``params["tools"]`` (names; default: all built-ins) against built-ins + registered tools."""
    for mod in params.get("tool_modules") or []:
        importlib.import_module(mod)
    available = {**builtin_tools(params), **_REGISTRY}
    names = params.get("tools")
    if names is None:
        names = ["calculator", "python_exec", "lookup"]
    out: list[Tool] = []
    for name in names:
        if name not in available:
            raise KeyError(f"unknown tool {name!r}; available: {sorted(available)}")
        out.append(available[name])
    return out


# -- parsing ---------------------------------------------------------------------------------


@dataclass(slots=True)
class ParsedToolCalls:
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_calls(self) -> bool:
        return bool(self.calls)


def _coerce_call(obj: Any, out: ParsedToolCalls) -> None:
    if not isinstance(obj, dict):
        out.errors.append(f"tool call must be an object, got {type(obj).__name__}")
        return
    if "function" in obj and isinstance(obj["function"], dict):  # OpenAI shape
        call_id = obj.get("id")
        obj = obj["function"]
    else:
        call_id = obj.get("id")
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        out.errors.append("tool call has no 'name'")
        return
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            out.errors.append(f"arguments of {name!r} are not valid JSON: {e.msg}")
            return
    if not isinstance(args, dict):
        out.errors.append(f"arguments of {name!r} must be an object")
        return
    out.calls.append(ToolCall(id=str(call_id or f"call_{uuid.uuid4().hex[:8]}"), name=name, arguments=args))


def _coerce_payload(obj: Any, out: ParsedToolCalls) -> bool:
    """Interpret a decoded JSON value as OpenAI-style calls. Returns True if it looked like one."""
    if isinstance(obj, dict) and isinstance(obj.get("tool_calls"), list):
        for c in obj["tool_calls"]:
            _coerce_call(c, out)
        return True
    if isinstance(obj, dict) and ("name" in obj or "function" in obj):
        _coerce_call(obj, out)
        return True
    if isinstance(obj, list) and obj and all(isinstance(c, dict) for c in obj):
        if all("name" in c or "function" in c for c in obj):
            for c in obj:
                _coerce_call(c, out)
            return True
    return False


def parse_tool_calls(text: str) -> ParsedToolCalls:
    out = ParsedToolCalls()
    blocks = TOOL_CALL_RE.findall(text)
    if blocks:
        for raw in blocks:
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                out.errors.append(f"malformed <tool_call> JSON: {e.msg}")
                continue
            if isinstance(obj, dict) and not isinstance(obj.get("tool_calls"), list):
                _coerce_call(obj, out)
            elif not _coerce_payload(obj, out):
                out.errors.append("<tool_call> block is not a tool call object")
        return out
    if "<tool_call>" in text:  # opened but never closed
        out.errors.append("unterminated <tool_call> block")
        return out
    candidates = _FENCE_RE.findall(text) or [text.strip()]
    for raw in candidates:
        raw = raw.strip()
        if not raw.startswith(("{", "[")):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        _coerce_payload(obj, out)
    return out


# -- environment ----------------------------------------------------------------------------


class ToolCallingEnv:
    def __init__(
        self,
        *,
        tools: list[Tool],
        max_turns: int = 4,
        system_prompt: str | None = None,
        invalid_call_penalty: float = 0.0,
        max_calls_per_turn: int = 8,
    ) -> None:
        self._tools = {t.name: t for t in tools}
        self.tools: list[dict[str, Any]] = [t.schema() for t in tools]
        self.max_turns = max(1, int(max_turns))
        self.system_prompt_template = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.invalid_call_penalty = float(invalid_call_penalty)
        self.max_calls_per_turn = int(max_calls_per_turn)
        self._sample: Sample | None = None
        self._turn = 0
        self.history: list[dict[str, Any]] = []

    @property
    def turn(self) -> int:
        return self._turn

    def render_system_prompt(self) -> str:
        tools_json = "\n".join(json.dumps(s, ensure_ascii=False) for s in self.tools)
        try:
            return self.system_prompt_template.format(tools=tools_json)
        except (KeyError, IndexError, ValueError):
            return self.system_prompt_template.replace("{tools}", tools_json)

    def reset(self, sample: Sample) -> list[Message]:
        self._sample = sample
        self._turn = 0
        self.history = []
        messages = list(sample.messages)
        system = self.render_system_prompt()
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": f"{system}\n\n{messages[0].get('content', '')}".strip()}
        else:
            messages.insert(0, {"role": "system", "content": system})
        return messages

    def step(self, action: Completion) -> StepResult:
        self._turn += 1
        parsed = parse_tool_calls(action.text)
        errors = list(parsed.errors)
        results: list[Message] = []
        executed: list[dict[str, Any]] = []
        calls = parsed.calls[: self.max_calls_per_turn]
        if len(parsed.calls) > self.max_calls_per_turn:
            errors.append(f"too many tool calls ({len(parsed.calls)} > {self.max_calls_per_turn}); extra calls ignored")
        for call in calls:
            name = call["name"]
            tool_ = self._tools.get(name)
            if tool_ is None:
                errors.append(f"unknown tool {name!r}")
                content = f"error: unknown tool {name!r}; available: {sorted(self._tools)}"
            else:
                content = tool_.call(call.get("arguments") or {})
            results.append({"role": "tool", "name": name, "tool_call_id": call["id"], "content": content})
            executed.append({"id": call["id"], "name": name, "arguments": call.get("arguments"), "result": content})
        for err in parsed.errors:
            results.append({"role": "tool", "name": "error", "content": f"error: {err}"})
        invalid = len(errors)
        reward = -self.invalid_call_penalty if invalid else 0.0
        final_answer = not parsed.calls and not parsed.errors
        done = final_answer or self._turn >= self.max_turns
        self.history.append({"turn": self._turn, "calls": executed, "errors": errors})
        if done:
            results = []
        return StepResult(
            messages=results,
            reward=reward,
            done=done,
            info={"turn": self._turn, "tool_calls": executed, "invalid_calls": invalid, "final_answer": final_answer},
        )

    def close(self) -> None:
        self._sample = None


def make_tool_calling_env(params: Mapping[str, Any], *, max_turns: int, system_prompt: str | None = None) -> ToolCallingEnv:
    return ToolCallingEnv(
        tools=load_tools(params),
        max_turns=max_turns,
        system_prompt=params.get("system_prompt", system_prompt),
        invalid_call_penalty=float(params.get("invalid_call_penalty", 0.0)),
        max_calls_per_turn=int(params.get("max_calls_per_turn", 8)),
    )


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "ParsedToolCalls",
    "Tool",
    "ToolCallingEnv",
    "builtin_tools",
    "calculator",
    "load_tools",
    "lookup",
    "make_tool_calling_env",
    "parse_tool_calls",
    "python_exec",
    "registered_tools",
    "tool",
]
