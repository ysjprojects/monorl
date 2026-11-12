"""Example harness agent: a tool-using loop with one tool, a calculator.

It does what any agent trained through the capture gateway does, with the standard library only:
read the task file, talk OpenAI chat completions (function calling) to ``OPENAI_BASE_URL``, run
the tools the model calls, print the final answer::

    python -m monorl.gateway.agents.calc [--task-file FILE] [--base-url URL] [--api-key KEY]
                                         [--model NAME] [--max-steps N] [--request-timeout S]
                                         [--no-examples]

The defaults come from the variables ``env.type: harness`` sets (``MONORL_TASK_FILE``,
``OPENAI_BASE_URL``, ``OPENAI_API_KEY``, ``MONORL_MODEL``), so the program runs unchanged against
any OpenAI-compatible server.

Small models' chat templates render neither the ``tools`` of a request nor the ``tool_calls`` of
earlier assistant messages, so the conversation itself shows how the tool is used: the system
prompt spells the call syntax out, and two worked examples precede the task (``--no-examples``
leaves them out). An example is a question, an assistant message that carries the call both as
text (``<tool_call>...</tool_call>``, what such a template shows the model) and as structured
``tool_calls`` (what the API expects), the ``tool`` result, and the final number.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import sys
import urllib.error
import urllib.request
from typing import Any

SYSTEM_PROMPT = (
    "You solve arithmetic word problems. You have a calculator: to use it, reply with\n"
    '<tool_call>{"name": "calculator", "arguments": {"expression": "<arithmetic expression>"}}</tool_call>\n'
    "and you will be given the result. When you know the answer, reply with the final number and "
    "no tool call."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression (+ - * / // % ** and parentheses).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expression to evaluate, e.g. '2*(3+4)'."}
                },
                "required": ["expression"],
            },
        },
    }
]

EXAMPLES = [
    ("What is 17 * 3?", "17 * 3"),
    ("A crate holds 12 bottles. How many bottles are in 4 crates plus 5 loose bottles?", "12 * 4 + 5"),
]
"""``(question, calculator expression)`` of the worked examples shown before the task."""

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    raise ValueError("only numbers, + - * / // % ** and parentheses are supported")


def calculator(expression: str) -> str:
    """The value of an arithmetic expression, or ``error: ...``; never evaluates code."""
    try:
        value = _evaluate(ast.parse(str(expression).strip(), mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        return f"error: {e}"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return str(value)


def run_tool(call: dict[str, Any]) -> str:
    """Execute one OpenAI tool call (``{"id", "function": {"name", "arguments": <JSON string>}}``)."""
    function = call.get("function") or {}
    if function.get("name") != "calculator":
        return f"error: unknown tool {function.get('name')!r}"
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as e:
        return f"error: arguments are not valid JSON: {e.msg}"
    if not isinstance(arguments, dict) or "expression" not in arguments:
        return "error: missing required argument: expression"
    return calculator(arguments["expression"])


def chat(base_url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """``POST {base_url}/chat/completions``; exits with the server's error on a failed call."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"policy call failed with HTTP {e.code}: {e.read().decode('utf-8', 'replace')}"
        ) from e


def example_messages() -> list[dict[str, Any]]:
    """:data:`EXAMPLES` as chat messages: question, tool call, tool result, final number."""
    messages: list[dict[str, Any]] = []
    for n, (question, expression) in enumerate(EXAMPLES, start=1):
        arguments = {"expression": expression}
        call = {
            "id": f"call_example_{n}",
            "type": "function",
            "function": {"name": "calculator", "arguments": json.dumps(arguments)},
        }
        text = f"<tool_call>\n{json.dumps({'name': 'calculator', 'arguments': arguments})}\n</tool_call>"
        result = calculator(expression)
        messages += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": text, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "name": "calculator", "content": result},
            {"role": "assistant", "content": result},
        ]
    return messages


def solve(
    task: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_steps: int,
    timeout: float,
    examples: bool = True,
) -> str:
    """Run the tool loop on ``task`` (``{"messages": [...]}``) for at most ``max_steps`` policy
    calls; the final answer is the content of the model's last message. The chat is the system
    prompt (followed by the task's own, if it has one), the worked examples, the task."""
    system, rest = SYSTEM_PROMPT, list(task["messages"])
    if rest and rest[0].get("role") == "system" and isinstance(rest[0].get("content"), str):
        system = f"{SYSTEM_PROMPT}\n\n{rest.pop(0)['content']}"
    messages = [{"role": "system", "content": system}, *(example_messages() if examples else []), *rest]
    message: dict[str, Any] = {}
    for step in range(max_steps):
        reply = chat(base_url, api_key, {"model": model, "messages": messages, "tools": TOOLS}, timeout)
        message = reply["choices"][0]["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls or step == max_steps - 1:
            break
        for call in calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": run_tool(call),
                }
            )
    return message.get("content") or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="monorl.gateway.agents.calc", description=__doc__.splitlines()[0])
    parser.add_argument("--task-file", default=os.environ.get("MONORL_TASK_FILE"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    parser.add_argument("--model", default=os.environ.get("MONORL_MODEL", "monorl"))
    parser.add_argument("--max-steps", type=int, default=4, help="policy calls per task (default 4)")
    parser.add_argument("--request-timeout", type=float, default=600.0, help="seconds per policy call")
    parser.add_argument("--no-examples", action="store_true", help="leave the worked examples out")
    args = parser.parse_args(argv)
    if not args.task_file or not args.base_url:
        parser.error("--task-file and --base-url are required (or MONORL_TASK_FILE and OPENAI_BASE_URL)")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    with open(args.task_file, encoding="utf-8") as f:
        task = json.load(f)
    answer = solve(
        task,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_steps=args.max_steps,
        timeout=args.request_timeout,
        examples=not args.no_examples,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
