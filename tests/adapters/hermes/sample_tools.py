"""A custom tool registered through Hermes' own API (``tools.registry.registry.register``).

Imported by ``HermesEnv`` via ``params["tool_modules"]``; the tool then becomes selectable by
name in ``params["tools"]`` like any built-in.
"""

from __future__ import annotations

import ast
import json
import operator
from typing import Any

from tools.registry import registry

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("unsupported expression")


def monorl_calc(args: dict[str, Any], **_: Any) -> str:
    try:
        value = _eval(ast.parse(str(args.get("expression", "")), mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return json.dumps({"error": str(e)})
    return json.dumps({"result": value})


if registry.get_entry("monorl_calc") is None:
    registry.register(
        name="monorl_calc",
        toolset="monorl-test",
        schema={
            "name": "monorl_calc",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. 2*(3+4)"}},
                "required": ["expression"],
            },
        },
        handler=monorl_calc,
        emoji="🧮",
    )
