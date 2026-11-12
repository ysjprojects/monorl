"""Example harness agent for task directories: a loop with one tool, a shell.

The model is asked to complete the task of ``MONORL_TASK_FILE`` in the working directory, one
shell command per turn (OpenAI function calling against ``OPENAI_BASE_URL``); each command runs
with ``sh -c`` and its exit status and output go back to the model. The loop ends when the model
answers without a command or after ``--max-steps`` policy calls::

    python3 shell_agent.py [--task-file FILE] [--base-url URL] [--api-key KEY] [--model NAME]
                           [--workdir DIR] [--max-steps N] [--command-timeout S]
                           [--request-timeout S] [--no-examples]

Standard library only and Python 3.8+, so it runs in any task image that has a ``python3``:
mount this directory into the sandbox and name the script in ``env.params.command``
(``configs/recipes/grpo_harness_tasks_docker_cpu.yaml``).

Like ``monorl.gateway.agents.calc`` it shows small models how the tool is called: the system
prompt spells the call syntax out and two worked examples precede the task (``--no-examples``
leaves them out), each with the call both as text (``<tool_call>...</tool_call>``) and as
structured ``tool_calls``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any

SYSTEM_PROMPT = (
    "You complete a task on a Linux machine by running shell commands, one at a time. To run a "
    "command, reply with\n"
    '<tool_call>{"name": "shell", "arguments": {"command": "<shell command>"}}</tool_call>\n'
    "and you will be given its exit status and output. When the task is complete, reply DONE "
    "and no tool call."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a command with sh in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The command line to run."}},
                "required": ["command"],
            },
        },
    }
]

EXAMPLES = [
    (
        "Create the file /tmp/greeting.txt. It must contain exactly one line: good morning",
        "echo 'good morning' > /tmp/greeting.txt",
    ),
    ("Write the number of entries in /etc to /tmp/count.txt.", "ls /etc | wc -l > /tmp/count.txt"),
]
"""``(task, shell command)`` of the worked examples shown before the task."""

MAX_OUTPUT_CHARS = 2000
"""How much of the end of a command's output the model is shown."""


def shell(command: str, *, cwd: str | None, timeout: float) -> str:
    """Run ``command`` with ``sh -c``; its exit status and the end of its output."""
    try:
        done = subprocess.run(
            ["sh", "-c", command],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"error: the command did not finish within {timeout:g} seconds"
    output = done.stdout.decode("utf-8", "replace")[-MAX_OUTPUT_CHARS:]
    return f"exit status {done.returncode}\n{output}".rstrip()


def run_tool(call: dict[str, Any], *, cwd: str | None, timeout: float) -> str:
    """Execute one OpenAI tool call (``{"id", "function": {"name", "arguments": <JSON string>}}``)."""
    function = call.get("function") or {}
    if function.get("name") != "shell":
        return f"error: unknown tool {function.get('name')!r}"
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError as e:
        return f"error: arguments are not valid JSON: {e.msg}"
    if not isinstance(arguments, dict) or not isinstance(arguments.get("command"), str):
        return "error: missing required argument: command"
    return shell(arguments["command"], cwd=cwd, timeout=timeout)


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
    """:data:`EXAMPLES` as chat messages: task, tool call, tool result, ``DONE``."""
    messages: list[dict[str, Any]] = []
    for n, (task, command) in enumerate(EXAMPLES, start=1):
        arguments = {"command": command}
        call = {
            "id": f"call_example_{n}",
            "type": "function",
            "function": {"name": "shell", "arguments": json.dumps(arguments)},
        }
        text = f"<tool_call>\n{json.dumps({'name': 'shell', 'arguments': arguments})}\n</tool_call>"
        messages += [
            {"role": "user", "content": task},
            {"role": "assistant", "content": text, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "name": "shell", "content": "exit status 0"},
            {"role": "assistant", "content": "DONE"},
        ]
    return messages


def solve(
    task: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    cwd: str | None,
    max_steps: int,
    command_timeout: float,
    request_timeout: float,
    examples: bool = True,
) -> str:
    """Run the tool loop on ``task`` (``{"messages": [...]}``) for at most ``max_steps`` policy
    calls; returns the model's last message. The chat is the system prompt (followed by the
    task's own, if it has one), the worked examples, the task."""
    system, rest = SYSTEM_PROMPT, list(task["messages"])
    if rest and rest[0].get("role") == "system" and isinstance(rest[0].get("content"), str):
        system = f"{SYSTEM_PROMPT}\n\n{rest.pop(0)['content']}"
    messages = [{"role": "system", "content": system}, *(example_messages() if examples else []), *rest]
    message: dict[str, Any] = {}
    for _ in range(max_steps):
        payload = {"model": model, "messages": messages, "tools": TOOLS}
        message = chat(base_url, api_key, payload, request_timeout)["choices"][0]["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            break
        for call in calls:  # also after the last policy call: what it asked for is still done
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": run_tool(call, cwd=cwd, timeout=command_timeout),
                }
            )
    return message.get("content") or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shell_agent.py", description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--task-file", default=os.environ.get("MONORL_TASK_FILE"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    parser.add_argument("--model", default=os.environ.get("MONORL_MODEL", "monorl"))
    parser.add_argument("--workdir", default=os.environ.get("MONORL_WORKDIR"), help="where commands run")
    parser.add_argument("--max-steps", type=int, default=4, help="policy calls per task (default 4)")
    parser.add_argument("--command-timeout", type=float, default=60.0, help="seconds per shell command")
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
        cwd=args.workdir,
        max_steps=args.max_steps,
        command_timeout=args.command_timeout,
        request_timeout=args.request_timeout,
        examples=not args.no_examples,
    )
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
