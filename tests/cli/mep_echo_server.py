"""Minimal MEP/1 stdio server used by the `env check` tests (one JSON object per line).

Flags: ``--bad-spec`` omits the name, ``--die-on-step`` exits before answering step.
"""

import json
import sys

bad_spec = "--bad-spec" in sys.argv
die_on_step = "--die-on-step" in sys.argv
turns: dict[str, int] = {}

for line in sys.stdin:
    req = json.loads(line)
    op, ep = req.get("op"), req.get("episode_id")
    if op == "spec":
        out = {
            "version": "1",
            "protocol": "MEP/1",
            "max_turns": 3,
            "parallel_episodes": 2,
            "description": "echo",
        }
        if not bad_spec:
            out["name"] = "echo"
    elif op == "health":
        out = {"ok": True}
    elif op == "reset":
        turns[ep] = 0
        out = {"messages": req["sample"]["messages"], "info": {"seed": req.get("seed")}}
    elif op == "step":
        if die_on_step:
            sys.exit(3)
        turns[ep] += 1
        text = req["action"]["text"]
        done = turns[ep] >= 3
        out = {
            "messages": [] if done else [{"role": "user", "content": f"echo:{text}"}],
            "reward": float(len(text)),
            "done": done,
            "info": {"turn": turns[ep]},
        }
    elif op == "close":
        turns.pop(ep, None)
        out = {}
    else:
        out = {"error": {"type": "bad_op", "message": f"unknown op {op!r}"}}
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
