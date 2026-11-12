"""Tiny MEP/1 stdio server for the env benchmark tests: one JSON object per line, two concurrent
episodes per process, a fixed 2 ms of "work" per step so parallelism is measurable."""

import json
import sys
import time

turns: dict[str, int] = {}
for line in sys.stdin:
    req = json.loads(line)
    op, ep = req.get("op"), req.get("episode_id")
    if op == "spec":
        out = {
            "name": "bench-echo",
            "version": "1",
            "protocol": "MEP/1",
            "max_turns": 3,
            "parallel_episodes": 2,
        }
    elif op == "health":
        out = {"ok": True}
    elif op == "reset":
        turns[ep] = 0
        out = {"messages": req["sample"]["messages"], "info": {}}
    elif op == "step":
        time.sleep(0.002)
        turns[ep] += 1
        done = turns[ep] >= 3
        out = {
            "messages": [] if done else [{"role": "user", "content": "again"}],
            "reward": 1.0,
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
