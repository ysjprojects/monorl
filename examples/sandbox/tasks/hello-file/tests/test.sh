#!/bin/bash
# Reward 1 when /app/hello.txt holds exactly the expected line, else 0.
if [ -f /app/hello.txt ] && [ "$(cat /app/hello.txt)" = "Hello, world!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
