#!/bin/bash
# Reward 1 when /app/answer.txt holds the number of lines the image's words.txt has, else 0.
answer=$(tr -d '[:space:]' < /app/answer.txt 2>/dev/null)
if [ "$answer" = "7" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
