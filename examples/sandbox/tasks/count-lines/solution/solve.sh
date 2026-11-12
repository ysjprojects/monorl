#!/bin/bash
wc -l < /app/words.txt | tr -d ' ' > /app/answer.txt
