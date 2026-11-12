"""Grade /app/sorted.txt: 1.0 for the numbers in ascending order, 0.5 for the right numbers in
another order, 0.0 otherwise. Writes the labelled metrics Harbor calls reward.json."""

import json
import sys

EXPECTED = [-3, 7, 8, 19, 42, 100]


def main(answer_file: str, reward_file: str) -> None:
    try:
        with open(answer_file, encoding="utf-8") as f:
            numbers = [int(line) for line in f.read().split()]
    except (OSError, ValueError):
        numbers = []
    complete = sorted(numbers) == EXPECTED
    ordered = numbers == EXPECTED
    with open(reward_file, "w", encoding="utf-8") as f:
        json.dump({"reward": 1.0 if ordered else 0.5 if complete else 0.0, "complete": int(complete)}, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
