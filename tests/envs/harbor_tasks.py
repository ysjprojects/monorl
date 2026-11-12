"""Harbor task directories written by tests."""

from __future__ import annotations

from pathlib import Path

# tests/test.sh of the fake-run tasks: Harbor's contract, with the container root made explicit.
TEST_SH = """#!/bin/bash
logs="${FAKE_ROOT:-}/logs/verifier"
if [ -e "$(dirname "$0")/planted.txt" ] || [ -e "$logs/reward.txt" ]; then
  echo "the agent's files are still here" >&2
  exit 3
fi
echo "verifying as '$FAKE_DOCKER_USER'"
if [ "$(cat answer.txt)" = "$EXPECTED" ]; then echo 1 > "$logs/reward.txt"; else echo 0 > "$logs/reward.txt"; fi
"""


def write_task(
    directory: Path,
    toml: str,
    *,
    instruction: str = "abc",
    test_sh: str = TEST_SH,
    dockerfile: str | None = None,
) -> Path:
    """A Harbor task directory; the test script is deliberately not executable on this side."""
    (directory / "tests").mkdir(parents=True)
    (directory / "solution").mkdir()
    (directory / "task.toml").write_text(toml)
    (directory / "instruction.md").write_text(instruction)
    (directory / "tests" / "test.sh").write_text(test_sh)
    (directory / "tests" / "expected.json").write_text('{"answer": "the tests know"}')
    (directory / "solution" / "solve.sh").write_text("#!/bin/bash\necho 'the reference solution'\n")
    if dockerfile is not None:
        (directory / "environment").mkdir()
        (directory / "environment" / "Dockerfile").write_text(dockerfile)
        (directory / "environment" / "seed.txt").write_text("from the image")
    return directory
