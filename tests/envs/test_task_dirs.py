"""Task directories (the Harbor task format) as a dataset: what a task becomes, what is refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.data import open_source
from monorl.data.browse import preview_uri
from monorl.envs.harness import task_payload
from tests.envs.harbor_tasks import write_task

EXAMPLES = Path(__file__).parents[2] / "examples" / "sandbox" / "tasks"

FULL_TASK = """
schema_version = "1.4"

[task]
name = "acme/full"
description = "everything a single-container task can say"

[metadata]
difficulty = "hard"

[agent]
timeout_sec = 900.0
user = 1000

[verifier]
timeout_sec = 120.0
user = "root"
env = { MODEL_NAME = "judge", API_KEY = "${JUDGE_API_KEY}" }

[environment]
build_timeout_sec = 1200.0
docker_image = "acme/full:3"
workdir = "/workspace"
cpus = 2
memory_mb = 4096
storage_mb = 10240
gpus = 1
env = { MODE = "test" }

[environment.healthcheck]
command = "curl -sf http://localhost:8080/"
retries = 5

[solution]
env = { API_KEY = "sk-solution" }
"""


def _rows(uri: Path | str, **extra: object) -> list:
    return list(open_source(DataConfig(uri=str(uri), shuffle=False, extra=dict(extra))))


def test_a_task_directory_becomes_a_sample_with_its_task_mapping(tmp_path: Path) -> None:
    # an environment/ directory next to docker_image: the prebuilt image wins
    task_dir = write_task(tmp_path / "tasks" / "full", FULL_TASK, dockerfile="FROM scratch\n")
    (sample,) = _rows(tmp_path / "tasks")
    assert (sample.id, sample.messages, sample.reference) == (
        "full",
        [{"role": "user", "content": "abc"}],
        None,
    )
    assert sample.metadata["row"]["task"] == {
        "name": "acme/full",
        "dir": str(task_dir.resolve()),
        "sandbox": {
            "image": "acme/full:3",
            "workdir": "/workspace",
            "keep_entrypoint": True,
            "build_timeout": 1200.0,
            "cpus": 2.0,
            "memory": "4096m",
            "gpus": "1",
        },
        "env": {"MODE": "test"},
        "timeout": 900.0,
        "user": "1000",
        "healthcheck": {
            "command": "curl -sf http://localhost:8080/",
            "interval_sec": 5.0,
            "timeout_sec": 30.0,
            "start_period_sec": 0.0,
            "start_interval_sec": 5.0,
            "retries": 5,
        },
        "verifier": {
            "tests": str(task_dir.resolve() / "tests"),
            "timeout": 120.0,
            "user": "root",
            "env": {"MODEL_NAME": "judge", "API_KEY": "${JUDGE_API_KEY}"},
        },
        "metadata": {"difficulty": "hard"},
    }
    # the agent's view of the sample has none of it
    assert task_payload(sample) == {
        "id": "full",
        "messages": [{"role": "user", "content": "abc"}],
        "metadata": {"row": {"id": "full"}},
    }


def test_defaults_legacy_fields_and_the_usual_data_options(tmp_path: Path) -> None:
    write_task(
        tmp_path / "tasks" / "plain",
        'version = "1.0"\n[environment]\nmemory = "2G"\n',
        instruction="# harbor-canary GUID 42\n<!-- canary: do not train -->\n\n\nFix the bug.\n\nThen run the tests.\n",
        dockerfile="FROM ubuntu:24.04\n",
    )
    config = DataConfig(
        uri=str(tmp_path / "tasks"), system_prompt="Be brief.", prompt_template="Task: {prompt}"
    )
    (sample,) = open_source(config)
    assert sample.messages == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Task: Fix the bug.\n\nThen run the tests."},
    ]
    task = sample.metadata["row"]["task"]
    assert task["name"] == "plain" and task["timeout"] is None and task["user"] is None
    assert task["sandbox"] == {
        "build": {"context": str((tmp_path / "tasks" / "plain" / "environment").resolve())},
        "workdir": "image",
        "keep_entrypoint": True,
        "build_timeout": 600.0,
        "memory": "2048m",
    }
    assert task["verifier"]["timeout"] == 600.0 and "healthcheck" not in task


def test_a_task_directory_or_a_glob_selects_tasks(tmp_path: Path) -> None:
    for name in ("alpha-1", "alpha-2", "beta"):
        write_task(tmp_path / "tasks" / name, '[environment]\ndocker_image = "box"\n')
    (tmp_path / "tasks" / "notes").mkdir()  # not a task: no task.toml
    assert [s.id for s in _rows(tmp_path / "tasks")] == ["alpha-1", "alpha-2", "beta"]
    assert [s.id for s in _rows(tmp_path / "tasks" / "beta")] == ["beta"]
    assert [s.id for s in _rows(f"{tmp_path}/tasks/alpha-*")] == ["alpha-1", "alpha-2"]
    limited = list(open_source(DataConfig(uri=str(tmp_path / "tasks"), limit=2, shuffle=True), seed=3))
    assert len(limited) == 2 and {s.id for s in limited} <= {"alpha-1", "alpha-2", "beta"}
    preview = preview_uri(str(tmp_path / "tasks"))
    assert preview["num_rows"] == 3 and preview["inferred"]["prompt_field"] == "instruction"


def test_tasks_that_cannot_be_honoured_are_refused_by_name(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    image = '[environment]\ndocker_image = "box"\n'
    write_task(tasks / "fine", image)
    write_task(tasks / "steps", image + "[[steps]]\nname = 'one'\n")
    write_task(tasks / "windows", image + 'os = "windows"\n')
    write_task(tasks / "offline", image + 'network_mode = "no-network"\n')
    write_task(tasks / "offline-legacy", image + "allow_internet = false\n")
    write_task(
        tasks / "filtered-agent",
        image + '[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n',
    )
    write_task(tasks / "separate", image + '[verifier]\nenvironment_mode = "separate"\n')
    write_task(tasks / "verifier-image", image + '[verifier.environment]\ndocker_image = "grader"\n')
    write_task(tasks / "no-environment", "")
    compose = write_task(tasks / "compose", image, dockerfile="FROM box\n")
    (compose / "environment" / "docker-compose.yaml").write_text("services: {}\n")
    (write_task(tasks / "no-tests", image) / "tests" / "test.sh").unlink()
    (write_task(tasks / "no-instruction", image) / "instruction.md").unlink()
    write_task(tasks / "broken", "[environment\n")
    write_task(tasks / "future", 'schema_version = "2.0"\n' + image)

    with pytest.raises(DataError) as refused:
        _rows(tasks)
    message = str(refused.value)
    for reason in (
        "13 of 14 task directories",
        "steps: multi-step task",
        "windows: [environment].os = 'windows'",
        "offline: [environment].network_mode = 'no-network'",
        "offline-legacy: [environment].network_mode = 'no-network'",
        "filtered-agent: [agent].network_mode = 'allowlist'",
        "separate: separate verifier environment",
        "verifier-image: separate verifier environment",
        "no-environment: no [environment].docker_image and no environment/Dockerfile",
        "compose: multi-container environment",
        "no-tests: no tests/test.sh",
        "no-instruction: no instruction.md",
        "broken: task.toml cannot be read",
        "future: schema_version 2.0",
        "data.extra.skip_unsupported",
    ):
        assert reason in message
    assert [s.id for s in _rows(tasks, skip_unsupported=True)] == ["fine"]
    with pytest.raises(DataError, match="none of the 1 task directories"):
        _rows(tasks / "steps", skip_unsupported=True)


def test_shipped_example_tasks_load() -> None:
    samples = _rows(EXAMPLES)
    assert [s.id for s in samples] == ["count-lines", "hello-file", "sort-numbers"]
    for sample in samples:
        task = sample.metadata["row"]["task"]
        assert sample.reference is None and sample.messages[-1]["content"].startswith(("Create", "The file"))
        assert ("image" in task["sandbox"]) != ("build" in task["sandbox"])
        assert Path(task["verifier"]["tests"], "test.sh").is_file()
        # every example ships its reference solution, and the agent is not told where
        assert Path(task["dir"], "solution", "solve.sh").is_file()
        assert "examples" not in str(task_payload(sample))
