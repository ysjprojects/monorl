from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from nemo_rl_fakes import FakeLogger

from monorl.adapters.nemo_rl import PATCHES, is_available
from monorl.adapters.nemo_rl.entry import install_bridges
from monorl.adapters.nemo_rl.env_bridge import ACTOR_FQN
from monorl.adapters.nemo_rl.patches import (
    RUN_DIR_ENV,
    append_metrics_row,
    build_log_metrics,
    phase_for,
    translate_metrics,
)
from monorl.core.reporting import iter_metrics
from monorl.patching import apply, is_applied, revert, status


def test_patchset_declaration() -> None:
    assert PATCHES.framework == "nemo_rl" and PATCHES.distribution == "nemo-rl"
    patch = PATCHES.get("nemo_rl.logger_metrics")
    assert patch.target == "nemo_rl.utils.logger.Logger.log_metrics"
    assert patch.mode == "wrap" and patch.versions == ">=0.3,<1"
    assert patch.version_applicable("0.7.0+abc123") == (True, "")
    assert patch.version_applicable("0.2.1")[0] is False
    assert patch.version_applicable("1.0.0")[0] is False


def test_translate_metrics_maps_contract_names() -> None:
    train = translate_metrics(
        {
            "loss": np.float64(0.25),
            "reward": torch.tensor(0.75),
            "kl_penalty": 0.01,
            "token_mult_prob_error": 1.02,
            "mean_response_length": 120,
            "mean_prompt_length": 30,
            "lr": 1e-6,
            "grad_norm": 0.3,
            "generation_logger_metrics": {"x": 1},
            "advantages/mean": 0.0,
            "nan_metric": float("nan"),
            "vector": torch.tensor([1.0, 2.0]),
        },
        "train",
    )
    assert train == {
        "loss/policy": 0.25,
        "reward/mean": 0.75,
        "loss/kl": 0.01,
        "policy/token_mult_prob_error": 1.02,
        "tokens/response_mean": 120.0,
        "tokens/prompt_mean": 30.0,
        "lr": 1e-6,
        "grad_norm": 0.3,
        "nemo/train/advantages/mean": 0.0,
    }
    assert translate_metrics({"mean_gen_tokens_per_sample": 64}, "train") == {"tokens/response_mean": 64.0}
    timing = translate_metrics({"total_step_time": 10.0, "generation": 4.0, "policy_training": 5.0, "weight_sync": 1.0, "data_processing": 0.1}, "timing/train")
    assert timing == {"time/step": 10.0, "time/rollout": 4.0, "time/train": 5.0, "time/sync": 1.0, "time/data": 0.1}
    assert translate_metrics({"accuracy": 0.5}, "validation") == {"eval/accuracy": 0.5}
    assert translate_metrics({"total_validation_time": 2.0}, "timing/validation") == {"time/eval/total_validation_time": 2.0}
    assert translate_metrics({"tokens_per_sec": 3.0}, "performance") == {"perf/tokens_per_sec": 3.0}
    assert translate_metrics({"x": 1}, None) == {"nemo/x": 1.0}
    assert (phase_for("train"), phase_for("validation"), phase_for("timing/validation"), phase_for("timing/setup")) == ("train", "eval", "eval", "setup")


def test_append_metrics_row(tmp_path: Path) -> None:
    row = append_metrics_row(tmp_path, 3, {"loss": 0.5, "generation_logger_metrics": {}}, "train")
    assert row["step"] == 3 and row["phase"] == "train" and row["loss/policy"] == 0.5
    assert append_metrics_row(tmp_path, 3, {"only": "strings"}, "train") is None
    rows = [r for _, r in iter_metrics(tmp_path)]
    assert len(rows) == 1 and rows[0]["loss/policy"] == 0.5


def test_build_log_metrics_wraps_and_mirrors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def original(self, metrics, step, prefix="", step_metric=None, step_finished=False):
        seen.append((metrics, step, prefix, step_finished))
        return "upstream"

    wrapped = build_log_metrics(original)
    monkeypatch.delenv(RUN_DIR_ENV, raising=False)
    assert wrapped(object(), {"loss": 1.0}, 1, prefix="train") == "upstream"
    assert not (tmp_path / "metrics.jsonl").exists()
    monkeypatch.setenv(RUN_DIR_ENV, str(tmp_path))
    wrapped(object(), {"loss": 1.0, "reward": 2.0}, 2, "train", None, True)
    wrapped(object(), metrics={"total_step_time": 1.5}, step=2, prefix="timing/train")
    assert len(seen) == 3 and seen[1][3] is True
    rows = [r for _, r in iter_metrics(tmp_path)]
    assert [r["step"] for r in rows] == [2, 2]
    assert rows[0]["loss/policy"] == 1.0 and rows[0]["reward/mean"] == 2.0
    assert rows[1]["time/step"] == 1.5


def test_apply_against_fake_nemo_rl(fake_nemo_rl, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert is_available()
    report = apply("nemo_rl")
    assert report.ok and report.applied == ["nemo_rl.logger_metrics"], report
    assert report.installed_version == "0.7.0"
    assert is_applied("nemo_rl.logger_metrics", "nemo_rl")
    st = status("nemo_rl")[0]
    assert st.applied and st.applicable and st.verified

    monkeypatch.setenv(RUN_DIR_ENV, str(tmp_path))
    logger = FakeLogger({"log_dir": str(tmp_path)})
    logger.log_metrics({"loss": 0.1, "reward": 0.9, "lr": 1e-6}, 5, prefix="train")
    logger.log_metrics({"total_step_time": 2.0}, 5, prefix="timing/train", step_finished=True)
    assert FakeLogger.calls == [({"loss": 0.1, "reward": 0.9, "lr": 1e-6}, 5, "train"), ({"total_step_time": 2.0}, 5, "timing/train")]
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["step"] == 5 and rows[0]["loss/policy"] == 0.1 and rows[0]["reward/mean"] == 0.9 and rows[0]["lr"] == 1e-6
    assert rows[1]["time/step"] == 2.0

    assert apply("nemo_rl").skipped == [("nemo_rl.logger_metrics", "already applied")]
    revert("nemo_rl")
    assert not is_applied("nemo_rl.logger_metrics", "nemo_rl")
    FakeLogger({}).log_metrics({"loss": 0.2}, 6, prefix="train")
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2  # original restored, nothing mirrored


def test_install_bridges_registers_env_and_actor(fake_nemo_rl) -> None:
    install_bridges()
    env_utils = fake_nemo_rl["nemo_rl.environments.utils"]
    actor_registry = fake_nemo_rl["nemo_rl.distributed.ray_actor_environment_registry"]
    assert env_utils.ENV_REGISTRY["monorl"] == {"actor_class_fqn": ACTOR_FQN}
    assert actor_registry.ACTOR_ENVIRONMENT_REGISTRY[ACTOR_FQN] == "python"
    assert is_applied("nemo_rl.logger_metrics", "nemo_rl")
    install_bridges()  # idempotent
    assert list(env_utils.ENV_REGISTRY) == ["math", "monorl"]


def test_install_bridges_requires_nemo_rl() -> None:
    from monorl.core.errors import BackendUnavailable

    with pytest.raises(BackendUnavailable, match="not importable"):
        install_bridges()
