"""The suite under pytest-xdist (``pytest tests -n 4``): what keeps its workers apart."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PROBE = """
import os
from pathlib import Path


def test_report_scratch_home():
    worker = os.environ["PYTEST_XDIST_WORKER"]
    (Path(os.environ["PROBE_OUT"]) / worker).write_text(os.environ["MONORL_HOME"])
"""


def test_each_xdist_worker_has_a_scratch_home_of_its_own(tmp_path: Path) -> None:
    """Two workers of one run must not share ``MONORL_HOME`` (Hermes keeps a state database there,
    the media cache its downloads), and the home the caller chose stays the parent of both."""
    probe = tmp_path / "probe" / "test_probe.py"
    probe.parent.mkdir()
    probe.write_text(PROBE)
    out = tmp_path / "out"
    out.mkdir()
    home = tmp_path / "home"
    # the run below must not take itself for a worker of the run this test is part of
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env.update(MONORL_HOME=str(home), PROBE_OUT=str(out))

    # ``--dist each`` runs the probe on every worker; ``-p tests.conftest`` because the probe is
    # not below ``tests/``, where pytest would find the conftest by itself.
    command = [sys.executable, "-m", "pytest", str(probe), "--rootdir", str(probe.parent)]
    command += ["-p", "tests.conftest", "-n", "2", "--dist", "each", "-q", "-p", "no:cacheprovider"]
    proc = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=300)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    homes = {path.name: Path(path.read_text()) for path in out.iterdir()}
    assert homes == {"gw0": home / "gw0", "gw1": home / "gw1"}
