"""``pipeline_backend: nemo_rl`` — compile, export, launch NeMo-RL in a subprocess, collect results.

The NeMo-RL checkout (``NEMO_RL_HOME`` / ``trainer.extra["nemo_rl_home"]``) is the subprocess cwd so
``uv run`` picks up its project environment (``uv.lock``) — the layout NeMo-RL documents for
``uv run python examples/run_grpo.py --config <yaml>``. The child runs
:mod:`monorl.adapters.nemo_rl.entry`, which applies the patches, registers the env bridge and then
executes ``examples/run_grpo.py`` itself.

``trainer.extra`` knobs: ``nemo_rl_home`` (checkout), ``nemo_rl_launcher`` (``"uv"`` | ``"python"``,
default ``uv`` when a ``uv`` binary and ``<home>/pyproject.toml`` exist), ``nemo_rl_python``
(interpreter for the ``python`` launcher; default ``sys.executable``), ``nemo_rl_overrides`` (list of
hydra-style overrides appended to the command), ``nemo_rl`` (config deep-merge, see ``compile``).
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import monorl
from monorl.adapters.nemo_rl.compile import compile_config, write_config
from monorl.adapters.nemo_rl.data import export_dataset, write_bridge_module
from monorl.adapters.nemo_rl.entry import HOME_ENV, RUN_SCRIPT, SFT_SCRIPT, resolve_home
from monorl.adapters.nemo_rl.patches import ALGORITHM_ENV, RUN_DIR_ENV
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.core.reporting import CHECKPOINT_DIR, RunReporter, iter_metrics
from monorl.core.types import RunResult, RunState

ENTRY_MODULE = "monorl.adapters.nemo_rl.entry"

#: Seconds between status.json step refreshes while the child is running.
STEP_REFRESH_INTERVAL = 5.0
#: Grace period after SIGTERM before SIGKILL on cancel.
CANCEL_GRACE_SECONDS = 30.0


def nemo_rl_home(config: RunConfig) -> Path:
    """Resolve the checkout from ``trainer.extra['nemo_rl_home']`` or ``$NEMO_RL_HOME``."""
    return resolve_home(config.trainer.extra.get("nemo_rl_home"))


def build_command(config: RunConfig, home: Path, yaml_path: Path, run_dir: Path) -> list[str]:
    """The argv launched with ``cwd=home`` (``--script run_sft.py`` for ``algorithm.name: sft``)."""
    extra = config.trainer.extra
    launcher = extra.get("nemo_rl_launcher")
    if launcher is None:
        launcher = "uv" if shutil.which("uv") and (home / "pyproject.toml").is_file() else "python"
    if launcher == "uv":
        prefix = ["uv", "run", "python"]
    elif launcher == "python":
        prefix = [str(extra.get("nemo_rl_python") or sys.executable)]
    else:
        raise ConfigError(f"trainer.extra['nemo_rl_launcher'] must be 'uv' or 'python', got {launcher!r}")
    overrides = extra.get("nemo_rl_overrides") or []
    if not isinstance(overrides, list) or not all(isinstance(o, str) for o in overrides):
        raise ConfigError("trainer.extra['nemo_rl_overrides'] must be a list of strings")
    return [
        *prefix,
        "-m",
        ENTRY_MODULE,
        "--config",
        str(yaml_path),
        "--run-dir",
        str(run_dir),
        "--nemo-rl-home",
        str(home),
        "--script",
        str(SFT_SCRIPT if config.is_sft else RUN_SCRIPT),
        *overrides,
    ]


def monorl_src_root() -> Path:
    """Directory that makes ``import monorl`` work (the ``src`` dir or site-packages)."""
    return Path(monorl.__file__).resolve().parent.parent


def build_env(config: RunConfig, home: Path, run_dir: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    """Child environment: ``RAY_ADDRESS`` (when ``cluster.ray_address``), ``MONORL_RUN_DIR``,
    ``MONORL_ALGORITHM`` (the logger patch names ``train/loss`` ``loss/sft`` for SFT), ``NEMO_RL_HOME``
    and a ``PYTHONPATH`` that resolves the run-dir shim module and ``monorl``."""
    env = dict(os.environ if base is None else base)
    env[RUN_DIR_ENV] = str(run_dir)
    env[ALGORITHM_ENV] = config.algorithm.name
    env[HOME_ENV] = str(home)
    env["PYTHONUNBUFFERED"] = "1"
    if config.cluster.ray_address:
        env["RAY_ADDRESS"] = config.cluster.ray_address
    entries = [str(run_dir), str(monorl_src_root())]
    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p not in entries]
    env["PYTHONPATH"] = os.pathsep.join(entries + existing)
    return env


def _checkpoint_step(path: Path) -> int | None:
    name = path.name
    for prefix in ("step_", "step-"):
        if name.startswith(prefix):
            try:
                return int(name[len(prefix) :])
            except ValueError:
                return None
    return None


def list_checkpoints(run_dir: Path) -> list[str]:
    """NeMo-RL ``step_<n>`` (and monorl ``step-<n>``) checkpoint dirs under ``<run_dir>/checkpoints``, by step."""
    root = run_dir / CHECKPOINT_DIR
    if not root.is_dir():
        return []
    found = [(s, p) for p in root.iterdir() if p.is_dir() and (s := _checkpoint_step(p)) is not None]
    return [str(p) for _, p in sorted(found)]


def final_model_path(checkpoints: list[str]) -> str | None:
    """HF-loadable weights of the newest checkpoint (NeMo-RL writes ``<ckpt>/policy/weights``)."""
    if not checkpoints:
        return None
    last = Path(checkpoints[-1])
    for candidate in (last / "policy" / "weights", last / "policy", last):
        if candidate.is_dir():
            return str(candidate)
    return None


def latest_step(run_dir: Path) -> int:
    step = 0
    for _, row in iter_metrics(run_dir):
        try:
            step = max(step, int(row.get("step", 0)))
        except (TypeError, ValueError):
            continue
    return step


class NemoRLBackend:
    """:class:`~monorl.core.interfaces.PipelineBackend` for NeMo-RL."""

    name = "nemo_rl"

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config
        self._proc: subprocess.Popen[str] | None = None
        self._cancelled = threading.Event()
        self._lock = threading.Lock()

    # -- cancel --------------------------------------------------------------------------
    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self, *, grace: float = CANCEL_GRACE_SECONDS) -> None:
        """Signal the NeMo-RL process group (SIGTERM, then SIGKILL after ``grace`` seconds)."""
        self._cancelled.set()
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        _signal_group(proc, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if proc.poll() is None:
            _signal_group(proc, signal.SIGKILL)

    # -- run -----------------------------------------------------------------------------
    def run(self, config: RunConfig, run_dir: Path, *, reporter: RunReporter) -> RunResult:
        run_dir = Path(run_dir).expanduser().resolve()
        home = nemo_rl_home(config)
        self._cancelled.clear()
        reporter.set_status("pending", "compiling NeMo-RL config")
        exported = export_dataset(config, run_dir, tokenizer=_load_tokenizer(config, reporter))
        write_bridge_module(run_dir)
        yaml_path = write_config(compile_config(config, run_dir), run_dir)
        reporter.log(
            f"exported {exported.num_train} train samples"
            + (f" and {exported.num_eval} eval samples" if exported.num_eval is not None else "")
            + f" to {exported.paths.root}; config {yaml_path}"
        )
        cmd = build_command(config, home, yaml_path, run_dir)
        env = build_env(config, home, run_dir)
        reporter.log(f"launching (cwd={home}): {shlex.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            cwd=str(home),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with self._lock:
            self._proc = proc
        reporter.set_status("running", "NeMo-RL training started", child_pid=proc.pid, backend=self.name)

        with _signal_handlers(self._on_signal):
            self._stream(proc, reporter)
            returncode = proc.wait()

        checkpoints = list_checkpoints(run_dir)
        step = latest_step(run_dir)
        state: RunState
        error: str | None = None
        if self.cancelled:
            state, message = "cancelled", "cancelled by user"
        elif returncode == 0:
            state, message = "succeeded", f"NeMo-RL finished at step {step}"
        else:
            state = "failed"
            message = error = f"NeMo-RL exited with code {returncode} (see run.log)"
        reporter.set_status(state, message, step=step, returncode=returncode)
        with self._lock:
            self._proc = None
        return RunResult(
            state=state,
            final_step=step,
            run_dir=str(run_dir),
            checkpoints=checkpoints,
            final_model_path=final_model_path(checkpoints),
            error=error,
        )

    def _on_signal(self, signum: int, _frame: Any) -> None:
        self.cancel()

    def _stream(self, proc: subprocess.Popen[str], reporter: RunReporter) -> None:
        assert proc.stdout is not None
        last_refresh = time.monotonic()
        last_step = -1
        for line in proc.stdout:
            text = line.rstrip("\n")
            if text:
                reporter.log(text)
            now = time.monotonic()
            if now - last_refresh >= STEP_REFRESH_INTERVAL:
                last_refresh = now
                step = latest_step(reporter.run_dir)
                if step != last_step:
                    last_step = step
                    reporter.set_status("running", step=step)


def _signal_group(proc: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


class _signal_handlers:
    """Install ``handler`` for SIGTERM/SIGINT while active (main thread only) and restore after."""

    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.signal(sig, self.handler)

    def __exit__(self, *exc: Any) -> None:
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)
        self._previous.clear()


def _load_tokenizer(config: RunConfig, reporter: RunReporter) -> Any:
    """Tokenizer for the ``max_prompt_tokens`` filter at export; ``None`` (no filter) when unavailable."""
    from transformers import AutoTokenizer

    name = config.model.tokenizer_name_or_path or config.model.name_or_path
    try:
        return AutoTokenizer.from_pretrained(
            name, revision=config.model.revision, trust_remote_code=config.model.trust_remote_code
        )
    except Exception as e:  # gated/offline model: NeMo-RL still masks over-long prompts itself
        reporter.log(
            f"tokenizer {name!r} not loadable here ({type(e).__name__}: {e}); exporting without max_prompt_tokens filter"
        )
        return None


def make_backend(config: RunConfig, **_: Any) -> NemoRLBackend:
    return NemoRLBackend(config)


__all__ = [
    "ENTRY_MODULE",
    "NemoRLBackend",
    "build_command",
    "build_env",
    "final_model_path",
    "latest_step",
    "list_checkpoints",
    "make_backend",
    "monorl_src_root",
    "nemo_rl_home",
]
