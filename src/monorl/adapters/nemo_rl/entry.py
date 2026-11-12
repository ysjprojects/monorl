"""``python -m monorl.adapters.nemo_rl.entry --config <yaml> [--run-dir D] [--nemo-rl-home H] [--script S] [overrides…]``

Runs inside the NeMo-RL checkout's environment (``uv run`` / its python). It applies the monorl
patches, registers the monorl environment bridge with NeMo-RL and then executes
``<NEMO_RL_HOME>/examples/run_grpo.py`` (or ``--script examples/run_sft.py`` for SFT) as
``__main__`` with the remaining arguments (``--config`` plus hydra-style overrides), exactly like
``uv run python examples/run_grpo.py``.

NeMo-RL v0.7.0 (``81aa43dda4765b0429cf31dab44441e4e4383911``) registries touched:
``nemo_rl.environments.utils.ENV_REGISTRY`` / ``register_env`` and
``nemo_rl.distributed.ray_actor_environment_registry.ACTOR_ENVIRONMENT_REGISTRY``
(``PY_EXECUTABLES.SYSTEM`` from ``nemo_rl.distributed.virtual_cluster``).
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from monorl.adapters.nemo_rl.compile import ACTOR_FQN, ENV_NAME
from monorl.adapters.nemo_rl.patches import FRAMEWORK, PATCHES, RUN_DIR_ENV
from monorl.core.errors import BackendUnavailable, PatchError
from monorl.patching import apply, register_patchset

HOME_ENV = "NEMO_RL_HOME"
RUN_SCRIPT = Path("examples") / "run_grpo.py"
SFT_SCRIPT = Path("examples") / "run_sft.py"


def resolve_home(explicit: str | os.PathLike[str] | None = None) -> Path:
    """The NeMo-RL checkout (``explicit`` → ``$NEMO_RL_HOME``); must contain ``examples/run_grpo.py``."""
    raw = explicit or os.environ.get(HOME_ENV)
    if not raw:
        raise BackendUnavailable(
            "NeMo-RL is git-only: clone https://github.com/NVIDIA-NeMo/RL and point NEMO_RL_HOME "
            "(or trainer.extra['nemo_rl_home']) at the checkout"
        )
    home = Path(raw).expanduser().resolve()
    script = home / RUN_SCRIPT
    if not script.is_file():
        raise BackendUnavailable(f"NEMO_RL_HOME={home} does not contain {RUN_SCRIPT}; not a NeMo-RL checkout")
    return home


def install_bridges() -> None:
    """Apply the ``nemo_rl`` patchset and register the monorl environment actor with NeMo-RL."""
    register_patchset(PATCHES)
    report = apply(FRAMEWORK)
    if report.installed_version is None:
        raise BackendUnavailable("nemo_rl is not importable in this interpreter; run the entry inside the NeMo-RL environment")
    if report.failed:
        raise PatchError("; ".join(f"{pid}: {reason}" for pid, reason in report.failed))
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import ENV_REGISTRY, register_env

    if ENV_NAME not in ENV_REGISTRY:
        register_env(ENV_NAME, ACTOR_FQN)
    ACTOR_ENVIRONMENT_REGISTRY.setdefault(ACTOR_FQN, PY_EXECUTABLES.SYSTEM)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="python -m monorl.adapters.nemo_rl.entry", description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, help="Compiled NeMo-RL yaml (monorl.adapters.nemo_rl.compile)")
    parser.add_argument("--run-dir", default=None, help=f"monorl run directory (default: ${RUN_DIR_ENV})")
    parser.add_argument("--nemo-rl-home", default=None, help=f"NeMo-RL checkout (default: ${HOME_ENV})")
    parser.add_argument(
        "--script", default=str(RUN_SCRIPT), help=f"Example script relative to the checkout (default: {RUN_SCRIPT})"
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    args, overrides = parse_args(argv)
    home = resolve_home(args.nemo_rl_home)
    run_dir = args.run_dir or os.environ.get(RUN_DIR_ENV)
    if run_dir:
        os.environ[RUN_DIR_ENV] = str(Path(run_dir).expanduser().resolve())
    os.environ.setdefault(HOME_ENV, str(home))
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))
    install_bridges()
    script = home / args.script
    if not script.is_file():
        raise BackendUnavailable(f"NEMO_RL_HOME={home} does not contain {args.script}")
    sys.argv = [str(script), "--config", str(Path(args.config).expanduser().resolve()), *overrides]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
