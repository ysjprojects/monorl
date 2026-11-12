"""Fakes shared by the nemo_rl adapter tests (importable as ``nemo_rl_fakes``)."""

from __future__ import annotations

import textwrap
from typing import Any

FAKE_VERSION = "0.7.0"


class FakeLogger:
    """Stand-in for ``nemo_rl.utils.logger.Logger`` (same ``log_metrics`` signature as v0.7.0)."""

    calls: list[tuple[dict[str, Any], int, str]] = []

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    def log_metrics(self, metrics, step, prefix="", step_metric=None, step_finished=False):
        type(self).calls.append((dict(metrics), step, prefix))



FAKE_HOME_FILES: dict[str, str] = {
    "pyproject.toml": '[project]\nname = "nemo-rl"\nversion = "0.7.0"\n',
    "nemo_rl/__init__.py": f'__version__ = "{FAKE_VERSION}"\n',
    "nemo_rl/utils/__init__.py": "",
    "nemo_rl/utils/logger.py": textwrap.dedent(
        """
        class Logger:
            def __init__(self, cfg):
                self.cfg = cfg

            def log_metrics(self, metrics, step, prefix="", step_metric=None, step_finished=False):
                print(f"LOG {prefix} step={step} {sorted(metrics)}", flush=True)
        """
    ),
    "nemo_rl/environments/__init__.py": "",
    "nemo_rl/environments/utils.py": textwrap.dedent(
        """
        ENV_REGISTRY = {}


        def register_env(name, fqn):
            if name in ENV_REGISTRY:
                raise ValueError(name)
            ENV_REGISTRY[name] = {"actor_class_fqn": fqn}
        """
    ),
    "nemo_rl/distributed/__init__.py": "",
    "nemo_rl/distributed/ray_actor_environment_registry.py": "ACTOR_ENVIRONMENT_REGISTRY = {}\n",
    "nemo_rl/distributed/virtual_cluster.py": "class PY_EXECUTABLES:\n    SYSTEM = 'python'\n",
    "examples/run_grpo.py": textwrap.dedent(
        """
        import argparse
        import json
        import os
        import sys
        import time

        import yaml

        from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
        from nemo_rl.environments.utils import ENV_REGISTRY
        from nemo_rl.utils.logger import Logger


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", required=True)
            args, overrides = parser.parse_known_args()
            config = yaml.safe_load(open(args.config))
            print(f"Loaded configuration from: {args.config}")
            print(f"Overrides: {overrides}")
            run_dir = os.environ["MONORL_RUN_DIR"]
            with open(os.path.join(run_dir, "child_env.json"), "w") as f:
                json.dump(
                    {
                        "cwd": os.getcwd(),
                        "argv": sys.argv,
                        "env_registry": ENV_REGISTRY,
                        "actor_registry": ACTOR_ENVIRONMENT_REGISTRY,
                        "ray_address": os.environ.get("RAY_ADDRESS"),
                        "patched": getattr(Logger.log_metrics, "__monorl_patch__", None),
                        "dataset_name": config["data"]["train"]["dataset_name"],
                    },
                    f,
                )
            if os.environ.get("FAKE_NEMO_RL_SLEEP"):
                print("sleeping", flush=True)
                while True:
                    time.sleep(0.1)
            logger = Logger(config["logger"])
            for step in (1, 2):
                logger.log_metrics({"loss": 0.5 / step, "reward": 0.25 * step, "lr": 1e-6, "kl_penalty": 0.01}, step, prefix="train")
                logger.log_metrics({"total_step_time": 3.0, "policy_training": 1.0}, step, prefix="timing/train")
            ckpt = os.path.join(config["checkpointing"]["checkpoint_dir"], "step_2", "policy", "weights")
            os.makedirs(ckpt, exist_ok=True)
            if os.environ.get("FAKE_NEMO_RL_FAIL"):
                print("boom", flush=True)
                sys.exit(3)
            print("training done", flush=True)


        if __name__ == "__main__":
            main()
        """
    ),
    "examples/run_sft.py": textwrap.dedent(
        """
        import argparse
        import json
        import os
        import sys

        import yaml

        from nemo_rl.utils.logger import Logger


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", required=True)
            args, overrides = parser.parse_known_args()
            config = yaml.safe_load(open(args.config))
            run_dir = os.environ["MONORL_RUN_DIR"]
            rows = [json.loads(line) for line in open(config["data"]["train"]["data_path"])]
            with open(os.path.join(run_dir, "child_env.json"), "w") as f:
                json.dump(
                    {
                        "argv": sys.argv,
                        "algorithm": os.environ.get("MONORL_ALGORITHM"),
                        "sections": sorted(config),
                        "processor": config["data"]["default"]["processor"],
                        "dataset_name": config["data"]["train"]["dataset_name"],
                        "rows": rows,
                    },
                    f,
                )
            logger = Logger(config["logger"])
            for step in (1, 2):
                logger.log_metrics({"loss": 1.0 / step, "lr": 1e-6, "global_valid_toks": 40}, step, prefix="train")
            logger.log_metrics({"val_loss": 0.75}, 2, prefix="validation")
            ckpt = os.path.join(config["checkpointing"]["checkpoint_dir"], "step_2", "policy", "weights")
            os.makedirs(ckpt, exist_ok=True)
            print("sft done", flush=True)


        if __name__ == "__main__":
            main()
        """
    ),
}
