"""``python -m monorl.adapters.verl.entry <hydra overrides>`` — verl's ``main_ppo`` with monorl patches.

Mirrors ``verl/trainer/main_ppo.py::main`` (v0.9.1): same Hydra config (``pkg://verl.trainer.config``,
``ppo_trainer``), same validation, same ``run_ppo``; the only difference is the ``TaskRunner``
class. verl runs the trainer inside a Ray actor (``TaskRunnerV1``), so patches applied in this
driver process would not reach the ``Tracking`` logger; :class:`MonorlTaskRunner` subclasses
verl's runner and applies :data:`monorl.adapters.verl.patches.PATCHES` inside the actor before
delegating to ``run``. The driver applies them too, for ``get_ppo_ray_runtime_env``.
"""

from __future__ import annotations

import sys

import hydra
import ray
from verl.trainer import main_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from monorl.adapters.verl.patches import PATCHES
from monorl.patching import apply, register_patchset

CONFIG_PATH = "pkg://verl.trainer.config"
CONFIG_NAME = "ppo_trainer"


def apply_patches() -> None:
    register_patchset(PATCHES)
    report = apply(PATCHES.framework)
    if report.failed:
        raise RuntimeError(f"monorl verl patches failed: {report.failed}")


_BaseRunner = main_ppo.TaskRunnerV1.__ray_actor_class__


@ray.remote
class MonorlTaskRunner(_BaseRunner):  # type: ignore[misc,valid-type]
    """verl's ``TaskRunnerV1`` with monorl patches applied in the actor process."""

    def run(self, config):  # type: ignore[override]
        apply_patches()
        return super().run(config)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(config) -> None:
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    main_ppo.run_ppo(config, task_runner_class=MonorlTaskRunner)


if __name__ == "__main__":
    apply_patches()
    sys.exit(main())
