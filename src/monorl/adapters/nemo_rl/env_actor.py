"""Ray actor wrapping :class:`~monorl.adapters.nemo_rl.env_bridge.MonorlEnvironmentBridge`.

Only importable inside a NeMo-RL environment (needs ``ray`` and ``nemo_rl``). NeMo-RL v0.7.0
(``81aa43dda4765b0429cf31dab44441e4e4383911``) creates environments through
``nemo_rl/environments/utils.py::create_env`` from ``ENV_REGISTRY[env_name]["actor_class_fqn"]`` and
looks the class up in ``nemo_rl/distributed/ray_actor_environment_registry.py`` for its python
executable; :func:`monorl.adapters.nemo_rl.entry.install_bridges` registers both for :data:`ACTOR_FQN`.
The decorator arguments mirror ``nemo_rl/environments/math_environment.py::MathEnvironment``.
"""

from __future__ import annotations

import ray
from nemo_rl.environments.interfaces import EnvironmentInterface

from monorl.adapters.nemo_rl.env_bridge import ACTOR_FQN, MonorlEnvironmentBridge


@ray.remote(max_restarts=-1, max_task_retries=-1, max_concurrency=1000)
class MonorlEnvironment(MonorlEnvironmentBridge, EnvironmentInterface):
    """``EnvironmentInterface`` actor: one monorl environment per episode, monorl rewards."""


__all__ = ["ACTOR_FQN", "MonorlEnvironment"]
