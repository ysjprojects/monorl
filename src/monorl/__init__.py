"""monorl: a unified RL harness.

One orchestrator, pluggable components (rollout engine, trainer, environment, reward,
data source, weight sync), and adapters that bridge upstream frameworks (vLLM, SGLang,
verl, TRL, NeMo-RL, Unsloth, OpenEnv, Hermes) through versioned monkeypatches.
"""

__version__ = "0.1.0"
