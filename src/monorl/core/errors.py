class MonorlError(Exception):
    """Base class for all monorl errors."""


class RegistryError(MonorlError):
    pass


class ConfigError(MonorlError):
    pass


class PatchError(MonorlError):
    """A monkeypatch could not be applied/verified against the installed upstream version."""


class BackendUnavailable(MonorlError):
    """The requested backend's upstream framework is not installed or not usable here."""


class DataError(MonorlError):
    pass


class EnvironmentError_(MonorlError):
    """Environment-side failure during an episode (named with a trailing underscore to avoid
    shadowing the builtin)."""


class WeightSyncError(MonorlError):
    pass


class GenerationCancelled(MonorlError):
    """A ``RolloutEngine.generate`` call was aborted through the engine's ``cancel()`` before it
    finished."""


class GenerationFailed(MonorlError):
    """The rollout engine failed while generating for an episode. Unlike an environment's failure
    this is never tolerated (``env.max_failed_frac``): the engine serves every episode."""


class EpisodeAborted(MonorlError):
    """Raised inside an episode when the abort event is set (async producer shutdown, cancelled
    sampling stage) or its ``generate`` call was cancelled."""
