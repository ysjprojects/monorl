"""Real openenv-core environments used by the adapter tests (importable by dotted path)."""

from __future__ import annotations

from openenv.core.env_server import Action, Environment, Observation, State


class ReverseAction(Action):
    text: str


class ReverseObservation(Observation):
    output: str


class ReverseEnv(Environment[ReverseAction, ReverseObservation, State]):
    """Reward 1.0 when the agent replies with the reversed word; 3 attempts."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, word: str = "monorl", attempts: int = 3) -> None:
        super().__init__()
        self.word = word
        self.attempts = attempts
        self._state = State()
        self.closed = False

    def reset(self, seed: int | None = None, episode_id: str | None = None, word: str | None = None, **kwargs) -> ReverseObservation:
        if word:
            self.word = word
        self._state = State(episode_id=episode_id, step_count=0)
        return ReverseObservation(output=f"Reverse the word: {self.word}")

    def step(self, action: ReverseAction, timeout_s: float | None = None, **kwargs) -> ReverseObservation:
        self._state.step_count += 1
        correct = action.text.strip() == self.word[::-1]
        done = correct or self._state.step_count >= self.attempts
        return ReverseObservation(
            output="correct" if correct else f"wrong (attempt {self._state.step_count})",
            reward=1.0 if correct else 0.0,
            done=done,
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        self.closed = True


class GuessAction(Action):
    guess: int


class GuessObservation(Observation):
    hint: str
    remaining: int


class GuessEnv(Environment[GuessAction, GuessObservation, State]):
    """Number guessing with JSON actions ``{"guess": n}``."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, target: int = 7, attempts: int = 5) -> None:
        super().__init__()
        self.target = target
        self.attempts = attempts
        self._state = State()

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs) -> GuessObservation:
        self._state = State(episode_id=episode_id, step_count=0)
        return GuessObservation(hint="guess a number between 1 and 10", remaining=self.attempts)

    def step(self, action: GuessAction, timeout_s: float | None = None, **kwargs) -> GuessObservation:
        self._state.step_count += 1
        remaining = self.attempts - self._state.step_count
        if action.guess == self.target:
            return GuessObservation(hint="correct", remaining=remaining, reward=1.0, done=True)
        hint = "higher" if action.guess < self.target else "lower"
        return GuessObservation(hint=hint, remaining=remaining, reward=-0.1, done=remaining <= 0)

    @property
    def state(self) -> State:
        return self._state
