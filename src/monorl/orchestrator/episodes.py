"""Parallel episode runner.

Each episode owns one environment instance. A stepped :class:`~monorl.core.interfaces.Environment`
follows the multi-turn loop from its docstring; an :class:`~monorl.core.interfaces.EpisodeDriver`
(an external agent harness behind the capture gateway) runs the whole episode itself. Episodes
run concurrently on a thread pool; their ``generate`` calls go through a
:class:`monorl.rollout.BatchingEngine`, so many concurrent single-prompt requests coalesce into a
few batched engine calls.

Text prompts are sent to the engine as token ids: turn 0 is rendered with :func:`chat_prompt_ids`
(with the environment's tool schemas, if any), every later turn is built by
:func:`monorl.rollout.render.continue_prompt_ids`. With ``rollout.turn_rendering: tokens`` the
previous prompt and the sampled completion stay in the context token for token and only the new
messages are rendered, so a turn's prompt is an exact extension of the tokens the policy was
sampled on and the batch builder trains the whole episode as one sequence. ``template`` re-renders
the conversation from its messages every turn instead.

Multimodal chats (image parts) are sent to the engine as chats on **every** turn: raw ids cannot
carry pixel tensors, so the engine re-runs the processor over the whole chat (image preprocessing
and placeholder expansion included) each turn and records the ids it conditioned on. This costs
one processor pass per turn and forgoes prefix caching. Tool schemas cannot be passed through
``RolloutEngine.generate``, so multimodal episodes against tool-exposing environments are
rejected.

Context budget (``max_context_tokens``): no prompt plus completion of a turn may exceed it. A
turn's generation is capped to the room that is left; an episode whose next prompt leaves no room
ends there with what it has (``Trajectory.info["context_exhausted"]``, ``done=False``); a first
prompt that does not fit is a failure of the episode. Multimodal prompts are tokenised by the
engine, so the budget is not applied to them.

Failures. An abort (:class:`EpisodeAborted`) and an engine failure
(:class:`~monorl.core.errors.GenerationFailed`) always end the batch. Anything else an episode
raises is a failure of its environment — a crashing ``reset``/``step``, an agent that dies, a
conversation the chat template rejects — and is retried ``retries`` times with a fresh
environment. What still fails is raised, or, when the caller passes a ``failures`` list, recorded
there and left out of the result.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from monorl.core.errors import DataError, EpisodeAborted, GenerationCancelled, GenerationFailed, MonorlError
from monorl.core.interfaces import Environment, EpisodeDriver, RolloutEngine
from monorl.core.types import (
    Completion,
    Message,
    PromptInput,
    Sample,
    SamplingParams,
    Trajectory,
    Turn,
    messages_have_media,
)
from monorl.rollout.render import chat_prompt_ids, continue_prompt_ids

log = logging.getLogger(__name__)

CANCEL_INTERVAL_S = 0.05
"""How often ``run_episodes`` re-issues ``engine.cancel()`` while an abort is being carried out."""


@dataclass(slots=True)
class EpisodeSpec:
    sample: Sample
    group_id: str
    engine: RolloutEngine
    """Engine (normally a ``BatchingEngine``) this episode generates with."""


@dataclass(slots=True)
class EpisodeFailure:
    """An episode whose environment failed on every attempt."""

    spec: EpisodeSpec
    error: Exception
    attempts: int


def run_episode(
    spec: EpisodeSpec,
    *,
    env_factory: Callable[[], Environment | EpisodeDriver],
    tokenizer: Any,
    params: SamplingParams,
    abort: threading.Event | None = None,
    turn_rendering: str = "tokens",
    max_context_tokens: int | None = None,
    retries: int = 0,
) -> Trajectory:
    """One episode; an environment failure is retried ``retries`` times, each time with a fresh
    environment, and raised when the last attempt fails too (module docstring)."""
    attempt = 0
    while True:
        try:
            return _run_once(
                spec,
                env_factory=env_factory,
                tokenizer=tokenizer,
                params=params,
                abort=abort,
                turn_rendering=turn_rendering,
                max_context_tokens=max_context_tokens,
            )
        except (EpisodeAborted, GenerationFailed):
            raise
        except Exception as e:
            if attempt >= retries or (abort is not None and abort.is_set()):
                raise
            attempt += 1
            log.warning(
                "episode %s failed (%s: %s); retry %d of %d",
                spec.group_id,
                type(e).__name__,
                e,
                attempt,
                retries,
            )


def _run_once(
    spec: EpisodeSpec,
    *,
    env_factory: Callable[[], Environment | EpisodeDriver],
    tokenizer: Any,
    params: SamplingParams,
    abort: threading.Event | None,
    turn_rendering: str,
    max_context_tokens: int | None,
) -> Trajectory:
    env = env_factory()
    try:
        if hasattr(env, "run_episode"):
            return env.run_episode(
                spec.sample,
                group_id=spec.group_id,
                engine=spec.engine,
                tokenizer=tokenizer,
                params=params,
                abort=abort,
                turn_rendering=turn_rendering,
                max_context_tokens=max_context_tokens,
            )
        return _step_episode(
            env,
            spec,
            tokenizer=tokenizer,
            params=params,
            abort=abort,
            turn_rendering=turn_rendering,
            max_context_tokens=max_context_tokens,
        )
    finally:
        env.close()


def _step_episode(
    env: Environment,
    spec: EpisodeSpec,
    *,
    tokenizer: Any,
    params: SamplingParams,
    abort: threading.Event | None,
    turn_rendering: str,
    max_context_tokens: int | None,
) -> Trajectory:
    messages: list[Message] = list(env.reset(spec.sample))
    # ``[]`` (remote/OpenEnv environments without tools) means no tool schemas, like ``None``.
    tools = getattr(env, "tools", None) or None
    max_turns = max(1, int(getattr(env, "max_turns", 1)))
    turns: list[Turn] = []
    done = False
    context_exhausted = False
    answered_at = 0  # index of the previous turn's assistant message in ``messages``
    for _ in range(max_turns):
        if abort is not None and abort.is_set():
            raise EpisodeAborted(f"episode {spec.group_id} aborted")
        prompt: PromptInput
        turn_params = params
        multimodal = messages_have_media(messages)
        if multimodal and tools is not None:
            raise MonorlError(
                f"episode {spec.group_id}: multimodal chats cannot be combined with environment tool schemas"
            )
        if multimodal:
            prompt = list(messages)
        else:
            if not turns:
                prompt = chat_prompt_ids(tokenizer, messages, tools)
            else:
                prompt = continue_prompt_ids(
                    tokenizer,
                    previous_messages=messages[:answered_at],
                    previous_prompt_ids=turns[-1].prompt_token_ids,
                    completion=turns[-1].completion,
                    new_messages=messages[answered_at + 1 :],
                    tools=tools,
                    mode=turn_rendering,
                ).ids
            if max_context_tokens is not None:
                room = max_context_tokens - len(prompt)
                if room < 1:
                    if not turns:
                        raise DataError(
                            f"episode {spec.group_id}: the prompt has {len(prompt)} tokens but the context budget "
                            f"is {max_context_tokens} (rollout.max_context_tokens, by default "
                            "trainer.max_seq_len); shorten the prompt (data.max_prompt_tokens) or raise the budget"
                        )
                    context_exhausted = True
                    break
                if room < params.max_new_tokens:
                    turn_params = dataclasses.replace(params, max_new_tokens=room)
        try:
            completion = spec.engine.generate([prompt], turn_params)[0][0]
        except GenerationCancelled as e:
            raise EpisodeAborted(f"episode {spec.group_id} aborted during generation") from e
        except Exception as e:
            raise GenerationFailed(f"episode {spec.group_id}: the rollout engine failed: {e}") from e
        if completion.prompt_token_ids is not None:
            prompt_ids = list(completion.prompt_token_ids)
        elif multimodal:
            prompt_ids = chat_prompt_ids(tokenizer, messages, None)
        else:
            prompt_ids = list(prompt)  # type: ignore[arg-type]
        answered_at = len(messages)
        messages.append(_assistant_message(completion))
        result = env.step(completion)
        turns.append(
            Turn(
                prompt_token_ids=prompt_ids,
                completion=completion,
                reward=float(result.reward),
                info=dict(result.info),
                message_index=answered_at,
            )
        )
        messages.extend(result.messages)
        done = bool(result.done)
        if completion.finish_reason == "length" and turn_params is not params:
            context_exhausted = True  # cut short by the budget, not by rollout.max_new_tokens
        if done or completion.finish_reason in ("length", "error"):
            break
    info: dict[str, Any] = {"finish_reason": turns[-1].completion.finish_reason, "num_turns": len(turns)}
    if context_exhausted:
        info["context_exhausted"] = True
    if tools is not None:
        info["tools"] = tools
    return Trajectory(
        sample=spec.sample,
        group_id=spec.group_id,
        messages=messages,
        turns=turns,
        done=done and not context_exhausted,
        info=info,
    )


def run_episodes(
    specs: Sequence[EpisodeSpec],
    *,
    env_factory: Callable[[], Environment | EpisodeDriver],
    tokenizer: Any,
    params: SamplingParams,
    max_parallel: int,
    abort: threading.Event | None = None,
    turn_rendering: str = "tokens",
    max_context_tokens: int | None = None,
    retries: int = 0,
    failures: list[EpisodeFailure] | None = None,
) -> list[Trajectory]:
    """Run every spec to completion, ``max_parallel`` episodes at a time; order is preserved.

    Without ``failures`` the first episode failure is re-raised after the other in-flight episodes
    finish. With a ``failures`` list, an episode whose environment failed on all ``retries + 1``
    attempts is appended to it and missing from the result; aborts and engine failures are still
    raised (module docstring).

    Once ``abort`` is set the episodes raise :class:`EpisodeAborted`: at their next turn boundary,
    or straight out of a ``generate`` call in flight when the spec's engine implements ``cancel()``
    (see :class:`~monorl.core.interfaces.RolloutEngine`). ``cancel()`` aborts every call in flight
    on that engine, so an engine must not be shared with work that is meant to outlive the abort.
    An ``env.step`` in flight finishes first. Either way this returns or raises only after every
    episode thread has ended and every environment is closed.
    """
    if not specs:
        return []
    workers = max(1, min(int(max_parallel), len(specs)))

    def attempt(spec: EpisodeSpec) -> Trajectory | None:
        try:
            return run_episode(
                spec,
                env_factory=env_factory,
                tokenizer=tokenizer,
                params=params,
                abort=abort,
                turn_rendering=turn_rendering,
                max_context_tokens=max_context_tokens,
                retries=retries,
            )
        except (EpisodeAborted, GenerationFailed):
            raise
        except Exception as e:
            if failures is None:
                raise
            failures.append(EpisodeFailure(spec=spec, error=e, attempts=retries + 1))
            log.warning("episode %s failed for good (%s: %s)", spec.group_id, type(e).__name__, e)
            return None

    with _cancel_generation_on_abort(specs, abort):
        if workers == 1:
            results = [attempt(spec) for spec in specs]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="monorl-episode") as pool:
                futures = [pool.submit(attempt, spec) for spec in specs]
                try:
                    results = [f.result() for f in futures]
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise
    return [trajectory for trajectory in results if trajectory is not None]


@contextlib.contextmanager
def _cancel_generation_on_abort(
    specs: Sequence[EpisodeSpec], abort: threading.Event | None
) -> Iterator[None]:
    """While the block runs, a watcher thread answers ``abort`` by calling ``cancel()`` on each
    distinct engine that has one — and keeps calling it every :data:`CANCEL_INTERVAL_S`, because an
    episode that was between two checks submits its next ``generate`` after the first cancel. The
    watcher has ended when the block exits, so no cancel can hit a later caller of the engines."""
    engines = list({id(s.engine): s.engine for s in specs if hasattr(s.engine, "cancel")}.values())
    if abort is None or not engines:
        yield
        return
    finished = threading.Event()

    def watch() -> None:
        while not finished.wait(CANCEL_INTERVAL_S):
            if not abort.is_set():
                continue
            for engine in list(engines):
                try:
                    engine.cancel()
                except Exception:  # noqa: BLE001 - the abort still takes effect at turn boundaries
                    log.exception("%s.cancel() failed; not cancelling it again", type(engine).__name__)
                    engines.remove(engine)

    watcher = threading.Thread(target=watch, name="monorl-episode-cancel", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        watcher.join()


def _assistant_message(completion: Completion) -> Message:
    return {"role": "assistant", "content": completion.text}


__all__ = [
    "EpisodeAborted",
    "EpisodeFailure",
    "EpisodeSpec",
    "chat_prompt_ids",
    "run_episode",
    "run_episodes",
]
