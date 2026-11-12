"""Parallel episode runner.

Each episode owns one environment instance. A stepped :class:`~monorl.core.interfaces.Environment`
follows the multi-turn loop from its docstring; an :class:`~monorl.core.interfaces.EpisodeDriver`
(an external agent harness behind the capture gateway) runs the whole episode itself. Episodes
run concurrently on a thread pool; their ``generate`` calls go through a
:class:`monorl.rollout.BatchingEngine`, so many concurrent single-prompt requests coalesce into a
few batched engine calls.

Turn 0 hands the chat to the engine (which applies the chat template and records the resulting
ids in ``Completion.prompt_token_ids``), or renders it here with :func:`chat_prompt_ids` when the
environment exposes tool schemas. Every later turn is sent as raw token ids built by
:func:`monorl.rollout.render.continue_prompt_ids`: with ``rollout.turn_rendering: tokens`` the
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
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from monorl.core.errors import GenerationCancelled, MonorlError
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


class EpisodeAborted(MonorlError):
    """Raised inside an episode when the abort event is set (async producer shutdown, cancelled
    sampling stage) or its ``generate`` call was cancelled."""


@dataclass(slots=True)
class EpisodeSpec:
    sample: Sample
    group_id: str
    engine: RolloutEngine
    """Engine (normally a ``BatchingEngine``) this episode generates with."""


def run_episode(
    spec: EpisodeSpec,
    *,
    env_factory: Callable[[], Environment | EpisodeDriver],
    tokenizer: Any,
    params: SamplingParams,
    abort: threading.Event | None = None,
    turn_rendering: str = "tokens",
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
            )
        return _step_episode(
            env, spec, tokenizer=tokenizer, params=params, abort=abort, turn_rendering=turn_rendering
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
) -> Trajectory:
    messages: list[Message] = list(env.reset(spec.sample))
    # ``[]`` (remote/OpenEnv environments without tools) means no tool schemas, like ``None``.
    tools = getattr(env, "tools", None) or None
    max_turns = max(1, int(getattr(env, "max_turns", 1)))
    turns: list[Turn] = []
    done = False
    answered_at = 0  # index of the previous turn's assistant message in ``messages``
    for _ in range(max_turns):
        if abort is not None and abort.is_set():
            raise EpisodeAborted(f"episode {spec.group_id} aborted")
        prompt: PromptInput
        multimodal = messages_have_media(messages)
        if multimodal and tools is not None:
            raise MonorlError(
                f"episode {spec.group_id}: multimodal chats cannot be combined with environment tool schemas"
            )
        if multimodal or (not turns and tools is None):
            prompt = list(messages)
        elif not turns:
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
        try:
            completion = spec.engine.generate([prompt], params)[0][0]
        except GenerationCancelled as e:
            raise EpisodeAborted(f"episode {spec.group_id} aborted during generation") from e
        if completion.prompt_token_ids is not None:
            prompt_ids = list(completion.prompt_token_ids)
        elif isinstance(prompt[0], dict):
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
        if done or completion.finish_reason in ("length", "error"):
            break
    return Trajectory(
        sample=spec.sample,
        group_id=spec.group_id,
        messages=messages,
        turns=turns,
        done=done,
        info={"finish_reason": turns[-1].completion.finish_reason, "num_turns": len(turns)},
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
) -> list[Trajectory]:
    """Run every spec to completion, ``max_parallel`` episodes at a time; order is preserved.

    The first episode failure is re-raised after the other in-flight episodes finish.

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
    kwargs = {
        "env_factory": env_factory,
        "tokenizer": tokenizer,
        "params": params,
        "abort": abort,
        "turn_rendering": turn_rendering,
    }
    with _cancel_generation_on_abort(specs, abort):
        if workers == 1:
            return [run_episode(spec, **kwargs) for spec in specs]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="monorl-episode") as pool:
            futures = [pool.submit(run_episode, spec, **kwargs) for spec in specs]
            try:
                return [f.result() for f in futures]
            except BaseException:
                for f in futures:
                    f.cancel()
                raise


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
