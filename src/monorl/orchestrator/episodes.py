"""Parallel episode runner.

Each episode owns one :class:`~monorl.core.interfaces.Environment` instance and follows the
multi-turn loop from the ``Environment`` docstring. Episodes run concurrently on a thread pool;
their ``generate`` calls go through a :class:`monorl.rollout.BatchingEngine`, so many concurrent
single-prompt requests coalesce into a few batched engine calls.

Turn 0 hands the chat to the engine (which applies the chat template and records the resulting
ids in ``Completion.prompt_token_ids``). Later turns — and every turn when the environment exposes
tool schemas — are chat-templated here with :func:`chat_prompt_ids` and sent as raw token ids so
the engine conditions on exactly the ids the batch builder will later train on.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from monorl.core.errors import MonorlError
from monorl.core.interfaces import Environment, RolloutEngine
from monorl.core.types import Completion, Message, PromptInput, Sample, SamplingParams, Trajectory, Turn


class EpisodeAborted(MonorlError):
    """Raised inside an episode when the abort event is set (async producer shutdown)."""


def chat_prompt_ids(tokenizer: Any, messages: Sequence[Message], tools: Any = None) -> list[int]:
    """Token ids of ``messages`` rendered with the chat template and a generation prompt.

    Rendered as text first and tokenised without special tokens, so the template alone decides
    about BOS/EOS; this is the same recipe rollout engines use for chat prompts, which is what
    makes turn ``k`` prompts a token-exact extension of turn ``k-1``.
    """
    text = tokenizer.apply_chat_template(
        list(messages), tools=tools, add_generation_prompt=True, tokenize=False
    )
    return [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]


@dataclass(slots=True)
class EpisodeSpec:
    sample: Sample
    group_id: str
    engine: RolloutEngine
    """Engine (normally a ``BatchingEngine``) this episode generates with."""


def run_episode(
    spec: EpisodeSpec,
    *,
    env_factory: Callable[[], Environment],
    tokenizer: Any,
    params: SamplingParams,
    abort: threading.Event | None = None,
) -> Trajectory:
    env = env_factory()
    try:
        messages: list[Message] = list(env.reset(spec.sample))
        tools = getattr(env, "tools", None)
        max_turns = max(1, int(getattr(env, "max_turns", 1)))
        turns: list[Turn] = []
        done = False
        for turn_index in range(max_turns):
            if abort is not None and abort.is_set():
                raise EpisodeAborted(f"episode {spec.group_id} aborted")
            prompt: PromptInput
            if turn_index == 0 and tools is None:
                prompt = list(messages)
            else:
                prompt = chat_prompt_ids(tokenizer, messages, tools)
            completion = spec.engine.generate([prompt], params)[0][0]
            if completion.prompt_token_ids is not None:
                prompt_ids = list(completion.prompt_token_ids)
            elif turn_index == 0 and tools is None:
                prompt_ids = chat_prompt_ids(tokenizer, messages, None)
            else:
                prompt_ids = list(prompt)  # type: ignore[arg-type]
            messages.append(_assistant_message(completion))
            result = env.step(completion)
            turns.append(
                Turn(
                    prompt_token_ids=prompt_ids,
                    completion=completion,
                    reward=float(result.reward),
                    info=dict(result.info),
                )
            )
            messages.extend(result.messages)
            done = bool(result.done)
            if done or completion.finish_reason in ("length", "error"):
                break
    finally:
        env.close()
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
    env_factory: Callable[[], Environment],
    tokenizer: Any,
    params: SamplingParams,
    max_parallel: int,
    abort: threading.Event | None = None,
) -> list[Trajectory]:
    """Run every spec to completion, ``max_parallel`` episodes at a time; order is preserved.

    The first episode failure is re-raised after the other in-flight episodes finish.
    """
    if not specs:
        return []
    workers = max(1, min(int(max_parallel), len(specs)))
    kwargs = {"env_factory": env_factory, "tokenizer": tokenizer, "params": params, "abort": abort}
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


def _assistant_message(completion: Completion) -> Message:
    return {"role": "assistant", "content": completion.text}
