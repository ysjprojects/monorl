"""Core data types shared by every component and adapter.

Torch-free on purpose: the web server and CLI import this module without paying for torch.
Tensor batches live in :mod:`monorl.core.batch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(TypedDict, total=False):
    id: str
    name: str
    arguments: dict[str, Any] | str


class Message(TypedDict, total=False):
    """OpenAI/HF chat-template compatible message."""

    role: Role
    content: str
    tool_calls: list[ToolCall]
    tool_call_id: str
    name: str


# A prompt is either a chat (chat template applied by the engine) or raw token ids
# (used for token-exact continuation in multi-turn rollouts).
PromptInput = list[Message] | list[int]

FinishReason = Literal["stop", "length", "tool_call", "error"]


@dataclass(slots=True)
class Sample:
    """One prompt-level unit yielded by a :class:`~monorl.core.interfaces.DataSource`."""

    id: str
    messages: list[Message]
    reference: Any = None
    """Ground truth used by verifiers (answer string, test cases, label...)."""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SamplingParams:
    n: int = 1
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    logprobs: bool = True
    """Return per-token logprobs under the generating policy (needed for importance ratios)."""


@dataclass(slots=True)
class Completion:
    text: str
    token_ids: list[int]
    logprobs: list[float] | None = None
    finish_reason: FinishReason = "stop"
    prompt_token_ids: list[int] | None = None
    """Exact prompt tokens the engine conditioned on (post chat template)."""


@dataclass(slots=True)
class Turn:
    """One policy action inside an episode. ``prompt_token_ids`` are the tokens the policy
    conditioned on for this turn; ``completion`` is what it produced."""

    prompt_token_ids: list[int]
    completion: Completion
    reward: float = 0.0
    """Per-turn reward from the environment (0 for single-turn tasks)."""
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Trajectory:
    """A full episode for one sample. Trajectories sharing ``group_id`` were generated from the
    same prompt and form a GRPO/RLOO group."""

    sample: Sample
    group_id: str
    messages: list[Message]
    turns: list[Turn]
    reward: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    done: bool = True
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def response_text(self) -> str:
        return "".join(t.completion.text for t in self.turns)

    @property
    def last_response_text(self) -> str:
        return self.turns[-1].completion.text if self.turns else ""

    @property
    def num_response_tokens(self) -> int:
        return sum(len(t.completion.token_ids) for t in self.turns)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.turns[0].prompt_token_ids) if self.turns else 0


@dataclass(slots=True)
class RewardResult:
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    """Returned by :meth:`Environment.step`."""

    messages: list[Message]
    """New messages to append to the conversation *after* the assistant turn (tool results,
    next user turn...). Empty when the episode is done."""
    reward: float = 0.0
    done: bool = True
    info: dict[str, Any] = field(default_factory=dict)


RunState = Literal["pending", "running", "succeeded", "failed", "cancelled"]


@dataclass(slots=True)
class RunResult:
    state: RunState
    final_step: int
    run_dir: str
    checkpoints: list[str] = field(default_factory=list)
    final_model_path: str | None = None
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
