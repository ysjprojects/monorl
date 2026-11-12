"""Deterministic fakes for every component the orchestrator consumes.

Importable from a subprocess (the cancel test), so nothing here depends on pytest. Call
:func:`install_fakes` to register the fake engine/trainer/env/reward in the registry and point the
orchestrator's lazy sibling-package lookups at the fake data/reward/advantage/batching helpers.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from monorl.config.schema import DataConfig, RunConfig
from monorl.core.errors import GenerationCancelled
from monorl.core.registry import registry
from monorl.core.types import (
    Completion,
    Message,
    RewardResult,
    Sample,
    SamplingParams,
    StepResult,
    Trajectory,
)

ENGINES: list[FakeEngine] = []
TRAINERS: list[FakeTrainer] = []
ADVANTAGE_CALLS: list[torch.Tensor] = []
BATCHING_ENGINES: list[FakeBatchingEngine] = []


# -- tokenizer ---------------------------------------------------------------------------------

_MARKERS = {"<|system|>": 1, "<|user|>": 2, "<|assistant|>": 3, "<|tool|>": 4, "<|end|>": 5, "<|tools|>": 6}
_MARKER_IDS = {v: k for k, v in _MARKERS.items()}
_MARKER_RE = re.compile("|".join(re.escape(m) for m in _MARKERS))
_CHAR_BASE = 10


class FakeTokenizer:
    """Char-level tokenizer with ChatML-like markers; token-exact and trivially invertible."""

    pad_token_id = 0
    eos_token_id = _MARKERS["<|end|>"]
    bos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        pos = 0
        for m in _MARKER_RE.finditer(text):
            ids.extend(_CHAR_BASE + ord(c) for c in text[pos : m.start()])
            ids.append(_MARKERS[m.group(0)])
            pos = m.end()
        ids.extend(_CHAR_BASE + ord(c) for c in text[pos:])
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in _MARKER_IDS:
                if not skip_special_tokens:
                    out.append(_MARKER_IDS[i])
            elif i >= _CHAR_BASE:
                out.append(chr(i - _CHAR_BASE))
        return "".join(out)

    def __call__(self, text: str, add_special_tokens: bool = False, **_: Any) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text)}

    def apply_chat_template(
        self,
        messages: Sequence[Message],
        tools: Any = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **_: Any,
    ) -> str | list[int]:
        text = ""
        if tools:
            text += "<|tools|>" + json.dumps(tools, sort_keys=True) + "<|end|>"
        for m in messages:
            text += f"<|{m['role']}|>{m.get('content', '')}<|end|>"
        if add_generation_prompt:
            text += "<|assistant|>"
        return self.encode(text) if tokenize else text


def last_user_text(prompt_text: str) -> str:
    users = re.findall(r"<\|user\|>(.*?)<\|end\|>", prompt_text, flags=re.S)
    return users[-1] if users else ""


# -- rollout engine ----------------------------------------------------------------------------


class FakeEngine:
    """Echo engine. Response policy is driven by ``rollout.extra``:

    * ``fake_mode``: ``"alternate"`` (default; even calls answer ``last_user.upper()``, odd calls
      ``"wrong"``), ``"always_right"``, ``"script"`` (consume ``fake_script`` in order).
    * ``fake_mismatch``: prepend a token that the text does not re-tokenise to (prefix mismatch).
    * ``fake_finish``: finish_reason for every completion.  ``fake_no_logprobs``: logprobs=None.
    * ``fake_latency``: seconds slept per ``generate`` call.
    """

    def __init__(self, config: RunConfig, replica: int = 0, **_: Any) -> None:
        self.config = config
        self.replica = replica
        self._tokenizer = FakeTokenizer()
        self._lock = threading.Lock()
        self._counter = 0
        self.calls: list[dict[str, Any]] = []
        self.weight_version: float = 0.0
        self.update_count = 0
        self.shutdown_count = 0
        extra = config.rollout.extra
        self.mode = extra.get("fake_mode", "alternate")
        self.script: list[str] = list(extra.get("fake_script", []))
        self.mismatch = bool(extra.get("fake_mismatch", False))
        self.finish = extra.get("fake_finish", "stop")
        self.no_logprobs = bool(extra.get("fake_no_logprobs", False))
        self.latency = float(extra.get("fake_latency", 0.0))
        ENGINES.append(self)

    @property
    def tokenizer(self) -> FakeTokenizer:
        return self._tokenizer

    def _respond(self, prompt_text: str) -> str:
        with self._lock:
            i = self._counter
            self._counter += 1
        if self.mode == "script":
            return self.script[i % len(self.script)]
        if self.mode == "always_right":
            return last_user_text(prompt_text).upper()
        return last_user_text(prompt_text).upper() if i % 2 == 0 else "wrong"

    def generate(self, prompts: Sequence[Any], params: SamplingParams) -> list[list[Completion]]:
        if self.latency:
            time.sleep(self.latency)
        with self._lock:
            self.calls.append({"num_prompts": len(prompts), "n": params.n, "temperature": params.temperature})
        out: list[list[Completion]] = []
        for prompt in prompts:
            if prompt and isinstance(prompt[0], dict):
                prompt_ids = self._tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=True
                )
            else:
                prompt_ids = [int(t) for t in prompt]
            prompt_text = self._tokenizer.decode(prompt_ids)
            comps = []
            for _ in range(params.n):
                text = self._respond(prompt_text)
                ids = self._tokenizer.encode(text) + [self._tokenizer.eos_token_id]
                if self.mismatch:
                    ids = [9999] + ids
                comps.append(
                    Completion(
                        text=text,
                        token_ids=ids,
                        logprobs=None if self.no_logprobs else [-(0.1 + 0.01 * j) for j in range(len(ids))],
                        finish_reason=self.finish,
                        prompt_token_ids=list(prompt_ids),
                    )
                )
            out.append(comps)
        return out

    def update_weights(self, weights: Any) -> None:
        for _name, tensor in weights:
            self.weight_version = float(tensor.reshape(-1)[0].item())
        self.update_count += 1

    def load_weights_from_path(self, path: Any) -> None:
        self.update_count += 1

    def sleep(self) -> None:
        pass

    def wake_up(self) -> None:
        pass

    def shutdown(self) -> None:
        self.shutdown_count += 1


class SlowCancellableEngine(FakeEngine):
    """:class:`FakeEngine` whose ``generate`` takes ``rollout.extra["fake_generate_seconds"]``
    (default 60; also the ``generate_seconds`` attribute) and implements the optional ``cancel()``
    capability: the calls in flight raise :class:`GenerationCancelled` at once. ``generating`` is
    set once a call is in flight; ``calls`` only lists calls that ran to completion."""

    def __init__(self, config: RunConfig, replica: int = 0, **kwargs: Any) -> None:
        # own state first: the base constructor publishes the instance in ``ENGINES``
        self.generate_seconds = float(config.rollout.extra.get("fake_generate_seconds", 60.0))
        self.generating = threading.Event()
        self.cancel_calls = 0
        self._cancel_cv = threading.Condition()
        self._cancel_epoch = 0
        super().__init__(config, replica, **kwargs)

    def generate(self, prompts: Sequence[Any], params: SamplingParams) -> list[list[Completion]]:
        with self._cancel_cv:
            epoch = self._cancel_epoch
            self.generating.set()
            if self._cancel_cv.wait_for(lambda: self._cancel_epoch != epoch, timeout=self.generate_seconds):
                raise GenerationCancelled("fake generation cancelled")
        return super().generate(prompts, params)

    def cancel(self) -> None:
        with self._cancel_cv:
            self.cancel_calls += 1
            self._cancel_epoch += 1
            self._cancel_cv.notify_all()


# -- trainer ----------------------------------------------------------------------------------


class FakeTrainer:
    """Records batches; ``trainer.extra``: ``fake_sleep`` seconds per train_step, ``fake_fail_at``
    step index that raises RuntimeError."""

    def __init__(self, config: RunConfig, **_: Any) -> None:
        self.config = config
        self._step = 0
        self.weight = torch.tensor([0.0])
        self.batches: list[Any] = []
        self.logprob_calls: list[bool] = []
        self.value_calls: list[Any] = []
        self.max_seq_len = config.trainer.max_seq_len
        self.learning_rate = config.trainer.learning_rate
        self.sleep = float(config.trainer.extra.get("fake_sleep", 0.0))
        self.fail_at = config.trainer.extra.get("fake_fail_at")
        self.loaded_from: Path | None = None
        TRAINERS.append(self)

    @property
    def step(self) -> int:
        return self._step

    def train_step(self, batch: Any) -> dict[str, float]:
        if self.sleep:
            time.sleep(self.sleep)
        self._step += 1
        self.weight = self.weight + 1.0
        self.batches.append(batch)
        if self.fail_at is not None and self._step == int(self.fail_at):
            raise RuntimeError(f"fake trainer failure at step {self._step}")
        n = max(1, batch.num_response_tokens)
        old = batch.old_logprobs if batch.old_logprobs is not None else torch.zeros_like(batch.advantages)
        return {
            "loss/policy": float(-(batch.advantages * old).sum().item() / n),
            "loss/kl": 0.0
            if batch.ref_logprobs is None
            else float(batch.ref_logprobs.abs().sum().item() / n),
            "loss/entropy": 0.0,
            "policy/clipfrac": 0.0,
            "policy/approx_kl": 0.0,
            "lr": self.learning_rate,
            "grad_norm": 1.0,
        }

    def compute_logprobs(self, batch: Any, *, reference: bool = False) -> torch.Tensor:
        self.logprob_calls.append(reference)
        return torch.full(batch.input_ids.shape, -0.5 if reference else -0.25, dtype=torch.float32)

    def compute_values(self, batch: Any) -> torch.Tensor:
        """A critic that values every state at ``trainer.extra['fake_value']`` (default 0.25)."""
        self.value_calls.append(batch)
        value = float(self.config.trainer.extra.get("fake_value", 0.25))
        return torch.full(batch.input_ids.shape, value, dtype=torch.float32)

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        yield "w", self.weight.clone()

    def save_checkpoint(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "state.json").write_text(
            json.dumps({"step": self._step, "weight": float(self.weight[0])})
        )

    def load_checkpoint(self, path: Path) -> None:
        state = json.loads((Path(path) / "state.json").read_text())
        self._step = int(state["step"])
        self.weight = torch.tensor([float(state["weight"])])
        self.loaded_from = Path(path)

    def save_pretrained(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text(json.dumps({"architectures": ["Fake"], "step": self._step}))


# -- environments -----------------------------------------------------------------------------


class FakeMultiTurnEnv:
    """``env.params``: ``turn_reward`` (per turn), ``done_after`` (turns), ``tools`` (schemas)."""

    def __init__(self, config: RunConfig, **_: Any) -> None:
        params = config.env.params
        self.max_turns = config.env.max_turns
        self.turn_reward = float(params.get("turn_reward", 0.0))
        self.done_after = int(params.get("done_after", config.env.max_turns))
        self.tools = params.get("tools")
        self._turns = 0
        self.closed = False

    def reset(self, sample: Sample) -> list[Message]:
        self._turns = 0
        return list(sample.messages)

    def step(self, action: Completion) -> StepResult:
        self._turns += 1
        done = self._turns >= self.done_after
        return StepResult(
            messages=[] if done else [{"role": "user", "content": f"again{self._turns}"}],
            reward=self.turn_reward,
            done=done,
            info={"turn": self._turns},
        )

    def close(self) -> None:
        self.closed = True


# -- rewards ----------------------------------------------------------------------------------


class ExactReward:
    name = "test_exact"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        out = []
        for t in trajectories:
            score = 1.0 if t.last_response_text.strip() == str(t.sample.reference).strip() else 0.0
            out.append(RewardResult(score=score, breakdown={self.name: score}))
        return out


class CompositeReward:
    name = "composite"

    def __init__(self, specs: Sequence[Any]) -> None:
        self.members = [
            (spec.type, float(spec.weight), registry.get("reward", spec.type)(dict(spec.params)))
            for spec in specs
        ]

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        results = [RewardResult(score=0.0) for _ in trajectories]
        for name, weight, fn in self.members:
            for res, member in zip(results, fn(trajectories), strict=True):
                res.score += weight * member.score
                res.breakdown[name] = member.score
        return results


def fake_build_reward(specs: Sequence[Any], *, config: RunConfig | None = None) -> CompositeReward:
    return CompositeReward(specs)


# -- algorithms -------------------------------------------------------------------------------


def fake_compute_advantages(
    rewards: torch.Tensor, group_ids: Sequence[str], *, algorithm: Any
) -> torch.Tensor:
    rewards = rewards.to(torch.float32)
    adv = rewards.clone()
    if algorithm.advantage_norm == "group":
        for gid in set(group_ids):
            idx = torch.tensor([i for i, g in enumerate(group_ids) if g == gid])
            r = rewards[idx]
            centred = r - r.mean()
            if algorithm.advantage_std_norm:
                centred = centred / (r.std(unbiased=False) + 1e-6)
            adv[idx] = centred
    elif algorithm.advantage_norm == "batch":
        adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
    ADVANTAGE_CALLS.append(adv.clone())
    return adv


# -- data ---------------------------------------------------------------------------------------


class FakeDataSource:
    name = "fake-local"

    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def schema(self) -> dict[str, str]:
        return {"prompt": "string", "answer": "string"}


def fake_open_source(
    data_config: DataConfig, *, split: str | None = None, tokenizer: Any = None, seed: int = 0
) -> FakeDataSource:
    prompt_field = data_config.prompt_field or "prompt"
    answer_field = data_config.answer_field or "answer"
    samples: list[Sample] = []
    with open(data_config.uri) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages: list[Message] = []
            if data_config.system_prompt:
                messages.append({"role": "system", "content": data_config.system_prompt})
            messages.append({"role": "user", "content": str(row[prompt_field])})
            samples.append(
                Sample(id=str(row.get("id", i)), messages=messages, reference=row.get(answer_field))
            )
    if data_config.limit:
        samples = samples[: data_config.limit]
    return FakeDataSource(samples)


# -- batching engine ----------------------------------------------------------------------------


class FakeBatchingEngine:
    """Pass-through stand-in for ``monorl.rollout.BatchingEngine`` that records construction."""

    def __init__(self, engine: Any, *, max_batch_size: int, max_wait_ms: float) -> None:
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.generate_calls = 0
        self._lock = threading.Lock()
        BATCHING_ENGINES.append(self)

    def generate(self, prompts: Sequence[Any], params: SamplingParams) -> list[list[Completion]]:
        with self._lock:
            self.generate_calls += 1
        return self.engine.generate(prompts, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)


# -- installation -------------------------------------------------------------------------------


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


DEFAULT_ROWS = [{"prompt": p, "answer": p.upper()} for p in ("abc", "def", "ghi", "jkl", "mno", "pqr")]

FAKE_MODEL_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 256,
    "max_position_embeddings": 512,
}


def fake_config_dict(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    data_file = write_jsonl(tmp_path / "data.jsonl", DEFAULT_ROWS)
    # a local model dir with a config.json keeps the placement planner's AutoConfig lookup offline
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(FAKE_MODEL_CONFIG))
    cfg: dict[str, Any] = {
        "run_name": "fake-run",
        "model": {"name_or_path": str(model_dir)},
        "data": {"uri": str(data_file), "prompt_field": "prompt", "answer_field": "answer"},
        "algorithm": {"name": "grpo"},
        "rollout": {
            "backend": "fake",
            "group_size": 2,
            "max_new_tokens": 16,
            "extra": {"max_parallel_episodes": 4},
        },
        "trainer": {
            "backend": "fake",
            "prompts_per_step": 2,
            "max_steps": 2,
            "save_every": 1,
            "max_seq_len": 64,
        },
        "rewards": [{"type": "test_exact"}],
        "env": {"type": "single_turn"},
        "logging": {"run_root": str(tmp_path / "runs")},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            for k2, v2 in value.items():
                if isinstance(v2, dict) and isinstance(cfg[key].get(k2), dict):
                    cfg[key][k2] = {**cfg[key][k2], **v2}
                else:
                    cfg[key][k2] = v2
        else:
            cfg[key] = value
    return cfg


def reset_recorders() -> None:
    ENGINES.clear()
    TRAINERS.clear()
    ADVANTAGE_CALLS.clear()
    BATCHING_ENGINES.clear()


def install_fakes(monkeypatch: Any = None) -> None:
    """Register fake components and route the orchestrator's lazy lookups to the fakes.

    With ``monkeypatch`` the routing is undone at test teardown; without it (subprocess) the
    patches are permanent for the process.
    """
    from monorl.orchestrator import loop

    registry.register("rollout_engine", "fake", FakeEngine, overwrite=True)
    registry.register("trainer", "fake", FakeTrainer, overwrite=True)
    registry.register("env", "fake_multi", FakeMultiTurnEnv, overwrite=True)
    registry.register("reward", "test_exact", lambda params, **kw: ExactReward(params), overwrite=True)
    routes = {
        "_resolve_open_source": lambda: fake_open_source,
        "_resolve_build_reward": lambda: fake_build_reward,
        "_resolve_compute_advantages": lambda: fake_compute_advantages,
        "_resolve_batching_engine": lambda: FakeBatchingEngine,
    }
    for name, value in routes.items():
        if monkeypatch is not None:
            monkeypatch.setattr(loop, name, value)
        else:
            setattr(loop, name, value)
    reset_recorders()
