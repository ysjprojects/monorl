"""``reward_model``: score the conversation with a sequence-classification reward model.

Params:
    model: HF id or path of an ``AutoModelForSequenceClassification`` checkpoint (required).
    device: ``cpu`` (default) / ``cuda`` / ``mps``.      dtype: ``float32`` (default) / ``bfloat16`` / ``float16``.
    batch_size (8), max_length (4096), label_index (0; which logit is the reward),
    normalize: ``none`` (default; raw logit) or ``sigmoid``.
    trust_remote_code (False).

The scored text is the full chat (prompt messages + the last assistant turn) rendered with the
model's chat template when it has one, else the messages joined as ``role: content`` lines.

Factory kwargs: ``model=``/``tokenizer=`` inject preloaded objects (tests build a tiny random model).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from monorl.core.types import Message, RewardResult, Trajectory
from monorl.rewards.base import check_trajectories, conversation, reward_name

_DTYPES = {"float32": "float32", "fp32": "float32", "bfloat16": "bfloat16", "bf16": "bfloat16", "float16": "float16", "fp16": "float16"}


def render_messages(tokenizer: Any, messages: list[Message]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


class RewardModelReward:
    def __init__(self, params: dict[str, Any], *, model: Any = None, tokenizer: Any = None) -> None:
        import torch

        self.name = reward_name(params, "reward_model")
        self.device = torch.device(str(params.get("device", "cpu")))
        self.batch_size = int(params.get("batch_size", 8))
        self.max_length = int(params.get("max_length", 4096))
        self.label_index = int(params.get("label_index", 0))
        self.normalize = str(params.get("normalize", "none"))
        if self.normalize not in ("none", "sigmoid"):
            raise ValueError("reward_model: normalize must be 'none' or 'sigmoid'")
        dtype_name = _DTYPES.get(str(params.get("dtype", "float32")))
        if dtype_name is None:
            raise ValueError(f"reward_model: unknown dtype {params.get('dtype')!r}")
        dtype = getattr(torch, dtype_name)

        if model is None or tokenizer is None:
            model_id = params.get("model")
            if not model_id:
                raise ValueError("reward_model needs params.model (HF id or path)")
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            trust = bool(params.get("trust_remote_code", False))
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
            if model is None:
                model = AutoModelForSequenceClassification.from_pretrained(model_id, dtype=dtype, trust_remote_code=trust)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()
        self._torch = torch

    def score_texts(self, texts: list[str]) -> list[float]:
        torch = self._torch
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(
                    texts[i : i + self.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.float()
                values = logits[:, self.label_index] if logits.ndim == 2 else logits.reshape(logits.shape[0], -1)[:, self.label_index]
                if self.normalize == "sigmoid":
                    values = torch.sigmoid(values)
                scores.extend(values.cpu().tolist())
        return scores

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        check_trajectories(trajectories)
        texts = [render_messages(self.tokenizer, conversation(t)) for t in trajectories]
        return [RewardResult(score=s, breakdown={self.name: s}) for s in self.score_texts(texts)]


def make_reward_model(params: dict[str, Any], *, model: Any = None, tokenizer: Any = None, **_: Any) -> RewardModelReward:
    return RewardModelReward(params, model=model, tokenizer=tokenizer)
