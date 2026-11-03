"""monorl :class:`~monorl.core.interfaces.DataSource` → ``datasets.Dataset`` for ``GRPOTrainer``.

Columns: ``prompt`` (chat messages, ``role``/``content`` only so Arrow gets one schema),
``sample_id``, ``reference`` (string form for TRL-native reward functions) and ``sample_json``
(the full :class:`~monorl.core.types.Sample`, used to rebuild exact samples on the reward side).
"""

from __future__ import annotations

import json
from typing import Any

from monorl.config.schema import DataConfig, RunConfig
from monorl.core.errors import DataError
from monorl.core.types import Sample

PROMPT_COLUMN = "prompt"
SAMPLE_COLUMN = "sample_json"


def sample_to_row(sample: Sample) -> dict[str, Any]:
    ref = sample.reference
    return {
        PROMPT_COLUMN: [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in sample.messages],
        "sample_id": str(sample.id),
        "reference": None if ref is None else (ref if isinstance(ref, str) else json.dumps(ref, default=str)),
        SAMPLE_COLUMN: json.dumps(
            {"id": sample.id, "messages": sample.messages, "reference": ref, "metadata": sample.metadata},
            default=str,
        ),
    }


def sample_from_row(row: dict[str, Any]) -> Sample:
    """Inverse of :func:`sample_to_row` (works on TRL's per-example input dicts)."""
    raw = row.get(SAMPLE_COLUMN)
    if raw is None:
        return Sample(id=str(row.get("sample_id", "")), messages=list(row[PROMPT_COLUMN]), reference=row.get("reference"))
    data = json.loads(raw)
    return Sample(
        id=str(data["id"]),
        messages=list(data["messages"]),
        reference=data.get("reference"),
        metadata=dict(data.get("metadata") or {}),
    )


def build_dataset(data_config: DataConfig, *, split: str | None = None, tokenizer: Any = None, seed: int = 0) -> Any:
    """Materialise ``data_config`` through :func:`monorl.data.open_source` into a ``datasets.Dataset``."""
    from datasets import Dataset

    from monorl.data import open_source

    source = open_source(data_config, split=split, tokenizer=tokenizer, seed=seed)
    rows = [sample_to_row(s) for s in source]
    if not rows:
        raise DataError(f"dataset {data_config.uri!r} yielded no samples")
    return Dataset.from_list(rows)


def build_train_and_eval(config: RunConfig, *, tokenizer: Any = None) -> tuple[Any, Any | None]:
    train = build_dataset(config.data, tokenizer=tokenizer, seed=config.seed)
    eval_ds = None
    if config.data.eval_uri:
        eval_cfg = config.data.model_copy(update={"uri": config.data.eval_uri, "split": config.data.eval_split})
        eval_ds = build_dataset(eval_cfg, tokenizer=tokenizer, seed=config.seed)
    return train, eval_ds
