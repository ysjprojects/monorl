"""Export monorl samples to verl's RL parquet layout.

verl v0.9.1 reads rows through ``verl/utils/dataset/rl_dataset.py`` (``RLHFDataset``): the chat
lives under ``data.prompt_key`` (``prompt``), the reward manager reads ``data_source``,
``reward_model.ground_truth`` and ``extra_info`` (``verl/experimental/reward_loop/reward_manager/naive.py``),
and the agent loop picks the loop from the optional ``agent_name`` column
(``verl/experimental/agent_loop/agent_loop.py``). Rows are:

``{prompt: [messages], data_source: "monorl", reward_model: {style: "rule", ground_truth: str},
extra_info: {sample_id, index, messages_json, metadata_json, reference_encoding}, agent_name}``

``extra_info`` round-trips the original :class:`~monorl.core.types.Sample` so the reward bridge
(:mod:`monorl.adapters.verl.reward`) can rebuild a :class:`~monorl.core.types.Trajectory`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from monorl.config.schema import RunConfig
from monorl.core.types import Sample

DATA_SOURCE = "monorl"
PROMPT_KEY = "prompt"
DATA_DIR = "data"
TRAIN_FILE = "train.parquet"
VAL_FILE = "val.parquet"
SINGLE_TURN_AGENT = "single_turn_agent"
TOOL_AGENT = "tool_agent"

PARQUET_COLUMNS = ("prompt", "data_source", "reward_model", "extra_info", "agent_name")


def train_parquet_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / DATA_DIR / TRAIN_FILE


def val_parquet_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / DATA_DIR / VAL_FILE


def encode_reference(reference: Any) -> tuple[str, str]:
    """``(ground_truth, encoding)``: strings pass through, everything else is JSON-encoded."""
    if reference is None:
        return "", "none"
    if isinstance(reference, str):
        return reference, "text"
    return json.dumps(reference, default=str), "json"


def decode_reference(ground_truth: Any, encoding: str | None) -> Any:
    if encoding == "none":
        return None
    if encoding == "json" and isinstance(ground_truth, str):
        return json.loads(ground_truth)
    return ground_truth


def sample_to_row(sample: Sample, index: int, *, agent_name: str = SINGLE_TURN_AGENT) -> dict[str, Any]:
    ground_truth, encoding = encode_reference(sample.reference)
    prompt = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in sample.messages]
    return {
        "prompt": prompt,
        "data_source": DATA_SOURCE,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "sample_id": str(sample.id),
            "index": int(index),
            "messages_json": json.dumps(list(sample.messages), default=str),
            "metadata_json": json.dumps(dict(sample.metadata), default=str),
            "reference_encoding": encoding,
        },
        "agent_name": agent_name,
    }


def rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            pa.field("data_source", pa.string()),
            pa.field("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
            pa.field(
                "extra_info",
                pa.struct(
                    [
                        ("sample_id", pa.string()),
                        ("index", pa.int64()),
                        ("messages_json", pa.string()),
                        ("metadata_json", pa.string()),
                        ("reference_encoding", pa.string()),
                    ]
                ),
            ),
            pa.field("agent_name", pa.string()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def write_parquet(samples: Iterable[Sample], path: str | Path, *, agent_name: str = SINGLE_TURN_AGENT) -> int:
    """Write ``samples`` to ``path``; returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [sample_to_row(s, i, agent_name=agent_name) for i, s in enumerate(samples)]
    if not rows:
        raise ValueError(f"no samples to export to {path}")
    pq.write_table(rows_to_table(rows), path)
    return len(rows)


def agent_name_for(config: RunConfig) -> str:
    return TOOL_AGENT if config.env.type == "tool_calling" else SINGLE_TURN_AGENT


def export_datasets(config: RunConfig, run_dir: str | Path) -> dict[str, Path]:
    """Materialise train (and eval) parquet files from ``config.data`` via :func:`monorl.data.open_source`."""
    from monorl.data import open_source

    agent = agent_name_for(config)
    out = {"train": train_parquet_path(run_dir)}
    write_parquet(open_source(config.data, seed=config.seed), out["train"], agent_name=agent)
    if config.data.eval_uri:
        eval_cfg = config.data.model_copy(update={"uri": config.data.eval_uri, "split": config.data.eval_split})
        out["val"] = val_parquet_path(run_dir)
        write_parquet(open_source(eval_cfg, seed=config.seed), out["val"], agent_name=agent)
    return out
