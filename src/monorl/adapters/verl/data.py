"""Export monorl samples to verl's parquet layouts (RL prompts and SFT conversations).

verl v0.9.1 reads RL rows through ``verl/utils/dataset/rl_dataset.py`` (``RLHFDataset``): the chat
lives under ``data.prompt_key`` (``prompt``), the reward manager reads ``data_source``,
``reward_model.ground_truth`` and ``extra_info`` (``verl/experimental/reward_loop/reward_manager/naive.py``),
and the agent loop picks the loop from the optional ``agent_name`` column
(``verl/experimental/agent_loop/agent_loop.py``). Rows are:

``{prompt: [messages], data_source: "monorl", reward_model: {style: "rule", ground_truth: str},
extra_info: {sample_id, index, messages_json, metadata_json, reference_encoding}, agent_name}``

``extra_info`` round-trips the original :class:`~monorl.core.types.Sample` so the reward bridge
(:mod:`monorl.adapters.verl.reward`) can rebuild a :class:`~monorl.core.types.Trajectory`.

Multimodal samples follow ``examples/data_preprocess/geo3k.py``: message ``content`` is a string
with ``<image>`` placeholders (``RLHFDataset._build_messages`` turns them into image parts) and an
``images`` column holds ``{"bytes": png, "path": null}`` structs (the ``datasets.Image`` parquet
encoding ``process_image`` decodes). SFT rows (``verl/utils/dataset/multiturn_sft_dataset.py``
``MultiTurnSFTDataset``) are ``{messages: [prompt..., assistant response], images?}``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from monorl.config.schema import RunConfig
from monorl.core.types import Message, Sample, message_text, messages_have_media

DATA_SOURCE = "monorl"
PROMPT_KEY = "prompt"
DATA_DIR = "data"
TRAIN_FILE = "train.parquet"
VAL_FILE = "val.parquet"
SINGLE_TURN_AGENT = "single_turn_agent"
TOOL_AGENT = "tool_agent"

PARQUET_COLUMNS = ("prompt", "data_source", "reward_model", "extra_info", "agent_name")
IMAGES_KEY = "images"
MESSAGES_KEY = "messages"
IMAGE_PLACEHOLDER = "<image>"
SFT_TRAIN_FILE = "sft_train.parquet"
SFT_VAL_FILE = "sft_val.parquet"


def sft_train_parquet_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / DATA_DIR / SFT_TRAIN_FILE


def sft_val_parquet_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / DATA_DIR / SFT_VAL_FILE


def placeholder_messages(messages: list[Message]) -> list[dict[str, str]]:
    """``role``/``content`` strings; image parts become ``<image>`` placeholders in order."""
    out: list[dict[str, str]] = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            text = content
        else:
            chunks: list[str] = []
            for part in content:
                kind = part.get("type")
                if kind == "text":
                    chunks.append(str(part.get("text", "")))
                elif kind == "image":
                    chunks.append(IMAGE_PLACEHOLDER)
                else:
                    raise ValueError(f"trainer.backend='verl' supports text and image parts, got {kind!r}")
            text = "".join(chunks)
        out.append({"role": str(m.get("role", "user")), "content": text})
    return out


def image_structs(messages: list[Message]) -> list[dict[str, Any]]:
    """``[{"bytes": png, "path": None}, ...]`` for every image part in message order."""
    from monorl.rollout.media import chat_images

    out: list[dict[str, Any]] = []
    for img in chat_images(messages):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append({"bytes": buf.getvalue(), "path": None})
    return out


def has_media(samples: list[Sample]) -> bool:
    return any(messages_have_media(s.messages) for s in samples)


def parquet_has_images(path: str | Path) -> bool:
    """``True`` when the exported parquet carries the ``images`` column."""
    return IMAGES_KEY in pq.read_schema(path).names


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


def sample_to_row(
    sample: Sample, index: int, *, agent_name: str = SINGLE_TURN_AGENT, multimodal: bool = False
) -> dict[str, Any]:
    ground_truth, encoding = encode_reference(sample.reference)
    if multimodal:
        prompt = placeholder_messages(sample.messages)
    else:
        prompt = [{"role": str(m.get("role", "user")), "content": message_text(m)} for m in sample.messages]
    row: dict[str, Any] = {
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
    if multimodal:
        row[IMAGES_KEY] = image_structs(sample.messages)
    return row


def sft_sample_to_row(
    sample: Sample, *, response_field: str | None, multimodal: bool = False
) -> dict[str, Any]:
    """``{messages: prompt + assistant response, images?}`` for ``MultiTurnSFTDataset``; the
    response comes from :func:`monorl.orchestrator.sft.sft_messages`."""
    from monorl.orchestrator.sft import sft_messages

    messages = sft_messages(sample, response_field=response_field)
    row: dict[str, Any] = {MESSAGES_KEY: placeholder_messages(messages)}
    if multimodal:
        row[IMAGES_KEY] = image_structs(messages)
    return row


_MESSAGES_TYPE = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
_IMAGES_TYPE = pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))


def rows_to_table(rows: list[dict[str, Any]], *, multimodal: bool = False) -> pa.Table:
    fields = [
        pa.field("prompt", _MESSAGES_TYPE),
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
    if multimodal:
        fields.append(pa.field(IMAGES_KEY, _IMAGES_TYPE))
    return pa.Table.from_pylist(rows, schema=pa.schema(fields))


def sft_rows_to_table(rows: list[dict[str, Any]], *, multimodal: bool = False) -> pa.Table:
    fields = [pa.field(MESSAGES_KEY, _MESSAGES_TYPE)]
    if multimodal:
        fields.append(pa.field(IMAGES_KEY, _IMAGES_TYPE))
    return pa.Table.from_pylist(rows, schema=pa.schema(fields))


def write_parquet(
    samples: Iterable[Sample],
    path: str | Path,
    *,
    agent_name: str = SINGLE_TURN_AGENT,
    multimodal: bool | None = None,
) -> int:
    """Write RL rows for ``samples`` to ``path``; returns the number of rows written. ``multimodal``
    defaults to whether any sample carries image parts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = list(samples)
    if not samples:
        raise ValueError(f"no samples to export to {path}")
    multimodal = has_media(samples) if multimodal is None else multimodal
    rows = [sample_to_row(s, i, agent_name=agent_name, multimodal=multimodal) for i, s in enumerate(samples)]
    pq.write_table(rows_to_table(rows, multimodal=multimodal), path)
    return len(rows)


def write_sft_parquet(
    samples: Iterable[Sample], path: str | Path, *, response_field: str | None, multimodal: bool | None = None
) -> int:
    """Write SFT ``messages`` rows for ``samples`` to ``path``; returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = list(samples)
    if not samples:
        raise ValueError(f"no samples to export to {path}")
    multimodal = has_media(samples) if multimodal is None else multimodal
    rows = [sft_sample_to_row(s, response_field=response_field, multimodal=multimodal) for s in samples]
    pq.write_table(sft_rows_to_table(rows, multimodal=multimodal), path)
    return len(rows)


def agent_name_for(config: RunConfig) -> str:
    return TOOL_AGENT if config.env.type == "tool_calling" else SINGLE_TURN_AGENT


def is_multimodal(config: RunConfig, samples: list[Sample]) -> bool:
    return config.data.image_field is not None or has_media(samples)


def export_datasets(config: RunConfig, run_dir: str | Path) -> dict[str, Path]:
    """Materialise train (and eval) parquet files from ``config.data`` via :func:`monorl.data.open_source`
    (RL prompt rows, or SFT ``messages`` rows when ``algorithm.name == 'sft'``)."""
    from monorl.data import open_source

    sft = config.is_sft
    train = list(open_source(config.data, seed=config.seed))
    multimodal = is_multimodal(config, train)
    out = {"train": sft_train_parquet_path(run_dir) if sft else train_parquet_path(run_dir)}
    agent = agent_name_for(config)
    response_field = config.data.response_field
    if sft:
        write_sft_parquet(train, out["train"], response_field=response_field, multimodal=multimodal)
    else:
        write_parquet(train, out["train"], agent_name=agent, multimodal=multimodal)
    if config.data.eval_uri:
        eval_cfg = config.data.model_copy(
            update={"uri": config.data.eval_uri, "split": config.data.eval_split}
        )
        out["val"] = sft_val_parquet_path(run_dir) if sft else val_parquet_path(run_dir)
        evaluation = list(open_source(eval_cfg, seed=config.seed))
        if sft:
            write_sft_parquet(evaluation, out["val"], response_field=response_field, multimodal=multimodal)
        else:
            write_parquet(evaluation, out["val"], agent_name=agent, multimodal=multimodal)
    return out
