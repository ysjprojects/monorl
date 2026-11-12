"""verl parquet images (geo3k layout), the SFT parquet/config and the ``--sft`` launch path."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from PIL import Image

from monorl.adapters.verl.backend import VerlBackend, final_model_path
from monorl.adapters.verl.compiler import compile_config, compile_sft_config, to_overrides
from monorl.adapters.verl.data import (
    PARQUET_COLUMNS,
    export_datasets,
    parquet_has_images,
    placeholder_messages,
    sample_to_row,
    sft_sample_to_row,
    write_parquet,
    write_sft_parquet,
)
from monorl.adapters.verl.patches import map_metrics, phase_for
from monorl.core.errors import ConfigError
from monorl.core.types import Sample


def _image_sample(tmp_path: Path, sid: str = "s1") -> Sample:
    path = tmp_path / f"{sid}.png"
    Image.new("RGB", (6, 6), (0, 128, 0)).save(path)
    return Sample(
        id=sid,
        messages=[
            {"role": "system", "content": "be precise"},
            {"role": "user", "content": [{"type": "image", "image": str(path)}, {"type": "text", "text": "How many sides?"}]},
        ],
        reference="4",
        metadata={"response": "four"},
    )


# -- RL parquet with images ------------------------------------------------------------------------


def test_placeholder_messages_and_image_structs(tmp_path: Path) -> None:
    sample = _image_sample(tmp_path)
    assert placeholder_messages(sample.messages) == [
        {"role": "system", "content": "be precise"},
        {"role": "user", "content": "<image>How many sides?"},
    ]
    row = sample_to_row(sample, 0, multimodal=True)
    assert tuple(row) == (*PARQUET_COLUMNS, "images")
    assert row["prompt"][1]["content"] == "<image>How many sides?"
    assert len(row["images"]) == 1 and row["images"][0]["path"] is None
    assert Image.open(io.BytesIO(row["images"][0]["bytes"])).size == (6, 6)
    assert json.loads(row["extra_info"]["messages_json"]) == sample.messages
    with pytest.raises(ValueError, match="video"):
        placeholder_messages([{"role": "user", "content": [{"type": "video", "video": "v.mp4"}]}])


def test_write_parquet_images_column(tmp_path: Path) -> None:
    path = tmp_path / "data" / "train.parquet"
    text_only = Sample(id="t", messages=[{"role": "user", "content": "q"}], reference="a")
    assert write_parquet([text_only], path) == 1
    assert not parquet_has_images(path)
    assert write_parquet([_image_sample(tmp_path), text_only], path) == 2
    assert parquet_has_images(path)
    table = pq.read_table(path)
    assert table.column_names == [*PARQUET_COLUMNS, "images"]
    rows = table.to_pylist()
    assert rows[0]["prompt"][1]["content"].startswith("<image>") and len(rows[0]["images"]) == 1
    assert rows[1]["images"] == []  # text-only rows in a multimodal export carry no images
    assert Image.open(io.BytesIO(rows[0]["images"][0]["bytes"])).convert("RGB").getpixel((0, 0)) == (0, 128, 0)


def test_export_datasets_with_image_field(make_config, tmp_path: Path) -> None:
    img = tmp_path / "pic.png"
    Image.new("RGB", (3, 3), (255, 255, 0)).save(img)
    data_file = tmp_path / "vqa.jsonl"
    data_file.write_text(json.dumps({"q": "colour?", "a": "yellow", "pic": str(img)}) + "\n")
    cfg = make_config(data={"uri": str(data_file), "prompt_field": "q", "answer_field": "a", "image_field": "pic"})
    paths = export_datasets(cfg, tmp_path)
    assert parquet_has_images(paths["train"])
    row = pq.read_table(paths["train"]).to_pylist()[0]
    assert "<image>" in row["prompt"][-1]["content"] and len(row["images"]) == 1


def test_compile_config_multimodal_keys(make_config, tmp_path: Path) -> None:
    cfg = make_config(data={"image_field": "pic"}, rollout={"extra": {"limit_mm_per_prompt": 2}})
    tree = compile_config(cfg, tmp_path)
    assert tree["data"]["image_key"] == "images"
    assert tree["actor_rollout_ref"]["model"]["use_remove_padding"] is True
    assert tree["actor_rollout_ref"]["rollout"]["engine_kwargs"] == {"vllm": {"+limit_mm_per_prompt": {"image": 2}}}
    assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=2" in to_overrides(tree)
    plain = compile_config(make_config(), tmp_path)
    assert "image_key" not in plain["data"] and "use_remove_padding" not in plain["actor_rollout_ref"]["model"]
    assert "image_key" in compile_config(make_config(), tmp_path, multimodal=True)["data"]


# -- SFT -------------------------------------------------------------------------------------------


def _sft_config(make_config, **overrides):
    return make_config(algorithm={"name": "sft"}, rewards=[], **overrides)


def test_sft_rows_and_parquet(tmp_path: Path) -> None:
    sample = Sample(id="s", messages=[{"role": "user", "content": "2+2?"}], reference="4", metadata={"response": "4"})
    assert sft_sample_to_row(sample, response_field=None) == {
        "messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
    }
    path = tmp_path / "sft.parquet"
    write_sft_parquet([sample, _image_sample(tmp_path)], path, response_field=None)
    table = pq.read_table(path)
    assert table.column_names == ["messages", "images"]
    rows = table.to_pylist()
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "4"} and rows[0]["images"] == []
    assert rows[1]["messages"][1]["content"] == "<image>How many sides?" and rows[1]["messages"][-1]["content"] == "four"
    assert len(rows[1]["images"]) == 1


def test_export_datasets_sft(make_config, tmp_path: Path, data_file: Path) -> None:
    cfg = _sft_config(make_config, data={"eval_uri": str(data_file), "shuffle": False})
    paths = export_datasets(cfg, tmp_path)
    assert paths["train"].name == "sft_train.parquet" and paths["val"].name == "sft_val.parquet"
    rows = pq.read_table(paths["train"]).to_pylist()
    assert pq.read_table(paths["train"]).column_names == ["messages"]
    assert {r["messages"][-1]["content"] for r in rows} == {"4", "8", "6"}
    assert all(r["messages"][-1]["role"] == "assistant" for r in rows)


def test_compile_sft_config_snapshot(make_config, tmp_path: Path, data_file: Path) -> None:
    cfg = _sft_config(
        make_config,
        data={"eval_uri": str(data_file)},
        trainer={"eval_every": 5, "max_seq_len": 2048, "precision": "bf16", "scheduler": "linear", "warmup_steps": 3},
        model={"name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "lora": {"enabled": True, "r": 8, "alpha": 16}},
    )
    run_dir = tmp_path / "run"
    tree = compile_config(cfg, run_dir)
    assert tree == compile_sft_config(cfg, run_dir)
    assert set(tree) == {"data", "model", "optim", "engine", "checkpoint", "trainer"}
    data, model, optim, trainer = tree["data"], tree["model"], tree["optim"], tree["trainer"]
    assert data["train_files"] == str((run_dir / "data" / "sft_train.parquet").resolve())
    assert data["val_files"] == str((run_dir / "data" / "sft_val.parquet").resolve())
    assert data["messages_key"] == "messages" and data["max_length"] == 2048 and data["truncation"] == "right"
    assert data["train_batch_size"] == 16 and data["micro_batch_size_per_gpu"] == 2 and data["use_dynamic_bsz"] is False
    assert "image_key" not in data
    assert model["path"] == "Qwen/Qwen2.5-0.5B-Instruct" and model["lora_rank"] == 8 and model["lora_alpha"] == 16
    assert model["use_remove_padding"] is True
    assert optim == {
        "lr": 2e-6,
        "weight_decay": cfg.trainer.weight_decay,
        "lr_warmup_steps": 3,
        "lr_scheduler_type": "cosine",
        "clip_grad": cfg.trainer.max_grad_norm,
        "total_training_steps": 30,
    }
    assert tree["engine"] == {"seed": 7, "dtype": "bfloat16"}
    assert tree["checkpoint"] == {"save_contents": ["model", "optimizer", "extra", "hf_model"]}
    assert trainer["total_training_steps"] == 30 and trainer["total_epochs"] == 30
    assert trainer["save_freq"] == 10 and trainer["test_freq"] == 5
    assert trainer["default_local_dir"] == str((run_dir / "checkpoints").resolve())
    assert trainer["nnodes"] == 2 and trainer["n_gpus_per_node"] == 8 and trainer["resume_mode"] == "auto"
    assert trainer["project_name"] == "monorl" and trainer["experiment_name"] == "verl-test"
    overrides = to_overrides(tree)
    assert "data.messages_key=messages" in overrides and "trainer.total_epochs=30" in overrides
    multimodal = compile_sft_config(_sft_config(make_config, data={"image_field": "pic"}), run_dir)
    assert multimodal["data"]["image_key"] == "images"
    assert multimodal["data"]["val_files"] is None


def test_sft_prepare_and_launch_command(make_config, tmp_path: Path) -> None:
    cfg = _sft_config(make_config)
    backend = VerlBackend(cfg, python="/opt/verl/bin/python")
    run_dir = tmp_path / "run"
    overrides = backend.prepare(cfg, run_dir)
    assert (run_dir / "data" / "sft_train.parquet").exists()
    assert not (run_dir / "monorl_reward.py").exists()
    assert any(o.startswith("data.train_files=") and o.endswith("sft_train.parquet") for o in overrides)
    cmd = backend.build_command(overrides, sft=cfg.is_sft)
    assert cmd[:4] == ["/opt/verl/bin/python", "-m", "monorl.adapters.verl.entry", "--sft"]
    assert backend.build_command(overrides)[3] != "--sft"
    assert (run_dir / "verl_config.yaml").exists()


def test_sft_checkpoint_hf_export_and_metrics(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints" / "global_step_4" / "huggingface"
    ckpt.mkdir(parents=True)
    assert final_model_path(tmp_path) == str(ckpt)
    metrics = map_metrics({"train/loss": 1.5, "train/lr": 1e-5, "train/global_tokens": 100, "train/mfu": 0.2})
    assert metrics["loss/sft"] == 1.5 and metrics["lr"] == 1e-5 and metrics["tokens/total"] == 100 and metrics["train/mfu"] == 0.2
    val = map_metrics({"val/loss": 0.9})
    assert val["eval/loss"] == 0.9 and phase_for({"val/loss": 0.9}) == "eval"


def test_sft_rejects_rl_only_options_in_prepare(make_config, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        make_config(algorithm={"name": "sft"}, env={"type": "tool_calling"})
