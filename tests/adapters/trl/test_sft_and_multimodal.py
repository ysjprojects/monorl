"""``algorithm.name: sft`` → ``SFTTrainer`` mapping and multimodal (``images``) dataset export for TRL."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from monorl.adapters.trl.backend import load_policy_model, load_processing_class
from monorl.adapters.trl.callbacks import map_metrics
from monorl.adapters.trl.compile import (
    compile_sft_config,
    sft_config_kwargs,
    sft_label_smoothing,
    split_batch,
)
from monorl.adapters.trl.data import (
    build_dataset,
    build_train_and_eval,
    sample_to_row,
    sft_sample_to_row,
    structured_messages,
)
from monorl.adapters.trl.sft import LabelSmoothedSFTTrainer
from monorl.config.schema import DataConfig, RunConfig
from monorl.core.errors import ConfigError
from monorl.core.registry import registry
from monorl.core.reporting import RunReporter, iter_metrics, list_checkpoints, read_status
from monorl.core.types import Sample

CPU = torch.device("cpu")


@pytest.fixture
def chat_file(tmp_path: Path) -> Path:
    rows = [
        {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
        {"messages": [{"role": "user", "content": "What is 3+5?"}, {"role": "assistant", "content": "8"}]},
        {"messages": [{"role": "user", "content": "What is 10-4?"}, {"role": "assistant", "content": "6"}]},
        {"messages": [{"role": "user", "content": "What is 7*3?"}, {"role": "assistant", "content": "21"}]},
    ]
    p = tmp_path / "chat.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


@pytest.fixture
def sft_config(tiny_model_dir: Path, chat_file: Path, tmp_path: Path) -> RunConfig:
    return RunConfig.from_dict(
        {
            "run_name": "trl-sft",
            "model": {"name_or_path": str(tiny_model_dir), "dtype": "float32"},
            "data": {"uri": str(chat_file), "format": "chat", "max_prompt_tokens": 128},
            "algorithm": {"name": "sft", "sft_label_smoothing": 0.1},
            "trainer": {
                "backend": "trl",
                "prompts_per_step": 2,
                "micro_batch_size": 1,
                "max_steps": 2,
                "precision": "fp32",
                "device": "cpu",
                "save_every": 1,
                "max_seq_len": 256,
            },
            "logging": {"run_root": str(tmp_path / "runs")},
        }
    )


# -- config mapping ----------------------------------------------------------------------------


def test_sft_config_mapping(sft_config: RunConfig, tmp_path: Path) -> None:
    kw = sft_config_kwargs(sft_config, tmp_path, device=CPU)
    assert kw["output_dir"] == str(tmp_path / "checkpoints")
    assert kw["max_length"] == 256 and kw["max_steps"] == 2 and kw["save_steps"] == 1
    assert kw["per_device_train_batch_size"] == 1 and kw["gradient_accumulation_steps"] == 2
    assert kw["completion_only_loss"] is True and kw["assistant_only_loss"] is False
    assert "label_smoothing_factor" not in kw and kw["packing"] is False and kw["loss_type"] == "nll"
    assert sft_label_smoothing(sft_config) == 0.1
    assert kw["use_cpu"] is True and kw["report_to"] == "none"
    assert "num_generations" not in kw and "eval_strategy" not in kw
    assert split_batch(sft_config) == (1, 2)


def test_sft_train_on_prompt_eval_and_extra(sft_config: RunConfig, tmp_path: Path, chat_file: Path) -> None:
    cfg = sft_config.model_copy(deep=True)
    cfg.algorithm.sft_train_on_prompt = True
    cfg.data.eval_uri = str(chat_file)
    cfg.trainer.eval_every = 1
    cfg.trainer.extra = {"packing": True}
    args = compile_sft_config(cfg, tmp_path, device=CPU)
    assert args.completion_only_loss is False and args.packing is True
    assert args.eval_strategy.value == "steps" and args.eval_steps == 1
    cfg.trainer.extra = {"num_generations": 4}
    with pytest.raises(ConfigError, match="num_generations"):
        compile_sft_config(cfg, tmp_path, device=CPU)


def test_extra_loss_overrides_take_smoothing_away_from_monorl(sft_config: RunConfig, tmp_path: Path) -> None:
    cfg = sft_config.model_copy(deep=True)
    cfg.trainer.extra = {"label_smoothing_factor": 0.2}  # TRL's own smoother: must not be applied twice
    assert compile_sft_config(cfg, tmp_path, device=CPU).label_smoothing_factor == 0.2
    assert sft_label_smoothing(cfg) == 0.0
    cfg.trainer.extra = {"loss_type": "dft"}
    args = compile_sft_config(cfg, tmp_path, device=CPU)
    assert args.loss_type == "dft" and args.label_smoothing_factor == 0.0
    assert sft_label_smoothing(cfg) == 0.0


def test_sft_kwargs_reject_rl_algorithms(trl_config: RunConfig, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="sft"):
        sft_config_kwargs(trl_config, tmp_path, device=CPU)


def test_sft_metric_names() -> None:
    logs = {
        "loss": 2.0,
        "mean_token_accuracy": 0.25,
        "entropy": 1.5,
        "num_tokens": 40,
        "learning_rate": 1e-5,
        "epoch": 0.5,
    }
    out = map_metrics(logs, sft=True)
    assert out["loss/sft"] == 2.0 and out["sft/token_accuracy"] == 0.25 and out["loss/entropy"] == 1.5
    assert out["tokens/total"] == 40 and out["lr"] == 1e-5 and out["train/epoch"] == 0.5
    ev = map_metrics({"eval_loss": 1.0, "eval_mean_token_accuracy": 0.5, "eval_entropy": 0.1}, sft=True)
    assert ev == {"eval/loss": 1.0, "eval/sft/token_accuracy": 0.5, "eval/loss/entropy": 0.1}
    assert map_metrics({"loss": 2.0})["loss/policy"] == 2.0  # GRPO mapping untouched


# -- rows ---------------------------------------------------------------------------------------


def test_sft_rows_use_prompt_completion_format() -> None:
    sample = Sample(
        id="s1", messages=[{"role": "user", "content": "Q?"}], reference="A", metadata={"response": "A!"}
    )
    row = sft_sample_to_row(sample, response_field=None)
    assert row == {
        "prompt": [{"role": "user", "content": "Q?"}],
        "completion": [{"role": "assistant", "content": "A!"}],
    }


def _image_sample(tmp_path: Path, sid: str = "img") -> tuple[Sample, Path]:
    path = tmp_path / f"{sid}.png"
    Image.new("RGB", (8, 6), (255, 0, 0)).save(path)
    sample = Sample(
        id=sid,
        messages=[
            {
                "role": "user",
                "content": [{"type": "image", "image": str(path)}, {"type": "text", "text": "What colour?"}],
            }
        ],
        reference="red",
        metadata={"response": "red"},
    )
    return sample, path


def test_multimodal_rows_carry_images_and_placeholders(tmp_path: Path) -> None:
    sample, _ = _image_sample(tmp_path)
    row = sample_to_row(sample, multimodal=True)
    assert row["prompt"] == [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What colour?"}]}
    ]
    assert len(row["images"]) == 1 and row["images"][0].size == (8, 6)
    assert json.loads(row["sample_json"])["messages"] == sample.messages
    sft = sft_sample_to_row(sample, response_field=None, multimodal=True)
    assert sft["completion"] == [{"role": "assistant", "content": [{"type": "text", "text": "red"}]}]
    assert len(sft["images"]) == 1
    with pytest.raises(Exception, match="audio"):
        structured_messages([{"role": "user", "content": [{"type": "audio", "audio": "x.wav"}]}])


def test_build_dataset_with_image_field(tmp_path: Path) -> None:
    from datasets import Image as ImageFeature

    img = tmp_path / "pic.png"
    Image.new("RGB", (4, 4), (0, 255, 0)).save(img)
    data_file = tmp_path / "vqa.jsonl"
    rows = [{"question": "What colour?", "answer": "green", "picture": str(img)} for _ in range(2)]
    data_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cfg = DataConfig(
        uri=str(data_file), prompt_field="question", answer_field="answer", image_field="picture"
    )
    ds = build_dataset(cfg)
    assert "images" in ds.column_names
    assert isinstance(ds.features["images"].feature, ImageFeature)
    example = ds[0]
    assert example["images"][0].size == (4, 4)
    parts = example["prompt"][0]["content"]
    assert [p["type"] for p in parts] == ["image", "text"]
    assert [p["type"] for p in build_dataset(cfg, sft=True)[0]["completion"][0]["content"]] == ["text"]


# -- label-smoothed loss ---------------------------------------------------------------------------

EPS = 0.1


@pytest.fixture
def smoothed(
    sft_config: RunConfig, tmp_path: Path
) -> tuple[LabelSmoothedSFTTrainer, dict[str, torch.Tensor]]:
    """The trainer the backend builds for ``sft_label_smoothing: 0.1`` and one collated 3-row batch."""
    tokenizer, processing_class = load_processing_class(sft_config)
    train_ds, _ = build_train_and_eval(sft_config, tokenizer=tokenizer)
    trainer = LabelSmoothedSFTTrainer(
        model=load_policy_model(sft_config, CPU),
        args=compile_sft_config(sft_config, tmp_path, device=CPU),
        train_dataset=train_ds,
        processing_class=processing_class,
        label_smoothing=sft_label_smoothing(sft_config),
    )
    batch = dict(trainer.data_collator([trainer.train_dataset[i] for i in (0, 2, 3)]))
    return trainer, batch


def _closed_form(
    trainer: LabelSmoothedSFTTrainer, batch: dict[str, torch.Tensor], eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(token mean, mean of per-sequence means)`` of the smoothed cross-entropy, straight from logits."""
    with torch.no_grad():
        logits = trainer.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits, labels = logits[:, :-1], batch["labels"][:, 1:]
    ce = torch.nn.functional.cross_entropy
    token = ce(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, label_smoothing=eps)
    per_seq = [
        ce(lg, lb, ignore_index=-100, label_smoothing=eps) for lg, lb in zip(logits, labels, strict=True)
    ]
    return token, torch.stack(per_seq).mean()


def test_smoothed_loss_matches_closed_form_and_trains(
    smoothed: tuple[LabelSmoothedSFTTrainer, dict[str, torch.Tensor]],
) -> None:
    trainer, batch = smoothed
    model = trainer.model.train()
    loss_tokens = int((batch["labels"][:, 1:] != -100).sum())
    # completion-only masking survives: the prompt and the padding are not loss positions
    assert 0 < loss_tokens < int(batch["attention_mask"].sum()) < batch["input_ids"].numel()
    expected, _ = _closed_form(trainer, batch, EPS)
    unsmoothed, _ = _closed_form(trainer, batch, 0.0)

    loss, outputs = trainer.compute_loss(model, dict(batch), return_outputs=True)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)
    assert abs(loss.item() - unsmoothed.item()) > 1e-3
    assert outputs.logits.shape == (*batch["input_ids"].shape, model.config.vocab_size)

    loss.backward()
    grad = model.get_output_embeddings().weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0

    # Trainer's accumulated-batch normalisation: the micro-batch sum over the window's token count
    scaled = trainer.compute_loss(model, dict(batch), num_items_in_batch=torch.tensor(2 * loss_tokens))
    assert scaled.item() == pytest.approx(expected.item() / 2, abs=1e-5)


def test_smoothed_loss_sequence_aggregation_and_unsmoothed_eval(
    smoothed: tuple[LabelSmoothedSFTTrainer, dict[str, torch.Tensor]],
) -> None:
    trainer, batch = smoothed
    model = trainer.model.train()
    token, sequence = _closed_form(trainer, batch, EPS)
    assert abs(token.item() - sequence.item()) > 1e-4  # rows have different completion lengths

    trainer.loss_agg = "sequence"
    assert trainer.compute_loss(model, dict(batch)).item() == pytest.approx(sequence.item(), abs=1e-5)
    trainer.current_gradient_accumulation_steps = 2  # set by Trainer's loop for every accumulation window
    scaled = trainer.compute_loss(model, dict(batch), num_items_in_batch=torch.tensor(99))
    assert scaled.item() == pytest.approx(sequence.item() / 2, abs=1e-5)

    model.eval()  # eval_loss is the held-out NLL, as in monorl's SFT loop
    unsmoothed, _ = _closed_form(trainer, batch, 0.0)
    with torch.no_grad():
        assert trainer.compute_loss(model, dict(batch)).item() == pytest.approx(unsmoothed.item(), abs=1e-5)


# -- real CPU run --------------------------------------------------------------------------------


@pytest.mark.slow
def test_sft_end_to_end_on_cpu(sft_config: RunConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    reporter = RunReporter(run_dir, echo=False)
    backend = registry.get("pipeline_backend", "trl")(sft_config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = backend.run(sft_config, run_dir, reporter=reporter)

    deprecated = [
        str(w.message)
        for w in caught
        if issubclass(w.category, FutureWarning)
        and re.search(r"label_smoothing_factor|compute_loss_func", str(w.message))
    ]
    assert deprecated == []
    assert isinstance(backend.trainer, LabelSmoothedSFTTrainer) and backend.trainer.label_smoothing == 0.1

    assert result.state == "succeeded", result.error
    assert result.final_step == 2
    rows: list[dict[str, Any]] = [row for _, row in iter_metrics(run_dir)]
    train_rows = [r for r in rows if "loss/sft" in r]
    assert [r["step"] for r in train_rows] == [1, 2]
    for r in train_rows:
        assert {"loss/sft", "sft/token_accuracy", "lr", "tokens/total"} <= set(r) and r["phase"] == "train"
        assert "loss/policy" not in r

    status = read_status(run_dir)
    assert status is not None and status["state"] == "succeeded" and status["step"] == 2
    assert (run_dir / "final" / "config.json").exists()
    assert result.final_model_path == str(run_dir / "final")
    AutoModelForCausalLM.from_pretrained(run_dir / "final")
    checkpoints = list_checkpoints(run_dir)
    assert [Path(p).name for p in checkpoints] == ["step-1", "step-2"]
    assert result.checkpoints == checkpoints
    assert result.metrics["loss/sft"] > 0
