"""SFT loop end to end: real ``reference`` trainer on the tiny random Llama, no rollout engine."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch
from monorl.core.errors import DataError
from monorl.core.reporting import RunReporter, iter_metrics, iter_trajectories, list_checkpoints, read_status
from monorl.core.types import Sample
from monorl.orchestrator import run
from monorl.orchestrator.sft import SFTLoop, encode_sft_example, sft_messages, sft_response
from monorl.trainer.reference import ReferenceTrainer

CONTRACT_KEYS = {
    "loss/sft",
    "sft/ppl",
    "sft/tokens",
    "tokens/response_mean",
    "tokens/prompt_mean",
    "tokens/prefix_mismatch",
    "lr",
    "time/train",
    "time/step",
}


@pytest.fixture(scope="module")
def tokenizer(tiny_model_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model_dir)


@pytest.fixture
def sft_config_dict(run_config_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        **run_config_dict,
        "run_name": "sft-test",
        "algorithm": {"name": "sft"},
        "rewards": [],
        "trainer": {
            **run_config_dict["trainer"],
            "max_steps": 5,
            "save_every": 2,
            "learning_rate": 1e-3,
            "max_seq_len": 64,
        },
    }


def _capture_batches(monkeypatch: pytest.MonkeyPatch) -> list[TrainBatch]:
    batches: list[TrainBatch] = []
    original = ReferenceTrainer.train_step

    def recording(self: ReferenceTrainer, batch: TrainBatch) -> dict[str, float]:
        batches.append(batch)
        return original(self, batch)

    monkeypatch.setattr(ReferenceTrainer, "train_step", recording)
    return batches


def test_sft_run_trains_checkpoints_and_exports(
    sft_config_dict: dict[str, Any], tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches = _capture_batches(monkeypatch)
    config = RunConfig.from_dict(sft_config_dict)
    result = run(config)

    assert result.state == "succeeded", result.error
    assert result.final_step == 5
    run_dir = Path(result.run_dir)
    assert read_status(run_dir)["state"] == "succeeded"

    rows = [row for _, row in iter_metrics(run_dir) if row["phase"] == "train"]
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
    missing = CONTRACT_KEYS - set(rows[0])
    assert not missing, f"missing metric keys: {sorted(missing)}; have {sorted(rows[0])}"
    assert rows[0]["tokens/prefix_mismatch"] == 0.0
    assert rows[0]["sft/tokens"] > 0 and rows[0]["tokens/response_mean"] > 0
    # Four samples, two per step: every sample is revisited, so the NLL must drop.
    assert rows[-1]["loss/sft"] < rows[0]["loss/sft"]
    assert rows[-1]["sft/ppl"] < rows[0]["sft/ppl"]
    assert "reward/mean" not in rows[0] and "loss/policy" not in rows[0]

    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-2", "step-4"]
    assert (run_dir / "checkpoints" / "step-2" / "trainer_state.pt").exists()
    assert (run_dir / "final" / "config.json").exists()
    assert json.loads((run_dir / "final" / "config.json").read_text())["architectures"] == [
        "LlamaForCausalLM"
    ]
    assert result.final_model_path == str(run_dir / "final")

    trajectories = [row for _, row in iter_trajectories(run_dir)]
    assert {t["step"] for t in trajectories} == {1, 2, 3, 4, 5}
    first = trajectories[0]
    assert first["prompt"].startswith("What is") and first["response"] in {"4", "8", "6", "21"}
    assert first["num_response_tokens"] > 0

    # Captured batches: the response mask excludes the prompt (and padding) and covers the answer.
    assert len(batches) == 5
    batch = batches[0]
    assert batch.response_mask.shape == batch.input_ids.shape and batch.old_logprobs is None
    assert batch.advantages.abs().sum().item() == 0.0
    for b, sample_id in enumerate(batch.meta["sample_ids"]):
        mask = batch.response_mask[b].bool()
        attn = batch.attention_mask[b].bool()
        assert sample_id and mask.sum() > 0 and (mask & ~attn).sum() == 0
        first_response = int(mask.nonzero()[0])
        assert mask[:first_response].sum() == 0  # prompt positions are not loss positions
        assert mask[
            first_response : int(attn.sum())
        ].all()  # the response span runs to the end of the sequence
        prompt = tokenizer.decode(batch.input_ids[b, :first_response])
        assert "What is" in prompt and prompt.rstrip().endswith("assistant")
        assert tokenizer.decode(batch.input_ids[b, first_response:][mask[first_response:]]).split(
            "<|im_end|>"
        )[0] in {"4", "8", "6", "21"}


def test_response_mask_excludes_prompt_and_train_on_prompt_covers_all(
    tiny_model_dir: Path, tokenizer
) -> None:
    messages = [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]
    example = encode_sft_example(tokenizer, messages)
    assert not example.prefix_mismatch
    prompt_text = tokenizer.decode(example.input_ids[: example.prompt_len])
    assert "What is 2+2?" in prompt_text and prompt_text.rstrip().endswith("assistant")
    assert example.response_mask[: example.prompt_len] == [0] * example.prompt_len
    assert all(example.response_mask[example.prompt_len :])
    assert tokenizer.decode(example.response_ids).startswith("4")
    assert example.response_ids[-2:] == [
        tokenizer.eos_token_id,
        tokenizer("\n", add_special_tokens=False)["input_ids"][0],
    ]

    everything = encode_sft_example(tokenizer, messages, train_on_prompt=True)
    assert everything.input_ids == example.input_ids
    assert everything.response_mask == [0] + [1] * (len(example.input_ids) - 1)

    truncated = encode_sft_example(tokenizer, messages, max_seq_len=example.prompt_len + 1)
    assert truncated.truncated and len(truncated.input_ids) == example.prompt_len + 1
    assert sum(truncated.response_mask) == 1

    with pytest.raises(DataError, match="assistant"):
        encode_sft_example(tokenizer, messages[:1])


def test_prefix_mismatch_falls_back_to_response_tokens(tokenizer) -> None:
    class RewritingTokenizer:
        """Chat template whose generation prompt does not survive in the full rendering."""

        chat_template = "x"
        eos_token_id = tokenizer.eos_token_id

        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):  # noqa: ANN001
            if add_generation_prompt:
                return "USER: " + messages[-1]["content"] + " ASSISTANT:"
            return "<<" + " ".join(m["content"] for m in messages) + ">>"

        def __call__(self, text, add_special_tokens):  # noqa: ANN001
            return tokenizer(text, add_special_tokens=add_special_tokens)

    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
    example = encode_sft_example(RewritingTokenizer(), messages)
    assert example.prefix_mismatch
    prompt_ids = tokenizer("USER: hello ASSISTANT:", add_special_tokens=False)["input_ids"]
    assert example.input_ids[: example.prompt_len] == prompt_ids
    assert example.response_ids == [
        *tokenizer("world", add_special_tokens=False)["input_ids"],
        tokenizer.eos_token_id,
    ]


def test_sft_response_lookup_order() -> None:
    base = [{"role": "user", "content": "q"}]
    assert (
        sft_response(Sample(id="a", messages=base, metadata={"response": "from-data-layer"}))
        == "from-data-layer"
    )
    assert sft_response(Sample(id="b", messages=base, metadata={"row": {"out": "col"}}), "out") == "col"
    turn = {"role": "assistant", "content": [{"type": "text", "text": "turn"}]}
    assert (
        sft_response(Sample(id="c", messages=base, reference="ref", metadata={"assistant_turn": turn}))
        == "turn"
    )
    assert sft_response(Sample(id="d", messages=base, reference="ref", metadata={"response": None})) == "ref"
    with pytest.raises(DataError, match="no SFT target"):
        sft_response(Sample(id="e", messages=base, reference=["not", "text"]))
    assert sft_messages(Sample(id="f", messages=base, reference="ref"))[-1] == {
        "role": "assistant",
        "content": "ref",
    }


def test_sft_resume_continues_from_checkpoint(sft_config_dict: dict[str, Any]) -> None:
    first_cfg = RunConfig.from_dict(
        {**sft_config_dict, "trainer": {**sft_config_dict["trainer"], "max_steps": 2}}
    )
    first = run(first_cfg)
    assert first.state == "succeeded" and first.final_step == 2
    run_dir = Path(first.run_dir)

    second_cfg = RunConfig.from_dict(
        {**sft_config_dict, "trainer": {**sft_config_dict["trainer"], "max_steps": 4}}
    )
    resumed = run(second_cfg, resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4
    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-2", "step-4"]
    assert [row["step"] for _, row in iter_metrics(run_dir) if row["phase"] == "train"] == [1, 2, 3, 4]
    assert "resumed from" in (run_dir / "run.log").read_text()


def test_sft_eval_loss_logged(sft_config_dict: dict[str, Any]) -> None:
    cfg = RunConfig.from_dict(
        {
            **sft_config_dict,
            "data": {**sft_config_dict["data"], "eval_uri": sft_config_dict["data"]["uri"]},
            "trainer": {**sft_config_dict["trainer"], "max_steps": 4, "eval_every": 2},
        }
    )
    result = run(cfg)
    assert result.state == "succeeded", result.error
    eval_rows = [row for _, row in iter_metrics(Path(result.run_dir)) if row["phase"] == "eval"]
    assert [r["step"] for r in eval_rows] == [2, 4]
    for r in eval_rows:
        assert r["eval/loss"] > 0 and r["eval/num_samples"] == 4 and r["eval/tokens"] > 0
        assert r["eval/ppl"] == pytest.approx(torch.exp(torch.tensor(r["eval/nll_token"])).item(), rel=1e-4)
    assert eval_rows[1]["eval/loss"] < eval_rows[0]["eval/loss"]


def test_sft_cancel_checkpoints_and_reports_cancelled(
    sft_config_dict: dict[str, Any], tmp_path: Path, monkeypatch
) -> None:
    config = RunConfig.from_dict(
        {**sft_config_dict, "trainer": {**sft_config_dict["trainer"], "save_every": 10}}
    )
    reporter = RunReporter(tmp_path / "cancel-run")
    cancel = threading.Event()
    original = ReferenceTrainer.train_step

    def cancelling(self: ReferenceTrainer, batch: TrainBatch) -> dict[str, float]:
        out = original(self, batch)
        if self.step == 3:
            cancel.set()
        return out

    monkeypatch.setattr(ReferenceTrainer, "train_step", cancelling)
    result = SFTLoop(config, reporter, cancel=cancel, resume=False).run()
    assert result.state == "cancelled" and result.final_step == 3
    assert [Path(p).name for p in result.checkpoints] == ["step-3"]
    assert result.final_model_path is None and not (reporter.run_dir / "final").exists()
    assert "cancel requested" in (reporter.run_dir / "run.log").read_text()


def test_sft_label_smoothing_and_sequence_agg_through_trainer(
    sft_config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trainer selects the SFT loss itself (no ``loss_fn`` injected) and feeds the vocab-mean
    logprobs the smoothing term needs."""
    import monorl.algorithms.sft as sft_module

    calls: list[dict[str, Any]] = []
    original = sft_module.sft_policy_loss

    def recording(logprobs, old, adv, mask, **kw):  # noqa: ANN001
        out = original(logprobs, old, adv, mask, **kw)
        calls.append({"logprobs": logprobs.detach(), "mask": mask, "loss": out.loss.item(), **kw})
        return out

    monkeypatch.setattr(sft_module, "sft_policy_loss", recording)
    cfg = RunConfig.from_dict(
        {
            **sft_config_dict,
            "algorithm": {"name": "sft", "sft_label_smoothing": 0.1, "loss_agg": "sequence"},
            "trainer": {**sft_config_dict["trainer"], "max_steps": 1, "save_every": 10},
        }
    )
    trainer = ReferenceTrainer(cfg)  # default loss resolved from algorithm.name
    assert trainer.loss_fn is recording
    reporter = RunReporter(Path(cfg.logging.run_root) / "smooth")
    loop = SFTLoop(cfg, reporter, cancel=threading.Event(), resume=False)
    loop._setup()
    loop.trainer = trainer
    batch, _, _ = loop._encode(loop.sampler.batch(1), 1)
    metrics = trainer.train_step(batch)
    assert calls, "trainer never called the SFT loss"
    call = calls[0]
    assert call["mean_logprob_all_tokens"].shape == call["logprobs"].shape
    expected = sft_module.sft_loss(
        call["logprobs"],
        call["mask"],
        label_smoothing=0.1,
        loss_agg="sequence",
        mean_logprob_all_tokens=call["mean_logprob_all_tokens"],
    ).loss.item()
    assert call["loss"] == pytest.approx(expected, rel=1e-6)
    plain = sft_module.sft_loss(call["logprobs"], call["mask"], loss_agg="sequence").loss.item()
    assert call["loss"] != pytest.approx(plain)
    assert metrics["loss/sft"] > 0 and metrics["sft/ppl"] > 1
    loop._teardown()
