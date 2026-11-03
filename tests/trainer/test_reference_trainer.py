from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from monorl.core.interfaces import Trainer
from monorl.rollout.hf_engine import HFRolloutEngine
from monorl.trainer.reference import ReferenceTrainer
from tests.trainer.conftest import make_batch, make_config, reinforce_loss

LORA = pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])


def target_logprob(trainer: ReferenceTrainer, batch) -> float:
    lp = trainer.compute_logprobs(batch)
    return float((lp * batch.response_mask).sum() / batch.response_mask.sum())


def test_registry_and_protocol(tiny_model_dir: Path) -> None:
    from monorl.core.registry import registry

    factory = registry.get("trainer", "reference")
    trainer = factory(make_config(tiny_model_dir), loss_fn=reinforce_loss)
    assert isinstance(trainer, Trainer)
    assert trainer.step == 0
    assert trainer.max_seq_len == 64


@LORA
def test_loss_decreases_over_five_steps(tiny_model_dir: Path, tokenizer, lora: bool) -> None:
    trainer = ReferenceTrainer(make_config(tiny_model_dir, lora=lora), loss_fn=reinforce_loss)
    if lora:
        assert any("lora_" in n for n, p in trainer.model.named_parameters() if p.requires_grad)
        assert all("lora_" in n for n, p in trainer.model.named_parameters() if p.requires_grad)
    batch = make_batch(tokenizer)
    before = target_logprob(trainer, batch)
    losses = []
    for i in range(5):
        metrics = trainer.train_step(batch)
        losses.append(metrics["loss/policy"])
        assert trainer.step == i + 1
        for key in ("loss/policy", "loss/total", "loss/entropy", "grad_norm", "lr", "tokens/response_mean"):
            assert key in metrics, key
        assert metrics["grad_norm"] > 0
    assert losses[-1] < losses[0]
    assert target_logprob(trainer, batch) > before


@LORA
def test_compute_logprobs_alignment_and_reference(tiny_model_dir: Path, tokenizer, lora: bool) -> None:
    trainer = ReferenceTrainer(make_config(tiny_model_dir, lora=lora), loss_fn=reinforce_loss)
    batch = make_batch(tokenizer)
    lp = trainer.compute_logprobs(batch)
    assert lp.shape == batch.input_ids.shape
    assert torch.all(lp[batch.response_mask == 0] == 0)
    assert torch.all(lp[batch.response_mask == 1] < 0)
    # Matches a plain forward pass over the unpadded first sequence.
    n = int(batch.attention_mask[0].sum())
    ids = batch.input_ids[0, :n][None]
    base = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
    with torch.no_grad():
        logits = base(input_ids=ids).logits[0]
    ref = torch.log_softmax(logits[:-1], -1).gather(1, ids[0, 1:, None]).squeeze(1)
    ref = torch.nn.functional.pad(ref, (1, 0)) * batch.response_mask[0, :n]
    torch.testing.assert_close(lp[0, :n], ref, atol=1e-5, rtol=0)
    assert trainer.model.training  # mode restored

    initial_ref = trainer.compute_logprobs(batch, reference=True)
    torch.testing.assert_close(initial_ref, lp, atol=1e-5, rtol=0)
    for _ in range(2):
        trainer.train_step(batch)
    after = trainer.compute_logprobs(batch)
    assert not torch.allclose(after, lp)
    # Reference policy is frozen at the initial weights (adapters disabled / frozen base copy).
    torch.testing.assert_close(trainer.compute_logprobs(batch, reference=True), initial_ref, atol=1e-5, rtol=0)
    if not lora:
        assert trainer._reference_model is not None


def test_micro_batching_matches_single_pass(tiny_model_dir: Path, tokenizer) -> None:
    batch = make_batch(tokenizer)
    results = []
    for mbs in (1, 4):
        torch.manual_seed(0)
        trainer = ReferenceTrainer(make_config(tiny_model_dir, micro_batch_size=mbs), loss_fn=reinforce_loss)
        metrics = trainer.train_step(batch)
        results.append((metrics, trainer.compute_logprobs(batch)))
    (m1, lp1), (m4, lp4) = results
    assert m1["loss/policy"] == pytest.approx(m4["loss/policy"], abs=1e-5)
    assert m1["grad_norm"] == pytest.approx(m4["grad_norm"], rel=1e-4)
    torch.testing.assert_close(lp1, lp4, atol=1e-5, rtol=0)


def test_truncates_to_max_seq_len(tiny_model_dir: Path, tokenizer) -> None:
    trainer = ReferenceTrainer(make_config(tiny_model_dir, max_seq_len=16), loss_fn=reinforce_loss)
    batch = make_batch(tokenizer)
    assert batch.input_ids.shape[1] > 16
    lp = trainer.compute_logprobs(batch)
    assert lp.shape == (len(batch), 16)
    trainer.train_step(batch)


@LORA
def test_checkpoint_roundtrip(tiny_model_dir: Path, tokenizer, tmp_path: Path, lora: bool) -> None:
    cfg = make_config(tiny_model_dir, lora=lora, scheduler="linear", warmup_steps=1)
    batch = make_batch(tokenizer)
    trainer = ReferenceTrainer(cfg, loss_fn=reinforce_loss)
    for _ in range(2):
        trainer.train_step(batch)
    ckpt = tmp_path / "ckpt"
    trainer.save_checkpoint(ckpt)
    assert (ckpt / "trainer_state.pt").exists()
    assert (ckpt / ("adapter" if lora else "model")).is_dir()

    restored = ReferenceTrainer(cfg, loss_fn=reinforce_loss)
    assert not torch.allclose(restored.compute_logprobs(batch), trainer.compute_logprobs(batch))
    restored.load_checkpoint(ckpt)
    assert restored.step == 2
    torch.testing.assert_close(restored.compute_logprobs(batch), trainer.compute_logprobs(batch), atol=1e-6, rtol=0)
    a, b = trainer.optimizer.state_dict(), restored.optimizer.state_dict()
    assert a["param_groups"] == b["param_groups"]
    assert a["state"].keys() == b["state"].keys() and len(a["state"]) > 0
    for key in a["state"]:
        for k, v in a["state"][key].items():
            torch.testing.assert_close(b["state"][key][k], v, atol=0, rtol=0)
    assert restored.scheduler.get_last_lr() == trainer.scheduler.get_last_lr()
    # Continuing training from the restored state is bit-for-bit the same as never stopping.
    m1 = trainer.train_step(batch)
    m2 = restored.train_step(batch)
    assert m1["lr"] == m2["lr"] and m1["loss/policy"] == pytest.approx(m2["loss/policy"], abs=1e-6)
    torch.testing.assert_close(restored.compute_logprobs(batch), trainer.compute_logprobs(batch), atol=1e-6, rtol=0)


@LORA
def test_export_update_roundtrip_matches_save_pretrained(tiny_model_dir: Path, tokenizer, tmp_path: Path, lora: bool) -> None:
    cfg = make_config(tiny_model_dir, lora=lora)
    batch = make_batch(tokenizer)
    trainer = ReferenceTrainer(cfg, loss_fn=reinforce_loss)
    trainer.train_step(batch)
    policy_lp = trainer.compute_logprobs(batch)
    probe = batch.input_ids[0, : int(batch.attention_mask[0].sum())][None]

    if lora:
        adapters_before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if "lora_" in n}
        assert any(t.abs().sum() > 0 for n, t in adapters_before.items() if "lora_B" in n)
    exported = {}
    engine = HFRolloutEngine(make_config(tiny_model_dir))
    try:
        engine_names = set(engine.model.state_dict())

        def stream():
            for name, tensor in trainer.export_weights():
                assert name in engine_names, name
                assert ".lora_" not in name and "base_layer" not in name
                exported[name] = tensor
                yield name, tensor

        engine.update_weights(stream())
        assert set(exported) == engine_names
        with torch.no_grad():
            engine_logits = engine.model(input_ids=probe).logits.float().clone()
    finally:
        engine.shutdown()
    # Export did not mutate the training model.
    torch.testing.assert_close(trainer.compute_logprobs(batch), policy_lp, atol=0, rtol=0)
    if lora:
        for n, p in trainer.model.named_parameters():
            if "lora_" in n:
                torch.testing.assert_close(p, adapters_before[n], atol=0, rtol=0)

    out = tmp_path / "export"
    trainer.save_pretrained(out)
    merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    assert not any("lora" in k for k in merged.state_dict())
    with torch.no_grad():
        merged_logits = merged(input_ids=probe).logits.float()
    torch.testing.assert_close(engine_logits, merged_logits, atol=1e-5, rtol=0)
    # Both agree with the trainer's own view of the policy.
    ref = torch.log_softmax(merged_logits[0, :-1], -1).gather(1, probe[0, 1:, None]).squeeze(1)
    n = probe.shape[1]
    torch.testing.assert_close(policy_lp[0, 1:n], ref * batch.response_mask[0, 1:n], atol=1e-4, rtol=0)
    # Untrained weights differ from the base model, so the roundtrip is not vacuous.
    base = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
    with torch.no_grad():
        assert not torch.allclose(base(input_ids=probe).logits.float(), merged_logits)
    assert (out / "tokenizer_config.json").exists()


@pytest.mark.parametrize("optimizer", ["adamw", "adam", "sgd"])
@pytest.mark.parametrize("scheduler", ["constant", "linear", "cosine"])
def test_optimizers_and_schedulers(tiny_model_dir: Path, tokenizer, optimizer: str, scheduler: str) -> None:
    cfg = make_config(tiny_model_dir, optimizer=optimizer, scheduler=scheduler, warmup_steps=2, max_steps=4, learning_rate=1e-3)
    trainer = ReferenceTrainer(cfg, loss_fn=reinforce_loss)
    batch = make_batch(tokenizer)
    lrs = [trainer.train_step(batch)["lr"] for _ in range(4)]
    assert lrs[0] == pytest.approx(0.5e-3)
    assert lrs[1] == pytest.approx(1e-3)
    if scheduler == "constant":
        assert lrs[2] == pytest.approx(1e-3) and lrs[3] == pytest.approx(1e-3)
    else:
        assert lrs[2] < 1e-3 and lrs[3] <= lrs[2]


@LORA
def test_gradient_checkpointing(tiny_model_dir: Path, tokenizer, lora: bool) -> None:
    trainer = ReferenceTrainer(make_config(tiny_model_dir, lora=lora, gradient_checkpointing=True), loss_fn=reinforce_loss)
    batch = make_batch(tokenizer)
    before = target_logprob(trainer, batch)
    metrics = trainer.train_step(batch)
    assert metrics["grad_norm"] > 0
    assert target_logprob(trainer, batch) > before


def test_default_loss_fn_is_monorl_algorithms(tiny_model_dir: Path, tokenizer) -> None:
    algorithms = pytest.importorskip("monorl.algorithms")
    if not hasattr(algorithms, "policy_loss"):
        pytest.skip("monorl.algorithms.policy_loss not available yet")
    trainer = ReferenceTrainer(make_config(tiny_model_dir))
    assert trainer.loss_fn is algorithms.policy_loss
    batch = make_batch(tokenizer)
    batch.old_logprobs = trainer.compute_logprobs(batch)
    metrics = trainer.train_step(batch)
    assert "loss/policy" in metrics and "policy/approx_kl" in metrics
