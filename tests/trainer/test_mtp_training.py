"""MTP (multi-token prediction) layers in the reference trainer on a tiny random Mellum: the
auxiliary loss trains them on the response tokens and leaves the policy's update alone when the
trunk is detached, composes with micro-batching, trainer ranks, LoRA and SFT, and the layers travel
in ``export_weights``, ``save_pretrained`` and checkpoints under the checkpoint's own keys."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, GenerationConfig

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch
from monorl.core.errors import ConfigError
from monorl.mtp import WIRE_PREFIX, MtpLayers, MtpLoss, checkpoint_key, is_wire_name
from monorl.trainer.reference import ReferenceTrainer
from tests.rollout.tiny_mtp import build_tiny_mtp, two_layer_mtp
from tests.trainer.conftest import make_config
from tests.trainer.test_distributed_trainer import WORLD_SIZE, _free_port, _shard

LORA = pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
PROMPTS = [[1, 5, 9, 17, 33, 21], [1, 40, 41, 42], [1, 7, 7, 300, 12, 90, 2000], [1, 77, 78]]
# rows hold different numbers of targets, so parts of the batch are unequal shares of its loss
NEW_TOKENS = [10, 4, 12, 7]
# plain SGD without clipping: the update is the gradient itself, so a wrongly weighed part shows
EXACT = {"optimizer": "sgd", "max_grad_norm": 1e6, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def mtp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny Mellum with a final norm that is not all ones. The MTP layer normalises the hidden
    states it reads, so with the builder's unit norms the output of the last decoder layer (what it
    has to read) and the input of the output head (final norm applied) would train it alike."""
    out = build_tiny_mtp(tmp_path_factory.mktemp("tiny-mtp"))
    weights = load_file(str(out / "model.safetensors"))
    weights["model.norm.weight"] = torch.linspace(0.25, 4.0, weights["model.norm.weight"].numel())
    save_file(weights, str(out / "model.safetensors"), metadata={"format": "pt"})
    return out


def _batch(rows: list[tuple[int, list[int]]], pad: int) -> TrainBatch:
    """Right-padded batch of ``(prompt length, tokens)`` rows; the tokens after the prompt are the
    response, advantage +1 on each."""
    width = max(len(tokens) for _, tokens in rows)
    input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    response_mask = torch.zeros_like(input_ids)
    for i, (prompt_len, tokens) in enumerate(rows):
        input_ids[i, : len(tokens)] = torch.tensor(tokens)
        attention[i, : len(tokens)] = 1
        response_mask[i, prompt_len : len(tokens)] = 1
    return TrainBatch(
        input_ids=input_ids,
        attention_mask=attention,
        response_mask=response_mask,
        advantages=response_mask.float(),
        rewards=torch.ones(len(rows)),
        group_ids=[f"g{i // 2}" for i in range(len(rows))],
    )


@pytest.fixture(scope="module")
def batch(mtp_dir: Path) -> TrainBatch:
    """The policy's own greedy continuations of ``PROMPTS``: what an MTP layer has to draft."""
    policy = AutoModelForCausalLM.from_pretrained(mtp_dir).eval()
    rows = []
    for prompt, new_tokens in zip(PROMPTS, NEW_TOKENS, strict=True):
        generation = GenerationConfig(
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=policy.config.pad_token_id,
            eos_token_id=policy.config.eos_token_id,
        )
        with torch.no_grad():
            tokens = policy.generate(input_ids=torch.tensor([prompt]), generation_config=generation)
        rows.append((len(prompt), tokens[0].tolist()))
    return _batch(rows, policy.config.pad_token_id)


def _config(
    model_dir: Path,
    *,
    lora: bool = False,
    mtp: dict[str, Any] | None = None,
    algorithm: str | None = None,
    **trainer: Any,
) -> RunConfig:
    cfg = make_config(model_dir, lora=lora, **trainer).to_dict()
    if mtp is not None:
        cfg["model"]["mtp"] = {"enabled": True, **mtp}
    if algorithm is not None:
        cfg["algorithm"]["name"] = algorithm
    return RunConfig.from_dict(cfg)


def _trainer(model_dir: Path, **settings: Any) -> ReferenceTrainer:
    torch.manual_seed(0)  # LoRA adapters are initialised at random
    return ReferenceTrainer(_config(model_dir, **settings))


def _layer_params(trainer: ReferenceTrainer) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in trainer.mtp.named_parameters()}


def _policy_params(trainer: ReferenceTrainer) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in trainer.model.named_parameters()}


def _assert_same(ours: dict[str, torch.Tensor], theirs: dict[str, torch.Tensor], **tolerance: float) -> None:
    assert ours.keys() == theirs.keys()
    for name, tensor in ours.items():
        torch.testing.assert_close(tensor, theirs[name], msg=lambda m, name=name: f"{name}: {m}", **tolerance)


def _differ(ours: dict[str, torch.Tensor], theirs: dict[str, torch.Tensor]) -> float:
    return max(float((ours[name] - theirs[name]).abs().max()) for name in ours)


def _loss_of(policy: Any, layers: MtpLayers, batch: TrainBatch) -> MtpLoss:
    """The MTP loss of ``batch`` straight from the layers, for a policy and layers that are given."""
    last_layer = [policy.config.num_hidden_layers - 1]
    with torch.no_grad():
        hidden = policy(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=last_layer
        ).hidden_states[-1]
        return layers.loss(hidden, batch.input_ids, batch.response_mask, detach_trunk=True)


def _stored_mtp_weights(model_dir: Path) -> dict[str, torch.Tensor]:
    """The MTP entries of the tiny checkpoint (a Mellum keeps them under ``mtp.``)."""
    stored = load_file(str(model_dir / "model.safetensors"))
    return {key: tensor for key, tensor in stored.items() if key.startswith("mtp.")}


@LORA
@pytest.mark.parametrize("algorithm", ["grpo", "sft"])
def test_the_layers_learn_to_draft_the_policys_continuations(
    mtp_dir: Path, batch: TrainBatch, lora: bool, algorithm: str
) -> None:
    trainer = _trainer(mtp_dir, lora=lora, mtp={"train": True}, algorithm=algorithm, learning_rate=3e-3)
    history = [trainer.train_step(batch) for _ in range(8)]
    first, last = history[0], history[-1]

    # the first step reports the loss of the checkpoint's layers on the response tokens
    policy = AutoModelForCausalLM.from_pretrained(mtp_dir)
    untrained = _loss_of(policy, MtpLayers(policy, mtp_dir), batch)
    assert untrained.targets == sum(NEW_TOKENS)
    assert first["loss/mtp"] == pytest.approx(float(untrained.loss), rel=1e-4)
    assert first["mtp/accuracy"] == pytest.approx(untrained.correct / untrained.targets)

    assert last["loss/mtp"] < first["loss/mtp"] - 0.5
    assert first["mtp/accuracy"] < 0.2 and last["mtp/accuracy"] > 0.6
    objective = "loss/sft" if algorithm == "sft" else "loss/policy"
    for metrics in history:
        assert metrics["loss/total"] == pytest.approx(
            metrics[objective] + 0.1 * metrics["loss/mtp"], abs=1e-5
        )
    assert "loss/mtp_1" not in last  # per-depth losses are for models with several MTP layers
    # a group of their own at the policy's learning rate
    assert [group["lr"] for group in trainer.optimizer.param_groups] == pytest.approx([3e-3, 3e-3])


@LORA
def test_with_a_detached_trunk_the_policy_takes_the_step_it_takes_without_mtp_training(
    mtp_dir: Path, batch: TrainBatch, lora: bool
) -> None:
    def run(mtp: dict[str, Any]) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
        trainer = _trainer(mtp_dir, lora=lora, mtp=mtp)
        metrics = [trainer.train_step(batch) for _ in range(2)]
        return _policy_params(trainer), metrics

    plain, plain_metrics = run({})
    # the clip is active, and the MTP gradient is large next to the policy's: a clip over both
    # together would shrink the policy's step
    assert plain_metrics[0]["grad_norm"] > 1.0

    detached, detached_metrics = run({"train": True, "loss_coef": 5.0})
    _assert_same(detached, plain, rtol=0, atol=1e-7)
    for ours, theirs in zip(detached_metrics, plain_metrics, strict=True):
        assert ours["grad_norm"] == pytest.approx(theirs["grad_norm"], rel=1e-6)
        assert ours["loss/policy"] == pytest.approx(theirs["loss/policy"], abs=1e-6)

    joint, _ = run({"train": True, "loss_coef": 5.0, "detach_trunk": False})
    assert _differ(joint, plain) > 1e-4


@pytest.mark.parametrize("detach_trunk", [True, False], ids=["detached", "joint"])
def test_micro_batched_step_equals_the_single_pass_step(
    mtp_dir: Path, batch: TrainBatch, detach_trunk: bool
) -> None:
    def step(micro_batch_size: int) -> tuple[ReferenceTrainer, dict[str, float]]:
        trainer = _trainer(
            mtp_dir,
            mtp={"train": True, "loss_coef": 1.0, "detach_trunk": detach_trunk},
            micro_batch_size=micro_batch_size,
            **EXACT,
        )
        return trainer, trainer.train_step(batch)

    initial = _layer_params(_trainer(mtp_dir, mtp={"train": True}))
    split, split_metrics = step(1)
    whole, whole_metrics = step(len(batch))
    for key in ("loss/mtp", "mtp/accuracy", "loss/total"):
        assert split_metrics[key] == pytest.approx(whole_metrics[key], rel=1e-4), key
    assert _differ(_layer_params(whole), initial) > 1e-3
    _assert_same(_layer_params(split), _layer_params(whole), rtol=1e-3, atol=1e-5)
    _assert_same(_policy_params(split), _policy_params(whole), rtol=1e-3, atol=1e-5)


def test_loss_coef_weighs_the_gradient_of_the_layers(mtp_dir: Path, batch: TrainBatch) -> None:
    def step(loss_coef: float) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        trainer = _trainer(mtp_dir, mtp={"train": True, "loss_coef": loss_coef}, **EXACT)
        before = _layer_params(trainer)
        metrics = trainer.train_step(batch)
        return {name: after - before[name] for name, after in _layer_params(trainer).items()}, metrics

    whole, whole_metrics = step(1.0)
    half, half_metrics = step(0.5)
    _assert_same(half, {name: 0.5 * moved for name, moved in whole.items()}, rtol=1e-4, atol=1e-7)
    assert half_metrics["loss/mtp"] == pytest.approx(whole_metrics["loss/mtp"], rel=1e-6)
    assert whole_metrics["loss/total"] - half_metrics["loss/total"] == pytest.approx(
        0.5 * whole_metrics["loss/mtp"], rel=1e-4
    )


def test_mtp_training_next_to_a_value_head(mtp_dir: Path, batch: TrainBatch) -> None:
    """``algorithm.name: ppo`` reads the input of the output head the MTP layers share with the
    policy: both losses train, and neither the policy nor the value head sees the MTP loss."""
    critic = batch.slice(0, len(batch))
    rows = torch.arange(len(batch))
    state = batch.response_mask.argmax(dim=1) - 1  # the last prompt token
    critic.value_mask = torch.zeros_like(batch.response_mask)
    critic.value_mask[rows, state] = 1
    critic.returns = torch.zeros(batch.input_ids.shape)
    critic.returns[rows, state] = torch.tensor([1.0, -1.0, 0.5, 2.0])
    critic.values = torch.zeros(batch.input_ids.shape)

    def run(mtp: dict[str, Any]) -> tuple[ReferenceTrainer, dict[str, float]]:
        trainer = _trainer(mtp_dir, mtp=mtp, algorithm="ppo")
        trainer.train_step(critic)  # moves the head off zero: the value loss reaches the policy from now on
        return trainer, trainer.train_step(critic)

    plain, plain_metrics = run({})
    trained, metrics = run({"train": True})
    assert metrics["loss/value"] == pytest.approx(plain_metrics["loss/value"], rel=1e-6)
    assert metrics["loss/total"] == pytest.approx(
        metrics["loss/policy"] + 0.5 * metrics["loss/value"] + 0.1 * metrics["loss/mtp"], abs=1e-5
    )
    _assert_same(_policy_params(trained), _policy_params(plain), rtol=0, atol=1e-7)
    _assert_same(trained.value_head.state_dict(), plain.value_head.state_dict(), rtol=0, atol=1e-7)
    assert _differ(_layer_params(trained), _layer_params(plain)) > 1e-4
    # policy, value head, MTP layers: the layers train at the policy's rate
    assert [g["lr"] for g in trained.optimizer.param_groups] == pytest.approx([1e-3, 1e-2, 1e-3])


class _TwoLayerTrainer(ReferenceTrainer):
    """Mellum checkpoints hold one MTP layer; the depth logic runs on two layers built in memory."""

    def _load_mtp(self) -> MtpLayers:
        mtp_model = two_layer_mtp()[1]
        mtp_model.tie_with_main_model(self.hf_model)  # embedding and head are this trainer's policy's
        return MtpLayers.of(self.hf_model, mtp_model)


def test_every_depth_reports_its_loss_and_parts_of_a_batch_add_up(mtp_dir: Path) -> None:
    """Two MTP layers and a response that starts at position 2: the first depth has one target
    more than the second, so every depth needs the batch's count of its own."""
    rows = [(2, [1, 5, 9, 17, 33]), (4, [1, 40, 41, 42, 64, 3, 90, 7]), (3, [1, 7, 7, 30, 12])]
    batch = _batch(rows, pad=0)
    policy, mtp_model = two_layer_mtp()
    expected = _loss_of(policy, MtpLayers.of(policy, mtp_model), batch)
    assert expected.counts == [9, 8]

    def step(micro_batch_size: int) -> tuple[ReferenceTrainer, dict[str, float]]:
        config = _config(
            mtp_dir, mtp={"train": True, "loss_coef": 1.0}, micro_batch_size=micro_batch_size, **EXACT
        )
        trainer = _TwoLayerTrainer(config, model=two_layer_mtp()[0])
        return trainer, trainer.train_step(batch)

    split, split_metrics = step(1)
    whole, whole_metrics = step(len(batch))
    depths = [entropy / count for entropy, count in zip(expected.sums, expected.counts, strict=True)]
    for metrics in (split_metrics, whole_metrics):
        assert metrics["loss/mtp_1"] == pytest.approx(depths[0], rel=1e-4)
        assert metrics["loss/mtp_2"] == pytest.approx(depths[1], rel=1e-4)
        assert metrics["loss/mtp"] == pytest.approx(sum(depths) / 2, rel=1e-4)
        assert metrics["mtp/accuracy"] == pytest.approx(expected.correct / 17)
        assert metrics["loss/total"] == pytest.approx(metrics["loss/policy"] + metrics["loss/mtp"], abs=1e-5)
    _assert_same(_layer_params(split), _layer_params(whole), rtol=1e-3, atol=1e-5)


@LORA
def test_export_and_final_checkpoint_hold_the_trained_layers(
    mtp_dir: Path, batch: TrainBatch, tmp_path: Path, lora: bool
) -> None:
    trainer = _trainer(mtp_dir, lora=lora, mtp={"train": True}, learning_rate=3e-3)
    for _ in range(3):
        trainer.train_step(batch)
    trained = _layer_params(trainer)
    plain = AutoModelForCausalLM.from_pretrained(mtp_dir)
    original = dict(MtpLayers(plain, mtp_dir).named_parameters())
    # full parameters, also next to LoRA adapters: every one of them moved
    assert all(not torch.equal(trained[name], original[name]) for name in trained)

    # weight sync: the policy under its own names (adapters merged), then the layers under the
    # checkpoint's keys, tagged as MTP weights
    exported = dict(trainer.export_weights())
    pushed = {checkpoint_key(name): tensor for name, tensor in exported.items() if is_wire_name(name)}
    stored = _stored_mtp_weights(mtp_dir)
    assert pushed.keys() == stored.keys()
    assert all(pushed[key].shape == stored[key].shape for key in stored)
    assert {name for name in exported if not is_wire_name(name)} == plain.state_dict().keys()
    assert all(is_wire_name(name) for name in list(exported)[-len(stored) :])
    engine_layers = MtpLayers(plain, mtp_dir)  # what an engine that drafts does with a pushed update
    engine_layers.load_checkpoint_weights(pushed)
    _assert_same(dict(engine_layers.named_parameters()), trained, rtol=0, atol=0)

    # final/: a complete checkpoint, the policy and the layers that were trained with it
    trainer.save_pretrained(tmp_path / "final")
    final_policy = AutoModelForCausalLM.from_pretrained(tmp_path / "final")
    final_layers = MtpLayers(final_policy, tmp_path / "final")
    _assert_same(dict(final_layers.named_parameters()), trained, rtol=0, atol=0)
    logits = final_policy(input_ids=batch.input_ids[:1], attention_mask=batch.attention_mask[:1]).logits
    actual = torch.log_softmax(logits[0, :-1], dim=-1).gather(-1, batch.input_ids[0, 1:, None])[:, 0]
    expected = trainer.compute_logprobs(batch)[0]
    torch.testing.assert_close(actual * batch.response_mask[0, 1:], expected[1:], atol=1e-4, rtol=0)
    # together they give the loss the trainer sees on its next step
    final_loss = _loss_of(final_policy, final_layers, batch)
    assert trainer.train_step(batch)["loss/mtp"] == pytest.approx(float(final_loss.loss), rel=1e-4)


@LORA
def test_checkpoint_restores_the_layers_and_their_optimizer_state(
    mtp_dir: Path, batch: TrainBatch, tmp_path: Path, lora: bool
) -> None:
    settings: dict[str, Any] = {"lora": lora, "mtp": {"train": True}, "learning_rate": 3e-3}
    trainer = _trainer(mtp_dir, **settings)
    for _ in range(3):
        trainer.train_step(batch)
    trainer.save_checkpoint(tmp_path / "ckpt")

    restored = _trainer(mtp_dir, **settings)
    assert _differ(_layer_params(restored), _layer_params(trainer)) > 1e-3
    restored.load_checkpoint(tmp_path / "ckpt")
    _assert_same(_layer_params(restored), _layer_params(trainer), rtol=0, atol=0)
    # Adam's moments came back too: both take the same next step
    before = _layer_params(trainer)
    metrics = [t.train_step(batch) for t in (trainer, restored)]
    assert metrics[1]["loss/mtp"] == pytest.approx(metrics[0]["loss/mtp"], rel=1e-6)
    assert _differ(_layer_params(trainer), before) > 1e-4
    _assert_same(_layer_params(restored), _layer_params(trainer), rtol=0, atol=1e-7)
    _assert_same(_policy_params(restored), _policy_params(trainer), rtol=0, atol=1e-7)

    # a checkpoint of a run without MTP layers cannot resume a run that holds them
    _trainer(mtp_dir, lora=lora).save_checkpoint(tmp_path / "plain")
    with pytest.raises(RuntimeError, match="holds no MTP layers"):
        _trainer(mtp_dir, lora=lora, mtp={}).load_checkpoint(tmp_path / "plain")


def test_enabled_without_training_carries_the_checkpoints_layers(
    mtp_dir: Path, tiny_model_dir: Path, batch: TrainBatch, tmp_path: Path
) -> None:
    trainer = _trainer(mtp_dir, mtp={})
    before = _policy_params(trainer)
    metrics = [trainer.train_step(batch) for _ in range(2)]
    assert _differ(_policy_params(trainer), before) > 1e-4  # the policy trains
    assert not [key for key in metrics[-1] if "mtp" in key]
    assert len(trainer.optimizer.param_groups) == 1

    # the layers are the checkpoint's, in weight sync, in final/ and after a checkpoint round trip
    stored = _stored_mtp_weights(mtp_dir)
    exported = dict(trainer.export_weights())
    _assert_same({key: exported[WIRE_PREFIX + key] for key in stored}, stored, rtol=0, atol=0)
    trainer.save_pretrained(tmp_path / "final")
    final = MtpLayers(AutoModelForCausalLM.from_pretrained(tmp_path / "final"), tmp_path / "final")
    _assert_same(dict(final.checkpoint_weights()), stored, rtol=0, atol=0)
    trainer.save_checkpoint(tmp_path / "ckpt")
    restored = _trainer(mtp_dir, mtp={})
    restored.load_checkpoint(tmp_path / "ckpt")
    _assert_same(dict(restored.mtp.checkpoint_weights()), stored, rtol=0, atol=0)

    # without the setting the export is the policy alone, as it always was
    assert not [name for name, _ in _trainer(mtp_dir).export_weights() if "mtp" in name]
    # a model that ships no MTP layers
    with pytest.raises(ConfigError, match="declares no MTP layers"):
        _trainer(tiny_model_dir, mtp={})


def _rank_worker(rank: int, port: int, model_dir: str, batch_path: str, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.manual_seed(0)
    config = _config(Path(model_dir), mtp={"train": True, "loss_coef": 1.0}, micro_batch_size=1, **EXACT)
    trainer = ReferenceTrainer(config, rank=rank, world_size=WORLD_SIZE)
    batch: TrainBatch = torch.load(batch_path, weights_only=False)
    metrics = trainer.train_step(_shard(batch, rank, WORLD_SIZE))
    torch.save({"metrics": metrics, "layers": _layer_params(trainer)}, Path(out_dir) / f"rank{rank}.pt")
    trainer.shutdown()


@pytest.mark.timeout(300)
def test_two_rank_step_matches_single_process(mtp_dir: Path, batch: TrainBatch, tmp_path: Path) -> None:
    """Rows 0-1 go to rank 0 and rows 2-3 to rank 1: the ranks hold 14 and 19 targets, and only the
    count over both ranks together with the all-reduce reproduces the full-batch gradient."""
    batch_path = tmp_path / "batch.pt"
    torch.save(batch, batch_path)
    mp.spawn(
        _rank_worker,
        args=(_free_port(), str(mtp_dir), str(batch_path), str(tmp_path)),
        nprocs=WORLD_SIZE,
        join=True,
    )
    ranks = [torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(WORLD_SIZE)]

    single = _trainer(mtp_dir, mtp={"train": True, "loss_coef": 1.0}, micro_batch_size=1, **EXACT)
    initial = _layer_params(single)
    expected = single.train_step(batch)
    assert _differ(_layer_params(single), initial) > 1e-3
    for r, data in enumerate(ranks):
        for key in ("loss/mtp", "mtp/accuracy", "loss/policy", "loss/total", "grad_norm"):
            assert data["metrics"][key] == pytest.approx(expected[key], rel=1e-4, abs=1e-5), f"rank {r} {key}"
        _assert_same(data["layers"], _layer_params(single), rtol=1e-3, atol=1e-5)
