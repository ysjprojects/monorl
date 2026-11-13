"""``monorl.mtp.MtpLayers`` on a tiny random Mellum on CPU: the checkpoint layout both ways, the
auxiliary loss and what it reaches, agreement between the training target and what drafting
computes, drafting with the held layers, and an export that every loader reads."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, GenerationConfig, MellumForCausalLM
from transformers.cache_utils import MtpCache
from transformers.generation.candidate_generator import MTPCandidateGenerator
from transformers.modeling_layers import MtpModel

from monorl.core.errors import ConfigError
from monorl.mtp import (
    MtpLayers,
    checkpoint_key,
    checkpoint_shards,
    is_wire_name,
    mtp_key_pattern,
    num_mtp_layers,
    write_checkpoint_weights,
)
from tests.rollout.tiny_mtp import build_tiny_mtp, fit_to_greedy, tiny_mellum_config, two_layer_mtp

PROMPTS = [[1, 5, 9, 17, 33, 21], [1, 40, 41, 42], [1, 7, 7, 300, 12, 90, 2000]]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_mtp(tmp_path_factory.mktemp("tiny-mtp"))


@pytest.fixture
def policy(model_dir: Path) -> Any:
    return AutoModelForCausalLM.from_pretrained(model_dir).eval()


@pytest.fixture
def layers(policy: Any, model_dir: Path) -> MtpLayers:
    return MtpLayers(policy, model_dir)


def _checkpoint(model_dir: Path, layers: MtpLayers) -> dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in load_file(str(model_dir / "model.safetensors")).items()
        if layers.is_checkpoint_key(k)
    }


def _generate(
    policy: Any, prompt: list[int], *, use_mtp: bool, max_new_tokens: int = 16, **config: Any
) -> Any:
    settings: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_logits": True,
        "pad_token_id": policy.config.pad_token_id,
        "eos_token_id": policy.config.eos_token_id,
        "use_mtp": use_mtp,
    }
    generation = GenerationConfig(**(settings | config))
    return policy.generate(input_ids=torch.tensor([prompt]), generation_config=generation)


def test_weights_travel_in_the_checkpoints_own_layout(
    policy: Any, layers: MtpLayers, model_dir: Path
) -> None:
    stored = _checkpoint(model_dir, layers)
    assert len(stored) > len(layers.parameters())  # the checkpoint holds one tensor per expert
    exported = dict(layers.checkpoint_weights())
    assert exported.keys() == stored.keys()
    assert all(torch.equal(exported[key], stored[key]) for key in stored)
    # in a weight stream the same weights are tagged, so an engine need not know the model
    wire = dict(layers.wire_weights())
    assert all(is_wire_name(name) for name in wire) and not is_wire_name("model.embed_tokens.weight")
    assert {checkpoint_key(name): tensor for name, tensor in wire.items()}.keys() == stored.keys()
    with pytest.raises(ValueError, match="not an MTP weight name"):
        checkpoint_key("model.embed_tokens.weight")

    # the way in: other keys are ignored, the layers' parameters change, what is shared stays shared
    # and what an engine froze stays frozen
    for parameter in layers.parameters():
        parameter.requires_grad_(False)
    pushed = {key: value + 0.5 for key, value in stored.items()}
    layers.load_checkpoint_weights(pushed | {"model.embed_tokens.weight": torch.zeros(1)})
    back = dict(layers.checkpoint_weights())
    assert all(torch.allclose(back[key], pushed[key]) for key in pushed)
    assert layers.model.shared_head is policy.lm_head
    assert layers.model.embed_tokens is policy.get_input_embeddings()
    assert not any(parameter.requires_grad for parameter in layers.parameters())

    # a partial update would leave layers that draft from two policies at once
    partial = dict(list(pushed.items())[:-1])
    with pytest.raises(KeyError, match="incomplete: 1 of"):
        layers.load_checkpoint_weights(partial)
    wrong = pushed | {"mtp.fc.weight": torch.zeros(3, 3)}
    with pytest.raises(ValueError, match=r"mtp\.fc\.weight.*shape"):
        layers.load_checkpoint_weights(wrong)


def test_the_loss_reaches_only_the_mtp_layers_when_the_trunk_is_detached(
    policy: Any, layers: MtpLayers
) -> None:
    torch.manual_seed(0)
    ids = torch.randint(3, 1000, (3, 12))
    mask = torch.zeros_like(ids)
    mask[:, 5:] = 1
    mask[2, 9:] = 0  # a shorter row, right-padded
    for parameter in layers.parameters():
        parameter.requires_grad_(True)
    last_layer = [policy.config.num_hidden_layers - 1]

    def gradients(detach_trunk: bool) -> tuple[float, float, float]:
        policy.zero_grad()
        for parameter in layers.parameters():
            parameter.grad = None
        hidden = policy(input_ids=ids, output_hidden_states=last_layer).hidden_states[-1]
        result = layers.loss(hidden, ids, mask, detach_trunk=detach_trunk)
        assert result.counts == [int(mask[:, 2:].sum())] == layers.target_counts(mask)
        assert float(result.loss.detach()) == pytest.approx(result.sums[0] / result.counts[0], rel=1e-6)
        result.loss.backward()
        trunk = sum(float(p.grad.abs().sum()) for p in policy.parameters() if p.grad is not None)
        own = sum(float(p.grad.abs().sum()) for p in layers.parameters())
        return float(result.loss.detach()), trunk, own

    detached, trunk, own = gradients(detach_trunk=True)
    assert trunk == 0.0 and own > 0.0  # embedding and output head are the policy's and get nothing
    joint, trunk_joint, own_joint = gradients(detach_trunk=False)
    assert joint == pytest.approx(detached, rel=1e-6)
    assert trunk_joint > 0.0 and own_joint == pytest.approx(own, rel=1e-5)

    hidden = policy(input_ids=ids, output_hidden_states=last_layer).hidden_states[-1]
    nothing = layers.loss(hidden, ids, torch.zeros_like(ids), detach_trunk=True)
    assert (float(nothing.loss), nothing.targets, nothing.sums) == (0.0, 0, [0.0])
    assert not nothing.loss.requires_grad


def test_parts_of_a_batch_add_up_to_the_batch(policy: Any, layers: MtpLayers) -> None:
    """Micro-batches and trainer ranks see parts of a batch: with the batch's target counts their
    losses are shares, and the accumulated gradient is the batch's gradient."""
    torch.manual_seed(1)
    ids = torch.randint(3, 1000, (4, 10))
    mask = torch.zeros_like(ids)
    mask[0, 3:] = 1
    mask[1, 6:] = 1
    mask[3, 2:9] = 1  # row 2 has no target at all
    for parameter in layers.parameters():
        parameter.requires_grad_(True)
    last_layer = [policy.config.num_hidden_layers - 1]

    def run(rows: slice, totals: list[int] | None) -> float:
        hidden = policy(input_ids=ids[rows], output_hidden_states=last_layer).hidden_states[-1]
        result = layers.loss(hidden, ids[rows], mask[rows], detach_trunk=True, totals=totals)
        if result.loss.requires_grad:
            result.loss.backward()
        return float(result.loss.detach())

    whole = run(slice(0, 4), None)
    expected = [parameter.grad.clone() for parameter in layers.parameters()]
    for parameter in layers.parameters():
        parameter.grad = None
    totals = layers.target_counts(mask)
    parts = [run(slice(0, 1), totals), run(slice(1, 3), totals), run(slice(3, 4), totals)]
    assert sum(parts) == pytest.approx(whole, rel=1e-4)
    for parameter, wanted in zip(layers.parameters(), expected, strict=True):
        assert torch.allclose(parameter.grad, wanted, rtol=1e-3, atol=1e-5)


def test_every_depth_is_trained_on_what_drafting_computes() -> None:
    """Two MTP layers: the loss of depth ``k`` at a position is the cross entropy of exactly the
    logits transformers' drafting loop produces there for the same tokens."""
    policy, mtp_model = two_layer_mtp()
    layers = MtpLayers.of(policy, mtp_model)
    assert layers.num_layers == 2 and len(layers.parameters()) == 2 * 15
    prefix = torch.tensor([[1, 5, 9, 17, 33, 21, 64]])
    length = prefix.shape[1]
    last_layer = [policy.config.num_hidden_layers - 1]
    with torch.no_grad():
        # Drafting after the policy has read all but the last prefix token: depth 0 drafts the
        # token after the prefix, depth 1 the one after that.
        hidden = policy(input_ids=prefix[:, :-1], output_hidden_states=last_layer).hidden_states[-1]
        positions = torch.arange(length).unsqueeze(0)
        drafted, draft_logits, _ = mtp_model(
            input_ids=prefix[:, 1:],
            last_hidden_states=hidden,
            attention_mask=None,
            position_ids=positions[:, 1:],
            mtp_cache=MtpCache(config=policy.config.get_mtp_config()),
        )
        sequence = torch.cat([prefix, drafted], dim=1)
        hidden = policy(input_ids=sequence, output_hidden_states=last_layer).hidden_states[-1]
        for depth in (0, 1):
            target = torch.zeros_like(sequence)
            target[0, length + depth] = 1  # only the token this depth drafted
            result = layers.loss(hidden, sequence, target, detach_trunk=True)
            # depth 0 predicts it from the last prefix position, depth 1 from the same position;
            # the other depth predicts it from elsewhere and is not what is compared
            expected = -torch.log_softmax(draft_logits[0, depth], dim=-1)[drafted[0, depth]]
            assert result.counts[depth] == 1
            assert result.sums[depth] == pytest.approx(float(expected), rel=1e-4, abs=1e-5)
        both = torch.zeros_like(sequence)
        both[0, length:] = 1
        assert layers.loss(hidden, sequence, both, detach_trunk=True).correct >= 2  # the drafts are argmaxes


def test_held_layers_draft_what_was_pushed_and_never_change_the_output(
    policy: Any, layers: MtpLayers
) -> None:
    plain = [_generate(policy, prompt, use_mtp=False) for prompt in PROMPTS]
    layers.install_drafting()

    # random MTP layers: nothing is accepted, the output is the policy's own
    for prompt, expected in zip(PROMPTS, plain, strict=True):
        drafted = _generate(policy, prompt, use_mtp=True)
        assert torch.equal(drafted.sequences, expected.sequences)
    cold = layers.take_stats()
    new_tokens = sum(out.sequences.shape[1] - len(prompt) for out, prompt in zip(plain, PROMPTS, strict=True))
    assert cold.emitted == new_tokens == cold.steps + cold.accepted
    assert layers.take_stats().steps == 0  # counters restart with every read

    # layers that know these continuations: drafts are accepted, several tokens per pass, and the
    # tokens and the policy's logits for them are still exactly those of plain decoding
    fit_to_greedy(layers, PROMPTS)
    for prompt, expected in zip(PROMPTS, plain, strict=True):
        drafted = _generate(policy, prompt, use_mtp=True)
        assert torch.equal(drafted.sequences, expected.sequences)
        assert len(drafted.logits) == len(expected.logits)
        for ours, theirs in zip(drafted.logits, expected.logits, strict=True):
            assert torch.allclose(ours, theirs, atol=1e-4)
    warm = layers.take_stats()
    assert warm.emitted == new_tokens and warm.accepted > 0.8 * warm.drafted
    assert warm.steps < 0.7 * cold.steps
    assert warm.metrics()["mtp/accept_length"] > 1.5 and warm.metrics()["mtp/accept_rate"] > 0.8


def test_weights_pushed_to_another_holder_are_the_ones_that_draft(model_dir: Path, layers: MtpLayers) -> None:
    """An engine's layers start from the checkpoint (random here); after a push they draft like
    the trainer's. transformers alone would reload the checkpoint's layers for every call."""
    fit_to_greedy(layers, PROMPTS)
    engine_policy = AutoModelForCausalLM.from_pretrained(model_dir).eval()
    engine_layers = MtpLayers(engine_policy, model_dir)
    engine_layers.install_drafting()
    _generate(engine_policy, PROMPTS[0], use_mtp=True)
    assert engine_layers.take_stats().accepted == 0

    engine_layers.load_checkpoint_weights(dict(layers.checkpoint_weights()))
    _generate(engine_policy, PROMPTS[0], use_mtp=True)
    stats = engine_layers.take_stats()
    assert stats.accepted > 0.8 * stats.drafted


def test_drafted_tokens_stop_at_eos_and_at_the_token_budget(policy: Any, layers: MtpLayers) -> None:
    fit_to_greedy(layers, PROMPTS[:1])
    plain = _generate(policy, PROMPTS[0], use_mtp=False)
    continuation = plain.sequences[0, len(PROMPTS[0]) :].tolist()
    layers.install_drafting()

    # a budget that ends in the middle of an accepted draft
    short = _generate(policy, PROMPTS[0], use_mtp=True, max_new_tokens=4)
    assert short.sequences[0, len(PROMPTS[0]) :].tolist() == continuation[:4]
    assert layers.take_stats().emitted == 4

    # an end-of-sequence token inside an accepted draft ends the sequence there
    eos = continuation[5]
    plain_eos = _generate(policy, PROMPTS[0], use_mtp=False, eos_token_id=[eos])
    drafted = _generate(policy, PROMPTS[0], use_mtp=True, eos_token_id=[eos])
    assert torch.equal(drafted.sequences, plain_eos.sequences)
    assert drafted.sequences[0, -1] == eos and drafted.sequences.shape[1] < plain.sequences.shape[1]


def test_an_export_with_the_mtp_layers_is_a_complete_checkpoint(
    policy: Any, layers: MtpLayers, tmp_path: Path
) -> None:
    with torch.no_grad():
        for parameter in layers.parameters():
            parameter.add_(0.25)  # "trained"
    for name, shard_size in (("single", None), ("sharded", "4MB")):
        out = tmp_path / name
        policy.save_pretrained(out, **({"max_shard_size": shard_size} if shard_size else {}))
        before = len(checkpoint_shards(out))
        assert (before > 1) == (name == "sharded")
        # save_pretrained drops the MTP layers: transformers' own loader finds nothing to load
        with pytest.raises(ConfigError, match="holds no MTP weights"):
            MtpLayers(AutoModelForCausalLM.from_pretrained(out), out)

        write_checkpoint_weights(out, layers.checkpoint_weights(tied_from=policy.state_dict()))
        shards = checkpoint_shards(out)
        assert len(shards) == before + 1 and all(shard.exists() for shard in shards)
        assert not (out / "model.safetensors").exists()

        reloaded = AutoModelForCausalLM.from_pretrained(out)
        for (name_, ours), (_, theirs) in zip(
            policy.state_dict().items(), reloaded.state_dict().items(), strict=True
        ):
            assert torch.equal(ours, theirs), name_
        again = MtpLayers(reloaded, out)
        upstream = MtpModel.from_pretrained(reloaded)  # transformers reads the directory too
        for name_, parameter in layers.named_parameters():
            assert torch.equal(dict(again.named_parameters())[name_], parameter), name_
            assert torch.equal(upstream.state_dict()[name_], parameter), name_


def test_checkpoint_copies_of_head_and_embedding_follow_the_policy(
    policy: Any, model_dir: Path, tmp_path: Path
) -> None:
    """DeepSeek-style checkpoints keep copies of the output head and the embedding under the MTP
    layer. The layers share the policy's modules instead: the copies are not weights of their
    own, never travel in weight sync, and an export writes them with the policy's current values."""
    source = tmp_path / "with-copies"
    shutil.copytree(model_dir, source)
    stored = load_file(str(source / "model.safetensors"))
    stored["mtp.shared_head.head.weight"] = torch.zeros_like(stored["lm_head.weight"])  # stale on purpose
    stored["mtp.embed_tokens.weight"] = torch.zeros_like(stored["model.embed_tokens.weight"])
    save_file(stored, str(source / "model.safetensors"), metadata={"format": "pt"})

    embedding_before = policy.get_input_embeddings().weight.clone()
    head_before = policy.lm_head.weight.clone()
    layers = MtpLayers(policy, source)
    copies = {"mtp.shared_head.head.weight", "mtp.embed_tokens.weight"}
    # loading the layers leaves the policy as it was: the checkpoint's copies are not the policy's
    assert torch.equal(policy.get_input_embeddings().weight, embedding_before)
    assert torch.equal(policy.lm_head.weight, head_before)
    assert layers.model.shared_head is policy.lm_head
    assert layers.model.embed_tokens is policy.get_input_embeddings()
    assert not copies & dict(layers.checkpoint_weights()).keys()
    assert not any(checkpoint_key(name) in copies for name, _ in layers.wire_weights())
    # a push that carries the copies anyway leaves the policy alone
    layers.load_checkpoint_weights({key: stored[key] for key in stored if layers.is_checkpoint_key(key)})
    assert torch.equal(policy.lm_head.weight, head_before)
    assert torch.equal(policy.get_input_embeddings().weight, embedding_before)

    with torch.no_grad():
        policy.lm_head.weight.add_(1.0)  # the policy has trained since
    out = tmp_path / "export"
    policy.save_pretrained(out)
    write_checkpoint_weights(out, layers.checkpoint_weights(tied_from=policy.state_dict()))
    written = {}
    for shard in checkpoint_shards(out):
        written.update(load_file(str(shard)))
    assert torch.equal(written["mtp.shared_head.head.weight"], policy.lm_head.weight)
    assert torch.equal(written["mtp.embed_tokens.weight"], policy.get_input_embeddings().weight)
    MtpLayers(AutoModelForCausalLM.from_pretrained(out), out)  # and it loads back


def test_models_without_mtp_layers_are_refused(tiny_model_dir: Path, tmp_path: Path) -> None:
    llama = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    assert num_mtp_layers(llama.config) == 0 and mtp_key_pattern(llama) is None
    with pytest.raises(ConfigError, match="declares no MTP layers"):
        MtpLayers(llama, tiny_model_dir)
    # layers built in memory have no checkpoint to be laid out like
    policy, mtp_model = two_layer_mtp()
    with pytest.raises(ConfigError, match="no checkpoint layout"):
        list(MtpLayers.of(policy, mtp_model).checkpoint_weights())
    # a model of an MTP family whose checkpoint holds no MTP layers
    bare = tmp_path / "no-mtp"
    MellumForCausalLM(tiny_mellum_config(128)).save_pretrained(bare)
    with pytest.raises(ConfigError, match="holds no MTP weights"):
        MtpLayers(AutoModelForCausalLM.from_pretrained(bare), bare)


def test_the_held_candidate_generator_sets_what_transformers_sets(policy: Any, layers: MtpLayers) -> None:
    """``MtpLayers.install_drafting`` re-does ``MTPCandidateGenerator.__init__`` without loading
    the layers from disk. A transformers release that gives the generator more state has to
    fail here, not during a rollout."""
    config = GenerationConfig(do_sample=False, use_mtp=True)
    upstream = MTPCandidateGenerator(policy, config, {}, None)
    layers.install_drafting()
    held = policy._get_candidate_generator(generation_config=config, logits_processor=None)
    assert isinstance(held, MTPCandidateGenerator) and held.mtp_model is layers.model
    assert set(vars(upstream)) <= set(vars(held))
    assert held.model_kwargs_overrides == upstream.model_kwargs_overrides
    assert held.num_mtp_layers == upstream.num_mtp_layers
