"""Builds a random tiny Mellum — a model family that ships MTP (multi-token prediction) layers —
with one MTP layer stored in the checkpoint's own layout and the SmolLM tokenizer, for MTP tests
on CPU. ``MellumForCausalLM.save_pretrained`` leaves MTP layers out, so they are written the way
JetBrains' checkpoints hold them (``mtp.fc``, ``mtp.pre_fc_norm_*``, ``mtp.norm``,
``mtp.layers.0.*`` with one tensor per expert); ``transformers.modeling_layers.MtpModel`` reads
them from there.

Mellum checkpoints hold at most one MTP layer. :func:`two_layer_mtp` builds a two-layer
``MtpModel`` in memory for the tests of the depth logic, on a policy with tied embeddings (the
saved model has an output head of its own).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TINY_TOKENIZER = "HuggingFaceTB/SmolLM-135M-Instruct"


def tiny_mellum_config(vocab_size: int, *, num_mtp_layers: int = 1, **overrides: Any) -> Any:
    from transformers import MellumConfig

    settings: dict[str, Any] = {
        "vocab_size": vocab_size,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "moe_intermediate_size": 64,
        "num_experts_per_tok": 2,
        "num_experts": 4,
        "max_position_embeddings": 512,
        "sliding_window": 128,
        # one layer of each attention type: the MTP layer is a sliding-attention layer and the
        # rotary embedding only knows the types the model uses
        "layer_types": ["full_attention", "sliding_attention"],
        "num_mtp_layers": num_mtp_layers,
        "mtp_layer_types": ["sliding_attention"] * num_mtp_layers,
        "mtp_mlp_layer_types": ["sparse"] * num_mtp_layers,
        "tie_word_embeddings": True,
    }
    return MellumConfig(**(settings | overrides))


def init_parameters(module: Any) -> None:
    """Give every parameter a seeded value. Mellum's packed expert weights are allocated with
    ``torch.empty`` and not touched by its weight initialisation: left alone they hold whatever
    the allocator returns, sometimes NaN. Norm weights get values away from 1: with unit weights
    an RMS norm applied twice equals one applied once, and reading a hidden state from the wrong
    side of a norm would go unnoticed."""
    import torch

    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if "norm" in name:
                parameter.uniform_(0.5, 1.5)
            else:
                parameter.normal_(mean=0.0, std=0.02)


def mtp_checkpoint_weights(model: Any, mtp_model: Any) -> dict[str, Any]:
    """``mtp_model``'s single layer under Mellum's checkpoint keys."""
    from transformers.core_model_loading import revert_weight_conversion

    state = mtp_model.state_dict()
    # The block is a decoder layer of the policy: its tensors are stored like any other layer's.
    block = {
        "model.layers.0." + key.split(".mtp_block.", 1)[1]: value
        for key, value in state.items()
        if ".mtp_block." in key
    }
    weights = {
        key.replace("model.layers.0.", "mtp.layers.0.", 1): value
        for key, value in revert_weight_conversion(model, block).items()
    }
    weights["mtp.fc.weight"] = state["layers.0.eh_proj.weight"]
    weights["mtp.pre_fc_norm_embedding.weight"] = state["layers.0.enorm.weight"]
    weights["mtp.pre_fc_norm_hidden.weight"] = state["layers.0.hnorm.weight"]
    weights["mtp.norm.weight"] = state["layers.0.post_norm.weight"]
    return {key: value.detach().clone() for key, value in weights.items()}


def build_tiny_mtp(out: Path, *, seed: int = 0) -> Path:
    """Random tiny Mellum with one MTP layer and its tokenizer saved to ``out``; returns ``out``."""
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, MellumForCausalLM
    from transformers.modeling_layers import MtpModel

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(TINY_TOKENIZER)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    # An output head of its own: with tied embeddings a random model mostly repeats its last
    # token, and drafts that only have to repeat a token test little.
    config = tiny_mellum_config(
        len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=pad,
        tie_word_embeddings=False,
    )
    model = MellumForCausalLM(config)
    init_parameters(model)
    mtp_model = MtpModel(model, 1)
    init_parameters(mtp_model.layers)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    weights = out / "model.safetensors"
    tensors = load_file(str(weights)) | mtp_checkpoint_weights(model, mtp_model)
    save_file(tensors, str(weights), metadata={"format": "pt"})
    return out


def two_layer_mtp(*, seed: int = 0, vocab_size: int = 128) -> tuple[Any, Any]:
    """``(policy, MtpModel)``: a random tiny Mellum and two random MTP layers on top of it, in
    memory only."""
    import torch
    from transformers import MellumForCausalLM
    from transformers.modeling_layers import MtpModel

    torch.manual_seed(seed)
    config = tiny_mellum_config(vocab_size, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    model = MellumForCausalLM(config).eval()
    init_parameters(model)
    # Mellum's config validation allows one MTP layer (its checkpoints hold one); the layer
    # classes do not care, and nothing here is saved.
    config.__dict__.update(
        num_nextn_predict_layers=2,
        mtp_layer_types=["sliding_attention"] * 2,
        mtp_mlp_layer_types=["sparse"] * 2,
    )
    mtp_model = MtpModel(model, 2).eval()
    init_parameters(mtp_model.layers)
    return model, mtp_model


def fit_to_greedy(
    layers: Any, prompts: list[list[int]], *, max_new_tokens: int = 16, steps: int = 300
) -> None:
    """Train ``layers`` (a ``monorl.mtp.MtpLayers``) until they predict the policy's own greedy
    continuations of ``prompts``, so that drafts for those prompts are accepted."""
    import torch
    from transformers import GenerationConfig

    model = layers.main_model
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
    )
    last_layer = [model.config.get_text_config().num_hidden_layers - 1]
    examples = []
    with torch.no_grad():
        for prompt in prompts:
            ids = model.generate(input_ids=torch.tensor([prompt]), generation_config=config)
            hidden = model(input_ids=ids, output_hidden_states=last_layer).hidden_states[-1]
            mask = torch.zeros_like(ids)
            mask[:, len(prompt) :] = 1
            examples.append((hidden, ids, mask))
    parameters = layers.parameters()
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=3e-3)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = sum(layers.loss(h, ids, mask, detach_trunk=True).loss for h, ids, mask in examples)
        loss.backward()
        optimizer.step()
    for parameter in parameters:
        parameter.requires_grad_(False)
