from __future__ import annotations

import pytest

from monorl.core.registry import registry
from monorl.rewards.reward_model import render_messages


@pytest.fixture(scope="module")
def tiny_rm(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import AutoTokenizer, LlamaConfig, LlamaForSequenceClassification

    from tests.conftest import TINY_TOKENIZER

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(TINY_TOKENIZER)
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        num_labels=1,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
    )
    model = LlamaForSequenceClassification(cfg)
    out = tmp_path_factory.mktemp("tiny-rm")
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    assert transformers.__version__
    return out


def test_reward_model_scores_last_assistant_turn(traj, tiny_rm):
    fn = registry.get("reward", "reward_model")({"model": str(tiny_rm), "batch_size": 2})
    trajs = [traj("The answer is 4.", "4"), traj("I refuse.", "4"), traj("Four, definitely.", "4")]
    results = fn(trajs)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert all(isinstance(s, float) for s in scores)
    assert results[0].breakdown == {"reward_model": scores[0]}
    # deterministic & independent of batch composition
    assert fn(trajs[:1])[0].score == pytest.approx(scores[0], abs=1e-4)
    # changing the response changes the score
    assert scores[0] != scores[1]


def test_reward_model_sigmoid_and_injected_model(traj, tiny_rm):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_rm)
    model = AutoModelForSequenceClassification.from_pretrained(tiny_rm)
    fn = registry.get("reward", "reward_model")({"normalize": "sigmoid"}, model=model, tokenizer=tok)
    [res] = fn([traj("hi")])
    assert 0.0 < res.score < 1.0


def test_render_messages_uses_chat_template_or_plain_join(tiny_rm):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_rm)
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    rendered = render_messages(tok, msgs)
    assert "hello" in rendered and "hi" in rendered

    class NoTemplate:
        chat_template = None

    assert render_messages(NoTemplate(), msgs) == "user: hi\nassistant: hello"


def test_reward_model_requires_model_param():
    with pytest.raises(ValueError):
        registry.get("reward", "reward_model")({})
