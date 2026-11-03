from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from monorl.config.schema import RunConfig
from monorl.core.interfaces import RolloutEngine
from monorl.core.types import SamplingParams
from monorl.rollout.hf_engine import HFRolloutEngine, _FinishRecorder, chat_prompt_ids

CHATS = [
    [{"role": "user", "content": "Say hello."}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "What is 2+2? Explain briefly."}],
]


def test_registry_and_protocol(engine: HFRolloutEngine) -> None:
    from monorl.core.registry import registry

    assert registry.get("rollout_engine", "hf").__name__ == "make_hf_engine"
    assert isinstance(engine, RolloutEngine)
    assert engine.tokenizer.pad_token_id is not None


def test_n_completions_order_and_prompt_ids(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=3, max_new_tokens=6, temperature=1.0, seed=1)
    out = engine.generate(CHATS, params)
    assert len(out) == 2
    for chat, group in zip(CHATS, out, strict=True):
        assert len(group) == 3
        expected = chat_prompt_ids(engine.tokenizer, chat)
        for c in group:
            assert c.prompt_token_ids == expected
            assert 1 <= len(c.token_ids) <= 6
            assert c.logprobs is not None and len(c.logprobs) == len(c.token_ids)
            assert c.finish_reason in ("stop", "length")
            assert all(lp <= 0.0 for lp in c.logprobs)
    # Different prompts condition on different ids, so groups are not interchangeable.
    assert out[0][0].prompt_token_ids != out[1][0].prompt_token_ids
    # Sampling at T=1 with a 49k vocab yields distinct completions within a group.
    assert len({tuple(c.token_ids) for c in out[0]}) > 1


def test_seed_is_deterministic(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=2, max_new_tokens=5, temperature=1.0, seed=123)
    a = engine.generate(CHATS, params)
    b = engine.generate(CHATS, params)
    assert [[c.token_ids for c in g] for g in a] == [[c.token_ids for c in g] for g in b]
    assert [[c.logprobs for c in g] for g in a] == [[c.logprobs for c in g] for g in b]


def test_logprobs_match_forward_pass_and_exclude_padding(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=2, max_new_tokens=7, temperature=1.0, seed=7, logprobs=True)
    out = engine.generate(CHATS, params)
    lengths = {len(chat_prompt_ids(engine.tokenizer, c)) for c in CHATS}
    assert len(lengths) == 2, "prompts must differ in length so the batch is left-padded"
    width = max(lengths)
    for chat, group in zip(CHATS, out, strict=True):
        for c in group:
            # Unpadded prompt ids are recorded even though the batch was left-padded to `width`.
            assert c.prompt_token_ids == chat_prompt_ids(engine.tokenizer, chat)
            assert len(c.prompt_token_ids) <= width
            assert len(c.logprobs) == len(c.token_ids)
            ids = torch.tensor([c.prompt_token_ids + c.token_ids])
            with torch.no_grad():
                logits = engine.model(input_ids=ids).logits[0].float()
            start = len(c.prompt_token_ids)
            ref = torch.log_softmax(logits[start - 1 : -1], dim=-1).gather(1, ids[0, start:, None]).squeeze(1)
            torch.testing.assert_close(torch.tensor(c.logprobs), ref, atol=1e-4, rtol=0.0)


def test_logprobs_are_temperature_scaled(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=1, max_new_tokens=4, temperature=0.5, seed=3)
    (c,) = engine.generate(CHATS[:1], params)[0]
    ids = torch.tensor([c.prompt_token_ids + c.token_ids])
    with torch.no_grad():
        logits = engine.model(input_ids=ids).logits[0].float()
    start = len(c.prompt_token_ids)
    ref = torch.log_softmax(logits[start - 1 : -1] / 0.5, dim=-1).gather(1, ids[0, start:, None]).squeeze(1)
    torch.testing.assert_close(torch.tensor(c.logprobs), ref, atol=1e-4, rtol=0.0)


def test_logprobs_disabled(engine: HFRolloutEngine) -> None:
    out = engine.generate(CHATS[:1], SamplingParams(n=1, max_new_tokens=3, logprobs=False, seed=0))
    assert out[0][0].logprobs is None


def test_raw_token_prompt_continues_exactly(engine: HFRolloutEngine) -> None:
    greedy = SamplingParams(n=1, max_new_tokens=8, temperature=0.0)
    (first,) = engine.generate(CHATS[1:], greedy)[0]
    assert len(first.token_ids) >= 4
    prefix = first.prompt_token_ids + first.token_ids[:3]
    (cont,) = engine.generate([prefix], SamplingParams(n=1, max_new_tokens=len(first.token_ids) - 3, temperature=0.0))[0]
    assert cont.prompt_token_ids == prefix
    assert cont.token_ids == first.token_ids[3:]


def test_greedy_replicates_n(engine: HFRolloutEngine) -> None:
    out = engine.generate(CHATS[:1], SamplingParams(n=3, max_new_tokens=4, temperature=0.0))
    assert len(out[0]) == 3
    assert len({tuple(c.token_ids) for c in out[0]}) == 1


def test_stop_string_sets_finish_reason(engine: HFRolloutEngine) -> None:
    sampled = SamplingParams(n=1, max_new_tokens=8, temperature=1.0, seed=11)
    (full,) = engine.generate(CHATS[:1], sampled)[0]
    assert full.finish_reason == "length" and len(full.token_ids) == 8
    # Stop on the text of the first few generated tokens: the same seed must now end early with "stop".
    stop_text = next(
        t for k in range(1, 7) if (t := engine.tokenizer.decode(full.token_ids[:k])).strip()
    )
    (stopped,) = engine.generate(
        CHATS[:1], SamplingParams(n=1, max_new_tokens=8, temperature=1.0, seed=11, stop=[stop_text])
    )[0]
    assert stopped.finish_reason == "stop"
    assert len(stopped.token_ids) < 8
    assert stopped.token_ids == full.token_ids[: len(stopped.token_ids)]
    assert len(stopped.logprobs) == len(stopped.token_ids)


def test_finish_recorder_eos_and_length() -> None:
    rec = _FinishRecorder(3, eos_ids=[2], stop=None)
    ids = torch.tensor([[5, 2], [5, 6], [2, 2]])
    rec(ids[:, :1], None)
    rec(ids, None)
    rec.finalize(2)
    assert rec.reason == ["stop", "length", "stop"]
    assert rec.num_tokens == [2, 2, 1]


def test_sampling_knobs_accepted(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=2, max_new_tokens=4, temperature=0.8, top_p=0.9, top_k=50, min_p=0.05, repetition_penalty=1.2, seed=5)
    out = engine.generate(CHATS, params)
    assert all(len(g) == 2 for g in out)
    assert all(len(c.logprobs) == len(c.token_ids) for g in out for c in g)


def _logits(engine: HFRolloutEngine, ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return engine.model(input_ids=torch.tensor([ids])).logits.float().clone()


def test_update_weights_and_load_from_path(tiny_config: RunConfig, tiny_model_dir: Path, tmp_path: Path) -> None:
    eng = HFRolloutEngine(tiny_config)
    probe = chat_prompt_ids(eng.tokenizer, CHATS[0])
    before = _logits(eng, probe)
    name = "model.layers.0.mlp.down_proj.weight"
    original = eng.model.state_dict()[name].clone()
    eng.update_weights([(name, original + 0.5)])
    assert not torch.allclose(_logits(eng, probe), before)
    with pytest.raises(KeyError):
        eng.update_weights([("model.layers.0.nope.weight", original)])
    with pytest.raises(ValueError):
        eng.update_weights([(name, original[:1])])
    eng.load_weights_from_path(tiny_model_dir)
    torch.testing.assert_close(_logits(eng, probe), before)
    # Weights exported by another model instance (HF names) load too.
    other = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
    eng.update_weights(other.state_dict().items())
    torch.testing.assert_close(_logits(eng, probe), before)
    eng.sleep()
    eng.wake_up()
    torch.testing.assert_close(_logits(eng, probe), before)
    eng.shutdown()


@pytest.mark.slow
def test_smollm2_generates_sensible_text(smol_model_id: str) -> None:
    config = RunConfig.from_dict(
        {
            "model": {"name_or_path": smol_model_id, "dtype": "float32"},
            "data": {"uri": "unused.jsonl"},
            "rollout": {"backend": "hf", "group_size": 2, "device": "cpu"},
            "rewards": [{"type": "exact_match"}],
        }
    )
    eng = HFRolloutEngine(config)
    try:
        out = eng.generate(
            [[{"role": "user", "content": "What is the capital of France? Answer in one word."}]],
            SamplingParams(n=1, max_new_tokens=24, temperature=0.0),
        )
    finally:
        eng.shutdown()
    (c,) = out[0]
    assert "paris" in c.text.lower()
    assert c.finish_reason == "stop"
    assert c.token_ids[-1] in (eng.tokenizer.eos_token_id,) or c.finish_reason == "stop"


def test_top_k_minus_one_disables_truncation(engine: HFRolloutEngine) -> None:
    """transformers backfills ``top_k=None`` with 50; the engine must pass 0 so sampling is
    full-vocabulary and matches the full-softmax logprobs it returns."""
    gc = engine._generation_config(SamplingParams(n=1, max_new_tokens=1, temperature=1.0, top_k=-1, min_p=0.0), 1)
    effective, _ = engine.model._prepare_generation_config(gc)
    assert effective.top_k == 0
    assert effective.min_p == 0.0
    gc = engine._generation_config(SamplingParams(n=1, max_new_tokens=1, temperature=1.0, top_k=7), 1)
    effective, _ = engine.model._prepare_generation_config(gc)
    assert effective.top_k == 7
