from __future__ import annotations

import torch

from monorl.core.types import SamplingParams
from monorl.rollout.hf_engine import HFRolloutEngine

CHATS = [
    [{"role": "user", "content": "Say hello."}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "What is 2+2?"}],
]


def test_top_logprobs_match_a_recomputation_from_the_model(engine: HFRolloutEngine) -> None:
    temperature, k = 0.05, 5  # peaked enough that the top-k order is far from float noise
    params = SamplingParams(n=2, max_new_tokens=6, temperature=temperature, seed=5, top_logprobs=k)
    for group in engine.generate(CHATS, params):
        for c in group:
            top = c.top_logprobs
            assert len(top) == len(c.token_ids) > 0 and top.k == k
            ids = torch.tensor([c.prompt_token_ids + c.token_ids])
            with torch.no_grad():
                logits = engine.model(input_ids=ids).logits[0].float()
            start = len(c.prompt_token_ids)
            logprobs = torch.log_softmax(logits[start - 1 : -1] / temperature, dim=-1)
            values, indices = logprobs.topk(k, dim=-1)
            assert torch.equal(torch.tensor(top.token_ids).view(-1, k), indices)
            captured = torch.tensor(top.logprobs).view(-1, k)
            torch.testing.assert_close(captured, values, atol=1e-3, rtol=0.0)
            # same distribution as the sampled-token logprobs, whatever token was sampled
            sampled = torch.tensor(c.token_ids)[:, None] == indices
            assert sampled.any()
            torch.testing.assert_close(
                captured[sampled], torch.tensor(c.logprobs)[sampled.any(dim=1)], atol=1e-6, rtol=0.0
            )


def test_top_logprobs_are_only_captured_on_request(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=1, max_new_tokens=4, temperature=1.0, seed=5)
    assert all(c.top_logprobs is None for group in engine.generate(CHATS, params) for c in group)
    # without sampled-token logprobs the top-k is still available
    params = SamplingParams(n=1, max_new_tokens=4, temperature=1.0, seed=5, logprobs=False, top_logprobs=3)
    for group in engine.generate(CHATS, params):
        for c in group:
            assert c.logprobs is None
            assert c.top_logprobs.k == 3 and len(c.top_logprobs) == len(c.token_ids)
