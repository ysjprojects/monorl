"""``HFRolloutEngine`` drafting with a model's MTP layers (``model.mtp.rollout``) on a tiny random
Mellum on CPU: completions are those of the engine without drafting (tokens, text, finish reasons,
logprobs) while one pass of the policy emits several tokens, sampled tokens and their logprobs are
the policy's, and pushed weights are the ones that draft."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, GenerationCancelled
from monorl.core.types import Completion, PromptInput, SamplingParams
from monorl.mtp import write_checkpoint_weights
from monorl.rollout.hf_engine import HFRolloutEngine, chat_prompt_ids
from monorl.rollout.media import load_processor
from tests.rollout.tiny_mtp import build_tiny_mtp, fit_to_greedy
from tests.rollout.tiny_vlm import image_chat

CHAT = [{"role": "user", "content": "Say hello."}]
PROMPTS: list[PromptInput] = [[1, 5, 9, 17, 33, 21], [1, 40, 41, 42], CHAT]
GREEDY = SamplingParams(n=1, max_new_tokens=16, temperature=0.0)


def _config(model_dir: Path, **mtp: Any) -> RunConfig:
    return RunConfig.from_dict(
        {
            "model": {"name_or_path": str(model_dir), "dtype": "float32", "mtp": mtp},
            "data": {"uri": "unused.jsonl"},
            "rollout": {"backend": "hf", "group_size": 2, "max_new_tokens": 8, "device": "cpu"},
            "trainer": {"backend": "reference", "device": "cpu", "precision": "fp32"},
            "rewards": [{"type": "exact_match"}],
        }
    )


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_mtp(tmp_path_factory.mktemp("tiny-mtp-engine"))


@pytest.fixture(scope="module")
def plain(model_dir: Path) -> Iterator[HFRolloutEngine]:
    """The same model without MTP: what every drafting engine has to reproduce."""
    engine = HFRolloutEngine(_config(model_dir))
    yield engine
    engine.shutdown()


@pytest.fixture(scope="module")
def fitted(model_dir: Path) -> Iterator[HFRolloutEngine]:
    """A drafting engine whose MTP layers know the policy's greedy continuations of ``PROMPTS``,
    so its drafts for them are accepted and a pass of the policy emits two tokens."""
    engine = HFRolloutEngine(_config(model_dir, enabled=True, rollout=True))
    prompts = [p if isinstance(p[0], int) else chat_prompt_ids(engine.tokenizer, p) for p in PROMPTS]
    fit_to_greedy(engine.mtp_layers, prompts, steps=50)
    yield engine
    engine.shutdown()


@pytest.fixture
def cold(model_dir: Path) -> Iterator[HFRolloutEngine]:
    """A drafting engine with the checkpoint's own (random) MTP layers."""
    engine = HFRolloutEngine(_config(model_dir, enabled=True, rollout=True, num_speculative_tokens=1))
    yield engine
    engine.shutdown()


def _assert_same(got: list[list[Completion]], expected: list[list[Completion]]) -> None:
    assert len(got) == len(expected)
    for group, expected_group in zip(got, expected, strict=True):
        assert len(group) == len(expected_group)
        for ours, theirs in zip(group, expected_group, strict=True):
            assert ours.token_ids == theirs.token_ids
            assert ours.text == theirs.text
            assert ours.finish_reason == theirs.finish_reason
            assert ours.prompt_token_ids == theirs.prompt_token_ids
            # float32 logprobs of order 10, computed over different shapes with and without drafts
            assert ours.logprobs == pytest.approx(theirs.logprobs, abs=1e-4)
            if theirs.top_logprobs is None:
                assert ours.top_logprobs is None
                continue
            assert ours.top_logprobs.k == theirs.top_logprobs.k
            assert ours.top_logprobs.token_ids == theirs.top_logprobs.token_ids
            assert list(ours.top_logprobs.logprobs) == pytest.approx(
                list(theirs.top_logprobs.logprobs), abs=1e-4
            )


def _num_tokens(completions: list[list[Completion]]) -> int:
    return sum(len(c.token_ids) for group in completions for c in group)


def _policy_logprobs(engine: HFRolloutEngine, completion: Completion) -> torch.Tensor:
    """``[tokens, vocab]``: the policy's log-probabilities at every generated position, from one
    teacher-forced forward pass over prompt and completion."""
    ids = torch.tensor([completion.prompt_token_ids + completion.token_ids])
    with torch.no_grad():
        logits = engine.model(input_ids=ids).logits[0].float()
    return torch.log_softmax(logits[len(completion.prompt_token_ids) - 1 : -1], dim=-1)


def test_greedy_completions_are_those_of_plain_decoding(
    plain: HFRolloutEngine, fitted: HFRolloutEngine
) -> None:
    params = SamplingParams(n=2, max_new_tokens=16, temperature=0.0, top_logprobs=3)
    expected = plain.generate(PROMPTS, params)
    fitted.speculation_stats()  # counters restart with every read
    got = fitted.generate(PROMPTS, params)
    _assert_same(got, expected)
    assert all(len(group) == 2 for group in got)

    stats = fitted.speculation_stats()
    # greedy decodes every prompt once, whatever n is; with accepted drafts in far fewer passes
    assert stats.emitted == sum(len(group[0].token_ids) for group in got) == 3 * 16
    assert stats.metrics()["mtp/accept_length"] > 1.5
    assert fitted.speculation_stats().steps == 0
    assert plain.speculation_stats() is None


def test_a_completion_ends_where_plain_decoding_ends(
    plain: HFRolloutEngine, fitted: HFRolloutEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pass of the policy emits two tokens here, and the stopping criteria see both at once:
    the token budget, a stop string and EOS all have to end the completion at the same token as
    decoding one token at a time."""
    continuation = plain.generate(PROMPTS, GREEDY)[0][0].token_ids

    for budget in (1, 2, 3):
        params = SamplingParams(n=1, max_new_tokens=budget, temperature=0.0)
        got = fitted.generate(PROMPTS, params)
        _assert_same(got, plain.generate(PROMPTS, params))
        assert all(len(c.token_ids) == budget and c.finish_reason == "length" for g in got for c in g)

    # Stop strings that are complete one token later each: some of them in the middle of what a
    # pass emitted, which then has to be cut.
    fitted.speculation_stats()
    lengths, cut_inside_a_pass = set(), False
    for k in (2, 3, 4):
        stop = fitted.tokenizer.decode(continuation[:k])
        params = SamplingParams(n=1, max_new_tokens=16, temperature=0.0, stop=[stop])
        expected = plain.generate(PROMPTS, params)
        got = fitted.generate(PROMPTS, params)
        _assert_same(got, expected)
        assert got[0][0].finish_reason == "stop" and len(got[0][0].token_ids) < 16
        lengths.add(len(got[0][0].token_ids))
        stats = fitted.speculation_stats()
        assert stats.emitted == _num_tokens(got)  # what was delivered, not what the passes produced
        cut_inside_a_pass |= stats.steps + stats.accepted > stats.emitted
    assert len(lengths) == 3 and cut_inside_a_pass

    # EOS: the first token of the continuation that differs from the ones before it.
    eos = next(token for token in continuation if token != continuation[0])
    for engine in (plain, fitted):
        monkeypatch.setattr(engine, "_eos_ids", [eos])
    expected = plain.generate(PROMPTS, GREEDY)
    got = fitted.generate(PROMPTS, GREEDY)
    _assert_same(got, expected)
    ended = got[0][0]
    assert ended.finish_reason == "stop" and ended.token_ids == continuation[: continuation.index(eos) + 1]
    assert 1 < len(ended.token_ids) < len(continuation)


@pytest.mark.parametrize("top_p", [1.0, 0.5])
def test_sampled_tokens_and_logprobs_are_the_policys(cold: HFRolloutEngine, top_p: float) -> None:
    prompts = [CHAT, PROMPTS[0]]
    params = SamplingParams(n=3, max_new_tokens=8, temperature=1.0, top_p=top_p, seed=5)
    out = cold.generate(prompts, params)
    stats = cold.speculation_stats()
    assert stats.accepted > 0 and stats.metrics()["mtp/accept_length"] > 1.1  # drafts were accepted

    assert [len(group) for group in out] == [3, 3]
    for prompt, group in zip(prompts, out, strict=True):
        for c in group:
            wanted = prompt if isinstance(prompt[0], int) else chat_prompt_ids(cold.tokenizer, prompt)
            assert c.prompt_token_ids == wanted
            assert 1 <= len(c.token_ids) <= 8 and len(c.logprobs) == len(c.token_ids)
            logprobs = _policy_logprobs(cold, c)
            tokens = torch.tensor(c.token_ids)[:, None]
            own = logprobs.gather(1, tokens)
            torch.testing.assert_close(torch.tensor(c.logprobs), own.squeeze(1), atol=1e-4, rtol=0.0)
            if top_p < 1.0:
                # Nucleus sampling drops the least likely tokens that together hold less than
                # 1 - top_p: a sampled token and the tokens no likelier than it hold more. A draft
                # the policy had not verified would fall outside for half of the tokens.
                probs = logprobs.exp()
                tail = torch.where(probs <= own.exp(), probs, 0.0).sum(dim=-1)
                assert bool((tail > 1.0 - top_p - 1e-4).all())


def test_rows_draw_from_one_seeded_stream(cold: HFRolloutEngine) -> None:
    """The rows are decoded one after another: no row repeats another, not even for the same
    prompt, and the same seed gives the same rows again."""
    prompts = [CHAT, PROMPTS[0], CHAT]
    params = SamplingParams(n=3, max_new_tokens=6, temperature=1.0, seed=7)
    out = cold.generate(prompts, params)
    assert [len(group) for group in out] == [3, 3, 3]
    assert len({tuple(c.token_ids) for group in (out[0], out[2]) for c in group}) == 6
    again = cold.generate(prompts, params)
    assert [[c.token_ids for c in g] for g in again] == [[c.token_ids for c in g] for g in out]
    assert [[c.logprobs for c in g] for g in again] == [[c.logprobs for c in g] for g in out]
    other = cold.generate(prompts, SamplingParams(n=3, max_new_tokens=6, temperature=1.0, seed=8))
    assert [[c.token_ids for c in g] for g in other] != [[c.token_ids for c in g] for g in out]


def test_pushed_weights_are_the_ones_that_draft(
    plain: HFRolloutEngine, fitted: HFRolloutEngine, cold: HFRolloutEngine
) -> None:
    expected = plain.generate(PROMPTS, GREEDY)
    fitted.speculation_stats()
    fitted.generate(PROMPTS, GREEDY)
    trained = fitted.speculation_stats()
    assert trained.accepted > 0.8 * trained.drafted

    # the checkpoint's random layers draft nothing the policy accepts
    _assert_same(cold.generate(PROMPTS, GREEDY), expected)
    assert cold.speculation_stats().accepted == 0
    name, weight = next(iter(cold.model.state_dict().items()))
    cold.update_weights([(name, torch.zeros_like(weight))])  # a policy that is out of date
    assert [g[0].token_ids for g in cold.generate(PROMPTS, GREEDY)] != [g[0].token_ids for g in expected]
    cold.speculation_stats()

    # What a trainer exports: the policy's weights, then the MTP layers' tagged as such.
    def export() -> Iterator[tuple[str, torch.Tensor]]:
        yield from fitted.model.state_dict().items()
        yield from fitted.mtp_layers.wire_weights()

    cold.update_weights(export())
    _assert_same(cold.generate(PROMPTS, GREEDY), expected)
    assert cold.speculation_stats() == trained

    # a push of the policy alone leaves the drafter as it is
    cold.update_weights(fitted.model.state_dict().items())
    _assert_same(cold.generate(PROMPTS, GREEDY), expected)
    assert cold.speculation_stats() == trained

    # MTP layers that would be a mixture of two pushes are refused; so are names nobody knows
    with pytest.raises(KeyError, match="incomplete"):
        cold.update_weights(list(export())[:-1])
    unknown = [("model.layers.0.nope.weight", torch.zeros(1))]
    with pytest.raises(KeyError, match="nope"):
        cold.update_weights(unknown)

    # An engine that does not draft takes the same stream and drops the MTP layers' weights.
    plain.update_weights(export())
    _assert_same(plain.generate(PROMPTS, GREEDY), expected)
    with pytest.raises(KeyError, match="nope"):
        plain.update_weights(unknown)


def test_weights_loaded_from_a_directory_reach_policy_and_drafter(
    plain: HFRolloutEngine, fitted: HFRolloutEngine, cold: HFRolloutEngine, model_dir: Path, tmp_path: Path
) -> None:
    expected = plain.generate(PROMPTS, GREEDY)
    export = tmp_path / "export"
    fitted.model.save_pretrained(export)
    write_checkpoint_weights(
        export, fitted.mtp_layers.checkpoint_weights(tied_from=fitted.model.state_dict())
    )

    # The directory stores the policy in the checkpoint's layout too (one tensor per expert):
    # every tensor of an engine that knows nothing any more has to come back from it.
    cold.update_weights([(name, torch.zeros_like(w)) for name, w in cold.model.state_dict().items()])
    assert [g[0].token_ids for g in cold.generate(PROMPTS, GREEDY)] != [g[0].token_ids for g in expected]
    cold.speculation_stats()
    cold.load_weights_from_path(export)
    _assert_same(cold.generate(PROMPTS, GREEDY), expected)
    stats = cold.speculation_stats()
    assert stats.accepted > 0.8 * stats.drafted

    cold.load_weights_from_path(model_dir)  # back to the checkpoint's random layers
    _assert_same(cold.generate(PROMPTS, GREEDY), expected)
    assert cold.speculation_stats().accepted == 0

    # a directory with MTP weights is a valid source for an engine that does not draft
    plain.load_weights_from_path(export)
    _assert_same(plain.generate(PROMPTS, GREEDY), expected)

    # names that belong to nothing are still refused
    write_checkpoint_weights(export, [("model.layers.0.nope.weight", torch.zeros(1))])
    for engine in (cold, plain):
        with pytest.raises(KeyError, match="nope"):
            engine.load_weights_from_path(export)


def test_cancel_stops_a_drafting_call_within_a_step(
    cold: HFRolloutEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    short = SamplingParams(n=2, max_new_tokens=6, temperature=1.0, seed=11)
    before = cold.generate([CHAT], short)

    decoding = threading.Event()
    model_generate = cold.model.generate

    def spy(*args: Any, **kwargs: Any) -> Any:
        decoding.set()
        return model_generate(*args, **kwargs)

    monkeypatch.setattr(cold.model, "generate", spy)
    # Three rows decoded one after another, each of them seconds long.
    long = SamplingParams(n=3, max_new_tokens=2000, temperature=1.0, seed=3)
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["result"] = cold.generate([CHAT], long)
        except BaseException as e:  # noqa: BLE001 - recorded for the assertions
            outcome["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert decoding.wait(30)
    time.sleep(0.1)  # a few decoding steps in
    t0 = time.monotonic()
    cold.cancel()
    thread.join(30)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), GenerationCancelled), outcome
    assert time.monotonic() - t0 < 3.0

    after = cold.generate([CHAT], short)
    assert [(c.token_ids, c.logprobs) for c in after[0]] == [(c.token_ids, c.logprobs) for c in before[0]]


def test_what_the_engine_cannot_draft_is_refused(
    cold: HFRolloutEngine,
    model_dir: Path,
    tiny_model_dir: Path,
    vlm_config: RunConfig,
    test_images: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # one token per MTP layer, and this checkpoint has one layer
    with pytest.raises(ConfigError, match="num_speculative_tokens=2"):
        HFRolloutEngine(_config(model_dir, enabled=True, rollout=True, num_speculative_tokens=2))
    with pytest.raises(ConfigError, match="no MTP layers"):
        HFRolloutEngine(_config(tiny_model_dir, enabled=True, rollout=True))

    # prompts with images, as a vision-language checkpoint with MTP layers would receive them
    monkeypatch.setattr(cold, "_processor", load_processor(vlm_config))
    with pytest.raises(ValueError, match="text prompts only"):
        cold.generate([image_chat(test_images["red"])], GREEDY)
    assert len(cold.generate([CHAT], GREEDY)[0][0].token_ids) == 16
