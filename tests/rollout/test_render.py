"""Token-level turn continuation (``monorl.rollout.render``) against a real BPE tokenizer.

The SmolLM tokenizer ships a ChatML template; the other templates below reproduce, in a few lines
each, behaviours of popular chat templates that matter for continuation: trimming content
(Llama 3, Gemma), dropping the reasoning of earlier turns (Qwen3, DeepSeek-R1) and rendering
structured tool calls (Qwen/Hermes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from transformers import AutoTokenizer

from monorl.core.types import Completion
from monorl.rollout.render import SENTINEL, chat_prompt_ids, continue_prompt_ids

TRIMMING = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' + m['role'] + '\n' + (m['content'] | trim) + '<|im_end|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)

# Reasoning is kept for assistant turns after the last user message and dropped before it.
REASONING_DROPPING = (
    "{%- set ns = namespace(last_query=-1) -%}"
    "{%- for m in messages -%}{%- if m['role'] == 'user' -%}{%- set ns.last_query = loop.index0 -%}{%- endif -%}"
    "{%- endfor -%}"
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'assistant' and loop.index0 < ns.last_query -%}"
    "{{- '<|im_start|>assistant\n' + (m['content'].split('</think>')[-1] | trim) + '<|im_end|>\n' -}}"
    "{%- else -%}"
    "{{- '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)

TOOL_CALLING = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' + m['role'] + '\n' + (m['content'] or '') -}}"
    "{%- for call in m['tool_calls'] or [] -%}"
    "{{- '\n<tool_call>\n' + (call['function'] | tojson) + '\n</tool_call>' -}}"
    "{%- endfor -%}"
    "{{- '<|im_end|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)

# An assistant turn with tool calls is rendered from the calls alone; its content is left out
# (Llama 3.1, Mistral).
CALL_ONLY = (
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'assistant' and m['tool_calls'] -%}"
    "{{- '<|im_start|>assistant\n' + (m['tool_calls'][0]['function'] | tojson) + '<|im_end|>\n' -}}"
    "{%- else -%}"
    "{{- '<|im_start|>' + m['role'] + '\n' + (m['content'] or '') + '<|im_end|>\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)

USER = [{"role": "user", "content": "What is 2+2?"}]
FOLLOW_UP = [{"role": "user", "content": "And 3+3?"}]


@pytest.fixture
def tok(tiny_model_dir: Path) -> Any:
    return AutoTokenizer.from_pretrained(tiny_model_dir)


def _ids(tok: Any, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def _completion(tok: Any, text: str, *, finish: str = "stop", ids: list[int] | None = None) -> Completion:
    ids = _ids(tok, text) if ids is None else ids
    if finish == "stop":
        ids = [*ids, tok.convert_tokens_to_ids("<|im_end|>")]
    return Completion(text=text, token_ids=ids, logprobs=None, finish_reason=finish)


def _continue(tok: Any, completion: Completion, new: list[dict[str, Any]] = FOLLOW_UP, **kwargs: Any) -> Any:
    tools = kwargs.get("tools")
    return continue_prompt_ids(
        tok,
        previous_messages=USER,
        previous_prompt_ids=chat_prompt_ids(tok, USER, tools),
        completion=completion,
        new_messages=new,
        **kwargs,
    )


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_continuation_equals_the_template_render_when_the_text_round_trips(tok: Any, finish: str) -> None:
    """A completion that ended with the end-of-turn token and one cut off by the length limit both
    continue into exactly what the chat template renders: the marker is neither doubled nor lost."""
    completion = _completion(tok, "The answer is 4.", finish=finish)
    out = _continue(tok, completion)
    assert out.bridged
    assert out.ids == chat_prompt_ids(
        tok, [*USER, {"role": "assistant", "content": completion.text}, *FOLLOW_UP]
    )


def test_sampled_tokens_stay_in_the_context_when_the_text_does_not_round_trip(tok: Any) -> None:
    """The same text tokenised the way the policy happened to sample it (not how the tokenizer
    would encode it) stays as sampled; re-rendering the chat replaces it."""
    text = "The answer is four."
    sampled = _ids(tok, "The ans") + _ids(tok, "wer is fo") + _ids(tok, "ur.")
    assert sampled != _ids(tok, text) and tok.decode(sampled) == text
    completion = _completion(tok, text, ids=sampled)
    prompt = chat_prompt_ids(tok, USER)
    sampled_context = prompt + completion.token_ids

    bridged = _continue(tok, completion)
    assert bridged.bridged and bridged.ids[: len(sampled_context)] == sampled_context
    assert tok.decode(bridged.ids[len(sampled_context) :]) == (
        "\n<|im_start|>user\nAnd 3+3?<|im_end|>\n<|im_start|>assistant\n"
    )

    rendered = _continue(tok, completion, mode="template")
    assert not rendered.bridged
    assert rendered.ids[: len(sampled_context)] != sampled_context
    assert rendered.ids == chat_prompt_ids(tok, [*USER, {"role": "assistant", "content": text}, *FOLLOW_UP])


def test_a_completion_ending_in_another_special_token_still_gets_its_turn_closed(tok: Any) -> None:
    other = tok.convert_tokens_to_ids("<|endoftext|>")
    end = tok.convert_tokens_to_ids("<|im_end|>")
    assert other != end
    completion = Completion(
        text="Four.", token_ids=[*_ids(tok, "Four."), other], logprobs=None, finish_reason="stop"
    )
    out = _continue(tok, completion)
    after = out.ids[len(chat_prompt_ids(tok, USER)) + len(completion.token_ids) :]
    assert out.bridged and after[0] == end


def test_whitespace_the_template_trims_is_kept_as_sampled(tok: Any) -> None:
    tok.chat_template = TRIMMING
    completion = _completion(tok, " The answer is 4.\n")
    out = _continue(tok, completion)
    context = chat_prompt_ids(tok, USER) + completion.token_ids
    assert out.bridged and out.ids[: len(context)] == context
    # the template itself would have dropped the sampled whitespace tokens
    full = chat_prompt_ids(tok, [*USER, {"role": "assistant", "content": completion.text}, *FOLLOW_UP])
    assert full[: len(context)] != context


def test_a_template_that_drops_earlier_reasoning_is_followed(tok: Any) -> None:
    """Token continuation must not change what the context says: once a new user message makes
    the template drop the previous turn's reasoning, the prompt is the template's rendering."""
    tok.chat_template = REASONING_DROPPING
    completion = _completion(tok, "<think>\n2+2 is 4\n</think>\n\nThe answer is 4.")
    out = _continue(tok, completion)
    assert not out.bridged
    assert out.ids == chat_prompt_ids(
        tok, [*USER, {"role": "assistant", "content": completion.text}, *FOLLOW_UP]
    )
    assert "2+2 is 4" not in tok.decode(out.ids)

    # Inside a tool loop (no new user message) the same template keeps the reasoning: continue.
    tool_result = [{"role": "tool", "content": "4"}]
    kept = _continue(tok, completion, tool_result)
    context = chat_prompt_ids(tok, USER) + completion.token_ids
    assert kept.bridged and kept.ids[: len(context)] == context


def test_reasoning_already_in_the_context_is_not_kept_once_the_template_drops_it(tok: Any) -> None:
    """Turn 1 reasons inside a tool loop (kept), turn 2 answers plainly, then the user speaks
    again: the template now drops turn 1's reasoning, so the previous prompt cannot be extended
    even though turn 2 itself would render unchanged."""
    tok.chat_template = REASONING_DROPPING
    first = "<think>\nuse the tool\n</think>\n\ncalc(2+2)"
    before_second = [*USER, {"role": "assistant", "content": first}, {"role": "tool", "content": "4"}]
    second = _completion(tok, "The answer is 4.")
    out = continue_prompt_ids(
        tok,
        previous_messages=before_second,
        previous_prompt_ids=chat_prompt_ids(tok, before_second),
        completion=second,
        new_messages=FOLLOW_UP,
    )
    assert not out.bridged
    assert "use the tool" not in tok.decode(out.ids)


def test_structured_tool_calls_keep_the_sampled_tokens(tok: Any) -> None:
    """The gateway hands back the assistant turn as structured ``tool_calls``; the template would
    re-serialise them (other JSON spacing), the continuation keeps what the policy sampled."""
    tok.chat_template = TOOL_CALLING
    sampled_text = '<tool_call>\n{"name":"calc","arguments":{"expression":"2+2"}}\n</tool_call>'
    completion = _completion(tok, sampled_text)
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": "calc", "arguments": {"expression": "2+2"}}}
        ],
    }
    tool_result = [{"role": "tool", "content": "4"}]
    out = _continue(tok, completion, tool_result, assistant_message=assistant)
    context = chat_prompt_ids(tok, USER) + completion.token_ids
    assert out.bridged and out.ids[: len(context)] == context
    assert tok.decode(out.ids[len(context) :]) == "\n<|im_start|>tool\n4<|im_end|>\n<|im_start|>assistant\n"
    # the template's own serialisation of the call differs from the sampled one
    assert '{"name": "calc"' in tok.decode(chat_prompt_ids(tok, [*USER, assistant, *tool_result]))


def test_a_tool_call_turn_rendered_without_its_content_is_continued_when_it_had_none(tok: Any) -> None:
    tok.chat_template = CALL_ONLY
    call = {"type": "function", "function": {"name": "calc", "arguments": {"expression": "2+2"}}}
    tool_result = [{"role": "tool", "content": "4"}]
    completion = _completion(tok, '{"name":"calc","arguments":{"expression":"2+2"}}')
    context = chat_prompt_ids(tok, USER) + completion.token_ids

    bare = {"role": "assistant", "content": "", "tool_calls": [call]}
    out = _continue(tok, completion, tool_result, assistant_message=bare)
    assert out.bridged and out.ids[: len(context)] == context
    assert tok.decode(out.ids[len(context) :]) == "\n<|im_start|>tool\n4<|im_end|>\n<|im_start|>assistant\n"

    # Text next to the call would be dropped by this template: follow it instead of keeping it.
    wordy = {"role": "assistant", "content": "Let me compute.", "tool_calls": [call]}
    out = _continue(tok, _completion(tok, "Let me compute."), tool_result, assistant_message=wordy)
    assert not out.bridged
    assert out.ids == chat_prompt_ids(tok, [*USER, wordy, *tool_result])


class _RejectsProbe:
    """Chat template that refuses the sentinel probe, like a template with strict validation."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, text: str, **kwargs: Any) -> Any:
        return self._tokenizer(text, **kwargs)

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        if any(m["content"] == SENTINEL for m in messages):
            raise ValueError("unexpected content")
        return self._tokenizer.apply_chat_template(messages, **kwargs)


def test_a_template_that_rejects_the_probe_falls_back_to_a_full_render(tok: Any) -> None:
    strict = _RejectsProbe(tok)
    completion = _completion(tok, "The answer is 4.")
    out = _continue(strict, completion)
    assert not out.bridged
    assert out.ids == chat_prompt_ids(
        tok, [*USER, {"role": "assistant", "content": completion.text}, *FOLLOW_UP]
    )


def test_unknown_mode_is_an_error(tok: Any) -> None:
    with pytest.raises(ValueError, match="turn rendering"):
        _continue(tok, _completion(tok, "4"), mode="bogus")
