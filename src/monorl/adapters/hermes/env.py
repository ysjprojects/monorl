"""``env: hermes`` — a Hermes Agent (hermes-agent 0.19) episode where the LLM is *our* policy.

Hermes' ``AIAgent.run_conversation`` owns the loop "call LLM → parse tool calls → execute tools →
repeat". For RL the policy must be the orchestrator's rollout engine, so this module inverts
control by stepping Hermes' loop manually instead of running it: Hermes never calls an LLM, no
API key is used.

Hermes modules/classes used (all from the installed ``hermes-agent`` distribution, whose
top-level import names are ``run_agent``, ``agent``, ``tools``, ``model_tools``, ``toolsets``,
``hermes_constants``):

* ``run_agent.AIAgent`` — constructed once per env with ``provider="custom"`` and an unreachable
  ``base_url`` (its OpenAI client is created lazily and never used). We use:
  ``AIAgent.tools`` (OpenAI tool schemas resolved from ``enabled_toolsets``),
  ``AIAgent.valid_tool_names``, ``AIAgent._build_system_prompt(system_message)``
  (``agent.system_prompt.build_system_prompt``: identity, tool guidance, skills index, environment
  hints), ``AIAgent._repair_tool_call`` (``agent.agent_runtime_helpers.repair_tool_call``),
  ``AIAgent._execute_tool_calls(assistant_message, messages, task_id)`` (the real tool runtime:
  ``agent.tool_executor.execute_tool_calls_{sequential,concurrent,segmented}`` →
  ``model_tools.handle_function_call`` → ``tools.registry.registry.dispatch``),
  ``AIAgent._cleanup_task_resources`` and ``AIAgent.close``.
* ``agent.conversation_loop._invalid_tool_name_error_content`` — the exact error text Hermes'
  loop feeds back for hallucinated tool names.
* ``tools.registry.registry`` + ``toolsets.create_custom_toolset`` / ``toolsets.validate_toolset``
  — tool allowlist resolution; ``params["tool_modules"]`` are imported so custom tools registered
  through ``registry.register(...)`` become selectable by name.
* ``tools.terminal_tool.register_task_env_overrides`` / ``clear_task_env_overrides`` /
  ``cleanup_vm`` — Hermes' own per-rollout sandbox hooks (documented upstream as the hooks RL
  environments use): per-episode ``cwd`` and container image.
* ``tools.tool_result_storage.maybe_persist_tool_result`` + ``tools.budget_config.BudgetConfig``
  — Hermes' tool-output truncation, driven by ``params["max_tool_output_chars"]``.
* ``tools.skills_tool.skill_view`` / ``agent.skill_utils.parse_frontmatter`` — skill bodies
  injected into the system prompt for ``params["skills"]``.
* ``hermes_constants.set_hermes_home_override`` — context-local ``HERMES_HOME`` for
  ``params["hermes_home"]``.
* ``openai.types.chat.ChatCompletionMessage`` / ``ChatCompletionMessageToolCall`` — the response
  object shape Hermes' executor expects (hermes-agent depends on the ``openai`` SDK).

Episode protocol:

* ``reset(sample)`` → ``[system (Hermes prompt), *sample.messages]``; ``tools`` exposes the
  schemas for the chat template.
* ``step(completion)``: tool calls are parsed from the assistant text with
  :func:`monorl.envs.tool_calling.parse_tool_calls` (``<tool_call>{...}</tool_call>`` or JSON),
  validated like Hermes' loop does (name repair, unknown-name errors), executed through Hermes'
  executor; the resulting ``tool`` messages are returned. ``done`` when the assistant answers
  without tool calls (Hermes' "final response") or after ``max_turns``.
* ``close()`` tears down the episode's sandbox/task resources and the agent.

Reward: Hermes has no task scoring API, so ``StepResult.reward`` is ``0`` except for
``-invalid_call_penalty`` on turns with malformed/unknown calls; reward functions score the
trajectory on top.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from monorl.core.errors import BackendUnavailable
from monorl.core.types import Completion, Message, Sample, StepResult, ToolCall, message_text
from monorl.envs.tool_calling import parse_tool_calls

log = logging.getLogger("monorl.adapters.hermes")

DEFAULT_TOOLS: tuple[str, ...] = ("file", "terminal")
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 8000
SANDBOX_BACKENDS: frozenset[str] = frozenset({"local", "docker", "singularity", "modal", "daytona", "ssh"})
_IMAGE_KEY_FOR_BACKEND = {
    "docker": "docker_image",
    "singularity": "singularity_image",
    "modal": "modal_image",
    "daytona": "daytona_image",
}

COMPACT_SYSTEM_PROMPT = (
    "You are Hermes Agent, a tool-using assistant. Solve the user's task using the tools below. "
    "To call a tool, reply with one or more blocks of the form\n"
    '<tool_call>{"name": "<tool name>", "arguments": {...}}</tool_call>\n'
    "Tool results come back as tool messages. When the task is complete, answer the user "
    "directly without any tool call.\n\nTools:\n{tools}"
)


def _require_hermes() -> None:
    try:
        import run_agent  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise BackendUnavailable(
            "env 'hermes' needs the 'hermes-agent' package (pip install monorl[hermes])"
        ) from e


# -- params ------------------------------------------------------------------------------------


def _str_list(value: Any, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(f"env.params.{key} must be a list of strings, got {value!r}")


def resolve_tool_names(tools: Sequence[str], *, tool_modules: Sequence[str] = ()) -> list[str]:
    """Resolve mixed Hermes tool / toolset names to a sorted list of registered tool names.

    ``tool_modules`` are imported first so tools they register (``tools.registry.registry.register``)
    are selectable. Unknown names raise ``ValueError`` listing what exists.
    """
    _require_hermes()
    for mod in tool_modules:
        importlib.import_module(mod)
    import model_tools  # discovers builtin tools at import
    from toolsets import resolve_toolset, validate_toolset

    registered = set(model_tools.get_all_tool_names())
    out: set[str] = set()
    unknown: list[str] = []
    for name in tools:
        if name in registered:
            out.add(name)
        elif validate_toolset(name):
            out.update(t for t in resolve_toolset(name) if t in registered)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"env.params.tools: unknown Hermes tool/toolset names {unknown}; "
            f"tools: {sorted(registered)}; toolsets: {sorted(__import__('toolsets').get_toolset_names())}"
        )
    return sorted(out)


def custom_toolset_for(tool_names: Sequence[str]) -> str:
    """Register (idempotently) a Hermes toolset containing exactly ``tool_names`` and return its name."""
    import hashlib

    from toolsets import TOOLSETS, create_custom_toolset

    names = sorted(set(tool_names))
    digest = hashlib.sha1(",".join(names).encode()).hexdigest()[:12]
    toolset = f"monorl-{digest}"
    if toolset not in TOOLSETS:
        create_custom_toolset(toolset, "monorl HermesEnv tool allowlist", tools=names)
    return toolset


def load_skill_prompts(names: Sequence[str], skills_dir: str | os.PathLike[str] | None = None) -> list[str]:
    """Return the body (frontmatter stripped) of each named skill's ``SKILL.md``.

    With ``skills_dir`` the directory is searched directly (``<dir>/**/<name>/SKILL.md``);
    otherwise Hermes' own resolution (``tools.skills_tool.skill_view``) is used.
    """
    if not names:
        return []
    _require_hermes()
    from agent.skill_utils import parse_frontmatter

    out: list[str] = []
    if skills_dir is not None:
        root = Path(skills_dir).expanduser()
        for name in names:
            matches = sorted(p for p in root.glob(f"**/{name}/SKILL.md") if p.is_file())
            if not matches:
                raise ValueError(f"env.params.skills: skill {name!r} not found under {root}")
            _front, body = parse_frontmatter(matches[0].read_text(encoding="utf-8"))
            out.append(f"# Skill: {name}\n{body.strip()}")
        return out
    from tools.skills_tool import skill_view

    for name in names:
        result = json.loads(skill_view(name, preprocess=False))
        if not result.get("success"):
            raise ValueError(f"env.params.skills: {result.get('error') or f'skill {name!r} not found'}")
        _front, body = parse_frontmatter(result.get("content") or "")
        out.append(f"# Skill: {name}\n{body.strip()}")
    return out


def default_workspace_root() -> Path:
    """``$MONORL_HOME/hermes-workspaces`` (``~/.monorl`` by default): parent of per-episode workspaces."""
    return Path(os.environ.get("MONORL_HOME") or "~/.monorl").expanduser() / "hermes-workspaces"


def sandbox_overrides(sandbox: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Validate ``params["sandbox"]`` → ``(backend, hermes task override keys)``."""
    sandbox = dict(sandbox or {})
    backend = str(sandbox.pop("backend", "local") or "local").lower()
    if backend not in SANDBOX_BACKENDS:
        raise ValueError(
            f"env.params.sandbox.backend must be one of {sorted(SANDBOX_BACKENDS)}, got {backend!r}"
        )
    image = sandbox.pop("image", None)
    if sandbox:
        raise ValueError(f"env.params.sandbox: unknown keys {sorted(sandbox)}; expected backend, image")
    overrides: dict[str, Any] = {}
    if image:
        key = _IMAGE_KEY_FOR_BACKEND.get(backend)
        if key is None:
            raise ValueError(f"env.params.sandbox.image is not supported by the {backend!r} backend")
        overrides[key] = str(image)
    if backend != "local":
        overrides["env_type"] = backend  # isolation signal: Hermes keys the sandbox by task id
    return backend, overrides


# -- message shaping ---------------------------------------------------------------------------


def to_openai_message(text: str, calls: Sequence[ToolCall]) -> Any:
    """Build the ``ChatCompletionMessage`` Hermes' executor expects from parsed calls."""
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

    tool_calls = [
        ChatCompletionMessageToolCall(
            id=str(c["id"]),
            type="function",
            function=Function(
                name=str(c["name"]), arguments=json.dumps(c.get("arguments") or {}, ensure_ascii=False)
            ),
        )
        for c in calls
    ]
    return ChatCompletionMessage(role="assistant", content=text, tool_calls=tool_calls or None)


def project_tool_message(msg: Mapping[str, Any]) -> Message:
    """Reduce Hermes' internal tool-result dict (``tool_name``, ``_tool_output_risk``...) to a chat message."""
    content = msg.get("content", "")
    if not isinstance(content, str):  # multimodal envelope: keep the text parts
        parts = content if isinstance(content, list) else [content]
        content = "\n".join(str(p.get("text", "")) if isinstance(p, Mapping) else str(p) for p in parts)
    out: Message = {"role": "tool", "content": content}
    if msg.get("name"):
        out["name"] = str(msg["name"])
    if msg.get("tool_call_id"):
        out["tool_call_id"] = str(msg["tool_call_id"])
    return out


# -- environment ---------------------------------------------------------------------------------


class HermesEnv:
    """monorl Environment stepping a Hermes ``AIAgent`` manually (see module docstring)."""

    def __init__(
        self,
        *,
        tools: Sequence[str] = DEFAULT_TOOLS,
        tool_modules: Sequence[str] = (),
        skills: Sequence[str] = (),
        skills_dir: str | None = None,
        hermes_home: str | None = None,
        workspace_dir: str | None = None,
        sandbox: Mapping[str, Any] | None = None,
        max_turns: int = 8,
        max_tool_output_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
        max_calls_per_turn: int = 8,
        invalid_call_penalty: float = 0.0,
        system_prompt: str | None = None,
        hermes_system_prompt: bool = True,
        model_name: str = "policy",
    ) -> None:
        _require_hermes()
        self.max_turns = max(1, int(max_turns))
        self.max_tool_output_chars = max(1, int(max_tool_output_chars))
        self.max_calls_per_turn = max(1, int(max_calls_per_turn))
        self.invalid_call_penalty = float(invalid_call_penalty)
        self.system_prompt = system_prompt
        self.hermes_system_prompt = bool(hermes_system_prompt)
        self.model_name = model_name or "policy"
        self.hermes_home = str(Path(hermes_home).expanduser()) if hermes_home else None
        self.workspace_dir = str(Path(workspace_dir).expanduser()) if workspace_dir else None
        self.sandbox_backend, self._sandbox_overrides = sandbox_overrides(sandbox)
        if self.sandbox_backend != "local":
            # Hermes reads the terminal backend from the process environment only.
            if os.environ.get("TERMINAL_ENV", "local") != self.sandbox_backend:
                log.info("hermes env: setting TERMINAL_ENV=%s (process-wide)", self.sandbox_backend)
            os.environ["TERMINAL_ENV"] = self.sandbox_backend

        with self._hermes_home_scope():
            self.tool_names = resolve_tool_names(
                list(tools) or list(DEFAULT_TOOLS), tool_modules=tool_modules
            )
            self.skill_prompts = load_skill_prompts(list(skills), skills_dir)
            self.agent = self._make_agent()
        self.tools: list[dict[str, Any]] = list(self.agent.tools)
        self.valid_tool_names: set[str] = set(self.agent.valid_tool_names)
        self._system_prompt_cache: str | None = None

        self._sample: Sample | None = None
        self._turn = 0
        self._task_id: str | None = None
        self._episode_dir: str | None = None
        self.cwd: str | None = None
        self._messages: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self._closed = False

    # -- hermes plumbing --------------------------------------------------------------------

    @contextmanager
    def _hermes_home_scope(self) -> Iterator[None]:
        """Context-local ``HERMES_HOME`` override (Hermes' ``set_hermes_home_override``) when configured."""
        if self.hermes_home is None:
            yield
            return
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        token = set_hermes_home_override(self.hermes_home)
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    def _make_agent(self) -> Any:
        from run_agent import AIAgent

        return AIAgent(
            base_url="http://127.0.0.1:9/v1",  # never contacted: the policy is external
            api_key="monorl-no-key",
            provider="custom",
            model=self.model_name,
            max_iterations=self.max_turns,
            tool_delay=0.0,
            enabled_toolsets=[custom_toolset_for(self.tool_names)],
            quiet_mode=True,
            tool_progress_mode="off",
            skip_context_files=True,
            skip_memory=True,
            save_trajectories=False,
        )

    @property
    def turn(self) -> int:
        return self._turn

    def render_system_prompt(self) -> str:
        """Hermes' full system prompt (or the compact variant) plus skills and extra instructions."""
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache
        extra = [p for p in (*self.skill_prompts, self.system_prompt) if p]
        if self.hermes_system_prompt:
            with self._hermes_home_scope():
                prompt = self.agent._build_system_prompt("\n\n".join(extra) or None)
        else:
            tools_json = "\n".join(json.dumps(t["function"], ensure_ascii=False) for t in self.tools)
            prompt = "\n\n".join([COMPACT_SYSTEM_PROMPT.replace("{tools}", tools_json), *extra])
        self._system_prompt_cache = prompt
        return prompt

    def _begin_task(self, sample: Sample) -> None:
        from tools.terminal_tool import register_task_env_overrides

        self._task_id = f"monorl-{sample.id}-{uuid.uuid4().hex[:8]}"
        if self.workspace_dir is None:
            # Not tempfile.gettempdir(): macOS TMPDIR resolves under /private/var, which
            # Hermes' file tools refuse to write to (tools.file_tools._SENSITIVE_PATH_PREFIXES).
            root = default_workspace_root()
            root.mkdir(parents=True, exist_ok=True)
            self._episode_dir = tempfile.mkdtemp(prefix=f"{sample.id}-", dir=root)
            cwd = self._episode_dir
        else:
            Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)
            cwd = self.workspace_dir
        self.cwd = cwd
        register_task_env_overrides(self._task_id, {"cwd": cwd, **self._sandbox_overrides})

    def _end_task(self) -> None:
        if self._task_id is None:
            return
        from tools.process_registry import process_registry
        from tools.terminal_tool import cleanup_vm, clear_task_env_overrides

        try:
            with self._hermes_home_scope():
                process_registry.kill_all(task_id=self._task_id)  # background terminal jobs
                self.agent._cleanup_task_resources(self._task_id)  # browser + non-persistent sandbox
                if "env_type" in self._sandbox_overrides:
                    # Isolated (non-local) sandbox keyed by our task id: it must not outlive the
                    # episode even when Hermes' persist-mode would keep it for the idle reaper.
                    cleanup_vm(self._task_id, force_remove=True)
        finally:
            clear_task_env_overrides(self._task_id)
            self._task_id = None
            if self._episode_dir is not None:
                shutil.rmtree(self._episode_dir, ignore_errors=True)
                self._episode_dir = None

    # -- Environment protocol ---------------------------------------------------------------

    def reset(self, sample: Sample) -> list[Message]:
        if self._closed:
            raise RuntimeError("HermesEnv is closed")
        self._end_task()
        self._sample = sample
        self._turn = 0
        self.history = []
        self._begin_task(sample)
        messages: list[Message] = list(sample.messages)
        system = f"{self.render_system_prompt()}\n\nWorking directory for this task: {self.cwd}"
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": f"{system}\n\n{message_text(messages[0])}".strip()}
        else:
            messages.insert(0, {"role": "system", "content": system})
        self._messages = [dict(m) for m in messages]
        return messages

    def _validate_calls(self, calls: list[ToolCall]) -> tuple[list[ToolCall], list[Message], list[str]]:
        """Mirror Hermes' loop: repair near-miss names, error out unknown ones, execute the rest."""
        from agent.conversation_loop import _invalid_tool_name_error_content

        ok: list[ToolCall] = []
        errors: list[Message] = []
        reasons: list[str] = []
        for call in calls:
            name = call["name"]
            if name not in self.valid_tool_names:
                repaired = self.agent._repair_tool_call(name)
                if repaired:
                    call = ToolCall(id=call["id"], name=repaired, arguments=call.get("arguments", {}))
                    name = repaired
            if name in self.valid_tool_names:
                ok.append(call)
            else:
                reasons.append(f"unknown tool {name!r}")
                errors.append(
                    {
                        "role": "tool",
                        "name": name,
                        "tool_call_id": call["id"],
                        "content": _invalid_tool_name_error_content(name, self.valid_tool_names),
                    }
                )
        return ok, errors, reasons

    def _truncate(self, msg: Message) -> Message:
        from tools.budget_config import BudgetConfig
        from tools.tool_result_storage import maybe_persist_tool_result

        content = msg.get("content", "")
        if len(content) <= self.max_tool_output_chars:
            return msg
        limit = self.max_tool_output_chars
        msg["content"] = maybe_persist_tool_result(
            content=content,
            tool_name=msg.get("name", ""),
            tool_use_id=msg.get("tool_call_id", "truncated"),
            env=None,
            config=BudgetConfig(default_result_size=limit, turn_budget=limit, preview_size=limit),
            threshold=limit,
        )
        return msg

    def step(self, action: Completion) -> StepResult:
        if self._sample is None or self._task_id is None:
            raise RuntimeError("HermesEnv.step() called before reset()")
        self._turn += 1
        parsed = parse_tool_calls(action.text)
        reasons = list(parsed.errors)
        calls = parsed.calls[: self.max_calls_per_turn]
        if len(parsed.calls) > self.max_calls_per_turn:
            reasons.append(
                f"too many tool calls ({len(parsed.calls)} > {self.max_calls_per_turn}); extra calls ignored"
            )
        calls, name_errors, name_reasons = self._validate_calls(calls)
        reasons.extend(name_reasons)

        final_answer = not parsed.calls and not parsed.errors
        self._messages.append({"role": "assistant", "content": action.text})
        results: list[Message] = []
        executed: list[dict[str, Any]] = []
        if calls:
            hermes_messages: list[dict[str, Any]] = list(self._messages)
            before = len(hermes_messages)
            with self._hermes_home_scope():
                self.agent._execute_tool_calls(
                    to_openai_message(action.text, calls), hermes_messages, self._task_id
                )
            produced = [project_tool_message(m) for m in hermes_messages[before:] if m.get("role") == "tool"]
            results.extend(self._truncate(m) for m in produced)
            by_id = {m.get("tool_call_id"): m for m in produced}
            for c in calls:
                executed.append(
                    {
                        "id": c["id"],
                        "name": c["name"],
                        "arguments": c.get("arguments"),
                        "result": (by_id.get(c["id"]) or {}).get("content"),
                    }
                )
        results.extend(name_errors)
        for err in parsed.errors:
            results.append({"role": "tool", "name": "error", "content": f"error: {err}"})
        self._messages.extend(dict(m) for m in results)

        invalid = len(reasons)
        done = final_answer or self._turn >= self.max_turns
        self.history.append({"turn": self._turn, "calls": executed, "errors": reasons})
        if done:
            self._end_task()
        return StepResult(
            messages=[] if done else results,
            reward=-self.invalid_call_penalty if invalid else 0.0,
            done=done,
            info={
                "turn": self._turn,
                "tool_calls": executed,
                "invalid_calls": invalid,
                "final_answer": final_answer,
                "truncated": done and not final_answer,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._end_task()
        finally:
            with self._hermes_home_scope():
                self.agent.close()
            self._sample = None


def make_hermes_env(
    params: Mapping[str, Any],
    *,
    max_turns: int,
    system_prompt: str | None = None,
    model_name: str | None = None,
) -> HermesEnv:
    known = {
        "tools", "tool_modules", "skills", "skills_dir", "hermes_home", "workspace_dir", "sandbox",
        "max_tool_output_chars", "max_calls_per_turn", "invalid_call_penalty", "system_prompt",
        "hermes_system_prompt", "model_name",
    }  # fmt: skip
    unknown = sorted(set(params) - known)
    if unknown:
        raise ValueError(f"env 'hermes': unknown params {unknown}; expected {sorted(known)}")
    return HermesEnv(
        tools=_str_list(params.get("tools"), "tools") or list(DEFAULT_TOOLS),
        tool_modules=_str_list(params.get("tool_modules"), "tool_modules"),
        skills=_str_list(params.get("skills"), "skills"),
        skills_dir=params.get("skills_dir"),
        hermes_home=params.get("hermes_home"),
        workspace_dir=params.get("workspace_dir"),
        sandbox=params.get("sandbox"),
        max_turns=max_turns,
        max_tool_output_chars=int(params.get("max_tool_output_chars", DEFAULT_MAX_TOOL_OUTPUT_CHARS)),
        max_calls_per_turn=int(params.get("max_calls_per_turn", 8)),
        invalid_call_penalty=float(params.get("invalid_call_penalty", 0.0)),
        system_prompt=params.get("system_prompt", system_prompt),
        hermes_system_prompt=bool(params.get("hermes_system_prompt", True)),
        model_name=str(params.get("model_name") or model_name or "policy"),
    )


__all__ = [
    "COMPACT_SYSTEM_PROMPT",
    "DEFAULT_MAX_TOOL_OUTPUT_CHARS",
    "DEFAULT_TOOLS",
    "HermesEnv",
    "SANDBOX_BACKENDS",
    "custom_toolset_for",
    "load_skill_prompts",
    "make_hermes_env",
    "project_tool_message",
    "resolve_tool_names",
    "sandbox_overrides",
    "to_openai_message",
]
