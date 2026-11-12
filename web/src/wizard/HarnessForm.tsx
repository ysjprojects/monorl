// `env.params` of `env.type: harness` (src/monorl/envs/harness.py, src/monorl/envs/sandbox.py) as a
// form. The fields edit single keys of the params in place and "Other params (JSON)" shows every
// key the fields above it do not own, so keys the form does not know survive untouched.

import { Field, JsonEditor, NumberInput, Select, TextInput } from "../components/ui";
import { isPlainObject } from "../defaults";
import type { JsonDict, JsonValue } from "../types";
import { HARNESS_EXAMPLE_COMMAND, type StepProps } from "./common";

/** Top-level params with a field of their own; the rest (`env`, `workdir`, `gateway`, `sandbox`) is edited as JSON. */
const FORM_PARAMS = ["command", "timeout", "verifier", "verifier_timeout", "model"];
const PLACEHOLDERS = ["{task_file}", "{workdir}", "{base_url}", "{api_key}", "{model}", "{python}", "{session}", "{reward_url}", "{sample_id}", "{anthropic_base_url}"];
const SANDBOX_BACKENDS = ["local", "docker"];

/**
 * A harness command as one editable line: the string as configured, or an argv list joined the way
 * a shell would read it back (the env splits strings like a shell and runs them without one).
 */
function commandLine(command: JsonValue | undefined): string {
  if (typeof command === "string") return command;
  if (!Array.isArray(command)) return "";
  return command.map((arg) => (typeof arg === "string" && /^[\w@%+=:,./{}-]+$/.test(arg) ? arg : `'${String(arg).replaceAll("'", `'\\''`)}'`)).join(" ");
}

/** `value` without `key` when `next` is `null`, else with `key: next`. */
function withKey(value: JsonDict, key: string, next: JsonValue): JsonDict {
  if (next !== null) return { ...value, [key]: next };
  return Object.fromEntries(Object.entries(value).filter(([k]) => k !== key));
}

const text = (value: JsonValue | undefined): string | null => (typeof value === "string" ? value : typeof value === "number" ? String(value) : null);

export function HarnessForm({ params, set, modelName }: { params: JsonDict; set: StepProps["set"]; modelName: string }) {
  /** Set one key of `env.params`; `null` removes it, so the env's default applies. */
  const setParam = (key: string, value: JsonValue) => set("env.params", withKey(params, key, value));

  const sandbox: JsonDict = isPlainObject(params.sandbox) ? params.sandbox : {};
  const backend = typeof sandbox.backend === "string" ? sandbox.backend.toLowerCase() : "local";
  const docker = backend === "docker";
  /** Set one option of `env.params.sandbox`; the last option to go takes the mapping with it. */
  const setSandbox = (key: string, value: JsonValue) => {
    const next = withKey(sandbox, key, value);
    setParam("sandbox", Object.keys(next).length ? next : null);
  };
  // `build` is a context directory or {context, dockerfile, args, target}; the form edits the first two.
  const build: JsonDict | null = typeof sandbox.build === "string" ? { context: sandbox.build } : isPlainObject(sandbox.build) ? sandbox.build : null;
  const setBuild = (key: "context" | "dockerfile", value: string | null) => {
    const next = withKey(build ?? {}, key, value);
    setSandbox("build", Object.keys(next).length ? next : null);
  };
  const image = text(sandbox.image);
  const context = build ? text(build.context) : null;
  const dockerOnly = Object.keys(sandbox).filter((k) => k !== "backend" && sandbox[k] !== null);

  return (
    <div className="mt-3 space-y-3 rounded-md border border-slate-800 p-3">
      <Field
        label="Command"
        description="The agent, started once per episode: a command line (split like a shell would, run without one)."
        error={commandLine(params.command).trim() ? null : "Required: the agent's command."}
      >
        <TextInput value={commandLine(params.command)} onChange={(v) => setParam("command", v ?? "")} className="font-mono" placeholder={HARNESS_EXAMPLE_COMMAND.join(" ")} />
      </Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Timeout (s)" description="Seconds the agent may run before it is killed.">
          <NumberInput value={typeof params.timeout === "number" ? params.timeout : null} onChange={(v) => setParam("timeout", v)} nullable min={0} step="any" placeholder="600" />
        </Field>
        <Field label="Model name" description="The model name the agent is told.">
          <TextInput value={typeof params.model === "string" ? params.model : null} onChange={(v) => setParam("model", v)} nullable className="font-mono" placeholder={modelName || "model.name_or_path"} />
        </Field>
      </div>
      <div className="grid grid-cols-[2fr_1fr] gap-3">
        <Field
          label="Verifier"
          description="Optional command run in the working directory after the agent ended. Its result is the episode's reward: the last line of its stdout when that is a number, else 1 for exit status 0 and 0 otherwise. It also gets {reference_file}, {stdout_file}, {stderr_file} and {exit_code}."
        >
          <TextInput value={params.verifier == null ? null : commandLine(params.verifier)} onChange={(v) => setParam("verifier", v)} nullable className="font-mono" placeholder="none" />
        </Field>
        <Field label="Verifier timeout (s)" description="A verifier that runs longer scores 0.">
          <NumberInput value={typeof params.verifier_timeout === "number" ? params.verifier_timeout : null} onChange={(v) => setParam("verifier_timeout", v)} nullable min={0} step="any" placeholder="60" disabled={params.verifier == null} />
        </Field>
      </div>

      <div className="space-y-3 border-t border-slate-800 pt-3">
        <div className="text-xs font-semibold text-slate-300">Sandbox</div>
        <Field
          label="Backend"
          description={docker ? "One container per episode, driven through the Docker CLI: the agent and the verifier are `docker exec` processes in it. Every path a placeholder or variable names is the container's." : "Where the agent and the verifier run. local: processes on the trainer host, each in its own process group."}
          error={
            params.sandbox != null && !isPlainObject(params.sandbox)
              ? "env.params.sandbox must be a mapping, e.g. {backend: docker, image: …}: fix it under Other params."
              : !SANDBOX_BACKENDS.includes(backend)
                ? `Unknown backend '${backend}': expected local or docker.`
                : !docker && dockerOnly.length
                  ? `${dockerOnly.join(", ")} need${dockerOnly.length === 1 ? "s" : ""} the docker backend: switch to it, or remove ${dockerOnly.length === 1 ? "it" : "them"} under Other params.`
                  : null
          }
        >
          <Select value={backend} onChange={(v) => setSandbox("backend", v === "local" ? null : v)} options={[...new Set([...SANDBOX_BACKENDS, backend])].map((v) => ({ value: v }))} />
        </Field>
        {docker && (
          <>
            <div className="grid grid-cols-3 gap-3">
              <Field label="Image" description="Pulled when it is not local." error={image && build ? "Takes an image or a build, not both." : null}>
                <TextInput value={image} onChange={(v) => setSandbox("image", v)} nullable className="font-mono" placeholder="python:3.12-slim" />
              </Field>
              <Field label="Build context" description="Or a directory to build the image from; built once and cached." error={build && !context ? "A build needs its context directory." : null}>
                <TextInput value={context} onChange={(v) => setBuild("context", v)} nullable className="font-mono" placeholder="./agent" />
              </Field>
              <Field label="Dockerfile" description="Relative to the build context.">
                <TextInput value={build ? text(build.dockerfile) : null} onChange={(v) => setBuild("dockerfile", v)} nullable className="font-mono" placeholder="Dockerfile" disabled={!build} />
              </Field>
            </div>
            {!image && !build && <div className="text-xs text-amber-300">The docker backend needs an image or a build, unless a task-directory dataset brings one with every task.</div>}
            <div className="grid grid-cols-4 gap-3">
              <Field label="Network" description="docker run --network." error={sandbox.network === "none" ? "'none' leaves the agent without a way to the policy: the gateway runs on the trainer host." : null}>
                <TextInput value={text(sandbox.network)} onChange={(v) => setSandbox("network", v)} nullable className="font-mono" placeholder="default" />
              </Field>
              <Field label="CPUs" description="CPU limit of the container.">
                <NumberInput value={typeof sandbox.cpus === "number" ? sandbox.cpus : null} onChange={(v) => setSandbox("cpus", v)} nullable min={0} step="any" placeholder="unlimited" />
              </Field>
              <Field label="Memory" description="Memory limit, e.g. 2g.">
                <TextInput value={text(sandbox.memory)} onChange={(v) => setSandbox("memory", v)} nullable className="font-mono" placeholder="unlimited" />
              </Field>
              <Field label="User" description="User the processes run as.">
                <TextInput value={text(sandbox.user)} onChange={(v) => setSandbox("user", v)} nullable className="font-mono" placeholder="image default" />
              </Field>
            </div>
          </>
        )}
      </div>

      <Field
        label="Other params (JSON)"
        description="env: extra environment variables for the agent and the verifier. workdir: {root, template} of the fresh per-episode working directory. gateway: {bind, port, host}. sandbox: all its options, including the ones above and build_timeout, workdir, mounts, pids_limit, gpus, platform, python, keep_entrypoint, keep_processes, args, start_timeout."
      >
        <JsonEditor
          value={Object.fromEntries(Object.entries(params).filter(([k]) => !FORM_PARAMS.includes(k)))}
          onChange={(v) => set("env.params", { ...Object.fromEntries(Object.entries(params).filter(([k]) => FORM_PARAMS.includes(k))), ...v })}
          rows={5}
          placeholder={'{"env": {"LOG_LEVEL": "info"}, "workdir": {"root": "/tmp/monorl-episodes", "template": "./task_template"}}'}
        />
      </Field>
      <div className="text-xs text-slate-500">
        Placeholders in the command, the verifier and <span className="font-mono">env</span> values: <span className="font-mono text-slate-400">{PLACEHOLDERS.join(" ")}</span> (the last when the gateway serves the Anthropic API; <span className="font-mono">{"{python}"}</span> is{" "}
        <span className="font-mono">sandbox.python</span> in a container). The agent's process gets <span className="font-mono text-slate-400">OPENAI_BASE_URL</span> / <span className="font-mono text-slate-400">OPENAI_API_KEY</span>, so an unmodified OpenAI SDK client reaches the policy, and{" "}
        <span className="font-mono text-slate-400">MONORL_TASK_FILE</span>: the sample's id, messages and metadata, never the reference.
      </div>
    </div>
  );
}
