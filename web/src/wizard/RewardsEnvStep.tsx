import { Card, Field, JsonEditor, NumberInput, Pill, Select, TextInput } from "../components/ui";
import { REWARD_TYPES_FALLBACK } from "../defaults";
import { ENV_TYPES, PIPELINE_BACKENDS, type EnvType, type JsonDict, type RewardSpec } from "../types";
import { HARNESS_EXAMPLE_COMMAND, type StepProps } from "./common";
import { HarnessForm } from "./HarnessForm";
import { SftNote } from "./TrainingStep";

const REWARD_HINTS: Record<string, string> = {
  exact_match: "1 if the extracted answer equals the reference answer (params: normalize, extract_regex).",
  format: "Regex/structure check on the response (params: pattern, flags).",
  regex: "Regex match against the response (params: pattern).",
  math_verify: "Symbolic equivalence via math-verify; robust to formatting.",
  code_exec: "Runs tests against generated code in a sandbox (params: timeout).",
  length: "Length shaping (params: target, penalty).",
  llm_judge: "LLM-as-judge via an OpenAI-compatible API (params: model, base_url, rubric).",
  reward_model: "Scores with a sequence-classification reward model (params: name_or_path).",
  composite: "Nested weighted combination (params: rewards).",
};

const ENV_HINTS: Record<EnvType, string> = {
  single_turn: "One prompt → one completion, scored by the rewards above.",
  tool_calling: "Multi-turn loop where the policy may call Python tools; params configure the tools.",
  remote: "External MEP/1 server (Go, C++, …) over stdio or HTTP; rewards come from the env unless rewards are set.",
  harness: "An external agent program is the environment: one process per episode that calls the policy through an OpenAI-compatible gateway. Its reward comes from a verifier, a reward the agent posts, or the rewards above.",
  openenv: "OpenEnv Gym-style HTTP environment (reset/step/state) in a container.",
  hermes: "Hermes Agent tool-using loop driven by the policy; skills configured in params.",
};

export default function RewardsEnvStep({ config, set, system, envs }: StepProps) {
  const rewardTypes = system?.components?.reward?.length ? system.components.reward : REWARD_TYPES_FALLBACK;
  const envTypes = envs?.types?.length ? envs.types : system?.components?.env?.length ? system.components.env : ENV_TYPES;
  const rewards = config.rewards;
  const env = config.env;
  const params = env.params;
  const transport = typeof params.transport === "string" ? params.transport : "stdio";
  const harness = env.type === "harness";
  // RunConfig._check: a pipeline backend cannot leave failed episodes out, and verl / NeMo-RL run their own rollouts.
  const backend = config.trainer.backend;
  const pipelineBackend = PIPELINE_BACKENDS[backend] === true;

  const updateReward = (i: number, r: Partial<RewardSpec>) =>
    set(
      "rewards",
      rewards.map((x, j) => (j === i ? { ...x, ...r } : x)),
    );

  if (config.algorithm.name === "sft")
    return (
      <Card title="Rewards & environment">
        <SftNote>rewards and environment settings are not used; the existing values are kept in the config so you can switch back to an RL objective without re-entering them.</SftNote>
        <div className="mt-3 text-xs text-slate-500">
          Current: {rewards.length} reward(s) · env <span className="font-mono">{env.type}</span>. Change the objective in the Training step to edit them.
        </div>
      </Card>
    );

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card
        title="Rewards"
        actions={
          <button type="button" className="btn" onClick={() => set("rewards", [...rewards, { type: rewardTypes[0], weight: 1, params: {} }])}>
            + Add reward
          </button>
        }
      >
        {rewards.length === 0 && <div className="text-xs text-amber-300">No rewards: only valid when the environment supplies its own rewards (remote/openenv/hermes, or harness with a verifier or a reward the agent posts).</div>}
        <div className="space-y-3">
          {rewards.map((r, i) => (
            <div key={i} className="rounded-md border border-slate-800 p-3">
              <div className="grid grid-cols-[2fr_1fr_auto] items-end gap-2">
                <Field label="Type" description={REWARD_HINTS[r.type]}>
                  <Select value={r.type} onChange={(v) => updateReward(i, { type: v })} options={[...new Set([...rewardTypes, r.type])].map((v) => ({ value: v }))} />
                </Field>
                <Field path="rewards.weight" label="Weight">
                  <NumberInput value={r.weight} onChange={(v) => updateReward(i, { weight: v ?? 1 })} step="any" />
                </Field>
                <button type="button" className="btn mb-5" onClick={() => set("rewards", rewards.filter((_, j) => j !== i))} title="Remove">
                  ✕
                </button>
              </div>
              <Field label="Params (JSON)" className="mt-2">
                <JsonEditor value={r.params} onChange={(v) => updateReward(i, { params: v as JsonDict })} rows={3} placeholder="{}" />
              </Field>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Environment">
        <Field path="env.type" label="Type" description={ENV_HINTS[env.type as EnvType] ?? undefined}>
          <Select
            value={env.type}
            onChange={(v) => {
              set("env.type", v);
              // Params of one type mean nothing to another: start remote and harness from their defaults, clear theirs when leaving.
              if (v === "harness") set("env.params", { command: HARNESS_EXAMPLE_COMMAND });
              else if (v === "remote" && !("transport" in params)) set("env.params", { transport: "stdio", command: [], cwd: null });
              else if (v !== "remote" && ("transport" in params || harness)) set("env.params", {});
              if (v === "single_turn") set("env.max_turns", 1);
              else if (env.max_turns === 1) set("env.max_turns", 8);
            }}
            options={[...new Set([...envTypes, env.type])].map((v) => ({ value: v }))}
          />
        </Field>
        <div className="mt-3 grid grid-cols-3 gap-3">
          <Field path="env.max_turns" label={harness ? "Max policy calls" : "Max turns"} description={harness ? "Caps the agent's calls to the policy per episode; further calls are refused with an OpenAI-style error." : undefined}>
            <NumberInput value={env.max_turns} onChange={(v) => set("env.max_turns", v ?? 1)} min={1} disabled={env.type === "single_turn"} />
          </Field>
          <Field path="env.retries" label="Retries" error={env.retries > 0 && pipelineBackend && backend !== "trl" ? `Not honoured by trainer.backend '${backend}', which runs its own rollouts.` : null}>
            <NumberInput value={env.retries} onChange={(v) => set("env.retries", v ?? 0)} min={0} />
          </Field>
          <Field
            path="env.max_failed_frac"
            label="Max failed fraction"
            error={env.max_failed_frac > 0 && pipelineBackend ? `Not supported by trainer.backend '${backend}': only the native loop (reference, unsloth) can leave failed episodes out of a batch.` : null}
          >
            <NumberInput value={env.max_failed_frac} onChange={(v) => set("env.max_failed_frac", v ?? 0)} min={0} max={1} step="any" />
          </Field>
        </div>

        {env.type === "remote" && (
          <div className="mt-3 space-y-3 rounded-md border border-slate-800 p-3">
            <Field label="Transport" description="stdio spawns the server as a subprocess per worker; http connects to a running server and runs `parallel` episodes concurrently.">
              <Select value={transport} onChange={(v) => set("env.params", v === "http" ? { transport: "http", url: typeof params.url === "string" ? params.url : "http://127.0.0.1:9000", parallel: 8 } : { transport: "stdio", command: Array.isArray(params.command) ? params.command : [], cwd: null })} options={[{ value: "stdio" }, { value: "http" }]} />
            </Field>
            {transport === "stdio" ? (
              <>
                <Field label="Command" description="Executable and arguments, space separated.">
                  <TextInput
                    value={Array.isArray(params.command) ? params.command.map(String).join(" ") : ""}
                    onChange={(v) => set("env.params.command", (v ?? "").trim() ? (v ?? "").trim().split(/\s+/) : [])}
                    className="font-mono"
                    placeholder="envs/sdk/go/bin/wordle --seed 1"
                  />
                </Field>
                <Field label="Working directory">
                  <TextInput value={typeof params.cwd === "string" ? params.cwd : null} onChange={(v) => set("env.params.cwd", v)} nullable className="font-mono" placeholder="repo root" />
                </Field>
              </>
            ) : (
              <>
                <Field label="URL">
                  <TextInput value={typeof params.url === "string" ? params.url : ""} onChange={(v) => set("env.params.url", v ?? "")} className="font-mono" placeholder="http://127.0.0.1:9000" />
                </Field>
                <Field label="Parallel episodes">
                  <NumberInput value={typeof params.parallel === "number" ? params.parallel : 8} onChange={(v) => set("env.params.parallel", v ?? 8)} min={1} />
                </Field>
              </>
            )}
            {envs?.examples && envs.examples.length > 0 && (
              <div>
                <div className="mb-1 text-xs text-slate-400">Example MEP/1 servers found under envs/sdk:</div>
                <div className="flex flex-wrap gap-1">
                  {envs.examples.map((ex) => (
                    <button
                      key={ex.path}
                      type="button"
                      className="btn py-1 text-xs"
                      onClick={() => set("env.params", { transport: ex.transport ?? "stdio", command: ex.command, cwd: null })}
                      title={ex.path}
                    >
                      {ex.name} <Pill>{ex.language}</Pill>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {harness && <HarnessForm params={params} set={set} modelName={config.model.name_or_path} />}

        {env.type !== "remote" && env.type !== "harness" && env.type !== "single_turn" && (
          <Field label="Params (JSON)" path="env.params" className="mt-3">
            <JsonEditor value={params} onChange={(v) => set("env.params", v)} rows={6} placeholder={env.type === "openenv" ? '{"url": "http://localhost:8001", "action_field": "message"}' : "{}"} />
          </Field>
        )}
      </Card>
    </div>
  );
}
