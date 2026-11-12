import { useMemo, useState } from "react";
import { api } from "../api";
import { Card, ErrorBox, Field, NumberInput, Pill, Select, TextInput, Toggle, YamlEditor } from "../components/ui";
import { useAsync } from "../hooks";
import { checkPipeline, defaultStage, describeOverrides, effectiveAlgorithm, r1LoopStages, uniqueStageName } from "../pipeline";
import { ALGORITHMS, STAGE_TYPES, type AlgorithmName, type DataConfig, type EvalStage, type JsonDict, type JsonSchema, type PipelineConfig, type RejectionSampleStage, type Stage, type StageType, type TrainStage, type ValidationError } from "../types";

export interface StagesStepProps {
  pipeline: PipelineConfig;
  setStages: (stages: Stage[]) => void;
}

const STAGE_DEFS: Record<StageType, string> = { train: "TrainStage", rejection_sample: "RejectionSampleStage", eval: "EvalStage" };
const STAGE_TONE: Record<StageType, "sky" | "amber" | "green"> = { train: "sky", rejection_sample: "amber", eval: "green" };
const STAGE_BLURB: Record<StageType, string> = {
  train: "A full run (RL or SFT): the base config merged with overrides. Starts from the previous train stage's final export unless init_from says otherwise.",
  rejection_sample: "Samples N completions per prompt with the current policy, keeps those scoring ≥ min_reward and writes a chat jsonl for the next SFT stage.",
  eval: "Evaluation of the current policy on a dataset: greedy, or several samples per prompt for pass@k. Logs eval/reward_mean and eval/success_rate.",
};

/** Field description from GET /api/config/pipeline_schema ($defs.<Stage>.properties.<field>). */
function stageDesc(schema: JsonSchema | null, type: StageType, field: string): string | undefined {
  return schema?.$defs?.[STAGE_DEFS[type]]?.properties?.[field]?.description;
}

const errorsFor = (errors: ValidationError[], i: number, field?: string) => errors.filter((e) => e.loc[0] === "stages" && e.loc[1] === i && (field === undefined || e.loc[2] === field));
const firstMsg = (errs: ValidationError[]) => (errs.length ? errs[0].msg : null);

export default function StagesStep({ pipeline, setStages }: StagesStepProps) {
  const stages = pipeline.stages;
  const schema = useAsync(() => api.pipelineSchema(), []);
  const [open, setOpen] = useState<number | null>(stages.length ? 0 : null);
  const errors = useMemo(() => checkPipeline(pipeline), [pipeline]);

  const update = (i: number, next: Stage) => setStages(stages.map((s, j) => (j === i ? next : s)));
  const add = (type: StageType) => {
    const next = [...stages, defaultStage(type, uniqueStageName(stages, type === "rejection_sample" ? "rejection-sample" : type))];
    setStages(next);
    setOpen(next.length - 1);
  };
  const remove = (i: number) => {
    setStages(stages.filter((_, j) => j !== i));
    setOpen((o) => (o === null ? null : o === i ? null : o > i ? o - 1 : o));
  };
  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir;
    if (j < 0 || j >= stages.length) return;
    const next = [...stages];
    [next[i], next[j]] = [next[j], next[i]];
    setStages(next);
    setOpen(j);
  };
  const insertR1 = () => {
    const fresh = r1LoopStages();
    const next = [...stages];
    for (const s of fresh) next.push({ ...s, name: uniqueStageName(next, s.name) });
    setStages(next);
    setOpen(stages.length);
  };

  return (
    <div className="space-y-4">
      <Card
        title={
          <span>
            Stages <span className="ml-2 text-xs font-normal text-slate-500">run in order; each stage gets its own run dir under the pipeline run dir</span>
          </span>
        }
        actions={
          <div className="flex flex-wrap gap-2">
            {STAGE_TYPES.map((t) => (
              <button key={t} type="button" className="btn" onClick={() => add(t)}>
                + {t}
              </button>
            ))}
            <button type="button" className="btn btn-primary" onClick={insertR1} title="cold-start SFT → GRPO → rejection sampling → SFT → GRPO">
              + R1 loop (5 stages)
            </button>
          </div>
        }
      >
        <ErrorBox error={schema.error ? `pipeline schema unavailable (${schema.error}); field help is limited` : null} className="mb-3" />
        {stages.length === 0 && <div className="text-sm text-slate-500">No stages yet. Add stages one by one or start from the R1 loop template.</div>}
        <ol className="space-y-2">
          {stages.map((s, i) => {
            const errs = errorsFor(errors, i);
            const expanded = open === i;
            return (
              <li key={i} className={`rounded-md border ${errs.length ? "border-rose-800" : expanded ? "border-sky-700" : "border-slate-800"} bg-slate-900/60`}>
                <div className="flex items-center gap-2 px-3 py-2">
                  <button type="button" className="flex min-w-0 flex-1 items-center gap-2 text-left" onClick={() => setOpen(expanded ? null : i)}>
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-slate-700 text-xs text-slate-200">{i + 1}</span>
                    <Pill tone={STAGE_TONE[s.type]}>{s.type}</Pill>
                    <span className="truncate font-mono text-sm text-slate-100">{s.name || <span className="text-rose-300">(unnamed)</span>}</span>
                    <span className="truncate text-xs text-slate-500">{stageSummary(pipeline, s, i)}</span>
                    {errs.length > 0 && <Pill tone="red">{errs.length} issue(s)</Pill>}
                  </button>
                  <button type="button" className="btn px-2 py-1" disabled={i === 0} onClick={() => move(i, -1)} title="Move up">
                    ↑
                  </button>
                  <button type="button" className="btn px-2 py-1" disabled={i === stages.length - 1} onClick={() => move(i, 1)} title="Move down">
                    ↓
                  </button>
                  <button type="button" className="btn px-2 py-1" onClick={() => remove(i)} title="Remove stage">
                    ✕
                  </button>
                </div>
                {expanded && (
                  <div className="border-t border-slate-800 px-3 py-3">
                    <div className="mb-3 text-xs text-slate-500">{STAGE_BLURB[s.type]}</div>
                    <div className="mb-3 grid gap-3 md:grid-cols-[1fr_2fr]">
                      <Field label="Type">
                        <Select<StageType> value={s.type} onChange={(t) => update(i, defaultStage(t, s.name))} options={STAGE_TYPES.map((t) => ({ value: t }))} />
                      </Field>
                      <Field label="Name" description={stageDesc(schema.data, s.type, "name") ?? "Stage name (used for the stage directory)."} error={firstMsg(errorsFor(errors, i, "name"))}>
                        <TextInput value={s.name} onChange={(v) => update(i, { ...s, name: v ?? "" })} className="font-mono" />
                      </Field>
                    </div>
                    {s.type === "train" && <TrainEditor stage={s} index={i} pipeline={pipeline} schema={schema.data} errors={errors} onChange={(next) => update(i, next)} />}
                    {s.type === "rejection_sample" && <RejectionEditor stage={s} index={i} schema={schema.data} errors={errors} onChange={(next) => update(i, next)} />}
                    {s.type === "eval" && <EvalEditor stage={s} index={i} schema={schema.data} errors={errors} onChange={(next) => update(i, next)} />}
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      </Card>
      {errors.some((e) => e.loc[0] !== "stages" || typeof e.loc[1] !== "number") && (
        <ErrorBox error={errors.filter((e) => e.loc[0] !== "stages" || typeof e.loc[1] !== "number").map((e) => e.msg).join("; ")} />
      )}
    </div>
  );
}

function stageSummary(p: PipelineConfig, s: Stage, i: number): string {
  if (s.type === "train") {
    const init = s.init_from === "previous" ? (i === 0 ? "base model" : "previous export") : s.init_from === "base" ? "base model" : s.init_from;
    return `${effectiveAlgorithm(p.base, s)} · init ${init}${s.data_from === "previous" ? " · data from previous stage" : ""} · ${describeOverrides(s.overrides, 3)}`;
  }
  if (s.type === "rejection_sample") return `${s.num_samples} samples/prompt · keep reward ≥ ${s.min_reward} · ≤${s.max_per_prompt}/prompt${s.mix.length ? ` · +${s.mix.length} mixed` : ""}`;
  return `${s.data?.uri ? s.data.uri : "base eval data"}${s.max_prompts ? ` · ${s.max_prompts} prompts` : ""}${s.num_samples > 1 ? ` · ${s.num_samples} samples/prompt (pass@${s.num_samples})` : " · greedy"}`;
}

// ---- train -----------------------------------------------------------------------------------

function TrainEditor({ stage: s, index: i, pipeline, schema, errors, onChange }: { stage: TrainStage; index: number; pipeline: PipelineConfig; schema: JsonSchema | null; errors: ValidationError[]; onChange: (s: TrainStage) => void }) {
  const initMode = s.init_from === "previous" || s.init_from === "base" ? s.init_from : "path";
  const prevTrain = pipeline.stages.slice(0, i).filter((x) => x.type === "train").pop();
  const prev = i > 0 ? pipeline.stages[i - 1] : null;
  const algo = effectiveAlgorithm(pipeline.base, s);
  const setAlgo = (name: AlgorithmName) => {
    const cur = pipeline.base.algorithm.name;
    const rest = { ...s.overrides };
    const algoOv = rest.algorithm && typeof rest.algorithm === "object" && !Array.isArray(rest.algorithm) ? { ...rest.algorithm } : {};
    if (name === cur) delete algoOv.name;
    else algoOv.name = name;
    if (Object.keys(algoOv).length) rest.algorithm = algoOv;
    else delete rest.algorithm;
    onChange({ ...s, overrides: rest });
  };
  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-3">
        <Field label="Objective" description="Shortcut for overrides.algorithm.name; the base objective is used when unset.">
          <Select<AlgorithmName> value={ALGORITHMS.includes(algo as AlgorithmName) ? (algo as AlgorithmName) : "grpo"} onChange={setAlgo} options={ALGORITHMS.map((v) => ({ value: v, label: v === pipeline.base.algorithm.name ? `${v} (base)` : v }))} />
        </Field>
        <Field
          label="Initialise from"
          description={stageDesc(schema, "train", "init_from")}
          error={firstMsg(errorsFor(errors, i, "init_from"))}
        >
          <div className="space-y-1">
            <Select<"previous" | "base" | "path">
              value={initMode}
              onChange={(m) => onChange({ ...s, init_from: m === "path" ? "" : m })}
              options={[
                { value: "previous", label: prevTrain ? `previous (${prevTrain.name} → final/)` : "previous (no earlier train stage: base model)" },
                { value: "base", label: `base (${pipeline.base.model.name_or_path || "base model"})` },
                { value: "path", label: "path…" },
              ]}
            />
            {initMode === "path" && <TextInput value={s.init_from} onChange={(v) => onChange({ ...s, init_from: v ?? "" })} className="font-mono" placeholder="/path/to/checkpoint or hf id" />}
          </div>
        </Field>
        <Field
          label="Dataset"
          description={stageDesc(schema, "train", "data_from")}
          error={firstMsg(errorsFor(errors, i, "data_from"))}
        >
          <Select<"config" | "previous">
            value={s.data_from}
            onChange={(v) => onChange({ ...s, data_from: v })}
            options={[
              { value: "config", label: "config (base data.uri or overrides.data.uri)" },
              { value: "previous", label: prev?.type === "rejection_sample" ? `previous (${prev.name} → dataset.jsonl)` : "previous (needs a rejection_sample stage right before)" },
            ]}
          />
        </Field>
      </div>
      <Field
        label="Overrides (YAML, RunConfig subset)"
        description={stageDesc(schema, "train", "overrides") ?? "RunConfig fields merged over the pipeline base."}
        error={firstMsg(errorsFor(errors, i, "overrides"))}
      >
        <YamlEditor<JsonDict> value={s.overrides} onChange={(v) => onChange({ ...s, overrides: v })} shape="object" empty={{}} rows={7} placeholder={"algorithm:\n  name: sft\ntrainer:\n  max_steps: 50\n  learning_rate: 1.0e-5"} />
      </Field>
      <div className="text-xs text-slate-500">
        Effective: <span className="font-mono text-slate-300">{algo}</span> · {describeOverrides(s.overrides)}
      </div>
    </div>
  );
}

// ---- rejection sampling ----------------------------------------------------------------------

function RejectionEditor({ stage: s, index: i, schema, errors, onChange }: { stage: RejectionSampleStage; index: number; schema: JsonSchema | null; errors: ValidationError[]; onChange: (s: RejectionSampleStage) => void }) {
  const d = (f: string) => stageDesc(schema, "rejection_sample", f);
  return (
    <div className="space-y-3">
      <div className="grid gap-3 md:grid-cols-4">
        <Field label="Samples / prompt" description={d("num_samples")} error={firstMsg(errorsFor(errors, i, "num_samples"))}>
          <NumberInput value={s.num_samples} onChange={(v) => onChange({ ...s, num_samples: v ?? 4 })} min={1} />
        </Field>
        <Field label="Min reward" description={d("min_reward")}>
          <NumberInput value={s.min_reward} onChange={(v) => onChange({ ...s, min_reward: v ?? 1 })} step="any" />
        </Field>
        <Field label="Keep / prompt" description={d("max_per_prompt")} error={firstMsg(errorsFor(errors, i, "max_per_prompt"))}>
          <NumberInput value={s.max_per_prompt} onChange={(v) => onChange({ ...s, max_per_prompt: v ?? 1 })} min={1} />
        </Field>
        <Field label="Max prompts" description={d("max_prompts")}>
          <NumberInput value={s.max_prompts} onChange={(v) => onChange({ ...s, max_prompts: v })} nullable min={1} placeholder="all" />
        </Field>
        <Field label="Max response tokens" description={d("max_response_tokens")}>
          <NumberInput value={s.max_response_tokens} onChange={(v) => onChange({ ...s, max_response_tokens: v })} nullable min={1} placeholder="no limit" />
        </Field>
        <Field label="Output" description={d("output")} className="md:col-span-3">
          <TextInput value={s.output} onChange={(v) => onChange({ ...s, output: v })} nullable className="font-mono" placeholder="<stage dir>/dataset.jsonl" />
        </Field>
      </div>
      <div className="flex flex-wrap gap-6">
        <Toggle checked={s.require_finish} onChange={(v) => onChange({ ...s, require_finish: v })} label="Require finish" description={d("require_finish")} />
        <Toggle checked={s.dedupe} onChange={(v) => onChange({ ...s, dedupe: v })} label="Dedupe" description={d("dedupe")} />
      </div>
      <DataOverride value={s.data} onChange={(data) => onChange({ ...s, data })} description={d("data") ?? "Prompts to sample from (defaults to the base data config)."} />
      <div className="grid gap-3 md:grid-cols-3">
        <Field label="Mix in datasets (one URI per line)" description={d("mix")}>
          <textarea
            className="input font-mono text-xs"
            rows={5}
            value={s.mix.join("\n")}
            onChange={(e) =>
              onChange({
                ...s,
                mix: e.target.value
                  .split("\n")
                  .map((x) => x.trim())
                  .filter(Boolean),
              })
            }
            placeholder="hf://org/sft-mix"
          />
        </Field>
        <Field label="Rollout overrides (YAML)" description={d("rollout_overrides")}>
          <YamlEditor<JsonDict> value={s.rollout_overrides} onChange={(v) => onChange({ ...s, rollout_overrides: v })} shape="object" empty={{}} rows={5} placeholder={"temperature: 0.7\nmax_new_tokens: 1024"} />
        </Field>
        <Field label="Reward overrides (YAML list)" description={d("reward_overrides")}>
          <YamlEditor<JsonDict[] | null> value={s.reward_overrides} onChange={(v) => onChange({ ...s, reward_overrides: v })} shape="array" empty={null} rows={5} placeholder={"- type: math_verify\n  weight: 1.0"} />
        </Field>
      </div>
    </div>
  );
}

// ---- eval ------------------------------------------------------------------------------------

function EvalEditor({ stage: s, index: i, schema, errors, onChange }: { stage: EvalStage; index: number; schema: JsonSchema | null; errors: ValidationError[]; onChange: (s: EvalStage) => void }) {
  return (
    <div className="space-y-3">
      <DataOverride value={s.data} onChange={(data) => onChange({ ...s, data })} description={stageDesc(schema, "eval", "data") ?? "Evaluation dataset (defaults to base data.eval_uri or data)."} />
      <div className="grid gap-3 md:grid-cols-4">
        <Field label="Max prompts" description={stageDesc(schema, "eval", "max_prompts")}>
          <NumberInput value={s.max_prompts} onChange={(v) => onChange({ ...s, max_prompts: v })} nullable min={1} placeholder="all" />
        </Field>
        <Field
          label="Samples / prompt"
          description={stageDesc(schema, "eval", "num_samples") ?? "1 = greedy. More samples are drawn at the rollout temperature and reported as eval/pass_at_k."}
          error={firstMsg(errorsFor(errors, i, "num_samples"))}
        >
          <NumberInput value={s.num_samples} onChange={(v) => onChange({ ...s, num_samples: v ?? 1 })} min={1} />
        </Field>
        <Field label="Rollout overrides (YAML)" description={stageDesc(schema, "eval", "rollout_overrides")} className="md:col-span-2">
          <YamlEditor<JsonDict> value={s.rollout_overrides} onChange={(v) => onChange({ ...s, rollout_overrides: v })} shape="object" empty={{}} rows={3} placeholder="max_new_tokens: 1024" />
        </Field>
      </div>
    </div>
  );
}

/** Optional DataConfig override: a few common fields inline, the rest via YAML. */
function DataOverride({ value, onChange, description }: { value: Partial<DataConfig> | null; onChange: (v: Partial<DataConfig> | null) => void; description: string }) {
  const on = value !== null;
  return (
    <div className="rounded-md border border-slate-800 p-3">
      <Toggle checked={on} onChange={(v) => onChange(v ? { uri: "" } : null)} label="Custom dataset" description={description} />
      {on && (
        <div className="mt-3 grid gap-3 md:grid-cols-[2fr_1fr_1fr]">
          <Field label="URI">
            <TextInput value={value.uri ?? ""} onChange={(v) => onChange({ ...value, uri: v ?? "" })} className="font-mono" placeholder="hf://org/name · /path/file.jsonl" />
          </Field>
          <Field label="Split">
            <TextInput value={value.split ?? ""} onChange={(v) => onChange({ ...value, split: v || undefined })} placeholder="train" />
          </Field>
          <Field label="Format">
            <Select value={value.format ?? "auto"} onChange={(v) => onChange({ ...value, format: v })} options={["auto", "chat", "prompt_answer", "text"].map((v) => ({ value: v as DataConfig["format"] }))} />
          </Field>
          <Field label="More fields (YAML)" className="md:col-span-3">
            <YamlEditor<Partial<DataConfig>> value={omit(value, ["uri", "split", "format"])} onChange={(v) => onChange({ uri: value.uri, split: value.split, format: value.format, ...v })} shape="object" empty={{}} rows={3} placeholder={"prompt_field: question\nanswer_field: answer"} />
          </Field>
        </div>
      )}
    </div>
  );
}

function omit<T extends object>(obj: T, keys: (keyof T)[]): Partial<T> {
  const out: Partial<T> = { ...obj };
  for (const k of keys) delete out[k];
  return out;
}
