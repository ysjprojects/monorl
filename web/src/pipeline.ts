// Pipeline-side counterpart of defaults.ts: stage defaults (mirroring src/monorl/config/pipeline.py),
// normalisation of partial documents (recipes, pasted YAML) and the structural checks the server's
// PipelineConfig validator applies, so the wizard can hint before — or without — a server round-trip.

import { deepMerge, fromPartial, isPlainObject } from "./defaults";
import type { EvalStage, JsonDict, PipelineConfig, RejectionSampleStage, RunConfig, Stage, StageType, TrainStage, ValidationError } from "./types";

export function defaultTrainStage(name = "train"): TrainStage {
  return { type: "train", name, overrides: {}, init_from: "previous", data_from: "config" };
}

export function defaultRejectionStage(name = "rejection_sample"): RejectionSampleStage {
  return {
    type: "rejection_sample",
    name,
    data: null,
    num_samples: 4,
    min_reward: 1.0,
    max_per_prompt: 1,
    max_response_tokens: null,
    max_context_tokens: null,
    require_finish: true,
    dedupe: true,
    mix: [],
    rollout_overrides: {},
    reward_overrides: null,
    output: null,
    max_prompts: null,
  };
}

export function defaultEvalStage(name = "eval"): EvalStage {
  return { type: "eval", name, data: null, rollout_overrides: {}, max_prompts: null, num_samples: 1 };
}

export function defaultStage(type: StageType, name?: string): Stage {
  if (type === "train") return defaultTrainStage(name);
  if (type === "rejection_sample") return defaultRejectionStage(name);
  return defaultEvalStage(name);
}

/** Fill a partial stage document with defaults for its `type` (unknown types become train). */
export function stageFromPartial(partial: unknown, fallbackName: string): Stage {
  const p = isPlainObject(partial) ? partial : {};
  const type: StageType = p.type === "rejection_sample" || p.type === "eval" ? p.type : "train";
  const name = typeof p.name === "string" && p.name ? p.name : fallbackName;
  const base = defaultStage(type, name);
  const merged = deepMerge(base, { ...p, type, name });
  if (merged.type === "train" && !isPlainObject(merged.overrides)) merged.overrides = {};
  if (merged.type === "rejection_sample") {
    if (!Array.isArray(merged.mix)) merged.mix = [];
    if (!isPlainObject(merged.rollout_overrides)) merged.rollout_overrides = {};
  }
  if (merged.type === "eval" && !isPlainObject(merged.rollout_overrides)) merged.rollout_overrides = {};
  return merged;
}

/** Normalise a partial pipeline (recipe / YAML) to a full PipelineConfig. */
export function pipelineFromPartial(partial: unknown): PipelineConfig {
  const p = isPlainObject(partial) ? partial : {};
  const stagesIn = Array.isArray(p.stages) ? p.stages : [];
  return {
    name: typeof p.name === "string" && p.name ? p.name : null,
    base: fromPartial(p.base ?? {}),
    stages: stagesIn.map((s, i) => stageFromPartial(s, `stage-${i + 1}`)),
  };
}

/** Rename a stage, keeping names unique within the pipeline (`sft`, `sft-2`, ...). */
export function uniqueStageName(stages: Stage[], wanted: string): string {
  const taken = new Set(stages.map((s) => s.name));
  if (!taken.has(wanted)) return wanted;
  for (let i = 2; ; i++) if (!taken.has(`${wanted}-${i}`)) return `${wanted}-${i}`;
}

/**
 * Structural checks mirroring `PipelineConfig._check` plus the shape rules pydantic would reject.
 * Returned `loc`s use the same path convention as the server (`stages.<i>.<field>`).
 */
export function checkPipeline(p: PipelineConfig): ValidationError[] {
  const errors: ValidationError[] = [];
  if (p.stages.length === 0) errors.push({ loc: ["stages"], msg: "List should have at least 1 item", type: "too_short" });
  const seen = new Map<string, number>();
  p.stages.forEach((s, i) => {
    if (!s.name.trim()) errors.push({ loc: ["stages", i, "name"], msg: "Stage name must not be empty", type: "missing" });
    else if (seen.has(s.name)) errors.push({ loc: ["stages", i, "name"], msg: `Value error, stage names must be unique (duplicate of stage ${seen.get(s.name)})`, type: "value_error" });
    else seen.set(s.name, i);
    if (s.type === "train") {
      if (s.data_from === "previous" && (i === 0 || p.stages[i - 1].type !== "rejection_sample"))
        errors.push({ loc: ["stages", i, "data_from"], msg: `Value error, stage '${s.name}': data_from='previous' needs a rejection_sample stage right before it`, type: "value_error" });
      if (!s.init_from.trim()) errors.push({ loc: ["stages", i, "init_from"], msg: "init_from must be 'previous', 'base' or a path", type: "missing" });
    }
    if (s.type === "rejection_sample") {
      if (s.num_samples < 1) errors.push({ loc: ["stages", i, "num_samples"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
      if (s.max_per_prompt < 1) errors.push({ loc: ["stages", i, "max_per_prompt"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
      if (s.max_per_prompt > s.num_samples) errors.push({ loc: ["stages", i, "max_per_prompt"], msg: "max_per_prompt cannot exceed num_samples", type: "value_error" });
    }
    if (s.type === "eval" && s.num_samples < 1) errors.push({ loc: ["stages", i, "num_samples"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
  });
  const sameIdx = (loc: (string | number)[], i: number) => loc[0] === "stages" && loc[1] === i;
  // A train stage reading a rejection-sampled dataset should be SFT (or at least not rely on the RL rewards it drops).
  p.stages.forEach((s, i) => {
    if (s.type !== "train" || s.data_from !== "previous" || errors.some((e) => sameIdx(e.loc, i))) return;
    const algo = effectiveAlgorithm(p.base, s);
    if (algo !== "sft") errors.push({ loc: ["stages", i, "overrides"], msg: `Value error, stage '${s.name}' trains on rejection-sampled responses but algorithm is '${algo}' (expected sft)`, type: "value_error" });
  });
  return errors;
}

/** Algorithm a train stage ends up with after overrides. */
export function effectiveAlgorithm(base: RunConfig, s: TrainStage): string {
  const algo = s.overrides.algorithm;
  if (isPlainObject(algo) && typeof algo.name === "string") return algo.name;
  return base.algorithm.name;
}

/** Short human summary of a train stage's overrides (`algorithm.name=sft · trainer.max_steps=50`). */
export function describeOverrides(overrides: JsonDict, limit = 4): string {
  const flat: string[] = [];
  const walk = (node: unknown, prefix: string) => {
    if (isPlainObject(node)) for (const [k, v] of Object.entries(node)) walk(v, prefix ? `${prefix}.${k}` : k);
    else flat.push(`${prefix}=${typeof node === "string" ? node : JSON.stringify(node)}`);
  };
  walk(overrides, "");
  if (flat.length === 0) return "no overrides";
  return flat.slice(0, limit).join(" · ") + (flat.length > limit ? ` · +${flat.length - limit}` : "");
}

/** DeepSeek-R1 style loop: cold-start SFT → GRPO → rejection sampling → SFT → GRPO. */
export function r1LoopStages(): Stage[] {
  return [
    { type: "train", name: "cold-start-sft", overrides: { algorithm: { name: "sft" }, trainer: { max_steps: 50, learning_rate: 1e-5 } }, init_from: "base", data_from: "config" },
    { type: "train", name: "grpo-1", overrides: { algorithm: { name: "grpo" }, trainer: { max_steps: 200 } }, init_from: "previous", data_from: "config" },
    { ...defaultRejectionStage("rejection-sample"), num_samples: 8, min_reward: 1.0, max_per_prompt: 2 },
    { type: "train", name: "sft-2", overrides: { algorithm: { name: "sft" }, trainer: { max_steps: 100, learning_rate: 5e-6 } }, init_from: "previous", data_from: "previous" },
    { type: "train", name: "grpo-2", overrides: { algorithm: { name: "grpo" }, trainer: { max_steps: 300 } }, init_from: "previous", data_from: "config" },
  ];
}
