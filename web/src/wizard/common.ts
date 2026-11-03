import type { EnvsResponse, RunConfig, SystemReport } from "../types";

export interface StepProps {
  config: RunConfig;
  /** Set a dotted path (e.g. "trainer.learning_rate") immutably. */
  set: (path: string, value: unknown) => void;
  /** Deep-merge a partial RunConfig. */
  patch: (partial: unknown) => void;
  system: SystemReport | null;
  envs: EnvsResponse | null;
}

export const STEPS = [
  { id: "model", label: "Model" },
  { id: "data", label: "Data" },
  { id: "training", label: "Training" },
  { id: "rewards", label: "Rewards & Env" },
  { id: "compute", label: "Compute" },
  { id: "review", label: "Review" },
] as const;

export type StepId = (typeof STEPS)[number]["id"];

/** Availability of a backend on this machine, derived from the doctor report. */
export function backendAvailability(system: SystemReport | null, backend: string): { ok: boolean; note: string } {
  if (!system) return { ok: true, note: "" };
  const fw: Record<string, string> = { vllm: "vllm", sglang: "sglang", trl: "trl", verl: "verl", nemo_rl: "nemo_rl", unsloth: "unsloth" };
  const name = fw[backend];
  if (!name) return { ok: true, note: "built-in" };
  const info = system.frameworks?.[name];
  if (!info) return { ok: false, note: "unknown" };
  if (!info.installed) return { ok: false, note: "not installed" };
  const total = info.patches?.length ?? 0;
  const applicable = info.patches?.filter((p) => p.applicable).length ?? 0;
  return { ok: true, note: `v${info.version}${total ? ` · ${applicable}/${total} patches applicable` : ""}` };
}
