import { useCallback, useState } from "react";
import { api } from "../api";
import { deepMerge, defaultRunConfig, fromPartial } from "../defaults";
import { useAsync } from "../hooks";
import { setPath } from "../schema";
import type { Recipe, RunConfig } from "../types";
import { STEPS, type StepId } from "../wizard/common";
import ComputeStep from "../wizard/ComputeStep";
import DataStep from "../wizard/DataStep";
import ModelStep from "../wizard/ModelStep";
import Recipes from "../wizard/Recipes";
import ReviewStep from "../wizard/ReviewStep";
import RewardsEnvStep from "../wizard/RewardsEnvStep";
import TrainingStep from "../wizard/TrainingStep";

const DRAFT_KEY = "monorl.newrun.draft";

function loadDraft(): RunConfig {
  try {
    const raw = sessionStorage.getItem(DRAFT_KEY);
    if (raw) return fromPartial(JSON.parse(raw));
  } catch {
    /* ignore corrupt draft */
  }
  return defaultRunConfig();
}

export default function NewRun() {
  const [config, setConfig] = useState<RunConfig>(loadDraft);
  const [step, setStep] = useState<StepId>("model");
  const [recipeName, setRecipeName] = useState<string | null>(null);
  const system = useAsync(() => api.system(), []);
  const envs = useAsync(() => api.envs(), []);

  const commit = useCallback((next: RunConfig) => {
    setConfig(next);
    sessionStorage.setItem(DRAFT_KEY, JSON.stringify(next));
  }, []);
  const set = useCallback((path: string, value: unknown) => setConfig((c) => {
    const next = setPath(c, path, value);
    sessionStorage.setItem(DRAFT_KEY, JSON.stringify(next));
    return next;
  }), []);
  const patch = useCallback((partial: unknown) => setConfig((c) => {
    const next = deepMerge(c, partial);
    sessionStorage.setItem(DRAFT_KEY, JSON.stringify(next));
    return next;
  }), []);

  function pickRecipe(r: Recipe) {
    commit(fromPartial(r.config));
    setRecipeName(r.name);
    setStep("review");
  }

  function reset() {
    commit(defaultRunConfig());
    setRecipeName(null);
    setStep("model");
  }

  const idx = STEPS.findIndex((s) => s.id === step);
  const props = { config, set, patch, system: system.data, envs: envs.data };
  const stepDone: Record<StepId, boolean> = {
    model: (config.model.name_or_path ?? "").trim().length > 0,
    data: (config.data.uri ?? "").trim().length > 0,
    training: true,
    rewards: config.rewards.length > 0 || config.env.type !== "single_turn",
    compute: true,
    review: false,
  };

  return (
    <div className="mx-auto max-w-7xl space-y-4 p-5">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">
          New run {recipeName && <span className="ml-2 font-mono text-sm text-slate-500">from recipe {recipeName}</span>}
        </h1>
        <button type="button" className="btn" onClick={reset}>
          Reset
        </button>
      </div>

      <Recipes onPick={pickRecipe} />

      <ol className="flex flex-wrap gap-1">
        {STEPS.map((s, i) => (
          <li key={s.id}>
            <button
              type="button"
              onClick={() => setStep(s.id)}
              className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm ${s.id === step ? "bg-slate-800 text-white" : "text-slate-400 hover:text-slate-200"}`}
            >
              <span className={`flex h-5 w-5 items-center justify-center rounded-full text-xs ${stepDone[s.id] ? "bg-emerald-700 text-white" : i < idx ? "bg-amber-700 text-white" : "bg-slate-700 text-slate-200"}`}>{i + 1}</span>
              {s.label}
            </button>
          </li>
        ))}
      </ol>

      <div>
        {step === "model" && <ModelStep {...props} />}
        {step === "data" && <DataStep {...props} />}
        {step === "training" && <TrainingStep {...props} />}
        {step === "rewards" && <RewardsEnvStep {...props} />}
        {step === "compute" && <ComputeStep {...props} />}
        {step === "review" && <ReviewStep {...props} replace={commit} />}
      </div>

      <div className="flex justify-between border-t border-slate-800 pt-4">
        <button type="button" className="btn" disabled={idx === 0} onClick={() => setStep(STEPS[idx - 1].id)}>
          ← {idx > 0 ? STEPS[idx - 1].label : "Back"}
        </button>
        {idx < STEPS.length - 1 && (
          <button type="button" className="btn btn-primary" onClick={() => setStep(STEPS[idx + 1].id)}>
            {STEPS[idx + 1].label} →
          </button>
        )}
      </div>
    </div>
  );
}
