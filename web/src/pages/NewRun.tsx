import { useCallback, useState } from "react";
import { api } from "../api";
import { deepMerge, defaultRunConfig, fromPartial, isPlainObject } from "../defaults";
import { useAsync } from "../hooks";
import { pipelineFromPartial } from "../pipeline";
import { setPath } from "../schema";
import { recipeIsPipeline, type PipelineConfig, type Recipe, type RunConfig, type Stage } from "../types";
import { stepsFor, type StepId, type WizardMode } from "../wizard/common";
import ComputeStep from "../wizard/ComputeStep";
import DataStep from "../wizard/DataStep";
import ModelStep from "../wizard/ModelStep";
import Recipes from "../wizard/Recipes";
import ReviewStep from "../wizard/ReviewStep";
import RewardsEnvStep from "../wizard/RewardsEnvStep";
import StagesStep from "../wizard/StagesStep";
import TrainingStep from "../wizard/TrainingStep";

const DRAFT_KEY = "monorl.newrun.draft";

/** Wizard state: the base/run config, plus the pipeline-only parts kept even in run mode. */
interface Draft {
  mode: WizardMode;
  config: RunConfig;
  stages: Stage[];
  pipelineName: string | null;
}

function loadDraft(): Draft {
  try {
    const raw = sessionStorage.getItem(DRAFT_KEY);
    if (raw) {
      const parsed: unknown = JSON.parse(raw);
      if (isPlainObject(parsed) && "mode" in parsed) {
        const p = pipelineFromPartial({ base: parsed.config, stages: parsed.stages, name: parsed.pipelineName });
        return { mode: parsed.mode === "pipeline" ? "pipeline" : "run", config: p.base, stages: p.stages, pipelineName: p.name };
      }
      // Pre-pipeline drafts stored the bare RunConfig.
      return { mode: "run", config: fromPartial(parsed), stages: [], pipelineName: null };
    }
  } catch {
    /* ignore corrupt draft */
  }
  return { mode: "run", config: defaultRunConfig(), stages: [], pipelineName: null };
}

export default function NewRun() {
  const [draft, setDraft] = useState<Draft>(loadDraft);
  const [step, setStep] = useState<StepId>("model");
  const [recipeName, setRecipeName] = useState<string | null>(null);
  const system = useAsync(() => api.system(), []);
  const envs = useAsync(() => api.envs(), []);

  const update = useCallback((fn: (d: Draft) => Draft) => {
    setDraft((d) => {
      const next = fn(d);
      sessionStorage.setItem(DRAFT_KEY, JSON.stringify(next));
      return next;
    });
  }, []);
  const commit = useCallback((config: RunConfig) => update((d) => ({ ...d, config })), [update]);
  const set = useCallback((path: string, value: unknown) => update((d) => ({ ...d, config: setPath(d.config, path, value) })), [update]);
  const patch = useCallback((partial: unknown) => update((d) => ({ ...d, config: deepMerge(d.config, partial) })), [update]);
  const setStages = useCallback((stages: Stage[]) => update((d) => ({ ...d, stages })), [update]);
  const setPipelineName = useCallback((pipelineName: string | null) => update((d) => ({ ...d, pipelineName })), [update]);
  const replacePipeline = useCallback((p: PipelineConfig) => update((d) => ({ ...d, config: p.base, stages: p.stages, pipelineName: p.name })), [update]);
  const setMode = (mode: WizardMode) => {
    update((d) => ({ ...d, mode }));
    if (mode === "run" && step === "stages") setStep("review");
  };

  function pickRecipe(r: Recipe) {
    if (recipeIsPipeline(r)) {
      const p = pipelineFromPartial(r.config);
      update((d) => ({ ...d, mode: "pipeline", config: p.base, stages: p.stages, pipelineName: p.name }));
      setStep("stages");
    } else {
      update((d) => ({ ...d, mode: "run", config: fromPartial(r.config) }));
      setStep("review");
    }
    setRecipeName(r.name);
  }

  function reset() {
    update((d) => ({ mode: d.mode, config: defaultRunConfig(), stages: [], pipelineName: null }));
    setRecipeName(null);
    setStep("model");
  }

  const { mode, config, stages, pipelineName } = draft;
  const steps = stepsFor(mode);
  const idx = steps.findIndex((s) => s.id === step);
  const props = { config, set, patch, system: system.data, envs: envs.data };
  const sft = config.algorithm.name === "sft";
  const stepDone: Record<StepId, boolean> = {
    model: (config.model.name_or_path ?? "").trim().length > 0,
    data: (config.data.uri ?? "").trim().length > 0,
    training: true,
    rewards: sft || config.rewards.length > 0 || config.env.type !== "single_turn",
    compute: true,
    stages: stages.length > 0,
    review: false,
  };
  const pipeline: PipelineConfig = { name: pipelineName, base: config, stages };

  return (
    <div className="mx-auto max-w-7xl space-y-4 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-lg font-semibold">
          {mode === "pipeline" ? "New pipeline" : "New run"} {recipeName && <span className="ml-2 font-mono text-sm text-slate-500">from recipe {recipeName}</span>}
        </h1>
        <div className="flex items-center gap-3">
          <div className="flex rounded-md border border-slate-700 p-0.5 text-sm" role="radiogroup" aria-label="Wizard mode">
            {(
              [
                { id: "run", label: "Single run" },
                { id: "pipeline", label: "Pipeline (SFT → RL loop)" },
              ] as { id: WizardMode; label: string }[]
            ).map((m) => (
              <button
                key={m.id}
                type="button"
                role="radio"
                aria-checked={mode === m.id}
                onClick={() => setMode(m.id)}
                className={`rounded px-3 py-1 ${mode === m.id ? "bg-sky-700 text-white" : "text-slate-400 hover:text-slate-200"}`}
              >
                {m.label}
              </button>
            ))}
          </div>
          <button type="button" className="btn" onClick={reset}>
            Reset
          </button>
        </div>
      </div>
      {mode === "pipeline" && (
        <div className="text-xs text-slate-500">
          Steps 1–5 define the <span className="font-mono">base</span> config every stage starts from; each train stage can override any subset of it. Stage order = execution order.
        </div>
      )}

      <Recipes onPick={pickRecipe} />

      <ol className="flex flex-wrap gap-1">
        {steps.map((s, i) => (
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
        {step === "stages" && <StagesStep pipeline={pipeline} setStages={setStages} />}
        {step === "review" && (
          <ReviewStep {...props} replace={commit} pipeline={mode === "pipeline" ? { stages, name: pipelineName, setName: setPipelineName, replace: replacePipeline } : undefined} />
        )}
      </div>

      <div className="flex justify-between border-t border-slate-800 pt-4">
        <button type="button" className="btn" disabled={idx <= 0} onClick={() => setStep(steps[idx - 1].id)}>
          ← {idx > 0 ? steps[idx - 1].label : "Back"}
        </button>
        {idx >= 0 && idx < steps.length - 1 && (
          <button type="button" className="btn btn-primary" onClick={() => setStep(steps[idx + 1].id)}>
            {steps[idx + 1].label} →
          </button>
        )}
      </div>
    </div>
  );
}
