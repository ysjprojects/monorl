import type { ReactNode } from "react";
import { Card, Field, NumberInput, Pill, Select, Toggle } from "../components/ui";
import { PRESETS } from "../defaults";
import { ALGORITHMS, PIPELINE_BACKENDS, ROLLOUT_BACKENDS, TRAINER_BACKENDS, type AlgorithmName } from "../types";
import { backendAvailability, type StepProps } from "./common";

const ALGO_NOTES: Record<AlgorithmName, string> = {
  grpo: "Group-relative advantages (mean/std over the G samples of a prompt). Needs group_size ≥ 2.",
  dapo: "GRPO + clip-higher, dynamic sampling and overlong penalty (ByteDance DAPO).",
  rloo: "Leave-one-out baseline per group; no std normalisation.",
  reinforce: "Raw rewards as advantages; simplest and highest variance.",
  ppo: "Clipped surrogate with a value head (GAE). Set lam / value_coef.",
  sft: "Supervised fine-tuning on the dataset responses: no rollouts, no rewards, no environment. Loss = token NLL of the response span.",
};

export function SftNote({ children }: { children?: ReactNode }) {
  return (
    <div className="rounded-md border border-amber-900/60 bg-amber-950/30 px-3 py-2 text-xs text-amber-200">
      <span className="font-semibold">algorithm.name = sft</span> — {children ?? "this run performs no rollouts and computes no rewards; rollout, rewards and environment settings are ignored."}
    </div>
  );
}

export default function TrainingStep({ config, set, patch, system }: StepProps) {
  const a = config.algorithm;
  const r = config.rollout;
  const t = config.trainer;
  const pipeline = PIPELINE_BACKENDS[t.backend] === true;
  const sft = a.name === "sft";
  return (
    <div className="space-y-4">
      <Card title="Presets">
        <div className="grid gap-3 md:grid-cols-3">
          {PRESETS.map((p) => (
            <button key={p.name} type="button" onClick={() => patch(p.patch)} className="rounded-md border border-slate-800 bg-slate-900 p-3 text-left hover:border-sky-600">
              <div className="text-sm font-medium text-slate-100">{p.name}</div>
              <div className="mt-1 text-xs text-slate-500">{p.description}</div>
            </button>
          ))}
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Algorithm">
          <Field path="algorithm.name" label="Objective" description={ALGO_NOTES[a.name]}>
            <Select<AlgorithmName> value={a.name} onChange={(v) => set("algorithm.name", v)} options={ALGORITHMS.map((v) => ({ value: v }))} />
          </Field>
          {sft ? (
            <div className="mt-3 space-y-3" data-testid="sft-fields">
              <SftNote />
              <div className="grid grid-cols-2 gap-3">
                <Field path="algorithm.sft_label_smoothing" label="Label smoothing">
                  <NumberInput value={a.sft_label_smoothing} onChange={(v) => set("algorithm.sft_label_smoothing", v ?? 0)} min={0} max={0.99} step="any" />
                </Field>
                <Field path="algorithm.loss_agg" label="Loss aggregation">
                  <Select value={a.loss_agg} onChange={(v) => set("algorithm.loss_agg", v)} options={["token", "sequence"].map((v) => ({ value: v as typeof a.loss_agg }))} />
                </Field>
              </div>
              <Toggle path="algorithm.sft_train_on_prompt" checked={a.sft_train_on_prompt} onChange={(v) => set("algorithm.sft_train_on_prompt", v)} label="Train on prompt tokens" />
              <div className="text-xs text-slate-500">
                The target response comes from <span className="font-mono">data.response_field</span> (Data step), else <span className="font-mono">answer_field</span>, else the trailing assistant turn of a chat sample.
              </div>
            </div>
          ) : (
            <>
              <div className="mt-3 grid grid-cols-2 gap-3">
                <Field path="rollout.group_size" label="Group size (G)">
                  <NumberInput value={r.group_size} onChange={(v) => set("rollout.group_size", v ?? 4)} min={1} />
                </Field>
                <Field path="rollout.max_new_tokens" label="Max new tokens">
                  <NumberInput value={r.max_new_tokens} onChange={(v) => set("rollout.max_new_tokens", v ?? 256)} min={1} />
                </Field>
                <Field path="algorithm.kl_coef" label="KL coef">
                  <NumberInput value={a.kl_coef} onChange={(v) => set("algorithm.kl_coef", v ?? 0)} min={0} step="any" />
                </Field>
                <Field path="algorithm.clip_eps" label="Clip ε">
                  <NumberInput value={a.clip_eps} onChange={(v) => set("algorithm.clip_eps", v ?? 0.2)} step="any" />
                </Field>
                {a.name === "dapo" && (
                  <>
                    <Field path="algorithm.clip_eps_high" label="Clip ε high">
                      <NumberInput value={a.clip_eps_high} onChange={(v) => set("algorithm.clip_eps_high", v)} nullable step="any" placeholder="= clip_eps" />
                    </Field>
                    <Field path="algorithm.overlong_penalty" label="Overlong penalty">
                      <NumberInput value={a.overlong_penalty} onChange={(v) => set("algorithm.overlong_penalty", v ?? 0)} min={0} step="any" />
                    </Field>
                  </>
                )}
                {a.name === "ppo" && (
                  <>
                    <Field path="algorithm.lam" label="GAE λ">
                      <NumberInput value={a.lam} onChange={(v) => set("algorithm.lam", v ?? 1)} min={0} max={1} step="any" />
                    </Field>
                    <Field path="algorithm.value_coef" label="Value coef">
                      <NumberInput value={a.value_coef} onChange={(v) => set("algorithm.value_coef", v ?? 0.5)} min={0} step="any" />
                    </Field>
                  </>
                )}
                <Field path="rollout.temperature" label="Temperature">
                  <NumberInput value={r.temperature} onChange={(v) => set("rollout.temperature", v ?? 1)} min={0} step="any" />
                </Field>
                <Field path="algorithm.kl_estimator" label="KL estimator">
                  <Select value={a.kl_estimator} onChange={(v) => set("algorithm.kl_estimator", v)} options={["k1", "k2", "k3"].map((v) => ({ value: v as typeof a.kl_estimator }))} />
                </Field>
                <Field path="algorithm.advantage_norm" label="Advantage norm">
                  <Select value={a.advantage_norm} onChange={(v) => set("algorithm.advantage_norm", v)} options={["group", "batch", "none"].map((v) => ({ value: v as typeof a.advantage_norm }))} />
                </Field>
                <Field path="algorithm.loss_agg" label="Loss aggregation">
                  <Select value={a.loss_agg} onChange={(v) => set("algorithm.loss_agg", v)} options={["token", "sequence"].map((v) => ({ value: v as typeof a.loss_agg }))} />
                </Field>
              </div>
              <div className="mt-3 space-y-2">
                <Toggle path="algorithm.advantage_std_norm" checked={a.advantage_std_norm} onChange={(v) => set("algorithm.advantage_std_norm", v)} label="Std-normalise advantages" />
                <Toggle path="algorithm.dynamic_sampling" checked={a.dynamic_sampling} onChange={(v) => set("algorithm.dynamic_sampling", v)} label="Dynamic sampling" />
              </div>
            </>
          )}
        </Card>

        <Card title="Optimisation">
          <div className="grid grid-cols-2 gap-3">
            <Field path="trainer.learning_rate" label="Learning rate">
              <NumberInput value={t.learning_rate} onChange={(v) => set("trainer.learning_rate", v ?? 1e-6)} step="any" />
            </Field>
            <Field path="trainer.max_steps" label="Max steps">
              <NumberInput value={t.max_steps} onChange={(v) => set("trainer.max_steps", v ?? 100)} min={1} />
            </Field>
            <Field
              path="trainer.prompts_per_step"
              label={sft ? "Sequences / step" : "Prompts / step"}
              description={sft ? `${t.prompts_per_step} dataset samples per optimizer step` : `${t.prompts_per_step} × ${r.group_size} = ${t.prompts_per_step * r.group_size} sequences per step`}
            >
              <NumberInput value={t.prompts_per_step} onChange={(v) => set("trainer.prompts_per_step", v ?? 8)} min={1} />
            </Field>
            <Field path="trainer.micro_batch_size" label="Micro batch">
              <NumberInput value={t.micro_batch_size} onChange={(v) => set("trainer.micro_batch_size", v ?? 4)} min={1} />
            </Field>
            <Field path="algorithm.num_epochs" label="Epochs / batch">
              <NumberInput value={a.num_epochs} onChange={(v) => set("algorithm.num_epochs", v ?? 1)} min={1} />
            </Field>
            <Field path="trainer.save_every" label="Save every">
              <NumberInput value={t.save_every} onChange={(v) => set("trainer.save_every", v ?? 50)} min={1} />
            </Field>
            {sft && (
              <Field path="trainer.eval_every" label="Eval every" description="Held-out NLL (eval/loss) on data.eval_uri every N steps.">
                <NumberInput value={t.eval_every} onChange={(v) => set("trainer.eval_every", v)} nullable min={1} placeholder="off" />
              </Field>
            )}
            <Field path="trainer.precision" label="Precision">
              <Select value={t.precision} onChange={(v) => set("trainer.precision", v)} options={["fp32", "bf16", "fp16"].map((v) => ({ value: v as typeof t.precision }))} />
            </Field>
            <Field path="trainer.scheduler" label="Scheduler">
              <Select value={t.scheduler} onChange={(v) => set("trainer.scheduler", v)} options={["constant", "linear", "cosine"].map((v) => ({ value: v as typeof t.scheduler }))} />
            </Field>
            <Field path="trainer.max_seq_len" label="Max seq len">
              <NumberInput value={t.max_seq_len} onChange={(v) => set("trainer.max_seq_len", v ?? 2048)} min={16} />
            </Field>
            <Field path="trainer.max_grad_norm" label="Max grad norm">
              <NumberInput value={t.max_grad_norm} onChange={(v) => set("trainer.max_grad_norm", v ?? 1)} step="any" />
            </Field>
          </div>
          <div className="mt-3">
            <Toggle checked={t.gradient_checkpointing} onChange={(v) => set("trainer.gradient_checkpointing", v)} label="Gradient checkpointing" />
          </div>
        </Card>
      </div>

      <Card title="Backends">
        <div className="grid gap-4 md:grid-cols-2">
          {sft ? (
            <div className="text-xs text-slate-500">
              <div className="mb-1 text-xs font-medium text-slate-300">Rollout engine</div>
              Not used by SFT — no completions are sampled during training.
            </div>
          ) : (
            <Field path="rollout.backend" label="Rollout engine">
              <div className="space-y-1">
                {ROLLOUT_BACKENDS.map((b) => {
                  const av = backendAvailability(system, b);
                  return (
                    <label key={b} className={`flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm ${r.backend === b ? "border-sky-600 bg-sky-950/40" : "border-slate-800"} ${pipeline ? "opacity-60" : ""}`}>
                      <input type="radio" name="rollout" checked={r.backend === b} onChange={() => set("rollout.backend", b)} disabled={pipeline} />
                      <span className="font-mono">{b}</span>
                      <span className="ml-auto">{av.ok ? <Pill tone="green">{av.note || "available"}</Pill> : <Pill tone="amber">{av.note}</Pill>}</span>
                    </label>
                  );
                })}
                {pipeline && <div className="text-xs text-slate-500">Pipeline backends own rollout; this selection is compiled into their native inference config.</div>}
              </div>
            </Field>
          )}
          <Field path="trainer.backend" label="Trainer">
            <div className="space-y-1">
              {TRAINER_BACKENDS.map((b) => {
                const av = backendAvailability(system, b);
                return (
                  <label key={b} className={`flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm ${t.backend === b ? "border-sky-600 bg-sky-950/40" : "border-slate-800"}`}>
                    <input type="radio" name="trainer" checked={t.backend === b} onChange={() => set("trainer.backend", b)} />
                    <span className="font-mono">{b}</span>
                    {PIPELINE_BACKENDS[b] && <span className="text-xs text-slate-500">{sft ? "pipeline · SFTTrainer / sft runner" : "pipeline"}</span>}
                    <span className="ml-auto">{av.ok ? <Pill tone="green">{av.note || "available"}</Pill> : <Pill tone="amber">{av.note}</Pill>}</span>
                  </label>
                );
              })}
            </div>
          </Field>
        </div>
        {!system && <div className="mt-2 text-xs text-slate-500">Availability unknown until /api/system responds.</div>}
      </Card>
    </div>
  );
}
