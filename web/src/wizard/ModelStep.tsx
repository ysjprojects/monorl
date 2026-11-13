import { useState } from "react";
import { api } from "../api";
import { Card, ErrorBox, Field, NumberInput, Pill, Select, Spinner, TextInput, Toggle } from "../components/ui";
import { defaultMtpConfig, isPlainObject, normalizeMtp } from "../defaults";
import { fmtCompact } from "../format";
import { useAsync, useDebounced } from "../hooks";
import type { MtpConfig } from "../types";
import type { StepProps } from "./common";

function MtpError({ children }: { children: string | null }) {
  return children ? <div className="mt-1 text-xs text-rose-400">{children}</div> : null;
}

/** `model.mtp`, with the cross-field rules of `RunConfig._check_mtp` (src/monorl/config/schema.py) inline. */
function MtpCard({ config, set }: Pick<StepProps, "config" | "set">) {
  const mtp = config.model.mtp ?? defaultMtpConfig();
  const update = (change: Partial<MtpConfig>) => set("model.mtp", normalizeMtp({ ...mtp, ...change }));
  const trainer = config.trainer.backend;
  const engine = config.rollout.backend;
  const minP = Number(config.rollout.extra.min_p ?? 0);
  const megatron = config.trainer.extra.megatron;
  const megatronReady = isPlainObject(megatron) && megatron.enabled === true && Number(megatron.mtp_num_layers ?? 0) > 0;

  const backendError =
    trainer === "unsloth"
      ? "Not supported with trainer.backend 'unsloth': Unsloth loads the model itself and trains MTP heads with a trainer of its own. Use the reference trainer."
      : trainer === "trl"
        ? "Not supported with trainer.backend 'trl': TRL neither trains MTP layers nor drafts with them. Use the reference trainer."
        : null;
  // verl and NeMo-RL run their own rollouts on vLLM / SGLang, whatever rollout.backend says.
  const served = engine === "vllm" || engine === "sglang" ? engine : trainer === "verl" || trainer === "nemo_rl" ? trainer : null;
  const rolloutError =
    mtp.rollout && served && minP > 0
      ? `Drafting on ${served} cannot be combined with rollout.extra.min_p (${minP}): vLLM rejects it under speculative decoding and SGLang's verification does not apply it, so the sampled distribution would differ from the recorded logprobs. Remove min_p${served === engine ? " or use the hf engine" : ""}.`
      : null;
  const trainError =
    mtp.train && trainer === "verl" ? "Not available with trainer.backend 'verl': verl trains MTP layers only with its Megatron engine, and monorl compiles the FSDP actor. Drafting during rollouts still works." : null;

  return (
    <Card title="Multi-token prediction (MTP)">
      <Toggle
        checked={mtp.enabled}
        onChange={(v) => set("model.mtp", v ? { ...defaultMtpConfig(), enabled: true } : undefined)}
        label="Use the model's MTP layers"
        description="Some models (DeepSeek-V3, GLM, Step, Mellum, …) ship extra layers that guess the tokens after the next one. On: they are loaded next to the policy and kept in weight sync, checkpoints and the final export. Needs a model that ships MTP layers."
      />
      {mtp.enabled && (
        <div className="mt-3 space-y-3">
          <MtpError>{backendError}</MtpError>
          <div>
            <Toggle
              checked={mtp.rollout}
              onChange={(v) => update({ rollout: v })}
              label="Draft during rollouts"
              description="The MTP layers guess several tokens ahead and the policy checks every guess (speculative decoding). Drafting never changes what is sampled: tokens and logprobs are the policy's, only generation gets faster."
            />
            <MtpError>{rolloutError}</MtpError>
          </div>
          <Field
            label="Tokens drafted per step"
            description="Empty = the engine's default: one per MTP layer on the hf engine, 1 on vLLM, SGLang's own default."
            error={mtp.num_speculative_tokens != null && !(Number.isInteger(mtp.num_speculative_tokens) && mtp.num_speculative_tokens >= 1) ? "Has to be a whole number of at least 1, or empty." : null}
          >
            <NumberInput value={mtp.num_speculative_tokens} onChange={(v) => update({ num_speculative_tokens: v })} nullable min={1} placeholder="engine default" disabled={!mtp.rollout} />
            {mtp.rollout && engine === "hf" && (
              <span className="mt-1 block text-xs text-amber-300">On the hf engine this has to be empty or exactly the model's number of MTP layers; any other value fails when the engine starts.</span>
            )}
          </Field>
          <div>
            <Toggle
              checked={mtp.train}
              onChange={(v) => update({ train: v })}
              label="Train the MTP layers"
              description="Adds an auxiliary loss on the response tokens of every batch (RL and SFT) that teaches the MTP layers to predict the tokens after the next one, so their drafts keep matching a policy that changes."
            />
            <MtpError>{trainError}</MtpError>
            {mtp.train && trainer === "nemo_rl" && !megatronReady && (
              <div className="mt-1 text-xs text-amber-300">
                NeMo-RL trains MTP layers only on its Megatron path: set <code>trainer.extra.megatron.enabled: true</code> and <code>trainer.extra.megatron.mtp_num_layers</code> (the model's number of MTP layers) in the YAML on the Review step.
              </div>
            )}
          </div>
          {mtp.train && (
            <>
              <Field label="MTP loss weight" description="Weight of the MTP loss in the total loss." error={mtp.loss_coef < 0 ? "Has to be 0 or more." : null}>
                <NumberInput value={mtp.loss_coef} onChange={(v) => update({ loss_coef: v ?? 0.1 })} min={0} step="any" />
              </Field>
              <Toggle
                checked={mtp.detach_trunk}
                onChange={(v) => update({ detach_trunk: v })}
                label="Update the MTP layers only"
                description="On: the MTP loss changes the MTP layers and nothing else. Off: it also flows into the policy's own parameters, as in pretraining."
              />
            </>
          )}
        </div>
      )}
    </Card>
  );
}

export default function ModelStep({ config, set }: StepProps) {
  const [query, setQuery] = useState("");
  const q = useDebounced(query, 300);
  const results = useAsync(() => (q.trim() ? api.searchModels(q.trim(), 15) : Promise.resolve([])), [q]);
  const selected = config.model.name_or_path;
  const info = useAsync(() => (selected && !selected.startsWith("/") && !selected.startsWith("~") && !selected.startsWith(".") ? api.modelInfo(selected) : Promise.resolve(null)), [selected]);
  const m = config.model;

  return (
    <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
      <Card title="Hugging Face Hub">
        <input className="input" placeholder="Search text-generation models (e.g. Qwen2.5, SmolLM2)…" value={query} onChange={(e) => setQuery(e.target.value)} autoFocus />
        <ErrorBox error={results.error} className="mt-2" />
        {results.loading && q && <div className="mt-2"><Spinner label="Searching…" /></div>}
        {results.data && results.data.length > 0 && (
          <ul className="mt-2 max-h-80 divide-y divide-slate-800 overflow-auto rounded-md border border-slate-800">
            {results.data.map((r) => (
              <li key={r.id}>
                <button
                  type="button"
                  onClick={() => set("model.name_or_path", r.id)}
                  className={`flex w-full items-center justify-between gap-3 px-3 py-2 text-left hover:bg-slate-800 ${selected === r.id ? "bg-sky-950/60" : ""}`}
                >
                  <span className="min-w-0">
                    <span className="block truncate font-mono text-sm text-slate-100">{r.id}</span>
                    <span className="block truncate text-xs text-slate-500">{r.tags?.slice(0, 5).join(" · ")}</span>
                  </span>
                  <span className="shrink-0 text-right text-xs text-slate-400">
                    <span className="block">↓ {fmtCompact(r.downloads)}</span>
                    <span className="block">♥ {fmtCompact(r.likes)}</span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
        {results.data && results.data.length === 0 && q && !results.loading && <div className="mt-2 text-xs text-slate-500">No models found.</div>}
      </Card>

      <div className="space-y-4">
        <Card title="Selected model">
          <Field path="model.name_or_path" label="Model id or local path">
            <TextInput value={m.name_or_path} onChange={(v) => set("model.name_or_path", v ?? "")} placeholder="org/model or /path/to/checkpoint" className="font-mono" />
          </Field>
          {info.data && (
            <div className="mt-2 flex flex-wrap gap-1 text-xs">
              {typeof info.data.architecture === "string" && <Pill>{info.data.architecture}</Pill>}
              {typeof info.data.params === "number" && <Pill>{fmtCompact(info.data.params)} params</Pill>}
              {info.data.gated === true && <Pill tone="amber">gated — needs HF token</Pill>}
            </div>
          )}
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Field path="model.revision" label="Revision">
              <TextInput value={m.revision} onChange={(v) => set("model.revision", v)} nullable placeholder="main" />
            </Field>
            <Field path="model.dtype" label="dtype">
              <Select value={m.dtype} onChange={(v) => set("model.dtype", v)} options={["auto", "float32", "bfloat16", "float16"].map((v) => ({ value: v as typeof m.dtype }))} />
            </Field>
            <Field path="model.attn_implementation" label="Attention impl">
              <TextInput value={m.attn_implementation} onChange={(v) => set("model.attn_implementation", v)} nullable placeholder="auto" />
            </Field>
            <Field path="model.tokenizer_name_or_path" label="Tokenizer override">
              <TextInput value={m.tokenizer_name_or_path} onChange={(v) => set("model.tokenizer_name_or_path", v)} nullable placeholder="same as model" />
            </Field>
          </div>
          <div className="mt-3">
            <Toggle checked={m.trust_remote_code} onChange={(v) => set("model.trust_remote_code", v)} label="Trust remote code" description="Required for some custom architectures on the Hub." />
          </div>
        </Card>

        <Card title="LoRA">
          <Toggle path="model.lora.enabled" checked={m.lora.enabled} onChange={(v) => set("model.lora.enabled", v)} label="Train LoRA adapters" />
          {m.lora.enabled && (
            <div className="mt-3 grid grid-cols-3 gap-3">
              <Field path="model.lora.r" label="rank">
                <NumberInput value={m.lora.r} onChange={(v) => set("model.lora.r", v ?? 16)} min={1} />
              </Field>
              <Field path="model.lora.alpha" label="alpha">
                <NumberInput value={m.lora.alpha} onChange={(v) => set("model.lora.alpha", v ?? 32)} min={1} />
              </Field>
              <Field path="model.lora.dropout" label="dropout">
                <NumberInput value={m.lora.dropout} onChange={(v) => set("model.lora.dropout", v ?? 0)} min={0} max={1} step="any" />
              </Field>
              <Field path="model.lora.target_modules" label="Target modules" className="col-span-3">
                <TextInput
                  value={Array.isArray(m.lora.target_modules) ? m.lora.target_modules.join(", ") : m.lora.target_modules}
                  onChange={(v) => {
                    const t = (v ?? "").trim();
                    set("model.lora.target_modules", t === "" || t === "all-linear" ? "all-linear" : t.split(/[,\s]+/).filter(Boolean));
                  }}
                  placeholder="all-linear or q_proj, v_proj"
                  className="font-mono"
                />
              </Field>
            </div>
          )}
        </Card>

        <MtpCard config={config} set={set} />
      </div>
    </div>
  );
}
