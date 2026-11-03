import { useState } from "react";
import { api } from "../api";
import { Card, ErrorBox, Field, NumberInput, Pill, Select, Spinner, TextInput, Toggle } from "../components/ui";
import { fmtCompact } from "../format";
import { useAsync, useDebounced } from "../hooks";
import type { StepProps } from "./common";

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
      </div>
    </div>
  );
}
