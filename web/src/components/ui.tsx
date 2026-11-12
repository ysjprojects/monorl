import { useContext, useState, type ReactNode } from "react";
import { parse as parseYaml, stringify as toYaml } from "yaml";
import { SchemaContext, schemaProp } from "../schema";
import type { RunState } from "../types";

const STATE_CLASSES: Record<string, string> = {
  pending: "bg-slate-700 text-slate-200",
  running: "bg-sky-900 text-sky-200 ring-1 ring-sky-500/50",
  succeeded: "bg-emerald-900 text-emerald-200",
  failed: "bg-rose-900 text-rose-200",
  cancelled: "bg-amber-900 text-amber-200",
};

export function StateBadge({ state }: { state: RunState | string }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${STATE_CLASSES[state] ?? STATE_CLASSES.pending}`}>
      {state === "running" && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-sky-400" />}
      {state}
    </span>
  );
}

export function Pill({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "green" | "red" | "amber" | "sky" }) {
  const cls: Record<string, string> = {
    slate: "bg-slate-800 text-slate-300",
    green: "bg-emerald-900/60 text-emerald-300",
    red: "bg-rose-900/60 text-rose-300",
    amber: "bg-amber-900/60 text-amber-300",
    sky: "bg-sky-900/60 text-sky-300",
  };
  return <span className={`inline-block rounded px-1.5 py-0.5 text-xs ${cls[tone]}`}>{children}</span>;
}

export function ErrorBox({ error, className = "" }: { error: string | null | undefined; className?: string }) {
  if (!error) return null;
  return <div className={`rounded-md border border-rose-800 bg-rose-950/60 px-3 py-2 text-sm text-rose-200 ${className}`}>{error}</div>;
}

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-slate-600 border-t-sky-400" />
      {label}
    </div>
  );
}

export function Card({ title, children, actions, className = "" }: { title?: ReactNode; children: ReactNode; actions?: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { id: T; label: ReactNode }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="flex gap-1 border-b border-slate-800">
      {tabs.map((t) => (
        <button
          key={t.id}
          type="button"
          onClick={() => onChange(t.id)}
          className={`-mb-px border-b-2 px-3 py-1.5 text-sm ${value === t.id ? "border-sky-500 text-sky-300" : "border-transparent text-slate-400 hover:text-slate-200"}`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

// ---- schema-aware form fields ----------------------------------------------------------------

interface FieldProps {
  /** Dotted RunConfig path; used to pull the description from /api/config/schema. */
  path?: string;
  label: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  className?: string;
  error?: string | null;
}

export function Field({ path, label, description, children, className = "", error }: FieldProps) {
  const schema = useContext(SchemaContext);
  const desc = description ?? (path ? schemaProp(schema, path)?.description : undefined);
  return (
    <label className={`block ${className}`}>
      <span className="mb-1 block text-xs font-medium text-slate-300">{label}</span>
      {children}
      {desc && <span className="mt-1 block text-xs text-slate-500">{desc}</span>}
      {error && <span className="mt-1 block text-xs text-rose-400">{error}</span>}
    </label>
  );
}

export function TextInput({ value, onChange, nullable = false, ...rest }: { value: string | null; onChange: (v: string | null) => void; nullable?: boolean; placeholder?: string; className?: string; disabled?: boolean; type?: string; id?: string }) {
  return (
    <input
      {...rest}
      className={`input ${rest.className ?? ""}`}
      value={value ?? ""}
      onChange={(e) => onChange(nullable && e.target.value === "" ? null : e.target.value)}
    />
  );
}

export function NumberInput({ value, onChange, nullable = false, step, min, max, placeholder, className = "", disabled }: { value: number | null; onChange: (v: number | null) => void; nullable?: boolean; step?: number | "any"; min?: number; max?: number; placeholder?: string; className?: string; disabled?: boolean }) {
  // Keep the raw text so users can type "1e-6" or "0." without the field snapping back.
  const [text, setText] = useState<string | null>(null);
  const shown = text ?? (value == null ? "" : String(value));
  return (
    <input
      type="text"
      inputMode="decimal"
      className={`input font-mono ${className}`}
      value={shown}
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e) => {
        const t = e.target.value;
        setText(t);
        if (t.trim() === "") {
          if (nullable) onChange(null);
          return;
        }
        const n = Number(t);
        if (Number.isFinite(n)) onChange(n);
      }}
      onBlur={() => setText(null)}
    />
  );
}

export function Select<T extends string>({ value, onChange, options, className = "", disabled }: { value: T; onChange: (v: T) => void; options: { value: T; label?: ReactNode; disabled?: boolean }[]; className?: string; disabled?: boolean }) {
  return (
    <select className={`input ${className}`} value={value} onChange={(e) => onChange(e.target.value as T)} disabled={disabled}>
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>
          {o.label ?? o.value}
        </option>
      ))}
    </select>
  );
}

export function Toggle({ checked, onChange, label, description, path }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode; description?: ReactNode; path?: string }) {
  const schema = useContext(SchemaContext);
  const desc = description ?? (path ? schemaProp(schema, path)?.description : undefined);
  return (
    <label className="flex cursor-pointer items-start gap-3">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition ${checked ? "bg-sky-600" : "bg-slate-700"}`}
      >
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition ${checked ? "left-4.5" : "left-0.5"}`} />
      </button>
      <span>
        <span className="block text-sm text-slate-200">{label}</span>
        {desc && <span className="block text-xs text-slate-500">{desc}</span>}
      </span>
    </label>
  );
}

/** Textarea bound to a JSON object; shows a parse error instead of committing invalid text. */
export function JsonEditor({ value, onChange, rows = 4, placeholder }: { value: Record<string, unknown>; onChange: (v: Record<string, unknown>) => void; rows?: number; placeholder?: string }) {
  const [text, setText] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const shown = text ?? JSON.stringify(value, null, 2);
  return (
    <div>
      <textarea
        className={`input font-mono text-xs ${err ? "border-rose-700" : ""}`}
        rows={rows}
        value={shown}
        placeholder={placeholder}
        spellCheck={false}
        onChange={(e) => {
          setText(e.target.value);
          try {
            const parsed: unknown = JSON.parse(e.target.value || "{}");
            if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
              onChange(parsed as Record<string, unknown>);
              setErr(null);
            } else setErr("must be a JSON object");
          } catch (ex) {
            setErr(ex instanceof Error ? ex.message : "invalid JSON");
          }
        }}
        onBlur={() => {
          if (!err) setText(null);
        }}
      />
      {err && <div className="mt-1 text-xs text-rose-400">{err}</div>}
    </div>
  );
}

export function Kv({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-slate-500">{k}</dt>
          <dd className="truncate font-mono text-slate-200">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * Textarea bound to a YAML document. `shape` restricts what is accepted ("object" = mapping,
 * "array" = sequence); an empty document maps to `empty`. Invalid text shows the parse error and
 * is not committed.
 */
export function YamlEditor<T>({ value, onChange, shape, empty, rows = 6, placeholder, className = "" }: { value: T; onChange: (v: T) => void; shape: "object" | "array"; empty: T; rows?: number; placeholder?: string; className?: string }) {
  const [text, setText] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const isEmpty = value == null || (Array.isArray(value) ? value.length === 0 : typeof value === "object" && Object.keys(value as object).length === 0);
  const shown = text ?? (isEmpty ? "" : toYaml(value));
  return (
    <div className={className}>
      <textarea
        className={`input font-mono text-xs ${err ? "border-rose-700" : ""}`}
        rows={rows}
        value={shown}
        placeholder={placeholder}
        spellCheck={false}
        onChange={(e) => {
          const t = e.target.value;
          setText(t);
          if (t.trim() === "") {
            onChange(empty);
            setErr(null);
            return;
          }
          try {
            const parsed: unknown = parseYaml(t);
            const ok = shape === "array" ? Array.isArray(parsed) : !!parsed && typeof parsed === "object" && !Array.isArray(parsed);
            if (!ok) setErr(shape === "array" ? "must be a YAML list" : "must be a YAML mapping");
            else {
              onChange(parsed as T);
              setErr(null);
            }
          } catch (ex) {
            setErr(ex instanceof Error ? ex.message : "invalid YAML");
          }
        }}
        onBlur={() => {
          if (!err) setText(null);
        }}
      />
      {err && <div className="mt-1 text-xs text-rose-400">{err}</div>}
    </div>
  );
}
