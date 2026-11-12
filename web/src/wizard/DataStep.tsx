import { useState } from "react";
import { api } from "../api";
import { Card, ErrorBox, Field, NumberInput, Pill, Select, Spinner, Tabs, TextInput, Toggle } from "../components/ui";
import { fmtBytes, fmtCompact } from "../format";
import { useAsync, useDebounced } from "../hooks";
import { isImageCell, type BrowseEntry, type DataFormat, type JsonValue, type PreviewResponse } from "../types";
import type { StepProps } from "./common";

const IMAGE_EXT = /\.(png|jpe?g|webp)(\?.*)?$/i;

/** `image_field` is a column or a list of columns; the input edits it as a comma list. */
function imageFieldText(v: string | string[] | null): string {
  return Array.isArray(v) ? v.join(", ") : (v ?? "");
}

function parseImageField(text: string): string | string[] | null {
  const parts = text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length === 0) return null;
  return parts.length === 1 ? parts[0] : parts;
}

/** Preview table cell: decoded image placeholders become a chip, image URIs a thumbnail, the rest text. */
export function PreviewCell({ value }: { value: JsonValue | undefined }) {
  if (isImageCell(value))
    return (
      <span className="inline-flex items-center gap-1 rounded border border-sky-800 bg-sky-950/60 px-1.5 py-0.5 text-sky-200" title={[value.path, value.mode ? `mode ${value.mode}` : null].filter(Boolean).join(" · ") || undefined}>
        <span aria-hidden>🖼</span>[image {value.width}×{value.height}]
      </span>
    );
  if (typeof value === "string" && (value.startsWith("data:image/") || (/^https?:\/\//.test(value) && IMAGE_EXT.test(value))))
    return (
      <span className="inline-flex items-center gap-2">
        <img src={value} alt="" loading="lazy" className="h-12 w-12 rounded border border-slate-700 object-cover" />
        {!value.startsWith("data:") && <span className="max-w-[10rem] truncate text-slate-500" title={value}>{value.split("/").pop()}</span>}
      </span>
    );
  if (typeof value === "string" && IMAGE_EXT.test(value))
    return (
      <span className="inline-flex items-center gap-1 rounded border border-slate-700 bg-slate-800 px-1.5 py-0.5 text-slate-300" title={value}>
        <span aria-hidden>🖼</span>[image {value.split("/").pop()}]
      </span>
    );
  if (Array.isArray(value) && value.length > 0 && value.every(isImageCell))
    return (
      <span className="inline-flex flex-wrap gap-1">
        {value.map((v, i) => (
          <PreviewCell key={i} value={v} />
        ))}
      </span>
    );
  return <div className="line-clamp-3 whitespace-pre-wrap">{typeof value === "string" ? value : JSON.stringify(value)}</div>;
}

type SourceTab = "local" | "s3" | "hf";

function tabForUri(uri: string): SourceTab {
  if (uri.startsWith("s3://")) return "s3";
  if (uri.startsWith("hf://")) return "hf";
  return "local";
}

function parentUri(uri: string): string {
  const trimmed = uri.replace(/\/+$/, "");
  const idx = trimmed.lastIndexOf("/");
  if (uri.startsWith("s3://")) return idx <= 4 ? "s3://" : trimmed.slice(0, idx);
  return idx <= 0 ? "/" : trimmed.slice(0, idx);
}

function Browser({ start, onPick, onPickDir }: { start: string; onPick: (e: BrowseEntry) => void; onPickDir?: (uri: string) => void }) {
  const [cwd, setCwd] = useState(start);
  const [typed, setTyped] = useState(start);
  const listing = useAsync(() => api.browse(cwd), [cwd]);
  return (
    <div>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (typed === cwd) listing.reload();
          else setCwd(typed);
        }}
      >
        <button type="button" className="btn" onClick={() => setCwd(parentUri(cwd))} title="Up">
          ↑
        </button>
        <input className="input font-mono" value={typed} onChange={(e) => setTyped(e.target.value)} />
        <button type="submit" className="btn">
          Go
        </button>
        {onPickDir && (
          <button type="button" className="btn" onClick={() => onPickDir(cwd)} title="Use this directory as the dataset">
            Use dir
          </button>
        )}
      </form>
      <ErrorBox error={listing.error} className="mt-2" />
      {listing.loading && !listing.data && <div className="mt-2"><Spinner /></div>}
      {listing.data && (
        <ul className="mt-2 max-h-72 divide-y divide-slate-800 overflow-auto rounded-md border border-slate-800 font-mono text-xs">
          {listing.data.length === 0 && <li className="px-3 py-2 text-slate-500">empty</li>}
          {listing.data.map((e) => (
            <li key={e.uri}>
              <button
                type="button"
                className="flex w-full items-center justify-between px-3 py-1.5 text-left hover:bg-slate-800"
                onClick={() => {
                  if (e.is_dir) {
                    setCwd(e.uri);
                    setTyped(e.uri);
                  } else onPick(e);
                }}
              >
                <span className={e.is_dir ? "text-sky-300" : "text-slate-200"}>
                  {e.is_dir ? "📁 " : "📄 "}
                  {e.name}
                </span>
                <span className="text-slate-500">{fmtBytes(e.size)}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function DataStep({ config, set, patch }: StepProps) {
  const d = config.data;
  const [tab, setTab] = useState<SourceTab>(tabForUri(d.uri));
  const [hfQuery, setHfQuery] = useState("");
  const hq = useDebounced(hfQuery, 300);
  const hfResults = useAsync(() => (hq.trim() ? api.searchDatasets(hq.trim(), 15) : Promise.resolve([])), [hq]);
  const connectors = useAsync(() => api.connectors(), []);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [previewedUri, setPreviewedUri] = useState<string | null>(null);

  async function runPreview(uri: string, applyInferred: boolean) {
    if (!uri) return;
    setPreviewing(true);
    setPreviewErr(null);
    try {
      const p = await api.preview({ uri, split: d.split, hf_config_name: d.hf_config_name, limit: 5 });
      setPreview(p);
      setPreviewedUri(uri);
      if (applyInferred && p.inferred) {
        const images = p.inferred.image_fields ?? [];
        patch({
          data: {
            format: p.inferred.format ?? "auto",
            prompt_field: p.inferred.prompt_field ?? null,
            answer_field: p.inferred.answer_field ?? null,
            messages_field: p.inferred.messages_field ?? null,
            // `null` would be kept by deepMerge; only overwrite when the sniffer found image columns.
            ...(images.length ? { image_field: images.length === 1 ? images[0] : images } : {}),
          },
        });
      }
    } catch (e) {
      setPreview(null);
      setPreviewErr(e instanceof Error ? e.message : String(e));
    } finally {
      setPreviewing(false);
    }
  }

  // Picking from a browser/search result previews immediately; typed URIs preview on demand.
  const pickUri = (uri: string) => {
    set("data.uri", uri);
    setPreview(null);
    setPreviewedUri(null);
    void runPreview(uri, true);
  };
  // Browser is keyed on its start uri so it re-roots when the selected dataset changes source.
  const localStart = tabForUri(d.uri) === "local" && d.uri ? parentUri(d.uri) : "~";
  const s3Start = tabForUri(d.uri) === "s3" ? parentUri(d.uri) : "s3://";

  const columns = preview ? Object.keys(preview.columns) : [];
  const colOptions = [{ value: "", label: "(none)" }, ...columns.map((c) => ({ value: c }))];
  const s3 = connectors.data?.s3;
  const sft = config.algorithm.name === "sft";
  // Image columns: what the sniffer reported, plus columns typed as images or holding decoded image cells.
  const imageColumns = preview
    ? [...new Set([...(preview.inferred?.image_fields ?? []), ...columns.filter((c) => /image/i.test(preview.columns[c] ?? "") || preview.rows.some((r) => isImageCell(r[c])))])]
    : [];

  return (
    <div className="space-y-4">
      <Card title="Source">
        <Tabs<SourceTab>
          tabs={[
            { id: "local", label: "Local file system" },
            { id: "s3", label: "S3" },
            { id: "hf", label: "HF Hub" },
          ]}
          value={tab}
          onChange={setTab}
        />
        <div className="mt-3">
          {tab === "local" && <Browser key={localStart} start={localStart} onPick={(e) => pickUri(e.uri)} onPickDir={pickUri} />}
          {tab === "s3" && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                {connectors.loading && <Spinner label="Checking credentials…" />}
                {s3 && (
                  <>
                    {s3.configured ? <Pill tone="green">S3 credentials detected</Pill> : <Pill tone="amber">no AWS credentials in the server environment</Pill>}
                    {s3.region && <Pill>region {s3.region}</Pill>}
                    {s3.endpoint_url && <Pill>endpoint {s3.endpoint_url}</Pill>}
                    {s3.vars_present.map((v) => (
                      <Pill key={v} tone="sky">
                        {v}
                      </Pill>
                    ))}
                  </>
                )}
                <ErrorBox error={connectors.error} />
              </div>
              {s3 && !s3.configured && <div className="text-xs text-slate-500">Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or AWS_PROFILE) and optionally AWS_DEFAULT_REGION / AWS_ENDPOINT_URL before starting the server. Values are never shown here.</div>}
              <Browser key={s3Start} start={s3Start} onPick={(e) => pickUri(e.uri)} onPickDir={pickUri} />
            </div>
          )}
          {tab === "hf" && (
            <div>
              <input className="input" placeholder="Search Hub datasets (e.g. gsm8k)…" value={hfQuery} onChange={(e) => setHfQuery(e.target.value)} />
              <ErrorBox error={hfResults.error} className="mt-2" />
              {hfResults.data && hfResults.data.length > 0 && (
                <ul className="mt-2 max-h-64 divide-y divide-slate-800 overflow-auto rounded-md border border-slate-800">
                  {hfResults.data.map((r) => (
                    <li key={r.id}>
                      <button type="button" className="flex w-full items-center justify-between px-3 py-2 text-left hover:bg-slate-800" onClick={() => pickUri(`hf://${r.id}`)}>
                        <span className="font-mono text-sm">{r.id}</span>
                        <span className="text-xs text-slate-400">
                          ↓ {fmtCompact(r.downloads)} · ♥ {fmtCompact(r.likes)}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              <div className="mt-3 grid grid-cols-2 gap-3">
                <Field path="data.split" label="Split">
                  <TextInput value={d.split} onChange={(v) => set("data.split", v ?? "train")} />
                </Field>
                <Field path="data.hf_config_name" label="Config / subset">
                  <TextInput value={d.hf_config_name} onChange={(v) => set("data.hf_config_name", v)} nullable placeholder="default" />
                </Field>
              </div>
            </div>
          )}
        </div>
        <div className="mt-4 flex items-end gap-2">
          <Field path="data.uri" label="Dataset URI" className="flex-1">
            <TextInput value={d.uri} onChange={(v) => set("data.uri", v ?? "")} className="font-mono" placeholder="hf://org/name · s3://bucket/key · /path/file.jsonl" />
          </Field>
          <button type="button" className="btn btn-primary mb-5" disabled={!d.uri || previewing} onClick={() => runPreview(d.uri, true)}>
            {previewing ? "Loading…" : "Preview"}
          </button>
        </div>
        <ErrorBox error={previewErr} />
      </Card>

      {preview && (
        <Card
          title={
            <span>
              Preview <span className="font-mono text-xs text-slate-500">{previewedUri}</span>
              {preview.num_rows != null && <span className="ml-2 text-xs text-slate-500">{fmtCompact(preview.num_rows)} rows</span>}
            </span>
          }
        >
          <div className="overflow-auto rounded-md border border-slate-800">
            <table className="w-full text-xs">
              <thead className="bg-slate-900 text-left text-slate-400">
                <tr>
                  {columns.map((c) => (
                    <th key={c} className="px-2 py-1 font-mono">
                      {c} <span className="text-slate-600">{preview.columns[c]}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {preview.rows.map((row, i) => (
                  <tr key={i} className="align-top">
                    {columns.map((c) => (
                      <td key={c} className="max-w-xs px-2 py-1 font-mono text-slate-300">
                        <PreviewCell value={row[c]} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      <Card title="Field mapping">
        <div className="grid gap-3 md:grid-cols-4">
          <Field path="data.format" label="Format">
            <Select<DataFormat> value={d.format} onChange={(v) => set("data.format", v)} options={["auto", "chat", "prompt_answer", "text"].map((v) => ({ value: v as DataFormat }))} />
          </Field>
          <Field path="data.prompt_field" label="Prompt field">
            {columns.length ? (
              <Select value={d.prompt_field ?? ""} onChange={(v) => set("data.prompt_field", v || null)} options={colOptions} />
            ) : (
              <TextInput value={d.prompt_field} onChange={(v) => set("data.prompt_field", v)} nullable placeholder="auto" />
            )}
          </Field>
          <Field path="data.answer_field" label="Answer field">
            {columns.length ? (
              <Select value={d.answer_field ?? ""} onChange={(v) => set("data.answer_field", v || null)} options={colOptions} />
            ) : (
              <TextInput value={d.answer_field} onChange={(v) => set("data.answer_field", v)} nullable placeholder="auto" />
            )}
          </Field>
          <Field path="data.messages_field" label="Messages field">
            {columns.length ? (
              <Select value={d.messages_field ?? ""} onChange={(v) => set("data.messages_field", v || null)} options={colOptions} />
            ) : (
              <TextInput value={d.messages_field} onChange={(v) => set("data.messages_field", v)} nullable placeholder="auto" />
            )}
          </Field>
          <Field path="data.image_field" label="Image field(s)" description="Column(s) holding images (HF Image feature, path/URI, bytes or base64); comma-separate several. Attached to the user message as image parts.">
            <TextInput value={imageFieldText(d.image_field)} onChange={(v) => set("data.image_field", parseImageField(v ?? ""))} placeholder={imageColumns.length ? imageColumns.join(", ") : "none (text-only)"} className="font-mono" />
          </Field>
          {sft && (
            <Field path="data.response_field" label="Response field" description="Target response column for SFT (defaults to answer_field / the trailing assistant turn).">
              {columns.length ? (
                <Select value={d.response_field ?? ""} onChange={(v) => set("data.response_field", v || null)} options={colOptions} />
              ) : (
                <TextInput value={d.response_field} onChange={(v) => set("data.response_field", v)} nullable placeholder="auto" />
              )}
            </Field>
          )}
          <Field path="data.system_prompt" label="System prompt" className="md:col-span-2">
            <textarea className="input" rows={2} value={d.system_prompt ?? ""} onChange={(e) => set("data.system_prompt", e.target.value || null)} placeholder="optional" />
          </Field>
          <Field path="data.prompt_template" label="Prompt template" className="md:col-span-2">
            <textarea className="input font-mono" rows={2} value={d.prompt_template ?? ""} onChange={(e) => set("data.prompt_template", e.target.value || null)} placeholder="{prompt}" />
          </Field>
          <Field path="data.max_prompt_tokens" label="Max prompt tokens">
            <NumberInput value={d.max_prompt_tokens} onChange={(v) => set("data.max_prompt_tokens", v ?? 1024)} min={1} />
          </Field>
          <Field path="data.limit" label="Limit samples">
            <NumberInput value={d.limit} onChange={(v) => set("data.limit", v)} nullable min={1} placeholder="all" />
          </Field>
          <div className="flex flex-col justify-end gap-2 md:col-span-2">
            <Toggle checked={d.shuffle} onChange={(v) => set("data.shuffle", v)} label="Shuffle" />
            <Toggle path="data.streaming" checked={d.streaming} onChange={(v) => set("data.streaming", v)} label="Streaming" />
          </div>
        </div>
      </Card>
    </div>
  );
}
