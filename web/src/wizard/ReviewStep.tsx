import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { parse as parseYaml, stringify as toYaml } from "yaml";
import { ApiError, api } from "../api";
import { Card, ErrorBox, Field, Pill, Spinner, TextInput } from "../components/ui";
import { fromPartial } from "../defaults";
import { useDebounced } from "../hooks";
import type { RunConfig, ValidationError } from "../types";
import type { StepProps } from "./common";

function errorsFromApi(e: unknown): ValidationError[] {
  // POST /api/jobs returns 422 with {detail: [errors]} when the config is invalid.
  if (e instanceof ApiError && e.body && typeof e.body === "object" && "detail" in e.body && Array.isArray(e.body.detail)) {
    return e.body.detail.filter((x): x is ValidationError => !!x && typeof x === "object" && "msg" in x);
  }
  return [];
}

export default function ReviewStep({ config, patch, replace }: StepProps & { replace: (c: RunConfig) => void }) {
  const navigate = useNavigate();
  const [raw, setRaw] = useState(false);
  const [text, setText] = useState("");
  const [yamlErr, setYamlErr] = useState<string | null>(null);
  // Validation result tagged with the config it was computed for; `validating` is derived.
  const [validation, setValidation] = useState<{ config: RunConfig; errors: ValidationError[] } | null>(null);
  const [launching, setLaunching] = useState(false);
  const [launchErr, setLaunchErr] = useState<string | null>(null);

  const yamlText = useMemo(() => toYaml(config), [config]);
  const debounced = useDebounced(config, 400);
  const validating = validation?.config !== debounced || debounced !== config;
  const errors = validation?.errors ?? null;

  useEffect(() => {
    let cancelled = false;
    api
      .validate(debounced)
      .then((r) => {
        if (!cancelled) setValidation({ config: debounced, errors: r.ok ? [] : r.errors });
      })
      .catch((e) => {
        if (!cancelled) setValidation({ config: debounced, errors: [{ loc: [], msg: `validation request failed: ${e instanceof Error ? e.message : String(e)}` }] });
      });
    return () => {
      cancelled = true;
    };
  }, [debounced]);

  function enterRaw() {
    setText(yamlText);
    setYamlErr(null);
    setRaw(true);
  }

  function applyRaw(t: string) {
    setText(t);
    try {
      const parsed: unknown = parseYaml(t);
      if (!parsed || typeof parsed !== "object") throw new Error("YAML must be a mapping");
      replace(fromPartial(parsed));
      setYamlErr(null);
    } catch (e) {
      setYamlErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function launch() {
    setLaunching(true);
    setLaunchErr(null);
    try {
      const job = await api.createJob(config);
      navigate(`/runs/${job.id}`);
    } catch (e) {
      const errs = errorsFromApi(e);
      if (errs.length) setValidation({ config, errors: errs });
      setLaunchErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLaunching(false);
    }
  }

  const ok = errors !== null && errors.length === 0 && !yamlErr;
  return (
    <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
      <Card
        title="Run configuration"
        actions={
          <div className="flex gap-2">
            <button type="button" className={`btn ${!raw ? "btn-primary" : ""}`} onClick={() => setRaw(false)}>
              Rendered
            </button>
            <button type="button" className={`btn ${raw ? "btn-primary" : ""}`} onClick={enterRaw}>
              Edit YAML
            </button>
          </div>
        }
      >
        {raw ? (
          <>
            <textarea className={`input h-[32rem] font-mono text-xs ${yamlErr ? "border-rose-700" : ""}`} value={text} spellCheck={false} onChange={(e) => applyRaw(e.target.value)} />
            <ErrorBox error={yamlErr} className="mt-2" />
          </>
        ) : (
          <pre className="h-[32rem] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-3 font-mono text-xs text-slate-300">{yamlText}</pre>
        )}
      </Card>

      <div className="space-y-4">
        <Card title="Run">
          <Field path="run_name" label="Run name">
            <TextInput value={config.run_name} onChange={(v) => patch({ run_name: v })} nullable placeholder="auto: <model>-<algorithm>-<timestamp>" className="font-mono" />
          </Field>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Field path="seed" label="Seed">
              <TextInput value={String(config.seed)} onChange={(v) => patch({ seed: Number(v) || 0 })} className="font-mono" />
            </Field>
            <Field path="logging.run_root" label="Run root">
              <TextInput value={config.logging.run_root} onChange={(v) => patch({ logging: { run_root: v ?? "~/.monorl/runs" } })} className="font-mono" />
            </Field>
          </div>
        </Card>

        <Card
          title="Validation"
          actions={validating ? <Spinner label="validating…" /> : ok ? <Pill tone="green">valid</Pill> : errors && errors.length > 0 ? <Pill tone="red">{errors.length} error(s)</Pill> : null}
        >
          {errors && errors.length > 0 && (
            <ul className="space-y-1 text-sm">
              {errors.map((e, i) => (
                <li key={i} className="rounded-md border border-rose-900 bg-rose-950/40 px-2 py-1">
                  <span className="font-mono text-xs text-rose-300">{e.loc?.length ? e.loc.join(".") : "config"}</span>
                  <span className="ml-2 text-rose-100">{e.msg}</span>
                </li>
              ))}
            </ul>
          )}
          {ok && <div className="text-sm text-slate-400">Server accepted the configuration. Launch spawns `python -m monorl.cli train` in a fresh run directory.</div>}
          <ErrorBox error={launchErr} className="mt-3" />
          <button type="button" className="btn btn-primary mt-4 w-full justify-center" disabled={!ok || launching || validating} onClick={launch}>
            {launching ? "Launching…" : "Launch run"}
          </button>
        </Card>
      </div>
    </div>
  );
}
