import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { parse as parseYaml, stringify as toYaml } from "yaml";
import { ApiError, api } from "../api";
import { Card, ErrorBox, Field, Pill, Spinner, TextInput } from "../components/ui";
import { fromPartial } from "../defaults";
import { useDebounced } from "../hooks";
import { checkPipeline, pipelineFromPartial } from "../pipeline";
import { isPipelineDoc, type PipelineConfig, type RunConfig, type ValidationError } from "../types";
import type { StepProps } from "./common";

function errorsFromApi(e: unknown): ValidationError[] {
  // POST /api/jobs returns 422 with {detail: [errors]} when the config is invalid.
  if (e instanceof ApiError && e.body && typeof e.body === "object" && "detail" in e.body && Array.isArray(e.body.detail)) {
    return e.body.detail.filter((x): x is ValidationError => !!x && typeof x === "object" && "msg" in x);
  }
  return [];
}

/** The document under review: a RunConfig, or a PipelineConfig whose `base` is the wizard config. */
type Doc = { kind: "run"; config: RunConfig } | { kind: "pipeline"; pipeline: PipelineConfig };

interface ReviewProps extends StepProps {
  replace: (c: RunConfig) => void;
  /** Present in pipeline mode. */
  pipeline?: { stages: PipelineConfig["stages"]; name: string | null; setName: (n: string | null) => void; replace: (p: PipelineConfig) => void };
}

type Validation = { doc: unknown; errors: ValidationError[]; source: "server" | "client" };

export default function ReviewStep({ config, patch, replace, pipeline }: ReviewProps) {
  const navigate = useNavigate();
  const [raw, setRaw] = useState(false);
  const [text, setText] = useState("");
  const [yamlErr, setYamlErr] = useState<string | null>(null);
  // Validation result tagged with the document it was computed for; `validating` is derived.
  const [validation, setValidation] = useState<Validation | null>(null);
  const [launching, setLaunching] = useState(false);
  const [launchErr, setLaunchErr] = useState<string | null>(null);

  const doc: Doc = useMemo(
    () => (pipeline ? { kind: "pipeline", pipeline: { name: pipeline.name, base: config, stages: pipeline.stages } } : { kind: "run", config }),
    [config, pipeline],
  );
  const payload = doc.kind === "pipeline" ? doc.pipeline : doc.config;
  const yamlText = useMemo(() => toYaml(payload), [payload]);
  const debounced = useDebounced(payload, 400);
  const validating = validation?.doc !== debounced || debounced !== payload;
  const errors = validation?.errors ?? null;

  useEffect(() => {
    let cancelled = false;
    const done = (errors: ValidationError[], source: Validation["source"]) => {
      if (!cancelled) setValidation({ doc: debounced, errors, source });
    };
    if (isPipelineDoc(debounced)) {
      const p = debounced;
      api
        .validatePipeline(p)
        .then((r) => done(r.ok ? [] : r.errors, "server"))
        .catch((e) => {
          // Servers without pipeline support reject the payload shape (422 for the missing
          // `config`, 404/400 before the route exists): fall back to the structural checks.
          if (e instanceof ApiError && [400, 404, 422].includes(e.status)) done(checkPipeline(p), "client");
          else done([{ loc: [], msg: `validation request failed: ${e instanceof Error ? e.message : String(e)}` }], "server");
        });
    } else {
      api
        .validate(debounced)
        .then((r) => done(r.ok ? [] : r.errors, "server"))
        .catch((e) => done([{ loc: [], msg: `validation request failed: ${e instanceof Error ? e.message : String(e)}` }], "server"));
    }
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
      if (pipeline) {
        if (!isPipelineDoc(parsed)) throw new Error("a pipeline document needs `base:` and `stages:`");
        pipeline.replace(pipelineFromPartial(parsed));
      } else {
        if (isPipelineDoc(parsed)) throw new Error("this is a pipeline document; switch the wizard to Pipeline mode to edit it");
        replace(fromPartial(parsed));
      }
      setYamlErr(null);
    } catch (e) {
      setYamlErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function launch() {
    setLaunching(true);
    setLaunchErr(null);
    try {
      const job = doc.kind === "pipeline" ? await api.createPipelineJob(doc.pipeline) : await api.createJob(doc.config);
      navigate(`/runs/${job.id}`);
    } catch (e) {
      const errs = errorsFromApi(e);
      if (errs.length) setValidation({ doc: payload, errors: errs, source: "server" });
      setLaunchErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLaunching(false);
    }
  }

  const ok = errors !== null && errors.length === 0 && !yamlErr;
  const isPipeline = doc.kind === "pipeline";
  const stageCount = isPipeline ? doc.pipeline.stages.length : 0;
  return (
    <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
      <Card
        title={isPipeline ? `Pipeline configuration (${stageCount} stage${stageCount === 1 ? "" : "s"})` : "Run configuration"}
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
        <Card title={isPipeline ? "Pipeline" : "Run"}>
          {pipeline ? (
            <Field label="Pipeline name" description="Defaults to '<model>-pipeline-<timestamp>'. Stage run dirs live under <run_root>/<name>/stages/.">
              <TextInput value={pipeline.name} onChange={pipeline.setName} nullable placeholder="auto" className="font-mono" />
            </Field>
          ) : (
            <Field path="run_name" label="Run name">
              <TextInput value={config.run_name} onChange={(v) => patch({ run_name: v })} nullable placeholder="auto: <model>-<algorithm>-<timestamp>" className="font-mono" />
            </Field>
          )}
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Field path="seed" label="Seed">
              <TextInput value={String(config.seed)} onChange={(v) => patch({ seed: Number(v) || 0 })} className="font-mono" />
            </Field>
            <Field path="logging.run_root" label="Run root">
              <TextInput value={config.logging.run_root} onChange={(v) => patch({ logging: { run_root: v ?? "~/.monorl/runs" } })} className="font-mono" />
            </Field>
          </div>
          {isPipeline && (
            <ol className="mt-3 space-y-1 text-xs">
              {doc.pipeline.stages.map((s, i) => (
                <li key={i} className="flex items-center gap-2">
                  <span className="w-4 text-right text-slate-500">{i + 1}</span>
                  <Pill tone={s.type === "train" ? "sky" : s.type === "rejection_sample" ? "amber" : "green"}>{s.type}</Pill>
                  <span className="font-mono text-slate-200">{s.name}</span>
                </li>
              ))}
            </ol>
          )}
        </Card>

        <Card
          title="Validation"
          actions={validating ? <Spinner label="validating…" /> : ok ? <Pill tone="green">valid</Pill> : errors && errors.length > 0 ? <Pill tone="red">{errors.length} error(s)</Pill> : null}
        >
          {validation?.source === "client" && !validating && (
            <div className="mb-2 rounded-md border border-amber-900/60 bg-amber-950/30 px-2 py-1 text-xs text-amber-200">
              The server rejected the pipeline payload shape (no pipeline validation yet); showing client-side structural checks only. Launch will still post the pipeline.
            </div>
          )}
          {errors && errors.length > 0 && (
            <ul className="space-y-1 text-sm">
              {errors.map((e, i) => (
                <li key={i} className="rounded-md border border-rose-900 bg-rose-950/40 px-2 py-1">
                  <span className="font-mono text-xs text-rose-300">{e.loc?.length ? e.loc.join(".") : isPipeline ? "pipeline" : "config"}</span>
                  <span className="ml-2 text-rose-100">{e.msg}</span>
                </li>
              ))}
            </ul>
          )}
          {ok && (
            <div className="text-sm text-slate-400">
              {isPipeline
                ? `${validation?.source === "client" ? "Structure looks valid." : "Server accepted the pipeline."} Launch spawns \`python -m monorl.cli pipeline\`; stages run in order in one run directory.`
                : "Server accepted the configuration. Launch spawns `python -m monorl.cli train` in a fresh run directory."}
            </div>
          )}
          <ErrorBox error={launchErr} className="mt-3" />
          <button type="button" className="btn btn-primary mt-4 w-full justify-center" disabled={!ok || launching || validating} onClick={launch}>
            {launching ? "Launching…" : isPipeline ? "Launch pipeline" : "Launch run"}
          </button>
        </Card>
      </div>
    </div>
  );
}
