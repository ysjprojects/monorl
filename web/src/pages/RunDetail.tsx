import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { stringify as toYaml } from "yaml";
import { api, jobPath, stagesStreamUrl } from "../api";
import { RunStreams } from "../components/RunStreams";
import { Card, ErrorBox, Kv, Pill, Spinner, StateBadge } from "../components/ui";
import { fmtDate, fmtDuration, fmtNum } from "../format";
import { useAsync, useLiveSnapshot } from "../hooks";
import { EVAL_METRICS, REJECTION_METRICS, RL_METRICS, SFT_METRICS } from "../metrics";
import { effectiveAlgorithm } from "../pipeline";
import { isPipelineJob, jobRunConfig, TERMINAL_STATES, type JobDetail, type JsonValue, type PipelineJob, type StageStatus, type StageType } from "../types";

const STAGE_TONE: Record<StageType, "sky" | "amber" | "green"> = { train: "sky", rejection_sample: "amber", eval: "green" };

/** Route element: remount the page per job id so per-job stream state starts fresh. */
export default function RunDetailRoute() {
  const { id = "" } = useParams();
  return <RunDetail key={id} id={id} />;
}

function RunDetail({ id }: { id: string }) {
  const [cancelErr, setCancelErr] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [jobTerminal, setJobTerminal] = useState(false);
  const job = useAsync(() => api.job(id), [id], jobTerminal ? 15000 : 2000);
  // useAsync keeps the last good payload on transient errors, so this never flips back to false.
  const terminal = !!job.data && TERMINAL_STATES[job.data.state] === true;
  if (terminal !== jobTerminal) setJobTerminal(terminal);

  const j = job.data;
  const state = j?.status?.state ?? j?.state ?? "pending";

  async function cancel() {
    setCancelling(true);
    setCancelErr(null);
    try {
      await api.cancelJob(id);
      job.reload();
    } catch (e) {
      setCancelErr(e instanceof Error ? e.message : String(e));
    } finally {
      setCancelling(false);
    }
  }

  return (
    <div className="mx-auto max-w-7xl space-y-4 p-5">
      <div className="flex flex-wrap items-center gap-3">
        <Link to="/" className="text-sm text-slate-500 hover:text-slate-300">
          ← Runs
        </Link>
        <h1 className="text-lg font-semibold">{j?.name ?? id}</h1>
        {j && (isPipelineJob(j) ? <Pill tone="amber">pipeline</Pill> : <Pill>run</Pill>)}
        <StateBadge state={state} />
        <div className="ml-auto flex items-center gap-2">
          {j && !TERMINAL_STATES[state] && (
            <button className="btn btn-danger" onClick={cancel} disabled={cancelling}>
              {cancelling ? "Cancelling…" : isPipelineJob(j) ? "Cancel pipeline" : "Cancel run"}
            </button>
          )}
        </div>
      </div>
      <ErrorBox error={job.error ?? cancelErr} />
      {!j && job.loading && <Spinner />}
      {j && (isPipelineJob(j) ? <PipelineView job={j} terminal={terminal} /> : <SingleRunView job={j} terminal={terminal} />)}
    </div>
  );
}

// ---- single run --------------------------------------------------------------------------------

function SingleRunView({ job: j, terminal }: { job: JobDetail; terminal: boolean }) {
  const base = jobPath(j.id);
  const status = j.status;
  const c = jobRunConfig(j);
  const ckpts = useAsync(() => api.checkpoints(base), [base], terminal ? 0 : 5000);
  const configYaml = useMemo(() => toYaml(j.config), [j.config]);
  const sft = c.algorithm?.name === "sft";
  return (
    <>
      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        <Card title="Status">
          <Kv
            items={[
              ["step", `${status?.step ?? j.step} / ${c?.trainer?.max_steps ?? "?"}`],
              ["message", status?.message ?? j.message ?? j.error ?? "–"],
              ["model", c?.model?.name_or_path ?? "–"],
              ["data", c?.data?.uri ?? "–"],
              ["algorithm", sft ? `sft · trainer ${c?.trainer?.backend}` : `${c?.algorithm?.name} · rollout ${c?.rollout?.backend} · trainer ${c?.trainer?.backend}`],
              ["run dir", j.run_dir],
              ["started", fmtDate(status?.started_at ?? j.started_at)],
              ["duration", fmtDuration(status?.started_at ?? j.started_at, j.finished_at)],
              ["pid", String(status?.pid ?? j.pid ?? "–")],
            ]}
          />
        </Card>
        <CheckpointsCard checkpoints={ckpts.data?.checkpoints ?? []} saveEvery={c?.trainer?.save_every} />
      </div>
      <RunStreams base={base} terminal={terminal} chartMetrics={sft ? SFT_METRICS : RL_METRICS} configYaml={configYaml} />
    </>
  );
}

function CheckpointsCard({ checkpoints, saveEvery }: { checkpoints: { step: number; path: string }[]; saveEvery?: number }) {
  return (
    <Card title="Checkpoints">
      {checkpoints.length ? (
        <ul className="space-y-1 text-xs">
          {checkpoints.map((c) => (
            <li key={c.path} className="flex justify-between gap-2">
              <span className="font-mono text-slate-200">step-{c.step}</span>
              <span className="truncate font-mono text-slate-500" title={c.path}>
                {c.path}
              </span>
            </li>
          ))}
        </ul>
      ) : (
        <div className="text-xs text-slate-500">No checkpoints yet{saveEvery ? ` (every ${saveEvery} steps)` : ""}.</div>
      )}
    </Card>
  );
}

// ---- pipeline ----------------------------------------------------------------------------------

function PipelineView({ job: j, terminal }: { job: PipelineJob & { status: JobDetail["status"] }; terminal: boolean }) {
  const status = j.status;
  const base = j.config.base;
  const manifest = useLiveSnapshot(stagesStreamUrl(j.id), "stages", () => api.stages(j.id), terminal);
  const stages = manifest.data?.stages ?? [];
  const total = j.config.stages.length;
  const finished = stages.filter((s) => s.state === "succeeded").length;
  const active = stages.find((s) => s.state === "running") ?? null;

  // Follow the active stage until the user picks one explicitly.
  const [picked, setPicked] = useState<number | null>(null);
  const fallback = active?.index ?? [...stages].reverse().find((s) => s.state !== "pending")?.index ?? (stages.length ? stages[0].index : null);
  const selectedIndex = picked !== null && stages.some((s) => s.index === picked) ? picked : fallback;
  const selected = stages.find((s) => s.index === selectedIndex) ?? null;
  const configYaml = useMemo(() => toYaml(j.config), [j.config]);

  return (
    <>
      <div className="grid gap-4 lg:grid-cols-[2fr_3fr]">
        <Card title="Pipeline status">
          <Kv
            items={[
              ["stages", `${status?.step ?? finished} / ${total} finished`],
              ["active", active ? `${active.index + 1}. ${active.name} (${active.type})` : terminal ? "–" : "starting…"],
              ["message", status?.message ?? j.message ?? j.error ?? "–"],
              ["base model", base?.model?.name_or_path ?? "–"],
              ["base data", base?.data?.uri ?? "–"],
              ["final model", manifest.data?.final_model_path ?? "–"],
              ["run dir", j.run_dir],
              ["started", fmtDate(status?.started_at ?? j.started_at)],
              ["duration", fmtDuration(status?.started_at ?? j.started_at, j.finished_at)],
            ]}
          />
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded bg-slate-800">
            <div className="h-full bg-emerald-600 transition-all" style={{ width: `${total ? (100 * finished) / total : 0}%` }} />
          </div>
        </Card>
        <Card title="Stages">
          <ErrorBox error={manifest.error} className="mb-2" />
          {!manifest.data && !manifest.error && <Spinner />}
          {manifest.data && stages.length === 0 && <div className="text-xs text-slate-500">No stage manifest yet.</div>}
          {stages.length > 0 && (
            <div className="overflow-hidden rounded-md border border-slate-800">
              <table className="w-full text-xs">
                <thead className="bg-slate-900 text-left uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="px-2 py-1.5">#</th>
                    <th className="px-2 py-1.5">Name</th>
                    <th className="px-2 py-1.5">Type</th>
                    <th className="px-2 py-1.5">State</th>
                    <th className="px-2 py-1.5 text-right">Step</th>
                    <th className="px-2 py-1.5">Summary</th>
                    <th className="px-2 py-1.5 text-right">Duration</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {stages.map((s) => {
                    const isActive = s.state === "running";
                    const isSelected = s.index === selectedIndex;
                    return (
                      <tr
                        key={s.index}
                        onClick={() => setPicked(s.index)}
                        className={`cursor-pointer ${isSelected ? "bg-sky-950/50 ring-1 ring-inset ring-sky-600" : "hover:bg-slate-900/60"} ${isActive ? "font-medium text-slate-100" : "text-slate-300"}`}
                        aria-selected={isSelected}
                      >
                        <td className="px-2 py-1.5 font-mono">{s.index + 1}</td>
                        <td className="px-2 py-1.5 font-mono">{s.name}</td>
                        <td className="px-2 py-1.5">
                          <Pill tone={STAGE_TONE[s.type]}>{s.type}</Pill>
                        </td>
                        <td className="px-2 py-1.5">
                          <StateBadge state={s.state} />
                        </td>
                        <td className="px-2 py-1.5 text-right font-mono">{s.step}</td>
                        <td className="max-w-[18rem] truncate px-2 py-1.5 text-slate-400" title={summaryText(s)}>
                          {summaryText(s)}
                        </td>
                        <td className="px-2 py-1.5 text-right font-mono text-slate-400">{fmtDuration(s.started_at, s.finished_at)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {active && picked !== null && picked !== active.index && (
            <button className="btn mt-2" onClick={() => setPicked(null)}>
              Follow active stage
            </button>
          )}
        </Card>
      </div>

      {selected && <StagePanel key={`${j.id}:${selected.index}`} jobId={j.id} stage={selected} job={j} jobTerminal={terminal} />}

      <details className="card">
        <summary className="cursor-pointer text-sm font-semibold text-slate-200">Pipeline config</summary>
        <pre className="mt-2 overflow-auto font-mono text-xs text-slate-300">{configYaml}</pre>
      </details>
    </>
  );
}

function summaryText(s: StageStatus): string {
  const sum = s.summary ?? {};
  if (s.error) return s.error;
  if (s.type === "rejection_sample") {
    const acc = sum.accepted;
    const sam = sum.sampled;
    const rate = sum.accept_rate;
    if (typeof acc === "number" && typeof sam === "number") return `accepted ${acc} / ${sam}${typeof rate === "number" ? ` (${(100 * rate).toFixed(1)}%)` : ""}`;
  }
  const keys = s.type === "eval" ? ["eval/reward_mean", "eval/num_samples"] : ["reward/mean", "loss/sft", "loss/policy", "eval/loss"];
  const parts = keys.filter((k) => typeof sum[k] === "number").map((k) => `${k}=${fmtNum(sum[k], 3)}`);
  if (parts.length) return parts.join(" · ");
  const first = Object.entries(sum).find(([, v]) => typeof v === "number");
  return first ? `${first[0]}=${fmtNum(first[1], 3)}` : s.state === "pending" ? "waiting" : "";
}

function StagePanel({ jobId, stage: s, job, jobTerminal }: { jobId: string; stage: StageStatus; job: PipelineJob; jobTerminal: boolean }) {
  const base = jobPath(jobId, s.index);
  const terminal = TERMINAL_STATES[s.state] === true || jobTerminal;
  const cfg = job.config.stages[s.index];
  const ckpts = useAsync(() => api.checkpoints(base), [base], terminal || s.type !== "train" ? 0 : 5000);
  const algo = cfg?.type === "train" ? effectiveAlgorithm(job.config.base, cfg) : null;
  const chartMetrics = s.type === "rejection_sample" ? REJECTION_METRICS : s.type === "eval" ? EVAL_METRICS : algo === "sft" ? SFT_METRICS : RL_METRICS;
  const stageYaml = useMemo(() => (cfg ? toYaml(cfg) : ""), [cfg]);
  const sum = s.summary ?? {};
  return (
    <div className="space-y-4 rounded-lg border border-sky-900/60 p-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm font-semibold text-slate-100">
          Stage {s.index + 1}: <span className="font-mono">{s.name}</span>
        </span>
        <Pill tone={STAGE_TONE[s.type]}>{s.type}</Pill>
        {algo && <Pill tone="sky">{algo}</Pill>}
        <StateBadge state={s.state} />
        <span className="text-xs text-slate-500">
          {fmtDate(s.started_at)} · {fmtDuration(s.started_at, s.finished_at)}
        </span>
        {s.run_dir && (
          <span className="ml-auto truncate font-mono text-xs text-slate-500" title={s.run_dir}>
            {s.run_dir}
          </span>
        )}
      </div>
      {s.error && <ErrorBox error={s.error} />}
      <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
        {s.type === "rejection_sample" ? <RejectionSummary summary={sum} /> : <SummaryCard summary={sum} />}
        {s.type === "train" ? (
          <CheckpointsCard checkpoints={ckpts.data?.checkpoints ?? []} />
        ) : (
          <Card title="Stage config">
            <pre className="max-h-40 overflow-auto font-mono text-xs text-slate-300">{stageYaml}</pre>
          </Card>
        )}
      </div>
      <RunStreams base={base} terminal={terminal} chartMetrics={chartMetrics} configYaml={s.type === "train" ? stageYaml : undefined} />
    </div>
  );
}

function SummaryCard({ summary }: { summary: Record<string, JsonValue> }) {
  const entries = Object.entries(summary).filter(([, v]) => typeof v === "number" || typeof v === "string");
  return (
    <Card title="Summary">
      {entries.length === 0 ? <div className="text-xs text-slate-500">No metrics yet.</div> : <Kv items={entries.map(([k, v]) => [k, typeof v === "number" ? fmtNum(v) : String(v)])} />}
    </Card>
  );
}

function RejectionSummary({ summary }: { summary: Record<string, JsonValue> }) {
  const num = (k: string) => (typeof summary[k] === "number" ? summary[k] : null);
  const accepted = num("accepted");
  const sampled = num("sampled");
  const rate = num("accept_rate") ?? (accepted !== null && sampled ? accepted / sampled : null);
  return (
    <Card title="Rejection sampling">
      <div className="grid grid-cols-3 gap-3 text-center">
        <Stat label="sampled" value={sampled} />
        <Stat label="accepted" value={accepted} />
        <Stat label="accept rate" value={rate === null ? null : `${(100 * rate).toFixed(1)}%`} />
      </div>
      <Kv
        items={[
          ["prompts", num("prompts") ?? "–"],
          ["mixed in", num("mixed") ?? "–"],
          ["dataset rows", num("rows") ?? "–"],
          ["dataset", typeof summary.dataset === "string" ? summary.dataset : "–"],
          ["reward/mean", num("reward/mean") !== null ? fmtNum(num("reward/mean"), 3) : "–"],
        ]}
      />
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: number | string | null }) {
  return (
    <div className="rounded-md border border-slate-800 bg-slate-950 p-2">
      <div className="font-mono text-lg text-slate-100">{value === null ? "–" : typeof value === "number" ? fmtNum(value) : value}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  );
}
