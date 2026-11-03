import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { stringify as toYaml } from "yaml";
import { api, subscribeSse } from "../api";
import { MetricChart } from "../components/MetricChart";
import { Card, ErrorBox, Kv, Pill, Spinner, StateBadge, Tabs } from "../components/ui";
import { fmtDate, fmtDuration, fmtNum } from "../format";
import { useAsync } from "../hooks";
import { TERMINAL_STATES, type MetricRow, type TrajectoryRow } from "../types";

const CHART_METRICS = ["reward/mean", "reward/std", "loss/policy", "loss/kl", "tokens/response_mean", "lr"];
const COLORS = ["#38bdf8", "#a78bfa", "#f472b6", "#fb923c", "#34d399", "#facc15"];

/**
 * Tail an append-only stream for a job: SSE first (`/<kind>/stream`), falling back to polling the
 * paginated JSON endpoint if the EventSource errors. Stops once the job is terminal.
 */
function useStream<T>(jobId: string, kind: "metrics" | "logs", terminal: boolean, poll: (since: number) => Promise<{ items: T[]; next: number }>, parse: (data: string) => T) {
  const [items, setItems] = useState<T[]>([]);
  const [mode, setMode] = useState<"sse" | "poll" | "done">("sse");
  const nextRef = useRef(0);
  const terminalRef = useRef(terminal);
  useEffect(() => {
    terminalRef.current = terminal;
  });

  useEffect(() => {
    if (mode !== "sse") return;
    const event = kind === "metrics" ? "metrics" : "log";
    let gotAny = false;
    const dispose = subscribeSse(
      `/api/jobs/${jobId}/${kind}/stream?since=${nextRef.current}`,
      [event],
      (name, data) => {
        if (name === "end") {
          setMode("done");
          return;
        }
        gotAny = true;
        nextRef.current += 1;
        setItems((prev) => [...prev, parse(data)]);
      },
      () => {
        // EventSource would retry forever; a closed or never-opened stream means the endpoint is
        // unavailable (older server / proxy without SSE) — fall back to polling.
        if (!gotAny || terminalRef.current) {
          dispose();
          setMode(terminalRef.current ? "done" : "poll");
        }
      },
    );
    return dispose;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, kind, mode]);

  useEffect(() => {
    if (mode !== "poll") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const res = await poll(nextRef.current);
        if (cancelled) return;
        if (res.items.length) {
          nextRef.current = res.next;
          setItems((prev) => [...prev, ...res.items]);
        }
      } catch {
        /* keep polling */
      }
      if (cancelled) return;
      if (terminalRef.current) setMode("done");
      else timer = setTimeout(tick, 2000);
    };
    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, kind, mode]);

  // Terminal while in SSE mode: the server sends `end`; if it never does, do one final poll.
  useEffect(() => {
    if (terminal && mode === "sse") {
      const t = setTimeout(() => setMode((m) => (m === "sse" ? "poll" : m)), 3000);
      return () => clearTimeout(t);
    }
  }, [terminal, mode]);

  return { items, mode };
}

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

  const metrics = useStream<MetricRow>(
    id,
    "metrics",
    terminal,
    async (since) => {
      const r = await api.metrics(id, since);
      return { items: r.rows, next: r.next };
    },
    (d) => JSON.parse(d) as MetricRow,
  );
  const logs = useStream<string>(
    id,
    "logs",
    terminal,
    async (since) => {
      const r = await api.logs(id, 500);
      // /logs returns the last N lines + the absolute `next` offset; only append unseen lines.
      const unseen = Math.max(0, r.next - since);
      return { items: unseen === 0 ? [] : r.lines.slice(-unseen), next: r.next };
    },
    (d) => {
      try {
        const v: unknown = JSON.parse(d);
        return typeof v === "string" ? v : d;
      } catch {
        return d;
      }
    },
  );

  const traj = useIncremental<TrajectoryRow>(id, terminal, async (since) => api.trajectories(id, since));
  const ckpts = useAsync(() => api.checkpoints(id), [id], terminal ? 0 : 5000);

  const [tab, setTab] = useState<"charts" | "trajectories" | "logs" | "config">("charts");
  const j = job.data;
  const status = j?.status;
  const state = status?.state ?? j?.state ?? "pending";
  const extraMetrics = useMemo(() => {
    const keys = new Set<string>();
    for (const r of metrics.items) for (const k of Object.keys(r)) if (typeof r[k] === "number" && !["step", "time"].includes(k) && !CHART_METRICS.includes(k)) keys.add(k);
    return [...keys].sort();
  }, [metrics.items]);
  const [showExtra, setShowExtra] = useState(false);

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
        <StateBadge state={state} />
        <Pill tone={metrics.mode === "sse" ? "sky" : "slate"}>{metrics.mode === "sse" ? "live (SSE)" : metrics.mode === "poll" ? "polling" : "finished"}</Pill>
        <div className="ml-auto flex items-center gap-2">
          {j && !TERMINAL_STATES[state] && (
            <button className="btn btn-danger" onClick={cancel} disabled={cancelling}>
              {cancelling ? "Cancelling…" : "Cancel run"}
            </button>
          )}
        </div>
      </div>
      <ErrorBox error={job.error ?? cancelErr} />
      {!j && job.loading && <Spinner />}
      {j && (
        <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
          <Card title="Status">
            <Kv
              items={[
                ["step", `${status?.step ?? j.step} / ${j.config?.trainer?.max_steps ?? "?"}`],
                ["message", status?.message ?? j.message ?? j.error ?? "–"],
                ["model", j.config?.model?.name_or_path ?? "–"],
                ["data", j.config?.data?.uri ?? "–"],
                ["algorithm", `${j.config?.algorithm?.name} · rollout ${j.config?.rollout?.backend} · trainer ${j.config?.trainer?.backend}`],
                ["run dir", j.run_dir],
                ["started", fmtDate(status?.started_at ?? j.started_at)],
                ["duration", fmtDuration(status?.started_at ?? j.started_at, j.finished_at)],
                ["pid", String(status?.pid ?? j.pid ?? "–")],
              ]}
            />
          </Card>
          <Card title="Checkpoints">
            {ckpts.data?.checkpoints?.length ? (
              <ul className="space-y-1 text-xs">
                {ckpts.data.checkpoints.map((c) => (
                  <li key={c.path} className="flex justify-between gap-2">
                    <span className="font-mono text-slate-200">step-{c.step}</span>
                    <span className="truncate font-mono text-slate-500" title={c.path}>
                      {c.path}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="text-xs text-slate-500">No checkpoints yet (every {j.config?.trainer?.save_every} steps).</div>
            )}
          </Card>
        </div>
      )}

      <Tabs
        tabs={[
          { id: "charts", label: `Charts (${metrics.items.length})` },
          { id: "trajectories", label: `Trajectories (${traj.length})` },
          { id: "logs", label: `Logs (${logs.items.length})` },
          { id: "config", label: "Config" },
        ]}
        value={tab}
        onChange={setTab}
      />

      {tab === "charts" && (
        <div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {CHART_METRICS.map((m, i) => (
              <MetricChart key={m} rows={metrics.items} metric={m} color={COLORS[i % COLORS.length]} />
            ))}
            {showExtra && extraMetrics.map((m, i) => <MetricChart key={m} rows={metrics.items} metric={m} color={COLORS[(i + 3) % COLORS.length]} />)}
          </div>
          {extraMetrics.length > 0 && (
            <button className="btn mt-3" onClick={() => setShowExtra((s) => !s)}>
              {showExtra ? "Hide" : "Show"} {extraMetrics.length} more metrics
            </button>
          )}
        </div>
      )}

      {tab === "trajectories" && <TrajectoryTable rows={traj} />}

      {tab === "logs" && <LogView lines={logs.items} />}

      {tab === "config" && j && <pre className="card overflow-auto font-mono text-xs text-slate-300">{toYaml(j.config)}</pre>}
    </div>
  );
}

function useIncremental<T>(jobId: string, terminal: boolean, fetchSince: (since: number) => Promise<{ rows: T[]; next: number }>) {
  const [rows, setRows] = useState<T[]>([]);
  const nextRef = useRef(0);
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const r = await fetchSince(nextRef.current);
        if (cancelled) return;
        if (r.rows.length) {
          nextRef.current = r.next;
          setRows((prev) => [...prev, ...r.rows]);
        }
      } catch {
        /* retry on next tick */
      }
      if (!cancelled && !terminal) timer = setTimeout(tick, 3000);
    };
    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, terminal]);
  return rows;
}

function TrajectoryTable({ rows }: { rows: TrajectoryRow[] }) {
  const [open, setOpen] = useState<number | null>(null);
  const [onlyStep, setOnlyStep] = useState<number | "all">("all");
  const steps = useMemo(() => [...new Set(rows.map((r) => r.step))].sort((a, b) => b - a), [rows]);
  const shown = useMemo(() => {
    const list = onlyStep === "all" ? rows : rows.filter((r) => r.step === onlyStep);
    return list.slice(-200).reverse();
  }, [rows, onlyStep]);
  if (rows.length === 0) return <div className="card text-sm text-slate-500">No trajectories logged yet.</div>;
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-xs text-slate-400">
        step
        <select className="input w-auto" value={String(onlyStep)} onChange={(e) => setOnlyStep(e.target.value === "all" ? "all" : Number(e.target.value))}>
          <option value="all">all (latest 200)</option>
          {steps.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>
      <div className="overflow-hidden rounded-lg border border-slate-800">
        <table className="w-full table-fixed text-sm">
          <thead className="bg-slate-900 text-left text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="w-14 px-3 py-2">Step</th>
              <th className="w-[38%] px-3 py-2">Prompt</th>
              <th className="px-3 py-2">Response</th>
              <th className="w-20 px-3 py-2 text-right">Reward</th>
              <th className="w-16 px-3 py-2 text-right">Tok</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {shown.map((r, i) => {
              const key = rows.indexOf(r);
              const expanded = open === key;
              return (
                <tr key={`${r.step}-${r.sample_id}-${i}`} className="cursor-pointer align-top hover:bg-slate-900/60" onClick={() => setOpen(expanded ? null : key)}>
                  <td className="px-3 py-2 font-mono text-xs">{r.step}</td>
                  <td className={`px-3 py-2 text-xs text-slate-300 ${expanded ? "whitespace-pre-wrap" : "truncate"}`}>{r.prompt}</td>
                  <td className={`px-3 py-2 text-xs text-slate-200 ${expanded ? "whitespace-pre-wrap" : "truncate"}`}>
                    {r.response}
                    {expanded && (
                      <div className="mt-2 space-y-1 border-t border-slate-800 pt-2 text-slate-400">
                        {r.reference != null && (
                          <div>
                            <span className="text-slate-500">reference:</span> <span className="font-mono">{typeof r.reference === "string" ? r.reference : JSON.stringify(r.reference)}</span>
                          </div>
                        )}
                        <div>
                          <span className="text-slate-500">breakdown:</span>{" "}
                          {Object.entries(r.reward_breakdown ?? {}).map(([k, v]) => (
                            <Pill key={k}>
                              {k}={fmtNum(v, 3)}
                            </Pill>
                          ))}
                        </div>
                        <div className="text-slate-500">
                          group {r.group_id} · sample {r.sample_id} · {r.num_turns} turn(s)
                        </div>
                      </div>
                    )}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono text-xs ${r.reward > 0 ? "text-emerald-300" : "text-slate-400"}`}>{fmtNum(r.reward, 3)}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs text-slate-500">{r.num_response_tokens}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LogView({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLPreElement>(null);
  const [follow, setFollow] = useState(true);
  useEffect(() => {
    if (follow && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines, follow]);
  return (
    <div className="space-y-2">
      <label className="flex items-center gap-2 text-xs text-slate-400">
        <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow tail
      </label>
      <pre ref={ref} className="card h-[32rem] overflow-auto whitespace-pre-wrap font-mono text-xs leading-5 text-slate-300">
        {lines.length ? lines.slice(-2000).join("\n") : "no log output yet"}
      </pre>
    </div>
  );
}
