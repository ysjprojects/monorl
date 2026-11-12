// Charts / trajectories / logs for one run directory. `base` is `jobPath(id)` for a run or
// `jobPath(id, stageIndex)` for a pipeline stage; the JSON routes under it share one shape.
// Remount (key on `base`) to switch directories — stream cursors live in component state.

import { useEffect, useMemo, useRef, useState } from "react";
import { api, subscribeSse } from "../api";
import { fmtNum } from "../format";
import type { MetricRow, TrajectoryRow } from "../types";
import { MetricChart } from "./MetricChart";
import { Pill, Tabs } from "./ui";

const COLORS = ["#38bdf8", "#a78bfa", "#f472b6", "#fb923c", "#34d399", "#facc15"];

/**
 * Tail an append-only stream: SSE first (`<base>/<kind>/stream`) when `sse` is set, falling back
 * to polling the paginated JSON endpoint if the EventSource errors. Stops once `terminal`.
 */
function useStream<T>(base: string, kind: "metrics" | "logs", sse: boolean, terminal: boolean, poll: (since: number) => Promise<{ items: T[]; next: number }>, parse: (data: string) => T) {
  const [items, setItems] = useState<T[]>([]);
  const [mode, setMode] = useState<"sse" | "poll" | "done">(sse ? "sse" : "poll");
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
      `${base}/${kind}/stream?since=${nextRef.current}`,
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
  }, [base, kind, mode]);

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
  }, [base, kind, mode]);

  // Terminal while in SSE mode: the server sends `end`; if it never does, do one final poll.
  useEffect(() => {
    if (terminal && mode === "sse") {
      const t = setTimeout(() => setMode((m) => (m === "sse" ? "poll" : m)), 3000);
      return () => clearTimeout(t);
    }
  }, [terminal, mode]);

  // Became terminal while polling-done was never reached (e.g. terminal from the start): one last fetch.
  useEffect(() => {
    if (terminal && mode === "done" && nextRef.current === 0) {
      void poll(0).then((res) => {
        if (res.items.length) {
          nextRef.current = res.next;
          setItems((prev) => (prev.length ? prev : res.items));
        }
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [terminal, mode]);

  return { items, mode };
}

function useIncremental<T>(base: string, terminal: boolean, fetchSince: (since: number) => Promise<{ rows: T[]; next: number }>) {
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
  }, [base, terminal]);
  return rows;
}

export interface RunStreamsProps {
  base: string;
  terminal: boolean;
  /** Try `<base>/metrics/stream` + `<base>/logs/stream` before polling (job-level routes only). */
  sse: boolean;
  /** Metrics charted by default; everything else numeric is behind "show more". */
  chartMetrics: string[];
  /** Rendered as an extra "Config" tab when given. */
  configYaml?: string;
}

export function RunStreams({ base, terminal, sse, chartMetrics, configYaml }: RunStreamsProps) {
  const metrics = useStream<MetricRow>(
    base,
    "metrics",
    sse,
    terminal,
    async (since) => {
      const r = await api.metrics(base, since);
      return { items: r.rows, next: r.next };
    },
    (d) => JSON.parse(d) as MetricRow,
  );
  const logs = useStream<string>(
    base,
    "logs",
    sse,
    terminal,
    async (since) => {
      const r = await api.logs(base, 500);
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
  const traj = useIncremental<TrajectoryRow>(base, terminal, async (since) => api.trajectories(base, since));

  const [tab, setTab] = useState<"charts" | "trajectories" | "logs" | "config">("charts");
  const extraMetrics = useMemo(() => {
    const keys = new Set<string>();
    for (const r of metrics.items) for (const k of Object.keys(r)) if (typeof r[k] === "number" && !["step", "time"].includes(k) && !chartMetrics.includes(k)) keys.add(k);
    return [...keys].sort();
  }, [metrics.items, chartMetrics]);
  const [showExtra, setShowExtra] = useState(false);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="flex-1">
          <Tabs
            tabs={[
              { id: "charts", label: `Charts (${metrics.items.length})` },
              { id: "trajectories", label: `Trajectories (${traj.length})` },
              { id: "logs", label: `Logs (${logs.items.length})` },
              ...(configYaml !== undefined ? [{ id: "config" as const, label: "Config" }] : []),
            ]}
            value={tab}
            onChange={setTab}
          />
        </div>
        <Pill tone={metrics.mode === "sse" ? "sky" : "slate"}>{metrics.mode === "sse" ? "live (SSE)" : metrics.mode === "poll" ? "polling" : "finished"}</Pill>
      </div>

      {tab === "charts" && (
        <div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {chartMetrics.map((m, i) => (
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

      {tab === "config" && configYaml !== undefined && <pre className="card overflow-auto font-mono text-xs text-slate-300">{configYaml}</pre>}
    </div>
  );
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
