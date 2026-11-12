// Charts / trajectories / logs for one run directory. `base` is `jobPath(id)` for a run or
// `jobPath(id, stageIndex)` for a pipeline stage; the JSON routes and the SSE tail under it share
// one shape.

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, eventsUrl, subscribeSse, type TailCursors } from "../api";
import { fmtNum } from "../format";
import type { MetricRow, TrajectoryRow } from "../types";
import { MetricChart } from "./MetricChart";
import { PhasePill, TranscriptView } from "./TranscriptView";
import { Pill, Tabs } from "./ui";

const COLORS = ["#38bdf8", "#a78bfa", "#f472b6", "#fb923c", "#34d399", "#facc15"];

/** Rows arriving over SSE are appended in batches this far apart, so a replay renders a few times, not once per row. */
const FLUSH_MS = 100;
const POLL_MS = 2000;

/** The append-only files of a run dir, as far as they have been read. */
interface Tails {
  metrics: MetricRow[];
  logs: string[];
  trajectories: TrajectoryRow[];
}

const emptyTails = (): Tails => ({ metrics: [], logs: [], trajectories: [] });
/** `prev` itself when there is nothing to add, so consumers memoised on one file ignore the others. */
const appended = <T,>(prev: T[], more: T[]): T[] => (more.length ? [...prev, ...more] : prev);
/** A page that failed to load counts as empty: the next tick asks again from the same cursor. */
const pageOr = <T,>(page: Promise<{ rows: T[]; next: number }>) => page.catch(() => ({ rows: [] as T[], next: 0 }));

/**
 * Tail the run dir at `base`: one EventSource on `<base>/events` carries all three files; if it
 * fails, each file is polled from its paginated JSON route instead. Stops once `terminal` and
 * drained. Cursors and rows are component state: they belong to the `base` the component mounted with.
 */
function useRunTails(base: string, terminal: boolean): Tails & { mode: "sse" | "poll" | "done" } {
  const [tails, setTails] = useState(emptyTails);
  const [mode, setMode] = useState<"sse" | "poll" | "done">("sse");
  /** Rows in `tails` per file, i.e. where to continue from; rows still buffered are not counted. */
  const cursors = useRef<TailCursors>({ metrics: 0, logs: 0, trajectories: 0 });
  const terminalRef = useRef(terminal);
  useEffect(() => {
    terminalRef.current = terminal;
  });

  useEffect(() => {
    if (mode !== "sse") return;
    let buffer = emptyTails();
    let flushTimer: ReturnType<typeof setTimeout> | undefined;
    const flush = () => {
      clearTimeout(flushTimer);
      flushTimer = undefined;
      const add = buffer;
      if (add.metrics.length + add.logs.length + add.trajectories.length === 0) return;
      buffer = emptyTails();
      cursors.current = {
        metrics: cursors.current.metrics + add.metrics.length,
        logs: cursors.current.logs + add.logs.length,
        trajectories: cursors.current.trajectories + add.trajectories.length,
      };
      setTails((prev) => ({ metrics: appended(prev.metrics, add.metrics), logs: appended(prev.logs, add.logs), trajectories: appended(prev.trajectories, add.trajectories) }));
    };
    const dispose = subscribeSse(
      eventsUrl(base, cursors.current),
      ["metrics", "log", "trajectory"],
      (name, data) => {
        if (name === "end") {
          flush();
          setMode("done");
          return;
        }
        if (name === "metrics") buffer.metrics.push(JSON.parse(data) as MetricRow);
        else if (name === "log") buffer.logs.push(JSON.parse(data) as string);
        else buffer.trajectories.push(JSON.parse(data) as TrajectoryRow);
        flushTimer ??= setTimeout(flush, FLUSH_MS);
      },
      () => {
        // The endpoint is unavailable (older server / proxy without SSE) or the connection broke.
        // EventSource would reconnect to the same URL and replay rows from the old cursors, so
        // close it and carry on by polling from the current ones.
        flush();
        dispose();
        setMode("poll");
      },
    );
    return () => {
      // Rows still buffered were never counted in the cursors: whatever reads next re-fetches them.
      clearTimeout(flushTimer);
      dispose();
    };
  }, [base, mode]);

  useEffect(() => {
    if (mode !== "poll") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      // Only fetches that started after the run ended are known to have seen everything.
      const ended = terminalRef.current;
      const from = cursors.current;
      const [metrics, logs, trajectories] = await Promise.all([
        pageOr(api.metrics(base, from.metrics)),
        // /logs returns the last N lines + the absolute `next` offset: keep the unseen ones.
        pageOr(api.logs(base, 500).then((r) => ({ rows: r.next > from.logs ? r.lines.slice(from.logs - r.next) : [], next: r.next }))),
        pageOr(api.trajectories(base, from.trajectories)),
      ]);
      if (cancelled) return;
      cursors.current = {
        metrics: metrics.rows.length ? metrics.next : from.metrics,
        logs: logs.rows.length ? logs.next : from.logs,
        trajectories: trajectories.rows.length ? trajectories.next : from.trajectories,
      };
      const got = metrics.rows.length + logs.rows.length + trajectories.rows.length;
      if (got) setTails((prev) => ({ metrics: appended(prev.metrics, metrics.rows), logs: appended(prev.logs, logs.rows), trajectories: appended(prev.trajectories, trajectories.rows) }));
      // The JSON routes are paginated: keep reading finished files until every page comes back empty.
      if (ended && got === 0) setMode("done");
      else timer = setTimeout(tick, ended ? 0 : POLL_MS);
    };
    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [base, mode]);

  // Terminal while in SSE mode: the server sends `end`; if it does not, finish by polling.
  useEffect(() => {
    if (terminal && mode === "sse") {
      const t = setTimeout(() => setMode((m) => (m === "sse" ? "poll" : m)), 3000);
      return () => clearTimeout(t);
    }
  }, [terminal, mode]);

  return { ...tails, mode };
}

export interface RunStreamsProps {
  base: string;
  terminal: boolean;
  /** Metrics charted by default; everything else numeric is behind "show more". */
  chartMetrics: string[];
  /** Rendered as an extra "Config" tab when given. */
  configYaml?: string;
}

/**
 * Streams of one run directory. A different `base` remounts the view, so the EventSources of the
 * previous directory are closed and none of its rows or cursors carry over.
 */
export function RunStreams(props: RunStreamsProps) {
  return <RunStreamsView key={props.base} {...props} />;
}

function RunStreamsView({ base, terminal, chartMetrics, configYaml }: RunStreamsProps) {
  const { metrics, logs, trajectories, mode } = useRunTails(base, terminal);

  const [tab, setTab] = useState<"charts" | "trajectories" | "logs" | "config">("charts");
  const extraMetrics = useMemo(() => {
    const keys = new Set<string>();
    for (const r of metrics) for (const k of Object.keys(r)) if (typeof r[k] === "number" && !["step", "time"].includes(k) && !chartMetrics.includes(k)) keys.add(k);
    return [...keys].sort();
  }, [metrics, chartMetrics]);
  const [showExtra, setShowExtra] = useState(false);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <div className="flex-1">
          <Tabs
            tabs={[
              { id: "charts", label: `Charts (${metrics.length})` },
              { id: "trajectories", label: `Trajectories (${trajectories.length})` },
              { id: "logs", label: `Logs (${logs.length})` },
              ...(configYaml !== undefined ? [{ id: "config" as const, label: "Config" }] : []),
            ]}
            value={tab}
            onChange={setTab}
          />
        </div>
        <Pill tone={mode === "sse" ? "sky" : "slate"}>{mode === "sse" ? "live (SSE)" : mode === "poll" ? "polling" : "finished"}</Pill>
      </div>

      {tab === "charts" && (
        <div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {chartMetrics.map((m, i) => (
              <MetricChart key={m} rows={metrics} metric={m} color={COLORS[i % COLORS.length]} />
            ))}
            {showExtra && extraMetrics.map((m, i) => <MetricChart key={m} rows={metrics} metric={m} color={COLORS[(i + 3) % COLORS.length]} />)}
          </div>
          {extraMetrics.length > 0 && (
            <button className="btn mt-3" onClick={() => setShowExtra((s) => !s)}>
              {showExtra ? "Hide" : "Show"} {extraMetrics.length} more metrics
            </button>
          )}
        </div>
      )}

      {tab === "trajectories" && <TrajectoryTable rows={trajectories} />}

      {tab === "logs" && <LogView lines={logs} />}

      {tab === "config" && configYaml !== undefined && <pre className="card overflow-auto font-mono text-xs text-slate-300">{configYaml}</pre>}
    </div>
  );
}

/** Rows rendered at most: the newest ones matching the filters. */
const MAX_TRAJECTORY_ROWS = 200;

function TrajectoryTable({ rows }: { rows: TrajectoryRow[] }) {
  /** Index into `rows` (append-only, so stable) of the expanded row. */
  const [open, setOpen] = useState<number | null>(null);
  const [onlyStep, setOnlyStep] = useState<number | "all">("all");
  const [onlyPhase, setOnlyPhase] = useState("all");
  const steps = useMemo(() => [...new Set(rows.map((r) => r.step))].sort((a, b) => b - a), [rows]);
  const phases = useMemo(() => [...new Set(rows.flatMap((r) => (r.phase ? [r.phase] : [])))], [rows]);
  const multiPhase = phases.length > 1;
  const phase = multiPhase ? onlyPhase : "all";
  const shown = useMemo(() => {
    const list: { row: TrajectoryRow; index: number }[] = [];
    for (let i = rows.length - 1; i >= 0 && list.length < MAX_TRAJECTORY_ROWS; i--) {
      const r = rows[i];
      if ((onlyStep === "all" || r.step === onlyStep) && (phase === "all" || r.phase === phase)) list.push({ row: r, index: i });
    }
    return list;
  }, [rows, onlyStep, phase]);
  const toggle = useCallback((index: number) => setOpen((o) => (o === index ? null : index)), []);
  if (rows.length === 0) return <div className="card text-sm text-slate-500">No trajectories logged yet.</div>;
  const columns = multiPhase ? 8 : 7;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
        step
        <select className="input w-auto" value={String(onlyStep)} onChange={(e) => setOnlyStep(e.target.value === "all" ? "all" : Number(e.target.value))}>
          <option value="all">all (latest {MAX_TRAJECTORY_ROWS})</option>
          {steps.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        {multiPhase && (
          <>
            <span className="ml-2">phase</span>
            <select className="input w-auto" aria-label="phase" value={phase} onChange={(e) => setOnlyPhase(e.target.value)}>
              <option value="all">all</option>
              {phases.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </>
        )}
        <span className="ml-auto text-slate-500">click a row for the full episode</span>
      </div>
      <div className="overflow-hidden rounded-lg border border-slate-800">
        <table className="w-full table-fixed text-sm">
          <thead className="bg-slate-900 text-left text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="w-16 px-3 py-2">Step</th>
              {multiPhase && <th className="w-24 px-3 py-2">Phase</th>}
              <th className="w-[32%] px-3 py-2">Prompt</th>
              <th className="px-3 py-2">Response</th>
              <th className="w-16 px-3 py-2 text-right">Turns</th>
              <th className="w-18 px-3 py-2 text-center">Solved</th>
              <th className="w-20 px-3 py-2 text-right">Reward</th>
              <th className="w-16 px-3 py-2 text-right">Tok</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {shown.map(({ row, index }) => (
              <TrajectoryRowView key={index} row={row} index={index} expanded={open === index} showPhase={multiPhase} columns={columns} onToggle={toggle} />
            ))}
            {shown.length === 0 && (
              <tr>
                <td colSpan={columns} className="px-3 py-4 text-center text-xs text-slate-500">
                  No trajectories match the filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * One rollout: a one-line summary and, while expanded, the episode below it. Memoised on the row
 * object, so rows appended by the live tail re-render only the new ones.
 */
const TrajectoryRowView = memo(function TrajectoryRowView({ row: r, index, expanded, showPhase, columns, onToggle }: { row: TrajectoryRow; index: number; expanded: boolean; showPhase: boolean; columns: number; onToggle: (index: number) => void }) {
  return (
    <>
      <tr className={`cursor-pointer align-top hover:bg-slate-900/60 ${expanded ? "bg-slate-900/60" : ""}`} aria-expanded={expanded} onClick={() => onToggle(index)}>
        <td className="whitespace-nowrap px-3 py-2 font-mono text-xs">
          <span className="mr-1 text-slate-600" aria-hidden>
            {expanded ? "▾" : "▸"}
          </span>
          {r.step}
        </td>
        {showPhase && <td className="px-3 py-2 text-xs">{r.phase ? <PhasePill phase={r.phase} /> : <span className="text-slate-600">–</span>}</td>}
        <td className="truncate px-3 py-2 text-xs text-slate-300">{r.prompt}</td>
        <td className="truncate px-3 py-2 text-xs text-slate-200">{r.response}</td>
        <td className="px-3 py-2 text-right font-mono text-xs text-slate-400">{r.num_turns}</td>
        <td className="px-3 py-2 text-center text-xs">
          {r.success === undefined ? (
            <span className="text-slate-600">–</span>
          ) : r.success ? (
            <span className="text-emerald-300" title="solved: reward reached logging.success_threshold" aria-label="solved">
              ✓
            </span>
          ) : (
            <span className="text-slate-500" title="not solved" aria-label="not solved">
              ✗
            </span>
          )}
        </td>
        <td className={`px-3 py-2 text-right font-mono text-xs ${r.reward > 0 ? "text-emerald-300" : "text-slate-400"}`}>{fmtNum(r.reward, 3)}</td>
        <td className="px-3 py-2 text-right font-mono text-xs text-slate-500">{r.num_response_tokens}</td>
      </tr>
      {expanded && (
        <tr className="bg-slate-950/40">
          <td colSpan={columns} className="px-4 py-3">
            <TranscriptView row={r} />
          </td>
        </tr>
      )}
    </>
  );
});

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
