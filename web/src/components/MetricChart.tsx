import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtNum } from "../format";
import type { MetricRow } from "../types";

export function MetricChart({ rows, metric, color = "#38bdf8" }: { rows: MetricRow[]; metric: string; color?: string }) {
  const data = rows.filter((r) => typeof r[metric] === "number").map((r) => ({ step: r.step, value: r[metric] as number }));
  const last = data.length ? data[data.length - 1].value : null;
  return (
    <div className="card p-3">
      <div className="mb-1 flex items-baseline justify-between">
        <span className="font-mono text-xs text-slate-400">{metric}</span>
        <span className="font-mono text-sm text-slate-100">{last == null ? "–" : fmtNum(last)}</span>
      </div>
      <div className="h-36">
        {data.length === 0 ? (
          <div className="flex h-full items-center justify-center text-xs text-slate-600">no data yet</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 10 }} tickLine={false} />
              <YAxis stroke="#475569" tick={{ fontSize: 10 }} tickLine={false} width={48} tickFormatter={(v: number) => fmtNum(v, 3)} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }}
                labelFormatter={(l) => `step ${l}`}
                formatter={(v) => [fmtNum(v as number), metric]}
              />
              <Line type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
