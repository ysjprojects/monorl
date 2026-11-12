import { Link } from "react-router-dom";
import { api } from "../api";
import { ErrorBox, Pill, Spinner, StateBadge } from "../components/ui";
import { fmtAgo, fmtDuration } from "../format";
import { useAsync } from "../hooks";
import { effectiveAlgorithm } from "../pipeline";
import { isPipelineJob, jobRunConfig, type Job } from "../types";

/** "sft → grpo → rs → sft → grpo" for pipelines, the objective for runs. */
function algoSummary(j: Job): string {
  const c = jobRunConfig(j);
  if (!isPipelineJob(j)) return c?.algorithm?.name ?? "–";
  return j.config.stages.map((s) => (s.type === "train" ? effectiveAlgorithm(c, s) : s.type === "rejection_sample" ? "rs" : "eval")).join(" → ");
}

export default function Runs() {
  const jobs = useAsync(() => api.jobs(), [], 3000);
  return (
    <div className="mx-auto max-w-6xl p-5">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-lg font-semibold">Runs</h1>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500">auto-refresh 3s</span>
          <Link to="/new" className="btn btn-primary">
            New run
          </Link>
        </div>
      </div>
      <ErrorBox error={jobs.error} className="mb-3" />
      {jobs.loading && !jobs.data && <Spinner />}
      {jobs.data && jobs.data.length === 0 && (
        <div className="card text-sm text-slate-400">
          No runs yet.{" "}
          <Link to="/new" className="text-sky-400 underline">
            Create one
          </Link>
          .
        </div>
      )}
      {jobs.data && jobs.data.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-slate-800">
          <table className="w-full text-sm">
            <thead className="bg-slate-900 text-left text-xs uppercase tracking-wide text-slate-500">
              <tr>
                <th className="px-3 py-2">Name</th>
                <th className="px-3 py-2">Kind</th>
                <th className="px-3 py-2">State</th>
                <th className="px-3 py-2">Model</th>
                <th className="px-3 py-2">Algo</th>
                <th className="px-3 py-2">Backends</th>
                <th className="px-3 py-2 text-right">Step</th>
                <th className="px-3 py-2 text-right">Duration</th>
                <th className="px-3 py-2 text-right">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {jobs.data.map((j) => {
                const c = jobRunConfig(j);
                const pipeline = isPipelineJob(j);
                return (
                  <tr key={j.id} className="hover:bg-slate-900/60">
                    <td className="px-3 py-2">
                      <Link to={`/runs/${j.id}`} className="font-medium text-sky-300 hover:underline">
                        {j.name}
                      </Link>
                      {j.message && j.state !== "running" && <div className="truncate text-xs text-slate-500">{j.message}</div>}
                    </td>
                    <td className="px-3 py-2">{pipeline ? <Pill tone="amber">pipeline</Pill> : <Pill>run</Pill>}</td>
                    <td className="px-3 py-2">
                      <StateBadge state={j.state} />
                    </td>
                    <td className="max-w-[16rem] truncate px-3 py-2 font-mono text-xs text-slate-300">{c?.model?.name_or_path}</td>
                    <td className="max-w-[14rem] truncate px-3 py-2 text-xs text-slate-300" title={algoSummary(j)}>
                      {algoSummary(j)}
                    </td>
                    <td className="px-3 py-2 text-xs text-slate-400">
                      {c?.rollout?.backend} / {c?.trainer?.backend}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-xs">
                      {j.step}
                      <span className="text-slate-600">/{pipeline ? `${j.config.stages.length} stages` : c?.trainer?.max_steps}</span>
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-xs text-slate-400">{fmtDuration(j.started_at, j.finished_at)}</td>
                    <td className="px-3 py-2 text-right text-xs text-slate-400">{fmtAgo(j.created_at)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
