import { api } from "../api";
import { Card, ErrorBox, Kv, Pill, Spinner } from "../components/ui";
import { useAsync } from "../hooks";
import type { PatchStatus } from "../types";

function patchSummary(patches: PatchStatus[]) {
  if (patches.length === 0) return <span className="text-slate-600">none</span>;
  const applied = patches.filter((p) => p.applied).length;
  const applicable = patches.filter((p) => p.applicable).length;
  const failed = patches.filter((p) => p.applied && p.verified === false).length;
  return (
    <span className="flex flex-wrap gap-1">
      <Pill tone={applied === applicable && applicable > 0 ? "green" : "slate"}>
        {applied}/{patches.length} applied
      </Pill>
      {applicable < patches.length && <Pill tone="amber">{patches.length - applicable} not applicable</Pill>}
      {failed > 0 && <Pill tone="red">{failed} failed verify</Pill>}
    </span>
  );
}

export default function System() {
  const sys = useAsync(() => api.system(), []);
  const r = sys.data;
  return (
    <div className="mx-auto max-w-6xl space-y-4 p-5">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">System</h1>
        <button className="btn" onClick={sys.reload}>
          Refresh
        </button>
      </div>
      <ErrorBox error={sys.error} />
      {sys.loading && !r && <Spinner label="Running doctor…" />}
      {r && (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <Card title="Runtime">
              <Kv
                items={[
                  ["python", r.python],
                  ["platform", r.platform],
                  ["torch", r.torch?.version ?? "not installed"],
                ]}
              />
            </Card>
            <Card title="Devices">
              <Kv
                items={[
                  ["cuda devices", String(r.torch?.cuda_device_count ?? 0)],
                  ["mps", r.torch?.mps ? "available" : "no"],
                  ...Object.entries(r.torch ?? {})
                    .filter(([k]) => !["version", "cuda_device_count", "mps"].includes(k))
                    .map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v) : String(v)] as [string, string]),
                ]}
              />
            </Card>
            <Card title="Toolchains">
              <Kv
                items={Object.entries(r.toolchains ?? {}).map(([k, v]) => [
                  k,
                  v.available ? (
                    <span>
                      <Pill tone="green">ok</Pill> {v.version ?? ""} <span className="text-slate-500">{v.path ?? ""}</span>
                    </span>
                  ) : (
                    <Pill tone="red">missing</Pill>
                  ),
                ])}
              />
            </Card>
          </div>

          <Card title="Frameworks & patches">
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-1 pr-3">Framework</th>
                  <th className="py-1 pr-3">Installed</th>
                  <th className="py-1 pr-3">Version</th>
                  <th className="py-1 pr-3">Patches</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {Object.entries(r.frameworks ?? {}).map(([name, f]) => (
                  <tr key={name} className="align-top">
                    <td className="py-2 pr-3 font-mono">{name}</td>
                    <td className="py-2 pr-3">{f.installed ? <Pill tone="green">installed</Pill> : <Pill tone="slate">absent</Pill>}</td>
                    <td className="py-2 pr-3 font-mono text-xs">{f.version ?? "–"}</td>
                    <td className="py-2 pr-3">
                      {patchSummary(f.patches ?? [])}
                      {f.patches?.length > 0 && (
                        <details className="mt-1">
                          <summary className="cursor-pointer text-xs text-slate-500">details</summary>
                          <table className="mt-1 w-full text-xs">
                            <tbody>
                              {f.patches.map((p) => (
                                <tr key={p.id} className="text-slate-400">
                                  <td className="py-0.5 pr-2 font-mono text-slate-300">{p.id}</td>
                                  <td className="py-0.5 pr-2 font-mono">{p.target}</td>
                                  <td className="py-0.5 pr-2">{p.version_spec}</td>
                                  <td className="py-0.5 pr-2">
                                    {p.applied ? (
                                      <Pill tone={p.verified === false ? "red" : "green"}>{p.verified === false ? "applied, unverified" : "applied"}</Pill>
                                    ) : p.applicable ? (
                                      <Pill tone="amber">pending</Pill>
                                    ) : (
                                      <Pill tone="slate">n/a</Pill>
                                    )}
                                  </td>
                                  <td className="py-0.5 text-slate-500">{p.reason ?? ""}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </details>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Registered components">
            <div className="grid gap-3 md:grid-cols-2">
              {Object.entries(r.components ?? {}).map(([ns, names]) => (
                <div key={ns}>
                  <div className="mb-1 font-mono text-xs text-slate-500">{ns}</div>
                  <div className="flex flex-wrap gap-1">
                    {names.map((n) => (
                      <Pill key={n} tone="sky">
                        {n}
                      </Pill>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
