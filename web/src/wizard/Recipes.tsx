import { api } from "../api";
import { ErrorBox, Pill, Spinner } from "../components/ui";
import { useAsync } from "../hooks";
import type { Recipe } from "../types";

export default function Recipes({ onPick }: { onPick: (r: Recipe) => void }) {
  const recipes = useAsync(() => api.recipes(), []);
  return (
    <details className="card" open>
      <summary className="cursor-pointer text-sm font-semibold text-slate-200">Recipes</summary>
      <ErrorBox error={recipes.error} className="mt-2" />
      {recipes.loading && !recipes.data && <div className="mt-2"><Spinner /></div>}
      {recipes.data && recipes.data.length === 0 && <div className="mt-2 text-xs text-slate-500">No recipes under configs/recipes.</div>}
      {recipes.data && recipes.data.length > 0 && (
        <div className="mt-3 grid gap-3 md:grid-cols-3">
          {recipes.data.map((r) => {
            const c = r.config;
            const model = c.model?.name_or_path;
            const algo = c.algorithm?.name;
            return (
              <button key={r.name} type="button" onClick={() => onPick(r)} className="rounded-md border border-slate-800 bg-slate-900 p-3 text-left hover:border-sky-600">
                <div className="font-mono text-sm text-slate-100">{r.name}</div>
                <div className="mt-1 line-clamp-2 text-xs text-slate-500">{r.description}</div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {algo && <Pill tone="sky">{algo}</Pill>}
                  {c.rollout?.backend && <Pill>{c.rollout.backend}</Pill>}
                  {c.trainer?.backend && <Pill>{c.trainer.backend}</Pill>}
                  {c.env?.type && c.env.type !== "single_turn" && <Pill tone="amber">{c.env.type}</Pill>}
                  {model && <Pill>{model.split("/").pop()}</Pill>}
                </div>
              </button>
            );
          })}
        </div>
      )}
    </details>
  );
}
