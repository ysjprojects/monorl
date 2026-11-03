import { Card, Field, NumberInput, Pill, Select, TextInput, Toggle } from "../components/ui";
import { WEIGHT_SYNCS } from "../types";
import type { StepProps } from "./common";

const SYNC_NOTES: Record<string, string> = {
  auto: "Pick per placement: inprocess for hf+reference, cuda_ipc when colocated on CUDA, broadcast when disaggregated, else checkpoint.",
  inprocess: "Trainer and engine share one process; weights are the same tensors (hf backend).",
  checkpoint: "Trainer saves to disk, engine reloads. Slow but works everywhere.",
  broadcast: "NCCL/Gloo broadcast from trainer rank 0 to engine workers (disaggregated).",
  cuda_ipc: "Zero-copy CUDA IPC handles between colocated processes on the same GPU.",
};

export default function ComputeStep({ config, set, system }: StepProps) {
  const c = config.cluster;
  const r = config.rollout;
  const totalGpus = c.num_nodes * c.gpus_per_node;
  const localGpus = system?.torch?.cuda_device_count ?? 0;
  const ray = system?.frameworks?.ray;
  const disagg = c.placement === "disaggregated";
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Cluster">
        <div className="mb-3 flex flex-wrap gap-1 text-xs">
          {system && <Pill tone={localGpus > 0 ? "green" : "slate"}>{localGpus} CUDA device(s) locally</Pill>}
          {system?.torch?.mps && <Pill tone="sky">MPS available</Pill>}
          {ray && <Pill tone={ray.installed ? "green" : "amber"}>ray {ray.installed ? ray.version : "not installed"}</Pill>}
        </div>
        <Field path="cluster.mode" label="Mode">
          <div className="flex gap-2">
            {(["local", "ray"] as const).map((m) => (
              <label key={m} className={`flex flex-1 items-center gap-2 rounded-md border px-3 py-2 text-sm ${c.mode === m ? "border-sky-600 bg-sky-950/40" : "border-slate-800"}`}>
                <input type="radio" checked={c.mode === m} onChange={() => set("cluster.mode", m)} />
                <span>
                  <span className="block font-medium">{m === "local" ? "Local" : "Ray"}</span>
                  <span className="block text-xs text-slate-500">{m === "local" ? "In-process; single machine" : "Ray worker groups; multi-node"}</span>
                </span>
              </label>
            ))}
          </div>
        </Field>
        {c.mode === "ray" && (
          <Field path="cluster.ray_address" label="Ray address" className="mt-3">
            <TextInput value={c.ray_address} onChange={(v) => set("cluster.ray_address", v)} nullable placeholder="auto (start/join local cluster)" className="font-mono" />
          </Field>
        )}
        <div className="mt-3 grid grid-cols-2 gap-3">
          <Field path="cluster.num_nodes" label="Nodes">
            <NumberInput value={c.num_nodes} onChange={(v) => set("cluster.num_nodes", v ?? 1)} min={1} disabled={c.mode === "local"} />
          </Field>
          <Field path="cluster.gpus_per_node" label="GPUs per node">
            <NumberInput value={c.gpus_per_node} onChange={(v) => set("cluster.gpus_per_node", v ?? 0)} min={0} />
          </Field>
        </div>
        <div className="mt-3 text-xs text-slate-500">
          Total: <span className="font-mono text-slate-300">{totalGpus}</span> GPU(s) · rollout TP {r.tensor_parallel_size} × DP {r.data_parallel_size}
        </div>
      </Card>

      <Card title="Placement & weight sync">
        <Field path="cluster.placement" label="Placement">
          <Select value={c.placement} onChange={(v) => set("cluster.placement", v)} options={["auto", "colocated", "disaggregated"].map((v) => ({ value: v as typeof c.placement }))} />
        </Field>
        {disagg && (
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Field path="cluster.rollout_gpus" label="Rollout GPUs">
              <NumberInput value={c.rollout_gpus} onChange={(v) => set("cluster.rollout_gpus", v)} nullable min={0} />
            </Field>
            <Field path="cluster.trainer_gpus" label="Trainer GPUs">
              <NumberInput value={c.trainer_gpus} onChange={(v) => set("cluster.trainer_gpus", v)} nullable min={0} />
            </Field>
            {totalGpus > 0 && (c.rollout_gpus ?? 0) + (c.trainer_gpus ?? 0) > totalGpus && (
              <div className="col-span-2 text-xs text-rose-400">
                rollout + trainer GPUs ({(c.rollout_gpus ?? 0) + (c.trainer_gpus ?? 0)}) exceed the {totalGpus} available.
              </div>
            )}
          </div>
        )}
        <Field path="cluster.weight_sync" label="Weight sync" description={SYNC_NOTES[c.weight_sync]} className="mt-3">
          <Select value={c.weight_sync} onChange={(v) => set("cluster.weight_sync", v)} options={[...new Set([...WEIGHT_SYNCS, ...(system?.components?.weight_sync ?? []), c.weight_sync])].map((v) => ({ value: v }))} />
        </Field>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <Field path="rollout.tensor_parallel_size" label="Tensor parallel">
            <NumberInput value={r.tensor_parallel_size} onChange={(v) => set("rollout.tensor_parallel_size", v ?? 1)} min={1} />
          </Field>
          <Field path="rollout.data_parallel_size" label="Engine replicas (DP)">
            <NumberInput value={r.data_parallel_size} onChange={(v) => set("rollout.data_parallel_size", v ?? 1)} min={1} />
          </Field>
          <Field path="rollout.gpu_memory_utilization" label="GPU mem util">
            <NumberInput value={r.gpu_memory_utilization} onChange={(v) => set("rollout.gpu_memory_utilization", v ?? 0.6)} min={0.05} max={1} step="any" />
          </Field>
          <Field path="rollout.async_steps_ahead" label="Async steps ahead">
            <NumberInput value={r.async_steps_ahead} onChange={(v) => set("rollout.async_steps_ahead", v ?? 0)} min={0} />
          </Field>
        </div>
        <div className="mt-3 space-y-2">
          <Toggle path="rollout.enable_sleep_mode" checked={r.enable_sleep_mode} onChange={(v) => set("rollout.enable_sleep_mode", v)} label="Engine sleep mode" />
        </div>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <Field path="rollout.device" label="Rollout device">
            <TextInput value={r.device} onChange={(v) => set("rollout.device", v)} nullable placeholder="auto" />
          </Field>
          <Field path="trainer.device" label="Trainer device">
            <TextInput value={config.trainer.device} onChange={(v) => set("trainer.device", v)} nullable placeholder="auto" />
          </Field>
        </div>
      </Card>
    </div>
  );
}
