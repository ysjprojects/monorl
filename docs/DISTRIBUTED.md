# Distributed execution

The orchestrator only sees protocol objects (`RolloutEngine`, `Trainer`, `WeightSync`,
`Environment`). Where they live — in-process, on other GPUs, on other nodes — is decided by
`monorl.distributed.build_runtime(config)` from the `cluster:` block of the `RunConfig`. Pipeline
backends (`trl`, `verl`, `nemo_rl`) use their own launchers and read the same `cluster:` fields
when compiling their native configs.

```yaml
cluster:
  mode: local | ray          # in-process vs Ray actors
  ray_address: auto          # join an existing cluster (ray start --head) or null to start one
  num_nodes: 1
  gpus_per_node: 8
  placement: auto | colocated | disaggregated
  rollout_gpus: 4            # disaggregated only
  trainer_gpus: 4
  weight_sync: auto | inprocess | checkpoint | broadcast | cuda_ipc
```

## Placement

| placement | what runs where | when |
|---|---|---|
| `colocated` | rollout engine replicas and the trainer share every GPU; the engine sleeps (`vllm: sleep(level)`, `sglang: release_memory_occupation`) during the optimizer step and wakes after weight sync | best throughput per GPU when both fit (default on 1 node) |
| `disaggregated` | `rollout_gpus` for engines, `trainer_gpus` for training, disjoint sets; generation for step *n+1* can overlap training of step *n* (`rollout.async_steps_ahead`) | large models, multi-node, or engines that cannot sleep |
| `auto` | `colocated` when every node has GPUs for both roles and `rollout.async_steps_ahead == 0`, else `disaggregated` with a 50/50 split (trainer gets the remainder) | default |

`rollout.tensor_parallel_size × rollout.data_parallel_size` GPUs are reserved per set of engine
replicas; the planner validates the sum against `num_nodes × gpus_per_node` (the schema already
rejects `rollout_gpus + trainer_gpus > total`). The resolved plan is written to
`Runtime.placement` and shown in the run's `status.json` / UI job page.

Without CUDA every placement degrades to CPU/MPS in-process: `mode: local`,
`hf` engine, `reference` trainer, `inprocess` sync — which is what the CPU recipes and tests use.

## Sizing a run

`monorl bench` measures the pieces the planner has to balance, on the hardware you have:

```
monorl bench plan --config RECIPE [--gpus G --gpu-mem-gb M --preset h100|a100|l40s|mps|cpu]   # placement + step-time estimate
monorl bench sync --config RECIPE [--all | --strategy S] [--fake-params N --engines E]   # weight sync timing
monorl bench env  (--command CMD... | --url URL) [--parallel 1,8,32] [--episodes N]       # MEP/1 episodes/s
monorl bench rollout --config RECIPE [--batch-sizes 1,4,16,64] [--prompts N]              # tokens/s per batch size
```

`plan` prints the same placement the orchestrator would use (`Runtime.placement`) plus a
back-of-envelope step time; `sync` builds the real trainer/engines (or synthetic ones) and times
every strategy that works here; `env` drives a MEP/1 server with a canned policy at several
parallelism levels. All of them accept `--json PATH`.

## Ray

`cluster.mode: ray` builds the same components as Ray actors:

* `ray_address: auto` connects to a running cluster (`ray start --head` on node 0, `ray start
  --address=<head>:6379` on the others); `null` starts a local Ray instance.
* A placement group per role (`STRICT_PACK` for TP groups, `SPREAD` for DP replicas) pins actors
  to GPUs; colocated runs put engine and trainer bundles on the same GPUs.
* Engine replicas are a worker group: `generate` fans prompts over replicas, `update_weights`
  broadcasts to all. The trainer is `trainer_world_size` actors (one per trainer GPU) built with
  `rank`/`world_size`; each `TrainBatch` is sharded contiguously across them. The reference
  trainer joins a `torch.distributed` group (NCCL on CUDA, Gloo on CPU), weights micro-batch
  losses by the global token/sequence count and all-reduce-sums the trainable gradients before
  clipping, so a multi-rank step equals the single-process step over the whole batch; rank 0
  writes checkpoints/exports behind a barrier.
* Environments stay on the driver (remote MEP/1 processes are spawned where the orchestrator
  runs) unless the env adapter chooses otherwise; rewards run in the driver's thread pool.

Everything the orchestrator calls is synchronous from its point of view; Ray futures are awaited
inside the proxies.

Keys of `cluster.extra` that the Ray runtime and the weight sync read:

| Key | Default | Effect |
|---|---|---|
| `ray_init` | `{}` | Keyword arguments forwarded to `ray.init` |
| `cpus_per_actor` | `1` | CPUs in each actor's bundle when the cluster has no GPUs |
| `placement_group_timeout_s` | `300` | Seconds to wait for the placement group; after that the run fails with a `ConfigError` |
| `providers` | `[]` | Provider modules imported inside the actors only, after the config's own top-level `providers` (which every process loads) |
| `sync_chunk_bytes` | 64 MiB | Chunk size of the object-store weight sync (what `inprocess` becomes under Ray) |
| `sync_dir` | a fresh temp dir | Directory of the `checkpoint` weight sync; it must be on a filesystem that every actor sees |

## Weight sync strategies

After every optimizer step the trainer's weights have to reach every engine replica.
`cluster.weight_sync` picks how:

| strategy | mechanism | use |
|---|---|---|
| `inprocess` | `trainer.export_weights()` streams `(name, tensor)` straight into `engine.update_weights()` | `mode: local`, same process (CPU/MPS/1 GPU) |
| `cuda_ipc` | tensors stay on the GPU; handles are shared via CUDA IPC with the colocated engine process (vLLM worker extension / SGLang `update_weights_from_tensor`) | colocated multi-process on one node — fastest |
| `broadcast` | a NCCL (CUDA) / Gloo (CPU) process group spanning trainer and engine ranks; the trainer rank 0 broadcasts each tensor, engines load it in place | disaggregated, multi-node |
| `checkpoint` | trainer saves an HF checkpoint to shared storage, engines reload from it | simplest, slowest; any topology, survives engine restarts |
| `auto` | `inprocess` for local; `cuda_ipc` for colocated Ray; `broadcast` for disaggregated | default |

Every strategy reports `time/sync` and `weight_sync/bytes`; LoRA runs export **merged** weights
so engines never need adapter support. With `model.mtp.enabled` the model's MTP layers travel in
the same stream, named `@mtp/<checkpoint key>`: engines hand them to their drafter, or drop them
when they do not draft ([`MTP.md`](MTP.md)); the `checkpoint` strategy carries them as an extra
shard of the saved checkpoint.

## Async rollouts

`rollout.async_steps_ahead: N` lets a producer thread generate up to `N` batches ahead of the
trainer (bounded staleness). Batches carry the policy version they were sampled under
(`policy/staleness`); `old_logprobs` from the sampling policy make the PPO/GRPO ratio correct for
slightly off-policy data, and `algorithm.clip_eps` bounds the update. For more staleness, or
engines that differ numerically from the trainer, `algorithm.off_policy` adds decoupled PPO,
truncated importance weights, ratio masks and score centering ([`OFF_POLICY.md`](OFF_POLICY.md)).
`N = 0` (default) is fully synchronous; `N = 1` typically hides most generation latency in
disaggregated runs. In colocated runs the engine cannot generate while it sleeps, so async only
helps with `enable_sleep_mode: false` and enough memory for both.

Episodes within a step are always parallel: `group_size × prompts_per_step` episodes run as
concurrent tasks against the environment pool (remote MEP/1 servers advertise
`parallel_episodes`), and their `generate` calls are coalesced by the batching engine so the
inference backend sees large batches even for multi-turn envs.

## Resuming and cancellation

Checkpoints (`checkpoints/step-N/`) hold trainer state + RNG; `monorl train --resume` or
`run(config, run_dir, resume=True)` continues from the latest one and re-syncs weights before
generating. `SIGTERM`/`SIGINT` (the UI's *cancel* button sends `SIGTERM`) finish the in-flight
step, write `status: cancelled` and tear down Ray actors / env processes.
