// Bun mock of the monorl server API (docs/ARCHITECTURE.md "Server API") with canned data and a
// fake job that writes metrics/logs/trajectories on a timer and streams them over SSE.
// Run: `pnpm dev:mock` (listens on 8000, the Vite dev proxy target).

import { readdirSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

const PORT = Number(process.env.PORT ?? 8000);
const HOME = homedir();

type Dict = Record<string, unknown>;
type RunState = "pending" | "running" | "succeeded" | "failed" | "cancelled";

interface MockJob {
  id: string;
  name: string;
  run_dir: string;
  config: Dict;
  state: RunState;
  pid: number | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
  updated_at: number;
  step: number;
  message: string | null;
  metrics: Dict[];
  logs: string[];
  trajectories: Dict[];
  checkpoints: { step: number; path: string }[];
  timer: Timer | null;
}

const now = () => Date.now() / 1000;
const jobs = new Map<string, MockJob>();
const listeners = new Map<string, Set<(kind: "metrics" | "log" | "end", payload: unknown) => void>>();

function emit(id: string, kind: "metrics" | "log" | "end", payload: unknown) {
  for (const fn of listeners.get(id) ?? []) fn(kind, payload);
}

const SCHEMA = {
  title: "RunConfig",
  type: "object",
  properties: {
    run_name: { anyOf: [{ type: "string" }, { type: "null" }], default: null, description: "Defaults to '<model>-<algorithm>-<timestamp>'." },
    seed: { type: "integer", default: 42 },
    model: { $ref: "#/$defs/ModelConfig" },
    data: { $ref: "#/$defs/DataConfig" },
    algorithm: { $ref: "#/$defs/AlgorithmConfig" },
    rollout: { $ref: "#/$defs/RolloutConfig" },
    trainer: { $ref: "#/$defs/TrainerConfig" },
    rewards: { type: "array", items: { $ref: "#/$defs/RewardSpec" }, description: "Combined as a weighted sum." },
    env: { $ref: "#/$defs/EnvConfig" },
    cluster: { $ref: "#/$defs/ClusterConfig" },
    logging: { $ref: "#/$defs/LoggingConfig" },
  },
  $defs: {
    LoraConfig: {
      properties: {
        enabled: { type: "boolean", default: false, description: "Train LoRA adapters instead of full parameters." },
        r: { type: "integer", default: 16, minimum: 1, description: "LoRA rank." },
        alpha: { type: "integer", default: 32, minimum: 1, description: "LoRA alpha (scaling = alpha / r)." },
        dropout: { type: "number", default: 0.05, minimum: 0, maximum: 1 },
        target_modules: { default: "all-linear", description: "Module names to adapt, or 'all-linear'." },
      },
    },
    ModelConfig: {
      properties: {
        name_or_path: { type: "string", description: "Hugging Face model id or local path." },
        revision: { description: "Hub revision (branch, tag or commit)." },
        tokenizer_name_or_path: { description: "Override tokenizer (defaults to the model)." },
        trust_remote_code: { type: "boolean", default: false },
        dtype: { enum: ["auto", "float32", "bfloat16", "float16"], default: "auto", description: "Parameter dtype; 'auto' picks bf16 on accelerators and fp32 on CPU." },
        attn_implementation: { description: "e.g. 'flash_attention_2', 'sdpa', 'eager'." },
        chat_template: { description: "Jinja chat template override." },
        lora: { $ref: "#/$defs/LoraConfig" },
      },
    },
    DataConfig: {
      properties: {
        uri: { type: "string", description: "Dataset location: 'hf://org/name' (Hub), 's3://bucket/prefix-or-file' (credentials from AWS_* env vars), or a local file/directory (.jsonl/.json/.parquet/.csv/.arrow)." },
        split: { type: "string", default: "train", description: "Split name for Hub datasets (ignored for files)." },
        hf_config_name: { description: "Hub dataset config/subset name." },
        format: { enum: ["auto", "chat", "prompt_answer", "text"], default: "auto", description: "'chat': a messages column; 'prompt_answer': prompt + reference answer columns; 'text': a single prompt column; 'auto' sniffs columns." },
        messages_field: { description: "Column holding chat messages (format=chat)." },
        prompt_field: { description: "Column holding the prompt/question." },
        answer_field: { description: "Column holding the reference answer (optional)." },
        system_prompt: { description: "System prompt prepended to every sample." },
        prompt_template: { description: "Python format string applied to the prompt column, e.g. '{prompt}\\nThink step by step.'" },
        max_prompt_tokens: { type: "integer", default: 1024, minimum: 1, description: "Samples with longer prompts are dropped." },
        limit: { description: "Use only the first N samples (after shuffling)." },
        shuffle: { type: "boolean", default: true },
        streaming: { type: "boolean", default: false, description: "Stream from Hub/S3 instead of materialising locally." },
        eval_uri: { description: "Optional evaluation dataset (same format)." },
        eval_split: { type: "string", default: "test" },
      },
    },
    AlgorithmConfig: {
      properties: {
        name: { enum: ["grpo", "dapo", "rloo", "reinforce", "ppo"], default: "grpo", description: "Policy-gradient objective." },
        kl_coef: { type: "number", default: 0, minimum: 0, description: "Coefficient of the KL penalty to the reference policy (0 disables)." },
        kl_estimator: { enum: ["k1", "k2", "k3"], default: "k3", description: "KL estimator (Schulman's k1/k2/k3)." },
        clip_eps: { type: "number", default: 0.2, exclusiveMinimum: 0, description: "PPO clipping epsilon (lower bound)." },
        clip_eps_high: { description: "Asymmetric upper clip (DAPO clip-higher)." },
        advantage_norm: { enum: ["group", "batch", "none"], default: "group", description: "Normalise rewards within the prompt group (GRPO), across the batch, or not at all." },
        advantage_std_norm: { type: "boolean", default: true, description: "Divide by std in group normalisation (False = Dr.GRPO style)." },
        loss_agg: { enum: ["token", "sequence"], default: "token", description: "Average the loss over all response tokens or per sequence first." },
        entropy_coef: { type: "number", default: 0, minimum: 0 },
        num_epochs: { type: "integer", default: 1, minimum: 1, description: "Optimizer passes over each rollout batch." },
        dynamic_sampling: { type: "boolean", default: false, description: "Drop groups whose rewards are all identical (DAPO)." },
        overlong_penalty: { type: "number", default: 0, minimum: 0, description: "Reward penalty for truncated (length) completions." },
        gamma: { type: "number", default: 1, description: "Discount for multi-turn per-turn rewards." },
        lam: { type: "number", default: 1, description: "GAE lambda (ppo only)." },
        value_coef: { type: "number", default: 0.5, description: "Value loss weight (ppo only)." },
      },
    },
    RolloutConfig: {
      properties: {
        backend: { type: "string", default: "hf", description: "Rollout engine: 'hf' (transformers), 'vllm', 'sglang'." },
        group_size: { type: "integer", default: 4, minimum: 1, description: "Completions sampled per prompt (G in GRPO)." },
        max_new_tokens: { type: "integer", default: 256, minimum: 1 },
        temperature: { type: "number", default: 1 },
        top_p: { type: "number", default: 1 },
        top_k: { type: "integer", default: -1, description: "-1 disables top-k." },
        tensor_parallel_size: { type: "integer", default: 1, minimum: 1 },
        data_parallel_size: { type: "integer", default: 1, minimum: 1, description: "Number of independent engine replicas." },
        gpu_memory_utilization: { type: "number", default: 0.6 },
        max_model_len: {},
        enable_sleep_mode: { type: "boolean", default: true, description: "Offload engine memory while training (colocated)." },
        async_steps_ahead: { type: "integer", default: 0, minimum: 0, description: "Generate up to N batches ahead of training (off-policy staleness bound). 0 = synchronous." },
        device: { description: "Device for the 'hf' backend (cpu/mps/cuda); auto if unset." },
      },
    },
    TrainerConfig: {
      properties: {
        backend: { type: "string", default: "reference", description: "Trainer: 'reference' (torch), 'unsloth', or a pipeline backend: 'trl', 'verl', 'nemo_rl'." },
        learning_rate: { type: "number", default: 1e-6, exclusiveMinimum: 0 },
        prompts_per_step: { type: "integer", default: 8, minimum: 1, description: "Prompts per optimizer step (x group_size = sequences)." },
        micro_batch_size: { type: "integer", default: 4, minimum: 1, description: "Sequences per forward/backward pass." },
        max_steps: { type: "integer", default: 100, minimum: 1 },
        warmup_steps: { type: "integer", default: 0, minimum: 0 },
        weight_decay: { type: "number", default: 0, minimum: 0 },
        max_grad_norm: { type: "number", default: 1 },
        precision: { enum: ["fp32", "bf16", "fp16"], default: "bf16", description: "Mixed-precision mode (fp32 forced on CPU)." },
        gradient_checkpointing: { type: "boolean", default: false },
        optimizer: { enum: ["adamw", "adam", "sgd"], default: "adamw" },
        scheduler: { enum: ["constant", "linear", "cosine"], default: "constant" },
        save_every: { type: "integer", default: 50, minimum: 1, description: "Checkpoint interval in steps." },
        eval_every: {},
        max_seq_len: { type: "integer", default: 2048, minimum: 16, description: "Prompt + response truncation for training." },
        device: { description: "cpu/mps/cuda; auto if unset." },
      },
    },
    RewardSpec: {
      properties: {
        type: { type: "string", description: "Reward function name (see /api/system for the list)." },
        weight: { type: "number", default: 1, description: "Multiplier when combining several rewards." },
        params: { type: "object" },
      },
    },
    EnvConfig: {
      properties: {
        type: { type: "string", default: "single_turn", description: "Environment: 'single_turn', 'tool_calling', 'openenv', 'hermes', ..." },
        max_turns: { type: "integer", default: 1, minimum: 1 },
        params: { type: "object", description: "Environment-specific parameters." },
      },
    },
    ClusterConfig: {
      properties: {
        mode: { enum: ["local", "ray"], default: "local", description: "Run in-process or on a Ray cluster." },
        ray_address: { description: "Ray address ('auto' to join an existing cluster)." },
        num_nodes: { type: "integer", default: 1, minimum: 1 },
        gpus_per_node: { type: "integer", default: 0, minimum: 0 },
        placement: { enum: ["auto", "colocated", "disaggregated"], default: "auto", description: "Share GPUs between rollout and training (colocated) or split them (disaggregated)." },
        rollout_gpus: { description: "GPUs for inference when disaggregated." },
        trainer_gpus: { description: "GPUs for training when disaggregated." },
        weight_sync: { type: "string", default: "auto", description: "Weight sync strategy: 'auto', 'inprocess', 'checkpoint', 'broadcast', 'cuda_ipc'." },
      },
    },
    LoggingConfig: {
      properties: {
        run_root: { type: "string", default: "~/.monorl/runs", description: "Directory that holds run directories." },
        log_every: { type: "integer", default: 1, minimum: 1 },
        save_trajectories: { type: "boolean", default: true, description: "Write sampled rollouts to trajectories.jsonl." },
        trajectories_per_step: { type: "integer", default: 8, minimum: 0 },
        wandb_project: {},
        tensorboard: { type: "boolean", default: false },
      },
    },
  },
};

const patch = (id: string, target: string, applicable: boolean, applied: boolean) => ({
  id,
  target,
  version_spec: ">=0.8",
  installed_version: applicable ? "0.33.0" : null,
  applicable,
  applied,
  verified: applied ? true : null,
  reason: applicable ? null : "framework not installed",
});

const SYSTEM = {
  python: "3.12.8 (.venv)",
  platform: "macOS-15.6-arm64",
  torch: { version: "2.15.0", cuda_device_count: 0, mps: true },
  toolchains: {
    go: { available: true, version: "go1.27", path: "/opt/homebrew/bin/go" },
    cxx: { available: true, version: "Apple clang 21", path: "/usr/bin/clang++" },
  },
  frameworks: {
    vllm: { installed: false, version: null, patches: [patch("vllm.sleep_mode", "vllm.LLM.sleep", false, false), patch("vllm.worker_update_weights", "vllm.worker.Worker", false, false)] },
    sglang: { installed: false, version: null, patches: [patch("sglang.update_weights", "sglang.srt.Engine.update_weights_from_tensor", false, false)] },
    verl: { installed: false, version: null, patches: [] },
    trl: { installed: true, version: "1.17.1", patches: [patch("trl.grpo_reporter", "trl.GRPOTrainer.log", true, true)] },
    nemo_rl: { installed: false, version: null, patches: [] },
    unsloth: { installed: false, version: null, patches: [] },
    openenv: { installed: true, version: "0.3.0", patches: [] },
    hermes: { installed: true, version: "0.19.0", patches: [] },
    ray: { installed: true, version: "2.59.0", patches: [] },
  },
  components: {
    rollout_engine: ["hf", "vllm", "sglang"],
    trainer: ["reference", "unsloth"],
    pipeline_backend: ["trl", "verl", "nemo_rl"],
    env: ["single_turn", "tool_calling", "remote", "openenv", "hermes"],
    reward: ["exact_match", "format", "regex", "math_verify", "code_exec", "length", "llm_judge", "reward_model", "composite"],
    data_source: ["local", "hf", "s3"],
    weight_sync: ["inprocess", "checkpoint", "broadcast", "cuda_ipc"],
  },
};

const RECIPES = [
  {
    name: "gsm8k-grpo-smol",
    description: "GRPO on GSM8K with SmolLM2-135M; CPU-friendly smoke recipe.",
    config: {
      model: { name_or_path: "HuggingFaceTB/SmolLM2-135M-Instruct" },
      data: { uri: "hf://openai/gsm8k", hf_config_name: "main", prompt_field: "question", answer_field: "answer", format: "prompt_answer", system_prompt: "Answer with the final number after '####'." },
      algorithm: { name: "grpo" },
      rollout: { backend: "hf", group_size: 4, max_new_tokens: 128, device: "cpu" },
      trainer: { backend: "reference", max_steps: 20, prompts_per_step: 4, precision: "fp32", device: "cpu" },
      rewards: [{ type: "math_verify", weight: 1.0, params: {} }, { type: "format", weight: 0.2, params: { pattern: "####\\s*-?\\d+" } }],
    },
  },
  {
    name: "dapo-qwen-vllm",
    description: "DAPO with clip-higher and dynamic sampling on a vLLM backend (single GPU).",
    config: {
      model: { name_or_path: "Qwen/Qwen2.5-1.5B-Instruct" },
      data: { uri: "hf://BytedTsinghua-SIA/DAPO-Math-17k" },
      algorithm: { name: "dapo", clip_eps: 0.2, clip_eps_high: 0.28, dynamic_sampling: true, overlong_penalty: 1.0 },
      rollout: { backend: "vllm", group_size: 8, max_new_tokens: 2048 },
      trainer: { backend: "reference", max_steps: 500, prompts_per_step: 32 },
      rewards: [{ type: "math_verify", weight: 1.0, params: {} }],
      cluster: { mode: "local", gpus_per_node: 1, placement: "colocated" },
    },
  },
  {
    name: "go-env-remote",
    description: "Multi-turn RL against the Go 'wordle' MEP/1 example env over stdio.",
    config: {
      model: { name_or_path: "HuggingFaceTB/SmolLM2-135M-Instruct" },
      data: { uri: "configs/data/wordle_prompts.jsonl" },
      env: { type: "remote", max_turns: 6, params: { transport: "stdio", command: ["envs/sdk/go/bin/wordle"] } },
      rewards: [],
      rollout: { group_size: 4, max_new_tokens: 16 },
      trainer: { max_steps: 50 },
    },
  },
];

const MODELS = [
  { id: "HuggingFaceTB/SmolLM2-135M-Instruct", downloads: 1_234_567, likes: 210, pipeline_tag: "text-generation", tags: ["llama", "instruct"], last_modified: "2025-09-10T00:00:00Z" },
  { id: "HuggingFaceTB/SmolLM-135M-Instruct", downloads: 456_000, likes: 120, pipeline_tag: "text-generation", tags: ["llama"], last_modified: "2025-05-01T00:00:00Z" },
  { id: "Qwen/Qwen2.5-1.5B-Instruct", downloads: 9_876_543, likes: 980, pipeline_tag: "text-generation", tags: ["qwen2"], last_modified: "2025-08-20T00:00:00Z" },
  { id: "Qwen/Qwen3-0.6B", downloads: 5_000_000, likes: 640, pipeline_tag: "text-generation", tags: ["qwen3"], last_modified: "2025-10-01T00:00:00Z" },
  { id: "meta-llama/Llama-3.2-1B-Instruct", downloads: 7_000_000, likes: 1500, pipeline_tag: "text-generation", tags: ["llama", "gated"], last_modified: "2025-01-15T00:00:00Z" },
  { id: "hf-internal-testing/tiny-random-LlamaForCausalLM", downloads: 3_000_000, likes: 10, pipeline_tag: "text-generation", tags: ["llama", "testing"], last_modified: "2024-02-01T00:00:00Z" },
];

const DATASETS = [
  { id: "openai/gsm8k", downloads: 2_000_000, likes: 700, tags: ["math"], last_modified: "2024-01-01T00:00:00Z" },
  { id: "BytedTsinghua-SIA/DAPO-Math-17k", downloads: 90_000, likes: 150, tags: ["math"], last_modified: "2025-03-01T00:00:00Z" },
  { id: "HuggingFaceH4/ultrafeedback_binarized", downloads: 500_000, likes: 300, tags: ["chat"], last_modified: "2024-06-01T00:00:00Z" },
  { id: "trl-lib/tldr", downloads: 100_000, likes: 40, tags: ["summarization"], last_modified: "2024-10-01T00:00:00Z" },
];

const S3_TREE: Record<string, { name: string; is_dir: boolean; size: number | null }[]> = {
  "s3://": [{ name: "monorl-data", is_dir: true, size: null }, { name: "team-datasets", is_dir: true, size: null }],
  "s3://monorl-data": [{ name: "gsm8k", is_dir: true, size: null }, { name: "prompts.jsonl", is_dir: false, size: 1_048_576 }],
  "s3://monorl-data/gsm8k": [{ name: "train.parquet", is_dir: false, size: 4_200_000 }, { name: "test.parquet", is_dir: false, size: 600_000 }],
  "s3://team-datasets": [{ name: "chat.jsonl", is_dir: false, size: 12_000_000 }],
};

function browseLocal(uri: string) {
  const dir = resolve(uri.replace(/^~(?=$|\/)/, HOME));
  const entries = readdirSync(dir, { withFileTypes: true })
    .filter((e) => !e.name.startsWith("."))
    .map((e) => {
      const full = join(dir, e.name);
      let size: number | null = null;
      if (!e.isDirectory()) {
        try {
          size = statSync(full).size;
        } catch {
          size = null;
        }
      }
      return { name: e.name, uri: full, is_dir: e.isDirectory(), size };
    })
    .sort((a, b) => Number(b.is_dir) - Number(a.is_dir) || a.name.localeCompare(b.name));
  return entries.slice(0, 500);
}

function browse(uri: string) {
  if (uri.startsWith("s3://")) {
    const trimmed = uri.replace(/\/+$/, "");
    const key = trimmed === "s3:" ? "s3://" : trimmed;
    const items = S3_TREE[key];
    if (!items) throw new Error(`no such prefix: ${uri}`);
    return items.map((e) => ({ ...e, uri: `${key === "s3://" ? "s3://" : `${key}/`}${e.name}` }));
  }
  return browseLocal(uri);
}

function preview(body: { uri: string; limit?: number }) {
  const limit = body.limit ?? 5;
  if (/chat|ultrafeedback/i.test(body.uri)) {
    const rows = Array.from({ length: limit }, (_, i) => ({
      messages: [
        { role: "user", content: `Write a haiku about ${["rain", "code", "mountains", "tea", "sleep"][i % 5]}.` },
        { role: "assistant", content: "Soft rain on the roof / lines of code fall into place / the compiler hums" },
      ],
      source: "mock",
    }));
    return { columns: { messages: "list", source: "string" }, rows, inferred: { format: "chat", prompt_field: null, answer_field: null, messages_field: "messages" }, num_rows: 1200 };
  }
  const rows = Array.from({ length: limit }, (_, i) => ({
    question: `Natalia sold clips to ${40 + i} of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?`,
    answer: `Natalia sold ${40 + i}/2 = <<${40 + i}/2=${(40 + i) / 2}>>${(40 + i) / 2} clips in May.\n#### ${(40 + i) * 1.5}`,
  }));
  return { columns: { question: "string", answer: "string" }, rows, inferred: { format: "prompt_answer", prompt_field: "question", answer_field: "answer", messages_field: null }, num_rows: 7473 };
}

// --- validation ------------------------------------------------------------------------------

function validate(config: unknown): { ok: boolean; errors: { loc: (string | number)[]; msg: string; type: string }[] } {
  const errors: { loc: (string | number)[]; msg: string; type: string }[] = [];
  const c = (config ?? {}) as Dict;
  const model = c.model as Dict | undefined;
  const data = c.data as Dict | undefined;
  const algo = (c.algorithm as Dict | undefined) ?? {};
  const rollout = (c.rollout as Dict | undefined) ?? {};
  const trainer = (c.trainer as Dict | undefined) ?? {};
  const cluster = (c.cluster as Dict | undefined) ?? {};
  const env = (c.env as Dict | undefined) ?? {};
  const rewards = (c.rewards as unknown[] | undefined) ?? [];
  if (!model || typeof model.name_or_path !== "string" || !model.name_or_path) errors.push({ loc: ["model", "name_or_path"], msg: "Field required", type: "missing" });
  if (!data || typeof data.uri !== "string" || !data.uri) errors.push({ loc: ["data", "uri"], msg: "Field required", type: "missing" });
  const name = (algo.name as string | undefined) ?? "grpo";
  if (!["grpo", "dapo", "rloo", "reinforce", "ppo"].includes(name)) errors.push({ loc: ["algorithm", "name"], msg: `Input should be 'grpo', 'dapo', 'rloo', 'reinforce' or 'ppo'`, type: "literal_error" });
  const gs = (rollout.group_size as number | undefined) ?? 4;
  if (typeof gs !== "number" || gs < 1) errors.push({ loc: ["rollout", "group_size"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
  const lr = trainer.learning_rate as number | undefined;
  if (lr !== undefined && !(lr > 0)) errors.push({ loc: ["trainer", "learning_rate"], msg: "Input should be greater than 0", type: "greater_than" });
  const steps = trainer.max_steps as number | undefined;
  if (steps !== undefined && !(steps >= 1)) errors.push({ loc: ["trainer", "max_steps"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
  if (errors.length === 0) {
    if (["grpo", "dapo", "rloo"].includes(name) && gs < 2) errors.push({ loc: [], msg: `Value error, ${name} needs rollout.group_size >= 2`, type: "value_error" });
    if (cluster.placement === "disaggregated" && Number(cluster.gpus_per_node ?? 0) > 0) {
      const r = cluster.rollout_gpus as number | null | undefined;
      const t = cluster.trainer_gpus as number | null | undefined;
      const total = Number(cluster.num_nodes ?? 1) * Number(cluster.gpus_per_node ?? 0);
      if (r == null || t == null) errors.push({ loc: [], msg: "Value error, disaggregated placement needs rollout_gpus and trainer_gpus", type: "value_error" });
      else if (r + t > total) errors.push({ loc: [], msg: `Value error, rollout_gpus + trainer_gpus = ${r + t} exceeds ${total} GPUs`, type: "value_error" });
    }
    if (rewards.length === 0 && ((env.type as string | undefined) ?? "single_turn") === "single_turn")
      errors.push({ loc: [], msg: "Value error, single_turn env needs at least one reward in `rewards`", type: "value_error" });
  }
  return { ok: errors.length === 0, errors };
}

// --- fake job --------------------------------------------------------------------------------

const PROMPTS = [
  "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether?",
  "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
  "Betty is saving money for a new wallet which costs $100. She has only half of the money she needs. How much more does she need?",
  "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
];

function startFakeJob(job: MockJob) {
  const maxSteps = Number((job.config.trainer as Dict | undefined)?.max_steps ?? 100);
  const saveEvery = Number((job.config.trainer as Dict | undefined)?.save_every ?? 50);
  const lr0 = Number((job.config.trainer as Dict | undefined)?.learning_rate ?? 1e-6);
  const groupSize = Number((job.config.rollout as Dict | undefined)?.group_size ?? 4);
  job.state = "running";
  job.started_at = now();
  job.pid = 40_000 + Math.floor(Math.random() * 1000);
  job.message = "training";
  const log = (line: string) => {
    const stamped = `${new Date().toISOString().replace("T", " ").slice(0, 19)} ${line}`;
    job.logs.push(stamped);
    emit(job.id, "log", stamped);
  };
  log(`[running] starting run ${job.name} (${maxSteps} steps, mock)`);
  log(`loading model ${(job.config.model as Dict).name_or_path}`);
  job.timer = setInterval(() => {
    if (job.state !== "running") return;
    const step = job.step + 1;
    const p = step / maxSteps;
    const row: Dict = {
      step,
      time: now(),
      phase: "train",
      "reward/mean": Math.min(1, 0.15 + 0.7 * (1 - Math.exp(-3 * p)) + (Math.random() - 0.5) * 0.1),
      "reward/std": Math.max(0.02, 0.45 * (1 - p) + (Math.random() - 0.5) * 0.05),
      "reward/exact_match": Math.min(1, 0.1 + 0.6 * p + (Math.random() - 0.5) * 0.1),
      "loss/policy": (Math.random() - 0.5) * 0.02 - 0.01 * p,
      "loss/kl": 0.002 * step * (0.8 + Math.random() * 0.4),
      "loss/entropy": 1.2 - 0.6 * p + (Math.random() - 0.5) * 0.05,
      "policy/clipfrac": 0.05 + Math.random() * 0.05,
      "policy/approx_kl": 0.001 * step,
      "tokens/response_mean": 60 + 80 * p + (Math.random() - 0.5) * 10,
      "tokens/prompt_mean": 85 + (Math.random() - 0.5) * 4,
      "time/rollout": 2.5 + Math.random(),
      "time/train": 1.1 + Math.random() * 0.4,
      "time/sync": 0.1,
      lr: lr0 * (1 - 0.5 * p),
    };
    job.metrics.push(row);
    job.step = step;
    job.updated_at = now();
    emit(job.id, "metrics", row);
    log(`step ${step}/${maxSteps} reward=${(row["reward/mean"] as number).toFixed(3)} kl=${(row["loss/kl"] as number).toFixed(4)}`);
    for (let g = 0; g < Math.min(2, groupSize); g++) {
      const prompt = PROMPTS[(step + g) % PROMPTS.length];
      const reward = Math.random() < (row["reward/mean"] as number) ? 1 : 0;
      job.trajectories.push({
        step,
        sample_id: `s${step}-${g}`,
        group_id: `g${step}`,
        prompt,
        response: reward ? "Let's think step by step. In May she sold 48/2 = 24 clips. 48 + 24 = 72.\n#### 72" : "The answer is 50 because 48 + 2 = 50.\n#### 50",
        reward,
        reward_breakdown: { exact_match: reward, format: 1 },
        num_turns: 1,
        num_response_tokens: 30 + Math.floor(Math.random() * 40),
        reference: "72",
      });
    }
    if (step % saveEvery === 0 || step === maxSteps) {
      job.checkpoints.push({ step, path: `${job.run_dir}/checkpoints/step-${step}` });
      log(`saved checkpoint step-${step}`);
    }
    if (step >= maxSteps) finishJob(job, "succeeded", "finished");
  }, 1000);
}

function finishJob(job: MockJob, state: RunState, message: string) {
  clearInterval(job.timer ?? undefined);
  job.timer = null;
  job.state = state;
  job.message = message;
  job.finished_at = now();
  job.updated_at = now();
  job.logs.push(`${new Date().toISOString().replace("T", " ").slice(0, 19)} [${state}] ${message}`);
  emit(job.id, "log", job.logs[job.logs.length - 1]);
  emit(job.id, "end", { state });
}

function newJob(config: Dict, state: RunState = "pending"): MockJob {
  const id = Math.random().toString(16).slice(2, 10);
  const modelName = String((config.model as Dict | undefined)?.name_or_path ?? "model").split("/").pop();
  const name = (config.run_name as string | null | undefined) ?? `${modelName}-${(config.algorithm as Dict | undefined)?.name ?? "grpo"}-${Date.now()}`;
  const job: MockJob = {
    id,
    name,
    run_dir: `${HOME}/.monorl/runs/${name}`,
    config,
    state,
    pid: null,
    created_at: now(),
    started_at: null,
    finished_at: null,
    error: null,
    updated_at: now(),
    step: 0,
    message: null,
    metrics: [],
    logs: [],
    trajectories: [],
    checkpoints: [],
    timer: null,
  };
  jobs.set(id, job);
  return job;
}

function publicJob(job: MockJob) {
  const { metrics: _m, logs: _l, trajectories: _t, checkpoints: _c, timer: _timer, ...rest } = job;
  return rest;
}

function statusJson(job: MockJob) {
  if (job.state === "pending") return null;
  return { state: job.state, message: job.message, step: job.step, started_at: job.started_at, updated_at: job.updated_at, pid: job.pid };
}

// seed: one finished run, one failed run
{
  const done = newJob({ ...RECIPES[0].config, run_name: "gsm8k-grpo-smol-seed", trainer: { ...RECIPES[0].config.trainer, max_steps: 12, save_every: 6 } });
  startFakeJob(done);
  for (let i = 0; i < 12; i++) {
    // fast-forward the timer deterministically
    const row = { step: i + 1, time: now() - (12 - i) * 60, phase: "train", "reward/mean": 0.2 + i * 0.05, "reward/std": 0.4 - i * 0.02, "loss/policy": -0.01 * i, "loss/kl": 0.001 * i, "tokens/response_mean": 50 + i * 3, lr: 1e-5 };
    done.metrics.push(row);
    done.step = i + 1;
  }
  done.checkpoints.push({ step: 6, path: `${done.run_dir}/checkpoints/step-6` }, { step: 12, path: `${done.run_dir}/checkpoints/step-12` });
  finishJob(done, "succeeded", "finished 12 steps");
  const failed = newJob({ ...RECIPES[1].config, run_name: "dapo-qwen-vllm-oom" });
  failed.state = "failed";
  failed.started_at = now() - 3600;
  failed.finished_at = now() - 3500;
  failed.error = "RuntimeError: CUDA out of memory";
  failed.message = "RuntimeError: CUDA out of memory";
  failed.logs.push("2026-11-02 09:00:00 [running] starting", "2026-11-02 09:01:40 [failed] RuntimeError: CUDA out of memory");
}

// --- http ------------------------------------------------------------------------------------

const json = (data: unknown, status = 200) => Response.json(data, { status, headers: { "access-control-allow-origin": "*" } });

function sse(id: string, kinds: ("metrics" | "log")[], replay: () => { kind: "metrics" | "log"; payload: unknown }[], terminal: () => RunState | null): Response {
  const encoder = new TextEncoder();
  let unsubscribe: (() => void) | null = null;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (event: string, data: unknown) => controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
      for (const r of replay()) send(r.kind, r.payload);
      const t = terminal();
      if (t) {
        send("end", { state: t });
        controller.close();
        return;
      }
      const fn = (kind: "metrics" | "log" | "end", payload: unknown) => {
        if (kind === "end") {
          send("end", payload);
          controller.close();
          unsubscribe?.();
          return;
        }
        if (kinds.includes(kind)) send(kind, payload);
      };
      const set = listeners.get(id) ?? new Set();
      set.add(fn);
      listeners.set(id, set);
      unsubscribe = () => listeners.get(id)?.delete(fn);
    },
    cancel() {
      unsubscribe?.();
    },
  });
  return new Response(stream, { headers: { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" } });
}

Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname;
    const method = req.method;
    try {
      if (path === "/api/system") return json(SYSTEM);
      if (path === "/api/config/schema") return json(SCHEMA);
      if (path === "/api/config/validate" && method === "POST") return json(validate(((await req.json()) as Dict).config));
      if (path === "/api/recipes") return json(RECIPES);
      if (path === "/api/models/search") {
        const needle = (url.searchParams.get("q") ?? "").toLowerCase();
        return json(MODELS.filter((m) => m.id.toLowerCase().includes(needle)).slice(0, Number(url.searchParams.get("limit") ?? 20)));
      }
      if (path === "/api/models/info") return json({ id: url.searchParams.get("id"), architecture: "LlamaForCausalLM", params: 134_515_008, gated: false });
      if (path === "/api/datasets/search") {
        const needle = (url.searchParams.get("q") ?? "").toLowerCase();
        return json(DATASETS.filter((d) => d.id.toLowerCase().includes(needle)));
      }
      if (path === "/api/datasets/browse") return json(browse(url.searchParams.get("uri") ?? "~"));
      if (path === "/api/datasets/preview" && method === "POST") return json(preview((await req.json()) as { uri: string; limit?: number }));
      if (path === "/api/connectors") {
        const vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_REGION", "AWS_ENDPOINT_URL"].filter((v) => process.env[v]);
        const configured = vars.includes("AWS_ACCESS_KEY_ID") || vars.includes("AWS_PROFILE") || process.env.MOCK_S3 === "1";
        return json({ s3: { configured, region: process.env.AWS_DEFAULT_REGION ?? (configured ? "us-east-1" : null), endpoint_url: process.env.AWS_ENDPOINT_URL ?? null, vars_present: vars } });
      }
      if (path === "/api/envs")
        return json({
          types: SYSTEM.components.env,
          examples: [
            { name: "wordle", language: "go", path: "envs/sdk/go/examples/wordle", transport: "stdio", command: ["envs/sdk/go/bin/wordle"] },
            { name: "countdown", language: "go", path: "envs/sdk/go/examples/countdown", transport: "stdio", command: ["envs/sdk/go/bin/countdown"] },
            { name: "gridworld", language: "cpp", path: "envs/sdk/cpp/examples/gridworld", transport: "stdio", command: ["envs/sdk/cpp/bin/gridworld"] },
          ],
        });
      if (path === "/api/jobs" && method === "GET") return json([...jobs.values()].sort((a, b) => b.created_at - a.created_at).map(publicJob));
      if (path === "/api/jobs" && method === "POST") {
        const { config } = (await req.json()) as { config: Dict };
        const v = validate(config);
        if (!v.ok) return json({ detail: v.errors }, 422);
        const job = newJob(config);
        setTimeout(() => startFakeJob(job), 800);
        return json(publicJob(job));
      }
      const m = path.match(/^\/api\/jobs\/([^/]+)(?:\/(.*))?$/);
      if (m) {
        const job = jobs.get(m[1]);
        if (!job) return json({ detail: "job not found" }, 404);
        const sub = m[2] ?? "";
        const since = Number(url.searchParams.get("since") ?? 0);
        if (sub === "" && method === "GET") return json({ ...publicJob(job), status: statusJson(job) });
        if (sub === "cancel" && method === "POST") {
          if (job.state === "running" || job.state === "pending") finishJob(job, "cancelled", "cancelled by user");
          return json(publicJob(job));
        }
        if (sub === "metrics") return json({ rows: job.metrics.slice(since), next: job.metrics.length });
        if (sub === "metrics/stream")
          return sse(job.id, ["metrics"], () => job.metrics.slice(since).map((payload) => ({ kind: "metrics" as const, payload })), () => (job.state === "running" || job.state === "pending" ? null : job.state));
        if (sub === "logs") {
          const tail = Number(url.searchParams.get("tail") ?? 200);
          return json({ lines: job.logs.slice(-tail), next: job.logs.length });
        }
        if (sub === "logs/stream")
          return sse(job.id, ["log"], () => job.logs.slice(since).map((payload) => ({ kind: "log" as const, payload })), () => (job.state === "running" || job.state === "pending" ? null : job.state));
        if (sub === "trajectories") return json({ rows: job.trajectories.slice(since), next: job.trajectories.length });
        if (sub === "checkpoints") return json({ checkpoints: job.checkpoints });
      }
      return json({ detail: `no route ${method} ${path}` }, 404);
    } catch (e) {
      return json({ detail: e instanceof Error ? e.message : String(e) }, 400);
    }
  },
});

console.log(`monorl mock API listening on http://127.0.0.1:${PORT}`);
