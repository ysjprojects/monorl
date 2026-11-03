// Hand-maintained mirror of src/monorl/config/schema.py (RunConfig) and the server API shapes
// documented in docs/ARCHITECTURE.md. Keep in sync with the pydantic schema; field descriptions
// are fetched at runtime from /api/config/schema, only the structure lives here.

export type JsonValue = string | number | boolean | null | JsonValue[] | { [k: string]: JsonValue };
export type JsonDict = Record<string, JsonValue>;

export interface LoraConfig {
  enabled: boolean;
  r: number;
  alpha: number;
  dropout: number;
  target_modules: string[] | "all-linear";
}

export interface ModelConfig {
  name_or_path: string;
  revision: string | null;
  tokenizer_name_or_path: string | null;
  trust_remote_code: boolean;
  dtype: "auto" | "float32" | "bfloat16" | "float16";
  attn_implementation: string | null;
  chat_template: string | null;
  lora: LoraConfig;
}

export type DataFormat = "auto" | "chat" | "prompt_answer" | "text";

export interface DataConfig {
  uri: string;
  split: string;
  hf_config_name: string | null;
  format: DataFormat;
  messages_field: string | null;
  prompt_field: string | null;
  answer_field: string | null;
  system_prompt: string | null;
  prompt_template: string | null;
  max_prompt_tokens: number;
  limit: number | null;
  shuffle: boolean;
  streaming: boolean;
  eval_uri: string | null;
  eval_split: string;
  extra: JsonDict;
}

export type AlgorithmName = "grpo" | "dapo" | "rloo" | "reinforce" | "ppo";
export const ALGORITHMS: AlgorithmName[] = ["grpo", "dapo", "rloo", "reinforce", "ppo"];

export interface AlgorithmConfig {
  name: AlgorithmName;
  kl_coef: number;
  kl_estimator: "k1" | "k2" | "k3";
  clip_eps: number;
  clip_eps_high: number | null;
  advantage_norm: "group" | "batch" | "none";
  advantage_std_norm: boolean;
  loss_agg: "token" | "sequence";
  entropy_coef: number;
  num_epochs: number;
  dynamic_sampling: boolean;
  overlong_penalty: number;
  gamma: number;
  lam: number;
  value_coef: number;
}

export type RolloutBackend = "hf" | "vllm" | "sglang";
export const ROLLOUT_BACKENDS: RolloutBackend[] = ["hf", "vllm", "sglang"];

export interface RolloutConfig {
  backend: RolloutBackend | string;
  group_size: number;
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  top_k: number;
  stop: string[];
  tensor_parallel_size: number;
  data_parallel_size: number;
  gpu_memory_utilization: number;
  max_model_len: number | null;
  enable_sleep_mode: boolean;
  async_steps_ahead: number;
  device: string | null;
  extra: JsonDict;
}

export type TrainerBackend = "reference" | "unsloth" | "trl" | "verl" | "nemo_rl";
export const TRAINER_BACKENDS: TrainerBackend[] = ["reference", "unsloth", "trl", "verl", "nemo_rl"];
export const PIPELINE_BACKENDS: Record<string, true> = { trl: true, verl: true, nemo_rl: true };

export interface TrainerConfig {
  backend: TrainerBackend | string;
  learning_rate: number;
  prompts_per_step: number;
  micro_batch_size: number;
  max_steps: number;
  warmup_steps: number;
  weight_decay: number;
  max_grad_norm: number;
  precision: "fp32" | "bf16" | "fp16";
  gradient_checkpointing: boolean;
  optimizer: "adamw" | "adam" | "sgd";
  scheduler: "constant" | "linear" | "cosine";
  save_every: number;
  eval_every: number | null;
  max_seq_len: number;
  device: string | null;
  extra: JsonDict;
}

export interface RewardSpec {
  type: string;
  weight: number;
  params: JsonDict;
}

export type EnvType = "single_turn" | "tool_calling" | "remote" | "openenv" | "hermes";
export const ENV_TYPES: EnvType[] = ["single_turn", "tool_calling", "remote", "openenv", "hermes"];

export interface EnvConfig {
  type: EnvType | string;
  max_turns: number;
  params: JsonDict;
}

export type WeightSync = "auto" | "inprocess" | "checkpoint" | "broadcast" | "cuda_ipc";
export const WEIGHT_SYNCS: WeightSync[] = ["auto", "inprocess", "checkpoint", "broadcast", "cuda_ipc"];

export interface ClusterConfig {
  mode: "local" | "ray";
  ray_address: string | null;
  num_nodes: number;
  gpus_per_node: number;
  placement: "auto" | "colocated" | "disaggregated";
  rollout_gpus: number | null;
  trainer_gpus: number | null;
  weight_sync: WeightSync | string;
  extra: JsonDict;
}

export interface LoggingConfig {
  run_root: string;
  log_every: number;
  save_trajectories: boolean;
  trajectories_per_step: number;
  wandb_project: string | null;
  tensorboard: boolean;
}

export interface RunConfig {
  run_name: string | null;
  seed: number;
  model: ModelConfig;
  data: DataConfig;
  algorithm: AlgorithmConfig;
  rollout: RolloutConfig;
  trainer: TrainerConfig;
  rewards: RewardSpec[];
  env: EnvConfig;
  cluster: ClusterConfig;
  logging: LoggingConfig;
}

// ---- server API ---------------------------------------------------------------------------

export type RunState = "pending" | "running" | "succeeded" | "failed" | "cancelled";
export const TERMINAL_STATES: Record<string, true> = { succeeded: true, failed: true, cancelled: true };

export interface Job {
  id: string;
  name: string;
  run_dir: string;
  config: RunConfig;
  state: RunState;
  pid: number | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
  updated_at: number;
  step: number;
  message: string | null;
}

export interface StatusJson {
  state: RunState;
  message: string | null;
  step: number;
  started_at: number;
  updated_at: number;
  pid: number;
  [k: string]: JsonValue | undefined;
}

export interface JobDetail extends Job {
  status: StatusJson | null;
}

export interface MetricRow {
  step: number;
  time: number;
  phase: string;
  [metric: string]: number | string;
}

export interface TrajectoryRow {
  step: number;
  sample_id: string;
  group_id: string;
  prompt: string;
  response: string;
  reward: number;
  reward_breakdown: Record<string, number>;
  num_turns: number;
  num_response_tokens: number;
  reference: JsonValue;
}

export interface Checkpoint {
  step: number;
  path: string;
}

export interface ValidationError {
  loc: (string | number)[];
  msg: string;
  type?: string;
}

export interface ValidateResponse {
  ok: boolean;
  errors: ValidationError[];
}

export interface Recipe {
  name: string;
  description: string;
  config: Partial<RunConfig> & Record<string, unknown>;
}

export interface HubModel {
  id: string;
  downloads: number;
  likes: number;
  pipeline_tag: string | null;
  tags: string[];
  last_modified: string | null;
}

export interface HubDataset {
  id: string;
  downloads: number;
  likes: number;
  tags: string[];
  last_modified: string | null;
}

export interface BrowseEntry {
  name: string;
  uri: string;
  is_dir: boolean;
  size: number | null;
}

export interface PreviewResponse {
  columns: Record<string, string>;
  rows: JsonDict[];
  inferred: { format: DataFormat; prompt_field: string | null; answer_field: string | null; messages_field: string | null };
  num_rows?: number | null;
}

export interface Connectors {
  s3: { configured: boolean; region: string | null; endpoint_url: string | null; vars_present: string[] };
}

export interface PatchStatus {
  id: string;
  target: string;
  version_spec: string;
  installed_version: string | null;
  applicable: boolean;
  applied: boolean;
  verified: boolean | null;
  reason: string | null;
}

export interface FrameworkInfo {
  installed: boolean;
  version: string | null;
  patches: PatchStatus[];
}

export interface SystemReport {
  python: string;
  platform: string;
  torch: { version: string | null; cuda_device_count: number; mps: boolean; [k: string]: JsonValue | undefined };
  toolchains: Record<string, { available: boolean; version: string | null; path?: string | null }>;
  frameworks: Record<string, FrameworkInfo>;
  components: Record<string, string[]>;
}

export interface EnvExample {
  name: string;
  language: string;
  path: string;
  transport: "stdio" | "http";
  command: string[];
}

export interface EnvsResponse {
  types: string[];
  examples: EnvExample[];
}

// JSON-schema subset we read for field descriptions/defaults.
export interface JsonSchemaProp {
  description?: string;
  default?: JsonValue;
  type?: string | string[];
  enum?: JsonValue[];
  anyOf?: JsonSchemaProp[];
  const?: JsonValue;
  $ref?: string;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  title?: string;
  properties?: Record<string, JsonSchemaProp>;
  items?: JsonSchemaProp;
}

export interface JsonSchema extends JsonSchemaProp {
  $defs?: Record<string, JsonSchemaProp>;
  properties?: Record<string, JsonSchemaProp>;
}
