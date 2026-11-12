// Hand-maintained mirror of src/monorl/config/schema.py (RunConfig) and the server API shapes
// documented in docs/ARCHITECTURE.md. Keep in sync with the pydantic schema; field descriptions
// are fetched at runtime from /api/config/schema, only the structure lives here.

export type JsonValue = string | number | boolean | null | JsonValue[] | { [k: string]: JsonValue };
export type JsonDict = Record<string, JsonValue>;

// ---- messages (monorl.core.types) ---------------------------------------------------------

export type ContentPart =
  | { type: "text"; text: string }
  /** `image` is a path, URL or data URL; in saved rollouts it is `null` with a `note` when the image was left out (inline and too large, or neither a path nor a URL). */
  | { type: "image"; image: string | null; note?: string }
  | { type: "audio"; audio: string }
  | { type: "video"; video: string };

/** A tool call requested by an assistant message. */
export interface ToolCall {
  id?: string;
  name: string;
  /** Parsed arguments (an object), or the raw string when the backend kept it unparsed. */
  arguments: JsonValue;
}

export interface Message {
  role: string;
  content: string | ContentPart[];
  /** `tool` messages: the tool that produced the content. */
  name?: string;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
}

/** Text parts of a message joined (mirrors `monorl.core.types.message_text`). */
export function messageText(m: Message): string {
  if (typeof m.content === "string") return m.content;
  return m.content
    .filter((p): p is Extract<ContentPart, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("");
}

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
  image_field: string | string[] | null;
  response_field: string | null;
  tools_field: string | null;
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

export type AlgorithmName = "grpo" | "dapo" | "rloo" | "reinforce" | "ppo" | "sft";
export const ALGORITHMS: AlgorithmName[] = ["grpo", "dapo", "rloo", "reinforce", "ppo", "sft"];

/** Corrections for batches sampled by another policy than the one being updated (`algorithm.off_policy`). */
export interface OffPolicyConfig {
  anchor: "auto" | "behavior" | "trainer";
  importance_weights: "none" | "token" | "sequence";
  /** `null` = untruncated. */
  importance_cap: number | null;
  ratio_mask: "none" | "token" | "sequence" | "sequence_mean";
  ratio_mask_low: number;
  ratio_mask_high: number;
  score_centering: boolean;
  score_centering_top_k: number;
}

export interface AlgorithmConfig {
  name: AlgorithmName;
  sft_label_smoothing: number;
  sft_train_on_prompt: boolean;
  sft_turns: "last" | "all";
  kl_coef: number;
  kl_estimator: "k1" | "k2" | "k3";
  clip_eps: number;
  clip_eps_high: number | null;
  advantage_norm: "group" | "batch" | "none";
  advantage_std_norm: boolean;
  loss_agg: "token" | "sequence" | "rollout";
  entropy_coef: number;
  num_epochs: number;
  dynamic_sampling: boolean;
  overlong_penalty: number;
  credit_assignment: "episode" | "turn";
  gamma: number;
  lam: number;
  value_coef: number;
  /** `null` = plain squared error. */
  value_clip: number | null;
  off_policy: OffPolicyConfig;
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
  /** Context budget of an episode; `null` = `trainer.max_seq_len`. */
  max_context_tokens: number | null;
  enable_sleep_mode: boolean;
  async_steps_ahead: number;
  turn_rendering: "tokens" | "template";
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

export type EnvType = "single_turn" | "tool_calling" | "remote" | "harness" | "openenv" | "hermes";
export const ENV_TYPES: EnvType[] = ["single_turn", "tool_calling", "remote", "harness", "openenv", "hermes"];

export interface EnvConfig {
  type: EnvType | string;
  max_turns: number;
  retries: number;
  max_failed_frac: number;
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
  save_transcripts: boolean;
  success_threshold: number;
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

// ---- pipelines (monorl.config.pipeline) ----------------------------------------------------

export type StageType = "train" | "rejection_sample" | "eval";
export const STAGE_TYPES: StageType[] = ["train", "rejection_sample", "eval"];

export interface TrainStage {
  type: "train";
  name: string;
  /** Partial RunConfig merged over the pipeline base. */
  overrides: JsonDict;
  init_from: "previous" | "base" | string;
  data_from: "previous" | "config";
}

export interface RejectionSampleStage {
  type: "rejection_sample";
  name: string;
  data: Partial<DataConfig> | null;
  num_samples: number;
  min_reward: number;
  max_per_prompt: number;
  max_response_tokens: number | null;
  max_context_tokens: number | null;
  require_finish: boolean;
  dedupe: boolean;
  mix: string[];
  rollout_overrides: JsonDict;
  reward_overrides: JsonDict[] | null;
  output: string | null;
  max_prompts: number | null;
}

export interface EvalStage {
  type: "eval";
  name: string;
  data: Partial<DataConfig> | null;
  rollout_overrides: JsonDict;
  max_prompts: number | null;
  /** Completions per prompt; above 1 the stage samples at the rollout temperature and reports `eval/pass_at_k`. */
  num_samples: number;
}

export type Stage = TrainStage | RejectionSampleStage | EvalStage;

export interface PipelineConfig {
  name: string | null;
  base: RunConfig;
  stages: Stage[];
}

/** A yaml/json document is a pipeline when it has `stages` and `base` (vs. a RunConfig). */
export function isPipelineDoc(doc: unknown): doc is { base: unknown; stages: unknown[] } {
  return !!doc && typeof doc === "object" && "stages" in doc && "base" in doc && Array.isArray(doc.stages);
}

// ---- server API ---------------------------------------------------------------------------

export type RunState = "pending" | "running" | "succeeded" | "failed" | "cancelled";
export const TERMINAL_STATES: Record<string, true> = { succeeded: true, failed: true, cancelled: true };

export type JobKind = "run" | "pipeline";

interface JobBase {
  id: string;
  name: string;
  run_dir: string;
  state: RunState;
  pid: number | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
  updated_at: number;
  /** Training step for runs; stages finished for pipelines. */
  step: number;
  message: string | null;
}

export interface RunJob extends JobBase {
  kind?: "run";
  config: RunConfig;
}

export interface PipelineJob extends JobBase {
  kind: "pipeline";
  config: PipelineConfig;
}

export type Job = RunJob | PipelineJob;

export function isPipelineJob(j: Job): j is PipelineJob {
  return j.kind === "pipeline";
}

/** The RunConfig a job trains with (a pipeline's base config). */
export function jobRunConfig(j: Job): RunConfig {
  return isPipelineJob(j) ? j.config.base : j.config;
}

/** One entry of a pipeline manifest (`pipeline.json`). */
export interface StageStatus {
  index: number;
  name: string;
  type: StageType;
  state: RunState;
  run_dir: string | null;
  started_at: number | null;
  finished_at: number | null;
  /** Training step (train), sampling chunk (rejection_sample) or 0. */
  step: number;
  /** Train: last train-phase metrics row. Rejection: cumulative `{prompts, sampled, accepted, accept_rate, mixed, rows, dataset, "reward/mean"}`. Eval: last eval row. */
  summary: Record<string, JsonValue>;
  error?: string | null;
}

/** GET /api/jobs/{id}/stages, and the data of every `stages` event of /api/jobs/{id}/stages/stream. */
export interface StagesResponse {
  job_id: string;
  kind: JobKind;
  name: string;
  state: RunState;
  final_model_path: string | null;
  /** Why the pipeline failed, from the manifest. */
  error?: string | null;
  stages: StageStatus[];
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

export type JobDetail = Job & { status: StatusJson | null };

export interface MetricRow {
  step: number;
  time: number;
  phase: string;
  [metric: string]: number | string;
}

export type TrajectoryPhase = "train" | "eval" | "rejection" | "sft";
export type FinishReason = "stop" | "length" | "tool_call" | "error";

/** A tool call the environment executed for a turn (`turns[i].info.tool_calls`). */
export interface ExecutedToolCall extends ToolCall {
  result: JsonValue;
}

/** One policy turn of a saved rollout. */
export interface TrajectoryTurn {
  index: number;
  /** Index into `TrajectoryRow.threads`. */
  thread: number;
  /** Position of this turn's assistant message in its thread; `null` when not recorded. */
  message_index: number | null;
  reward: number;
  finish_reason: FinishReason | null;
  prompt_tokens: number;
  response_tokens: number;
  /** The environment's step info. Tool-calling environments set `tool_calls` (`ExecutedToolCall[]`), `invalid_calls` and `final_answer`. */
  info: JsonDict;
}

/** One row of `trajectories.jsonl`. The optional keys are absent on rows of older runs and of backends that do not record them. */
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
  phase?: TrajectoryPhase;
  done?: boolean;
  /** Finish reason of the last turn. */
  finish_reason?: FinishReason | null;
  num_context_tokens?: number;
  /** `reward >= logging.success_threshold`. */
  success?: boolean;
  /** Conversations of the episode: thread 0 is the one it ended in, the others are further contexts
   * (before a compaction, a sub-agent). Omitted with `logging.save_transcripts: false`. */
  threads?: Message[][];
  /** Policy turns in order. Omitted with `logging.save_transcripts: false`. */
  turns?: TrajectoryTurn[];
  /** Episode-level details. */
  info?: JsonDict;
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
  /** Absent on older servers: a config with `base` + `stages` is a pipeline either way. */
  kind?: JobKind;
  config: Record<string, unknown>;
}

export function recipeIsPipeline(r: Recipe): boolean {
  return r.kind === "pipeline" || isPipelineDoc(r.config);
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

/** Placeholder the preview endpoint emits for decoded image cells (PIL / HF `Image` feature). */
export interface ImageCell {
  __image__: true;
  width: number;
  height: number;
  mode?: string;
  path?: string;
}

export function isImageCell(v: unknown): v is ImageCell {
  return !!v && typeof v === "object" && "__image__" in v && v.__image__ === true;
}

export interface PreviewResponse {
  columns: Record<string, string>;
  rows: JsonDict[];
  inferred: {
    format: DataFormat;
    prompt_field: string | null;
    answer_field: string | null;
    messages_field: string | null;
    image_fields?: string[];
  };
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
