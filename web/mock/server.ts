// Bun mock of the monorl server API (docs/ARCHITECTURE.md "Server API") with canned data and a
// fake job that writes metrics/logs/trajectories on a timer and streams them over SSE. Rollout rows
// carry transcripts (`threads` / `turns`); tool-calling runs (`env.type: tool_calling`, e.g. the
// seeded `tool-calling-grpo-seed`) cycle through scripted multi-turn episodes.
// Run: `pnpm dev:mock` (listens on 8000, the Vite dev proxy target).

import { readdirSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { deflateSync } from "node:zlib";

const PORT = Number(process.env.PORT ?? 8000);
const HOME = homedir();

type Dict = Record<string, unknown>;
type RunState = "pending" | "running" | "succeeded" | "failed" | "cancelled";
type StageType = "train" | "rejection_sample" | "eval";

/** Append-only run-dir streams (metrics.jsonl, run.log, trajectories.jsonl, checkpoints/). */
interface RunData {
  metrics: Dict[];
  logs: string[];
  trajectories: Dict[];
  checkpoints: { step: number; path: string }[];
}

const emptyRunData = (): RunData => ({ metrics: [], logs: [], trajectories: [], checkpoints: [] });

interface MockStage extends RunData {
  index: number;
  name: string;
  type: StageType;
  state: RunState;
  run_dir: string;
  started_at: number | null;
  finished_at: number | null;
  step: number;
  summary: Dict;
  error: string | null;
  /** Steps (train) / chunks (rejection) the fake stage runs for. */
  length: number;
  /** Train: the stage's merged RunConfig. Rejection / eval: the stage document. */
  config: Dict;
  /** The RunConfig the stage rolls out with (`config` for train stages, else the pipeline base). */
  run: Dict;
}

interface MockJob extends RunData {
  id: string;
  kind: "run" | "pipeline";
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
  stages: MockStage[];
  timer: Timer | null;
}

const now = () => Date.now() / 1000;
const jobs = new Map<string, MockJob>();

const stamp = () => new Date().toISOString().replace("T", " ").slice(0, 19);

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
        image_field: { description: "Column(s) holding images (HF Image feature, file path/URI, bytes or base64) attached to the prompt as image content parts. Use '<image>' in prompt_template to position them." },
        response_field: { description: "Column holding the target response for SFT (defaults to answer_field / the trailing assistant turn)." },
        tools_field: { description: "Column holding the tool schemas (a list of OpenAI-style function schemas, or its JSON string) a chat row's prompts are rendered with. Default: a 'tools' column of a chat dataset when it holds one." },
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
    OffPolicyConfig: {
      properties: {
        anchor: { enum: ["auto", "behavior", "trainer"], default: "auto", description: "Proximal policy the clipped ratio is anchored on. 'behavior': the policy that sampled the tokens (the engine's log-probs). 'trainer': the trainer's policy when the update on the batch starts (decoupled PPO; one extra forward pass per batch). 'auto': 'trainer' when importance_weights or ratio_mask is set, otherwise the backend's default." },
        importance_weights: { enum: ["none", "token", "sequence"], default: "none", description: "Weigh the policy loss by the truncated importance ratio between the proximal and the behaviour policy, per token or per training sequence (product of its token ratios)." },
        importance_cap: { anyOf: [{ type: "number", minimum: 1 }, { type: "null" }], default: 2, description: "Upper bound of the importance weights (null = untruncated)." },
        ratio_mask: { enum: ["none", "token", "sequence", "sequence_mean"], default: "none", description: "Drop from the policy loss the tokens, or whole training sequences, whose proximal/behaviour importance ratio is outside [ratio_mask_low, ratio_mask_high]. 'sequence' tests the product of a sequence's token ratios, 'sequence_mean' their geometric mean." },
        ratio_mask_low: { type: "number", default: 0.5, exclusiveMinimum: 0, maximum: 1, description: "Lower bound of the accepted importance ratio." },
        ratio_mask_high: { type: "number", default: 5, minimum: 1, description: "Upper bound of the accepted importance ratio." },
        score_centering: { type: "boolean", default: false, description: "Subtract the expected score under the sampling policy from the policy gradient (arXiv:2609.20807). Needs the engine's top-k next-token log-probs and the 'behavior' anchor." },
        score_centering_top_k: { type: "integer", default: 128, minimum: 1, description: "Number of most likely next tokens the engine records per sampled token." },
      },
    },
    AlgorithmConfig: {
      properties: {
        name: { enum: ["grpo", "dapo", "rloo", "reinforce", "ppo", "sft"], default: "grpo", description: "Policy-gradient objective, or 'sft' for supervised fine-tuning on the dataset responses (no rollouts)." },
        sft_label_smoothing: { type: "number", default: 0, minimum: 0, exclusiveMaximum: 1, description: "Label smoothing for sft." },
        sft_train_on_prompt: { type: "boolean", default: false, description: "sft: include prompt tokens in the loss (default: response only)." },
        sft_turns: { enum: ["last", "all"], default: "last", description: "sft: assistant turns of a conversation that carry the loss. 'last': only the final one, earlier turns are context. 'all': every assistant turn (multi-turn and tool-using conversations), except those a row marks with 'weight': 0." },
        kl_coef: { type: "number", default: 0, minimum: 0, description: "Coefficient of the KL penalty to the reference policy (0 disables)." },
        kl_estimator: { enum: ["k1", "k2", "k3"], default: "k3", description: "KL estimator (Schulman's k1/k2/k3)." },
        clip_eps: { type: "number", default: 0.2, exclusiveMinimum: 0, description: "PPO clipping epsilon (lower bound)." },
        clip_eps_high: { description: "Asymmetric upper clip (DAPO clip-higher)." },
        advantage_norm: { enum: ["group", "batch", "none"], default: "group", description: "Normalise rewards within the prompt group (GRPO), across the batch, or not at all." },
        advantage_std_norm: { type: "boolean", default: true, description: "Divide by std in group normalisation (False = Dr.GRPO style)." },
        loss_agg: { enum: ["token", "sequence", "rollout"], default: "token", description: "Average the loss over all response tokens, per training sequence first, or per rollout first ('rollout': every episode counts equally however many sequences its turns form)." },
        entropy_coef: { type: "number", default: 0, minimum: 0 },
        num_epochs: { type: "integer", default: 1, minimum: 1, description: "Optimizer passes over each rollout batch." },
        dynamic_sampling: { type: "boolean", default: false, description: "Drop groups whose rewards are all identical (DAPO)." },
        overlong_penalty: { type: "number", default: 0, minimum: 0, description: "Reward penalty for truncated (length) completions." },
        credit_assignment: { enum: ["episode", "turn"], default: "episode", description: "Which rewards a response token is credited with. 'episode': the reward of its whole episode, the same advantage for every token. 'turn': the return from its own turn on (later per-turn rewards discounted by gamma; the reward-function score arrives at the final turn), compared with the same turn of the other episodes of its prompt group. See docs/CREDIT_ASSIGNMENT.md." },
        gamma: { type: "number", default: 1, minimum: 0, maximum: 1, description: "Discount for multi-turn per-turn rewards." },
        lam: { type: "number", default: 1, minimum: 0, maximum: 1, description: "GAE lambda (ppo only): 1 = returns minus the value baseline, 0 = one-step TD error." },
        value_coef: { type: "number", default: 0.5, description: "Value loss weight (ppo only)." },
        value_clip: { anyOf: [{ type: "number", exclusiveMinimum: 0 }, { type: "null" }], default: 0.5, description: "ppo only: the value loss also penalises a value estimate clipped to this distance from the estimate the advantages were computed with (null = plain squared error)." },
        off_policy: { $ref: "#/$defs/OffPolicyConfig", description: "Off-policy corrections (off by default)." },
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
        max_context_tokens: { anyOf: [{ type: "integer", minimum: 2 }, { type: "null" }], default: null, description: "Context budget of an episode in tokens: no prompt plus completion of a turn may exceed it. A turn's generation is capped to the room that is left, an episode without room for another turn ends, and an agent behind the gateway gets a context_length_exceeded error. Default: trainer.max_seq_len (and rollout.max_model_len when set), so that nothing an episode samples is cut off in training." },
        enable_sleep_mode: { type: "boolean", default: true, description: "Offload engine memory while training (colocated)." },
        async_steps_ahead: { type: "integer", default: 0, minimum: 0, description: "Generate up to N batches ahead of training (off-policy staleness bound). 0 = synchronous." },
        turn_rendering: { enum: ["tokens", "template"], default: "tokens", description: "How a multi-turn prompt is built. 'tokens': the sampled tokens of earlier turns stay in the context and only new messages are rendered. 'template': the chat template re-renders the whole conversation every turn." },
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
        type: { type: "string", default: "single_turn", description: "Environment: 'single_turn', 'tool_calling', 'remote', 'harness', 'openenv', 'hermes', ..." },
        max_turns: { type: "integer", default: 1, minimum: 1 },
        retries: { type: "integer", default: 0, minimum: 0, description: "Times an episode whose environment failed (a crashed agent, a failing step) is started again." },
        max_failed_frac: { type: "number", default: 0, minimum: 0, maximum: 1, description: "Share of a rollout batch's episodes that may fail for good; they are left out of the batch. Above it the run fails. 0 (default): any failed episode fails the run." },
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
        save_transcripts: { type: "boolean", default: true, description: "Include the full conversation and per-turn details of every saved rollout." },
        success_threshold: { type: "number", default: 1, description: "Total reward at or above which an episode counts as solved (success-rate and pass@k metrics)." },
        wandb_project: {},
        tensorboard: { type: "boolean", default: false },
      },
    },
  },
};

// Mirrors PipelineConfig.model_json_schema() (src/monorl/config/pipeline.py).
const PIPELINE_SCHEMA = {
  title: "PipelineConfig",
  type: "object",
  properties: {
    name: { anyOf: [{ type: "string" }, { type: "null" }], default: null, description: "Pipeline run name (defaults to '<model>-pipeline-<timestamp>')." },
    base: { $ref: "#/$defs/RunConfig", description: "Base run configuration every stage starts from." },
    stages: { type: "array", minItems: 1, items: { oneOf: [{ $ref: "#/$defs/TrainStage" }, { $ref: "#/$defs/RejectionSampleStage" }, { $ref: "#/$defs/EvalStage" }], discriminator: { propertyName: "type" } } },
  },
  required: ["base", "stages"],
  $defs: {
    ...SCHEMA.$defs,
    RunConfig: { properties: SCHEMA.properties },
    TrainStage: {
      properties: {
        type: { const: "train", default: "train" },
        name: { type: "string", description: "Stage name (used for the stage directory)." },
        overrides: { type: "object", description: "RunConfig fields merged over the pipeline base (e.g. {algorithm: {name: sft}, trainer: {max_steps: 50}})." },
        init_from: { type: "string", default: "previous", description: "Policy initialisation: 'previous' (last train stage's final export), 'base' (base model), or a path." },
        data_from: { enum: ["previous", "config"], default: "config", description: "'previous' uses the dataset written by the preceding rejection_sample stage as data.uri; an SFT stage then trains every policy turn of its conversations (algorithm.sft_turns: all)." },
      },
      required: ["name"],
    },
    RejectionSampleStage: {
      properties: {
        type: { const: "rejection_sample", default: "rejection_sample" },
        name: { type: "string" },
        data: { anyOf: [{ $ref: "#/$defs/DataConfig" }, { type: "null" }], default: null, description: "Prompts to sample from (defaults to the base data config)." },
        num_samples: { type: "integer", default: 4, minimum: 1, description: "Episodes sampled per prompt." },
        min_reward: { type: "number", default: 1, description: "Keep episodes whose total reward is >= this value." },
        max_per_prompt: { type: "integer", default: 1, minimum: 1, description: "Keep at most this many accepted episodes per prompt." },
        max_response_tokens: { anyOf: [{ type: "integer", minimum: 1 }, { type: "null" }], default: null, description: "Drop episodes in which a policy turn is longer than this many tokens." },
        max_context_tokens: { anyOf: [{ type: "integer", minimum: 1 }, { type: "null" }], default: null, description: "Drop episodes whose context (prompt plus completion of their longest turn) exceeds this many tokens, e.g. the trainer.max_seq_len of the SFT stage that follows." },
        require_finish: { type: "boolean", default: true, description: "Drop episodes in which a policy turn was cut off by the token limit or failed." },
        dedupe: { type: "boolean", default: true, description: "Drop episodes whose conversation repeats one already kept for the same prompt." },
        mix: { type: "array", items: { type: "string" }, description: "Extra chat/prompt_answer dataset URIs appended to the output." },
        rollout_overrides: { type: "object", description: "RolloutConfig overrides for sampling." },
        reward_overrides: { anyOf: [{ type: "array", items: { type: "object" } }, { type: "null" }], default: null, description: "Replace the base rewards for scoring." },
        output: { anyOf: [{ type: "string" }, { type: "null" }], default: null, description: "Output jsonl path (default: <stage dir>/dataset.jsonl)." },
        max_prompts: { anyOf: [{ type: "integer", minimum: 1 }, { type: "null" }], default: null, description: "Sample from at most this many prompts." },
      },
      required: ["name"],
    },
    EvalStage: {
      properties: {
        type: { const: "eval", default: "eval" },
        name: { type: "string" },
        data: { anyOf: [{ $ref: "#/$defs/DataConfig" }, { type: "null" }], default: null, description: "Evaluation dataset (defaults to base data.eval_uri or data)." },
        rollout_overrides: { type: "object", description: "RolloutConfig overrides for evaluation." },
        max_prompts: { anyOf: [{ type: "integer", minimum: 1 }, { type: "null" }], default: null, description: "Evaluate at most this many prompts." },
        num_samples: { type: "integer", default: 1, minimum: 1, description: "Completions per prompt. 1 is greedy; more are sampled at the rollout temperature and reported as eval/pass_at_k." },
      },
      required: ["name"],
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
    env: ["single_turn", "tool_calling", "remote", "harness", "openenv", "hermes"],
    reward: ["exact_match", "format", "regex", "math_verify", "code_exec", "length", "llm_judge", "reward_model", "composite"],
    data_source: ["local", "hf", "s3"],
    weight_sync: ["inprocess", "checkpoint", "broadcast", "cuda_ipc"],
  },
};

const RECIPES: { name: string; description: string; kind: "run" | "pipeline"; config: Dict }[] = [
  {
    kind: "run",
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
    kind: "run",
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
    kind: "run",
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
  {
    kind: "run",
    name: "sft-chat-smol",
    description: "Classic SFT on a chat dataset (no rollouts/rewards); response = trailing assistant turn.",
    config: {
      model: { name_or_path: "HuggingFaceTB/SmolLM2-135M-Instruct" },
      data: { uri: "hf://HuggingFaceH4/ultrafeedback_binarized", format: "chat", messages_field: "messages" },
      algorithm: { name: "sft", sft_label_smoothing: 0.05 },
      trainer: { backend: "reference", max_steps: 200, prompts_per_step: 16, learning_rate: 1e-5, eval_every: 50 },
      rewards: [],
    },
  },
  {
    kind: "run",
    name: "vlm-grpo-vqa",
    description: "GRPO on a VQA dataset with a Qwen2-VL policy; images attached from the `image` column.",
    config: {
      model: { name_or_path: "Qwen/Qwen2-VL-2B-Instruct" },
      data: { uri: "hf://HuggingFaceM4/VQAv2-sample", image_field: "image", prompt_field: "question", answer_field: "answer", format: "prompt_answer", prompt_template: "<image>\n{prompt}\nAnswer with a single word." },
      rollout: { backend: "vllm", group_size: 4, max_new_tokens: 32 },
      trainer: { max_steps: 100 },
      rewards: [{ type: "exact_match", weight: 1.0, params: { normalize: true } }],
      cluster: { gpus_per_node: 1, placement: "colocated" },
    },
  },
  {
    kind: "run",
    name: "tool-calling-grpo",
    description: "Multi-turn GRPO where the policy calls the `calculator` / `python_exec` tools over up to 6 turns; scored on the final answer.",
    config: {
      model: { name_or_path: "Qwen/Qwen2.5-1.5B-Instruct" },
      data: { uri: "hf://openai/gsm8k", hf_config_name: "main", prompt_field: "question", answer_field: "answer", format: "prompt_answer" },
      algorithm: { name: "grpo", loss_agg: "rollout" },
      rollout: { backend: "sglang", group_size: 4, max_new_tokens: 512, max_context_tokens: 1536, stop: ["</tool_call>"] },
      trainer: { backend: "reference", max_steps: 40, prompts_per_step: 8, eval_every: 10 },
      rewards: [{ type: "exact_match", weight: 1.0, params: { extract: "last_number", scope: "last" } }, { type: "format", weight: 0.2, params: { patterns: ["<tool_call>.*</tool_call>"], require_all: false } }],
      env: { type: "tool_calling", max_turns: 6, retries: 1, max_failed_frac: 0.1, params: { tools: ["calculator", "python_exec"], invalid_call_penalty: 0.1 } },
    },
  },
  {
    kind: "pipeline",
    name: "r1-sft-grpo-loop",
    description: "DeepSeek-R1 style loop: cold-start SFT → GRPO → rejection sampling → SFT → GRPO on GSM8K.",
    config: {
      name: null,
      base: {
        model: { name_or_path: "HuggingFaceTB/SmolLM2-135M-Instruct" },
        data: { uri: "hf://openai/gsm8k", hf_config_name: "main", prompt_field: "question", answer_field: "answer", format: "prompt_answer", system_prompt: "Answer with the final number after '####'." },
        algorithm: { name: "grpo" },
        rollout: { backend: "hf", group_size: 4, max_new_tokens: 128, device: "cpu" },
        trainer: { backend: "reference", max_steps: 20, prompts_per_step: 4, precision: "fp32", device: "cpu" },
        rewards: [{ type: "math_verify", weight: 1.0, params: {} }, { type: "format", weight: 0.2, params: { pattern: "####\\s*-?\\d+" } }],
      },
      stages: [
        { type: "train", name: "cold-start-sft", init_from: "base", overrides: { algorithm: { name: "sft" }, trainer: { max_steps: 8, learning_rate: 1e-5 } } },
        { type: "train", name: "grpo-1", overrides: { trainer: { max_steps: 10 } } },
        { type: "rejection_sample", name: "rejection-sample", num_samples: 8, min_reward: 1.0, max_per_prompt: 2, max_prompts: 64 },
        { type: "train", name: "sft-2", data_from: "previous", overrides: { algorithm: { name: "sft" }, trainer: { max_steps: 6, learning_rate: 5e-6 } } },
        { type: "train", name: "grpo-2", overrides: { trainer: { max_steps: 12 } } },
        { type: "eval", name: "final-eval", max_prompts: 100 },
      ],
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

// --- tiny PNG generator (for image-URI preview thumbnails) ---------------------------------------

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(buf: Uint8Array): number {
  let c = 0xffffffff;
  for (const b of buf) c = CRC_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  out.set(new TextEncoder().encode(type), 4);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** A `size`×`size` RGB PNG with a hue-based gradient (distinct per seed). */
function makePng(size: number, seed: number): Uint8Array {
  const raw = new Uint8Array(size * (1 + size * 3));
  for (let y = 0; y < size; y++) {
    raw[y * (1 + size * 3)] = 0; // filter: none
    for (let x = 0; x < size; x++) {
      const o = y * (1 + size * 3) + 1 + x * 3;
      raw[o] = (seed * 60 + x * 2) & 0xff;
      raw[o + 1] = (seed * 90 + y * 2) & 0xff;
      raw[o + 2] = (seed * 130 + ((x + y) & 0x7f)) & 0xff;
    }
  }
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, size);
  dv.setUint32(4, size);
  ihdr.set([8, 2, 0, 0, 0], 8); // 8-bit RGB
  const parts = [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), pngChunk("IHDR", ihdr), pngChunk("IDAT", new Uint8Array(deflateSync(raw))), pngChunk("IEND", new Uint8Array())];
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

const PNG_CACHE = new Map<number, Uint8Array>();
function mockPng(seed: number): Uint8Array {
  let png = PNG_CACHE.get(seed);
  if (!png) PNG_CACHE.set(seed, (png = makePng(64, seed)));
  return png;
}

function preview(body: { uri: string; limit?: number }) {
  const limit = body.limit ?? 5;
  if (/vqa|image|coco|vision/i.test(body.uri)) {
    // Mirrors monorl.data.browse: decoded images become {__image__, width, height}; URI columns pass through.
    const questions = ["What color is the bus?", "How many people are in the picture?", "Is it raining?", "What sport is being played?", "What is written on the sign?"];
    const answers = ["yellow", "3", "no", "tennis", "stop"];
    const rows = Array.from({ length: limit }, (_, i) => ({
      image: { __image__: true, width: 640, height: 480 - (i % 2) * 60, mode: "RGB" },
      image_url: `http://127.0.0.1:${PORT}/api/mock/images/${i % 5}.png`,
      image_path: `/data/vqa/images/${String(100 + i).padStart(6, "0")}.jpg`,
      question: questions[i % 5],
      answer: answers[i % 5],
      question_id: 1000 + i,
    }));
    return {
      columns: { image: "Image", image_url: "string", image_path: "string", question: "string", answer: "string", question_id: "int64" },
      rows,
      inferred: { format: "prompt_answer", prompt_field: "question", answer_field: "answer", messages_field: null, image_fields: ["image"] },
      num_rows: 2048,
    };
  }
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

// Mirrors RunConfig validation (src/monorl/config/schema.py); `sft` skips rollout/reward checks.
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
  const ALGOS = ["grpo", "dapo", "rloo", "reinforce", "ppo", "sft"];
  if (!ALGOS.includes(name)) errors.push({ loc: ["algorithm", "name"], msg: `Input should be ${ALGOS.map((a) => `'${a}'`).join(", ").replace(/, '(\w+)'$/, " or '$1'")}`, type: "literal_error" });
  const smoothing = algo.sft_label_smoothing as number | undefined;
  if (smoothing !== undefined && !(smoothing >= 0 && smoothing < 1)) errors.push({ loc: ["algorithm", "sft_label_smoothing"], msg: "Input should be greater than or equal to 0 and less than 1", type: "value_error" });
  const gs = (rollout.group_size as number | undefined) ?? 4;
  if (typeof gs !== "number" || gs < 1) errors.push({ loc: ["rollout", "group_size"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
  const lr = trainer.learning_rate as number | undefined;
  if (lr !== undefined && !(lr > 0)) errors.push({ loc: ["trainer", "learning_rate"], msg: "Input should be greater than 0", type: "greater_than" });
  const steps = trainer.max_steps as number | undefined;
  if (steps !== undefined && !(steps >= 1)) errors.push({ loc: ["trainer", "max_steps"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
  // OffPolicyConfig: field bounds, then its two combination rules.
  const off = (algo.off_policy as Dict | undefined) ?? {};
  const offLoc = (field: string) => ["algorithm", "off_policy", field];
  if (typeof off.importance_cap === "number" && off.importance_cap < 1) errors.push({ loc: offLoc("importance_cap"), msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
  if (typeof off.ratio_mask_low === "number" && !(off.ratio_mask_low > 0 && off.ratio_mask_low <= 1)) errors.push({ loc: offLoc("ratio_mask_low"), msg: "Input should be greater than 0 and less than or equal to 1", type: "value_error" });
  if (typeof off.ratio_mask_high === "number" && off.ratio_mask_high < 1) errors.push({ loc: offLoc("ratio_mask_high"), msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
  if (typeof off.score_centering_top_k === "number" && off.score_centering_top_k < 1) errors.push({ loc: offLoc("score_centering_top_k"), msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
  const anchor = (off.anchor as string | undefined) ?? "auto";
  const corrects = ((off.importance_weights as string | undefined) ?? "none") !== "none" || ((off.ratio_mask as string | undefined) ?? "none") !== "none";
  if (corrects && anchor === "behavior")
    errors.push({ loc: ["algorithm", "off_policy"], msg: "Value error, off_policy.importance_weights / off_policy.ratio_mask compare the proximal with the behaviour policy, which are the same policy under off_policy.anchor='behavior'; use anchor 'trainer'", type: "value_error" });
  else if (off.score_centering === true && (anchor === "trainer" || (anchor === "auto" && corrects)))
    errors.push({ loc: ["algorithm", "off_policy"], msg: "Value error, off_policy.score_centering weighs scores by the ratio of the current to the behaviour policy and needs off_policy.anchor='behavior' (or 'auto' without importance_weights / ratio_mask)", type: "value_error" });
  if (typeof env.retries === "number" && env.retries < 0) errors.push({ loc: ["env", "retries"], msg: "Input should be greater than or equal to 0", type: "greater_than_equal" });
  const maxFailed = Number(env.max_failed_frac ?? 0);
  if (!(maxFailed >= 0 && maxFailed <= 1)) errors.push({ loc: ["env", "max_failed_frac"], msg: "Input should be greater than or equal to 0 and less than or equal to 1", type: "value_error" });
  const budget = (rollout.max_context_tokens as number | null | undefined) ?? null;
  if (budget !== null && budget < 2) errors.push({ loc: ["rollout", "max_context_tokens"], msg: "Input should be greater than or equal to 2", type: "greater_than_equal" });
  if (errors.length === 0) {
    const sft = name === "sft";
    if (!sft && ["grpo", "dapo", "rloo"].includes(name) && gs < 2) errors.push({ loc: [], msg: `Value error, ${name} needs rollout.group_size >= 2`, type: "value_error" });
    if (cluster.placement === "disaggregated" && Number(cluster.gpus_per_node ?? 0) > 0) {
      const r = cluster.rollout_gpus as number | null | undefined;
      const t = cluster.trainer_gpus as number | null | undefined;
      const total = Number(cluster.num_nodes ?? 1) * Number(cluster.gpus_per_node ?? 0);
      if (r == null || t == null) errors.push({ loc: [], msg: "Value error, disaggregated placement needs rollout_gpus and trainer_gpus", type: "value_error" });
      else if (r + t > total) errors.push({ loc: [], msg: `Value error, rollout_gpus + trainer_gpus = ${r + t} exceeds ${total} GPUs`, type: "value_error" });
    }
    if (!sft && rewards.length === 0 && ((env.type as string | undefined) ?? "single_turn") === "single_turn")
      errors.push({ loc: [], msg: "Value error, single_turn env needs at least one reward in `rewards`", type: "value_error" });
    const maxSeqLen = Number(trainer.max_seq_len ?? 2048);
    const backend = String(trainer.backend ?? "reference");
    if (!sft && budget !== null && budget > maxSeqLen)
      errors.push({ loc: [], msg: `Value error, rollout.max_context_tokens=${budget} exceeds trainer.max_seq_len=${maxSeqLen}: training would cut such episodes off; raise trainer.max_seq_len or lower the budget`, type: "value_error" });
    if (!sft && SYSTEM.components.pipeline_backend.includes(backend)) {
      if (maxFailed > 0)
        errors.push({ loc: [], msg: `Value error, env.max_failed_frac is not supported by trainer.backend='${backend}': failed episodes can only be left out of a batch by the native loop ('reference', 'unsloth')`, type: "value_error" });
      if (backend !== "trl" && (Number(env.retries ?? 0) > 0 || budget !== null))
        errors.push({ loc: [], msg: `Value error, env.retries and rollout.max_context_tokens are not honoured by trainer.backend='${backend}', which runs its own rollouts; use its own length limits`, type: "value_error" });
    }
    const extra = (rollout.extra as Dict | undefined) ?? {};
    if (!sft && off.score_centering === true && (Number(rollout.temperature ?? 1) <= 0 || Number(rollout.top_k ?? -1) > 0 || Number(rollout.top_p ?? 1) < 1 || Number(extra.min_p ?? 0) > 0 || Number(extra.repetition_penalty ?? 1) !== 1))
      errors.push({ loc: [], msg: "Value error, off_policy.score_centering needs the full sampling distribution: rollout.temperature > 0 and no top_k / top_p / min_p / repetition_penalty truncation", type: "value_error" });
  }
  return { ok: errors.length === 0, errors };
}

function deepMerge(base: unknown, patch: unknown): unknown {
  const isObj = (v: unknown): v is Dict => typeof v === "object" && v !== null && !Array.isArray(v);
  if (!isObj(base) || !isObj(patch)) return patch === undefined ? base : patch;
  const out: Dict = { ...base };
  for (const [k, v] of Object.entries(patch)) out[k] = isObj(out[k]) && isObj(v) ? deepMerge(out[k], v) : v;
  return out;
}

// Mirrors PipelineConfig validation (src/monorl/config/pipeline.py).
function validatePipeline(pipeline: unknown): { ok: boolean; errors: { loc: (string | number)[]; msg: string; type: string }[] } {
  const errors: { loc: (string | number)[]; msg: string; type: string }[] = [];
  const p = (pipeline ?? {}) as Dict;
  const base = validate(p.base);
  for (const e of base.errors) errors.push({ ...e, loc: ["base", ...e.loc] });
  const stages = Array.isArray(p.stages) ? (p.stages as Dict[]) : null;
  if (!stages) errors.push({ loc: ["stages"], msg: "Field required", type: "missing" });
  else if (stages.length === 0) errors.push({ loc: ["stages"], msg: "List should have at least 1 item after validation, not 0", type: "too_short" });
  else {
    const names = stages.map((s) => s.name);
    stages.forEach((s, i) => {
      if (!["train", "rejection_sample", "eval"].includes(String(s.type ?? "train"))) errors.push({ loc: ["stages", i], msg: "Input tag 'type' found using 'type' does not match any of the expected tags: 'train', 'rejection_sample', 'eval'", type: "union_tag_invalid" });
      if (typeof s.name !== "string" || !s.name) errors.push({ loc: ["stages", i, "name"], msg: "Field required", type: "missing" });
      if ((s.type ?? "train") === "train") {
        const merged = validate(deepMerge(p.base, s.overrides ?? {}));
        for (const e of merged.errors) errors.push({ ...e, loc: ["stages", i, "overrides", ...e.loc] });
      }
      if (s.type === "rejection_sample") {
        if (typeof s.num_samples === "number" && s.num_samples < 1) errors.push({ loc: ["stages", i, "num_samples"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
        if (typeof s.max_per_prompt === "number" && s.max_per_prompt < 1) errors.push({ loc: ["stages", i, "max_per_prompt"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
      }
      if (s.type === "eval" && typeof s.num_samples === "number" && s.num_samples < 1) errors.push({ loc: ["stages", i, "num_samples"], msg: "Input should be greater than or equal to 1", type: "greater_than_equal" });
    });
    if (errors.length === 0) {
      if (new Set(names).size !== names.length) errors.push({ loc: [], msg: `Value error, stage names must be unique, got ${JSON.stringify(names)}`, type: "value_error" });
      stages.forEach((s, i) => {
        if ((s.type ?? "train") === "train" && s.data_from === "previous" && (i === 0 || stages[i - 1].type !== "rejection_sample"))
          errors.push({ loc: [], msg: `Value error, stage '${s.name}': data_from='previous' needs a rejection_sample stage right before it`, type: "value_error" });
      });
    }
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

/** What the fake rollouts of a run look like: tool-calling agent episodes or single-turn completions. */
function rolloutShape(config: Dict): { tools: boolean; maxTurns: number } {
  const env = (config.env as Dict | undefined) ?? {};
  const tools = env.type === "tool_calling";
  return { tools, maxTurns: tools ? Math.max(1, Number(env.max_turns ?? 1)) : 1 };
}

const jitter = (width: number) => (Math.random() - 0.5) * width;

/** Agent metrics of one batch at progress `p` (0..1), under `episode/` for train rows and `eval/` for eval rows. */
function episodeMetrics(prefix: "episode" | "eval", p: number, config: Dict): Dict {
  const { tools, maxTurns } = rolloutShape(config);
  const turns = tools ? Math.min(maxTurns, 2.6 + 1.2 * (1 - p) + jitter(0.3)) : 1;
  const truncated = Math.max(0, 0.12 * (1 - p) + jitter(0.03));
  return {
    [`${prefix}/turns_mean`]: turns,
    [`${prefix}/turns_max`]: tools ? Math.min(maxTurns, Math.ceil(turns) + 1) : 1,
    [`${prefix}/done_frac`]: tools ? 1 - truncated / 2 : 1,
    [`${prefix}/truncated_frac`]: truncated,
    [`${prefix}/context_tokens_mean`]: (tools ? 420 * turns : 150) + jitter(20),
    ...(tools ? { [`${prefix}/tool_calls_mean`]: Math.max(0, turns - 1 + jitter(0.2)), [`${prefix}/invalid_calls_mean`]: Math.max(0, 0.6 * (1 - p) + jitter(0.1)), [`${prefix}/context_exhausted_frac`]: Math.max(0, 0.2 * (1 - p) + jitter(0.04)) } : {}),
  };
}

/** One fake train-phase metrics row at `step` of `maxSteps` (RL or SFT keys depending on `sft`). */
function trainRow(step: number, maxSteps: number, lr0: number, sft: boolean, config: Dict): Dict {
  const p = step / maxSteps;
  if (sft) {
    const loss = 2.4 * Math.exp(-2.5 * p) + 0.35 + (Math.random() - 0.5) * 0.05;
    return {
      step,
      time: now(),
      phase: "train",
      "loss/sft": loss,
      "sft/ppl": Math.exp(loss),
      "sft/tokens": 1800 + Math.floor(Math.random() * 300),
      "time/train": 0.9 + Math.random() * 0.3,
      lr: lr0 * (1 - 0.5 * p),
      ...(step % 5 === 0 ? { "eval/loss": loss + 0.08 + Math.random() * 0.03 } : {}),
    };
  }
  const solved = Math.max(0, Math.min(1, 0.1 + 0.6 * p + (Math.random() - 0.5) * 0.1));
  const groupSize = Number((config.rollout as Dict | undefined)?.group_size ?? 4);
  const maxFailedFrac = Number((config.env as Dict | undefined)?.max_failed_frac ?? 0);
  const episode = episodeMetrics("episode", p, config);
  return {
    step,
    time: now(),
    phase: "train",
    "reward/mean": Math.min(1, 0.15 + 0.7 * (1 - Math.exp(-3 * p)) + (Math.random() - 0.5) * 0.1),
    "reward/std": Math.max(0.02, 0.45 * (1 - p) + (Math.random() - 0.5) * 0.05),
    "reward/exact_match": solved,
    "reward/success_rate": solved,
    "reward/pass_at_k": 1 - (1 - solved) ** groupSize,
    ...episode,
    "episode/context_tokens_max": Math.round((episode["episode/context_tokens_mean"] as number) * 1.6),
    // Reported only by runs that tolerate failed episodes (`env.max_failed_frac` > 0).
    ...(maxFailedFrac > 0 ? { "rollout/failed_frac": Math.max(0, Math.min(maxFailedFrac, 0.04 + jitter(0.06))) } : {}),
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
}

/** One fake eval-phase metrics row: `prompts` prompts, `numSamples` completions each (`eval/pass_at_k` only above 1). */
function evalRow(step: number, p: number, config: Dict, prompts: number, numSamples: number): Dict {
  const solved = Math.max(0, Math.min(1, 0.15 + 0.55 * p + jitter(0.1)));
  return {
    step,
    time: now(),
    phase: "eval",
    "eval/reward_mean": solved + 0.19,
    "eval/reward_std": 0.4,
    "eval/exact_match": solved,
    "eval/format": 0.95,
    "eval/success_rate": solved,
    ...(numSamples > 1 ? { "eval/pass_at_k": 1 - (1 - solved) ** numSamples } : {}),
    ...episodeMetrics("eval", p, config),
    "eval/num_samples": prompts * numSamples,
    "eval/time": 12.5,
  };
}

// --- fake rollouts (rows of trajectories.jsonl) ------------------------------------------------

type Phase = "train" | "eval" | "rejection";

interface Msg {
  role: "system" | "user" | "assistant" | "tool";
  content: string | Dict[];
  name?: string;
  tool_call_id?: string;
  tool_calls?: Dict[];
}

/** A scripted episode: its conversations (thread 0 = the one it ends in) and its policy turns in order. */
interface Episode {
  threads: Msg[][];
  turns: { thread: number; message_index: number; reward: number; finish_reason: string; info: Dict }[];
  solved: boolean;
  reference: string;
  info: Dict;
}

const messageText = (m: Msg) => (typeof m.content === "string" ? m.content : m.content.map((part) => (part.type === "text" ? String(part.text) : "")).join(""));
/** ~4 characters per token, and a fixed budget per image. */
const messageTokens = (m: Msg) => 4 + Math.ceil(messageText(m).length / 4) + (typeof m.content === "string" ? 0 : 256 * m.content.filter((part) => part.type === "image").length);

/** The trajectories.jsonl row of `episode`; `threads` / `turns` are left out when `logging.save_transcripts` is off. */
function episodeRow(episode: Episode, at: { step: number; sample_id: string; group_id: string; phase: Phase }, config: Dict): Dict {
  const logging = (config.logging as Dict | undefined) ?? {};
  const turns = episode.turns.map((t, index) => {
    const thread = episode.threads[t.thread];
    return { index, ...t, prompt_tokens: thread.slice(0, t.message_index).reduce((n, m) => n + messageTokens(m), 0), response_tokens: messageTokens(thread[t.message_index]) };
  });
  const first = turns[0];
  const last = turns[turns.length - 1];
  const calls = turns.reduce((n, t) => n + ((t.info.tool_calls as unknown[] | undefined)?.length ?? 0), 0);
  const { tools } = rolloutShape(config);
  const breakdown: Record<string, number> = { exact_match: episode.solved ? 1 : 0, format: tools ? Number(calls > 0) : 1 };
  const penalties = turns.reduce((sum, t) => sum + t.reward, 0);
  if (penalties !== 0) breakdown.env = penalties;
  // Tool-calling runs weigh the format reward 0.2 and add the env's per-turn penalties; the others report the exact match alone.
  const reward = breakdown.exact_match + (tools ? 0.2 * breakdown.format : 0) + penalties;
  return {
    ...at,
    prompt: messageText(episode.threads[first.thread].find((m) => m.role === "user")!),
    response: messageText(episode.threads[last.thread][last.message_index]),
    reward,
    reward_breakdown: breakdown,
    num_turns: turns.length,
    num_response_tokens: turns.reduce((n, t) => n + t.response_tokens, 0),
    reference: episode.reference,
    done: last.finish_reason !== "tool_call" && episode.info.context_exhausted !== true,
    finish_reason: last.finish_reason,
    num_context_tokens: last.prompt_tokens + last.response_tokens,
    success: reward >= Number(logging.success_threshold ?? 1),
    info: episode.info,
    ...(logging.save_transcripts === false ? {} : { threads: episode.threads, turns }),
  };
}

/** A single-turn completion; the user message carries an image part when the run has `data.image_field`. */
function singleTurnEpisode(seed: number, solved: boolean, config: Dict): Episode {
  const data = (config.data as Dict | undefined) ?? {};
  const prompt = PROMPTS[seed % PROMPTS.length];
  const messages: Msg[] = [];
  if (typeof data.system_prompt === "string" && data.system_prompt) messages.push({ role: "system", content: data.system_prompt });
  messages.push({
    role: "user",
    content: data.image_field ? [{ type: "image", image: `http://127.0.0.1:${PORT}/api/mock/images/${seed % 5}.png` }, { type: "text", text: prompt }] : prompt,
  });
  messages.push({ role: "assistant", content: solved ? "Let's think step by step. In May she sold 48/2 = 24 clips. 48 + 24 = 72.\n#### 72" : "The answer is 50 because 48 + 2 = 50.\n#### 50" });
  return { threads: [messages], turns: [{ thread: 0, message_index: messages.length - 1, reward: 0, finish_reason: "stop", info: {} }], solved, reference: "72", info: {} };
}

// Tool-calling episodes follow monorl.envs.tool_calling: the assistant writes <tool_call> blocks, the
// env answers with one `tool` message per call (or an `error` one per unparsable block) and reports
// `{turn, tool_calls, invalid_calls, final_answer}` as the step info; an invalid call costs the penalty.

const INVALID_CALL_PENALTY = 0.1;

const TOOL_SYSTEM_PROMPT = `You are a careful assistant that solves word problems step by step. You may call tools.

# Tools
<tools>
${JSON.stringify({ name: "calculator", description: "Evaluate an arithmetic expression.", parameters: { type: "object", properties: { expression: { type: "string", description: "e.g. '48 / 2 + 7'" } }, required: ["expression"] } })}
${JSON.stringify({ name: "python_exec", description: "Run Python code and return what it prints.", parameters: { type: "object", properties: { code: { type: "string" } }, required: ["code"] } })}
</tools>

To call a tool, reply with one block per call:
<tool_call>
{"name": <tool name>, "arguments": <arguments object>}
</tool_call>
The tool results come back as tool messages. When you know the answer, reply with your reasoning and the final number after '####'.`;

interface ToolTask {
  question: string;
  /** Calculator expressions and their values, in solving order. */
  calc: [expression: string, value: string][];
  answer: string;
}

const TOOL_TASKS: ToolTask[] = [
  { question: PROMPTS[0], calc: [["48 / 2", "24"], ["48 + 24", "72"]], answer: "72" },
  { question: PROMPTS[1], calc: [["12 / 60", "0.2"], ["0.2 * 50", "10"]], answer: "10" },
  { question: PROMPTS[3], calc: [["3 * 2 * 2", "12"], ["12 * 52", "624"]], answer: "624" },
];

interface Call {
  name: string;
  arguments: Dict;
  result: string;
  /** The env rejected it (an unknown tool): counted in `invalid_calls`. */
  invalid?: boolean;
}

const calculate = ([expression, value]: [string, string]): Call => ({ name: "calculator", arguments: { expression }, result: value });

/** A python_exec call whose output is long enough to need collapsing in the viewer. */
function tabulate(factor: number): Call {
  const code = `factor = ${factor}\nprint(" n |  n*factor |    n**2")\nfor n in range(1, 41):\n    print(f"{n:>2} | {n * factor:>9} | {n * n:>7}")`;
  const lines = Array.from({ length: 40 }, (_, i) => `${String(i + 1).padStart(2)} | ${String(Number(((i + 1) * factor).toFixed(6))).padStart(9)} | ${String((i + 1) ** 2).padStart(7)}`);
  return { name: "python_exec", arguments: { code }, result: [" n |  n*factor |    n**2", ...lines].join("\n") };
}

/**
 * Append one policy turn to `episode.threads[thread]`: the assistant message (`text` plus a block
 * per call), then the tool messages the env answers with. `errors` are parse failures of blocks
 * already written in `text`.
 */
function act(episode: Episode, thread: number, text: string, opts: { calls?: Call[]; errors?: string[]; finish?: string } = {}) {
  const { calls = [], errors = [] } = opts;
  const messages = episode.threads[thread];
  const turn = episode.turns.length + 1;
  const executed = calls.map((c, i) => ({ id: `call_${turn}_${i + 1}`, name: c.name, arguments: c.arguments, result: c.result }));
  const blocks = executed.map((c) => `<tool_call>\n${JSON.stringify({ name: c.name, arguments: c.arguments })}\n</tool_call>`);
  messages.push({
    role: "assistant",
    content: [text, ...blocks].join("\n"),
    ...(executed.length ? { tool_calls: executed.map((c) => ({ id: c.id, name: c.name, arguments: c.arguments })) } : {}),
  });
  const invalid = errors.length + calls.filter((c) => c.invalid).length;
  episode.turns.push({
    thread,
    message_index: messages.length - 1,
    reward: invalid ? -INVALID_CALL_PENALTY : 0,
    finish_reason: opts.finish ?? (executed.length || errors.length ? "tool_call" : "stop"),
    info: { turn, tool_calls: executed, invalid_calls: invalid, final_answer: executed.length === 0 && errors.length === 0 },
  });
  for (const c of executed) messages.push({ role: "tool", name: c.name, tool_call_id: c.id, content: c.result });
  for (const e of errors) messages.push({ role: "tool", name: "error", content: `error: ${e}` });
}

const toolEpisode = (task: ToolTask, user: Msg["content"], solved: boolean, info: Dict = {}): Episode => ({
  threads: [[{ role: "system", content: TOOL_SYSTEM_PROMPT }, { role: "user", content: user }]],
  turns: [],
  solved,
  reference: task.answer,
  info,
});

const finalAnswer = (task: ToolTask, solved: boolean) => (solved ? `Putting the two results together gives ${task.answer}.\n#### ${task.answer}` : `So the answer is ${task.calc[0][1]}.\n#### ${task.calc[0][1]}`);

/** The fake tool-calling episodes; a run cycles through them so every part of the transcript viewer shows up. */
const TOOL_SCRIPTS: ((task: ToolTask) => Episode)[] = [
  // Three turns in one conversation: two calculator calls, then the right final answer.
  (task) => {
    const ep = toolEpisode(task, task.question, true);
    act(ep, 0, "I need the first quantity.", { calls: [calculate(task.calc[0])] });
    act(ep, 0, `That is ${task.calc[0][1]}. Now the total.`, { calls: [calculate(task.calc[1])] });
    act(ep, 0, finalAnswer(task, true));
    return ep;
  },
  // Invalid calls (a block that is not JSON, an unknown tool) with their penalties, a long tool result, and an answer cut off by max_new_tokens.
  (task) => {
    const ep = toolEpisode(task, task.question, false);
    act(ep, 0, `Let me compute.\n<tool_call>\n{"name": "calculator", "arguments": {"expression": ${task.calc[0][0]}}\n</tool_call>`, { errors: ["invalid JSON in <tool_call>: Expecting ',' delimiter: line 1 column 52 (char 51)"] });
    act(ep, 0, "The call was malformed. I will tabulate the multiples in Python instead.", { calls: [tabulate(Number(task.calc[0][1]))] });
    act(ep, 0, "Let me double-check with a computer algebra system.", { calls: [{ name: "wolfram", arguments: { input: task.calc[1][0] }, result: "error: unknown tool 'wolfram'; available: ['calculator', 'python_exec']", invalid: true }] });
    act(ep, 0, `Going by the table above, the first quantity is ${task.calc[0][1]}. The question then asks for the combined amount, which means I still have to take that value and`, { finish: "length" });
    return ep;
  },
  // Two threads: the context was compacted after turn 2, so turns 1-2 live in thread 1 and the episode ends in thread 0.
  (task) => {
    const ep = toolEpisode(
      task,
      `[The conversation was compacted after 2 turns to stay within the context window.]\n\nTask: ${task.question}\n\nProgress so far:\n- python_exec printed the multiples of ${task.calc[0][1]} (not needed any more)\n- calculator: ${task.calc[0][0]} = ${task.calc[0][1]}`,
      true,
      { compactions: 1, compacted_after_turn: 2 },
    );
    ep.threads.push([{ role: "system", content: TOOL_SYSTEM_PROMPT }, { role: "user", content: task.question }]);
    act(ep, 1, "I will start with the first quantity.", { calls: [calculate(task.calc[0])] });
    act(ep, 1, `That is ${task.calc[0][1]}. Let me look at its multiples before going on.`, { calls: [tabulate(Number(task.calc[0][1]))] });
    act(ep, 0, "Continuing from the summary: only the total is missing.", { calls: [calculate(task.calc[1])] });
    act(ep, 0, finalAnswer(task, true));
    return ep;
  },
  // Out of context: two long tool results fill the budget (rollout.max_context_tokens), the third turn is cut short and the episode ends unfinished.
  (task) => {
    const ep = toolEpisode(task, task.question, false, { context_exhausted: true });
    act(ep, 0, "Let me tabulate the multiples first.", { calls: [tabulate(Number(task.calc[0][1]))] });
    act(ep, 0, "And the multiples of the second quantity.", { calls: [tabulate(Number(task.calc[1][1]))] });
    act(ep, 0, `From the two tables, ${task.calc[0][0]} is ${task.calc[0][1]}, and the total should then be`, { finish: "length" });
    return ep;
  },
  // Image parts in the user message (a URL, a file path, one the backend left out), two calls in one turn, and a wrong final answer.
  (task) => {
    const ep = toolEpisode(task, [{ type: "image", image: `http://127.0.0.1:${PORT}/api/mock/images/2.png` }, { type: "image", image: "/data/worksheets/scan_000123.jpg" }, { type: "image", image: null, note: "inline image omitted (2.4 MB)" }, { type: "text", text: `The scanned worksheet above states the problem:\n${task.question}` }], false);
    act(ep, 0, "I will compute both quantities at once.", { calls: task.calc.map(calculate) });
    act(ep, 0, finalAnswer(task, false));
    return ep;
  },
];

/** `count` fake rollouts of `phase` at `step`: scripted agent episodes for tool-calling runs, else single-turn completions solved at `solveRate`. */
function fakeTrajectories(step: number, solveRate: number, count: number, phase: Phase, config: Dict): Dict[] {
  const { tools } = rolloutShape(config);
  return Array.from({ length: count }, (_, g) => {
    const episode = tools ? TOOL_SCRIPTS[(step + g) % TOOL_SCRIPTS.length](TOOL_TASKS[(step + g) % TOOL_TASKS.length]) : singleTurnEpisode(step + g, Math.random() < solveRate, config);
    return episodeRow(episode, { step, sample_id: `${phase[0]}${step}-${g}`, group_id: `g${step}`, phase }, config);
  });
}

/** Rows as runs wrote them before transcripts existed: no `phase`, `success`, `threads` or `turns`. */
function legacyTrajectories(step: number, rewardMean: number, count: number): Dict[] {
  return Array.from({ length: count }, (_, g) => {
    const reward = Math.random() < rewardMean ? 1 : 0;
    return {
      step,
      sample_id: `s${step}-${g}`,
      group_id: `g${step}`,
      prompt: PROMPTS[(step + g) % PROMPTS.length],
      response: reward ? "Let's think step by step. In May she sold 48/2 = 24 clips. 48 + 24 = 72.\n#### 72" : "The answer is 50 because 48 + 2 = 50.\n#### 50",
      reward,
      reward_breakdown: { exact_match: reward, format: 1 },
      num_turns: 1,
      num_response_tokens: 30 + Math.floor(Math.random() * 40),
      reference: "72",
    };
  });
}

/** Advance one train step on `run` (a job or a train stage). Returns the metrics row. */
function trainTick(run: RunData & { step: number; run_dir: string }, config: Dict, log: (line: string) => void): Dict {
  const trainer = (config.trainer as Dict | undefined) ?? {};
  const maxSteps = Number(trainer.max_steps ?? 100);
  const saveEvery = Number(trainer.save_every ?? 50);
  const sft = ((config.algorithm as Dict | undefined)?.name ?? "grpo") === "sft";
  const step = run.step + 1;
  const row = trainRow(step, maxSteps, Number(trainer.learning_rate ?? 1e-6), sft, config);
  run.metrics.push(row);
  run.step = step;
  if (sft) log(`step ${step}/${maxSteps} loss=${(row["loss/sft"] as number).toFixed(3)} ppl=${(row["sft/ppl"] as number).toFixed(2)}`);
  else {
    log(`step ${step}/${maxSteps} reward=${(row["reward/mean"] as number).toFixed(3)} kl=${(row["loss/kl"] as number).toFixed(4)}`);
    run.trajectories.push(...fakeTrajectories(step, row["reward/success_rate"] as number, Math.min(2, Number((config.rollout as Dict | undefined)?.group_size ?? 4)), "train", config));
    const evalEvery = Number(trainer.eval_every ?? 0);
    if (evalEvery > 0 && step % evalEvery === 0) {
      const evalMetrics = evalRow(step, step / maxSteps, config, 50, 1);
      run.metrics.push(evalMetrics);
      run.trajectories.push(...fakeTrajectories(step, evalMetrics["eval/success_rate"] as number, 2, "eval", config));
      log(`eval at step ${step}: reward=${(evalMetrics["eval/reward_mean"] as number).toFixed(3)} success=${(evalMetrics["eval/success_rate"] as number).toFixed(3)}`);
    }
  }
  if (step % saveEvery === 0 || step === maxSteps) {
    run.checkpoints.push({ step, path: `${run.run_dir}/checkpoints/step-${step}` });
    log(`saved checkpoint step-${step}`);
  }
  return row;
}

function startFakeJob(job: MockJob) {
  const maxSteps = Number((job.config.trainer as Dict | undefined)?.max_steps ?? 100);
  job.state = "running";
  job.started_at = now();
  job.pid = 40_000 + Math.floor(Math.random() * 1000);
  job.message = "training";
  const log = (line: string) => {
    job.logs.push(`${stamp()} ${line}`);
  };
  log(`[running] starting run ${job.name} (${maxSteps} steps, mock)`);
  log(`loading model ${(job.config.model as Dict).name_or_path}`);
  job.timer = setInterval(() => {
    if (job.state !== "running") return;
    trainTick(job, job.config, log);
    job.updated_at = now();
    if (job.step >= maxSteps) finishJob(job, "succeeded", "finished");
  }, 1000);
}

// --- fake pipeline ---------------------------------------------------------------------------

function buildStages(job: MockJob): MockStage[] {
  const base = (job.config.base as Dict | undefined) ?? {};
  const stages = (job.config.stages as Dict[] | undefined) ?? [];
  return stages.map((s, index) => {
    const type = (s.type as StageType | undefined) ?? "train";
    const name = String(s.name ?? `stage-${index + 1}`);
    const config = type === "train" ? (deepMerge(base, s.overrides ?? {}) as Dict) : s;
    const length = type === "train" ? Number((config.trainer as Dict | undefined)?.max_steps ?? 10) : type === "rejection_sample" ? Math.max(2, Math.ceil(Number(s.max_prompts ?? 64) / 16)) : 1;
    return {
      index,
      name,
      type,
      state: "pending",
      run_dir: `${job.run_dir}/stages/${String(index).padStart(2, "0")}-${name}`,
      started_at: null,
      finished_at: null,
      step: 0,
      summary: {},
      error: null,
      length,
      config,
      run: type === "train" ? config : base,
      ...emptyRunData(),
    };
  });
}

const stageLogger = (st: MockStage) => (line: string) => {
  st.logs.push(`${stamp()} ${line}`);
};

function pipelineLog(job: MockJob, line: string) {
  job.logs.push(`${stamp()} ${line}`);
}

function beginStage(job: MockJob, st: MockStage) {
  st.state = "running";
  st.started_at = now();
  job.message = `stage ${st.index + 1}/${job.stages.length} ${st.name} (${st.type})`;
  pipelineLog(job, `[stage ${st.index + 1}/${job.stages.length}] ${st.name} (${st.type}) starting in ${st.run_dir}`);
  const log = stageLogger(st);
  if (st.type === "train") {
    const prevTrain = job.stages.slice(0, st.index).filter((x) => x.type === "train").pop();
    const init = st.config.init_from ?? "previous";
    log(`[running] ${(st.config.algorithm as Dict | undefined)?.name ?? "grpo"} run, ${st.length} steps; init_from=${init === "previous" && prevTrain ? `${prevTrain.run_dir}/final` : init === "previous" ? "base model" : init}`);
    if (st.index > 0 && job.stages[st.index - 1].type === "rejection_sample" && st.config.data_from === "previous") log(`data.uri = ${job.stages[st.index - 1].run_dir}/dataset.jsonl`);
  } else if (st.type === "rejection_sample") log(`[running] sampling ${st.config.num_samples ?? 4} episodes/prompt, keep reward >= ${st.config.min_reward ?? 1}`);
  else log(`[running] ${Number(st.config.num_samples ?? 1) > 1 ? `eval with ${st.config.num_samples} samples/prompt` : "greedy eval"}${st.config.max_prompts ? ` on ${st.config.max_prompts} prompts` : ""}`);
}

/** One timer tick of a stage: appends a metrics row and updates summary; returns true when finished. */
function stageTick(st: MockStage): boolean {
  const log = stageLogger(st);
  if (st.type === "train") {
    const row = trainTick(st, st.config, log);
    st.summary = Object.fromEntries(Object.entries(row).filter(([k, v]) => typeof v === "number" && !["step", "time"].includes(k)));
    return st.step >= st.length;
  }
  if (st.type === "rejection_sample") {
    const step = st.step + 1;
    const prompts = 16;
    const numSamples = Number(st.config.num_samples ?? 4);
    const sampled = prompts * numSamples;
    const rate = Math.min(0.9, 0.08 + 0.05 * step + (Math.random() - 0.5) * 0.04);
    const accepted = Math.min(prompts * Number(st.config.max_per_prompt ?? 1), Math.round(sampled * rate));
    const row: Dict = {
      step,
      time: now(),
      phase: "rejection",
      "rs/prompts": prompts,
      "rs/sampled": sampled,
      "rs/accepted": accepted,
      "rs/accept_rate": accepted / sampled,
      "reward/mean": rate * 0.9 + (Math.random() - 0.5) * 0.05,
      "reward/std": 0.3,
      "tokens/response_mean": 70 + Math.random() * 20,
      "time/rollout": 3 + Math.random(),
    };
    st.metrics.push(row);
    st.step = step;
    st.trajectories.push(...fakeTrajectories(step, rate, 2, "rejection", st.run));
    const prev = st.summary;
    const tPrompts = Number(prev.prompts ?? 0) + prompts;
    const tSampled = Number(prev.sampled ?? 0) + sampled;
    const tAccepted = Number(prev.accepted ?? 0) + accepted;
    st.summary = { prompts: tPrompts, sampled: tSampled, accepted: tAccepted, accept_rate: tAccepted / tSampled, mixed: 0, rows: tAccepted, dataset: `${st.run_dir}/dataset.jsonl`, "reward/mean": row["reward/mean"] };
    log(`chunk ${step}/${st.length}: sampled=${sampled} accepted=${accepted} (${(100 * (accepted / sampled)).toFixed(1)}%)`);
    if (step >= st.length) log(`wrote ${tAccepted} rows to ${st.run_dir}/dataset.jsonl`);
    return step >= st.length;
  }
  const numSamples = Math.max(1, Number(st.config.num_samples ?? 1));
  const row = evalRow(1, 0.9, st.run, Number(st.config.max_prompts ?? 100), numSamples);
  st.metrics.push(row);
  st.step = 1;
  st.trajectories.push(...fakeTrajectories(1, row["eval/success_rate"] as number, 4, "eval", st.run));
  st.summary = Object.fromEntries(Object.entries(row).filter(([k]) => !["step", "time", "phase"].includes(k)));
  log(`eval/reward_mean=${(row["eval/reward_mean"] as number).toFixed(3)} eval/success_rate=${(row["eval/success_rate"] as number).toFixed(3)} on ${row["eval/num_samples"]} samples`);
  return true;
}

function endStage(job: MockJob, st: MockStage) {
  st.state = "succeeded";
  st.finished_at = now();
  stageLogger(st)(`[succeeded] stage finished`);
  pipelineLog(job, `[stage ${st.index + 1}/${job.stages.length}] ${st.name} succeeded`);
  job.step = job.stages.filter((s) => s.state === "succeeded").length;
}

function startFakePipeline(job: MockJob) {
  job.state = "running";
  job.started_at = now();
  job.pid = 41_000 + Math.floor(Math.random() * 1000);
  pipelineLog(job, `[running] pipeline ${job.name}: ${job.stages.length} stages (mock)`);
  let current = job.stages.find((s) => s.state !== "succeeded") ?? null;
  if (!current) {
    finishJob(job, "succeeded", "all stages finished");
    return;
  }
  beginStage(job, current);
  job.timer = setInterval(() => {
    if (job.state !== "running" || !current) return;
    const done = stageTick(current);
    job.updated_at = now();
    if (done) {
      endStage(job, current);
      const next = job.stages[current.index + 1];
      if (!next) {
        finishJob(job, "succeeded", `all ${job.stages.length} stages finished`);
        return;
      }
      current = next;
      beginStage(job, current);
    }
  }, 700);
}

function finishJob(job: MockJob, state: RunState, message: string) {
  clearInterval(job.timer ?? undefined);
  job.timer = null;
  job.state = state;
  job.message = message;
  job.finished_at = now();
  job.updated_at = now();
  for (const st of job.stages) {
    if (st.state === "running") {
      st.state = state;
      st.finished_at = now();
      st.logs.push(`${stamp()} [${state}] ${message}`);
    }
  }
  job.logs.push(`${stamp()} [${state}] ${message}`);
}

function newJob(config: Dict, state: RunState = "pending"): MockJob {
  const id = Math.random().toString(16).slice(2, 10);
  const modelName = String((config.model as Dict | undefined)?.name_or_path ?? "model").split("/").pop();
  const name = (config.run_name as string | null | undefined) ?? `${modelName}-${(config.algorithm as Dict | undefined)?.name ?? "grpo"}-${Date.now()}`;
  const job: MockJob = {
    id,
    kind: "run",
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
    stages: [],
    timer: null,
    ...emptyRunData(),
  };
  jobs.set(id, job);
  return job;
}

function newPipelineJob(pipeline: Dict): MockJob {
  const base = (pipeline.base as Dict | undefined) ?? {};
  const modelName = String((base.model as Dict | undefined)?.name_or_path ?? "model").split("/").pop();
  const name = (pipeline.name as string | null | undefined) ?? `${modelName}-pipeline-${Date.now()}`;
  const job = newJob({ ...pipeline, name });
  job.kind = "pipeline";
  job.name = name;
  job.run_dir = `${HOME}/.monorl/runs/${name}`;
  job.stages = buildStages(job);
  return job;
}

function publicJob(job: MockJob) {
  const { metrics: _m, logs: _l, trajectories: _t, checkpoints: _c, timer: _timer, stages: _s, ...rest } = job;
  return rest;
}

function stageStatus(st: MockStage) {
  const { metrics: _m, logs: _l, trajectories: _t, checkpoints: _c, length: _len, config: _cfg, ...rest } = st;
  return rest;
}

/** GET /api/jobs/{id}/stages, and the data of each `stages` event of its stream. */
function stagesPayload(job: MockJob) {
  const last = [...job.stages].reverse().find((s) => s.type === "train" && s.state === "succeeded");
  return { job_id: job.id, kind: job.kind, name: job.name, state: job.state, final_model_path: last ? `${last.run_dir}/final` : null, error: job.error, stages: job.stages.map(stageStatus) };
}

function statusJson(job: MockJob) {
  if (job.state === "pending") return null;
  return { state: job.state, message: job.message, step: job.step, started_at: job.started_at, updated_at: job.updated_at, pid: job.pid };
}

// seed: one finished run from before transcripts existed, one finished tool-calling agent run, one
// failed run, one finished SFT run, one finished + one running pipeline
{
  const done = newJob({ ...RECIPES[0].config, run_name: "gsm8k-grpo-smol-seed", trainer: { ...(RECIPES[0].config.trainer as Dict), max_steps: 12, save_every: 6 } });
  startFakeJob(done);
  for (let i = 0; i < 12; i++) {
    // fast-forward the timer deterministically
    const row = { step: i + 1, time: now() - (12 - i) * 60, phase: "train", "reward/mean": 0.2 + i * 0.05, "reward/std": 0.4 - i * 0.02, "loss/policy": -0.01 * i, "loss/kl": 0.001 * i, "tokens/response_mean": 50 + i * 3, lr: 1e-5 };
    done.metrics.push(row);
    done.trajectories.push(...legacyTrajectories(i + 1, row["reward/mean"], 2));
    done.step = i + 1;
  }
  done.checkpoints.push({ step: 6, path: `${done.run_dir}/checkpoints/step-6` }, { step: 12, path: `${done.run_dir}/checkpoints/step-12` });
  finishJob(done, "succeeded", "finished 12 steps");

  const agentRecipe = RECIPES.find((r) => r.name === "tool-calling-grpo")!;
  const agent = newJob({ ...agentRecipe.config, run_name: "tool-calling-grpo-seed", trainer: { ...(agentRecipe.config.trainer as Dict), max_steps: 10, save_every: 5, eval_every: 5 } });
  startFakeJob(agent);
  while (agent.step < 10) trainTick(agent, agent.config, (l) => agent.logs.push(`${stamp()} ${l}`));
  finishJob(agent, "succeeded", "finished 10 steps");

  const failed = newJob({ ...RECIPES[1].config, run_name: "dapo-qwen-vllm-oom" });
  failed.state = "failed";
  failed.started_at = now() - 3600;
  failed.finished_at = now() - 3500;
  failed.error = "RuntimeError: CUDA out of memory";
  failed.message = "RuntimeError: CUDA out of memory";
  failed.logs.push("2026-11-02 09:00:00 [running] starting", "2026-11-02 09:01:40 [failed] RuntimeError: CUDA out of memory");

  const sft = newJob({ ...RECIPES[3].config, run_name: "sft-chat-smol-seed", trainer: { ...(RECIPES[3].config.trainer as Dict), max_steps: 15, save_every: 5 } });
  startFakeJob(sft);
  while (sft.step < 15) trainTick(sft, sft.config, (l) => sft.logs.push(`${stamp()} ${l}`));
  finishJob(sft, "succeeded", "finished 15 steps");

  const pipelineRecipe = RECIPES.find((r) => r.kind === "pipeline")!;
  const finishedPipeline = newPipelineJob({ ...pipelineRecipe.config, name: "r1-loop-seed-finished" });
  startFakePipeline(finishedPipeline);
  // Fast-forward every stage synchronously.
  for (const st of finishedPipeline.stages) {
    if (st.state === "pending") beginStage(finishedPipeline, st);
    while (!stageTick(st)) {
      /* advance */
    }
    endStage(finishedPipeline, st);
  }
  finishJob(finishedPipeline, "succeeded", `all ${finishedPipeline.stages.length} stages finished`);

  const live = newPipelineJob({ ...pipelineRecipe.config, name: "r1-loop-live" });
  setTimeout(() => startFakePipeline(live), 1500);
}

// --- http ------------------------------------------------------------------------------------

const json = (data: unknown, status = 200) => Response.json(data, { status, headers: { "access-control-allow-origin": "*" } });

// SSE, polled like the real server: every stream re-reads its source on a timer, so a stream opened
// before its job or stage started simply stays quiet until rows appear.
const SSE_POLL_MS = 250;

function sse(poll: (send: (event: string, data: unknown) => void) => boolean): Response {
  const encoder = new TextEncoder();
  let timer: Timer | null = null;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const send = (event: string, data: unknown) => controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
      /** One poll; closes the stream and returns true once `poll` reports it finished. */
      const step = () => {
        if (!poll(send)) return false;
        if (timer) clearInterval(timer);
        controller.close();
        return true;
      };
      if (!step()) timer = setInterval(step, SSE_POLL_MS);
    },
    cancel() {
      if (timer) clearInterval(timer);
    },
  });
  return new Response(stream, { headers: { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive", "access-control-allow-origin": "*" } });
}

interface StreamEnd {
  state: RunState;
  error: string | null;
}

const isTerminal = (state: RunState) => state === "succeeded" || state === "failed" || state === "cancelled";
const jobEnd = (job: MockJob): StreamEnd | null => (isTerminal(job.state) ? { state: job.state, error: job.error } : null);
/** A stage is over when it is terminal or its job is (a cancelled pipeline leaves later stages pending). */
const stageEnd = (job: MockJob, st: MockStage): StreamEnd | null => (isTerminal(st.state) ? { state: st.state, error: st.error } : jobEnd(job));

const TAIL_EVENT = { metrics: "metrics", logs: "log", trajectories: "trajectory" } as const;
type TailKind = keyof typeof TAIL_EVENT;

type TailCursors = Partial<Record<TailKind, number>>;

/**
 * Tail of a job's or stage's run data for the kinds in `cursors`: one event per row / line from
 * each cursor, then `end` `{state, error, next}` with `next = endNext(cursors)`.
 */
function sseTail(run: RunData, cursors: TailCursors, ended: () => StreamEnd | null, endNext: (cursors: TailCursors) => unknown): Response {
  const kinds = Object.keys(cursors) as TailKind[];
  return sse((send) => {
    const end = ended(); // before draining, so nothing appended in between is lost
    for (const kind of kinds) {
      const items: unknown[] = run[kind];
      let cursor = cursors[kind] ?? 0;
      for (; cursor < items.length; cursor++) send(TAIL_EVENT[kind], items[cursor]);
      cursors[kind] = cursor;
    }
    if (!end) return false;
    send("end", { ...end, next: endNext(cursors) });
    return true;
  });
}

/** `<base>/events`: metrics, log and trajectories of one run dir on one connection, each from its own cursor. */
function sseEvents(run: RunData, params: URLSearchParams, ended: () => StreamEnd | null): Response {
  const cursor = (kind: TailKind) => Number(params.get(kind) ?? 0);
  return sseTail(run, { metrics: cursor("metrics"), logs: cursor("logs"), trajectories: cursor("trajectories") }, ended, (cursors) => cursors);
}

/** `stages/stream`: the stages payload now and whenever it changes, then `end`. */
function sseStages(job: MockJob): Response {
  let sent = "";
  return sse((send) => {
    const end = jobEnd(job);
    const payload = stagesPayload(job);
    const text = JSON.stringify(payload);
    if (text !== sent) {
      send("stages", payload);
      sent = text;
    }
    if (!end) return false;
    send("end", { ...end, next: null });
    return true;
  });
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
      if (path === "/api/config/pipeline_schema") return json(PIPELINE_SCHEMA);
      if (path === "/api/config/validate" && method === "POST") {
        const body = (await req.json()) as { config?: Dict; pipeline?: Dict };
        if (body.pipeline !== undefined) return json({ ...validatePipeline(body.pipeline), pipeline: body.pipeline });
        if (body.config === undefined) return json({ detail: [{ loc: ["body", "config"], msg: "Field required", type: "missing" }] }, 422);
        return json(validate(body.config));
      }
      const img = path.match(/^\/api\/mock\/images\/(\d+)\.png$/);
      if (img) return new Response(mockPng(Number(img[1]) % 5), { headers: { "content-type": "image/png", "cache-control": "max-age=3600", "access-control-allow-origin": "*" } });
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
        const body = (await req.json()) as { config?: Dict; pipeline?: Dict };
        if (body.pipeline !== undefined) {
          const v = validatePipeline(body.pipeline);
          if (!v.ok) return json({ detail: v.errors }, 422);
          const job = newPipelineJob(body.pipeline);
          setTimeout(() => startFakePipeline(job), 800);
          return json(publicJob(job));
        }
        if (body.config === undefined) return json({ detail: [{ loc: ["body"], msg: "expected `config` or `pipeline`", type: "missing" }] }, 422);
        const v = validate(body.config);
        if (!v.ok) return json({ detail: v.errors }, 422);
        const job = newJob(body.config);
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
        if (sub === "stages") return json(stagesPayload(job));
        if (sub === "stages/stream") return sseStages(job);
        const stageEvents = sub.match(/^stages\/(\d+)\/events$/);
        if (stageEvents) {
          const st = job.stages[Number(stageEvents[1])];
          if (!st) return json({ detail: "stage not found" }, 404);
          return sseEvents(st, url.searchParams, () => stageEnd(job, st));
        }
        const sm = sub.match(/^stages\/(\d+)\/(metrics|logs|trajectories|checkpoints)$/);
        if (sm) {
          const st = job.stages[Number(sm[1])];
          if (!st) return json({ detail: "stage not found" }, 404);
          if (sm[2] === "metrics") return json({ rows: st.metrics.slice(since), next: st.metrics.length });
          if (sm[2] === "logs") {
            const tail = Number(url.searchParams.get("tail") ?? 200);
            return json({ lines: st.logs.slice(-tail), next: st.logs.length });
          }
          if (sm[2] === "trajectories") return json({ rows: st.trajectories.slice(since), next: st.trajectories.length });
          return json({ checkpoints: st.checkpoints });
        }
        if (sub === "cancel" && method === "POST") {
          if (job.state === "running" || job.state === "pending") finishJob(job, "cancelled", "cancelled by user");
          return json(publicJob(job));
        }
        if (sub === "events") return sseEvents(job, url.searchParams, () => jobEnd(job));
        const single = sub.match(/^(metrics|logs)\/stream$/);
        if (single) {
          const kind = single[1] as TailKind;
          return sseTail(job, { [kind]: since }, () => jobEnd(job), (cursors) => cursors[kind]);
        }
        if (sub === "metrics") return json({ rows: job.metrics.slice(since), next: job.metrics.length });
        if (sub === "logs") {
          const tail = Number(url.searchParams.get("tail") ?? 200);
          return json({ lines: job.logs.slice(-tail), next: job.logs.length });
        }
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
