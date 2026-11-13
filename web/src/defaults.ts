import type { MtpConfig, RunConfig } from "./types";

/** Mirrors `MtpConfig` in src/monorl/config/schema.py. */
export function defaultMtpConfig(): MtpConfig {
  return { enabled: false, rollout: false, num_speculative_tokens: null, train: false, loss_coef: 0.1, detach_trunk: true };
}

/** `model.mtp` as the config stores it: the full section, or `undefined` when it holds the defaults
 * (an absent section means the same to the server and keeps the YAML of other runs unchanged). */
export function normalizeMtp(mtp: unknown): MtpConfig | undefined {
  if (!isPlainObject(mtp)) return undefined;
  const defaults = defaultMtpConfig();
  const full = deepMerge(defaults, mtp);
  return (Object.keys(full) as (keyof MtpConfig)[]).every((k) => full[k] === defaults[k]) ? undefined : full;
}

/** `providers` as the config stores it: the module paths, trimmed and without empty entries, or
 * `undefined` for an empty list (the schema default), which keeps the key out of the YAML. */
export function normalizeProviders(providers: unknown): string[] | undefined {
  if (!Array.isArray(providers)) return undefined;
  const modules = providers.filter((p): p is string => typeof p === "string").map((p) => p.trim()).filter(Boolean);
  return modules.length ? modules : undefined;
}

/** Mirrors the pydantic defaults in src/monorl/config/schema.py (`providers: []` is stored as absent). */
export function defaultRunConfig(): RunConfig {
  return {
    run_name: null,
    seed: 42,
    model: {
      name_or_path: "",
      revision: null,
      tokenizer_name_or_path: null,
      trust_remote_code: false,
      dtype: "auto",
      attn_implementation: null,
      chat_template: null,
      lora: { enabled: false, r: 16, alpha: 32, dropout: 0.05, target_modules: "all-linear" },
    },
    data: {
      uri: "",
      split: "train",
      hf_config_name: null,
      format: "auto",
      messages_field: null,
      prompt_field: null,
      answer_field: null,
      image_field: null,
      response_field: null,
      tools_field: null,
      system_prompt: null,
      prompt_template: null,
      max_prompt_tokens: 1024,
      limit: null,
      shuffle: true,
      streaming: false,
      eval_uri: null,
      eval_split: "test",
      extra: {},
    },
    algorithm: {
      name: "grpo",
      sft_label_smoothing: 0.0,
      sft_train_on_prompt: false,
      sft_turns: "last",
      kl_coef: 0.0,
      kl_estimator: "k3",
      clip_eps: 0.2,
      clip_eps_high: null,
      advantage_norm: "group",
      advantage_std_norm: true,
      loss_agg: "token",
      entropy_coef: 0.0,
      num_epochs: 1,
      dynamic_sampling: false,
      overlong_penalty: 0.0,
      credit_assignment: "episode",
      gamma: 1.0,
      lam: 1.0,
      value_coef: 0.5,
      value_clip: 0.5,
      off_policy: {
        anchor: "auto",
        importance_weights: "none",
        importance_cap: 2.0,
        ratio_mask: "none",
        ratio_mask_low: 0.5,
        ratio_mask_high: 5.0,
        score_centering: false,
        score_centering_top_k: 128,
      },
    },
    rollout: {
      backend: "hf",
      group_size: 4,
      max_new_tokens: 256,
      temperature: 1.0,
      top_p: 1.0,
      top_k: -1,
      stop: [],
      tensor_parallel_size: 1,
      data_parallel_size: 1,
      gpu_memory_utilization: 0.6,
      max_model_len: null,
      max_context_tokens: null,
      enable_sleep_mode: true,
      async_steps_ahead: 0,
      turn_rendering: "tokens",
      device: null,
      extra: {},
    },
    trainer: {
      backend: "reference",
      learning_rate: 1e-6,
      prompts_per_step: 8,
      micro_batch_size: 4,
      max_steps: 100,
      warmup_steps: 0,
      weight_decay: 0.0,
      max_grad_norm: 1.0,
      precision: "bf16",
      gradient_checkpointing: false,
      optimizer: "adamw",
      scheduler: "constant",
      save_every: 50,
      eval_every: null,
      max_seq_len: 2048,
      device: null,
      extra: {},
    },
    rewards: [{ type: "exact_match", weight: 1.0, params: {} }],
    env: { type: "single_turn", max_turns: 1, retries: 0, max_failed_frac: 0.0, params: {} },
    cluster: {
      mode: "local",
      ray_address: null,
      num_nodes: 1,
      gpus_per_node: 0,
      placement: "auto",
      rollout_gpus: null,
      trainer_gpus: null,
      weight_sync: "auto",
      extra: {},
    },
    logging: {
      run_root: "~/.monorl/runs",
      log_every: 1,
      save_trajectories: true,
      trajectories_per_step: 8,
      save_transcripts: true,
      success_threshold: 1.0,
      wandb_project: null,
      tensorboard: false,
    },
  };
}

export function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** Deep-merge `patch` over `base`; arrays and scalars in `patch` replace. A `null` where `base`
 * holds a non-null default (an object section, array or required string — e.g. `model:` or
 * `rewards:` left empty while editing YAML) keeps the default: every nullable field of the schema
 * already defaults to `null`, so a null patch over a non-null default is a partial document. */
export function deepMerge<T>(base: T, patch: unknown): T {
  if (!isPlainObject(base) || !isPlainObject(patch)) return (patch === undefined ? base : patch) as T;
  const out: Record<string, unknown> = { ...base };
  for (const [k, v] of Object.entries(patch)) {
    const cur = out[k];
    if (isPlainObject(cur)) out[k] = isPlainObject(v) ? deepMerge(cur, v) : cur;
    else if (v === null && cur !== null && cur !== undefined) out[k] = cur;
    else out[k] = v;
  }
  return out as T;
}

/** Fill a partial config (e.g. a recipe) with schema defaults. */
export function fromPartial(partial: unknown): RunConfig {
  const config = deepMerge(defaultRunConfig(), partial);
  const { mtp: _mtp, ...model } = config.model;
  const mtp = normalizeMtp(_mtp);
  const { providers: _providers, ...rest } = config;
  const providers = normalizeProviders(_providers);
  return { ...rest, ...(providers ? { providers } : {}), model: mtp ? { ...model, mtp } : model };
}

export interface Preset {
  name: string;
  description: string;
  patch: Partial<Record<keyof RunConfig, unknown>>;
}

export const PRESETS: Preset[] = [
  {
    name: "CPU smoke",
    description: "Tiny settings that finish on a laptop CPU in minutes. Use with a ~100M model.",
    patch: {
      rollout: { backend: "hf", group_size: 2, max_new_tokens: 32, device: "cpu" },
      trainer: {
        backend: "reference",
        learning_rate: 1e-5,
        prompts_per_step: 2,
        micro_batch_size: 2,
        max_steps: 5,
        precision: "fp32",
        save_every: 5,
        max_seq_len: 512,
        device: "cpu",
      },
      cluster: { mode: "local", gpus_per_node: 0, placement: "auto", weight_sync: "inprocess" },
    },
  },
  {
    name: "Single GPU",
    description: "Colocated rollout + training on one GPU; vLLM sleeps while the trainer runs.",
    patch: {
      rollout: { backend: "vllm", group_size: 8, max_new_tokens: 512, gpu_memory_utilization: 0.5, enable_sleep_mode: true },
      trainer: { backend: "reference", learning_rate: 1e-6, prompts_per_step: 16, micro_batch_size: 4, max_steps: 200, precision: "bf16" },
      cluster: { mode: "local", gpus_per_node: 1, placement: "colocated", weight_sync: "auto" },
    },
  },
  {
    name: "Multi-GPU",
    description: "Ray cluster with disaggregated placement: dedicated inference GPUs, broadcast weight sync.",
    patch: {
      rollout: { backend: "vllm", group_size: 16, max_new_tokens: 1024, tensor_parallel_size: 2, data_parallel_size: 1, async_steps_ahead: 1 },
      trainer: { backend: "reference", learning_rate: 1e-6, prompts_per_step: 64, micro_batch_size: 8, max_steps: 1000, precision: "bf16", gradient_checkpointing: true },
      cluster: {
        mode: "ray",
        ray_address: "auto",
        num_nodes: 1,
        gpus_per_node: 8,
        placement: "disaggregated",
        rollout_gpus: 2,
        trainer_gpus: 6,
        weight_sync: "broadcast",
      },
    },
  },
];

export const REWARD_TYPES_FALLBACK = ["exact_match", "format", "regex", "math_verify", "code_exec", "length", "llm_judge", "reward_model", "composite"];
