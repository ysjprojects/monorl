# Config reference

Every field of a run config (`RunConfig`) and of a pipeline config (`PipelineConfig`), with its type
and default. This file is generated from the schema in `src/monorl/config/`; do not edit it, run
`python -m monorl.config.reference --write docs/CONFIG.md` instead.
`monorl recipes show NAME` prints complete example configs. A config is validated by
`RunConfig.from_dict` and by `POST /api/config/validate`;
rules that span several fields are enforced at validation and described in the docs linked under
each section.

Unknown keys are rejected. A key marked "required" has no default. Bounds on numbers are listed
with the type.

## Top-level fields

See: [RUNS.md](RUNS.md)

| Key | Type, default | Description |
| --- | --- | --- |
| `run_name` | string or null; default `null` | Defaults to '\<model>-\<algorithm>-\<timestamp>'. |
| `seed` | integer; default `42` | Seed of the run: Python and torch are seeded with it at the start and with seed + step at every step, and data sources and backends receive it. |

## `model`

See: [BACKENDS.md](BACKENDS.md), [MTP.md](MTP.md)

This section is required.

| Key | Type, default | Description |
| --- | --- | --- |
| `model.name_or_path` | string; required | Hugging Face model id or local path. |
| `model.revision` | string or null; default `null` | Hub revision (branch, tag or commit). |
| `model.tokenizer_name_or_path` | string or null; default `null` | Override tokenizer (defaults to the model). |
| `model.trust_remote_code` | boolean; default `false` | Let the model repository's own code run when the model is loaded. |
| `model.dtype` | `auto`, `float32`, `bfloat16`, `float16`; default `auto` | Parameter dtype; 'auto' picks bf16 on accelerators and fp32 on CPU. |
| `model.attn_implementation` | string or null; default `null` | e.g. 'flash_attention_2', 'sdpa', 'eager'. |
| `model.chat_template` | string or null; default `null` | Jinja chat template override. |
| `model.lora.enabled` | boolean; default `false` | Train LoRA adapters instead of full parameters. |
| `model.lora.r` | integer, >= 1; default `16` | LoRA rank. |
| `model.lora.alpha` | integer, >= 1; default `32` | LoRA alpha (scaling = alpha / r). |
| `model.lora.dropout` | number, >= 0.0, <= 1.0; default `0.05` | LoRA dropout. |
| `model.lora.target_modules` | list of string or `all-linear`; default `all-linear` | Module names to adapt, or 'all-linear'. |
| `model.mtp.enabled` | boolean; default `false` | Load the checkpoint's MTP (multi-token prediction) layers next to the policy and keep them in weight sync, checkpoints and the final export. Needs a model that ships MTP layers. |
| `model.mtp.rollout` | boolean; default `false` | Draft tokens with the MTP layers during rollouts (speculative decoding). The policy verifies every draft, so the sampled tokens and their logprobs are the policy's; only speed changes. |
| `model.mtp.num_speculative_tokens` | integer or null, >= 1; default `null` | Tokens drafted per decoding step. Default: one per MTP layer on the hf engine, 1 on vLLM, SGLang's own default. |
| `model.mtp.train` | boolean; default `false` | Train the MTP layers with an auxiliary loss (predict the tokens after the next one) on the response tokens of every batch, in RL and SFT, so drafts keep matching a policy that changes. |
| `model.mtp.loss_coef` | number, >= 0.0; default `0.1` | Weight of the MTP loss in the total loss. |
| `model.mtp.detach_trunk` | boolean; default `true` | true: the MTP loss updates the MTP layers only. false: it also flows into the policy's own parameters, as in pretraining. |

## `data`

See: [DATA.md](DATA.md)

This section is required.

| Key | Type, default | Description |
| --- | --- | --- |
| `data.uri` | string; required | Dataset location: 'hf://org/name' (Hub), 's3://bucket/prefix-or-file' (credentials from AWS_* env vars), or a local file/directory (.jsonl/.json/.parquet/.csv/.arrow). |
| `data.split` | string; default `train` | Split name for Hub datasets (ignored for files). |
| `data.hf_config_name` | string or null; default `null` | Hub dataset config/subset name. |
| `data.format` | `auto`, `chat`, `prompt_answer`, `text`; default `auto` | 'chat': a messages column; 'prompt_answer': prompt + reference answer columns; 'text': a single prompt column; 'auto' sniffs columns. |
| `data.messages_field` | string or null; default `null` | Column holding chat messages (format=chat). |
| `data.prompt_field` | string or null; default `null` | Column holding the prompt/question. |
| `data.answer_field` | string or null; default `null` | Column holding the reference answer (optional). |
| `data.image_field` | string or list of string or null; default `null` | Column(s) holding images (HF Image feature, file path/URI, bytes or base64) attached to the prompt as image content parts. Use '\<image>' in prompt_template to position them. |
| `data.response_field` | string or null; default `null` | Column holding the target response for SFT (defaults to answer_field / the trailing assistant turn). |
| `data.tools_field` | string or null; default `null` | Column holding the tool schemas (a list of OpenAI-style function schemas, or its JSON string) a chat row's prompts are rendered with. Default: a 'tools' column of a chat dataset when it holds one. |
| `data.system_prompt` | string or null; default `null` | System prompt prepended to every sample. |
| `data.prompt_template` | string or null; default `null` | Python format string applied to the prompt column, e.g. '{prompt}\nThink step by step.' |
| `data.max_prompt_tokens` | integer, >= 1; default `1024` | Samples with longer prompts are dropped. |
| `data.limit` | integer or null, >= 1; default `null` | Use only the first N samples (after shuffling). |
| `data.shuffle` | boolean; default `true` | Shuffle the rows with the run's seed (a streaming source shuffles within a buffer). |
| `data.streaming` | boolean; default `false` | Stream from Hub/S3 instead of materialising locally. |
| `data.eval_uri` | string or null; default `null` | Optional evaluation dataset (same format). |
| `data.eval_split` | string; default `test` | Split of eval_uri that is evaluated. |
| `data.extra` | mapping; default `{}` | Loader-specific options. Keys: see [DATA.md](DATA.md). |

## `algorithm`

See: [ALGORITHMS.md](ALGORITHMS.md), [CREDIT_ASSIGNMENT.md](CREDIT_ASSIGNMENT.md),
[OFF_POLICY.md](OFF_POLICY.md)

| Key | Type, default | Description |
| --- | --- | --- |
| `algorithm.name` | `grpo`, `dapo`, `rloo`, `reinforce`, `ppo`, `sft`; default `grpo` | Policy-gradient objective, or 'sft' for supervised fine-tuning on the dataset responses (no rollouts). |
| `algorithm.sft_label_smoothing` | number, >= 0.0, < 1.0; default `0.0` | Label smoothing for sft. |
| `algorithm.sft_train_on_prompt` | boolean; default `false` | sft: include prompt tokens in the loss (default: response only). |
| `algorithm.sft_turns` | `last`, `all`; default `last` | sft: assistant turns of a conversation that carry the loss. 'last': only the final one, earlier turns are context. 'all': every assistant turn (multi-turn and tool-using conversations), except those a row marks with 'weight': 0. |
| `algorithm.kl_coef` | number, >= 0.0; default `0.0` | Coefficient of the KL penalty to the reference policy (0 disables). |
| `algorithm.kl_estimator` | `k1`, `k2`, `k3`; default `k3` | KL estimator (Schulman's k1/k2/k3). |
| `algorithm.clip_eps` | number, > 0.0; default `0.2` | PPO clipping epsilon (lower bound). |
| `algorithm.clip_eps_high` | number or null, > 0.0; default `null` | Asymmetric upper clip (DAPO clip-higher). |
| `algorithm.advantage_norm` | `group`, `batch`, `none`; default `group` | Normalise rewards within the prompt group (GRPO), across the batch, or not at all. |
| `algorithm.advantage_std_norm` | boolean; default `true` | Divide by std in group normalisation (False = Dr.GRPO style). |
| `algorithm.loss_agg` | `token`, `sequence`, `rollout`; default `token` | Average the loss over all response tokens, per training sequence first, or per rollout first ('rollout': every episode counts equally however many sequences its turns form). |
| `algorithm.entropy_coef` | number, >= 0.0; default `0.0` | Weight of the entropy bonus: entropy_coef * entropy is subtracted from the loss. |
| `algorithm.num_epochs` | integer, >= 1; default `1` | Optimizer passes over each rollout batch. |
| `algorithm.dynamic_sampling` | boolean; default `false` | Drop groups whose rewards are all identical (DAPO). |
| `algorithm.overlong_penalty` | number, >= 0.0; default `0.0` | Reward penalty for truncated (length) completions. |
| `algorithm.credit_assignment` | `episode`, `turn`; default `episode` | Which rewards a response token is credited with. 'episode': the reward of its whole episode, the same advantage for every token. 'turn': the return from its own turn on (later per-turn rewards discounted by gamma; the reward-function score arrives at the final turn), compared with the same turn of the other episodes of its prompt group. See docs/CREDIT_ASSIGNMENT.md. |
| `algorithm.gamma` | number, >= 0.0, <= 1.0; default `1.0` | Discount for multi-turn per-turn rewards. |
| `algorithm.lam` | number, >= 0.0, <= 1.0; default `1.0` | GAE lambda (ppo only): 1 = returns minus the value baseline, 0 = one-step TD error. |
| `algorithm.value_coef` | number, >= 0.0; default `0.5` | Value loss weight (ppo only). |
| `algorithm.value_clip` | number or null, > 0.0; default `0.5` | ppo only: the value loss also penalises a value estimate clipped to this distance from the estimate the advantages were computed with (null = plain squared error). |
| `algorithm.off_policy.anchor` | `auto`, `behavior`, `trainer`; default `auto` | Proximal policy the clipped ratio is anchored on. 'behavior': the policy that sampled the tokens (the engine's log-probs). 'trainer': the trainer's policy when the update on the batch starts (decoupled PPO; one extra forward pass per batch). 'auto': 'trainer' when importance_weights or ratio_mask is set, otherwise the backend's default. |
| `algorithm.off_policy.importance_weights` | `none`, `token`, `sequence`; default `none` | Weigh the policy loss by the truncated importance ratio between the proximal and the behaviour policy, per token or per training sequence (product of its token ratios). |
| `algorithm.off_policy.importance_cap` | number or null, >= 1.0; default `2.0` | Upper bound of the importance weights (null = untruncated). |
| `algorithm.off_policy.ratio_mask` | `none`, `token`, `sequence`, `sequence_mean`; default `none` | Drop from the policy loss the tokens, or whole training sequences, whose proximal/behaviour importance ratio is outside [ratio_mask_low, ratio_mask_high]. 'sequence' tests the product of a sequence's token ratios, 'sequence_mean' their geometric mean. |
| `algorithm.off_policy.ratio_mask_low` | number, > 0.0, <= 1.0; default `0.5` | Lower bound of the accepted importance ratio. |
| `algorithm.off_policy.ratio_mask_high` | number, >= 1.0; default `5.0` | Upper bound of the accepted importance ratio. |
| `algorithm.off_policy.score_centering` | boolean; default `false` | Subtract the expected score under the sampling policy from the policy gradient (arXiv:2609.20807). Needs the engine's top-k next-token log-probs and the 'behavior' anchor. |
| `algorithm.off_policy.score_centering_top_k` | integer, >= 1; default `128` | Number of most likely next tokens the engine records per sampled token. |

## `rollout`

See: [BACKENDS.md](BACKENDS.md), [DISTRIBUTED.md](DISTRIBUTED.md)

| Key | Type, default | Description |
| --- | --- | --- |
| `rollout.backend` | string; default `hf` | Rollout engine: 'hf' (transformers), 'vllm', 'sglang'. |
| `rollout.group_size` | integer, >= 1; default `4` | Completions sampled per prompt (G in GRPO). |
| `rollout.max_new_tokens` | integer, >= 1; default `256` | Tokens a turn may generate at most. |
| `rollout.temperature` | number, >= 0.0; default `1.0` | Sampling temperature; 0 decodes greedily. |
| `rollout.top_p` | number, > 0.0, <= 1.0; default `1.0` | Nucleus sampling threshold; 1.0 turns it off. |
| `rollout.top_k` | integer; default `-1` | -1 disables top-k. |
| `rollout.stop` | list of string; default `[]` | Strings at which a turn's generation stops. |
| `rollout.tensor_parallel_size` | integer, >= 1; default `1` | GPUs one engine replica spans (tensor parallelism). |
| `rollout.data_parallel_size` | integer, >= 1; default `1` | Number of independent engine replicas. |
| `rollout.gpu_memory_utilization` | number, > 0.0, <= 1.0; default `0.6` | Share of a GPU's memory the engine may take (vLLM gpu_memory_utilization, SGLang mem_fraction_static). |
| `rollout.max_model_len` | integer or null; default `null` | Context length the engine is started with (vLLM max_model_len, SGLang context_length). Default: the model's own. |
| `rollout.max_context_tokens` | integer or null, >= 2; default `null` | Context budget of an episode in tokens: no prompt plus completion of a turn may exceed it. A turn's generation is capped to the room that is left, an episode without room for another turn ends, and an agent behind the gateway gets a context_length_exceeded error. Default: trainer.max_seq_len (and rollout.max_model_len when set), so that nothing an episode samples is cut off in training. |
| `rollout.enable_sleep_mode` | boolean; default `true` | Offload engine memory while training (colocated). |
| `rollout.async_steps_ahead` | integer, >= 0; default `0` | Generate up to N batches ahead of training (off-policy staleness bound). 0 = synchronous. |
| `rollout.turn_rendering` | `tokens`, `template`; default `tokens` | How a multi-turn prompt is built. 'tokens': the sampled tokens of earlier turns stay in the context and only new messages are rendered. 'template': the chat template re-renders the whole conversation every turn. |
| `rollout.device` | string or null; default `null` | Device for the 'hf' backend (cpu/mps/cuda); auto if unset. |
| `rollout.extra` | mapping; default `{}` | Engine-specific options passed through. Keys: see [BACKENDS.md](BACKENDS.md). |

## `trainer`

See: [RUNS.md](RUNS.md), [BACKENDS.md](BACKENDS.md)

| Key | Type, default | Description |
| --- | --- | --- |
| `trainer.backend` | string; default `reference` | Trainer: 'reference' (torch), 'unsloth', or a pipeline backend: 'trl', 'verl', 'nemo_rl'. |
| `trainer.learning_rate` | number, > 0.0; default `1.0e-06` | Learning rate (its peak under warmup or a schedule). |
| `trainer.prompts_per_step` | integer, >= 1; default `8` | Prompts per optimizer step (x group_size = sequences). |
| `trainer.micro_batch_size` | integer, >= 1; default `4` | Sequences per forward/backward pass. |
| `trainer.max_steps` | integer, >= 1; default `100` | Training steps of the run. |
| `trainer.warmup_steps` | integer, >= 0; default `0` | Steps over which the learning rate rises linearly from 0. |
| `trainer.weight_decay` | number, >= 0.0; default `0.0` | Weight decay of the optimizer. |
| `trainer.max_grad_norm` | number, > 0.0; default `1.0` | Norm the gradient is clipped to at every step. |
| `trainer.precision` | `fp32`, `bf16`, `fp16`; default `bf16` | Mixed-precision mode (fp32 forced on CPU). |
| `trainer.gradient_checkpointing` | boolean; default `false` | Recompute activations in the backward pass instead of storing them: less memory, more compute. |
| `trainer.optimizer` | `adamw`, `adam`, `sgd`; default `adamw` | Optimizer. |
| `trainer.scheduler` | `constant`, `linear`, `cosine`; default `constant` | Learning rate after the warmup: constant, or a linear or cosine decay to 0 at the last step. |
| `trainer.save_every` | integer, >= 1; default `50` | Checkpoint interval in steps. |
| `trainer.eval_every` | integer or null, >= 1; default `null` | Evaluate on data.eval_uri every this many steps. Null, or no eval_uri: no evaluation during the run. |
| `trainer.max_seq_len` | integer, >= 16; default `2048` | Prompt + response truncation for training. |
| `trainer.device` | string or null; default `null` | cpu/mps/cuda; auto if unset. |
| `trainer.extra` | mapping; default `{}` | Backend-specific options (e.g. verl/nemo overrides). Keys: see [BACKENDS.md](BACKENDS.md). |

## `rewards`

See: [REWARDS.md](REWARDS.md)

A list of entries (default `[]`); each entry has the keys below. Combined as a weighted sum.

| Key | Type, default | Description |
| --- | --- | --- |
| `rewards[].type` | string; required | Reward function name (see /api/system for the list). |
| `rewards[].weight` | number; default `1.0` | Multiplier when combining several rewards. |
| `rewards[].params` | mapping; default `{}` | Parameters of the reward function. Keys: see [REWARDS.md](REWARDS.md). |

## `env`

See: [ENVIRONMENTS.md](ENVIRONMENTS.md), [GATEWAY.md](GATEWAY.md), [SANDBOX.md](SANDBOX.md),
[ENV_SDK.md](ENV_SDK.md)

| Key | Type, default | Description |
| --- | --- | --- |
| `env.type` | string; default `single_turn` | Environment: 'single_turn', 'tool_calling', 'openenv', 'hermes', ... |
| `env.max_turns` | integer, >= 1; default `1` | Policy turns an episode may take at most. |
| `env.retries` | integer, >= 0; default `0` | Times an episode whose environment failed (a crashed agent, a failing step) is started again. |
| `env.max_failed_frac` | number, >= 0.0, <= 1.0; default `0.0` | Share of a rollout batch's episodes that may fail for good; they are left out of the batch. Above it the run fails. 0 (default): any failed episode fails the run. |
| `env.params` | mapping; default `{}` | Environment-specific parameters. Keys: see [ENVIRONMENTS.md](ENVIRONMENTS.md), [GATEWAY.md](GATEWAY.md), [SANDBOX.md](SANDBOX.md), [ENV_SDK.md](ENV_SDK.md). |

## `cluster`

See: [DISTRIBUTED.md](DISTRIBUTED.md)

| Key | Type, default | Description |
| --- | --- | --- |
| `cluster.mode` | `local`, `ray`; default `local` | Run in-process or on a Ray cluster. |
| `cluster.ray_address` | string or null; default `null` | Ray address ('auto' to join an existing cluster). |
| `cluster.num_nodes` | integer, >= 1; default `1` | Nodes of the cluster. |
| `cluster.gpus_per_node` | integer, >= 0; default `0` | GPUs on each node; 0 runs on CPU. |
| `cluster.placement` | `auto`, `colocated`, `disaggregated`; default `auto` | Share GPUs between rollout and training (colocated) or split them (disaggregated). |
| `cluster.rollout_gpus` | integer or null, >= 0; default `null` | GPUs for inference when disaggregated. |
| `cluster.trainer_gpus` | integer or null, >= 0; default `null` | GPUs for training when disaggregated. |
| `cluster.weight_sync` | string; default `auto` | Weight sync strategy: 'auto', 'inprocess', 'checkpoint', 'broadcast', 'cuda_ipc'. |
| `cluster.extra` | mapping; default `{}` | Options of the runtime and the weight sync (ray_init, sync_dir, ...). Keys: see [DISTRIBUTED.md](DISTRIBUTED.md). |

## `logging`

See: [RUNS.md](RUNS.md)

| Key | Type, default | Description |
| --- | --- | --- |
| `logging.run_root` | string; default `~/.monorl/runs` | Directory that holds run directories. |
| `logging.log_every` | integer, >= 1; default `1` | Write metrics every this many steps (the last step is always written). |
| `logging.save_trajectories` | boolean; default `true` | Write sampled rollouts to trajectories.jsonl. |
| `logging.trajectories_per_step` | integer, >= 0; default `8` | Rollouts saved per logged step at most; 0 saves none. |
| `logging.save_transcripts` | boolean; default `true` | Include the full conversation and per-turn details of every saved rollout. |
| `logging.success_threshold` | number; default `1.0` | Total reward at or above which an episode counts as solved (success-rate and pass@k metrics). |
| `logging.wandb_project` | string or null; default `null` | Weights & Biases project the metrics are also logged to (needs the wandb package). |
| `logging.tensorboard` | boolean; default `false` | Also write the metrics as TensorBoard events under \<run_dir>/tensorboard. |

## Pipelines

See: [PIPELINES.md](PIPELINES.md)

A document with both `base` and `stages` is a pipeline config. Each entry of `stages` is one
of the stage types below, selected by its `type` key.

| Key | Type, default | Description |
| --- | --- | --- |
| `name` | string or null; default `null` | Pipeline run name (defaults to '\<model>-pipeline-\<timestamp>'). |
| `base` | run config (the sections above); required | Base run configuration every stage starts from. |
| `stages` | list of (`train` stage or `rejection_sample` stage or `eval` stage), at least 1 item(s); required | The stages, run in order. |

### Stage `train`

| Key | Type, default | Description |
| --- | --- | --- |
| `stages[].type` | `train`; default `train` | Stage kind. |
| `stages[].name` | string; required | Stage name (used for the stage directory). |
| `stages[].overrides` | mapping; default `{}` | RunConfig fields merged over the pipeline base (e.g. {algorithm: {name: sft}, trainer: {max_steps: 50}}). |
| `stages[].init_from` | `previous`, `base` or string; default `previous` | Policy initialisation: 'previous' (last train stage's final export), 'base' (base model), or a path. |
| `stages[].data_from` | `previous`, `config`; default `config` | 'previous' uses the dataset written by the preceding rejection_sample stage as data.uri; an SFT stage then trains every policy turn of its conversations (algorithm.sft_turns: all). |

### Stage `rejection_sample`

| Key | Type, default | Description |
| --- | --- | --- |
| `stages[].type` | `rejection_sample`; default `rejection_sample` | Stage kind. |
| `stages[].name` | string; required | Stage name (used for the stage directory). |
| `stages[].data` | `data` section or null; default `null` | Prompts to sample from (defaults to the base data config). |
| `stages[].num_samples` | integer, >= 1; default `4` | Episodes sampled per prompt. |
| `stages[].min_reward` | number; default `1.0` | Keep episodes whose total reward is >= this value. |
| `stages[].max_per_prompt` | integer, >= 1; default `1` | Keep at most this many accepted episodes per prompt. |
| `stages[].max_response_tokens` | integer or null, >= 1; default `null` | Drop episodes in which a policy turn is longer than this many tokens. |
| `stages[].max_context_tokens` | integer or null, >= 1; default `null` | Drop episodes whose context (prompt plus completion of their longest turn) exceeds this many tokens, e.g. the trainer.max_seq_len of the SFT stage that follows. |
| `stages[].require_finish` | boolean; default `true` | Drop episodes in which a policy turn was cut off by the token limit or failed. |
| `stages[].dedupe` | boolean; default `true` | Drop episodes whose conversation repeats one already kept for the same prompt. |
| `stages[].mix` | list of string; default `[]` | Extra chat/prompt_answer dataset URIs appended to the output. |
| `stages[].rollout_overrides` | mapping; default `{}` | RolloutConfig overrides for sampling. |
| `stages[].reward_overrides` | list of mapping or null; default `null` | Replace the base rewards for scoring. |
| `stages[].output` | string or null; default `null` | Output jsonl path (default: \<stage dir>/dataset.jsonl). |
| `stages[].max_prompts` | integer or null, >= 1; default `null` | Sample from at most this many prompts. |

### Stage `eval`

| Key | Type, default | Description |
| --- | --- | --- |
| `stages[].type` | `eval`; default `eval` | Stage kind. |
| `stages[].name` | string; required | Stage name (used for the stage directory). |
| `stages[].data` | `data` section or null; default `null` | Evaluation dataset (defaults to base data.eval_uri or data). |
| `stages[].rollout_overrides` | mapping; default `{}` | RolloutConfig overrides for the evaluation rollouts. |
| `stages[].max_prompts` | integer or null, >= 1; default `null` | Evaluate at most this many prompts. |
| `stages[].num_samples` | integer, >= 1; default `1` | Completions per prompt. 1 evaluates greedily; more samples that many at the rollout temperature and reports eval/pass_at_k. |
