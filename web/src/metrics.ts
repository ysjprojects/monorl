// Default chart sets per run kind (metrics.jsonl keys written by the orchestrator / pipeline stages).
export const RL_METRICS = ["reward/mean", "reward/std", "loss/policy", "loss/kl", "tokens/response_mean", "lr"];
export const SFT_METRICS = ["loss/sft", "sft/ppl", "eval/loss", "sft/tokens", "lr", "time/train"];
export const REJECTION_METRICS = ["rs/accept_rate", "rs/accepted", "rs/sampled", "reward/mean", "tokens/response_mean", "time/rollout"];
export const EVAL_METRICS = ["eval/reward_mean", "eval/reward_std", "eval/num_samples"];
