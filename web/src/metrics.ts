// Default chart sets per run kind (metrics.jsonl keys written by the orchestrator / pipeline stages).
export const RL_METRICS = ["reward/mean", "reward/success_rate", "reward/std", "episode/turns_mean", "loss/policy", "loss/kl", "tokens/response_mean", "lr"];
export const SFT_METRICS = ["loss/sft", "sft/ppl", "eval/loss", "sft/tokens", "lr", "time/train"];
export const REJECTION_METRICS = ["rs/accept_rate", "rs/accepted", "rs/sampled", "reward/mean", "tokens/response_mean", "time/rollout"];
export const EVAL_METRICS = ["eval/reward_mean", "eval/success_rate", "eval/reward_std", "eval/turns_mean", "eval/num_samples"];
