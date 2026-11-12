# Off-policy corrections

With `rollout.async_steps_ahead > 0` a batch is sampled by an older policy than the one being
updated (`policy/staleness` counts the weight syncs in between, per batch), and an engine may also
differ numerically from the trainer (precision, kernels). `algorithm.off_policy` configures how the
policy loss accounts for that. Everything is off by default: a default config computes exactly the
loss it computed before this section existed.

## Three policies

| | Symbol | Where it comes from |
|---|---|---|
| behaviour policy | `mu` | the engine's log-probs of the sampled tokens (`Completion.logprobs`) |
| proximal policy | `pi_prox` | `TrainBatch.old_logprobs`: the anchor of the clipped ratio |
| current policy | `pi_theta` | the trainer's forward pass in `train_step` |

`off_policy.anchor` chooses `pi_prox`:

- `behavior`: `pi_prox = mu`, the engine's log-probs (verl "bypass mode"). One policy pair; the
  ratio `pi_theta / mu` is clipped.
- `trainer`: `pi_prox` is the trainer's policy when the update on the batch starts, computed once
  per batch with `Trainer.compute_logprobs` and kept for every epoch of `algorithm.num_epochs`
  (decoupled PPO). The engine's log-probs then travel next to it as
  `TrainBatch.behavior_logprobs`.
- `auto` (default): `trainer` when `importance_weights` or `ratio_mask` is set; otherwise the
  behaviour before this section existed, i.e. the engine's log-probs when they are comparable with
  the trainer's and the trainer's otherwise (`tokens/old_logprobs_recomputed`).

## Decoupled PPO with truncated importance weights and a ratio mask

With `rho_t = pi_prox(a_t) / mu(a_t)` and `r_t = pi_theta(a_t) / pi_prox(a_t)`, the per-token policy
loss is

    L_t = - w_t * keep_t * min(r_t A_t, clip(r_t, 1 - eps, 1 + eps_high) A_t)

This is the decoupled objective of Hilton et al. as used by AReaL (Eq. 5) and verl's "decoupled
mode": the clip is anchored on a recent policy, the importance weight corrects for the policy that
actually sampled. `w` and `keep` are constants of the batch (no gradient).

| Option | Values | Effect |
|---|---|---|
| `importance_weights` | `none` \| `token` \| `sequence` | `w_t = min(rho_t, cap)` per token, or `w = min(prod_t rho_t, cap)` for all tokens of a training sequence (verl `rollout_is`) |
| `importance_cap` | float >= 1 (default 2.0) or null | the cap; null leaves the weights untruncated |
| `ratio_mask` | `none` \| `token` \| `sequence` \| `sequence_mean` | `keep = 0` where the token ratio, the product of a sequence's ratios, or their geometric mean is outside the band |
| `ratio_mask_low`, `ratio_mask_high` | default 0.5 and 5.0 | the accepted band (must contain 1) |

A training sequence is a row of the `TrainBatch`; an episode whose turns do not continue each other
token for token yields several rows and each is weighed and masked on its own. Log-ratios are
clamped to `[-20, 20]` before exponentiation (verl's `SAFETY_BOUND`).

Masked tokens contribute zero and stay in the denominator of the loss mean, as in IcePop, AReaL's
rejection mask and TRL/NeMo-RL masked importance sampling. verl's rejection sampling instead removes
them from the response mask (shrinking the denominator and dropping them from the KL and entropy
terms too). Keeping the denominator makes the correction a per-token coefficient, which is why it
composes with all three `loss_agg` modes, with `TrainBatch.loss_weights` and with micro-batching:
the reference trainer's micro-batched gradient equals the full-batch gradient.

Reference bands from the sources: `0.5 .. 5` for token masks (IcePop; prime-rl uses `0.2 .. 5`),
about `0.999 .. 1.002` for `sequence_mean` (verl/NeMo-RL geometric-mean rejection).

## Score centering

`off_policy.score_centering` implements Marek & Ryabinin's correction. A policy-gradient estimator
that weighs the sampled token's score `s_y = grad log p(y)` by `f(p_y / q_y)` (`p` the trainer, `q`
the sampler) has a non-zero expected update even under a constant reward when `p != q`; score
centering subtracts that expectation, `sum_v q_v f(p_v / q_v) s_v`. The engine records its `K`
most likely next tokens `H` at every sampled position (`score_centering_top_k`, default 128; the
paper uses 128 and finds 32 sufficient); the rest of the sampler's distribution is modelled by the
trainer's tail scaled by `rho = (1 - sum_H q_v) / (1 - sum_H p_v)`. The loss per token is the
paper's Eq. 14:

    L_t = - A_t ( sg[f(p_y / q_y)] log p_y  -  sum_{v in H} sg[ q_v f(p_v / q_v) - alpha p_v ] log p_v ),
    alpha = rho f(1 / rho)

Both tail masses are floored at `1e-6`. monorl's policy term is the clipped surrogate
`-min(r A, clip(r) A)`, whose gradient is `-A f(r) s_y` with `f(r) = r` where the clip is inactive
and `0` where it binds (`r > 1 + eps_high` for `A > 0`, `r < 1 - eps` for `A < 0`); that `f` is used
for the head and the tail, so the centering term composes with the clipped ratio as section A.2 of
the paper prescribes for any weighting function. On-policy (`p = q`) every coefficient is zero.

Requirements, all checked when the config is loaded or the loop starts:

- the behaviour anchor (`anchor: behavior` or `auto`): the weights are functions of
  `pi_theta / mu`. It cannot be combined with `anchor: trainer`, `importance_weights` or
  `ratio_mask`; in the two-policy form the clipped ratio is the importance weight.
- untruncated sampling at `rollout.temperature > 0` (no `top_k`, `top_p`, `min_p`,
  `repetition_penalty`): the recorded head must be the distribution the token was drawn from.
- an engine that returns the top-k of the sampling distribution for every generated token
  (`SamplingParams.top_logprobs`, `Completion.top_logprobs`):

| Engine | Source of the top-k |
|---|---|
| `hf` | `topk` of the temperature-scaled softmax of every step's logits |
| `vllm` (colocated) | `SamplingParams(logprobs=K)` with `logprobs_mode=processed_logprobs`; the engine is started with `max_logprobs >= K` |
| `vllm` (`rollout.extra.mode: server`) | `/v1/completions` `logprobs=K` + `return_tokens_as_token_ids`; `/v1/chat/completions` (image prompts) `top_logprobs=K+1`. Start the server with `--max-logprobs` >= `K + 1` and `--logprobs-mode processed_logprobs` |
| `sglang` (colocated and server) | `top_logprobs_num=K` -> `meta_info.output_top_logprobs` |

vLLM returns the sampled token together with the top-k; the adapter keeps the `K` most likely
entries so the head does not depend on the sampled token. The top-k is never written to
`trajectories.jsonl`.

## Diagnostics

Logged per step when a separate proximal policy and usable engine log-probs exist (`anchor: trainer`
or a correction), over the batch's response tokens, before the first epoch:

| Metric | Meaning |
|---|---|
| `offpolicy/weight_mean`, `offpolicy/weight_max` | applied coefficients `w * keep` (masked tokens count as 0) |
| `offpolicy/truncated_frac` | share of tokens whose weight was cut to the cap |
| `offpolicy/masked_frac` | share of tokens dropped by the ratio mask |
| `offpolicy/ess` | effective sample size `(sum w)^2 / (n sum w^2)` as a fraction of the tokens |
| `offpolicy/kl` | `mean(log mu - log pi_prox)`, the direct estimate of `KL(mu || pi_prox)` (verl `rollout_corr/kl`) |
| `offpolicy/kl_k3` | `mean(rho - 1 - log rho)`, its non-negative k3 estimate (verl `rollout_corr/k3_kl`) |
| `offpolicy/skipped` | 1 when the batch had to train without the configured correction (below) |
| `offpolicy/sc_head_mass` | score centering: mean sampler mass covered by the recorded top-k |
| `offpolicy/sc_coef_l1`, `offpolicy/sc_tail_ratio` | score centering: mean `sum_v |c_v|` of the centering coefficients (0 on-policy) and mean tail-mass ratio `rho` |

With `anchor: trainer` and no correction the loss is unchanged and only the diagnostics are added.
With the behaviour anchor `policy/approx_kl` already is the k3 estimate of `KL(mu || pi_theta)`.

## When a correction cannot be applied

- The engine's log-probs are not comparable with the trainer's (a non-`hf` engine sampling through
  `top_k`/`top_p`/`min_p`/`repetition_penalty`, or `rollout.extra.trust_engine_logprobs: false`) and
  `importance_weights`, `ratio_mask`, `score_centering` or `anchor: behavior` is set: `ConfigError`
  when the loop starts. `anchor: trainer` alone still runs (it needs no behaviour log-probs).
- A batch arrives without engine log-probs, or without top-k for score centering: that batch trains
  uncorrected on the trainer anchor and reports `offpolicy/skipped = 1`.
- Custom loss functions: the reference trainer passes `behavior_logprobs`, `top_logprobs` and
  `behavior_top_logprobs` only for batches that carry them, so a loss with the old signature keeps
  working with default configs.

## Pipeline backends

The options are compiled to the backend's own mechanism; a combination without an exact counterpart
raises `ConfigError` naming the option. `score_centering` is rejected by all three.

| Option | verl 0.9.1 `algorithm.rollout_correction` | NeMo-RL 0.7.0 `loss_fn` | TRL 1.17 `GRPOConfig` (vLLM rollouts only) |
|---|---|---|---|
| default | untouched (decoupled, metrics only) | `use_importance_sampling_correction` = async rollouts | TRL defaults |
| `anchor: behavior` | `bypass_mode: true` | error | error |
| `anchor: trainer` | `bypass_mode: false` | correction off (error with async rollouts) | `vllm_importance_sampling_correction: false` |
| `importance_weights` + cap | `rollout_is`, `rollout_is_threshold` | `tis` (+ `sequence_level_importance_ratios`) | `token_truncate` / `sequence_truncate`, `clip_max` |
| uncapped weights | error | correction without truncation | error |
| `ratio_mask` | `rollout_rs: token_k1 / seq_sum_k1 / seq_mean_k1`, threshold `1/high_1/low` (verl thresholds `-log rho`); rejected tokens leave the denominator | with weights at the same level: `icepop`; `sequence_mean` + uncapped token weights: `seq-mask-tis` | with weights at the same level: `token_mask` / `sequence_mask`, `clip_min`, `clip_max` |

## Not implemented

- verl's k2/k3 and `seq_max` rejection statistics and batch-normalised importance weights.
- AReaL's log-linear approximation of the proximal log-probs (it needs per-token policy versions,
  which monorl does not record) and prime-rl's per-token staleness masking; staleness here is per
  batch and bounded by `rollout.async_steps_ahead`.
- prime-rl's IPO trust region on absolute probability differences.
- Score centering with a separate proximal policy (it would need the proximal policy's
  probabilities of the head tokens) and with truncated sampling (prime-rl replays the sampler's
  support for that).

## Sources

- verl rollout correction: <https://verl.readthedocs.io/en/latest/algo/rollout_corr.html>, and at the
  pinned commit `docs/algo/rollout_corr.md`, `docs/algo/rollout_corr_math.md`,
  `verl/trainer/ppo/rollout_corr_helper.py`, `verl/trainer/ppo/core_algos.py`,
  `verl/trainer/config/algorithm.py`.
- Hilton, Cobbe, Schulman, "Batch size-invariance for policy optimization":
  <https://arxiv.org/abs/2110.00641>. Not read directly; the decoupled objective is taken from
  verl's `rollout_corr_math.md` section 1.3 and AReaL Eq. 5, which both cite it and agree.
- AReaL: <https://github.com/inclusionAI/AReaL> (`areal/utils/functional/functional.py`
  `ppo_actor_loss_fn`, `apply_rejection_sampling`; `docs/en/algorithms/async.md`), paper
  <https://arxiv.org/abs/2505.24298> section 5.2.
- prime-rl algorithm notes (staleness, IcePop, score centering in practice):
  <https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/algorithms.md>. IcePop
  (<https://arxiv.org/abs/2510.18855>) is used as stated there and as implemented by verl
  (`rollout_is_threshold: "low_high"`) and NeMo-RL (`icepop`); the paper itself was not read.
- Marek, Ryabinin, "Score Centering Stabilizes Off-policy Reinforcement Learning":
  <https://arxiv.org/abs/2609.20807> (sections 4, 4.1, appendix A.1-A.3).
- Pinned upstream code read for the mappings and the top-k capture: NeMo-RL
  `nemo_rl/algorithms/loss/loss_functions.py`; TRL `trl/trainer/grpo_trainer.py`,
  `grpo_config.py`; vLLM `vllm/v1/sample/sampler.py`, `vllm/logprobs.py`,
  `vllm/entrypoints/generate/base/serving.py`; SGLang `srt/layers/sampler.py`,
  `srt/layers/logprob_processor.py`, `srt/managers/tokenizer_manager.py`.
