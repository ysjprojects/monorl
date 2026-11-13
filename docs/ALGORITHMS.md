# Algorithms

This document describes the `algorithm:` section of a run config: what each value of
`algorithm.name` computes in the native loop (`trainer.backend: reference`) and what every other
`algorithm.*` option does. It is for anyone choosing an objective or reading the loss and
advantage metrics of a run. The functions named here live in `src/monorl/algorithms/` and are
pure torch; the loop that calls them is `src/monorl/orchestrator/loop.py`.

## Shared setup

Every training step of the RL algorithms (all names except `sft`) does the following.

1. `PromptSampler.batch(step)` draws `trainer.prompts_per_step` prompts (default 8).
2. Every prompt is rolled out `rollout.group_size` times (default 4). The episodes of one prompt
   in one step share a group id (`s<step>-p<index>`) and form a group, so a step has
   `prompts_per_step * group_size` episodes.
3. `ComponentLoop._score` gives every episode one scalar reward `R`: the score of the reward
   functions ([REWARDS.md](REWARDS.md)), plus `sum_k gamma^k * r_k` over the environment's
   per-turn rewards `r_k`, minus `overlong_penalty` when the last turn was cut off.
4. The advantage `A` of an episode is computed from the rewards by the estimator that
   `algorithm.name` selects (`compute_advantages`). It carries no gradient, and every response
   token of the episode gets it (`build_train_batch`). Prompt tokens are not trained.
5. The trainer minimises `policy_loss`, the clipped surrogate, over the response tokens:

```text
ratio_t = exp(logp_theta(a_t) - logp_old(a_t))
high    = clip_eps_high if clip_eps_high is set, else clip_eps
pg_t    = -min(ratio_t * A_t, clip(ratio_t, 1 - clip_eps, 1 + high) * A_t)
loss    = agg(pg) + kl_coef * agg(kl) - entropy_coef * agg(entropy)
```

`agg` is the mean chosen by `loss_agg` (see [Options](#options)). `logp_theta` is the trainer's
`log_softmax(logits / rollout.temperature)` at the sampled token (a temperature of 0 counts as 1).

`logp_old` is the log-probability under the behaviour policy, the policy that sampled the tokens:
the per-token logprob the rollout engine recorded (`Completion.logprobs`, stored as
`TrainBatch.old_logprobs`). It stays fixed for all epochs of a batch. Before the first epoch the
loop replaces it with a no-gradient pass of the trainer (`Trainer.compute_logprobs`) and reports
`tokens/old_logprobs_recomputed: 1` in three cases: a completion has no recorded logprobs;
`rollout.backend` is not `hf` and sampling uses `top_k`, `top_p`, `rollout.extra.min_p` or
`rollout.extra.repetition_penalty`, or `rollout.extra.trust_engine_logprobs` is `false` (the
engine may report logprobs of the truncated distribution); or `algorithm.off_policy` anchors on
the trainer ([OFF_POLICY.md](OFF_POLICY.md)).

`clip_eps` defaults to `0.2` and `clip_eps_high` to `null`. `num_epochs` (default `1`) is the
number of `Trainer.train_step` calls on the same batch. One `train_step` is one optimizer step:
gradients are accumulated over micro-batches of `trainer.micro_batch_size` sequences and clipped
to `trainer.max_grad_norm`. The learning-rate schedule spans `trainer.max_steps * num_epochs`
optimizer steps, and the reported trainer metrics are means over the epochs. The ratio differs
from 1 only as far as `logp_old` differs from the trainer's current logprobs: after an earlier
epoch on the same batch, when older weights sampled the batch (`rollout.async_steps_ahead`), or
through numerical differences between engine and trainer.

## Algorithm names

`algorithm.name` defaults to `grpo`. In the native loop the name selects the advantage estimator
(and, for `ppo` and `sft`, a different batch or loop). It does not change the default of any other
field. `R_i` is the reward of episode `i`; `mean_g` and `std_g` are taken over its group.

### `grpo`

```text
A_i = (R_i - mean_g) / (std_g + 1e-6)
```

`compute_advantages` with `advantage_norm: group` (the default). `std_g` is the unbiased standard
deviation (divisor `n - 1`); a group whose rewards are all equal gets advantage 0 for every
member. `advantage_norm` and `advantage_std_norm` change the formula (see [Options](#options)).
Requires `rollout.group_size >= 2`.

```yaml
algorithm:
  name: grpo
  kl_coef: 0.001
```

### `dapo`

The same code path as `grpo`: same estimator, same loss, same `group_size >= 2` requirement. The
name switches nothing on. The DAPO ingredients are separate options that must be set explicitly
and that work under every RL name: `clip_eps_high`, `dynamic_sampling`, `overlong_penalty` and
`loss_agg: token` (already the default). Pipeline backends read the name when they compile the
config (see [Other backends](#other-backends)).

```yaml
algorithm:
  name: dapo
  clip_eps_high: 0.28
  dynamic_sampling: true
  overlong_penalty: 1.0
```

### `rloo`

```text
A_i = R_i - mean of R_j over the other members j != i of the group
```

`compute_advantages`: a leave-one-out baseline, no division by a standard deviation.
`advantage_norm` and `advantage_std_norm` are not read. Requires `rollout.group_size >= 2`.

```yaml
algorithm:
  name: rloo
  num_epochs: 2
```

### `reinforce`

`A_i = R_i`: `compute_advantages` returns the raw rewards, with no baseline, and does not read
`advantage_norm` or `advantage_std_norm`. The loss is still the clipped surrogate. The name does
not need a group, but config validation rejects `rollout.group_size: 1` while `advantage_norm` is
`group` (its default), so one rollout per prompt needs `advantage_norm: none` or `batch`:

```yaml
algorithm:
  name: reinforce
  advantage_norm: none
rollout:
  group_size: 1
```

### `ppo`

A learned baseline instead of a group baseline. The reference trainer adds a linear value head,
and `ComponentLoop._critic_batch` computes the advantages with `gae` from the head's estimates.
With `credit_assignment: episode` an episode is one step, so `A = R - V(s_0)`, with `V` read at
the last prompt token of the first turn. The loss gains `value_coef * value_loss`. A trainer
without `compute_values` is rejected with a `ConfigError` when the first batch is built. Details:
[CREDIT_ASSIGNMENT.md](CREDIT_ASSIGNMENT.md).

```yaml
algorithm:
  name: ppo
  lam: 0.95
```

### `sft`

Supervised fine-tuning on dataset responses: no rollouts, rewards, groups or advantages. The run
is driven by `SFTLoop` (`src/monorl/orchestrator/sft.py`) and the loss is `sft_loss`. Validation
requires `env.type: single_turn` and skips the group-size and reward checks. See [SFT](#sft).

```yaml
algorithm:
  name: sft
  sft_turns: all
  sft_label_smoothing: 0.1
```

### Per-turn credit and off-policy corrections

`credit_assignment: turn` gives every turn of a multi-turn episode its own advantage: the
discounted return from that turn on (`turn_returns`), passed through the estimator of
`algorithm.name` per turn index (`turn_advantages`), or through `gae` for `ppo`. With one turn per
episode it equals `credit_assignment: episode`. See [CREDIT_ASSIGNMENT.md](CREDIT_ASSIGNMENT.md).

`algorithm.off_policy.*` (all off by default) adds importance weights, a ratio mask or score
centering to the policy term and chooses the anchor of `logp_old`: [OFF_POLICY.md](OFF_POLICY.md).

## Options

| Field | Default | Effect |
|---|---|---|
| `advantage_norm` | `group` | Read by `grpo`/`dapo`. `group`: mean and std over the prompt group. `batch`: mean and unbiased std over all episodes of the step. `none`: `A = R`. Under every RL name `group` needs `rollout.group_size >= 2`. |
| `advantage_std_norm` | `true` | `grpo`/`dapo` with `group` or `batch`. `false`: `A = R - mean`, no division. |
| `loss_agg` | `token` | Mean used for the policy, KL and entropy terms (`masked_aggregate`). |
| `kl_coef` | `0.0` | Weight of the KL penalty to the reference policy. `0` skips the reference pass. |
| `kl_estimator` | `k3` | Per-token estimator in `kl_penalty`: `k1`, `k2` or `k3`. |
| `entropy_coef` | `0.0` | Subtracts `entropy_coef * agg(entropy)`: the entropy of the trainer's temperature-scaled next-token distribution at the response positions. `loss/entropy` is reported whatever the value. |
| `dynamic_sampling` | `false` | Drops groups whose rewards are all identical and resamples. |
| `overlong_penalty` | `0.0` | Subtracted from `R` when the last turn's `finish_reason` is `length`. |
| `credit_assignment` | `episode` | `episode` or `turn` (previous section). |
| `gamma` | `1.0` | Discount of per-turn environment rewards in `R`, in `turn_returns` and in `gae`. |
| `lam` | `1.0` | `ppo` only. GAE lambda. No effect with `credit_assignment: episode` (one step). |
| `value_coef` | `0.5` | `ppo` only. Weight of the value loss in the total loss. |
| `value_clip` | `0.5` | `ppo` only. Clip distance in `value_loss`. `null`: plain squared error. |

**`loss_agg`.** `token`: the mean over all response tokens of the batch. `sequence`: the token
mean of every training sequence, then the mean over the sequences that have a response token.
`rollout`: every response token weighs one over the number of response tokens of its episode
(`TrainBatch.loss_weights`, set by `build_train_batch`), so every episode counts equally however
many sequences its turns form. The `ppo` value loss is always a mean over the value positions.

**KL penalty.** With `kl_coef > 0` the loop computes `logp_ref` once per batch
(`Trainer.compute_logprobs(batch, reference=True)`). The reference policy of the reference
trainer is: with `model.lora.enabled`, the same model with its adapters disabled; otherwise a
frozen copy loaded from `model.name_or_path` on first use (a second copy of the weights in
memory). With `d = logp_theta - logp_ref` at a response token, `kl_penalty` returns:

```text
k1 = d            k2 = 0.5 * d^2            k3 = exp(-d) + d - 1
```

**`dynamic_sampling`.** After scoring (overlong penalty included), `_drop_uniform_groups` removes
every group with more than one member whose rewards are all equal. While fewer than
`prompts_per_step` groups remain, the loop draws `prompts_per_step` random prompts (seeded by
seed, step and round), rolls them out, filters them the same way and keeps as many groups as are
missing, for at most `rollout.extra.dynamic_sampling_max_rounds` rounds (default 3). A batch can
therefore hold fewer groups; with none left the step makes no update (`rollout/skipped: 1`).

## SFT

`SFTLoop` draws `trainer.prompts_per_step` samples per step (`rollout.group_size` is not used),
encodes each as its prompt messages plus an assistant message holding the response, and calls
`train_step` `num_epochs` times. Where the response text comes from (`data.response_field`,
`data.answer_field` or a trailing assistant turn) is in [DATA.md](DATA.md). SFT stages inside
pipelines are in [PIPELINES.md](PIPELINES.md).

```text
nll_t  = -logp_theta(y_t)
loss_t = (1 - eps) * nll_t + eps * (-mean over the vocabulary v of logp_theta(v))
loss   = agg(loss_t over the trained tokens)
```

`sft_loss`, with `eps = sft_label_smoothing`. `logp_theta` is the same temperature-scaled
log-softmax as in the RL loss, so `rollout.temperature` (default 1.0) applies. Apart from
`loss_agg`, `num_epochs` and the three fields below, no `algorithm.*` field is read.
`loss_agg: rollout` reduces like `sequence` here, because the SFT batch has no loss weights.

| Field | Default | Effect |
|---|---|---|
| `sft_label_smoothing` | `0.0` | `eps` above, in `[0, 1)`. `sft/ppl` always uses the unsmoothed NLL. |
| `sft_turns` | `last` | `last`: only the final assistant message carries the loss. `all`: every assistant message except those a row marks with `"weight": 0`; the final one is always trained. |
| `sft_train_on_prompt` | `false` | `true`: every token of a sequence except its first is trained, prompt included. |

Trained tokens (`encode_sft_conversation`): for every trained turn, the chat before it is rendered
with the generation prompt and the chat including it without one. The tokens the second rendering
adds are the turn's loss span: the response and whatever the chat template closes the turn with.
All other tokens are context. Sequences are cut from the right at `trainer.max_seq_len`.

## Metrics

Names only; meanings are in [RUNS.md](RUNS.md#metrics).

| Reported | Metric names |
|---|---|
| every RL step | `reward/mean`, `reward/std`, `reward/<reward name>`, `advantage/mean`, `advantage/abs_mean`, `rollout/num_sequences`, `rollout/groups_dropped`, `rollout/resample_rounds`, `policy/staleness` |
| every RL update | `loss/policy`, `loss/kl`, `loss/entropy`, `loss/total`, `policy/clipfrac`, `policy/approx_kl`, `policy/ratio_mean`, `grad_norm`, `lr` |
| conditionally | `tokens/old_logprobs_recomputed`, `rollout/skipped`, `reward/overlong_penalty`, `reward/env` |
| `credit_assignment: turn` | `advantage/turn_spread`, `advantage/turn_spread_max`, `advantage/first_turn_abs`, `advantage/last_turn_abs` |
| `ppo` | `loss/value`, `value/clipfrac`, `value/mean`, `value/return_mean`, `value/explained_variance`, `time/values` |
| `off_policy` | `offpolicy/*` (listed in [OFF_POLICY.md](OFF_POLICY.md)) |
| `sft` | `loss/sft`, `sft/ppl`, `sft/tokens`, `sft/turns`, `data/epoch`, `loss/entropy`, `loss/total`, `grad_norm`, `lr` |

`loss/kl` is 0 unless `kl_coef > 0`. `loss/total` is the optimised loss: the policy, KL and
entropy terms, plus `value_coef * loss/value` for `ppo`.

## Other backends

Everything above is the native loop with the reference trainer. The pipeline backends
(`trainer.backend: trl`, `verl`, `nemo_rl`) compile the `algorithm:` section into the framework's
own config, and the framework computes its own advantages and loss, so not every name and option
is available there. Which backend implements which algorithm is in the
[support matrix](../README.md#support-matrix); per-backend options and limits are in
[BACKENDS.md](BACKENDS.md).

## Example

The functions take plain tensors and an `AlgorithmConfig`. Below: two groups, `p0` with rewards
`[1, 0, 0, 1]` and `p1` with rewards `[1, 1]`; then one sequence of three tokens with advantage 1
whose ratios to the old policy are 1.4, 1.0 and 0.6, under `clip_eps_high: 0.28`.

```python
import torch

from monorl.algorithms import compute_advantages, policy_loss
from monorl.config.schema import AlgorithmConfig

rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
groups = ["p0", "p0", "p0", "p0", "p1", "p1"]
for overrides in (
    {"name": "grpo"},
    {"name": "grpo", "advantage_std_norm": False},
    {"name": "rloo"},
    {"name": "reinforce"},
):
    adv = compute_advantages(rewards, groups, algorithm=AlgorithmConfig(**overrides))
    print(overrides, [round(a, 3) for a in adv.tolist()])

algo = AlgorithmConfig(clip_eps=0.2, clip_eps_high=0.28)
old = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
new = torch.log(torch.tensor([[0.7, 0.5, 0.3]]))
out = policy_loss(new, old, torch.tensor([1.0]), torch.ones(1, 3), algorithm=algo)
print({k: round(out.metrics[k], 4) for k in ("loss/policy", "policy/clipfrac")})
```

Output with `.venv/bin/python`:

```text
{'name': 'grpo'} [0.866, -0.866, -0.866, 0.866, 0.0, 0.0]
{'name': 'grpo', 'advantage_std_norm': False} [0.5, -0.5, -0.5, 0.5, 0.0, 0.0]
{'name': 'rloo'} [0.667, -0.667, -0.667, 0.667, 0.0, 0.0]
{'name': 'reinforce'} [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
{'loss/policy': -0.96, 'policy/clipfrac': 0.3333}
```

The policy loss is `-(1.28 + 1.0 + 0.6) / 3`: the first ratio is clipped at `1 + clip_eps_high`;
the third stays at 0.6 because, with a positive advantage, the unclipped term is the smaller one.
