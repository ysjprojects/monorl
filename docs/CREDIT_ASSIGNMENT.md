# Credit assignment in multi-turn episodes

An episode is a sequence of turns: the policy acts (one completion), the environment answers, the
policy acts again. By default every token of an episode carries the same advantage, computed from
one scalar per episode. `algorithm.credit_assignment: turn` credits the tokens of a turn with what
happened from that turn on, and `algorithm.name: ppo` replaces the group baseline with a learned
value head. Everything here is off by default: a default config computes exactly the advantages it
computed before this document existed.

| Option | Values | Effect |
|---|---|---|
| `algorithm.credit_assignment` | `episode` (default) \| `turn` | what a response token is credited with: its episode's reward, or the return from its own turn on |
| `algorithm.gamma` | 0..1 (default 1) | discount per turn of rewards that arrive later |
| `algorithm.name: ppo` | | advantages from a value head on the policy's hidden states instead of the prompt group |
| `algorithm.lam` | 0..1 (default 1) | GAE lambda (`ppo`) |
| `algorithm.value_coef` | default 0.5 | weight of the value loss (`ppo`) |
| `algorithm.value_clip` | default 0.5, `null` = off | clip range of the value loss (`ppo`) |

## Rewards and returns

Episode `i` has turns `k = 0 .. K_i - 1` in the order the policy took them (`Trajectory.turns`; for
an episode with several conversations, a sub-agent or a compacted history, that is the order of the
calls, not of a thread). The environment pays `r_{i,k}` after turn `k` (`Turn.reward`) and the
reward function scores the finished episode with `S_i` (minus `algorithm.overlong_penalty` when the
last completion was cut off).

The **episode reward** is what `reward/mean`, `trajectories.jsonl`, dynamic sampling and the
`episode` estimators use. It is unchanged:

    R_i = S_i + sum_k gamma^k r_{i,k}

For turn-level credit the score is a reward of the final turn. The **reward of a turn** and the
**return from a turn on** are

    rho_{i,k} = r_{i,k} + [k = K_i - 1] S_i
    G_{i,k}   = sum_{j >= k} gamma^(j-k) rho_{i,j}  =  rho_{i,k} + gamma G_{i,k+1}

Rewards paid before turn `k` are not in `G_{i,k}`: they cannot have been caused by it, and leaving
them out removes noise without changing the expected gradient (the reward-to-go form of the policy
gradient). With `gamma = 1`, `G_{i,0} = R_i`. With `gamma < 1` the two differ in one respect: `R_i`
does not discount the score, `G_{i,0}` discounts it by `gamma^(K_i - 1)` like any reward `K_i - 1`
turns away, which is what makes the turn-level estimators and GAE agree with each other.

## `credit_assignment: episode` (default)

Every response token of episode `i` gets `A_i`, computed from the episode rewards of its prompt
group `g(i)` of `n` episodes (`monorl.algorithms.compute_advantages`):

| `algorithm.name` | `A_i` |
|---|---|
| `grpo`, `dapo` with `advantage_norm: group` | `(R_i - mean_g R) / (std_g R + 1e-6)`; without the division when `advantage_std_norm: false` |
| `grpo`, `dapo` with `advantage_norm: batch` | the same with mean and std over the whole batch |
| `grpo`, `dapo` with `advantage_norm: none`, `reinforce` | `R_i` |
| `rloo` | `R_i - mean_{j in g, j != i} R_j` (no baseline in a group of one) |
| `ppo` | `R_i - V(s_{i,0})`: the episode is one step, valued at the state of its first turn (below) |

## `credit_assignment: turn`

The tokens of turn `k` of episode `i` get `A_{i,k}`: the estimator above, applied per turn index to
the returns from that turn on (`monorl.algorithms.turn_advantages`).

### Comparing episodes with different numbers of turns

Episodes of a prompt group share the prompt and nothing else; after the first action their states
differ and so may their lengths. Turn `k` of an episode is compared with turn `k` of every other
member of its group, and **a member that has fewer turns stands in with its last turn**:

    x_{j,k} = G_{j, min(k, K_j - 1)}                   aligned return of member j at turn index k
    A_{i,k} = Est( x_{1,k}, ..., x_{n,k} )_i           for k < K_i

where `Est` is the row of the table above for the configured estimator, e.g.

    rloo:                       A_{i,k} = G_{i,k} - mean_{j != i} x_{j,k}
    grpo (group, std):          A_{i,k} = (G_{i,k} - mean_j x_{j,k}) / (std_j x_{j,k} + 1e-6)
    grpo/dapo (batch):          the same with mean and std over the aligned returns of the whole batch
    reinforce, advantage_norm none:   A_{i,k} = G_{i,k}

In words: an episode that has ended keeps its final return-to-go, the reward of its last turn
including the score, and is compared as such with the later turns of longer episodes.

What this definition guarantees:

1. **No dependence on the turn's own action beyond the episode-level estimator's.** Every
   comparison has all `n` members in it. For `rloo` the baseline of a turn is built from other
   episodes only, which are independent of this episode's actions given the prompt, so the estimator
   stays unbiased. For the group mean, `G_{i,k} - mean_j x_{j,k} = (1 - 1/n) (G_{i,k} - mean_{j != i}
   x_{j,k})`: the own return shrinks the advantage by `1 - 1/n`, the same factor at every turn and
   the same factor the episode-level group mean has. The std includes the own return exactly as it
   does at episode level.
2. **Only per-turn rewards change anything.** With one turn per episode, or with rewards that arrive
   at the final turn only and `gamma = 1`, `x_{j,k} = R_j` for every `k`, so every turn gets exactly
   the episode-level advantage, whatever the turn counts.
3. **An episode's earlier rewards never reach a turn.** `A_{i,k}` is a function of episode `i`'s
   rewards at turns `>= k` and of every other member's rewards from its aligned turn on.

What it costs, and the alternatives that were not chosen:

- The baseline of a late turn is not conditioned on "the episode is still running". If episodes
  that end early earn systematically more (or less) at their last turn than long episodes earn late,
  the late turns of long episodes get a baseline that is off by that difference: more variance,
  and for the mean/std estimators a constant offset within the group at that turn index. It is
  never worse than `episode` in this respect, which compares every turn with whole-episode rewards.
- *Survivors only* (compare turn `k` with the members that have a turn `k`) conditions on the
  episode still running, but the comparison shrinks with `k`: the group-mean shrink factor
  `1 - 1/n_k` then varies by turn and reaches 0 for a lone survivor, whose baseline would be its own
  return; a std over two survivors turns any difference into `+-0.707`; and final-turn-only rewards
  no longer reproduce the episode-level numbers.
- *Zero after the end* (a finished episode contributes return 0 to later turns) drags the baseline
  of late turns towards 0 for every episode that finished earlier, so late turns look good merely
  for existing.
- *State-aligned comparison* (Wei et al., MT-GRPO: sample the `G` actions of turn `k` from the same
  state and normalise each turn's reward within that group) compares like with like but needs a
  rollout tree, `G^(K-1)` episodes per prompt; monorl samples `G` independent episodes. Their
  advantage is the discounted sum of per-turn normalised rewards; without the division by the std
  that sum is the centred return-to-go, the quantity used here, over their state-aligned groups.
- A learned state value does condition on the state: that is `ppo`, below.

`advantage_norm: batch` uses the same alignment with the batch as the comparison set. Dynamic
sampling (`algorithm.dynamic_sampling`) still drops the groups whose *episode* rewards are all
equal, also when their per-turn rewards differ.

## Tokens, training rows and loss aggregation

`monorl.orchestrator.batching.build_train_batch` takes one advantage per trajectory (`[N]`) or one
per turn (an `[N, K]` table, `K` the largest turn count). An episode whose turns do not continue
each other token for token spans several training rows (`trajectory_rows`); every turn is a loss
position in exactly one of them, and every response token carries the advantage of its own turn in
whichever row that is. `algorithm.loss_agg` is untouched: `token`, `sequence` and `rollout` decide
how tokens are averaged, not what they are credited with, so with `rollout` an episode still
counts once however its turns are spread over rows. `TrainBatch.rewards` stays the episode reward.

## `algorithm.name: ppo`: a learned value baseline

The reference trainer (and the Unsloth trainer, which wraps it) adds a value head to the policy:

- **Where.** A linear layer `hidden -> 1` on the policy's final hidden states, the input of its
  output embedding (`lm_head`). The value of turn `k`, `V(s_{i,k})`, is read at the turn's *state*:
  the last prompt token before its completion, in the row the turn is a loss position in. That
  position has seen the whole context of the turn and none of its action.
- **Parameters.** float32, zero-initialised: the first advantages are plain returns and the policy
  receives no value gradient until the head has moved. It trains with the policy's optimizer in its
  own parameter group at 10 times `trainer.learning_rate` (the factor the verl compilation of `ppo`
  gives its critic), under the same schedule, weight decay and gradient clipping. The value loss
  back-propagates into the policy's trainable parameters (LoRA adapters or all weights).
- **Values for the advantages.** Once per rollout batch, before the first epoch, the loop calls
  `Trainer.compute_values(batch)`: one forward pass without gradient.
- **Advantages** (`monorl.algorithms.gae`, one step per turn, the episode ends after its last turn):

      delta_{i,k} = rho_{i,k} + gamma V(s_{i,k+1}) - V(s_{i,k})        V(s_{i,K_i}) = 0
      A_{i,k}     = sum_{l >= 0} (gamma lam)^l delta_{i,k+l}
      T_{i,k}     = A_{i,k} + V(s_{i,k})                                 value target

  `lam = 1` gives `A_{i,k} = G_{i,k} - V(s_{i,k})` and `T_{i,k} = G_{i,k}`; `lam = 0` the one-step
  TD error. With `credit_assignment: episode` the episode is a single step: `A_i = R_i - V(s_{i,0})`
  on every token, `T_i = R_i`, and only the first state of an episode is a value position. The
  advantages are used as they are: `advantage_norm` applies to `grpo`/`dapo` only, there is no
  batch whitening.
- **Value loss.** With `v` the head's current estimate, `v_old` the estimate the advantages were
  computed with and `c = value_clip`:

      L_V = 0.5 * mean_states max( (v - T)^2, (clip(v, v_old - c, v_old + c) - T)^2 )
      L   = L_policy + value_coef * L_V

  The mean runs over all value positions of the batch, whatever `loss_agg` says. `v_old` and `T`
  travel in the batch (`TrainBatch.values`, `returns`, `value_mask`) and stay fixed over
  `algorithm.num_epochs`, which is what the clip is for: it bounds how far the value function moves
  on one batch. `value_clip: null` gives the plain squared error.
- **Micro-batches and trainer ranks.** A micro-batch's value loss is weighed by its share of the
  batch's value positions (the global count across ranks), so the accumulated gradient is the
  full-batch gradient. The head is part of the gradient all-reduce.
- **Checkpoints and export.** `trainer_state.pt` holds the head next to the optimizer state;
  resuming a `ppo` run from a checkpoint without one is an error. `export_weights`, weight sync and
  `save_pretrained` are the policy alone: `final/` is a plain causal LM.
- **Turns without a state in the batch.** A turn with an empty prompt, a turn cut off by
  `trainer.max_seq_len`, and the turns of a multimodal episode outside its single training row have
  no value position: their value counts as 0 in the recursion above and they contribute no value
  loss. The default context budget (`rollout.max_context_tokens`) keeps episodes inside
  `max_seq_len`, so in practice this concerns multimodal episodes with several contexts only.

Trainers without a value head cannot run `ppo`: the loop raises `ConfigError` at the first batch.

## Composition

- `algorithm.off_policy`: importance weights, ratio masks and score centering multiply the policy
  term of every token as before, whatever its advantage is. Values are computed by the trainer when
  the update on a batch starts, like the `trainer` anchor; the value loss is not importance-weighted
  and there is no V-trace correction of the targets, so with stale batches
  (`rollout.async_steps_ahead > 0`) the head regresses returns of the behaviour policy.
- `algorithm.num_epochs`: advantages, old values and targets are computed once per batch.
- `algorithm.kl_coef`, `entropy_coef`, clipping: unchanged; they act on the policy term.
- `cluster.mode: ray` with several trainer ranks: `compute_values` is sharded like
  `compute_logprobs`; GAE runs on the driver over whole episodes, so an episode may be split across
  ranks.
- Evaluation and rejection sampling use the episode reward only.

## Metrics

Per training step, in addition to `advantage/mean` and `advantage/abs_mean` (over response tokens):

| Metric | When | Meaning |
|---|---|---|
| `advantage/turn_spread`, `advantage/turn_spread_max` | `turn` | largest minus smallest advantage among the turns of an episode: mean and maximum over episodes. 0 means every turn is credited alike, as with `episode` |
| `advantage/first_turn_abs`, `advantage/last_turn_abs` | `turn` | mean magnitude of the advantage of first and of final turns |
| `value/explained_variance` | `ppo` | `1 - Var(T - v_old) / Var(T)` over the batch's value positions before the update: 1 = the head predicts the targets, 0 = no better than a constant, negative = worse. Omitted when the targets do not vary |
| `value/mean`, `value/return_mean` | `ppo` | mean value estimate and mean target at the value positions |
| `loss/value`, `value/clipfrac` | `ppo` | `L_V` and the share of value positions where the clipped term is the larger one, averaged over epochs; `loss/total` includes `value_coef * L_V` |
| `time/values` | `ppo` | seconds spent in the value pass |

## What each option costs

| Option | Cost |
|---|---|
| `credit_assignment: turn` | arithmetic on an `[episodes, turns]` table per step; no model pass, no extra memory in the batch |
| `ppo` | one forward pass over the batch without gradient per rollout batch; three `[B, T]` tensors in the batch; `hidden + 1` parameters; the value loss' backward pass shares the policy's. The hidden states are read where the model hands them to its output embedding, so no per-layer outputs are kept |
| `ppo`, statistical | a value head that explains nothing adds noise through `gamma V(s')` when `lam < 1`; with `lam = 1` a bad head costs variance only. Its gradient shares the policy's trainable parameters and the gradient-norm clip with the policy loss; lower `value_coef` if `loss/value` dominates |

## Pipeline backends

TRL, verl and NeMo-RL compute their own advantages (one per completion or rollout; verl's `ppo`
runs token-level GAE with its own critic model). `credit_assignment: turn` is rejected with a
`ConfigError` naming the backend. For verl's `ppo`, `value_clip` is compiled to
`critic.cliprange_value` (0.5 is the default on both sides); `value_clip: null` is rejected there
because verl's value loss always clips.

## Not implemented

- Per-turn rollout trees (state-aligned group baselines as in MT-GRPO).
- Token-level values and GAE inside a turn (verl's and NeMo-RL's GAE estimators), advantage
  whitening for `ppo`, separate lambdas for policy and value targets, return normalisation.
- Off-policy value targets (V-trace) for stale batches.
- Dynamic sampling on per-turn rewards.

## Sources

- Reward-to-go and state-dependent baselines leave the expected policy gradient unchanged: OpenAI
  Spinning Up, "Intro to Policy Optimization", sections "Don't Let the Past Distract You" and
  "Baselines in Policy Gradients": <https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html>.
- Turn-level MDP for LLM agents, per-turn rewards, MT-GRPO (Eq. 3-5) and its per-turn rollouts: Wei
  et al., "Reinforcing Multi-Turn Reasoning in LLM Agents via Fine-Grained Reward Structure and
  Credit Assignment": <https://arxiv.org/abs/2505.11821> (v3, sections 3-5).
- GAE: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage
  Estimation": <https://arxiv.org/abs/1506.02438>. Not read directly; the recursion is the one in
  Wei et al. section 5.1 and in verl `verl/trainer/ppo/core_algos.py`
  `compute_gae_advantage_return` at the pinned commit, which agree. verl whitens the advantages
  afterwards; monorl does not.
- Clipped value loss: verl `core_algos.py` `compute_value_loss` (which credits TRL's former
  `ppo_trainer.py`) and `verl/trainer/config/critic/critic.yaml` (`cliprange_value: 0.5`,
  `optim.lr: 1e-5` against the actor's `1e-6`). The objective with a value-loss coefficient is from
  Schulman et al., "Proximal Policy Optimization Algorithms": <https://arxiv.org/abs/1707.06347>
  (not read directly).
- Leave-one-out baseline: Kool, van Hoof, Welling, "Buy 4 REINFORCE Samples, Get a Baseline for
  Free!" (ICLR 2019 workshop), as cited by Wei et al.; not read directly. monorl's episode-level
  definition predates this document (`monorl.algorithms.compute_advantages`).
- Backends' advantage granularity, read at the pinned versions: TRL
  `trl/trainer/grpo_trainer.py` (`advantages = rewards - mean_grouped_rewards`), NeMo-RL
  `nemo_rl/algorithms/advantage_estimator.py`, verl `core_algos.py`
  (`compute_grpo_outcome_advantage`).
- Unsloth's forward calls `lm_head` as a module when it returns logits:
  `unsloth/models/llama.py` `CausalLM_fast_forward` at the pinned commit.
- The alignment of unequal turn counts, its guarantees and the identity
  `x_i - mean_j x_j = (1 - 1/n)(x_i - mean_{j != i} x_j)` are original to this repo.
