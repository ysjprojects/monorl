# Multi-token prediction (MTP)

Some models ship extra decoder layers that predict the tokens *after* the next one: DeepSeek-V3,
GLM, Step, MiMo, Mellum and others. MTP layer `k` (0-based) reads, at position `t`, the
hidden state the previous depth produced there (depth `-1` is the policy's last layer) and the
embedding of token `t+k+1`, and predicts token `t+k+2`. They serve two purposes:

- **Drafting.** During generation the layers propose the next tokens and the policy verifies them
  in one pass (speculative decoding). A step emits the accepted drafts plus one token of the
  policy's own.
- **An auxiliary loss.** The layers are trained to predict tokens further ahead.

monorl does three things with them, all off by default, switched by `model.mtp`:

| Option | Default | Effect |
|---|---|---|
| `enabled` | `false` | The trainer holds the checkpoint's MTP layers. They travel in weight sync, are saved in checkpoints and are written into the final export. Needs a model that ships MTP layers. |
| `rollout` | `false` | Rollout engines draft with the layers. Sampled tokens and recorded logprobs are the policy's; only speed changes. |
| `num_speculative_tokens` | unset | Tokens drafted per decoding step. Unset: one per MTP layer on the `hf` engine, 1 on vLLM, SGLang's own default (3 steps). |
| `train` | `false` | Train the layers with the MTP loss on the response tokens of every batch, in RL and in SFT. |
| `loss_coef` | `0.1` | Weight of the MTP loss in the total loss. |
| `detach_trunk` | `true` | `true`: the MTP loss updates the MTP layers only. `false`: it also flows into the policy's parameters, as in pretraining. |

```yaml
model:
  name_or_path: <a model with MTP layers>
  mtp: {enabled: true, rollout: true, train: true}
```

`rollout` and `train` need `enabled`. Why train them during RL: the layers were fitted to the
policy the checkpoint holds. RL moves the policy, and layers that are not trained keep predicting
the old one. How fast acceptance falls as a real policy moves has not been measured here; the
tiny run below shows the other direction (layers that start useless and are trained).

## What each backend does

| | `enabled` | `rollout` | `train` |
|---|---|---|---|
| `reference` trainer | holds, syncs, checkpoints, exports the layers | – | MTP loss, own parameter group |
| `hf` engine | – | drafts with transformers' assisted decoding, one sequence at a time | – |
| vLLM engine | – | `speculative_config {method: mtp}`; pushed weights reach the drafter | – |
| SGLang engine | – | `speculative_algorithm: EAGLE` (its path for native MTP layers); pushed weights reach the draft runner | – |
| verl | `actor_rollout_ref.model.mtp.enable` | `enable_rollout` plus vLLM or SGLang keys | rejected: verl trains MTP only with its Megatron engine, monorl compiles the FSDP actor |
| NeMo-RL | nothing compiled | `policy.generation.vllm_kwargs.speculative_config` | Megatron policy only: `trainer.extra.megatron.enabled` and `mtp_num_layers` required |
| TRL, Unsloth | rejected | rejected | rejected |

Run here on CPU: the `hf` engine and the `reference` trainer, with a tiny random Mellum (one MTP
layer). vLLM, SGLang, verl and NeMo-RL are written against the pinned sources and tested with
fakes. Other model families go through the same transformers loader and have not been run.

## Rollouts

Speculative decoding does not change what is sampled. Greedy: a draft is accepted when it is the
policy's argmax. Sampling: a draft `x ~ q` is accepted with probability `min(1, p(x) / q(x))` and
otherwise replaced by a sample from the normalised `max(0, p - q)`, which leaves every emitted
token distributed as the policy's `p` (Leviathan et al., algorithm 1). The recorded logprob of a
token is the policy's, `log_softmax(logits / temperature)`, from the verification pass: the same
definition as without drafting, so losses and off-policy corrections are unaffected.

Per engine:

- **`hf`.** transformers reloads a model's MTP layers from disk on every `generate(use_mtp=True)`
  call; monorl holds them for the engine's lifetime (`monorl.mtp.MtpLayers.install_drafting`) so
  that the weights pushed by weight sync are the ones that draft. transformers' assisted decoding
  takes one sequence per call, so the engine decodes its rows one after another: this engine is
  for checking behaviour, not for throughput. It drafts one token per MTP layer
  (`num_speculative_tokens` must be unset or that number) and text prompts only.
- **vLLM.** `speculative_config = {"method": "mtp", "num_speculative_tokens": N or 1}`; keys in
  `rollout.extra.engine_kwargs.speculative_config` are laid over it. A `rejection_sample_method`
  other than `standard` is rejected (the lossy ones sample from another distribution than the
  recorded logprobs describe), as is sleep level 2 with drafting, which discards the drafter's
  weights. Weight updates reach the drafter as well as the policy, including the quantized copy
  of the output head that `draft_lm_head_quantization` keeps. In server mode
  (`rollout.extra.mode: server`) monorl does not start the server: the client refuses a server
  that does not draft as configured and names the `--speculative-config` to start it with.
- **SGLang.** `speculative_algorithm: EAGLE`, and with `num_speculative_tokens = N`
  `speculative_num_steps: N`, `speculative_eagle_topk: 1`, `speculative_num_draft_tokens: N + 1`;
  with the memory saver the CPU backups of target and draft weights are switched on. Accept
  thresholds below 1.0 are rejected for the same reason as vLLM's lossy samplers. Policy and
  draft runner are updated in one weight-update session. In server mode the client checks that
  the server drafts as configured before the first generation.
- `min_p` cannot be combined with drafting on vLLM and SGLang (and on verl / NeMo-RL, whose
  rollouts run on them): vLLM rejects such requests and SGLang's verification does not apply it.

Metrics, per training step and per evaluation (`eval/mtp/...`), from engines that draft:

| Metric | Meaning |
|---|---|
| `mtp/accept_rate` | accepted share of the drafted tokens |
| `mtp/accept_length` | tokens emitted per verification pass of the policy; 1.0 means no draft was accepted |

A caveat from verl's MTP guide (`third_party/verl/docs/advance/mtp.md`): on their hardware
drafting raised acceptance by about 14% but did not raise rollout throughput, and lowered it by
about half for a 7B model on H20 GPUs with SGLang. Measure before relying on it
(`monorl bench rollout`).

## Training

With `train`, every batch adds

    L = L_policy (+ value_coef * L_V) + loss_coef * L_MTP
    L_MTP = mean over depths k of  mean over targets  -log q_k(x_{t+k+2} | h^{k-1}_t, x_{t+k+1})

- **Targets** are the response tokens of the batch (`TrainBatch.response_mask`): depth `k` is
  trained at every position whose token `k+2` places ahead is a response token. Prompts are
  context only. In SFT the response tokens are the assistant turns that carry the SFT loss.
- **Teacher forcing.** The layers read the batch's own tokens, and the policy's last-layer hidden
  states from the same forward pass the policy loss uses. The logits a depth is trained on are
  the logits drafting computes for the same tokens (tested against transformers' drafting loop
  for two depths).
- **`detach_trunk: true`** (default): hidden states, embeddings and the shared output head enter
  as constants, so the policy's update is exactly what it is without MTP training. `loss_coef`
  then scales a gradient that only the MTP layers receive, which an Adam-type optimizer
  normalises away.
- **`detach_trunk: false`** lets the loss shape the policy too, with `loss_coef` setting how hard
  it pulls. verl's guide reports that this is the only setting that changed their training
  results, and recommends the detached form.
- **Parameters.** The layers' own parameters train as full parameters (also next to a LoRA
  policy) in a parameter group of their own at `trainer.learning_rate`, under the policy's
  schedule and weight decay. Their gradient is clipped to `trainer.max_grad_norm` on its own:
  a joint clip would let the MTP gradient change the size of the policy's step. `grad_norm`
  stays the policy's. Embedding and output head are the policy's and are shared, never copied.
- **Micro-batches and trainer ranks.** Each part's loss is divided by the whole batch's target
  count per depth, so the accumulated gradient is the full-batch gradient.

Metrics: `loss/mtp`, `mtp/accuracy` (share of targets whose most likely draft is the target),
`loss/mtp_<k>` per depth when the model has several MTP layers; `loss/total` includes
`loss_coef * loss/mtp`.

Cost: one pass through the MTP layers per micro-batch (one decoder layer per depth) and their
gradients. Without `train` the layers cost no forward pass in the trainer.

## Weights: sync, checkpoints, export

MTP weights travel **under the checkpoint's own keys and layout** (Mellum: `mtp.fc.weight`,
`mtp.layers.0.mlp.experts.3.down_proj.weight`; DeepSeek-V3: `model.layers.61.*`), because
transformers, vLLM and SGLang all load MTP layers from those keys:

- `Trainer.export_weights()` yields the policy's weights and then the MTP layers', each named
  `@mtp/<checkpoint key>` (`monorl.mtp.WIRE_PREFIX`). The tag lets an engine tell them from the
  policy's weights without knowing the model: tagged weights go to its drafter under the bare
  checkpoint key, or nowhere when it does not draft, and never into the policy. Every weight
  sync strategy carries them.
- `checkpoints/step-N` holds the layers with the optimizer state; resuming a run with MTP from a
  checkpoint without them is an error.
- `final/` is a complete checkpoint again: `save_pretrained` leaves a model's MTP layers out, so
  they are added as one more safetensors shard listed in `model.safetensors.index.json`
  (including the copies of output head and embedding that DeepSeek-style checkpoints keep under
  the MTP layer, written with the policy's current values). transformers
  (`MtpModel.from_pretrained`) and monorl load it back.
- Loading MTP layers never changes the policy. Where transformers does not ignore a checkpoint's
  MTP copy of the embedding (seen with a copy stored as `mtp.embed_tokens.weight`), it writes it
  into the embedding module the layers share with the policy; monorl puts the policy's values
  back.

transformers 5.22 can load MTP layers but not write them: `MtpModel` is not saved with the
model, and reverting its key renamings produces invalid keys. monorl records, when it loads the
layers, which checkpoint key each parameter came from, lets transformers undo the tensor
conversions (for example unpacking per-expert weights) and puts the recorded keys back
(`monorl.mtp.MtpLayers`). The round trip is checked at load time, before any training.

## Try it on CPU

```bash
.venv/bin/python scripts/dev/tiny_mtp_model.py /tmp/monorl-tiny-mtp   # random tiny Mellum, one MTP layer
monorl train --config configs/recipes/grpo_mtp_tiny_cpu.yaml
```

The model is random and knows nothing; the recipe shows the mechanics. Its MTP layer starts
random too, so the first rollouts accept no draft (`mtp/accept_length` 1.0). Each step trains
the layer on the batch, weight sync pushes it to the engine, and the next rollouts are drafted
by the trained layer: `loss/mtp` falls while `mtp/accuracy`, `mtp/accept_rate` and
`mtp/accept_length` rise. `final/` loads with its MTP layer in transformers. One run of the
recipe (16 steps, seed 42): `loss/mtp` 38.1 → 16.2, `mtp/accuracy` 0 → 0.27, `mtp/accept_rate`
0 → 0.17 on the sampled rollouts and 0.41 in the greedy evaluation after step 16
(`eval/mtp/accept_length` 1.38). With `model.mtp.train: false` the same recipe drafts with the
untouched layer: over 8 steps `mtp/accept_rate` stayed 0 in every step and in the evaluation, and
`final/` held the layer as the checkpoint had it.

## Limits

- Models: what `transformers.modeling_layers.MtpModel.from_pretrained` loads for the model's
  class. A model whose class registers no MTP key patterns, or a checkpoint without MTP weights,
  is a `ConfigError`. The native path needs transformers 5.22 or later (tested with 5.22.1 and
  5.22.2); it reaches into loader internals that a later release may move.
- `hf` engine: one sequence at a time, text prompts only, one drafted token per MTP layer.
- vLLM and SGLang decide themselves which architectures they can draft for; an unsupported one
  fails when the engine starts.
- verl: drafting only. NeMo-RL: drafting on vLLM; training with the Megatron policy, whose
  drafter weights reach vLLM by NeMo-RL's own refit.
- No MTP loss on sampled drafts, no distillation of the policy's distribution into the layers
  (targets are the tokens of the batch), no separate learning rate for the layers.
- Evaluation and rejection sampling draft when `rollout` is on; only training runs report the
  acceptance metrics.

## Sources

- MTP layers and drafting: `transformers/modeling_layers.py` (`MtpModel`, `MtpLayer`),
  `transformers/generation/candidate_generator.py` (`MTPCandidateGenerator`),
  `transformers/generation/utils.py` (`_assisted_decoding`, `_speculative_sampling`) at 5.22.1
  and 5.22.2.
- Speculative sampling: Leviathan, Kalman, Matias, "Fast Inference from Transformers via
  Speculative Decoding", <https://arxiv.org/abs/2211.17192>, as implemented by transformers.
- vLLM: `third_party/vllm/docs/features/speculative_decoding/mtp.md`,
  `vllm/config/speculative.py`, `vllm/v1/worker/gpu_worker.py` (`get_draft_model`).
- SGLang: `python/sglang/srt/arg_groups/fields/spec.py`, `arg_groups/speculative_hook.py`,
  `managers/scheduler_components/weight_updater.py`.
- verl: `third_party/verl/docs/advance/mtp.md`, `verl/workers/config/model.py` (`MtpConfig`).
- NeMo-RL: `nemo_rl/models/megatron/setup.py` (`_apply_mtp_config`),
  `nemo_rl/models/generation/vllm/vllm_worker.py`.
- The MTP objective as an auxiliary loss with shared embedding and output head: DeepSeek-V3
  technical report, <https://arxiv.org/abs/2412.19437>, section 2.2 (not re-read for this work;
  the layer structure used here is transformers' implementation of it).
