"""Multi-token prediction (MTP) layers held next to a transformers policy (``docs/MTP.md``).

Models such as DeepSeek-V3, GLM, Step, Kimi and Mellum ship extra decoder layers that predict the
tokens *after* the next one. MTP layer ``k`` (0-based) reads, at position ``t``, the hidden state
the previous depth produced there (depth ``-1`` is the policy's last layer) and the embedding of
token ``t+k+1``, and predicts token ``t+k+2``. They draft tokens for speculative decoding and are
trained with an auxiliary loss.

transformers keeps these layers outside the model (``transformers.modeling_layers.MtpModel``,
which shares embedding, head and rotary embedding with it), reloads them from disk for every
``generate(use_mtp=True)`` and cannot write them back to a checkpoint. :class:`MtpLayers` holds
one ``MtpModel`` for the lifetime of an engine or trainer and adds what RL needs:

* drafting with the held layers (:meth:`MtpLayers.install_drafting`), so that weights pushed by
  weight sync are the ones that draft, with exact acceptance counters;
* the teacher-forced loss (:meth:`MtpLayers.loss`);
* the layers' weights **in the checkpoint's own layout** (:meth:`MtpLayers.checkpoint_weights`,
  :meth:`MtpLayers.load_checkpoint_weights`). That layout is how MTP weights travel in weight
  sync (:meth:`MtpLayers.wire_weights`: the checkpoint key behind :data:`WIRE_PREFIX`) and how
  they are written next to an exported model (:func:`write_checkpoint_weights`): transformers,
  vLLM and SGLang all read MTP layers from the checkpoint's keys.

Written against transformers 5.22 (``MtpModel``, ``MTPCandidateGenerator``,
``core_model_loading``).
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading
import types
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn

from monorl.core.errors import ConfigError
from monorl.core.types import SpeculationStats

WIRE_PREFIX = "@mtp/"
"""Prefix of an MTP weight's name in a weight stream (``Trainer.export_weights()`` →
``RolloutEngine.update_weights()``): ``"@mtp/" + <checkpoint key>``. The policy's weights come
under their ``state_dict`` names, so an engine tells the two apart without knowing the model:
tagged names go to its drafter (untagged again, as checkpoint keys), or nowhere when it does not
draft."""

TIED_HEAD = "shared_head.head.weight"
TIED_EMBEDDING = "embed_tokens.weight"
"""Checkpoint entries of an MTP layer that are copies of the policy's output head and input
embedding (DeepSeek-V3 stores them under the MTP layer). ``MtpModel`` shares the policy's modules
instead of loading them."""


def is_wire_name(name: str) -> bool:
    """Whether a name in a weight stream is an MTP layer's weight (:data:`WIRE_PREFIX`)."""
    return name.startswith(WIRE_PREFIX)


def checkpoint_key(name: str) -> str:
    """The checkpoint key of an MTP weight's name in a weight stream."""
    if not name.startswith(WIRE_PREFIX):
        raise ValueError(f"{name!r} is not an MTP weight name")
    return name[len(WIRE_PREFIX) :]


def _mtp_model_class() -> Any:
    """transformers' ``MtpModel``, or a :class:`ConfigError` for a release that has none."""
    try:
        from transformers.modeling_layers import MtpModel
    except ImportError as e:
        import transformers

        raise ConfigError(
            f"model.mtp needs a transformers release with MTP support (5.22 or later); "
            f"{transformers.__version__} is installed"
        ) from e
    return MtpModel


def num_mtp_layers(model_config: Any) -> int:
    """MTP layers a transformers model config declares (0: the model has none)."""
    text = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
    return int(getattr(text, "num_mtp_layers", None) or 0)


def mtp_key_pattern(main_model: nn.Module) -> re.Pattern[str] | None:
    """Regex matching the checkpoint keys of ``main_model``'s MTP layers; ``None`` when the model
    has none. For reading a checkpoint directory, whose keys carry no :data:`WIRE_PREFIX`."""
    if num_mtp_layers(main_model.config) == 0:
        return None
    try:
        return _mtp_model_class().get_mtp_weights_regex(main_model)
    except ValueError:  # the model class registers no MTP key patterns
        return None


def resolve_checkpoint_dir(name_or_path: str, revision: str | None = None) -> Path:
    """Local directory of an HF checkpoint: the path itself, or the Hub snapshot of ``revision``
    (already in the cache once the model has been loaded)."""
    if Path(name_or_path).is_dir():
        return Path(name_or_path)
    from huggingface_hub import snapshot_download

    patterns = ["*.safetensors", "model.safetensors.index.json", "config.json"]
    return Path(snapshot_download(name_or_path, revision=revision, allow_patterns=patterns))


def checkpoint_shards(root: str | os.PathLike[str]) -> list[Path]:
    """The safetensors files of an HF-format directory (sharded or not)."""
    root = Path(root)
    index = root / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (root / "model.safetensors").exists():
        names = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"no model.safetensors(.index.json) under {root}")
    return [root / name for name in names]


def _safetensors_header(path: Path) -> dict[str, Any]:
    """Tensor table of a safetensors file: ``{name: {"dtype", "shape", "data_offsets"}}``."""
    with path.open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header


class MtpLoss(NamedTuple):
    loss: torch.Tensor
    """Mean over the MTP depths of the token-mean cross entropy (see :meth:`MtpLayers.loss` for
    parts of a batch); a zero without gradient when there is no target."""
    sums: list[float]
    """Summed cross entropy of every depth, one entry per MTP layer."""
    counts: list[int]
    """Targets of every depth."""
    correct: int
    """Targets whose most likely draft was the target token, over all depths."""

    @property
    def targets(self) -> int:
        return sum(self.counts)


class _Projection(nn.Module):
    """``MtpModel._project_to_logits`` as a module, so the shared output head can be swapped for
    a detached copy by ``torch.func.functional_call`` without touching the policy."""

    def __init__(self, mtp_model: nn.Module) -> None:
        super().__init__()
        self.mtp = mtp_model

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mtp._project_to_logits(hidden)


class MtpLayers:
    """The MTP layers of ``main_model`` (a transformers causal LM, not a PEFT wrapper), loaded
    from the checkpoint directory ``source`` and kept on the device of the policy's last layer.

    ``parameters()`` are the layers' own parameters: the embedding and the output head belong to
    the policy and are shared, never copied."""

    def __init__(self, main_model: nn.Module, source: str | os.PathLike[str]) -> None:
        MtpModel = _mtp_model_class()  # noqa: N806 - a class
        if num_mtp_layers(main_model.config) == 0:
            raise ConfigError(
                f"model.mtp is enabled but {type(main_model).__name__} declares no MTP layers "
                "(num_mtp_layers / num_nextn_predict_layers in its config)"
            )
        pattern = mtp_key_pattern(main_model)
        if pattern is None:
            raise ConfigError(
                f"{type(main_model).__name__} declares MTP layers but transformers knows no checkpoint keys for "
                "them, so they cannot be loaded"
            )
        self.pattern: re.Pattern[str] | None = pattern
        self.source: Path | None = Path(source)
        table = self._checkpoint_table()
        device = next(main_model.get_decoder().layers[-1].parameters()).device
        # A checkpoint may keep a copy of the embedding with its MTP weights. Where the loader does
        # not ignore it (a copy directly under the MTP prefix), it is written into the embedding the
        # layers share with the policy. That embedding is the policy's: its values are put back.
        embedding = main_model.get_input_embeddings()
        policy_embedding = (
            {name: parameter.detach().clone() for name, parameter in embedding.named_parameters()}
            if any(key.endswith(TIED_EMBEDDING) for key in table)
            else {}
        )
        # MtpModel.from_pretrained reads the directory the policy's config names; point it at `source`.
        name = main_model.config.name_or_path
        main_model.config.name_or_path = os.fspath(self.source)
        try:
            model = MtpModel.from_pretrained(main_model, device_map={"": device})
        except (ValueError, RuntimeError, OSError) as e:
            raise ConfigError(f"cannot load the MTP layers of {self.source}: {e}") from e
        finally:
            main_model.config.name_or_path = name
            with torch.no_grad():
                for parameter_name, parameter in embedding.named_parameters():
                    if parameter_name in policy_embedding:
                        parameter.copy_(policy_embedding[parameter_name])
        self._attach(main_model, model)
        self._record_layout(table)

    @classmethod
    def of(cls, main_model: nn.Module, mtp_model: nn.Module) -> MtpLayers:
        """Layers around an ``MtpModel`` that exists already. They draft and train; without a
        checkpoint they have no checkpoint layout to be exported in."""
        layers = cls.__new__(cls)
        layers.pattern = None
        layers.source = None
        layers._attach(main_model, mtp_model)
        return layers

    def _attach(self, main_model: nn.Module, model: nn.Module) -> None:
        self.main_model = main_model
        self.model = model
        self.device = next(main_model.get_decoder().layers[-1].parameters()).device
        self.num_layers = len(model.layers)
        self._projection = _Projection(model)
        shared = {id(p) for p in main_model.parameters()}
        # remove_duplicate=False: with tied embeddings the head and the embedding are one parameter
        # under two names, and both names have to be known as shared.
        self._shared_names = {
            name for name, p in model.named_parameters(remove_duplicate=False) if id(p) in shared
        }
        self._head_names = [f"mtp.shared_head.{name}" for name, _ in model.shared_head.named_parameters()]
        self._lock = threading.Lock()
        self._stats = SpeculationStats()
        self._conversions: list[Any] = []
        self._converters: list[Any] = []
        self._checkpoint_key: dict[str, str] = {}
        self._tied: dict[str, str] = {}
        self._shapes: dict[str, tuple[int, ...]] = {}

    def _record_layout(self, table: dict[str, tuple[int, ...]]) -> None:
        """How the checkpoint names and lays out these layers. Loading renames a key first and
        then converts tensors (e.g. packs per-expert weights); transformers can undo the
        conversion but not every renaming, so the renamed form of every checkpoint key is kept."""
        from transformers.core_model_loading import WeightConverter, WeightRenaming, apply_renamings

        conversions = getattr(self.model, "_weight_conversions", None) or self.model.get_weight_conversions(
            self.main_model
        )
        self._conversions = list(conversions)
        self._converters = [c for c in self._conversions if isinstance(c, WeightConverter)]
        renamings = [c for c in self._conversions if isinstance(c, WeightRenaming)]
        for key, shape in table.items():
            renamed = apply_renamings(key, renamings)
            tied = next((t for t in (TIED_HEAD, TIED_EMBEDDING) if renamed.endswith(t)), None)
            if tied is not None:
                self._tied[key] = tied
                continue
            if renamed in self._checkpoint_key:
                raise ConfigError(
                    f"MTP checkpoint keys {self._checkpoint_key[renamed]!r} and {key!r} of {self.source} name the "
                    "same parameter"
                )
            self._checkpoint_key[renamed] = key
            self._shapes[key] = shape
        self._verify_layout()

    # -- parameters -----------------------------------------------------------------------
    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """The layers' own parameters under their ``MtpModel`` names."""
        for name, parameter in self.model.named_parameters():
            if name not in self._shared_names:
                yield name, parameter

    def parameters(self) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.named_parameters()]

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Own parameters and buffers under ``MtpModel`` names (what a resumable checkpoint stores)."""
        return {k: v for k, v in self.model.state_dict().items() if k not in self._shared_names}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Inverse of :meth:`state_dict`; copies in place, so optimizers keep their parameters."""
        own = self.state_dict()
        missing, unexpected = sorted(set(own) - set(state)), sorted(set(state) - set(own))
        if missing or unexpected:
            raise KeyError(
                f"MTP state does not match the layers: missing {missing[:3]}, unexpected {unexpected[:3]}"
            )
        with torch.no_grad():
            for name, tensor in own.items():
                tensor.copy_(state[name].to(device=tensor.device, dtype=tensor.dtype))

    # -- checkpoint layout ----------------------------------------------------------------
    def is_checkpoint_key(self, name: str) -> bool:
        return self.pattern is not None and self.pattern.search(name) is not None

    def _layout(self) -> tuple[Path, re.Pattern[str]]:
        if self.source is None or self.pattern is None:
            raise ConfigError(
                "these MTP layers were not loaded from a checkpoint and have no checkpoint layout"
            )
        return self.source, self.pattern

    def checkpoint_weights(
        self, tied_from: Mapping[str, torch.Tensor] | None = None
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """The layers' weights under the checkpoint's own keys and layout (detached, not copied).

        ``tied_from`` (the policy's ``state_dict``-named weights as they are exported) adds the
        checkpoint's copies of the output head and the embedding, for a directory that is to look
        like the original checkpoint. Weight sync leaves them out: engines share those modules
        with the policy."""
        yield from self._to_checkpoint_layout().items()
        if tied_from is None:
            return
        # exported names: a LoRA-wrapped module keeps its weight under `.base_layer`
        main_names = {
            id(p): name.replace(".base_layer", "") for name, p in self.main_model.named_parameters()
        }
        sources = {
            TIED_HEAD: main_names[id(self.main_model.get_output_embeddings().weight)],
            TIED_EMBEDDING: main_names[id(self.main_model.get_input_embeddings().weight)],
        }
        for key, tied in self._tied.items():
            yield key, tied_from[sources[tied]].detach()

    def wire_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """:meth:`checkpoint_weights` named for a weight stream (:data:`WIRE_PREFIX`); what a
        trainer appends to ``export_weights()``."""
        for key, tensor in self.checkpoint_weights():
            yield WIRE_PREFIX + key, tensor

    def load_checkpoint_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Replace the layers' weights by ``weights`` (checkpoint keys and layout; other keys and
        the tied copies are ignored). Every own weight has to be there: a partial update would
        leave layers that draft from a mixture of two policies.

        Parameters are re-created by transformers' loader: for engines, not for a trainer whose
        optimizer holds them (:meth:`load_state_dict` copies in place)."""
        from transformers.core_model_loading import convert_and_load_state_dict_in_model
        from transformers.modeling_utils import LoadStateDictConfig

        self._layout()

        own = {key: tensor for key, tensor in weights.items() if key in self._shapes}
        missing = sorted(set(self._shapes) - set(own))
        if missing:
            raise KeyError(
                f"MTP weights are incomplete: {len(missing)} of {len(self._shapes)} keys missing, e.g. {missing[:3]}"
            )
        for key, tensor in own.items():
            if tuple(tensor.shape) != self._shapes[key]:
                raise ValueError(
                    f"MTP weight {key!r} has shape {tuple(tensor.shape)}, the checkpoint has {self._shapes[key]}"
                )
        trainable = {name: p.requires_grad for name, p in self.named_parameters()}
        training = self.model.training
        dtype = next(iter(self.parameters())).dtype
        with self._lock:
            info, _ = convert_and_load_state_dict_in_model(
                model=self.model,
                state_dict=own,
                load_config=LoadStateDictConfig(
                    weight_mapping=self._conversions, device_map={"": self.device}, dtype=dtype
                ),
            )
            self.model._adjust_missing_and_unexpected_keys(info)
            if info.missing_keys:
                raise RuntimeError(f"MTP weights did not reach {sorted(info.missing_keys)[:3]}")
            self.model.tie_with_main_model(self.main_model)
            self.model.train(training)
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(trainable.get(name, False))

    def _checkpoint_table(self) -> dict[str, tuple[int, ...]]:
        """Shapes of the checkpoint's MTP tensors by key."""
        source, pattern = self._layout()
        table: dict[str, tuple[int, ...]] = {}
        for shard in checkpoint_shards(source):
            for key, entry in _safetensors_header(shard).items():
                if pattern.search(key) is not None:
                    table[key] = tuple(entry["shape"])
        if not table:
            raise ConfigError(f"{source} holds no MTP weights (no key matches {pattern.pattern!r})")
        return table

    def _to_checkpoint_layout(self) -> dict[str, torch.Tensor]:
        from transformers.core_model_loading import revert_weight_conversion

        self._layout()
        state = {key: tensor.detach() for key, tensor in self.state_dict().items()}
        if self._converters:
            # Only the tensor conversions are undone here; the key renamings are undone from the
            # table recorded at load time.
            with self._lock:
                attached = getattr(self.model, "_weight_conversions", None)
                self.model._weight_conversions = self._converters
                try:
                    state = revert_weight_conversion(self.model, state)
                finally:
                    self.model._weight_conversions = attached
        unknown = sorted(set(state) - set(self._checkpoint_key))
        if unknown:
            raise ConfigError(
                f"MTP parameters {unknown[:3]} of {type(self.main_model).__name__} have no key in {self.source}"
            )
        return {self._checkpoint_key[key]: tensor for key, tensor in state.items()}

    def _verify_layout(self) -> None:
        """The way back to the checkpoint has to reproduce its keys and shapes, or an export would
        be unreadable: found out now, not after training."""
        produced = {key: tuple(tensor.shape) for key, tensor in self._to_checkpoint_layout().items()}
        if produced != self._shapes:
            missing = sorted(set(self._shapes) - set(produced))
            changed = sorted(k for k in produced if k in self._shapes and produced[k] != self._shapes[k])
            raise ConfigError(
                f"the MTP layers of {self.source} cannot be written back in the checkpoint's layout "
                f"(keys not reproduced: {missing[:3]}, shapes changed: {changed[:3]})"
            )

    # -- training -------------------------------------------------------------------------
    def target_counts(self, target_mask: torch.Tensor) -> list[int]:
        """Targets of every depth in ``target_mask`` ``[B, T]``: depth ``k`` is trained on the
        target tokens from position ``k+2`` on."""
        return [int(target_mask[:, depth + 2 :].sum()) for depth in range(self.num_layers)]

    def loss(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        detach_trunk: bool,
        totals: Sequence[int] | None = None,
    ) -> MtpLoss:
        """Teacher-forced MTP loss of a right-padded batch: the mean over the depths of the
        token-mean cross entropy.

        ``hidden`` ``[B, T, H]`` are the policy's last-layer hidden states for ``input_ids``
        ``[B, T]`` (``output_hidden_states=[num_hidden_layers - 1]``: what generation hands to the
        MTP layers). ``target_mask`` ``[B, T]`` marks the tokens that are training targets
        (response tokens). Depth ``k`` at position ``t`` is trained to predict token ``t+k+2``
        when that token is a target.

        ``detach_trunk``: the loss reaches the MTP layers only: hidden states, embeddings and the
        shared output head enter as constants.

        ``totals``: :meth:`target_counts` of the whole batch when this call sees a part of it
        (a micro-batch, one trainer rank). Every depth's cross entropy is then divided by the
        batch's count, so the parts' losses add up to the batch's loss and their gradients to
        its gradient."""
        if hidden.shape[:2] != input_ids.shape or target_mask.shape != input_ids.shape:
            raise ValueError("hidden, input_ids and target_mask must agree on [B, T]")
        if totals is not None and len(totals) != self.num_layers:
            raise ValueError(f"totals needs one count per MTP layer ({self.num_layers})")
        batch, width = input_ids.shape
        hidden = hidden.to(self.device)
        input_ids = input_ids.to(self.device)
        targets_mask = target_mask.to(self.device).bool()
        if detach_trunk:
            hidden = hidden.detach()
        positions = torch.arange(width, device=self.device).unsqueeze(0).expand(batch, -1)
        frozen_head: dict[str, torch.Tensor] = {}
        if detach_trunk:
            head = dict(self._projection.named_parameters(remove_duplicate=False))
            frozen_head = {name: head[name].detach() for name in self._head_names}
        entropies: list[torch.Tensor | None] = [None] * self.num_layers
        counts = [0] * self.num_layers
        correct = 0
        for depth in range(self.num_layers):
            length = width - depth - 1  # positions t that have an input token t+depth+1
            if length < 2:
                break
            embeds = self.model.embed_tokens(input_ids[:, depth + 1 :]).to(hidden.device)
            if detach_trunk:
                embeds = embeds.detach()
            hidden = self.model.forward_layer(
                depth, hidden[:, :length], positions[:, depth + 1 :], None, inputs_embeds=embeds
            )
            labels = input_ids[:, depth + 2 :]
            mask = targets_mask[:, depth + 2 :]
            counts[depth] = int(mask.sum())
            if counts[depth] == 0:
                continue
            states = hidden[:, :-1][mask]  # [count, H]: only the positions that are trained
            if frozen_head:
                logits = torch.func.functional_call(self._projection, frozen_head, (states,))
            else:
                logits = self._projection(states)
            wanted = labels[mask]
            entropies[depth] = nn.functional.cross_entropy(logits.float(), wanted, reduction="sum")
            correct += int((logits.argmax(dim=-1) == wanted).sum())
        sums = [0.0 if entropy is None else float(entropy.detach()) for entropy in entropies]
        denominators = list(totals) if totals is not None else counts
        depths = sum(1 for count in denominators if count > 0)
        parts = [
            entropy / denominators[depth] for depth, entropy in enumerate(entropies) if entropy is not None
        ]
        if not parts:
            return MtpLoss(hidden.new_zeros((), dtype=torch.float32), sums, counts, 0)
        return MtpLoss(torch.stack(parts).sum() / depths, sums, counts, correct)

    # -- drafting -------------------------------------------------------------------------
    def install_drafting(self) -> None:
        """Make ``main_model.generate`` with ``generation_config.use_mtp`` draft with these layers
        instead of a copy transformers reloads from disk on every call. Assisted decoding handles
        one sequence per call (a transformers restriction)."""
        layers = self
        upstream = self.main_model._get_candidate_generator

        def _get_candidate_generator(model: Any, generation_config: Any, **kwargs: Any) -> Any:
            if getattr(generation_config, "use_mtp", False):
                return _held_candidate_generator(layers, generation_config, kwargs.get("logits_processor"))
            return upstream(generation_config=generation_config, **kwargs)

        self.main_model._get_candidate_generator = types.MethodType(_get_candidate_generator, self.main_model)

    def take_stats(self) -> SpeculationStats:
        """Drafting counters since the previous call."""
        with self._lock:
            stats, self._stats = self._stats, SpeculationStats()
        return stats

    def drop_emitted(self, tokens: int) -> None:
        """Take back ``tokens`` that a pass emitted beyond the end of its sequence (a stop string
        that ends in the middle of a pass): the caller did not deliver them."""
        with self._lock:
            self._stats.emitted -= tokens

    def _count(self, drafted: int, accepted: int) -> None:
        with self._lock:
            self._stats.add(
                SpeculationStats(steps=1, drafted=drafted, accepted=accepted, emitted=accepted + 1)
            )


def _held_candidate_generator(layers: MtpLayers, generation_config: Any, logits_processor: Any) -> Any:
    """transformers' ``MTPCandidateGenerator`` over ``layers`` instead of freshly loaded ones."""
    from transformers.cache_utils import MtpCache
    from transformers.generation.candidate_generator import MTPCandidateGenerator

    class HeldCandidateGenerator(MTPCandidateGenerator):
        def __init__(self) -> None:
            # MTPCandidateGenerator.__init__ minus MtpModel.from_pretrained (transformers 5.22).
            main = layers.main_model
            self.num_mtp_layers = layers.num_layers
            self.model_kwargs_overrides = {
                "output_hidden_states": [main.config.get_text_config().num_hidden_layers - 1]
            }
            self.device = layers.device
            self.mtp_model = layers.model
            self.mtp_cache = MtpCache(config=main.config.get_mtp_config())
            self.mtp_cache.activate_past_recording()
            self.do_sample = generation_config.do_sample
            self.logits_processor = logits_processor
            self.is_main_model_prefill = True
            self._drafted = 0

        def get_candidates(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            candidates, logits = super().get_candidates(input_ids, *args, **kwargs)
            self._drafted = candidates.shape[1] - input_ids.shape[1]
            return candidates, logits

        def update_candidate_strategy(
            self, input_ids: torch.Tensor, scores: torch.Tensor, num_matches: int
        ) -> None:
            layers._count(self._drafted, int(num_matches))

    return HeldCandidateGenerator()


def write_checkpoint_weights(
    directory: str | os.PathLike[str], weights: Iterable[tuple[str, torch.Tensor]]
) -> None:
    """Add ``weights`` to an HF-format directory written by ``save_pretrained`` as one more
    safetensors shard, listed in ``model.safetensors.index.json`` like the others (an unsharded
    ``model.safetensors`` becomes the first of two shards). ``save_pretrained`` leaves a model's
    MTP layers out; with them the directory is a complete checkpoint again."""
    from safetensors.torch import save_file

    root = Path(directory)
    tensors = {key: tensor.detach().to("cpu").contiguous() for key, tensor in weights}
    if not tensors:
        return
    shards = checkpoint_shards(root)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"metadata": {}, "weight_map": {}}
    total = len(shards) + 1
    weight_map: dict[str, str] = {}
    size = 0
    for number, shard in enumerate(shards, start=1):
        name = f"model-{number:05d}-of-{total:05d}.safetensors"
        for key, entry in _safetensors_header(shard).items():
            if key in tensors:
                raise ValueError(f"{shard.name} already holds {key!r}")
            weight_map[key] = name
            size += entry["data_offsets"][1] - entry["data_offsets"][0]
        shard.rename(root / name)
    name = f"model-{total:05d}-of-{total:05d}.safetensors"
    save_file(tensors, str(root / name), metadata={"format": "pt"})
    for key, tensor in tensors.items():
        weight_map[key] = name
        size += tensor.numel() * tensor.element_size()
    index["metadata"] = {**index.get("metadata", {}), "total_size": size}
    index["weight_map"] = weight_map
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


__all__ = [
    "TIED_EMBEDDING",
    "TIED_HEAD",
    "WIRE_PREFIX",
    "MtpLayers",
    "MtpLoss",
    "checkpoint_key",
    "checkpoint_shards",
    "is_wire_name",
    "mtp_key_pattern",
    "num_mtp_layers",
    "resolve_checkpoint_dir",
    "write_checkpoint_weights",
]
