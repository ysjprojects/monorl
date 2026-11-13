#!/usr/bin/env python
"""Build a random tiny model with an MTP layer, for trying ``model.mtp`` on CPU (docs/MTP.md).

    .venv/bin/python scripts/dev/tiny_mtp_model.py /tmp/monorl-tiny-mtp
    monorl train --config configs/recipes/grpo_mtp_tiny_cpu.yaml

The model is a Mellum (a family that ships MTP layers) with two decoder layers of hidden size 64,
one MTP layer stored the way JetBrains' checkpoints store it, and the SmolLM tokenizer; it is the
model the MTP tests use (``tests/rollout/tiny_mtp.py``). It knows nothing. Its output head is
scaled by ``--sharpness`` so that the random policy has preferences instead of sampling
near-uniformly: only a policy with preferences can be predicted, and only then does training the
MTP layer show up as accepted drafts. At the default its most likely token has a probability of
about 0.6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="directory to write the model to")
    parser.add_argument(
        "--sharpness", type=float, default=50.0, help="scale of the embedding and the output head"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    from safetensors.torch import load_file, save_file
    from tests.rollout.tiny_mtp import build_tiny_mtp

    out = build_tiny_mtp(args.out, seed=args.seed)
    weights_file = out / "model.safetensors"
    weights = load_file(str(weights_file))
    for key in weights:
        if key.endswith("embed_tokens.weight") or key == "lm_head.weight":
            weights[key] = weights[key] * args.sharpness
    save_file(weights, str(weights_file), metadata={"format": "pt"})
    mtp_keys = sum(key.startswith("mtp.") for key in weights)
    print(f"wrote {out}: {len(weights) - mtp_keys} policy tensors, {mtp_keys} MTP tensors")


if __name__ == "__main__":
    main()
