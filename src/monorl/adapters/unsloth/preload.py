"""Import Unsloth before anything else (``unsloth/_gpu_init.py`` warns when ``trl``, ``transformers``
or ``peft`` are already in ``sys.modules`` and skips part of its import-time patching).

Two ways to use it, both guaranteeing ``import unsloth`` precedes monorl's heavy imports:

* ``import monorl.adapters.unsloth.preload`` as the first line of a script;
* ``python -m monorl.adapters.unsloth.preload monorl.cli train --config run.yaml`` — imports
  unsloth, then runs the named module as ``__main__`` with the remaining arguments.

The import is skipped (``IMPORTED = False``) when unsloth is not installed or CUDA is absent, so
the module is harmless on CPU hosts. ``MONORL_DISABLE_PATCHES`` does not affect it: this is
Unsloth's own import, not a monorl patch.
"""

from __future__ import annotations

import importlib.util
import runpy
import sys

#: ``trl``/``transformers``/``peft`` members that were already imported when this module ran.
PRE_IMPORTED: tuple[str, ...] = tuple(m for m in ("trl", "transformers", "peft") if m in sys.modules)


def _import_first() -> bool:
    if importlib.util.find_spec("unsloth") is None:
        return False
    import torch

    if not torch.cuda.is_available():
        return False
    import unsloth  # noqa: F401

    return True


IMPORTED: bool = _import_first()


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            f"unsloth preloaded: {IMPORTED} (pre-imported: {list(PRE_IMPORTED)})\n"
            "usage: python -m monorl.adapters.unsloth.preload <module> [args...]",
            file=sys.stderr,
        )
        raise SystemExit(0 if IMPORTED else 1)
    module, rest = argv[0], argv[1:]
    sys.argv = [module, *rest]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
