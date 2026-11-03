#!/usr/bin/env python3
"""Clone pinned upstream frameworks into ``third_party/<name>`` for reading / debugging.

    scripts/vendor.py --list
    scripts/vendor.py --fetch vllm
    scripts/vendor.py --fetch all [--force]

Pins come from ``third_party/pins.toml``. Clones are shallow (``--depth 1`` at the pinned tag,
then verified against the pinned commit). ``third_party/*/`` is gitignored.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
PINS_FILE = THIRD_PARTY / "pins.toml"


def load_pins(path: Path = PINS_FILE) -> dict[str, dict[str, str]]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def list_pins(pins: dict[str, dict[str, str]]) -> None:
    width = max(len(n) for n in pins)
    print(f"{'name':<{width}}  {'version':<12} {'tag':<16} {'commit':<12} status")
    for name, pin in pins.items():
        dest = THIRD_PARTY / name
        status = f"vendored ({_head(dest)[:12]})" if (dest / ".git").exists() else "-"
        print(f"{name:<{width}}  {pin['version']:<12} {pin['tag']:<16} {pin['commit'][:12]} {status}")


def _head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def fetch(name: str, pin: dict[str, str], *, force: bool = False) -> Path:
    dest = THIRD_PARTY / name
    if dest.exists():
        if not force and _head(dest) == pin["commit"]:
            print(f"{name}: already at {pin['commit'][:12]}")
            return dest
        shutil.rmtree(dest)
    THIRD_PARTY.mkdir(exist_ok=True)
    print(f"{name}: cloning {pin['git']} @ {pin['tag']} -> {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", pin["tag"], pin["git"], str(dest)],
        check=True,
    )
    head = _head(dest)
    if head != pin["commit"]:
        # Annotated tags resolve to the tag object; compare against the peeled commit too.
        peeled = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", f"{pin['tag']}^{{commit}}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if peeled != pin["commit"]:
            raise SystemExit(
                f"{name}: tag {pin['tag']} resolved to {head}, but pins.toml says {pin['commit']} - "
                "upstream moved the tag; update pins.toml deliberately."
            )
    print(f"{name}: ok at {pin['commit'][:12]}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Show the pins and what is vendored.")
    group.add_argument("--fetch", metavar="NAME", help="Clone NAME (or 'all') at its pin.")
    parser.add_argument("--force", action="store_true", help="Re-clone even if already present.")
    args = parser.parse_args(argv)

    pins = load_pins()
    if args.list:
        list_pins(pins)
        return 0
    names = list(pins) if args.fetch == "all" else [args.fetch]
    unknown = [n for n in names if n not in pins]
    if unknown:
        parser.error(f"unknown pin(s) {unknown}; known: {sorted(pins)}")
    for name in names:
        fetch(name, pins[name], force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
