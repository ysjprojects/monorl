"""The documentation stays navigable: relative links and anchors resolve, ``docs/README.md`` lists
every document, and the README lists every recipe."""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: Not ours (environments, vendored upstream sources, build output).
PRUNED = {".git", ".venv", ".venv-unsloth", "node_modules", "third_party", "dist", "__pycache__"}

FENCE = re.compile(r"^(```|~~~)")
LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")


def markdown_files() -> list[Path]:
    found: list[Path] = []
    for directory, subdirs, files in os.walk(ROOT):
        subdirs[:] = [d for d in subdirs if d not in PRUNED and not d.startswith(".pytest")]
        found.extend(Path(directory) / name for name in files if name.endswith(".md"))
    return sorted(found)


def prose_lines(path: Path) -> list[str]:
    """The lines of ``path`` outside fenced code blocks."""
    lines: list[str] = []
    fenced = False
    for line in path.read_text().splitlines():
        if FENCE.match(line.strip()):
            fenced = not fenced
            continue
        if not fenced:
            lines.append(line)
    return lines


def links(path: Path) -> list[str]:
    """Link targets of ``path``, without those inside code blocks and code spans."""
    targets: list[str] = []
    for line in prose_lines(path):
        targets.extend(LINK.findall(re.sub(r"`[^`]*`", "``", line)))
    return targets


def anchors(path: Path) -> set[str]:
    """The heading anchors GitHub generates for ``path`` (github-slugger's rule)."""
    seen: dict[str, int] = {}
    out: set[str] = set()
    for line in prose_lines(path):
        match = HEADING.match(line)
        if match is None:
            continue
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", match.group(1))  # a link contributes its text
        slug = re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        out.add(slug if count == 0 else f"{slug}-{count}")
    return out


def test_relative_links_and_anchors_resolve() -> None:
    broken: list[str] = []
    for path in markdown_files():
        for target in links(path):
            if re.match(r"[a-z][a-z0-9+.-]*:", target):  # http:, https:, mailto:
                continue
            file_part, _, anchor = target.partition("#")
            destination = (path.parent / file_part).resolve() if file_part else path
            where = f"{path.relative_to(ROOT)}: ({target})"
            if not destination.exists():
                broken.append(f"{where} does not exist")
            elif anchor and destination.suffix == ".md" and anchor not in anchors(destination):
                broken.append(f"{where} has no such heading")
    assert not broken, "broken links:\n  " + "\n  ".join(broken)


def test_the_docs_index_lists_every_document() -> None:
    index = ROOT / "docs" / "README.md"
    listed = {(index.parent / target.partition("#")[0]).resolve() for target in links(index)}
    missing = sorted(p.name for p in (ROOT / "docs").glob("*.md") if p != index and p.resolve() not in listed)
    assert not missing, f"docs/README.md does not link {missing}"


def test_the_readme_lists_every_recipe() -> None:
    rows = re.findall(r"(?m)^\| `([a-z0-9_]+)` \|", (ROOT / "README.md").read_text())
    recipes = {p.stem for p in (ROOT / "configs" / "recipes").glob("*.yaml")}
    assert recipes - set(rows) == set(), (
        f"recipes missing from the README tables: {sorted(recipes - set(rows))}"
    )
    assert set(rows) - recipes == set(), (
        f"the README lists recipes that do not exist: {sorted(set(rows) - recipes)}"
    )
