"""Field reference of the run config, generated from the pydantic models.

``render_reference()`` returns the Markdown of ``docs/CONFIG.md``. Torch-free and deterministic::

    python -m monorl.config.reference                         # print
    python -m monorl.config.reference --write docs/CONFIG.md  # regenerate
    python -m monorl.config.reference --check docs/CONFIG.md  # exit 1 when stale
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import yaml
from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from monorl.config.pipeline import EvalStage, PipelineConfig, RejectionSampleStage, TrainStage
from monorl.config.schema import DataConfig, RunConfig

REGENERATE_COMMAND = "python -m monorl.config.reference --write docs/CONFIG.md"

# Docs linked under each section heading (file names relative to docs/).
SECTION_DOCS: dict[str, tuple[str, ...]] = {
    "": ("RUNS.md",),
    "model": ("BACKENDS.md", "MTP.md"),
    "data": ("DATA.md",),
    "algorithm": ("ALGORITHMS.md", "CREDIT_ASSIGNMENT.md", "OFF_POLICY.md"),
    "rollout": ("BACKENDS.md", "DISTRIBUTED.md"),
    "trainer": ("RUNS.md", "BACKENDS.md"),
    "rewards": ("REWARDS.md",),
    "env": ("ENVIRONMENTS.md", "GATEWAY.md", "SANDBOX.md", "ENV_SDK.md"),
    "cluster": ("DISTRIBUTED.md",),
    "logging": ("RUNS.md",),
}
PIPELINE_DOCS = ("PIPELINES.md",)
# Free-form mapping fields whose keys are listed in fewer docs than their section links.
MAPPING_DOCS: dict[str, tuple[str, ...]] = {
    "rollout.extra": ("BACKENDS.md",),
    "trainer.extra": ("BACKENDS.md",),
}

# How a model is named when it appears as the type of a field that is not flattened.
_MODEL_LABELS: dict[type[BaseModel], str] = {
    RunConfig: "run config (the sections above)",
    DataConfig: "`data` section",
    TrainStage: "`train` stage",
    RejectionSampleStage: "`rejection_sample` stage",
    EvalStage: "`eval` stage",
}
_SCALAR_NAMES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    Any: "any",
    types.NoneType: "null",
}

_INTRO = f"""\
Every field of a run config (`RunConfig`) and of a pipeline config (`PipelineConfig`), with its type
and default. This file is generated from the schema in `src/monorl/config/`; do not edit it, run
`{REGENERATE_COMMAND}` instead.
`monorl recipes show NAME` prints complete example configs. A config is validated by
`RunConfig.from_dict` and by `POST /api/config/validate`;
rules that span several fields are enforced at validation and described in the docs linked under
each section.

Unknown keys are rejected. A key marked "required" has no default. Bounds on numbers are listed
with the type.
"""


def _is_model(tp: Any) -> bool:
    return isinstance(tp, type) and issubclass(tp, BaseModel)


def _list_item_model(tp: Any) -> type[BaseModel] | None:
    """The model ``M`` when ``tp`` is ``list[M]``."""
    if get_origin(tp) is list:
        (item,) = get_args(tp)
        if _is_model(item):
            return item
    return None


def _is_mapping(tp: Any) -> bool:
    return tp is dict or get_origin(tp) is dict


def _code(text: str) -> str:
    return f"`{text}`"


def _type_text(tp: Any) -> str:
    """A readable name for an annotation: no ``typing.`` noise, literals as code spans."""
    origin = get_origin(tp)
    if origin is Annotated:
        return _type_text(get_args(tp)[0])
    if origin is Literal:
        return ", ".join(_code(_yaml_scalar(v)) for v in get_args(tp))
    if origin in (Union, types.UnionType):
        args = list(get_args(tp))
        # null last, whatever the order in the annotation
        ordered = [a for a in args if a is not types.NoneType] + [a for a in args if a is types.NoneType]
        return " or ".join(_type_text(a) for a in ordered)
    if origin is list:
        (item,) = get_args(tp)
        text = _type_text(item)
        return f"list of ({text})" if " or " in text else f"list of {text}"
    if _is_mapping(tp):
        args = get_args(tp)
        if not args or args[1] is Any:
            return "mapping"
        return f"mapping of {_type_text(args[1])}"
    if _is_model(tp):
        return _MODEL_LABELS.get(tp, tp.__name__)
    if tp in _SCALAR_NAMES:
        return _SCALAR_NAMES[tp]
    return getattr(tp, "__name__", str(tp))


def _constraints(field: FieldInfo) -> str:
    out: list[str] = []
    for meta in field.metadata:
        if isinstance(meta, Ge):
            out.append(f">= {_yaml_scalar(meta.ge)}")
        elif isinstance(meta, Gt):
            out.append(f"> {_yaml_scalar(meta.gt)}")
        elif isinstance(meta, Le):
            out.append(f"<= {_yaml_scalar(meta.le)}")
        elif isinstance(meta, Lt):
            out.append(f"< {_yaml_scalar(meta.lt)}")
        elif isinstance(meta, MinLen):
            out.append(f"at least {meta.min_length} item(s)")
        elif isinstance(meta, MaxLen):
            out.append(f"at most {meta.max_length} item(s)")
    return ", ".join(out)


def _yaml_scalar(value: Any) -> str:
    """``value`` as YAML prints it on one line (``null``, ``true``, ``1.0e-06``, ``[]``, ``{}``)."""
    text = yaml.safe_dump(value, default_flow_style=True, width=10**6).strip()
    return text.removesuffix("...").strip()


def _default_text(field: FieldInfo) -> str:
    if field.default_factory is not None:
        return _code(_yaml_scalar(field.default_factory()))  # type: ignore[call-arg]
    if field.default is PydanticUndefined:
        return "required"
    return _code(_yaml_scalar(field.default))


def _cell(text: str) -> str:
    """Escape text for a Markdown table cell (pipes, raw angle brackets, newlines)."""
    return " ".join(text.split()).replace("|", "\\|").replace("<", "\\<")


def _links(docs: tuple[str, ...]) -> str:
    return ", ".join(f"[{d}]({d})" for d in docs)


def _see(docs: tuple[str, ...]) -> list[str]:
    """The ``See: ...`` paragraph under a heading, wrapped at 100 columns between links."""
    lines = ["See:"]
    for i, doc in enumerate(docs):
        item = f"[{doc}]({doc})" + ("," if i < len(docs) - 1 else "")
        if len(lines[-1]) + 1 + len(item) > 100:
            lines.append(item)
        else:
            lines[-1] = f"{lines[-1]} {item}"
    return lines


def _rows(
    model: type[BaseModel], prefix: str, docs: tuple[str, ...] = (), flatten: bool = True
) -> list[tuple[str, ...]]:
    """One row per field. ``flatten``: a nested model becomes dotted keys instead of a row.

    ``docs``: where the keys of free-form mapping fields are documented (empty: no pointer).
    """
    rows: list[tuple[str, ...]] = []
    for name, field in model.model_fields.items():
        key = f"{prefix}{name}"
        tp = field.annotation
        if flatten and _is_model(tp):
            rows.extend(_rows(tp, f"{key}.", docs))
            continue
        type_text = _type_text(tp)
        bounds = _constraints(field)
        if bounds:
            type_text = f"{type_text}, {bounds}"
        description = _cell(field.description or "")
        if docs and _is_mapping(tp):
            pointer = f"Keys: see {_links(MAPPING_DOCS.get(key, docs))}."
            description = f"{description} {pointer}" if description else pointer
        rows.append((_code(key), type_text, _default_text(field), description))
    return rows


def _table(rows: list[tuple[str, ...]]) -> list[str]:
    """Three columns: type and default share one, which keeps the widest table inside GitHub's page."""
    lines = ["| Key | Type, default | Description |", "| --- | --- | --- |"]
    for key, type_text, default, description in rows:
        default = default if default == "required" else f"default {default}"
        lines.append(f"| {key} | {type_text}; {default} | {description} |")
    return lines


def _section(title: str, docs: tuple[str, ...], rows: list[tuple[str, ...]], note: str = "") -> list[str]:
    lines = [f"## {title}", "", *_see(docs), ""]
    if note:
        lines += [note, ""]
    return lines + _table(rows) + [""]


def render_reference() -> str:
    """The Markdown field reference (the exact content of ``docs/CONFIG.md``)."""
    lines = ["# Config reference", "", _INTRO.rstrip(), ""]

    scalars = [
        (_code(n), _type_text(f.annotation), _default_text(f), _cell(f.description or ""))
        for n, f in RunConfig.model_fields.items()
        if not _is_model(f.annotation) and _list_item_model(f.annotation) is None
    ]
    lines += _section("Top-level fields", SECTION_DOCS[""], scalars)

    for name, field in RunConfig.model_fields.items():
        tp = field.annotation
        docs = SECTION_DOCS.get(name, ())
        item = _list_item_model(tp)
        if _is_model(tp):
            required = field.default is PydanticUndefined and field.default_factory is None
            note = "This section is required." if required else ""
            lines += _section(_code(name), docs, _rows(tp, f"{name}.", docs), note)
        elif item is not None:
            note = f"A list of entries (default {_default_text(field)}); each entry has the keys below."
            if field.description:
                note = f"{note} {_cell(field.description)}"
            lines += _section(_code(name), docs, _rows(item, f"{name}[].", docs), note)

    lines += [
        "## Pipelines",
        "",
        *_see(PIPELINE_DOCS),
        "",
        "A document with both `base` and `stages` is a pipeline config. Each entry of `stages` is one",
        "of the stage types below, selected by its `type` key.",
        "",
        *_table(_rows(PipelineConfig, "", flatten=False)),
        "",
    ]
    for stage in (TrainStage, RejectionSampleStage, EvalStage):
        kind = _yaml_scalar(stage.model_fields["type"].default)
        lines += [f"### Stage `{kind}`", "", *_table(_rows(stage, "stages[].", flatten=False)), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m monorl.config.reference", description="Render the config field reference."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", metavar="PATH", help="write the reference to PATH")
    group.add_argument("--check", metavar="PATH", help="exit 1 when PATH differs from the rendering")
    args = parser.parse_args(argv)
    text = render_reference()
    if args.write:
        Path(args.write).write_text(text, encoding="utf-8")
    elif args.check:
        path = Path(args.check)
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            print(f"{path} is out of date with the config schema; run: {REGENERATE_COMMAND}", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
