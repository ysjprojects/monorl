"""System, config, recipes, connectors and environment discovery routes."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Request

from monorl.config.loader import load_yaml
from monorl.config.pipeline import PipelineConfig, is_pipeline_payload
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, RegistryError
from monorl.server.app import validation_errors

router = APIRouter()

S3_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
    "S3_ENDPOINT_URL",
)


@router.get("/system")
def system() -> dict[str, Any]:
    from monorl.doctor import detect

    return detect().to_dict()


@router.get("/config/schema")
def config_schema() -> dict[str, Any]:
    return RunConfig.json_schema()


@router.get("/config/pipeline_schema")
def pipeline_schema() -> dict[str, Any]:
    return PipelineConfig.json_schema()


@router.post("/config/validate")
def config_validate(body: dict[str, Any]) -> dict[str, Any]:
    """``{config: RunConfig}`` → ``{ok, errors, config}``; ``{pipeline: PipelineConfig}`` → ``{ok, errors, pipeline}``."""
    if isinstance(body, dict) and "pipeline" in body:
        payload: Any = body["pipeline"]
        key = "pipeline"
    else:
        payload = body.get("config", body) if isinstance(body, dict) and "config" in body else body
        key = "pipeline" if is_pipeline_payload(payload) else "config"
    if not isinstance(payload, dict):
        return {
            "ok": False,
            "errors": [{"loc": [key], "msg": f"{key} must be an object", "type": "type_error"}],
        }
    try:
        if key == "pipeline":
            return {"ok": True, "errors": [], "pipeline": PipelineConfig.from_dict(payload).to_dict()}
        return {"ok": True, "errors": [], "config": RunConfig.from_dict(payload).to_dict()}
    except ConfigError as e:
        return {"ok": False, "errors": validation_errors(e)}


def recipe_description(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#!"):
            continue
        if line.startswith("#"):
            lines.append(line[1:].strip())
            continue
        if line == "" and not lines:
            continue
        break
    return " ".join(part for part in lines if part).strip()


def load_recipes(recipes_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not recipes_dir.is_dir():
        return out
    for path in sorted(recipes_dir.glob("*.yaml")) + sorted(recipes_dir.glob("*.yml")):
        text = path.read_text()
        try:
            config: Any = load_yaml(path)
        except ConfigError:
            try:
                config = yaml.safe_load(text)
            except yaml.YAMLError:
                continue
        if not isinstance(config, dict):
            continue
        kind = "pipeline" if is_pipeline_payload(config) else "run"
        valid, errors = True, []
        try:
            if kind == "pipeline":
                PipelineConfig.from_dict(config)
            else:
                RunConfig.from_dict(config)
        except ConfigError as e:
            valid, errors = False, validation_errors(e)
        out.append(
            {
                "name": path.stem,
                "kind": kind,
                "description": recipe_description(text),
                "config": config,
                "path": str(path),
                "valid": valid,
                "errors": errors,
            }
        )
    return out


@router.get("/recipes")
def recipes(request: Request) -> list[dict[str, Any]]:
    return load_recipes(request.app.state.settings.recipes_dir)


@router.get("/connectors")
def connectors() -> dict[str, Any]:
    present = [v for v in S3_VARS if os.environ.get(v)]
    has_keys = bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
    has_profile = bool(os.environ.get("AWS_PROFILE"))
    shared_credentials = Path(
        os.environ.get("AWS_SHARED_CREDENTIALS_FILE") or "~/.aws/credentials"
    ).expanduser()
    return {
        "s3": {
            "configured": has_keys or has_profile or shared_credentials.is_file(),
            "region": os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION"),
            "endpoint_url": os.environ.get("AWS_ENDPOINT_URL_S3")
            or os.environ.get("AWS_ENDPOINT_URL")
            or os.environ.get("S3_ENDPOINT_URL"),
            "profile": os.environ.get("AWS_PROFILE"),
            "vars_present": present,
            "shared_credentials_file": str(shared_credentials) if shared_credentials.is_file() else None,
        },
        "hf": {
            "configured": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            "vars_present": [
                v
                for v in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HOME", "HF_ENDPOINT")
                if os.environ.get(v)
            ],
        },
    }


def _is_executable(path: Path) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and bool(st.st_mode & stat.S_IXUSR)


def _language_for(path: Path, sdk_root: Path) -> str:
    name = path.name.lower()
    if name.endswith("-go") or name.startswith("go-"):
        return "go"
    if name.endswith("-cpp") or name.startswith("cpp-"):
        return "cpp"
    for lang in ("go", "cpp"):
        marker = sdk_root / lang
        if marker.is_dir() and path.is_relative_to(marker):
            return lang
    return "unknown"


def discover_env_examples(envs_bin: Path, envs_sdk: Path) -> list[dict[str, Any]]:
    """MEP/1 example servers: executables under ``envs/bin`` (built by the SDK makefiles) plus
    example sources under ``envs/sdk/{go,cpp}/examples`` that have been compiled in place."""
    seen: dict[str, dict[str, Any]] = {}
    candidates: list[Path] = []
    if envs_bin.is_dir():
        candidates.extend(sorted(envs_bin.iterdir()))
    for lang in ("go", "cpp"):
        examples = envs_sdk / lang / "examples"
        if examples.is_dir():
            for sub in sorted(examples.iterdir()):
                if sub.is_dir():
                    candidates.extend(sorted(p for p in sub.iterdir() if _is_executable(p)))
    for path in candidates:
        if not _is_executable(path) or path.name in seen:
            continue
        seen[path.name] = {
            "name": path.name,
            "language": _language_for(path, envs_sdk),
            "path": str(path),
            "transport": "stdio",
            "command": [str(path)],
        }
    return list(seen.values())


@router.get("/envs")
def envs(request: Request) -> dict[str, Any]:
    from monorl.adapters import PROVIDERS
    from monorl.core.registry import registry

    # Load only the envs provider: the other builtin providers pull in torch.
    try:
        registry.load_provider("monorl.envs")
    except RegistryError:
        pass
    types = set(registry.names("env", load_builtins=False))
    types.update(name for name in ("openenv", "hermes") if name in PROVIDERS)
    settings = request.app.state.settings
    return {
        "types": sorted(types),
        "examples": discover_env_examples(settings.envs_bin, settings.envs_sdk),
    }
