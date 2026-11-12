"""Hugging Face Hub search + dataset browsing/preview routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()

_HUB_TIMEOUT = 20.0


def _hf_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _hub_error(e: Exception) -> HTTPException:
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status in (401, 403):
        return HTTPException(status_code=403, detail=f"Hub access denied (gated/private?): {e}")
    if status == 404:
        return HTTPException(status_code=404, detail=str(e))
    return HTTPException(status_code=502, detail=f"Hugging Face Hub request failed: {e}")


@router.get("/models/search")
def models_search(
    q: str = Query("", alias="q"), limit: int = Query(20, ge=1, le=100)
) -> list[dict[str, Any]]:
    api = _hf_api()
    try:
        models = list(
            api.list_models(search=q or None, pipeline_tag="text-generation", sort="downloads", limit=limit)
        )
    except Exception as e:  # noqa: BLE001 - any network/HTTP failure maps to a 5xx/4xx detail
        raise _hub_error(e) from e
    return [
        {
            "id": m.id,
            "downloads": m.downloads,
            "likes": m.likes,
            "pipeline_tag": m.pipeline_tag,
            "tags": list(m.tags or []),
            "last_modified": _iso(m.last_modified),
            "gated": m.gated,
            "private": m.private,
            "library_name": getattr(m, "library_name", None),
        }
        for m in models
    ]


def _architecture(info: Any) -> tuple[str | None, str | None]:
    config = getattr(info, "config", None) or {}
    arch = None
    archs = config.get("architectures") if isinstance(config, dict) else None
    if isinstance(archs, list) and archs:
        arch = str(archs[0])
    model_type = config.get("model_type") if isinstance(config, dict) else None
    return arch, model_type


@router.get("/models/info")
def models_info(id: str = Query(..., min_length=1)) -> dict[str, Any]:
    api = _hf_api()
    try:
        info = api.model_info(id, timeout=_HUB_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        raise _hub_error(e) from e
    arch, model_type = _architecture(info)
    safetensors = getattr(info, "safetensors", None)
    params = None
    dtypes: dict[str, int] = {}
    if safetensors is not None:
        total = (
            safetensors.get("total") if isinstance(safetensors, dict) else getattr(safetensors, "total", None)
        )
        params = int(total) if total else None
        raw = (
            safetensors.get("parameters")
            if isinstance(safetensors, dict)
            else getattr(safetensors, "parameters", None)
        )
        dtypes = {str(k): int(v) for k, v in (raw or {}).items()}
    card = getattr(info, "card_data", None)
    siblings = getattr(info, "siblings", None) or []
    files = [getattr(s, "rfilename", None) or str(s) for s in siblings]
    return {
        "id": info.id,
        "architecture": arch,
        "model_type": model_type,
        "params": params,
        "param_dtypes": dtypes,
        "gated": bool(getattr(info, "gated", False) or False),
        "gated_mode": getattr(info, "gated", None) if isinstance(getattr(info, "gated", None), str) else None,
        "private": bool(getattr(info, "private", False) or False),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "library_name": getattr(info, "library_name", None),
        "tags": list(getattr(info, "tags", None) or []),
        "license": _card_field(card, "license"),
        "base_model": _card_field(card, "base_model"),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "last_modified": _iso(getattr(info, "last_modified", None)),
        "sha": getattr(info, "sha", None),
        "has_safetensors": any(f.endswith(".safetensors") for f in files),
        "has_chat_template": any(f in ("chat_template.jinja", "chat_template.json") for f in files)
        or "chat_template" in str(getattr(info, "config", None) or {}),
        "files": files,
    }


def _card_field(card: Any, key: str) -> Any:
    if card is None:
        return None
    if isinstance(card, dict):
        return card.get(key)
    getter = getattr(card, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:  # noqa: BLE001
            return None
    return getattr(card, key, None)


@router.get("/datasets/search")
def datasets_search(
    q: str = Query("", alias="q"), limit: int = Query(20, ge=1, le=100)
) -> list[dict[str, Any]]:
    api = _hf_api()
    try:
        datasets = list(api.list_datasets(search=q or None, sort="downloads", limit=limit))
    except Exception as e:  # noqa: BLE001
        raise _hub_error(e) from e
    return [
        {
            "id": d.id,
            "downloads": d.downloads,
            "likes": d.likes,
            "tags": list(d.tags or []),
            "last_modified": _iso(d.last_modified),
            "gated": d.gated,
            "private": d.private,
            "uri": f"hf://{d.id}",
        }
        for d in datasets
    ]


@router.get("/datasets/browse")
def datasets_browse(uri: str = Query(..., min_length=1)) -> list[dict[str, Any]]:
    from monorl.data.browse import browse_uri

    try:
        return browse_uri(uri)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class PreviewRequest(BaseModel):
    uri: str = Field(..., min_length=1)
    split: str | None = None
    hf_config_name: str | None = None
    limit: int = Field(20, ge=1, le=500)


@router.post("/datasets/preview")
def datasets_preview(body: PreviewRequest) -> dict[str, Any]:
    from monorl.data.browse import preview_uri

    try:
        return preview_uri(body.uri, split=body.split, hf_config_name=body.hf_config_name, limit=body.limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
