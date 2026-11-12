"""Non-job routes: schema, validate, recipes, hub search (monkeypatched HfApi), browse/preview,
connectors, envs, static SPA."""

from __future__ import annotations

import os
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from monorl.server.app import ServerSettings, create_app
from monorl.server.routes_system import recipe_description


def test_server_import_is_torch_free() -> None:
    import subprocess

    code = (
        "import sys, monorl.server, monorl.server.routes_hub, monorl.server.routes_jobs, "
        "monorl.server.routes_system; sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


# -- config ------------------------------------------------------------------------------------


def test_config_schema(client: TestClient) -> None:
    r = client.get("/api/config/schema")
    assert r.status_code == 200
    schema = r.json()
    assert schema["title"] == "RunConfig"
    assert set(schema["properties"]) >= {
        "model",
        "data",
        "algorithm",
        "rollout",
        "trainer",
        "rewards",
        "logging",
    }
    assert "ModelConfig" in schema["$defs"]


def test_config_validate_ok(client: TestClient, config_dict: dict[str, Any]) -> None:
    r = client.post("/api/config/validate", json={"config": config_dict})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["errors"] == []
    assert body["config"]["rollout"]["group_size"] == 2


def test_config_validate_bare_payload(client: TestClient, config_dict: dict[str, Any]) -> None:
    r = client.post("/api/config/validate", json=config_dict)
    assert r.status_code == 200 and r.json()["ok"] is True


def test_config_validate_invalid(client: TestClient, config_dict: dict[str, Any]) -> None:
    bad = dict(config_dict)
    bad["rollout"] = {"group_size": 1}
    bad["model"] = {"name_or_path": "x", "dtype": "int4"}
    r = client.post("/api/config/validate", json={"config": bad})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    locs = [tuple(e["loc"]) for e in body["errors"]]
    assert ("model", "dtype") in locs
    assert all(isinstance(e["msg"], str) and e["msg"] for e in body["errors"])


def test_config_validate_group_size_rule(client: TestClient, config_dict: dict[str, Any]) -> None:
    bad = dict(config_dict)
    bad["rollout"] = {"group_size": 1}
    body = client.post("/api/config/validate", json={"config": bad}).json()
    assert body["ok"] is False
    assert any("group_size" in e["msg"] for e in body["errors"])


def test_config_validate_not_an_object(client: TestClient) -> None:
    r = client.post("/api/config/validate", json={"config": [1, 2]})
    assert r.status_code == 200 and r.json()["ok"] is False


def test_malformed_body_is_422(client: TestClient) -> None:
    r = client.post("/api/config/validate", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert r.json()["ok"] is False and r.json()["errors"]


# -- recipes -----------------------------------------------------------------------------------


def test_recipe_description() -> None:
    text = (
        "#!/usr/bin/env monorl\n# GRPO on GSM8K\n#   with a tiny model\n\nrun_name: x\n# trailing comment\n"
    )
    assert recipe_description(text) == "GRPO on GSM8K with a tiny model"
    assert recipe_description("run_name: x\n") == ""


def test_recipes(client: TestClient, recipes_dir: Path, config_dict: dict[str, Any]) -> None:
    import yaml

    (recipes_dir / "b-recipe.yaml").write_text("# Second recipe.\n" + yaml.safe_dump(config_dict))
    (recipes_dir / "a-recipe.yaml").write_text(
        "# First recipe\n# spans two lines.\n" + yaml.safe_dump(config_dict)
    )
    (recipes_dir / "broken.yaml").write_text("# Broken\nmodel: {name_or_path: x}\n")
    (recipes_dir / "notes.txt").write_text("ignored")
    r = client.get("/api/recipes")
    assert r.status_code == 200
    recipes = r.json()
    assert [x["name"] for x in recipes] == ["a-recipe", "b-recipe", "broken"]
    assert recipes[0]["description"] == "First recipe spans two lines."
    assert recipes[0]["config"]["model"]["name_or_path"] == "org/tiny-model"
    assert recipes[0]["valid"] is True
    assert recipes[2]["valid"] is False and recipes[2]["errors"]


def test_recipes_missing_dir(client: TestClient, recipes_dir: Path) -> None:
    recipes_dir.rmdir()
    assert client.get("/api/recipes").json() == []


# -- hub ---------------------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, id: str, **kw: Any) -> None:
        self.id = id
        self.downloads = kw.get("downloads", 10)
        self.likes = kw.get("likes", 1)
        self.pipeline_tag = kw.get("pipeline_tag", "text-generation")
        self.tags = kw.get("tags", ["pytorch"])
        self.last_modified = kw.get("last_modified", datetime(2025, 1, 2, tzinfo=UTC))
        self.gated = kw.get("gated", False)
        self.private = False
        self.library_name = "transformers"
        self.config = kw.get("config", {"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
        self.safetensors = kw.get("safetensors")
        self.card_data = kw.get("card_data")
        self.siblings = [types.SimpleNamespace(rfilename=f) for f in kw.get("files", ["config.json"])]
        self.sha = "abc"


class _FakeHfApi:
    calls: list[tuple[str, dict[str, Any]]] = []

    def list_models(self, **kw: Any):
        self.calls.append(("list_models", kw))
        yield _FakeModel("org/alpha", downloads=100)
        yield _FakeModel("org/beta", downloads=50, gated="auto")

    def list_datasets(self, **kw: Any):
        self.calls.append(("list_datasets", kw))
        yield types.SimpleNamespace(
            id="org/ds",
            downloads=5,
            likes=0,
            tags=["task:qa"],
            last_modified=None,
            gated=False,
            private=False,
        )

    def model_info(self, repo_id: str, **kw: Any) -> _FakeModel:
        self.calls.append(("model_info", {"repo_id": repo_id, **kw}))
        if repo_id == "missing/model":
            import httpx

            resp = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))
            raise httpx.HTTPStatusError("not found", request=resp.request, response=resp)
        return _FakeModel(
            repo_id,
            gated="manual",
            safetensors={"total": 135_000_000, "parameters": {"BF16": 135_000_000}},
            files=["config.json", "model.safetensors", "chat_template.jinja"],
            card_data={"license": "apache-2.0"},
        )


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> type[_FakeHfApi]:
    import huggingface_hub

    _FakeHfApi.calls = []
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    return _FakeHfApi


def test_models_search(client: TestClient, fake_hub: type[_FakeHfApi]) -> None:
    r = client.get("/api/models/search", params={"q": "smol", "limit": 5})
    assert r.status_code == 200
    rows = r.json()
    assert [m["id"] for m in rows] == ["org/alpha", "org/beta"]
    assert rows[0] == {
        "id": "org/alpha",
        "downloads": 100,
        "likes": 1,
        "pipeline_tag": "text-generation",
        "tags": ["pytorch"],
        "last_modified": "2025-01-02T00:00:00+00:00",
        "gated": False,
        "private": False,
        "library_name": "transformers",
    }
    name, kw = fake_hub.calls[0]
    assert name == "list_models"
    assert kw == {"search": "smol", "pipeline_tag": "text-generation", "sort": "downloads", "limit": 5}


def test_models_search_limit_bounds(client: TestClient, fake_hub: type[_FakeHfApi]) -> None:
    assert client.get("/api/models/search", params={"limit": 0}).status_code == 422


def test_models_info(client: TestClient, fake_hub: type[_FakeHfApi]) -> None:
    r = client.get("/api/models/info", params={"id": "org/alpha"})
    assert r.status_code == 200
    info = r.json()
    assert info["architecture"] == "LlamaForCausalLM"
    assert info["model_type"] == "llama"
    assert info["params"] == 135_000_000
    assert info["param_dtypes"] == {"BF16": 135_000_000}
    assert info["gated"] is True and info["gated_mode"] == "manual"
    assert info["license"] == "apache-2.0"
    assert info["has_safetensors"] is True and info["has_chat_template"] is True


def test_models_info_not_found(client: TestClient, fake_hub: type[_FakeHfApi]) -> None:
    r = client.get("/api/models/info", params={"id": "missing/model"})
    assert r.status_code == 404
    assert "detail" in r.json()


def test_datasets_search(client: TestClient, fake_hub: type[_FakeHfApi]) -> None:
    r = client.get("/api/datasets/search", params={"q": "gsm"})
    assert r.status_code == 200
    assert r.json() == [
        {
            "id": "org/ds",
            "downloads": 5,
            "likes": 0,
            "tags": ["task:qa"],
            "last_modified": None,
            "gated": False,
            "private": False,
            "uri": "hf://org/ds",
        }
    ]
    assert fake_hub.calls[0][1] == {"search": "gsm", "sort": "downloads", "limit": 20}


# -- datasets browse / preview -----------------------------------------------------------------


@pytest.fixture
def fake_data(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from monorl.core.errors import DataError

    seen: dict[str, Any] = {}

    def browse_uri(uri: str) -> list[dict[str, Any]]:
        seen["browse"] = uri
        if uri == "s3://nope":
            raise DataError("bad bucket")
        if uri == "/missing":
            raise FileNotFoundError("/missing")
        return [{"name": "a.jsonl", "uri": f"{uri}/a.jsonl", "is_dir": False, "size": 12}]

    def preview_uri(uri: str, *, split: str | None, hf_config_name: str | None, limit: int) -> dict[str, Any]:
        seen["preview"] = {"uri": uri, "split": split, "hf_config_name": hf_config_name, "limit": limit}
        return {
            "columns": {"prompt": "string", "answer": "string"},
            "rows": [{"prompt": "2+2?", "answer": "4"}][:limit],
            "inferred": {
                "format": "prompt_answer",
                "prompt_field": "prompt",
                "answer_field": "answer",
                "messages_field": None,
            },
            "num_rows": 1,
        }

    mod = types.ModuleType("monorl.data.browse")
    mod.browse_uri = browse_uri  # type: ignore[attr-defined]
    mod.preview_uri = preview_uri  # type: ignore[attr-defined]
    pkg = sys.modules.get("monorl.data") or types.ModuleType("monorl.data")
    monkeypatch.setitem(sys.modules, "monorl.data", pkg)
    monkeypatch.setitem(sys.modules, "monorl.data.browse", mod)
    monkeypatch.setattr(pkg, "browse", mod, raising=False)
    return seen


def test_datasets_browse(client: TestClient, fake_data: dict[str, Any]) -> None:
    r = client.get("/api/datasets/browse", params={"uri": "/data"})
    assert r.status_code == 200
    assert r.json() == [{"name": "a.jsonl", "uri": "/data/a.jsonl", "is_dir": False, "size": 12}]
    assert fake_data["browse"] == "/data"
    assert client.get("/api/datasets/browse", params={"uri": "s3://nope"}).status_code == 400
    assert client.get("/api/datasets/browse", params={"uri": "/missing"}).status_code == 404
    assert client.get("/api/datasets/browse").status_code == 422


def test_datasets_preview(client: TestClient, fake_data: dict[str, Any]) -> None:
    r = client.post("/api/datasets/preview", json={"uri": "hf://org/ds", "split": "test", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["inferred"]["format"] == "prompt_answer"
    assert body["columns"] == {"prompt": "string", "answer": "string"}
    assert fake_data["preview"] == {"uri": "hf://org/ds", "split": "test", "hf_config_name": None, "limit": 5}
    assert client.post("/api/datasets/preview", json={"limit": 5}).status_code == 422


# -- connectors --------------------------------------------------------------------------------


def test_connectors_unconfigured(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in list(os.environ):
        if var.startswith("AWS_") or var in ("S3_ENDPOINT_URL", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            monkeypatch.delenv(var)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "nope"))
    body = client.get("/api/connectors").json()
    assert body["s3"]["configured"] is False
    assert body["s3"]["vars_present"] == []
    assert body["hf"]["configured"] is False


def test_connectors_never_echo_secrets(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIASECRETKEYID")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "supersecretvalue")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    r = client.get("/api/connectors")
    body = r.json()
    assert body["s3"]["configured"] is True
    assert body["s3"]["region"] == "eu-west-1"
    assert body["s3"]["endpoint_url"] == "http://localhost:9000"
    assert set(body["s3"]["vars_present"]) >= {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    }
    assert body["hf"]["configured"] is True
    assert "supersecretvalue" not in r.text and "AKIASECRETKEYID" not in r.text and "hf_secret" not in r.text


# -- envs --------------------------------------------------------------------------------------


def test_envs(client: TestClient, envs_bin: Path, tmp_path: Path) -> None:
    exe = envs_bin / "echo-env-go"
    exe.write_text("#!/bin/sh\necho hi\n")
    exe.chmod(0o755)
    (envs_bin / "README").write_text("not executable")
    example = tmp_path / "envs-sdk" / "cpp" / "examples" / "counter"
    example.mkdir(parents=True)
    bin_cpp = example / "counter_env"
    bin_cpp.write_text("#!/bin/sh\n")
    bin_cpp.chmod(0o755)
    (example / "main.cpp").write_text("int main(){}")
    r = client.get("/api/envs")
    assert r.status_code == 200
    body = r.json()
    assert "single_turn" in body["types"]
    assert {"openenv", "hermes"} <= set(body["types"])
    assert body["examples"] == [
        {
            "name": "echo-env-go",
            "language": "go",
            "path": str(exe),
            "transport": "stdio",
            "command": [str(exe)],
        },
        {
            "name": "counter_env",
            "language": "cpp",
            "path": str(bin_cpp),
            "transport": "stdio",
            "command": [str(bin_cpp)],
        },
    ]


# -- errors / static ---------------------------------------------------------------------------


def test_unknown_job_404(client: TestClient) -> None:
    for path in (
        "/api/jobs/nope",
        "/api/jobs/nope/metrics",
        "/api/jobs/nope/logs",
        "/api/jobs/nope/checkpoints",
    ):
        r = client.get(path)
        assert r.status_code == 404, path
        assert r.json() == {"detail": "unknown job 'nope'"}
    assert client.post("/api/jobs/nope/cancel").status_code == 404
    assert client.delete("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/metrics/stream").status_code == 404


def test_cors_for_vite(client: TestClient) -> None:
    r = client.options(
        "/api/config/schema",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_static_spa_fallback(settings: ServerSettings, tmp_path: Path) -> None:
    dist = tmp_path / "web-dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>monorl ui</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("nope")
    with TestClient(create_app(settings)) as c:
        assert c.get("/").text == "<html>monorl ui</html>"
        assert c.get("/jobs/abc").text == "<html>monorl ui</html>"
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/../secret.txt").text == "<html>monorl ui</html>"
        assert c.get("/api/does-not-exist").status_code == 404
        assert c.get("/api/config/schema").status_code == 200


def test_no_static_without_dist(client: TestClient) -> None:
    assert client.get("/").status_code == 404
