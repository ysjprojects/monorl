from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path

import pytest

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.core.types import message_media, message_text, messages_have_media
from monorl.data import open_source, sniff_format
from monorl.data.browse import preview_uri
from monorl.data.formats import infer_columns
from monorl.data.media import MediaResolver, media_cache_dir, normalize_content_parts, place_images
from tests.data.conftest import write_jsonl

pytest.importorskip("PIL")


def _png(color: tuple[int, int, int], size: tuple[int, int] = (8, 6)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(color: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (5, 5), color).save(buf, format="JPEG")
    return buf.getvalue()


RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


@pytest.fixture(autouse=True)
def media_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("MONORL_HOME", str(home))
    return home


def _open(path: Path, **kw):
    kw.setdefault("shuffle", False)
    return open_source(DataConfig(uri=str(path), **kw))


def _cached(data: bytes, ext: str) -> str:
    return str(media_cache_dir() / f"{hashlib.sha256(data).hexdigest()}.{ext}")


# -- HF Image feature ---------------------------------------------------------------------------


@pytest.fixture
def image_parquet(tmp_path: Path) -> Path:
    from datasets import Dataset, Image
    from PIL import Image as PILImage

    pngs = [_png(RED), _png(GREEN), _png(BLUE)]
    ds = Dataset.from_dict(
        {
            "image": [PILImage.open(io.BytesIO(b)) for b in pngs],
            "question": [f"What colour is this? ({i})" for i in range(3)],
            "answer": ["red", "green", "blue"],
        }
    ).cast_column("image", Image())
    path = tmp_path / "vqa.parquet"
    ds.to_parquet(path)
    return path


def test_parquet_hf_image_feature_becomes_cached_image_part(image_parquet: Path, media_home: Path) -> None:
    src = _open(image_parquet, image_field="image")
    assert len(src) == 3
    s = src[1]
    assert s.messages[-1]["role"] == "user"
    parts = s.messages[-1]["content"]
    assert [p["type"] for p in parts] == ["image", "text"]
    assert parts[1]["text"] == "What colour is this? (1)"
    ref = parts[0]["image"]
    assert ref.startswith(str(media_home / "cache" / "media")) and ref.endswith(".png")
    assert Path(ref).is_file()
    assert message_text(s.messages[-1]) == "What colour is this? (1)"
    assert message_media(s.messages[-1]) == [ref]
    assert messages_have_media(s.messages)
    assert s.reference == "green"
    assert s.metadata["response"] == "green"
    assert "image" not in s.metadata["row"]
    json.dumps(s.messages)  # samples stay JSON serialisable

    from PIL import Image

    with Image.open(ref) as img:
        assert img.size == (8, 6) and img.getpixel((0, 0)) == GREEN


def test_hf_image_feature_via_datasets_loader_pil_values(
    image_parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows coming from ``datasets`` carry decoded PIL images (HubRows); they are cached too."""
    from datasets import load_dataset

    from monorl.data import rows as rows_mod

    ds = load_dataset("parquet", data_files=str(image_parquet), split="train")
    assert type(ds[0]["image"]).__module__.startswith("PIL.")

    def fake_load_dataset(repo_id, name=None, split=None, streaming=False, **kw):
        return ds

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    src = open_source(DataConfig(uri="hf://org/vqa", image_field="image", shuffle=False))
    parts = src[2].messages[-1]["content"]
    assert parts[0]["type"] == "image" and Path(parts[0]["image"]).is_file()
    assert src.rows.media_base("hf://org/vqa#train#0") == "hf://datasets/org/vqa"
    assert rows_mod.HubRows is type(src.rows)


# -- jsonl: paths, base64, data URLs, multi-image ----------------------------------------------


@pytest.fixture
def image_jsonl(tmp_path: Path) -> Path:
    img_dir = tmp_path / "data" / "imgs"
    img_dir.mkdir(parents=True)
    (img_dir / "a.png").write_bytes(_png(RED))
    (img_dir / "b.jpg").write_bytes(_jpeg(BLUE))
    rows = [
        {"id": "rel", "prompt": "relative path", "image": "imgs/a.png", "answer": "r"},
        {"id": "abs", "prompt": "absolute path", "image": str(img_dir / "b.jpg"), "answer": "b"},
        {
            "id": "b64",
            "prompt": "bare base64",
            "image": base64.b64encode(_png(GREEN)).decode(),
            "answer": "g",
        },
        {
            "id": "data",
            "prompt": "data url",
            "image": "data:image/png;base64," + base64.b64encode(_png(GREEN)).decode(),
            "answer": "g",
        },
        {"id": "http", "prompt": "remote", "image": "https://example.com/x.png", "answer": "?"},
        {"id": "none", "prompt": "no image", "image": None, "answer": "-"},
    ]
    return write_jsonl(tmp_path / "data" / "rows.jsonl", rows)


def test_jsonl_path_base64_and_url_images(image_jsonl: Path, tmp_path: Path) -> None:
    src = _open(image_jsonl, image_field="image")
    by_id = {s.id: s for s in src}
    img_dir = tmp_path / "data" / "imgs"
    # relative paths resolve against the data file's directory
    assert by_id["rel"].messages[-1]["content"][0] == {"type": "image", "image": str(img_dir / "a.png")}
    assert by_id["abs"].messages[-1]["content"][0] == {"type": "image", "image": str(img_dir / "b.jpg")}
    green = _cached(_png(GREEN), "png")
    assert by_id["b64"].messages[-1]["content"][0] == {"type": "image", "image": green}
    assert by_id["data"].messages[-1]["content"][0] == {"type": "image", "image": green}
    assert by_id["http"].messages[-1]["content"][0] == {"type": "image", "image": "https://example.com/x.png"}
    assert by_id["none"].messages[-1]["content"] == "no image"
    assert by_id["b64"].metadata["response"] == "g"


def test_relative_path_resolves_against_data_file_dir(tmp_path: Path) -> None:
    d = tmp_path / "ds"
    (d / "pics").mkdir(parents=True)
    (d / "pics" / "p.png").write_bytes(_png(RED))
    path = write_jsonl(d / "rows.jsonl", [{"prompt": "q", "image": "pics/p.png"}])
    src = _open(path, image_field="image")
    assert src[0].messages[-1]["content"][0]["image"] == str(d / "pics" / "p.png")
    missing = write_jsonl(d / "bad.jsonl", [{"prompt": "q", "image": "pics/nope.png"}])
    with pytest.raises(DataError, match="image file not found"):
        _open(missing, image_field="image")


def test_multi_image_rows_list_column_and_several_columns(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "compare",
            "images": [base64.b64encode(_png(RED)).decode(), base64.b64encode(_png(BLUE)).decode()],
            "extra": base64.b64encode(_png(GREEN)).decode(),
        }
    ]
    path = write_jsonl(tmp_path / "multi.jsonl", rows)
    src = _open(path, image_field=["images", "extra"])
    parts = src[0].messages[-1]["content"]
    assert [p["type"] for p in parts] == ["image", "image", "image", "text"]
    assert [p["image"] for p in parts[:3]] == [
        _cached(_png(RED), "png"),
        _cached(_png(BLUE), "png"),
        _cached(_png(GREEN), "png"),
    ]
    assert parts[3] == {"type": "text", "text": "compare"}
    assert set(src[0].metadata["row"]) == set()


def test_image_placeholder_in_prompt_template(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "which is bigger?",
            "a": base64.b64encode(_png(RED)).decode(),
            "b": base64.b64encode(_png(BLUE)).decode(),
        }
    ]
    path = write_jsonl(tmp_path / "ph.jsonl", rows)
    src = _open(path, image_field=["a", "b"], prompt_template="First: <image> Second: <image>\n{prompt}")
    parts = src[0].messages[-1]["content"]
    assert parts == [
        {"type": "text", "text": "First: "},
        {"type": "image", "image": _cached(_png(RED), "png")},
        {"type": "text", "text": " Second: "},
        {"type": "image", "image": _cached(_png(BLUE), "png")},
        {"type": "text", "text": "\nwhich is bigger?"},
    ]
    # fewer placeholders than images: the extra image goes first
    src = _open(path, image_field=["a", "b"], prompt_template="{prompt} <image>")
    parts = src[0].messages[-1]["content"]
    assert [p["type"] for p in parts] == ["image", "text", "image"]
    assert parts[0]["image"] == _cached(_png(BLUE), "png") and parts[2]["image"] == _cached(_png(RED), "png")


def test_place_images_without_placeholder_puts_images_first() -> None:
    img = {"type": "image", "image": "/x.png"}
    assert place_images("hello", [img]) == [img, {"type": "text", "text": "hello"}]
    assert place_images("", [img]) == [img]
    assert place_images("a<image>b", []) == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


def test_cache_dedupes_by_content_hash(tmp_path: Path, media_home: Path) -> None:
    red = base64.b64encode(_png(RED)).decode()
    rows = [{"prompt": f"q{i}", "image": red} for i in range(4)] + [
        {"prompt": "q", "image": base64.b64encode(_jpeg(RED)).decode()}
    ]
    path = write_jsonl(tmp_path / "dup.jsonl", rows)
    src = _open(path, image_field="image")
    refs = [s.messages[-1]["content"][0]["image"] for s in src]
    assert len(set(refs[:4])) == 1 and refs[4] != refs[0] and refs[4].endswith(".jpg")
    files = sorted(p.name for p in (media_home / "cache" / "media").iterdir())
    assert len(files) == 2 and not any(f.startswith(".tmp") for f in files)
    # a second source re-using the same bytes writes nothing new
    before = {p: p.stat().st_mtime_ns for p in (media_home / "cache" / "media").iterdir()}
    _open(path, image_field="image")
    assert {p: p.stat().st_mtime_ns for p in (media_home / "cache" / "media").iterdir()} == before


def test_streaming_source_materialises_lazily(image_jsonl: Path, media_home: Path) -> None:
    src = _open(image_jsonl, image_field="image", streaming=True)
    assert not (media_home / "cache" / "media").exists()
    it = iter(src)
    first = next(it)
    assert first.id == "rel"
    assert not (media_home / "cache" / "media").exists()  # a path reference needs no copy
    next(it)
    third = next(it)
    assert third.id == "b64" and Path(third.messages[-1]["content"][0]["image"]).is_file()


def test_image_field_must_exist(image_jsonl: Path) -> None:
    with pytest.raises(DataError, match="column 'nope' not found"):
        _open(image_jsonl, image_field="nope")


def test_image_bytes_in_a_jsonl_row_require_user_turn(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "sys.jsonl",
        [
            {
                "messages": [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}],
                "image": base64.b64encode(_png(RED)).decode(),
            }
        ],
    )
    with pytest.raises(DataError, match="no user turn"):
        _open(path, image_field="image")


# -- OpenAI-style chat parts ----------------------------------------------------------------------


def test_chat_rows_with_openai_parts_are_normalised(tmp_path: Path) -> None:
    img_path = tmp_path / "c.png"
    img_path.write_bytes(_png(BLUE))
    data_url = "data:image/png;base64," + base64.b64encode(_png(RED)).decode()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look: "},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                        {"type": "image", "image": "c.png"},
                        "and a bare string part",
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I see "}, {"type": "text", "text": "two images."}],
                },
            ]
        },
        {"messages": [{"role": "user", "content": "plain"}, {"role": "assistant", "content": "ok"}]},
    ]
    path = write_jsonl(tmp_path / "chat.jsonl", rows)
    src = _open(path)
    s = src[0]
    assert s.messages[0] == {"role": "system", "content": "sys"}
    assert s.messages[1]["content"] == [
        {"type": "text", "text": "Look: "},
        {"type": "image", "image": _cached(_png(RED), "png"), "detail": "low"},
        {"type": "image", "image": str(img_path)},
        {"type": "text", "text": "and a bare string part"},
    ]
    assert s.reference == "I see two images."
    assert s.metadata["response"] == "I see two images."
    assert s.metadata["assistant_turn"]["content"][0]["type"] == "text"
    assert src[1].messages == [{"role": "user", "content": "plain"}]
    assert src[1].metadata["response"] == "ok"


def test_chat_parts_with_prompt_template_and_extra_image_field(tmp_path: Path) -> None:
    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": "https://h/x.png"},
                        {"type": "text", "text": "q"},
                    ],
                }
            ],
            "shot": base64.b64encode(_png(GREEN)).decode(),
        }
    ]
    path = write_jsonl(tmp_path / "chat2.jsonl", rows)
    src = _open(path, image_field="shot", prompt_template="{prompt}? <image>")
    assert src[0].messages[-1]["content"] == [
        {"type": "image", "image": "https://h/x.png"},
        {"type": "text", "text": "q? "},
        {"type": "image", "image": _cached(_png(GREEN), "png")},
    ]


def test_normalize_content_parts_rejects_unknown_types() -> None:
    with pytest.raises(DataError, match="unsupported content part type"):
        normalize_content_parts([{"type": "sticker", "id": 1}], MediaResolver(), where="r")
    assert normalize_content_parts("text", MediaResolver(), where="r") == "text"
    assert normalize_content_parts(None, MediaResolver(), where="r") == ""
    assert normalize_content_parts(
        [{"type": "video", "video_url": {"url": "s3://b/v.mp4"}}], MediaResolver(), where="r"
    ) == [{"type": "video", "video": "s3://b/v.mp4"}]


# -- MediaResolver edge cases -------------------------------------------------------------------


def test_resolver_handles_image_struct_pil_bytes_and_file_uri(tmp_path: Path) -> None:
    from PIL import Image

    r = MediaResolver(base=str(tmp_path))
    png = _png(RED)
    assert r.image_refs({"bytes": png, "path": "ignored.png"}, where="w") == [_cached(png, "png")]
    assert r.image_refs(png, where="w") == [_cached(png, "png")]
    assert r.image_refs(Image.open(io.BytesIO(png)), where="w") == [_cached(png, "png")]
    f = tmp_path / "f.png"
    f.write_bytes(png)
    assert r.image_refs(f"file://{f}", where="w") == [str(f)]
    assert r.image_refs({"bytes": None, "path": "f.png"}, where="w") == [str(f)]
    assert r.image_refs([None, [png, "f.png"]], where="w") == [_cached(png, "png"), str(f)]
    with pytest.raises(DataError, match="neither bytes nor path"):
        r.image_refs({"bytes": None, "path": None}, where="w")
    with pytest.raises(DataError, match="unsupported image value"):
        r.image_refs(3.5, where="w")
    with pytest.raises(DataError, match="not a decodable image"):
        r.image_refs(b"definitely not an image at all", where="w")


def test_resolver_reencodes_unknown_containers_as_png() -> None:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (3, 3), RED).save(buf, format="TIFF")
    [ref] = MediaResolver().image_refs(buf.getvalue(), where="w")
    assert ref.endswith(".png")
    with Image.open(ref) as img:
        assert img.size == (3, 3)


def test_resolver_fetches_remote_objects_via_fsspec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fsspec
    from fsspec.implementations.memory import MemoryFileSystem

    mem = MemoryFileSystem()
    mem.store.clear()
    mem.pipe_file("/bucket/imgs/r.png", _png(RED))
    monkeypatch.setattr(
        fsspec,
        "filesystem",
        lambda scheme, **kw: mem if scheme == "memory" else fsspec.get_filesystem_class(scheme)(**kw),
    )
    assert MediaResolver().image_refs("memory://bucket/imgs/r.png", where="w") == [_cached(_png(RED), "png")]
    # relative path against a remote base
    assert MediaResolver(base="memory://bucket/imgs").image_refs("r.png", where="w") == [
        _cached(_png(RED), "png")
    ]
    with pytest.raises(DataError, match="image not found"):
        MediaResolver().image_refs("memory://bucket/imgs/missing.png", where="w")


# -- response metadata --------------------------------------------------------------------------


def test_response_metadata_prompt_answer_uses_raw_answer(qa_jsonl: Path) -> None:
    s = _open(qa_jsonl)[3]
    assert s.reference == "6"
    assert s.metadata["response"] == "Add them. #### 6"


def test_response_field_overrides(tmp_path: Path) -> None:
    rows = [
        {"question": "q", "answer": "#### 1", "solution_text": "Long solution ending in 1."},
        {
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
            "solution_text": None,
        },
    ]
    p = write_jsonl(tmp_path / "r.jsonl", rows[:1])
    s = _open(p, response_field="solution_text")[0]
    assert s.reference == "1" and s.metadata["response"] == "Long solution ending in 1."
    c = write_jsonl(tmp_path / "c.jsonl", rows[1:])
    s = _open(c, response_field="solution_text")[0]
    assert s.metadata["response"] == "yo"  # null response_field falls back to the assistant turn
    t = write_jsonl(tmp_path / "t.jsonl", [{"text": "just a prompt"}])
    assert _open(t)[0].metadata["response"] is None
    with pytest.raises(DataError, match="column 'nope' not found"):
        _open(p, response_field="nope")


def test_chat_response_is_text_of_trailing_assistant_turn(chat_jsonl: Path) -> None:
    src = _open(chat_jsonl)
    assert [s.metadata["response"] for s in src] == [str(i) for i in range(6)]


# -- token counting -------------------------------------------------------------------------------


def test_max_prompt_tokens_counts_text_only(tmp_path: Path, tiny_tokenizer) -> None:
    big = base64.b64encode(_png(RED, size=(64, 64))).decode()
    rows = [{"prompt": "short", "image": big}, {"prompt": "a much longer prompt " * 20, "image": big}]
    path = write_jsonl(tmp_path / "tok.jsonl", rows)
    cfg = DataConfig(uri=str(path), image_field="image", max_prompt_tokens=16, shuffle=False)
    src = open_source(cfg, tokenizer=tiny_tokenizer)
    assert [s.messages[-1]["content"][-1]["text"] for s in src] == ["short"]
    assert src[0].metadata["prompt_tokens"] <= 16
    assert src.mapper.dropped == 1


# -- preview / sniff ------------------------------------------------------------------------------


def test_sniff_reports_image_field_candidates(image_parquet: Path) -> None:
    out = preview_uri(str(image_parquet), limit=2)
    assert out["inferred"]["image_fields"] == ["image"]
    assert out["inferred"]["format"] == "prompt_answer"
    assert out["columns"]["image"] == "struct<bytes: binary, path: string>"
    assert out["rows"][0]["image"] == {"__image__": True, "width": 8, "height": 6}
    assert "bytes" not in json.dumps(out["rows"])
    assert sniff_format({"img": "string", "prompt": "string"}, [])["image_fields"] == ["img"]
    assert sniff_format({"picture": "string", "text": "string"}, [])["image_fields"] == ["picture"]
    assert sniff_format({"q": "string", "shot": "image"}, [])["image_fields"] == ["shot"]
    assert sniff_format({"question": "string", "answer": "string"}, [])["image_fields"] == []


def test_preview_shows_uris_and_pil_without_bytes(image_jsonl: Path) -> None:
    out = preview_uri(str(image_jsonl), limit=10)
    rows = {r["id"]: r for r in out["rows"]}
    assert rows["rel"]["image"] == "imgs/a.png"
    assert rows["http"]["image"] == "https://example.com/x.png"
    assert out["inferred"]["image_fields"] == ["image"]
    assert out["inferred"]["prompt_field"] == "prompt"

    from PIL import Image

    from monorl.data.browse import _jsonable

    pil = Image.open(io.BytesIO(_png(RED, size=(3, 2))))
    assert _jsonable({"x": pil}) == {"x": {"__image__": True, "width": 3, "height": 2}}
    assert _jsonable(_png(RED, size=(3, 2))) == {"__image__": True, "width": 3, "height": 2}
    assert _jsonable({"bytes": _png(RED), "path": "p.png"}) == {
        "__image__": True,
        "width": 8,
        "height": 6,
        "path": "p.png",
    }
    assert _jsonable({"bytes": None, "path": "p.png"}) == "p.png"
    assert _jsonable(b"\x00\x01")["__bytes__"] == 2
    assert infer_columns([{"i": pil}]) == {"i": "image"}


def test_image_column_is_not_mistaken_for_the_prompt() -> None:
    out = sniff_format({"image": "string", "caption": "string"}, [{"image": "a.png", "caption": "c"}])
    assert out["prompt_field"] == "caption" and out["image_fields"] == ["image"]


# -- message_text adoption downstream -------------------------------------------------------------


def test_reporting_rewards_and_envs_read_text_parts(tmp_path: Path) -> None:
    from monorl.core.reporting import _last_user_text
    from monorl.core.types import Completion, Sample, Trajectory, Turn
    from monorl.envs.tool_calling import ToolCallingEnv
    from monorl.rewards.base import conversation, prompt_text
    from monorl.rewards.reward_model import render_messages

    user = {
        "role": "user",
        "content": [{"type": "image", "image": "/x.png"}, {"type": "text", "text": "what is it?"}],
    }
    system = {"role": "system", "content": [{"type": "text", "text": "be brief"}]}
    assert _last_user_text([system, user]) == "what is it?"
    assert _last_user_text([system]) == "be brief"

    sample = Sample(id="s", messages=[system, user], reference="cat")
    done = Turn(prompt_token_ids=[1], completion=Completion(text="a cat", token_ids=[2]))
    traj = Trajectory(
        sample=sample,
        group_id="g",
        messages=[system, user, {"role": "assistant", "content": [{"type": "text", "text": "a cat"}]}],
        turns=[done],
    )
    assert prompt_text(traj) == "be brief\n\nwhat is it?"
    assert conversation(traj)[-1]["content"] == [{"type": "text", "text": "a cat"}]  # not duplicated

    class Tok:
        chat_template = None

    assert (
        render_messages(Tok(), conversation(traj)) == "system: be brief\nuser: what is it?\nassistant: a cat"
    )

    env = ToolCallingEnv(tools=[], system_prompt="TOOLS: {tools}")
    out = env.reset(sample)
    assert out[0] == {"role": "system", "content": "TOOLS: \n\nbe brief"}
    assert out[1] is user
