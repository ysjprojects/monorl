"""Raw row providers: the part of a data source that knows where bytes live.

A :class:`RowSource` yields ``(row_id, row)`` pairs. Two flavours:

* :class:`FileRows` — local / fsspec files (one or many), either materialised into memory or
  streamed file-by-file. Row ids are ``<file-uri>#<index-in-file>`` and therefore stable across
  shuffles and seeds.
* :class:`HubRows` — ``datasets.load_dataset`` over a Hub repo, ``Dataset`` (materialised,
  indexable) or ``IterableDataset`` (streaming).
"""

from __future__ import annotations

import abc
import itertools
import random
from collections.abc import Iterator, Sequence
from typing import Any

from monorl.core.errors import DataError
from monorl.data import formats
from monorl.data.uri import ParsedURI, filesystem_for

SHUFFLE_BUFFER = 1000

RowRef = tuple[str, dict[str, Any]]


def buffered_shuffle(items: Iterator[RowRef], seed: int, buffer_size: int = SHUFFLE_BUFFER) -> Iterator[RowRef]:
    """Reservoir-style approximate shuffle for streams (same idea as ``IterableDataset.shuffle``)."""
    rng = random.Random(seed)
    buffer: list[RowRef] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        j = rng.randrange(buffer_size)
        yield buffer[j]
        buffer[j] = item
    rng.shuffle(buffer)
    yield from buffer


class RowSource(abc.ABC):
    """Common surface used by :mod:`monorl.data.sources` and :mod:`monorl.data.browse`."""

    uri: str
    streaming: bool

    @abc.abstractmethod
    def columns(self) -> dict[str, str]: ...

    def head(self, n: int) -> list[dict[str, Any]]:
        return [row for _, row in itertools.islice(self.iter_rows(), n)]

    @abc.abstractmethod
    def iter_rows(self, *, shuffle_seed: int | None = None) -> Iterator[RowRef]:
        """All rows; ``shuffle_seed`` requests a deterministic shuffled order."""

    def known_len(self) -> int | None:
        """Row count when it is cheap to know (materialised, or file metadata), else ``None``."""
        return None


# -- files ------------------------------------------------------------------------------------


def expand_files(fs, parsed: ParsedURI) -> list[str]:
    """Resolve a file / directory / glob into concrete data-file paths (sorted)."""
    path = parsed.path
    if any(ch in path for ch in "*?["):
        fs.invalidate_cache()
        matches = [p for p in fs.glob(path) if fs.isfile(p)]
        files = formats.select_files(matches)
        if not files:
            raise DataError(f"glob {parsed.raw!r} matched no supported data files")
        return files
    if not fs.exists(path):
        raise FileNotFoundError(f"dataset path not found: {parsed.raw}")
    if fs.isdir(path):
        fs.invalidate_cache(path)
        files = formats.select_files(fs.find(path))
        if not files:
            raise DataError(
                f"directory {parsed.raw!r} contains no supported data files ({sorted(formats.FORMAT_EXTENSIONS)})"
            )
        return files
    if formats.format_of(path) is None:
        raise DataError(f"unsupported file type: {parsed.raw} (expected {sorted(formats.FORMAT_EXTENSIONS)})")
    return [path]


class FileRows(RowSource):
    def __init__(self, parsed: ParsedURI, *, streaming: bool = False) -> None:
        self.parsed = parsed
        self.uri = parsed.raw
        self.streaming = streaming
        self.fs = filesystem_for(parsed)
        self.files = expand_files(self.fs, parsed)
        self.format = formats.format_of(self.files[0])
        self._rows: list[RowRef] | None = None
        self._columns: dict[str, str] | None = None

    # -- helpers
    def file_uri(self, path: str) -> str:
        return self.parsed.with_path(path)

    def _iter_file(self, path: str) -> Iterator[RowRef]:
        uri = self.file_uri(path)
        for i, row in enumerate(formats.iter_file_rows(self.fs, path, self.format)):
            yield f"{uri}#{i}", row

    def _iter_files(self, order: Sequence[str]) -> Iterator[RowRef]:
        for path in order:
            yield from self._iter_file(path)

    def _materialise(self) -> list[RowRef]:
        if self._rows is None:
            self._rows = list(self._iter_files(self.files))
        return self._rows

    # -- RowSource
    def columns(self) -> dict[str, str]:
        if self._columns is None:
            cols = formats.arrow_schema_of(self.fs, self.files[0], self.format)
            if cols is None:
                cols = formats.infer_columns(self.head(formats.SNIFF_ROWS))
            self._columns = cols
        return self._columns

    def head(self, n: int) -> list[dict[str, Any]]:
        if self._rows is not None:
            return [row for _, row in self._rows[:n]]
        return [row for _, row in itertools.islice(self._iter_files(self.files), n)]

    def iter_rows(self, *, shuffle_seed: int | None = None) -> Iterator[RowRef]:
        if not self.streaming:
            rows = self._materialise()
            if shuffle_seed is None:
                return iter(rows)
            order = list(range(len(rows)))
            random.Random(shuffle_seed).shuffle(order)
            return (rows[i] for i in order)
        if shuffle_seed is None:
            return self._iter_files(self.files)
        files = list(self.files)
        random.Random(shuffle_seed).shuffle(files)
        return buffered_shuffle(self._iter_files(files), shuffle_seed)

    def known_len(self) -> int | None:
        if self._rows is not None:
            return len(self._rows)
        if self.format == "parquet":
            return sum(formats.parquet_num_rows(self.fs, p) for p in self.files)
        if not self.streaming:
            return len(self._materialise())
        return None


# -- hugging face hub -------------------------------------------------------------------------


class HubRows(RowSource):
    def __init__(
        self,
        parsed: ParsedURI,
        *,
        split: str,
        config_name: str | None = None,
        streaming: bool = False,
        load_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from datasets import load_dataset

        self.parsed = parsed
        self.uri = parsed.raw
        self.split = split
        self.streaming = streaming
        kwargs: dict[str, Any] = dict(load_kwargs or {})
        if parsed.revision:
            kwargs.setdefault("revision", parsed.revision)
        if parsed.subpath:
            kwargs.setdefault("data_files", parsed.subpath)
        try:
            self.dataset = load_dataset(
                parsed.repo_id, name=config_name, split=split, streaming=streaming, **kwargs
            )
        except (ValueError, FileNotFoundError) as e:
            raise DataError(f"cannot load {parsed.raw!r} (split={split!r}, config={config_name!r}): {e}") from e
        self._columns: dict[str, str] | None = None

    def _row_id(self, index: int) -> str:
        return f"{self.uri}#{self.split}#{index}"

    def columns(self) -> dict[str, str]:
        if self._columns is None:
            features = getattr(self.dataset, "features", None)
            if features:
                self._columns = formats.features_to_columns(features)
            else:
                self._columns = formats.infer_columns(self.head(formats.SNIFF_ROWS))
        return self._columns

    def head(self, n: int) -> list[dict[str, Any]]:
        if self.streaming:
            return list(self.dataset.take(n))
        return [self.dataset[i] for i in range(min(n, len(self.dataset)))]

    def iter_rows(self, *, shuffle_seed: int | None = None) -> Iterator[RowRef]:
        if self.streaming:
            ds = self.dataset if shuffle_seed is None else self.dataset.shuffle(seed=shuffle_seed, buffer_size=SHUFFLE_BUFFER)
            return ((self._row_id(i), row) for i, row in enumerate(ds))
        n = len(self.dataset)
        order = list(range(n))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(order)
        return ((self._row_id(i), self.dataset[i]) for i in order)

    def known_len(self) -> int | None:
        if not self.streaming:
            return len(self.dataset)
        info = getattr(self.dataset, "info", None)
        splits = getattr(info, "splits", None) or {}
        split_info = splits.get(self.split) if hasattr(splits, "get") else None
        n = getattr(split_info, "num_examples", None)
        return int(n) if n else None


def open_rows(parsed: ParsedURI, *, split: str, config_name: str | None, streaming: bool, extra: dict[str, Any] | None = None) -> RowSource:
    if parsed.kind == "hf":
        return HubRows(parsed, split=split, config_name=config_name, streaming=streaming, load_kwargs=extra)
    return FileRows(parsed, streaming=streaming)
