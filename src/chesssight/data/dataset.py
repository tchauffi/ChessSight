"""Reading and writing a dataset on disk.

Layout::

    <run>/
      meta.json          provenance: config, git commit, master seed, versions
      index.jsonl        one IndexEntry per line
      images/000123.jpg
      samples/000123.json

``index.jsonl`` is append-only and is the source of truth for "what has been
generated so far", which is what makes an interrupted run resumable: the runner
reads the ids already present and renders only the rest. Appending a line only
after the sample record has been written means a crash mid-write can leave an
orphan sample file, but never an index entry pointing at a missing one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from chesssight.data.schema import DatasetMeta, IndexEntry, Sample

META_FILENAME = "meta.json"
INDEX_FILENAME = "index.jsonl"
IMAGES_DIRNAME = "images"
SAMPLES_DIRNAME = "samples"
FAILURES_FILENAME = "failures.jsonl"


class DatasetWriter:
    """Creates the run directory and appends samples to it."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.images_dir = self.root / IMAGES_DIRNAME
        self.samples_dir = self.root / SAMPLES_DIRNAME
        self.index_path = self.root / INDEX_FILENAME
        self.meta_path = self.root / META_FILENAME
        self.failures_path = self.root / FAILURES_FILENAME

    def initialise(self, meta: DatasetMeta) -> None:
        """Create directories and write ``meta.json``. Safe to call on a resume."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.write_meta(meta)

    def write_meta(self, meta: DatasetMeta) -> None:
        _atomic_write_text(self.meta_path, meta.model_dump_json(indent=2) + "\n")

    def sample_path(self, sample_id: str) -> Path:
        return self.samples_dir / f"{sample_id}.json"

    def image_path(self, sample_id: str, suffix: str = ".jpg") -> Path:
        return self.images_dir / f"{sample_id}{suffix}"

    def add(self, sample: Sample) -> IndexEntry:
        """Write the sample record, then append its index entry.

        The image itself is written by the renderer; this only records it.
        """
        sample_path = self.sample_path(sample.id)
        _atomic_write_text(sample_path, sample.model_dump_json(indent=2) + "\n")

        entry = IndexEntry(
            id=sample.id,
            image=sample.image,
            sample=str(sample_path.relative_to(self.root)),
            source=sample.source,
            split=sample.split,
            fen=sample.fen,
        )
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        return entry

    def record_failure(self, sample_id: str, reason: str) -> None:
        """Append a permanent failure so a run can finish despite bad samples."""
        with self.failures_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": sample_id, "reason": reason}) + "\n")

    def existing_ids(self) -> set[str]:
        """Ids already present in ``index.jsonl``, for resuming a run."""
        return {entry.id for entry in read_index(self.root)}


class DatasetReader:
    """Read-only access to a generated run."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        if not (self.root / INDEX_FILENAME).exists():
            raise FileNotFoundError(f"no {INDEX_FILENAME} under {self.root}")

    def meta(self) -> DatasetMeta:
        text = (self.root / META_FILENAME).read_text(encoding="utf-8")
        return DatasetMeta.model_validate_json(text)

    def entries(self, split: str | None = None) -> list[IndexEntry]:
        entries = list(read_index(self.root))
        if split is not None:
            entries = [entry for entry in entries if entry.split == split]
        return entries

    def load(self, sample_id: str) -> Sample:
        path = self.root / SAMPLES_DIRNAME / f"{sample_id}.json"
        return Sample.model_validate_json(path.read_text(encoding="utf-8"))

    def image_path(self, sample: Sample) -> Path:
        return self.root / sample.image

    def __iter__(self) -> Iterator[Sample]:
        for entry in self.entries():
            yield self.load(entry.id)

    def __len__(self) -> int:
        return len(self.entries())


def read_index(root: Path) -> Iterator[IndexEntry]:
    """Yield index entries, skipping a truncated final line.

    A run killed mid-append can leave a partial last line; that is expected during
    resume and must not be fatal.
    """
    index_path = Path(root) / INDEX_FILENAME
    if not index_path.exists():
        return
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield IndexEntry.model_validate_json(line)
            except ValueError:
                continue


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file and rename, so readers never see a partial file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
