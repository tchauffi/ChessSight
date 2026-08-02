from __future__ import annotations

from pathlib import Path

import pytest

from chesssight.data.dataset import (
    INDEX_FILENAME,
    DatasetReader,
    DatasetWriter,
    read_index,
)
from chesssight.data.schema import DatasetMeta
from tests.conftest import make_sample


def make_meta() -> DatasetMeta:
    return DatasetMeta(
        name="test-run",
        created_at="2026-07-29T12:00:00Z",
        source="synthetic",
        master_seed=42,
        git_commit="abc123",
    )


@pytest.fixture
def writer(tmp_path: Path) -> DatasetWriter:
    writer = DatasetWriter(tmp_path / "run")
    writer.initialise(make_meta())
    return writer


def test_initialise_creates_the_expected_layout(writer: DatasetWriter):
    assert writer.images_dir.is_dir()
    assert writer.samples_dir.is_dir()
    assert writer.meta_path.is_file()


def test_add_writes_sample_and_index_entry(writer: DatasetWriter):
    sample = make_sample(sample_id="000000")
    entry = writer.add(sample)

    assert writer.sample_path("000000").is_file()
    assert entry.sample == "samples/000000.json"
    assert entry.fen == sample.fen

    lines = writer.index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_reader_round_trips_samples(writer: DatasetWriter):
    written = [make_sample(sample_id=f"{index:06d}") for index in range(3)]
    for sample in written:
        writer.add(sample)

    reader = DatasetReader(writer.root)
    assert len(reader) == 3
    assert reader.meta().master_seed == 42
    assert [sample.id for sample in reader] == ["000000", "000001", "000002"]
    assert reader.load("000001") == written[1]


def test_reader_filters_by_split(writer: DatasetWriter):
    writer.add(make_sample(sample_id="000000"))

    val = make_sample(sample_id="000001")
    payload = val.model_dump()
    payload["split"] = "val"
    writer.add(type(val).model_validate(payload))

    reader = DatasetReader(writer.root)
    assert [entry.id for entry in reader.entries(split="val")] == ["000001"]
    assert [entry.id for entry in reader.entries(split="train")] == ["000000"]


def test_existing_ids_supports_resume(writer: DatasetWriter):
    for index in range(4):
        writer.add(make_sample(sample_id=f"{index:06d}"))
    assert writer.existing_ids() == {"000000", "000001", "000002", "000003"}


def test_truncated_final_index_line_is_skipped(writer: DatasetWriter):
    writer.add(make_sample(sample_id="000000"))
    with writer.index_path.open("a", encoding="utf-8") as handle:
        handle.write('{"id": "000001", "image": "images/0000')

    # A run killed mid-append must still be readable and resumable.
    assert [entry.id for entry in read_index(writer.root)] == ["000000"]
    assert writer.existing_ids() == {"000000"}


def test_record_failure_appends(writer: DatasetWriter):
    writer.record_failure("000005", "blender exited with code 1")
    writer.record_failure("000009", "timeout")
    lines = writer.failures_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "timeout" in lines[1]


def test_reader_requires_an_index(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        DatasetReader(tmp_path)


def test_read_index_on_missing_file_is_empty(tmp_path: Path):
    assert list(read_index(tmp_path)) == []


def test_meta_is_written_atomically(writer: DatasetWriter):
    writer.write_meta(make_meta())
    assert not (writer.root / f"{INDEX_FILENAME}.tmp").exists()
    assert not writer.meta_path.with_suffix(".json.tmp").exists()


def test_image_path_honours_suffix(writer: DatasetWriter):
    assert writer.image_path("000000").name == "000000.jpg"
    assert writer.image_path("000000", ".png").name == "000000.png"
