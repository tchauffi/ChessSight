from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chesssight.data.dataset import DatasetReader, DatasetWriter
from chesssight.data.export import (
    SEMANTIC_LABELS,
    board_bbox,
    mask_area_check,
    write_coco,
    write_masks,
)
from chesssight.data.fen import CLASS_NAMES, NUM_CLASSES
from chesssight.data.masks import (
    BOARD_LABEL,
    colorize,
    decode_sample_masks,
    instance_mask,
    rle_encode,
    semantic_mask,
)
from chesssight.data.schema import DatasetMeta, Sample
from tests.conftest import make_sample


def sample_with_masks(sample_id: str = "000000") -> Sample:
    """A sample whose pieces carry real (blocky) masks matching their boxes."""
    sample = make_sample(sample_id=sample_id)
    payload = sample.model_dump()
    for instance_id, piece in enumerate(payload["pieces"], start=1):
        piece["instance_id"] = instance_id
        box = piece["bbox"]
        mask = np.zeros((sample.height, sample.width), dtype=bool)
        x0, y0 = int(box["x"]), int(box["y"])
        x1 = min(sample.width, x0 + int(box["width"]))
        y1 = min(sample.height, y0 + int(box["height"]))
        x0, y0 = max(0, x0), max(0, y0)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        piece["mask"] = rle_encode(mask).model_dump()
        piece["visible_pixels"] = int(mask.sum())
        piece["visible"] = bool(mask.sum())
    return Sample.model_validate(payload)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    writer = DatasetWriter(tmp_path / "run")
    writer.initialise(
        DatasetMeta(
            name="run",
            created_at="2026-07-29T00:00:00Z",
            source="synthetic",
            master_seed=1,
        )
    )
    for index in range(3):
        sample = sample_with_masks(f"{index:06d}")
        writer.add(sample)
        Image.new("RGB", (sample.width, sample.height), (40, 40, 40)).save(
            writer.root / sample.image
        )
    return writer.root


class TestInstanceMask:
    def test_round_trips_from_stored_rle(self):
        sample = sample_with_masks()
        labels = instance_mask(sample)
        assert labels.shape == (sample.height, sample.width)
        painted = {int(value) for value in np.unique(labels) if value}
        assert painted == {
            piece.instance_id for piece in sample.pieces if piece.visible
        }

    def test_decode_sample_masks_matches_recorded_areas(self):
        sample = sample_with_masks()
        decoded = decode_sample_masks(sample)
        for piece in sample.pieces:
            assert piece.instance_id is not None
            assert int(decoded[piece.instance_id].sum()) == piece.visible_pixels

    def test_works_without_the_id_pass(self, run_dir: Path):
        # The whole point of storing RLE: a dataset stays usable after the id_pass
        # PNGs are deleted, which is worth doing since they are far larger.
        assert not (run_dir / "id_pass").exists()
        sample = DatasetReader(run_dir).load("000000")
        assert instance_mask(sample).any()


class TestSemanticMask:
    def test_uses_class_ids_not_instance_ids(self):
        sample = sample_with_masks()
        labels = semantic_mask(sample)
        present = {int(value) for value in np.unique(labels)}
        # Every non-zero label is a piece class or the board, never an instance id.
        assert present <= set(range(NUM_CLASSES)) | {BOARD_LABEL}
        assert BOARD_LABEL in present

    def test_board_label_can_be_omitted(self):
        sample = sample_with_masks()
        present = {
            int(v) for v in np.unique(semantic_mask(sample, include_board=False))
        }
        assert BOARD_LABEL not in present

    def test_pieces_paint_over_the_board(self):
        sample = sample_with_masks()
        labels = semantic_mask(sample)
        piece = sample.pieces[0]
        assert piece.bbox is not None
        row = int(piece.bbox.y + piece.bbox.height / 2)
        column = int(piece.bbox.x + piece.bbox.width / 2)
        assert labels[row, column] == piece.class_id

    def test_labels_cover_every_class_name(self):
        assert len(SEMANTIC_LABELS) == NUM_CLASSES + 1
        assert SEMANTIC_LABELS[:NUM_CLASSES] == CLASS_NAMES
        assert SEMANTIC_LABELS[BOARD_LABEL] == "board"


class TestWriteMasks:
    def test_writes_a_pair_per_sample(self, run_dir: Path):
        result = write_masks(DatasetReader(run_dir), run_dir / "masks")
        assert result == {"written": 3, "skipped": 0}
        assert len(list((run_dir / "masks" / "semantic").glob("*.png"))) == 3
        assert len(list((run_dir / "masks" / "instance").glob("*.png"))) == 3

    def test_written_masks_are_single_channel_label_images(self, run_dir: Path):
        write_masks(DatasetReader(run_dir), run_dir / "masks")
        with Image.open(run_dir / "masks" / "semantic" / "000000.png") as image:
            assert image.mode == "L"
            array = np.asarray(image)
        # Values are label indices, not colours -- a viewer shows it nearly black.
        assert array.max() <= BOARD_LABEL

    def test_round_trips_back_to_the_source_arrays(self, run_dir: Path):
        reader = DatasetReader(run_dir)
        write_masks(reader, run_dir / "masks")
        sample = reader.load("000000")
        written = np.asarray(Image.open(run_dir / "masks" / "instance" / "000000.png"))
        np.testing.assert_array_equal(written, instance_mask(sample))

    def test_writes_a_self_describing_label_map(self, run_dir: Path):
        write_masks(DatasetReader(run_dir), run_dir / "masks")
        labels = json.loads((run_dir / "masks" / "labels.json").read_text())
        assert labels["semantic"]["0"] == "empty"
        assert labels["semantic"][str(BOARD_LABEL)] == "board"
        assert labels["board_label"] == BOARD_LABEL

    def test_previews_are_rgb(self, run_dir: Path):
        write_masks(DatasetReader(run_dir), run_dir / "masks", previews=True)
        with Image.open(run_dir / "masks" / "mask_preview" / "000000.png") as image:
            assert image.mode == "RGB"

    def test_limit_is_honoured(self, run_dir: Path):
        result = write_masks(DatasetReader(run_dir), run_dir / "masks", limit=2)
        assert result["written"] == 2

    def test_samples_without_masks_are_skipped_not_failed(self, tmp_path: Path):
        writer = DatasetWriter(tmp_path / "nomask")
        writer.initialise(
            DatasetMeta(
                name="n",
                created_at="2026-07-29T00:00:00Z",
                source="synthetic",
                master_seed=1,
            )
        )
        writer.add(make_sample(sample_id="000000"))  # boxes but no masks
        result = write_masks(DatasetReader(writer.root), tmp_path / "out")
        assert result == {"written": 0, "skipped": 1}


class TestCoco:
    def test_structure(self, run_dir: Path):
        result = write_coco(DatasetReader(run_dir), run_dir / "coco.json")
        document = json.loads((run_dir / "coco.json").read_text())

        assert result["images"] == 3
        assert {"images", "annotations", "categories", "info"} <= set(document)
        # 12 piece classes plus the board, which is emitted as one more category so
        # a detector can localise the playing surface in the same forward pass.
        assert len(document["categories"]) == NUM_CLASSES
        assert document["categories"][0]["name"] == "white_pawn"
        assert document["categories"][-1]["name"] == "board"

    def test_the_board_can_be_left_out(self, run_dir: Path):
        write_coco(DatasetReader(run_dir), run_dir / "coco.json", include_board=False)
        document = json.loads((run_dir / "coco.json").read_text())
        assert all(c["name"] != "board" for c in document["categories"])
        assert all(a["category_id"] != BOARD_LABEL for a in document["annotations"])

    def test_the_board_box_wraps_the_corners(self, run_dir: Path):
        sample = DatasetReader(run_dir).load("000000")
        box = board_bbox(sample)
        assert box is not None
        corners = sample.board.corners_px
        # Clipped to the image, since real boards are routinely cropped by the frame.
        assert box.x >= 0 and box.y >= 0
        assert box.x + box.width <= sample.width
        assert box.y + box.height <= sample.height
        inside = [
            c
            for c in corners
            if 0 <= c[0] <= sample.width and 0 <= c[1] <= sample.height
        ]
        for x, y in inside:
            assert box.x - 0.5 <= x <= box.x + box.width + 0.5
            assert box.y - 0.5 <= y <= box.y + box.height + 0.5

    def test_annotations_use_the_modal_box(self, run_dir: Path):
        write_coco(DatasetReader(run_dir), run_dir / "coco.json")
        document = json.loads((run_dir / "coco.json").read_text())
        sample = DatasetReader(run_dir).load("000000")

        first = document["annotations"][0]
        piece = sample.pieces[0]
        assert piece.bbox is not None
        assert first["bbox"] == [
            piece.bbox.x,
            piece.bbox.y,
            piece.bbox.width,
            piece.bbox.height,
        ]
        assert first["category_id"] == piece.class_id

    def test_segmentation_is_column_major_as_coco_expects(self, run_dir: Path):
        # Our RLE is row-major; COCO's is column-major. Getting this wrong produces
        # masks that look like diagonal static, so the transpose is asserted here.
        write_coco(DatasetReader(run_dir), run_dir / "coco.json")
        document = json.loads((run_dir / "coco.json").read_text())
        sample = DatasetReader(run_dir).load("000000")
        annotation = document["annotations"][0]
        piece = sample.pieces[0]
        assert piece.instance_id is not None

        counts = annotation["segmentation"]["counts"]
        height, width = annotation["segmentation"]["size"]
        flat = np.zeros(height * width, dtype=bool)
        position, value = 0, False
        for run in counts:
            flat[position : position + run] = value
            position += run
            value = not value
        # Column-major means reshaping transposed recovers the original mask.
        recovered = flat.reshape(width, height).T
        np.testing.assert_array_equal(
            recovered, decode_sample_masks(sample)[piece.instance_id]
        )

    def test_masks_can_be_omitted(self, run_dir: Path):
        write_coco(DatasetReader(run_dir), run_dir / "coco.json", with_masks=False)
        document = json.loads((run_dir / "coco.json").read_text())
        assert all("segmentation" not in a for a in document["annotations"])

    def test_invisible_pieces_are_excluded(self, tmp_path: Path):
        writer = DatasetWriter(tmp_path / "run")
        writer.initialise(
            DatasetMeta(
                name="r",
                created_at="2026-07-29T00:00:00Z",
                source="synthetic",
                master_seed=1,
            )
        )
        sample = sample_with_masks()
        payload = sample.model_dump()
        payload["pieces"][0]["visible"] = False
        writer.add(Sample.model_validate(payload))

        write_coco(
            DatasetReader(writer.root), tmp_path / "coco.json", include_board=False
        )
        document = json.loads((tmp_path / "coco.json").read_text())
        assert len(document["annotations"]) == len(sample.pieces) - 1


def test_mask_area_check_passes_on_consistent_data():
    assert mask_area_check(sample_with_masks()) == []


def test_mask_area_check_catches_a_wrong_pixel_count():
    sample = sample_with_masks()
    payload = sample.model_dump()
    payload["pieces"][0]["visible_pixels"] = 99999
    problems = mask_area_check(Sample.model_validate(payload))
    assert len(problems) == 1
    assert "visible_pixels" in problems[0]


def test_colorize_gives_distinct_colours_per_label():
    labels = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    rgb = colorize(labels)
    assert rgb.shape == (2, 3, 3)
    assert tuple(rgb[0, 0]) == (0, 0, 0)  # background stays black
    assert len({tuple(rgb[r, c]) for r in range(2) for c in range(3)}) == 6
