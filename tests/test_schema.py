from __future__ import annotations

import pytest
from pydantic import ValidationError

from chesssight.data.fen import STARTING_FEN, fen_to_grid
from chesssight.data.schema import (
    BoundingBox,
    DatasetMeta,
    IndexEntry,
    MaskRLE,
    PieceAnnotation,
    RenderInfo,
    Sample,
)
from tests.conftest import make_sample


class TestMaskRLE:
    def test_counts_must_cover_the_whole_image(self):
        assert MaskRLE(height=2, width=3, counts=[2, 4]).counts == [2, 4]
        with pytest.raises(ValidationError):
            MaskRLE(height=2, width=3, counts=[2, 3])

    def test_negative_counts_rejected(self):
        with pytest.raises(ValidationError):
            MaskRLE(height=1, width=4, counts=[-1, 5])


class TestBoundingBox:
    def test_xyxy_conversion(self):
        assert BoundingBox(x=1.0, y=2.0, width=3.0, height=4.0).xyxy == (
            1.0,
            2.0,
            4.0,
            6.0,
        )

    def test_zero_size_rejected(self):
        with pytest.raises(ValidationError):
            BoundingBox(x=0, y=0, width=0, height=4)


class TestPieceAnnotation:
    def test_square_name_must_match_indices(self):
        with pytest.raises(ValidationError, match="disagrees with indices"):
            PieceAnnotation(class_id=1, square="a1", rank_index=0, file_index=0)

    def test_class_name_lookup(self):
        piece = PieceAnnotation(class_id=6, square="e1", rank_index=7, file_index=4)
        assert piece.class_name == "white_king"

    def test_empty_class_rejected(self):
        with pytest.raises(ValidationError):
            PieceAnnotation(class_id=0, square="a8", rank_index=0, file_index=0)


class TestSample:
    def test_a_consistent_sample_validates(self, sample: Sample):
        assert sample.fen == STARTING_FEN
        assert len(sample.squares) == 64
        assert len(sample.pieces) == 32
        assert sample.board.reprojection_error_px == pytest.approx(0.0, abs=1e-6)

    def test_json_round_trip(self, sample: Sample):
        assert Sample.model_validate_json(sample.model_dump_json()) == sample

    def test_fen_and_grid_must_agree(self, sample: Sample):
        payload = sample.model_dump()
        payload["grid"] = fen_to_grid("8/8/8/8/8/8/8/8")
        with pytest.raises(ValidationError, match="fen and grid disagree"):
            Sample.model_validate(payload)

    def test_squares_must_be_in_reading_order(self, sample: Sample):
        payload = sample.model_dump()
        payload["squares"][0], payload["squares"][1] = (
            payload["squares"][1],
            payload["squares"][0],
        )
        with pytest.raises(ValidationError, match="grid reading order"):
            Sample.model_validate(payload)

    def test_square_occupant_must_match_the_grid(self, sample: Sample):
        payload = sample.model_dump()
        payload["squares"][0]["occupant"] = 1
        with pytest.raises(ValidationError, match="disagrees with grid value"):
            Sample.model_validate(payload)

    def test_piece_class_must_match_the_grid(self, sample: Sample):
        payload = sample.model_dump()
        payload["pieces"][0]["class_id"] = 5
        with pytest.raises(ValidationError, match="but the grid says"):
            Sample.model_validate(payload)

    def test_wrong_number_of_squares_rejected(self, sample: Sample):
        payload = sample.model_dump()
        payload["squares"] = payload["squares"][:63]
        with pytest.raises(ValidationError):
            Sample.model_validate(payload)

    def test_unknown_fields_are_rejected(self, sample: Sample):
        payload = sample.model_dump()
        payload["surprise"] = 1
        with pytest.raises(ValidationError):
            Sample.model_validate(payload)

    def test_empty_board_sample_validates(self):
        empty = make_sample("8/8/8/8/8/8/8/8")
        assert empty.pieces == []

    def test_real_photo_sample_omits_renderer_only_fields(self):
        real = make_sample(sample_id="real_0001")
        payload = real.model_dump()
        payload["source"] = "real"
        payload["split"] = "test"
        payload["camera"] = None
        payload["render"] = None
        payload["board"]["reprojection_error_px"] = None
        for piece in payload["pieces"]:
            piece["mask"] = None
            piece["visible_pixels"] = None
            piece["bbox"] = None

        parsed = Sample.model_validate(payload)
        assert parsed.source == "real"
        assert parsed.split == "test"
        assert all(piece.mask is None for piece in parsed.pieces)


class TestRenderInfo:
    def test_engine_enum_matches_blender_52_identifiers(self):
        assert (
            RenderInfo(
                engine="BLENDER_EEVEE", samples=8, seed=1, blender_version="5.2.0"
            ).engine
            == "BLENDER_EEVEE"
        )
        with pytest.raises(ValidationError):
            # The 4.2-4.5 identifier; Blender 5.2 rejects it, so the schema must too.
            RenderInfo(
                engine="BLENDER_EEVEE_NEXT", samples=8, seed=1, blender_version="5.2.0"
            )


def test_index_entry_round_trip():
    entry = IndexEntry(
        id="000001",
        image="images/000001.jpg",
        sample="samples/000001.json",
        source="synthetic",
        split="train",
        fen=STARTING_FEN,
    )
    assert IndexEntry.model_validate_json(entry.model_dump_json()) == entry


def test_dataset_meta_defaults():
    meta = DatasetMeta(
        name="run", created_at="2026-07-29T00:00:00Z", source="synthetic", master_seed=1
    )
    assert meta.counts == {}
    assert meta.git_commit is None


class TestCapturedPieces:
    """A captured piece is in the image but not in play."""

    def test_a_captured_piece_needs_no_square(self):
        piece = PieceAnnotation(class_id=1, on_board=False)
        assert piece.square is None
        assert piece.rank_index is None
        assert not piece.on_board

    def test_a_captured_piece_may_not_claim_a_square(self):
        with pytest.raises(ValidationError, match="must not claim a square"):
            PieceAnnotation(
                class_id=1, on_board=False, square="e4", rank_index=4, file_index=4
            )

    def test_a_piece_on_the_board_still_needs_its_square(self):
        with pytest.raises(ValidationError, match="needs a square and indices"):
            PieceAnnotation(class_id=1, on_board=True)

    def test_captured_pieces_are_excluded_from_the_grid_check(self, sample: Sample):
        payload = sample.model_dump()
        payload["pieces"].append(
            {
                "class_id": 5,
                "square": None,
                "rank_index": None,
                "file_index": None,
                "on_board": False,
                "instance_id": 99,
                "bbox": None,
                "bbox_amodal": None,
                "mask": None,
                "visible_pixels": 300,
                "visibility": 1.0,
                "visible": True,
                "upright": True,
                "base_center_px": [10.0, 20.0],
            }
        )
        parsed = Sample.model_validate(payload)
        captured = [piece for piece in parsed.pieces if not piece.on_board]
        assert len(captured) == 1
        # The grid is untouched: the piece is beside the board, not on it.
        assert parsed.grid == sample.grid
        assert sum(1 for piece in parsed.pieces if piece.on_board) == 32
