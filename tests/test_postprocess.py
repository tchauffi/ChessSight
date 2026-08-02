from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from chesssight.data.fen import STARTING_FEN, fen_to_grid, iter_occupied, square_name
from chesssight.data.geometry import (
    BOARD_CORNERS,
    all_square_centers_board,
    apply_homography,
)
from chesssight.data.masks import ROLE_CODES
from chesssight.synth.postprocess import (
    MIN_VISIBLE_PIXELS,
    PostprocessError,
    build_sample,
)
from tests.conftest import TEST_HOMOGRAPHY

# Tall enough that TEST_HOMOGRAPHY projects the whole board inside the frame --
# otherwise the bottom-rank pieces are legitimately invisible and the fixture would
# be testing clipping rather than mask decoding.
WIDTH, HEIGHT = 512, 512


def make_raw(fen: str = STARTING_FEN, *, id_pass_path: str | None = None) -> dict:
    """A raw record exactly as the Blender side would emit it."""
    grid = fen_to_grid(fen)
    corners = apply_homography(TEST_HOMOGRAPHY, np.asarray(BOARD_CORNERS))
    centers = apply_homography(TEST_HOMOGRAPHY, all_square_centers_board())

    pieces = []
    for instance_id, (rank, file, class_id) in enumerate(iter_occupied(grid), start=1):
        center = centers[rank * 8 + file]
        pieces.append(
            {
                "instance_id": instance_id,
                "class_id": class_id,
                "rank_index": rank,
                "file_index": file,
                "bbox_amodal": [
                    float(center[0]) - 8.0,
                    float(center[1]) - 18.0,
                    float(center[0]) + 8.0,
                    float(center[1]) + 4.0,
                ],
                "base_center_px": [float(center[0]), float(center[1])],
                "apex_px": [float(center[0]), float(center[1]) - 18.0],
                "depth": 12.0,
                "behind_camera": False,
            }
        )

    return {
        "id": "000000",
        "fen": fen,
        "grid": grid,
        "width": WIDTH,
        "height": HEIGHT,
        "image_path": "/tmp/unused.png",
        "id_pass_path": id_pass_path,
        "board": {
            "corners_px": [[float(x), float(y)] for x, y in corners],
            "corner_depths": [10.0] * 4,
            "all_corners_in_front": True,
            "all_corners_in_frame": True,
        },
        "square_centers_px": [[float(x), float(y)] for x, y in centers],
        "square_center_depths": [12.0] * 64,
        "pieces": pieces,
        "camera": {
            "intrinsics": [[500.0, 0.0, 256.0], [0.0, 500.0, 192.0], [0.0, 0.0, 1.0]],
            "extrinsics": [
                [1.0, 0, 0, 0],
                [0, 1.0, 0, 0],
                [0, 0, 1.0, 12.0],
                [0, 0, 0, 1.0],
            ],
            "focal_mm": 50.0,
            "sensor_width_mm": 36.0,
            "resolution": [WIDTH, HEIGHT],
        },
        "render": {
            "engine": "BLENDER_EEVEE",
            "samples": 16,
            "seed": 12345,
            "blender_version": "5.2.0",
            "render_seconds": 0.5,
        },
    }


def write_id_pass(path, raw: dict, *, hide_instance: int | None = None) -> str:
    """Paint each piece's amodal box into an id image."""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for piece in raw["pieces"]:
        if piece["instance_id"] == hide_instance:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in piece["bbox_amodal"])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(WIDTH, x1), min(HEIGHT, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        image[y0:y1, x0:x1, 0] = piece["instance_id"]
        image[y0:y1, x0:x1, 1] = ROLE_CODES["piece"]
    Image.fromarray(image).save(path)
    return str(path)


class TestBuildSample:
    def test_produces_a_valid_sample(self):
        sample = build_sample(make_raw(), image_rel_path="images/000000.png")
        assert sample.id == "000000"
        assert sample.fen == STARTING_FEN
        assert len(sample.squares) == 64
        assert len(sample.pieces) == 32
        assert sample.source == "synthetic"
        assert sample.annotation_method == "synthetic_gt"
        assert sample.seed == 12345

    def test_homography_reproduces_the_projected_centres(self):
        sample = build_sample(make_raw(), image_rel_path="images/000000.png")
        assert sample.board.reprojection_error_px == pytest.approx(0.0, abs=1e-6)

    def test_homography_inverse_is_stored(self):
        sample = build_sample(make_raw(), image_rel_path="images/000000.png")
        forward = np.asarray(sample.board.homography)
        inverse = np.asarray(sample.board.homography_inv)
        np.testing.assert_allclose(forward @ inverse, np.eye(3), atol=1e-9)

    def test_squares_are_in_reading_order_with_correct_occupants(self):
        sample = build_sample(make_raw(), image_rel_path="images/000000.png")
        assert sample.squares[0].name == "a8"
        assert sample.squares[63].name == "h1"
        grid = fen_to_grid(STARTING_FEN)
        for index, square in enumerate(sample.squares):
            rank, file = divmod(index, 8)
            assert square.occupant == grid[rank][file]

    def test_pieces_carry_their_square_names(self):
        sample = build_sample(make_raw(), image_rel_path="images/000000.png")
        by_square = {piece.square: piece for piece in sample.pieces}
        assert by_square["e1"].class_name == "white_king"
        assert by_square["d8"].class_name == "black_queen"
        assert by_square["a1"].class_name == "white_rook"

    def test_amodal_boxes_survive_without_a_mask(self):
        sample = build_sample(make_raw(), image_rel_path="images/000000.png")
        assert all(piece.bbox_amodal is not None for piece in sample.pieces)
        assert all(piece.bbox is None for piece in sample.pieces)
        assert all(piece.visible for piece in sample.pieces)

    def test_json_round_trip(self):
        sample = build_sample(make_raw(), image_rel_path="images/000000.png")
        assert type(sample).model_validate_json(sample.model_dump_json()) == sample


class TestWithIdPass:
    def test_masks_and_modal_boxes_are_decoded(self, tmp_path):
        raw = make_raw()
        raw["id_pass_path"] = write_id_pass(tmp_path / "ids.png", raw)

        sample = build_sample(
            raw, image_rel_path="images/000000.png", mask_rel_path="id_pass/000000.png"
        )
        assert sample.mask_image == "id_pass/000000.png"
        for piece in sample.pieces:
            assert piece.bbox is not None
            assert piece.mask is not None
            assert piece.visible_pixels > 0
            assert piece.visibility == pytest.approx(1.0, abs=0.15)

    def test_a_hidden_piece_is_marked_invisible_but_kept(self, tmp_path):
        raw = make_raw()
        raw["id_pass_path"] = write_id_pass(tmp_path / "ids.png", raw, hide_instance=5)

        sample = build_sample(
            raw, image_rel_path="images/000000.png", mask_rel_path="id_pass/000000.png"
        )
        hidden = next(piece for piece in sample.pieces if piece.instance_id == 5)
        assert hidden.visible_pixels == 0
        assert not hidden.visible
        assert hidden.bbox is None
        # Still present, and still on the grid: the square really is occupied even
        # though this viewpoint cannot see it.
        assert len(sample.pieces) == 32
        rank, file = hidden.rank_index, hidden.file_index
        assert sample.grid[rank][file] == hidden.class_id
        assert sample.squares[rank * 8 + file].occupant == hidden.class_id

    def test_masks_can_be_skipped_to_save_space(self, tmp_path):
        raw = make_raw()
        raw["id_pass_path"] = write_id_pass(tmp_path / "ids.png", raw)

        sample = build_sample(
            raw,
            image_rel_path="images/000000.png",
            mask_rel_path="id_pass/000000.png",
            store_masks=False,
        )
        assert all(piece.mask is None for piece in sample.pieces)
        # Boxes and pixel counts still come from the pass.
        assert all(piece.bbox is not None for piece in sample.pieces)

    def test_visibility_threshold(self):
        assert MIN_VISIBLE_PIXELS > 0


class TestFailures:
    def test_fen_grid_disagreement_is_rejected(self):
        raw = make_raw()
        raw["grid"] = fen_to_grid("8/8/8/8/8/8/8/8")
        with pytest.raises(PostprocessError, match="fen and grid disagree"):
            build_sample(raw, image_rel_path="images/000000.png")

    def test_projection_disagreement_is_rejected(self):
        # Nudging one projected centre simulates a coordinate-convention bug; the
        # cross-check between the homography and Blender's own projection must
        # refuse to emit a sample rather than bake the error into the dataset.
        raw = make_raw()
        raw["square_centers_px"][20][0] += 12.0
        with pytest.raises(PostprocessError, match="disagree by"):
            build_sample(raw, image_rel_path="images/000000.png")

    def test_a_tiny_disagreement_is_tolerated(self):
        raw = make_raw()
        raw["square_centers_px"][20][0] += 0.05
        sample = build_sample(raw, image_rel_path="images/000000.png")
        assert sample.board.reprojection_error_px < 0.5

    def test_empty_board_has_no_pieces(self):
        raw = make_raw("8/8/8/8/8/8/8/8")
        sample = build_sample(raw, image_rel_path="images/000000.png")
        assert sample.pieces == []
        assert all(square.occupant == 0 for square in sample.squares)


def test_square_names_follow_grid_order():
    assert square_name(0, 0) == "a8"
    assert square_name(7, 4) == "e1"
