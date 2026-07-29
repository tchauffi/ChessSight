"""End-to-end render tests.

The geometry assertions here are the ones that matter. A y-flip, a transposed board,
a wrong corner order or a mis-scaled resolution all produce a perfectly plausible
image and perfectly well-formed JSON -- and every one of them is caught by checking
that two independently-computed projections of the same 64 points agree, and that
the id-pass pixels belong to the piece the labels claim is there.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from chesssight.data.fen import CLASS_NAMES, STARTING_FEN, fen_to_grid, iter_occupied
from chesssight.data.geometry import (
    board_to_image_homography,
    is_mirrored,
    polygon_contains,
    polygon_signed_area,
    project_square_centers,
)
from chesssight.data.masks import instance_ids, load_id_image, role_codes
from chesssight.data.qa import contact_sheet, render_overlay
from chesssight.synth.positions import StartingPositionSampler
from chesssight.synth.postprocess import build_sample, load_raw
from tests.integration.conftest import make_run, run_blender

pytestmark = pytest.mark.blender


@pytest.fixture(scope="module")
def rendered(render_root):
    """Render two starting-position scenes once and reuse them."""
    root = render_root / "smoke"
    _, writer, shard = make_run(root, count=2, sampler=StartingPositionSampler())
    result = run_blender(shard)
    assert result.returncode == 0, result.stderr[-4000:]

    samples = []
    for index in range(2):
        sample_id = f"{index:06d}"
        raw = load_raw(root / "raw_labels" / f"{sample_id}.json")
        samples.append(
            build_sample(
                raw,
                image_rel_path=f"images/{sample_id}.png",
                mask_rel_path=f"id_pass/{sample_id}.png",
            )
        )
    return root, writer, samples


class TestRenderOutput:
    def test_blender_exits_cleanly(self, rendered):
        _, _, samples = rendered
        assert len(samples) == 2

    def test_images_exist_and_are_not_blank(self, rendered):
        root, _, samples = rendered
        for sample in samples:
            array = np.asarray(Image.open(root / sample.image).convert("RGB"))
            assert array.shape[:2] == (sample.height, sample.width)
            # A uniform frame means the scene failed to build or was unlit.
            assert array.std() > 5.0

    def test_render_metadata_is_recorded(self, rendered):
        _, _, samples = rendered
        for sample in samples:
            assert sample.render is not None
            assert sample.render.engine == "BLENDER_EEVEE"
            assert sample.render.blender_version.startswith("5.")
            assert sample.render.render_seconds >= 0


class TestGeometry:
    def test_homography_agrees_with_blenders_own_projection(self, rendered):
        # Two completely different computations of the same 64 points. Sub-pixel
        # agreement is what proves the coordinate conventions line up.
        root, _, samples = rendered
        for sample in samples:
            raw = load_raw(root / "raw_labels" / f"{sample.id}.json")
            direct = np.asarray(raw["square_centers_px"])
            viaH = project_square_centers(
                board_to_image_homography(np.asarray(sample.board.corners_px))
            )
            assert np.max(np.linalg.norm(viaH - direct, axis=1)) < 0.01

    def test_reprojection_error_is_subpixel(self, rendered):
        _, _, samples = rendered
        for sample in samples:
            assert sample.board.reprojection_error_px < 0.5

    def test_every_square_centre_lies_inside_the_board_outline(self, rendered):
        _, _, samples = rendered
        for sample in samples:
            corners = sample.board.corners_px
            for square in sample.squares:
                assert polygon_contains(
                    corners, (square.center_px[0], square.center_px[1])
                ), f"{sample.id} {square.name} fell outside the board"

    def test_square_geometry_is_ordered_and_sane(self, rendered):
        _, _, samples = rendered
        for sample in samples:
            assert sample.squares[0].name == "a8"
            assert sample.squares[63].name == "h1"
            for square in sample.squares:
                assert len(square.quad_px) == 4

    def test_the_rendered_board_is_not_mirrored(self, rendered):
        # Flipping a world axis in the renderer leaves every other assertion in this
        # file passing -- corners, centres and pieces all move together, so the
        # labels stay self-consistent. Only the winding order changes, and a
        # mirrored board is one that could never be photographed.
        _, _, samples = rendered
        for sample in samples:
            area = polygon_signed_area(sample.board.corners_px)
            assert area > 0, f"{sample.id}: board renders mirrored (area {area:.1f})"
            assert not is_mirrored(sample.board.corners_px)

    def test_homography_inverse_round_trips(self, rendered):
        _, _, samples = rendered
        for sample in samples:
            forward = np.asarray(sample.board.homography)
            inverse = np.asarray(sample.board.homography_inv)
            np.testing.assert_allclose(forward @ inverse, np.eye(3), atol=1e-8)


class TestLabels:
    def test_fen_matches_the_requested_position(self, rendered):
        _, _, samples = rendered
        for sample in samples:
            assert sample.fen.split()[0] == STARTING_FEN.split()[0]
            assert sample.grid == fen_to_grid(STARTING_FEN)

    def test_one_piece_per_occupied_square(self, rendered):
        _, _, samples = rendered
        expected = len(iter_occupied(fen_to_grid(STARTING_FEN)))
        for sample in samples:
            assert len(sample.pieces) == expected
            occupied = {
                square.name for square in sample.squares if square.occupant != 0
            }
            assert {piece.square for piece in sample.pieces} == occupied

    def test_piece_bases_land_on_their_own_squares(self, rendered):
        _, _, samples = rendered
        for sample in samples:
            centers = {square.name: square.center_px for square in sample.squares}
            pitch = np.linalg.norm(
                np.asarray(sample.squares[0].center_px)
                - np.asarray(sample.squares[1].center_px)
            )
            for piece in sample.pieces:
                offset = np.linalg.norm(
                    np.asarray(piece.base_center_px) - np.asarray(centers[piece.square])
                )
                assert offset < 0.5 * pitch, f"{piece.square} drifted off its square"

    def test_instance_ids_are_unique(self, rendered):
        _, _, samples = rendered
        for sample in samples:
            ids = [piece.instance_id for piece in sample.pieces]
            assert len(set(ids)) == len(ids)


class TestIdPass:
    def test_id_pass_contains_exactly_the_visible_pieces(self, rendered):
        root, _, samples = rendered
        for sample in samples:
            id_image = load_id_image(root / sample.mask_image)
            painted = {
                int(value)
                for value in np.unique(
                    instance_ids(id_image)[role_codes(id_image) == 1]
                )
                if value != 0
            }
            visible = {piece.instance_id for piece in sample.pieces if piece.visible}
            assert painted == visible

    def test_most_pieces_are_visible_from_a_valid_pose(self, rendered):
        _, _, samples = rendered
        for sample in samples:
            visible = sum(1 for piece in sample.pieces if piece.visible)
            assert visible >= 0.8 * len(sample.pieces)

    def test_mask_pixels_belong_to_the_piece_the_labels_claim(self, rendered):
        # Walks up from each square centre until it hits a piece pixel, then asks
        # which piece that is. This ties the rendered pixels back to the FEN, and it
        # catches a self-consistently transposed or mirrored board -- which every
        # algebraic check would happily accept.
        #
        # An exact match is not required: in an oblique view a tall piece standing
        # one square behind genuinely is the first thing above a square's centre.
        # What must hold is that the occluder is a *neighbour*. Any real
        # orientation bug puts the hit several squares away.
        root, _, samples = rendered
        for sample in samples:
            id_image = load_id_image(root / sample.mask_image)
            ids, roles = instance_ids(id_image), role_codes(id_image)
            by_instance = {piece.instance_id: piece for piece in sample.pieces}
            pitch = np.linalg.norm(
                np.asarray(sample.squares[0].center_px)
                - np.asarray(sample.squares[1].center_px)
            )

            checked = exact = 0
            for index, square in enumerate(sample.squares):
                if square.occupant == 0 or not square.in_frame:
                    continue
                rank_index, file_index = divmod(index, 8)
                x, y = (int(round(value)) for value in square.center_px)

                for offset in range(0, int(pitch * 2)):
                    row, column = y - offset, x
                    if not (0 <= row < ids.shape[0] and 0 <= column < ids.shape[1]):
                        break
                    if roles[row, column] != 1:
                        continue

                    checked += 1
                    hit = by_instance[int(ids[row, column])]
                    distance = max(
                        abs(hit.rank_index - rank_index),
                        abs(hit.file_index - file_index),
                    )
                    # How far the occluder can legitimately be depends on the camera
                    # elevation: at the 18-degree end of the range a king two or
                    # three ranks back really does cover the square in front of it,
                    # and that is the hard case the low angles exist for. What no
                    # legitimate pose produces is a hit halfway across the board, so
                    # the bound is set to catch a transposed or rotated grid while
                    # leaving real occlusion alone.
                    assert distance <= 3, (
                        f"{sample.id}: {square.name} resolved to {hit.square}, "
                        f"{distance} squares away -- the board orientation is wrong"
                    )
                    exact += hit.square == square.name
                    break

            assert checked >= 20, "too few squares resolved to make this meaningful"
            assert exact / checked >= 0.6, (
                f"{sample.id}: only {exact}/{checked} squares resolved to their own "
                f"piece, which is more occlusion than this pose should produce"
            )

    def test_king_and_queen_are_not_transposed(self, rendered):
        # d/e file confusion is the classic orientation bug and survives every
        # symmetric check, so it gets its own assertion.
        _, _, samples = rendered
        for sample in samples:
            by_square = {piece.square: piece for piece in sample.pieces}
            assert CLASS_NAMES[by_square["e1"].class_id] == "white_king"
            assert CLASS_NAMES[by_square["d1"].class_id] == "white_queen"
            assert CLASS_NAMES[by_square["e8"].class_id] == "black_king"
            assert CLASS_NAMES[by_square["d8"].class_id] == "black_queen"


class TestOverlay:
    def test_overlay_renders(self, rendered):
        root, _, samples = rendered
        image = render_overlay(samples[0], root / samples[0].image)
        assert image.width == samples[0].width
        assert image.height > samples[0].height  # header strip

    def test_contact_sheet_tiles_samples(self, rendered):
        root, _, samples = rendered
        sheet = contact_sheet(
            [render_overlay(sample, root / sample.image) for sample in samples],
            columns=2,
        )
        assert sheet.width == 2 * 420


class TestDeterminism:
    def test_the_same_seed_reproduces_the_same_labels(self, render_root):
        root_a = render_root / "det_a"
        root_b = render_root / "det_b"
        for root in (root_a, root_b):
            _, _, shard = make_run(root, count=1, seed=99)
            assert run_blender(shard).returncode == 0

        raw_a = load_raw(root_a / "raw_labels" / "000000.json")
        raw_b = load_raw(root_b / "raw_labels" / "000000.json")
        assert raw_a["fen"] == raw_b["fen"]
        np.testing.assert_allclose(
            raw_a["board"]["corners_px"], raw_b["board"]["corners_px"], atol=1e-9
        )
        np.testing.assert_allclose(
            raw_a["square_centers_px"], raw_b["square_centers_px"], atol=1e-9
        )


class TestCapturedPieces:
    """Pieces beside the board must be labelled, masked, and off the grid."""

    @pytest.fixture(scope="class")
    def captured(self, render_root):
        root = render_root / "captured"
        _, _, shard = make_run(
            root,
            count=2,
            seed=555,
            overrides={
                "pieces": {
                    "captured_probability": 1.0,
                    "captured_count": {"min": 4, "max": 8},
                },
                "positions": {
                    "pgn_paths": [],
                    "weight_pgn": 0.0,
                    "weight_random": 1.0,
                    "random_min_pieces": 4,
                    "random_max_pieces": 12,
                },
            },
        )
        assert run_blender(shard).returncode == 0
        return [
            build_sample(
                load_raw(root / "raw_labels" / f"{index:06d}.json"),
                image_rel_path=f"images/{index:06d}.png",
                mask_rel_path=f"id_pass/{index:06d}.png",
            )
            for index in range(2)
        ]

    def test_captured_pieces_are_rendered_and_labelled(self, captured):
        for sample in captured:
            off_board = [piece for piece in sample.pieces if not piece.on_board]
            assert off_board, "expected captured pieces at probability 1.0"
            for piece in off_board:
                assert piece.square is None
                assert piece.rank_index is None and piece.file_index is None

    def test_captured_pieces_stay_out_of_the_grid(self, captured):
        for sample in captured:
            on_board = [piece for piece in sample.pieces if piece.on_board]
            occupied = sum(1 for square in sample.squares if square.occupant)
            # The grid describes the board only; the pile beside it is extra.
            assert len(on_board) == occupied
            assert len(sample.pieces) > occupied

    def test_captured_pieces_get_their_own_masks(self, captured):
        for sample in captured:
            for piece in sample.pieces:
                if piece.on_board or not piece.visible:
                    continue
                assert piece.mask is not None
                assert piece.visible_pixels > 0

    def test_instance_ids_do_not_collide(self, captured):
        for sample in captured:
            ids = [piece.instance_id for piece in sample.pieces]
            assert len(set(ids)) == len(ids)

    def test_the_board_is_not_floating_above_the_table(self, captured):
        # Captured pieces rest on the table, so if the table were offset from the
        # board's underside they would hover. Their masks touching the rendered
        # image at all is the cheap end-to-end check that the two agree.
        for sample in captured:
            off_board = [
                piece for piece in sample.pieces if not piece.on_board and piece.visible
            ]
            assert off_board
            assert all(piece.bbox is not None for piece in off_board)
