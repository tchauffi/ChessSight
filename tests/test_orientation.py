"""Resolving which corner is a8.

The failure this file is built to catch is a *silent* one. Every rotation of a
board is a perfectly plausible board, so a sign error in the colour test or a
mismatch between how the grid and the luminance map are turned does not raise
anything -- it returns a confident, consistent, wrong position. So the tests
below check each half of the decision on its own, then check that all four
rotations of a known board come back to the same answer.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from chesssight.data.fen import BOARD_SIZE, fen_to_grid
from chesssight.train.orientation import (
    colour_score,
    orient,
    orient_position,
    pawn_home_score,
    piece_score,
    rotate,
    square_luminance,
)

START = fen_to_grid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")


def canonical_luminance() -> np.ndarray:
    """A board whose a8 square is light, as a real board's is."""
    ranks, files = np.indices((BOARD_SIZE, BOARD_SIZE))
    return np.where((ranks + files) % 2 == 0, 0.8, 0.2)


def checkerboard_image(size: int = 512) -> Image.Image:
    """An 8x8 board filling the image, light at the top-left."""
    cell = size // BOARD_SIZE
    array = np.zeros((size, size), dtype=np.uint8)
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            value = 210 if (rank + file) % 2 == 0 else 45
            array[rank * cell : (rank + 1) * cell, file * cell : (file + 1) * cell] = (
                value
            )
    return Image.fromarray(array).convert("RGB")


class TestColour:
    def test_a_canonical_board_scores_positive(self):
        # a8 is a light square; that is what "positive" means here.
        assert colour_score(canonical_luminance()) > 0

    def test_a_quarter_turn_flips_the_sign(self):
        # This is the whole mechanism: 90 degrees moves a dark square to a8, so
        # two of the four candidates are eliminated by pixels alone.
        canonical = canonical_luminance()
        assert colour_score(rotate(canonical, 1)) < 0
        assert colour_score(rotate(canonical, 3)) < 0

    def test_a_half_turn_does_not(self):
        # 180 degrees preserves the colour pattern, which is exactly why the
        # piece vote is needed and why colour alone cannot finish the job.
        canonical = canonical_luminance()
        assert colour_score(rotate(canonical, 2)) == pytest.approx(
            colour_score(canonical)
        )

    def test_a_board_with_no_contrast_scores_near_zero(self):
        assert colour_score(np.full((BOARD_SIZE, BOARD_SIZE), 0.5)) == pytest.approx(
            0.0
        )


class TestPieces:
    def test_white_at_the_bottom_scores_positive(self):
        assert piece_score(START) > 0.9

    def test_a_half_turn_reverses_it(self):
        assert piece_score(rotate(START, 2).tolist()) == pytest.approx(
            -piece_score(START)
        )

    def test_an_empty_board_has_no_opinion(self):
        assert piece_score([[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]) == 0.0

    def test_black_alone_still_votes(self):
        # A black-only ending must not abstain: black at the top is as much
        # evidence as white at the bottom.
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board[0][4] = 12  # black king on the far rank
        assert piece_score(board) > 0


class TestPawnHome:
    def test_the_starting_position_is_fully_home(self):
        assert pawn_home_score(START) == pytest.approx(1.0)

    def test_a_half_turn_reads_every_pawn_as_invading(self):
        assert pawn_home_score(rotate(START, 2).tolist()) == pytest.approx(-1.0)

    def test_one_colour_without_pawns_abstains(self):
        # A lone runner is exactly the pawn whose position lies about the
        # orientation; without the other colour to balance it, no vote.
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board[1][3] = 1  # white pawn one step from promotion
        assert pawn_home_score(board) == 0.0

    def test_a_pawnless_board_abstains(self):
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board[0][4] = 12
        board[7][4] = 6
        assert pawn_home_score(board) == 0.0


class TestOrient:
    def test_pawns_overrule_a_lying_material_vote(self):
        # A game photographed with White sitting at the top: the material vote
        # reads it flipped (white pieces far, black pieces near), and only the
        # pawns -- each on its own half -- know which way the game ran. This is
        # the real failure that cost 45 of ChessReD val's 330 boards.
        grid = fen_to_grid("QRN5/8/1p6/8/8/1P6/8/qrn5 w - - 0 1")
        assert piece_score(grid) < 0  # material alone would flip this board
        turns, evidence = orient(grid, canonical_luminance())
        assert turns == 0
        assert evidence["pawns"] == pytest.approx(1.0)

    @pytest.mark.parametrize("turns", [0, 1, 2, 3])
    def test_every_rotation_of_a_board_comes_back(self, turns):
        # A quad ordered from a different starting corner reads the board rotated
        # by that many quarter-turns; orient must undo exactly that.
        rotated_grid = rotate(START, turns).tolist()
        rotated_luminance = rotate(canonical_luminance(), turns)
        recovered, _ = orient(rotated_grid, rotated_luminance)
        assert rotate(rotated_grid, recovered).tolist() == START

    def test_the_colour_filter_leaves_two_candidates(self):
        _, evidence = orient(START, canonical_luminance())
        assert evidence["candidates"] == 2  # the 180-degree pair
        assert evidence["colour"] > 0

    def test_an_empty_board_reports_the_tie_instead_of_hiding_it(self):
        # Colour narrows this to two, and nothing can choose between them. The
        # caller is told, rather than handed a confident coin flip.
        empty = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        _, evidence = orient(empty, canonical_luminance())
        assert evidence["pieces"] == 0.0
        assert evidence["margin"] == 0.0

    def test_a_decided_board_reports_a_margin(self):
        _, evidence = orient(START, canonical_luminance())
        assert evidence["margin"] > 1.0  # +1 against -1


class TestOrientPosition:
    def test_the_corners_are_rotated_with_the_grid(self):
        # If only the grid turned, the returned position and the returned
        # homography would disagree about which way up the board is -- and
        # nothing downstream would notice.
        image = checkerboard_image()
        corners = [[0.0, 0.0], [512.0, 0.0], [512.0, 512.0], [0.0, 512.0]]
        rolled = corners[1:] + corners[:1]
        grid, ordered, _ = orient_position(rotate(START, 1).tolist(), rolled, image)
        assert grid == START
        assert ordered == corners

    def test_an_already_oriented_board_is_left_alone(self):
        image = checkerboard_image()
        corners = [[0.0, 0.0], [512.0, 0.0], [512.0, 512.0], [0.0, 512.0]]
        grid, ordered, evidence = orient_position(START, corners, image)
        assert grid == START
        assert ordered == corners
        assert evidence["colour"] > 0


def test_square_luminance_reads_the_squares_not_the_grid_lines():
    # Sampling the full square would average in its high-contrast border and
    # collapse the light/dark difference the colour test depends on.
    from chesssight.data.geometry import board_to_image_homography

    image = checkerboard_image()
    corners = [[0.0, 0.0], [512.0, 0.0], [512.0, 512.0], [0.0, 512.0]]
    luminance = square_luminance(
        image, board_to_image_homography(np.asarray(corners, dtype=np.float64))
    )
    assert luminance[0][0] == pytest.approx(210 / 255, abs=0.02)
    assert luminance[0][1] == pytest.approx(45 / 255, abs=0.02)
