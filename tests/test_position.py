"""Reading an 8x8 position out of detections plus board geometry.

mAP cannot answer the question these tests ask. A detector that puts a perfect box
one square to the left scores well and reads the board wrongly, so square assignment
needs its own tests -- built on a known homography, so an error here is an error in
the assignment and not in the geometry it rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from chesssight.data.fen import BOARD_SIZE
from chesssight.data.geometry import apply_homography, board_to_image_homography
from chesssight.train.labels import BOARD_INDEX, CORNER_INDEX, class_id_to_index
from chesssight.train.position import (
    best_rotation,
    foot,
    grid_from,
    read_position,
    score_grid,
)

#: A board filling a 800x800 image, seen square-on. Deliberately not a perspective
#: view: these tests are about assignment, and a projective board would make a
#: failure ambiguous between the two.
CORNERS = [[100.0, 100.0], [700.0, 100.0], [700.0, 700.0], [100.0, 700.0]]
HOMOGRAPHY = board_to_image_homography(np.asarray(CORNERS))
SQUARE_PX = 600.0 / BOARD_SIZE


def detection_on(rank: int, file: int, class_id: int, score: float = 0.9) -> dict:
    """A piece box whose foot sits at the centre of one square."""
    centre = apply_homography(HOMOGRAPHY, [[file + 0.5, rank + 0.5]])[0]
    x, y = float(centre[0]), float(centre[1])
    # Tall box standing on the square, as a real piece's box is.
    return {
        "label": class_id_to_index(class_id),
        "score": score,
        "box": [x - SQUARE_PX / 3, y - SQUARE_PX * 2, x + SQUARE_PX / 3, y],
    }


def test_the_foot_is_near_the_bottom_centre_not_the_middle():
    # Pieces are tall and shot from above the table; the box centre floats up the
    # piece and lands a square or more behind where it actually stands.
    x, y = foot([10.0, 20.0, 30.0, 60.0])
    assert x == 20.0  # horizontally centred
    middle, bottom = 40.0, 60.0
    assert bottom - 0.1 * (bottom - middle) <= y <= bottom
    assert y > middle


def test_the_foot_sits_just_inside_the_bottom_edge():
    # Not *on* the edge: the bottom of a box is the nearest-to-camera point of the
    # base, forward of where the piece's axis meets the board, so a piece standing
    # near a square boundary is otherwise read one square over. Worth 5 points of
    # board-exact accuracy on synthetic boards.
    assert foot([0.0, 0.0, 10.0, 100.0])[1] < 100.0


class TestGridFrom:
    def test_a_piece_lands_on_its_own_square(self):
        grid = grid_from([detection_on(0, 0, 5)], HOMOGRAPHY)
        assert grid[0][0] == 5
        assert sum(sum(row) for row in grid) == 5  # nothing anywhere else

    @pytest.mark.parametrize("rank,file", [(0, 0), (0, 7), (7, 0), (7, 7), (3, 4)])
    def test_every_corner_and_the_middle_assign_correctly(self, rank, file):
        # An off-by-one in the u/v -> index conversion shows up at the edges first.
        grid = grid_from([detection_on(rank, file, 11)], HOMOGRAPHY)
        assert grid[rank][file] == 11

    def test_two_pieces_on_one_square_keep_the_better_one(self):
        grid = grid_from(
            [
                detection_on(2, 3, 1, score=0.4),
                detection_on(2, 3, 9, score=0.8),
            ],
            HOMOGRAPHY,
        )
        assert grid[2][3] == 9

    def test_a_piece_beside_the_board_is_not_placed(self):
        # Captured pieces stand off the board; the synthetic set labels them, and
        # they must not be forced onto a square.
        outside = detection_on(0, 0, 1)
        outside["box"] = [10.0, 10.0, 40.0, 60.0]
        assert grid_from([outside], HOMOGRAPHY) == [
            [0] * BOARD_SIZE for _ in range(BOARD_SIZE)
        ]

    def test_the_board_and_corner_classes_are_not_pieces(self):
        board = detection_on(4, 4, 1)
        board["label"] = BOARD_INDEX
        corner = detection_on(2, 2, 1)
        corner["label"] = CORNER_INDEX
        grid = grid_from([board, corner], HOMOGRAPHY)
        assert all(value == 0 for row in grid for value in row)


class TestScoring:
    def test_a_perfect_read_is_exact(self):
        truth = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        truth[0][0], truth[7][7] = 1, 12
        accuracy, exact = score_grid(truth, truth)
        assert accuracy == 1.0 and exact

    def test_one_wrong_square_costs_exactness(self):
        truth = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        predicted = [row[:] for row in truth]
        predicted[3][3] = 7
        accuracy, exact = score_grid(predicted, truth)
        assert not exact
        assert accuracy == pytest.approx(63 / 64)

    def test_a_rotated_read_scores_under_best_rotation(self):
        # Four interchangeable corners fix the geometry only up to a rotation, so a
        # correct read of a rotated board must not be reported as a failure.
        truth = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        truth[0][1] = 4
        rotated = np.rot90(np.asarray(truth), -1).tolist()
        assert best_rotation(rotated, truth) == (1.0, True)
        # ...while the unrotated comparison does see it as wrong.
        assert not score_grid(rotated, truth)[1]


class TestReadPosition:
    def test_missing_corners_give_no_position_rather_than_a_guess(self):
        assert read_position([detection_on(0, 0, 1)], None) is None
        assert read_position([detection_on(0, 0, 1)], CORNERS[:3]) is None

    def test_degenerate_corners_are_a_failure_not_a_crash(self):
        collapsed = [[100.0, 100.0]] * 4
        assert read_position([detection_on(0, 0, 1)], collapsed) is None

    def test_a_full_read_round_trips(self):
        truth = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        detections = []
        for rank, file, class_id in [(0, 0, 4), (1, 5, 1), (6, 2, 7), (7, 7, 10)]:
            truth[rank][file] = class_id
            detections.append(detection_on(rank, file, class_id))
        grid = read_position(detections, CORNERS)
        assert grid == truth


def test_position_threshold_is_below_a_detection_operating_point() -> None:
    """Reading a board wants recall; detecting objects wants balance.

    Pinned because the two thresholds are easy to conflate, and using the
    detection one for board reading costs 37 points of board accuracy.
    """
    from chesssight.train.predict_position import POSITION_THRESHOLD

    assert 0.0 < POSITION_THRESHOLD < 0.33
