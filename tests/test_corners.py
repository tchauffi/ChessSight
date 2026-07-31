"""Carrying board corners as a detection class, and reading them back.

The point of the exercise is a *position*, not four more boxes: corner centres give
the homography that says which square each piece stands on. So these tests care about
whether a quad survives the round trip, not merely whether four boxes come back.
"""

from __future__ import annotations

import numpy as np
import pytest

from chesssight.train.corners import (
    corner_error,
    homography_from,
    order_clockwise,
    select_quad,
)
from chesssight.train.labels import BOARD_INDEX, CORNER_INDEX


def corner(x: float, y: float, score: float = 0.9, size: float = 20.0) -> dict:
    return {
        "label": CORNER_INDEX,
        "name": "corner",
        "score": score,
        "box": [x - size / 2, y - size / 2, x + size / 2, y + size / 2],
    }


SQUARE = [(100.0, 100.0), (400.0, 100.0), (400.0, 400.0), (100.0, 400.0)]


class TestSelectQuad:
    def test_four_corners_round_trip_through_their_boxes(self):
        quad = select_quad([corner(x, y) for x, y in SQUARE])
        assert quad is not None
        # Same four points, whatever order they came back in.
        assert sorted(tuple(p) for p in quad) == sorted(SQUARE)

    def test_fewer_than_four_is_a_failure_not_a_guess(self):
        assert select_quad([corner(x, y) for x, y in SQUARE[:3]]) is None

    def test_two_boxes_on_one_corner_do_not_make_a_degenerate_quad(self):
        # The failure this guards: the model puts two boxes on the same physical
        # corner, the top four scores include both, and the homography that comes
        # out is degenerate rather than obviously wrong.
        detections = [
            corner(100, 100, score=0.99),
            corner(104, 103, score=0.98),  # the same corner again
            corner(400, 100, score=0.97),
            corner(400, 400, score=0.96),
            corner(100, 400, score=0.95),
        ]
        quad = select_quad(detections)
        assert quad is not None
        assert sorted(tuple(p) for p in quad) == sorted(SQUARE)

    def test_pieces_and_the_board_are_ignored(self):
        detections = [corner(x, y) for x, y in SQUARE] + [
            {"label": 0, "name": "white_pawn", "score": 0.99, "box": [0, 0, 10, 10]},
            {
                "label": BOARD_INDEX,
                "name": "board",
                "score": 0.99,
                "box": [0, 0, 500, 500],
            },
        ]
        quad = select_quad(detections)
        assert quad is not None and len(quad) == 4


class TestOrdering:
    def test_a_rotated_board_is_ordered_around_its_edge_not_by_coordinate(self):
        # A board seen from a low, rotated viewpoint: sorting by x or y interleaves
        # opposite corners and yields a bow-tie. Ordering by angle does not.
        diamond = [(250.0, 80.0), (420.0, 250.0), (250.0, 420.0), (80.0, 250.0)]
        ordered = order_clockwise(diamond)
        assert not _self_intersecting(ordered)

    def test_ordering_is_stable_for_the_same_board(self):
        first = order_clockwise(list(SQUARE))
        second = order_clockwise(list(reversed(SQUARE)))
        assert first == second


def _self_intersecting(quad: list[list[float]]) -> bool:
    """Whether the polygon's two diagonals fail to cross -- a bow-tie."""

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    a, b, c, d = quad
    # In a simple quadrilateral the diagonals a-c and b-d intersect.
    d1 = cross(a, c, b) * cross(a, c, d)
    d2 = cross(b, d, a) * cross(b, d, c)
    return not (d1 < 0 and d2 < 0)


class TestHomography:
    def test_a_clean_quad_gives_a_usable_homography(self):
        matrix = homography_from([corner(x, y) for x, y in SQUARE])
        assert matrix is not None
        # The board plane's own corners must land back on the detected ones.
        board = np.array(
            [[0.0, 0.0, 1.0], [8.0, 0.0, 1.0], [8.0, 8.0, 1.0], [0.0, 8.0, 1.0]]
        )
        projected = (matrix @ board.T).T
        projected = projected[:, :2] / projected[:, 2:]
        assert corner_error(projected.tolist(), [list(p) for p in SQUARE]) < 1e-6

    def test_too_few_corners_yields_no_geometry_rather_than_raising(self):
        # A frame with no readable board is normal, not exceptional: the caller
        # falls back to the previous frame's geometry.
        assert homography_from([corner(100, 100)]) is None


class TestCornerError:
    def test_error_does_not_punish_a_different_starting_corner(self):
        rotated = SQUARE[2:] + SQUARE[:2]
        assert (
            corner_error([list(p) for p in rotated], [list(p) for p in SQUARE]) == 0.0
        )

    def test_error_is_the_mean_distance(self):
        moved = [(x + 3.0, y + 4.0) for x, y in SQUARE]  # 5px each
        assert corner_error(
            [list(p) for p in moved], [list(p) for p in SQUARE]
        ) == pytest.approx(5.0)


class TestCornerAnnotations:
    def test_labels_reject_a_corner_as_a_class_id(self):
        # A corner is a keypoint, not a piece; converting it to a ChessSight class id
        # would silently produce class 14 and shift every downstream lookup.
        from chesssight.train.labels import index_to_class_id, is_piece

        with pytest.raises(ValueError, match="keypoint"):
            index_to_class_id(CORNER_INDEX)
        assert not is_piece(CORNER_INDEX)
        assert not is_piece(BOARD_INDEX)
        assert is_piece(0)
