"""ChessReD ingestion.

The category mapping gets the most attention here, because ChessReD and this
project order their pieces differently and a positional mapping would be wrong in
a way that no shape or count check would ever notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from chesssight.data.chessred import (
    CORNER_KEYS,
    ChessReDError,
    build_category_map,
    grid_from_pieces,
)
from chesssight.data.fen import CLASS_NAMES, EMPTY, LETTER_TO_CLASS

# ChessReD's own ordering, verbatim from its annotations.json.
CHESSRED_CATEGORIES = [
    {"id": 0, "name": "white-pawn"},
    {"id": 1, "name": "white-rook"},
    {"id": 2, "name": "white-knight"},
    {"id": 3, "name": "white-bishop"},
    {"id": 4, "name": "white-queen"},
    {"id": 5, "name": "white-king"},
    {"id": 6, "name": "black-pawn"},
    {"id": 7, "name": "black-rook"},
    {"id": 8, "name": "black-knight"},
    {"id": 9, "name": "black-bishop"},
    {"id": 10, "name": "black-queen"},
    {"id": 11, "name": "black-king"},
    {"id": 12, "name": "empty"},
]


class TestCategoryMap:
    def test_maps_by_name_not_position(self):
        mapping = build_category_map(CHESSRED_CATEGORIES)
        # ChessReD id 1 is a rook; a positional mapping would make it a knight.
        assert mapping[1] == LETTER_TO_CLASS["R"]
        assert mapping[2] == LETTER_TO_CLASS["N"]
        assert mapping[3] == LETTER_TO_CLASS["B"]
        assert mapping[7] == LETTER_TO_CLASS["r"]
        assert mapping[8] == LETTER_TO_CLASS["n"]

    def test_the_orderings_really_do_differ(self):
        # If this ever stops being true the mapping is trivial and this whole
        # module can be simplified -- so assert it rather than assume it.
        theirs = [c["name"].replace("-", "_") for c in CHESSRED_CATEGORIES[:12]]
        ours = list(CLASS_NAMES[1:13])
        assert theirs != ours

    def test_every_category_is_covered(self):
        mapping = build_category_map(CHESSRED_CATEGORIES)
        assert len(mapping) == 13
        assert mapping[12] == EMPTY

    def test_names_round_trip_to_the_same_piece(self):
        mapping = build_category_map(CHESSRED_CATEGORIES)
        for category in CHESSRED_CATEGORIES:
            if category["name"] == "empty":
                continue
            assert CLASS_NAMES[mapping[category["id"]]] == category["name"].replace(
                "-", "_"
            )

    def test_an_unknown_category_is_rejected(self):
        with pytest.raises(ChessReDError, match="unknown ChessReD category"):
            build_category_map([{"id": 0, "name": "white-dragon"}])


class TestGridFromPieces:
    def test_builds_the_grid_and_skips_empties(self):
        mapping = build_category_map(CHESSRED_CATEGORIES)
        records = [
            {"category_id": 5, "chessboard_position": "e1"},  # white king
            {"category_id": 11, "chessboard_position": "e8"},  # black king
            {"category_id": 12, "chessboard_position": "d4"},  # empty
        ]
        grid, occupied = grid_from_pieces(records, mapping)

        assert len(occupied) == 2
        assert grid[7][4] == LETTER_TO_CLASS["K"]  # e1
        assert grid[0][4] == LETTER_TO_CLASS["k"]  # e8
        assert grid[4][3] == EMPTY  # d4 stays empty

    def test_a_rook_does_not_become_a_knight(self):
        mapping = build_category_map(CHESSRED_CATEGORIES)
        grid, _ = grid_from_pieces(
            [{"category_id": 1, "chessboard_position": "a1"}], mapping
        )
        assert grid[7][0] == LETTER_TO_CLASS["R"]
        assert grid[7][0] != LETTER_TO_CLASS["N"]


def test_corner_keys_are_the_four_chessred_names():
    assert set(CORNER_KEYS) == {
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    }


def test_corner_resolution_scores_orderings_by_placement():
    """The resolver must prefer the ordering that puts pieces on their squares."""
    from chesssight.data.chessred import _corner_array, _placement_error
    from chesssight.data.fen import empty_grid

    # A square board, 800 px, with a piece whose foot sits on a1's centre.
    corners = {
        "top_left": [0.0, 0.0],
        "top_right": [800.0, 0.0],
        "bottom_right": [800.0, 800.0],
        "bottom_left": [0.0, 800.0],
    }
    good = ("top_left", "top_right", "bottom_right", "bottom_left")
    # a1 is grid[7][0]; with this mapping its centre is at (50, 750).
    records = [
        {
            "category_id": 1,
            "chessboard_position": "a1",
            "bbox": [30.0, 700.0, 40.0, 50.0],
        }
    ]

    error_good = _placement_error(_corner_array(corners, good), records, empty_grid())
    rotated = ("top_right", "bottom_right", "bottom_left", "top_left")
    error_rotated = _placement_error(
        _corner_array(corners, rotated), records, empty_grid()
    )

    assert error_good == pytest.approx(0.0, abs=1.0)
    assert error_rotated > 100.0


def test_placement_error_uses_the_foot_not_the_centre():
    """A piece stands on its square, so the box bottom is what should coincide."""
    from chesssight.data.chessred import _corner_array, _placement_error
    from chesssight.data.fen import empty_grid

    corners = {
        "top_left": [0.0, 0.0],
        "top_right": [800.0, 0.0],
        "bottom_right": [800.0, 800.0],
        "bottom_left": [0.0, 800.0],
    }
    order = ("top_left", "top_right", "bottom_right", "bottom_left")
    # Tall piece on a1: its centre is far above the square, its foot is on it.
    tall = [
        {
            "category_id": 1,
            "chessboard_position": "a1",
            "bbox": [30.0, 550.0, 40.0, 200.0],
        }
    ]
    assert _placement_error(_corner_array(corners, order), tall, empty_grid()) < 5.0


def test_canonical_corner_reference_is_the_project_convention():
    from chesssight.data.chessred import canonical_corner_reference
    from chesssight.data.geometry import BOARD_CORNERS

    np.testing.assert_array_equal(
        canonical_corner_reference(), np.asarray(BOARD_CORNERS)
    )
