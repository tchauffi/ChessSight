"""Combining the two position readers.

The rule is asymmetric on purpose -- occupancy from one model, naming from the
other -- so the tests are about which model wins where, and specifically about
the case that motivated the whole thing: a piece the detector missed must
survive, and a piece the detector named must be renamed.
"""

from __future__ import annotations

import pytest

from chesssight.data.fen import BOARD_SIZE, LETTER_TO_CLASS
from chesssight.train.fuse import agreement, fuse

EMPTY = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]


def board(**squares: int) -> list[list[int]]:
    """A grid from ``r0f0=class`` keyword pairs, for readable fixtures."""
    grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for key, value in squares.items():
        rank, file = int(key[1]), int(key[3])
        grid[rank][file] = value
    return grid


class TestOccupancy:
    def test_the_grid_classifier_decides_a_square_is_empty(self):
        # It misses 0.06% of occupied squares against the detector's 6.66%, so
        # its "nothing here" is the more reliable of the two.
        fused = fuse(EMPTY, board(r3f3=LETTER_TO_CLASS["Q"]))
        assert fused == EMPTY

    def test_a_piece_the_detector_missed_survives(self):
        # The 6.66%. Emptying this square would hand back the detector's single
        # largest error, which is the reason the fusion exists.
        grid = board(r2f5=LETTER_TO_CLASS["n"])
        assert fuse(grid, EMPTY)[2][5] == LETTER_TO_CLASS["n"]


class TestNaming:
    def test_the_detector_renames_what_it_found(self):
        # It misnames 1.05% against the grid's 12.65%, so where it has an
        # opinion it wins.
        grid = board(r4f4=LETTER_TO_CLASS["b"])
        detector = board(r4f4=LETTER_TO_CLASS["n"])
        assert fuse(grid, detector)[4][4] == LETTER_TO_CLASS["n"]

    def test_colour_comes_from_the_detector_too(self):
        # The detector's wrong-colour rate is 0.00%; the grid's is 0.25%.
        grid = board(r0f0=LETTER_TO_CLASS["P"])
        detector = board(r0f0=LETTER_TO_CLASS["p"])
        assert fuse(grid, detector)[0][0] == LETTER_TO_CLASS["p"]

    def test_agreement_is_left_alone(self):
        grid = board(r1f1=LETTER_TO_CLASS["R"])
        assert fuse(grid, grid) == grid


class TestShape:
    def test_a_wrong_sized_grid_is_rejected(self):
        with pytest.raises(ValueError, match="8x8"):
            fuse([[0] * BOARD_SIZE], EMPTY)

    def test_the_result_is_a_fresh_grid(self):
        grid = board(r0f0=LETTER_TO_CLASS["K"])
        fused = fuse(grid, EMPTY)
        fused[0][0] = 0
        assert grid[0][0] == LETTER_TO_CLASS["K"]  # inputs untouched


class TestAgreement:
    def test_it_counts_each_kind_of_disagreement(self):
        grid = board(r0f0=LETTER_TO_CLASS["P"], r1f1=LETTER_TO_CLASS["n"])
        detector = board(r0f0=LETTER_TO_CLASS["B"], r2f2=LETTER_TO_CLASS["q"])
        counts = agreement(grid, detector)
        assert counts["both"] == 1
        assert counts["named_differently"] == 1
        assert counts["grid_only"] == 1  # the detector's miss
        assert counts["detector_only"] == 1
        assert counts["neither"] == BOARD_SIZE * BOARD_SIZE - 3

    def test_a_detector_only_square_is_dropped_by_the_fusion(self):
        # Consistency between the diagnostic and the rule: anything counted as
        # detector_only is a square the fusion discards, and if that number ever
        # grows the occupancy choice needs revisiting.
        detector = board(r5f5=LETTER_TO_CLASS["q"])
        assert agreement(EMPTY, detector)["detector_only"] == 1
        assert fuse(EMPTY, detector)[5][5] == 0
