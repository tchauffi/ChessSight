"""Warping a board to a canonical square.

A wrong warp is the worst kind of bug here: the output still looks like a clean
rectified board, the classifier trains happily on it, and the position comes out
consistently one square off. So these tests pin the geometry numerically against
a known homography rather than checking that an image came back.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from chesssight.data.fen import BOARD_SIZE
from chesssight.train.rectify import (
    FAR_MARGIN,
    SIDE_MARGIN,
    playing_area_px,
    rectify,
    square_centre_px,
)

SIZE = 256
#: A board filling a 400x400 image, square-on, so the warp should be close to a
#: crop-and-scale and any error is unambiguous.
SQUARE_ON = [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]


def checkerboard(size: int = 400) -> Image.Image:
    """The board of SQUARE_ON drawn into a 400x400 image, a8 light."""
    array = np.full((size, size, 3), 30, dtype=np.uint8)
    cell = 200 // BOARD_SIZE
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            value = 220 if (rank + file) % 2 == 0 else 60
            y0, x0 = 100 + rank * cell, 100 + file * cell
            array[y0 : y0 + cell, x0 : x0 + cell] = value
    return Image.fromarray(array)


class TestPlayingArea:
    def test_the_board_is_centred_with_margin_around_it(self):
        x0, y0, x1, y1 = playing_area_px(SIZE)
        assert 0 < x0 < x1 < SIZE
        assert 0 < y0 < y1 < SIZE
        # Symmetric: the same margin each side horizontally, the far margin
        # applied at both ends vertically.
        assert x0 == pytest.approx(SIZE - x1)
        assert y0 == pytest.approx(SIZE - y1)

    def test_the_margin_is_actually_beyond_the_board(self):
        # If the playing area filled the output, a tall piece's smear would be
        # cropped away -- which is the failure this margin exists to prevent.
        x0, _, x1, _ = playing_area_px(SIZE)
        assert x1 - x0 < SIZE

    def test_a_bigger_far_margin_shrinks_the_board_vertically(self):
        _, tight_y0, _, tight_y1 = playing_area_px(SIZE, far=1.0)
        _, loose_y0, _, loose_y1 = playing_area_px(SIZE, far=4.0)
        assert (loose_y1 - loose_y0) < (tight_y1 - tight_y0)


class TestSquareCentres:
    def test_a1_and_h8_are_at_opposite_corners(self):
        a8 = square_centre_px(0, 0, SIZE)
        h1 = square_centre_px(7, 7, SIZE)
        assert a8[0] < h1[0] and a8[1] < h1[1]

    def test_centres_are_evenly_spaced(self):
        xs = [square_centre_px(0, f, SIZE)[0] for f in range(BOARD_SIZE)]
        gaps = np.diff(xs)
        assert np.allclose(gaps, gaps[0])

    def test_every_centre_lands_inside_the_playing_area(self):
        x0, y0, x1, y1 = playing_area_px(SIZE)
        for rank in range(BOARD_SIZE):
            for file in range(BOARD_SIZE):
                cx, cy = square_centre_px(rank, file, SIZE)
                assert x0 < cx < x1 and y0 < cy < y1


class TestRectify:
    def test_the_output_has_the_requested_size(self):
        assert rectify(checkerboard(), SQUARE_ON, size=SIZE).size == (SIZE, SIZE)

    def test_the_squares_land_where_the_geometry_says(self):
        # The load-bearing test: sample each square's centre in the rectified
        # image and check the light/dark pattern matches the board's own. A
        # transposed or flipped warp fails here and nowhere else.
        warped = np.asarray(rectify(checkerboard(), SQUARE_ON, size=SIZE).convert("L"))
        for rank in range(BOARD_SIZE):
            for file in range(BOARD_SIZE):
                cx, cy = square_centre_px(rank, file, SIZE)
                value = warped[int(cy), int(cx)]
                light = (rank + file) % 2 == 0
                assert (value > 128) == light, f"square {rank},{file}"

    def test_the_margin_contains_the_surroundings_not_the_board(self):
        # Just outside the playing area should be the dark backdrop, not more
        # squares -- otherwise the warp is scaled wrongly.
        warped = np.asarray(rectify(checkerboard(), SQUARE_ON, size=SIZE).convert("L"))
        x0, y0, _, _ = playing_area_px(SIZE)
        assert warped[int(y0) - 6, int(x0) - 6] < 60

    def test_a_rotated_quad_still_rectifies(self):
        # Corners given from a different starting corner rotate the board; the
        # warp must still produce a full board rather than a sliver.
        rolled = SQUARE_ON[1:] + SQUARE_ON[:1]
        warped = np.asarray(rectify(checkerboard(), rolled, size=SIZE).convert("L"))
        centre = warped[
            int(playing_area_px(SIZE)[1]) + 10 : int(playing_area_px(SIZE)[3]) - 10
        ]
        assert centre.std() > 40  # a real chequer pattern, not flat fill

    def test_margins_are_configurable_end_to_end(self):
        tight = rectify(checkerboard(), SQUARE_ON, size=SIZE, side=0.25, far=0.25)
        loose = rectify(checkerboard(), SQUARE_ON, size=SIZE, side=2.0, far=2.0)
        # The looser warp packs the board into fewer pixels, so its centre
        # crop covers more of the surrounding table.
        assert (
            np.asarray(tight.convert("L")).std() > np.asarray(loose.convert("L")).std()
        )


def test_the_defaults_keep_room_beyond_the_far_edge():
    # Stated as a property rather than a magic number: the far margin has to be
    # larger than the side margin, because that is where the piece smear goes.
    assert FAR_MARGIN > SIDE_MARGIN
