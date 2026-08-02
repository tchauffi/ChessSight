"""Vector chess pieces.

Drawing code fails quietly -- a piece that renders as nothing, or two pieces that
render identically, produces a diagram that still looks like a diagram. These
tests check the properties a reader actually depends on: every piece draws
something, no two look the same, both colours share a silhouette, and nothing
escapes its square.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageDraw

from chesssight.data.fen import CLASS_TO_LETTER, LETTER_TO_CLASS
from chesssight.train.glyphs import SHAPES, draw_piece

SIZE = 64
WHITE_BODY = (248, 246, 240)
BLACK_BODY = (26, 26, 30)


def render(class_id: int, *, body=WHITE_BODY, ink=BLACK_BODY, size=SIZE) -> Image.Image:
    image = Image.new("RGB", (size, size), (128, 128, 128))
    draw_piece(
        ImageDraw.Draw(image),
        class_id,
        origin=(0, 0),
        size=size,
        body=body,
        ink=ink,
    )
    return image


def ink_pixels(image: Image.Image) -> int:
    """How many pixels differ from the flat background."""
    background = Image.new("RGB", image.size, (128, 128, 128))
    difference = ImageChops.difference(image, background)
    # Via numpy rather than `getdata()`: Pillow types that as an ImagingCore,
    # which mypy does not consider iterable, and the per-pixel Python loop is
    # slower than the array reduction anyway.
    return int(np.asarray(difference).any(axis=2).sum())


WHITE_PIECES = [LETTER_TO_CLASS[letter] for letter in "PNBRQK"]
BLACK_PIECES = [LETTER_TO_CLASS[letter] for letter in "pnbrqk"]


class TestCoverage:
    @pytest.mark.parametrize("class_id", WHITE_PIECES + BLACK_PIECES)
    def test_every_piece_draws_something(self, class_id):
        assert ink_pixels(render(class_id)) > SIZE * SIZE * 0.05

    def test_all_six_kinds_have_a_shape(self):
        assert set(SHAPES) == set("pnbrqk")

    def test_an_empty_square_draws_nothing(self):
        # Class 0 has no letter; it must be a no-op rather than a stray mark.
        assert ink_pixels(render(0)) == 0


class TestDistinctness:
    @pytest.mark.parametrize(
        "first,second", list(itertools.combinations(WHITE_PIECES, 2))
    )
    def test_no_two_pieces_look_alike(self, first, second):
        # The one thing a diagram must never do is render a bishop as a pawn.
        difference = ImageChops.difference(render(first), render(second))
        changed = int(np.asarray(difference).any(axis=2).sum())
        assert changed > SIZE * SIZE * 0.02

    def test_the_two_colours_share_a_silhouette(self):
        # A white knight and a black knight are the same piece seen differently.
        # If the shapes diverged, the diagram would imply something untrue.
        white = render(LETTER_TO_CLASS["N"])
        black = render(LETTER_TO_CLASS["n"], body=BLACK_BODY, ink=WHITE_BODY)
        assert ink_pixels(white) == pytest.approx(ink_pixels(black), rel=0.15)

    def test_colour_actually_changes_the_drawing(self):
        white = render(LETTER_TO_CLASS["Q"])
        black = render(LETTER_TO_CLASS["q"], body=BLACK_BODY, ink=WHITE_BODY)
        assert ImageChops.difference(white, black).getbbox() is not None


class TestGeometry:
    @pytest.mark.parametrize("class_id", WHITE_PIECES)
    def test_a_piece_stays_inside_its_square(self, class_id):
        # Squares abut, so anything drawn outside the box lands on a neighbour.
        margin = 4
        image = Image.new(
            "RGB", (SIZE + 2 * margin, SIZE + 2 * margin), (128, 128, 128)
        )
        draw_piece(
            ImageDraw.Draw(image),
            class_id,
            origin=(margin, margin),
            size=SIZE,
            body=WHITE_BODY,
            ink=BLACK_BODY,
        )
        box = ImageChops.difference(
            image, Image.new("RGB", image.size, (128, 128, 128))
        ).getbbox()
        assert box is not None
        assert box[0] >= margin - 1 and box[1] >= margin - 1
        assert box[2] <= margin + SIZE + 1 and box[3] <= margin + SIZE + 1

    @pytest.mark.parametrize("size", [16, 24, 96])
    def test_the_shapes_survive_a_small_square(self, size):
        # A video overlay draws these at whatever the frame allows; a piece that
        # collapses to nothing at 16px would silently empty the diagram.
        for class_id in WHITE_PIECES:
            assert ink_pixels(render(class_id, size=size)) > size * size * 0.03

    def test_pieces_share_a_baseline(self):
        # They stand on one board. A piece floating above the others reads as a
        # rendering fault even when the position is right.
        bottoms = []
        for class_id in WHITE_PIECES:
            image = render(class_id)
            box = ImageChops.difference(
                image, Image.new("RGB", image.size, (128, 128, 128))
            ).getbbox()
            bottoms.append(box[3])
        assert max(bottoms) - min(bottoms) <= 2


def test_letters_and_classes_still_line_up():
    # draw_piece keys off CLASS_TO_LETTER; if that mapping moved, every piece
    # would draw as the wrong shape without anything failing.
    assert CLASS_TO_LETTER[LETTER_TO_CLASS["K"]] == "K"
    assert CLASS_TO_LETTER[LETTER_TO_CLASS["k"]] == "k"
