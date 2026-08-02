"""Corners expressed relative to the board box.

The whole point of this representation is that a corner outside the picture is a
perfectly ordinary number, so most of these tests are about *not* clamping,
clipping or otherwise quietly moving a point back inside -- each of which would
look like a working model and silently discard the case the model exists for.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chesssight.train.boxcorners import (
    BoxCornerNet,
    crop_box,
    jitter,
    to_box_space,
    to_image_space,
)

BOX = (100.0, 200.0, 300.0, 400.0)  # 200 x 200


class TestCropBox:
    def test_the_margin_grows_every_side(self):
        x0, y0, x1, y1 = crop_box(BOX, 0.25)
        assert (x0, y0, x1, y1) == (50.0, 150.0, 350.0, 450.0)

    def test_it_is_not_clipped_to_the_image(self):
        # Clipping would shift the crop back inside the frame and move every
        # target with it, which is precisely the bug this representation avoids.
        assert crop_box((0.0, 0.0, 10.0, 10.0), 0.5)[0] == -5.0

    def test_zero_margin_is_the_box(self):
        assert crop_box(BOX, 0.0) == BOX


class TestBoxSpace:
    def test_the_box_corners_map_to_the_unit_square(self):
        mapped = to_box_space(
            [[100.0, 200.0], [300.0, 200.0], [300.0, 400.0], [100.0, 400.0]], BOX
        )
        assert np.allclose(mapped, [[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_a_point_outside_the_box_keeps_going(self):
        # The representation's reason to exist: a corner below the frame is
        # y > 1, not a clamped 1.0.
        mapped = to_box_space([[100.0, 500.0]], BOX)
        assert mapped[0][1] == pytest.approx(1.5)

    def test_a_point_before_the_box_is_negative(self):
        mapped = to_box_space([[0.0, 200.0]], BOX)
        assert mapped[0][0] == pytest.approx(-0.5)

    def test_it_round_trips(self):
        points = [[120.0, 640.0], [-40.0, 210.0], [305.0, 399.0], [99.0, 201.0]]
        assert np.allclose(to_image_space(to_box_space(points, BOX), BOX), points)

    def test_a_degenerate_box_is_an_error(self):
        with pytest.raises(ValueError, match="degenerate"):
            to_box_space([[1.0, 1.0]], (5.0, 5.0, 5.0, 9.0))


class TestJitter:
    def test_it_stays_near_the_original(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            x0, y0, x1, y1 = jitter(BOX, rng, scale=0.12, shift=0.08)
            assert 140 < (x1 - x0) < 260  # within the scale range, generously
            assert abs((x0 + x1) / 2 - 200.0) < 200 * 0.2

    def test_zero_jitter_is_the_identity(self):
        rng = np.random.default_rng(0)
        assert jitter(BOX, rng, scale=0.0, shift=0.0) == pytest.approx(BOX)

    def test_it_is_reproducible_from_a_seed(self):
        a = jitter(BOX, np.random.default_rng(7), scale=0.1, shift=0.1)
        b = jitter(BOX, np.random.default_rng(7), scale=0.1, shift=0.1)
        assert a == b


class TestModel:
    def test_it_predicts_four_points(self):
        model = BoxCornerNet("resnet18", pretrained=False, image_size=224)
        out = model(torch.zeros(2, 3, 224, 224))
        assert out.shape == (2, 4, 2)

    def test_it_starts_near_the_box_corners(self):
        # A square-on board's corners *are* the box's corners, so starting there
        # means the model begins at a plausible geometry rather than at the
        # crop's origin, which is one corner and therefore a degenerate quad.
        model = BoxCornerNet("resnet18", pretrained=False, image_size=224).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 224, 224))[0]
        assert out[:, 0].min() < 0.45 and out[:, 0].max() > 0.55
        assert out[:, 1].min() < 0.45 and out[:, 1].max() > 0.55

    def test_a_plain_vit_is_accepted_here(self):
        # The opposite of the heatmap's constraint: this head pools to a vector,
        # so a patch-16 ViT's lack of stride-4 features does not matter.
        model = BoxCornerNet("vit_small_patch16_224", pretrained=False, image_size=224)
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 4, 2)
