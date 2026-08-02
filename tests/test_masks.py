from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from chesssight.data import masks
from chesssight.data.schema import BoundingBox


def make_id_image(height: int = 20, width: int = 30) -> np.ndarray:
    """An id image with two pieces, a board region and background."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Board fills the lower half.
    image[10:, :, 1] = masks.ROLE_CODES["board"]
    # Piece 1: a 4x5 block.
    image[2:6, 3:8, 0] = 1
    image[2:6, 3:8, 1] = masks.ROLE_CODES["piece"]
    # Piece 7: a 3x3 block.
    image[12:15, 20:23, 0] = 7
    image[12:15, 20:23, 1] = masks.ROLE_CODES["piece"]
    return image


class TestInstanceStats:
    def test_finds_present_instances(self):
        image = make_id_image()
        stats = masks.instance_stats(image, 1)
        assert stats.present
        assert stats.area == 4 * 5
        assert stats.bbox == BoundingBox(x=3.0, y=2.0, width=5.0, height=4.0)

    def test_second_instance(self):
        stats = masks.instance_stats(make_id_image(), 7)
        assert stats.area == 9
        assert stats.bbox == BoundingBox(x=20.0, y=12.0, width=3.0, height=3.0)

    def test_absent_instance(self):
        stats = masks.instance_stats(make_id_image(), 5)
        assert not stats.present
        assert stats.area == 0
        assert stats.bbox is None
        assert stats.visible_pixels == 0

    def test_board_pixels_are_not_mistaken_for_a_piece(self):
        # The board also carries instance id 0; the role channel is what separates
        # it from the background, and neither may leak into a piece mask.
        image = make_id_image()
        assert masks.instance_stats(image, 0).area == 0

    def test_present_instance_ids(self):
        assert masks.present_instance_ids(make_id_image()) == {1, 7}


class TestRLE:
    @pytest.mark.parametrize("seed", range(8))
    def test_round_trip_on_random_masks(self, seed: int):
        rng = np.random.default_rng(seed)
        mask = rng.random((14, 9)) > 0.6
        assert np.array_equal(masks.rle_decode(masks.rle_encode(mask)), mask)

    def test_round_trip_all_false(self):
        mask = np.zeros((5, 4), dtype=bool)
        rle = masks.rle_encode(mask)
        assert rle.counts == [20]
        assert np.array_equal(masks.rle_decode(rle), mask)

    def test_round_trip_all_true(self):
        mask = np.ones((5, 4), dtype=bool)
        rle = masks.rle_encode(mask)
        # Must start with a zero-length run of zeros to keep the parity right.
        assert rle.counts[0] == 0
        assert np.array_equal(masks.rle_decode(rle), mask)

    def test_mask_starting_with_one_keeps_parity(self):
        mask = np.array([[True, False], [False, True]])
        assert np.array_equal(masks.rle_decode(masks.rle_encode(mask)), mask)

    def test_counts_cover_the_image(self):
        mask = np.zeros((7, 3), dtype=bool)
        mask[2, 1] = True
        assert sum(masks.rle_encode(mask).counts) == 21

    def test_round_trip_through_real_id_image(self):
        image = make_id_image()
        mask = masks.piece_mask(image, 1)
        assert np.array_equal(masks.rle_decode(masks.rle_encode(mask)), mask)


class TestVisibility:
    def test_ratio_is_one_when_fully_visible(self):
        amodal = BoundingBox(x=0, y=0, width=10, height=10)
        assert masks.visibility_ratio(amodal, amodal, 100) == pytest.approx(1.0)

    def test_ratio_drops_when_occluded(self):
        amodal = BoundingBox(x=0, y=0, width=10, height=10)
        modal = BoundingBox(x=0, y=0, width=5, height=5)
        assert masks.visibility_ratio(modal, amodal, 25) == pytest.approx(0.25)

    def test_ratio_is_clamped_to_one(self):
        amodal = BoundingBox(x=0, y=0, width=2, height=2)
        assert masks.visibility_ratio(amodal, amodal, 999) == 1.0

    def test_no_amodal_box_gives_none(self):
        assert masks.visibility_ratio(None, None, 10) is None


def test_load_id_image_is_top_down(tmp_path):
    # Pillow reads rows top-down. Blender's own image.pixels is bottom-up, so this
    # asserts we are on the side of the boundary that needs no flip.
    array = np.zeros((4, 4, 3), dtype=np.uint8)
    array[0, :, 0] = 200  # top row
    path = tmp_path / "ids.png"
    Image.fromarray(array).save(path)

    loaded = masks.load_id_image(path)
    assert loaded[0, 0, 0] == 200
    assert loaded[3, 0, 0] == 0


def test_bbox_from_empty_mask_is_none():
    assert masks.bbox_from_mask(np.zeros((4, 4), dtype=bool)) is None
