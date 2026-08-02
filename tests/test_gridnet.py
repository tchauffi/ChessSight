"""Reading 64 squares off a rectified board.

The alignment between output cell (rank, file) and board square (rank, file) is
the whole design, and it is exactly the thing that fails silently: a transposed
or half-cell-shifted readout trains to a plausible accuracy and returns a
position that is wrong in a consistent, hard-to-spot way. So these tests check
the correspondence directly, by feeding the model's own geometry a synthetic
board and asking which cell lights up.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chesssight.data.fen import BOARD_SIZE, NUM_CLASSES
from chesssight.train.gridnet import GridNet
from chesssight.train.rectify import square_centre_px

SIZE = 224


def model() -> GridNet:
    return GridNet("resnet18", pretrained=False, image_size=SIZE, channels=32).eval()


class TestShape:
    def test_the_output_is_one_class_vector_per_square(self):
        out = model()(torch.zeros(2, 3, SIZE, SIZE))
        assert out.shape == (2, NUM_CLASSES, BOARD_SIZE, BOARD_SIZE)

    def test_it_taps_the_requested_stride(self):
        # Resolution is the knob that governs piece-type confusion: at stride 8 a
        # square had five feature positions to describe a whole silhouette, and
        # 12.65% of occupied squares came back the right colour and the wrong
        # piece. The default is now the finer map.
        assert model().stride == 4
        coarse = GridNet(
            "resnet18",
            pretrained=False,
            image_size=SIZE,
            channels=32,
            feature_stride=8,
        )
        assert coarse.stride == 8

    def test_more_than_one_cell_is_sampled_per_square(self):
        # The first version pooled each square to a single cell, discarding
        # within-square detail before anything could use it, and plateaued at
        # 90% per-square.
        net = model()
        assert net.cells_per_square >= 2
        cells = net.pool_cells(torch.zeros(1, 3, SIZE, SIZE))
        assert cells.shape[-1] == BOARD_SIZE * net.cells_per_square

    def test_a_square_can_see_its_neighbours(self):
        # The head must mix across squares: a piece's identifying detail smears
        # onto neighbouring cells, so a 1x1 classifier is reading mostly board.
        # Perturbing one square has to change its neighbour's logits.
        net = model()
        base = torch.zeros(1, 3, SIZE, SIZE)
        bumped = base.clone()
        cx, cy = square_centre_px(4, 4, SIZE)
        bumped[:, :, int(cy) - 5 : int(cy) + 5, int(cx) - 5 : int(cx) + 5] = 3.0
        with torch.no_grad():
            before = net(base)[0, :, 4, 5]
            after = net(bumped)[0, :, 4, 5]
        assert not torch.allclose(before, after, atol=1e-5)


class TestAlignment:
    @pytest.mark.parametrize(
        "rank,file", [(0, 0), (0, 7), (7, 0), (7, 7), (3, 4), (5, 2)]
    )
    def test_a_bright_square_lights_its_own_cell(self, rank, file):
        # A white patch at one square's centre must produce the largest response
        # at that cell. This catches a transpose, a flip, and an off-by-one in
        # the playing-area rectangle -- none of which any shape check would.
        net = model()
        image = torch.zeros(1, 3, SIZE, SIZE)
        cx, cy = square_centre_px(rank, file, SIZE)
        half = 5
        image[
            :, :, int(cy) - half : int(cy) + half, int(cx) - half : int(cx) + half
        ] = 3.0

        # Read the pooled cells rather than the classifier's logits: an untrained
        # classifier has no reason to prefer any class, but the features still
        # carry where the energy was. Via the model's own method, so the test
        # cannot drift from the path that actually runs.
        with torch.no_grad():
            cells = net.pool_cells(image)
        # Several cells per square, so fold them back down before locating the
        # square: the grid is the board's, the resolution is the model's.
        per = net.cells_per_square
        energy = cells[0].abs().mean(dim=0)
        squares = energy.reshape(BOARD_SIZE, per, BOARD_SIZE, per).mean(dim=(1, 3))
        hottest = np.unravel_index(int(squares.argmax()), squares.shape)
        assert hottest == (rank, file)

    def test_the_grid_is_not_symmetric_under_transpose(self):
        # A guard on the guard: if the readout were transposed, the test above
        # would still pass for every symmetric (r, r) square, so confirm an
        # asymmetric square actually distinguishes the two.
        assert square_centre_px(1, 6, SIZE) != square_centre_px(6, 1, SIZE)


class TestBackboneChoice:
    def test_a_swin_backbone_works_here_too(self):
        net = GridNet(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            image_size=SIZE,
            channels=32,
        ).eval()
        out = net(torch.zeros(1, 3, SIZE, SIZE))
        assert out.shape == (1, NUM_CLASSES, BOARD_SIZE, BOARD_SIZE)

    def test_channels_last_features_are_transposed(self):
        net = model()
        net.width = 96
        assert net._as_nchw(torch.zeros(1, 7, 7, 96)).shape == (1, 96, 7, 7)

    def test_a_map_with_the_channels_nowhere_is_an_error(self):
        with pytest.raises(ValueError, match="channels nowhere"):
            model()._as_nchw(torch.zeros(1, 5, 7, 9))
