"""The corner heatmap model's targets, loss and decoding.

The model itself cannot be unit tested -- it has to be trained -- but everything
around it can, and that is where the errors that matter hide. A target rendered
half a cell off, or a decode that reads the peak's cell index instead of its
sub-cell position, produces a model that trains happily and localises badly, with
no failure anywhere to notice.
"""

from __future__ import annotations

import math

import pytest
import torch

from chesssight.train.heatmap import (
    CORNERS,
    decode,
    focal_loss,
    gaussian_peak,
    quad_from_logits,
    render_target,
    square_size,
)

SIZE = 256
STRIDE = 4
CELLS = SIZE // STRIDE


def logits_with_peaks(points, *, sigma: float = 2.0, strength: float = 6.0):
    """Logits whose sigmoid has Gaussian peaks at ``points`` (input pixels)."""
    target = render_target(points, SIZE, stride=STRIDE, sigma=sigma)
    # A target in [0, 1] read as a probability, inverted to logits and scaled so
    # the peak is confident. Background sits well below.
    return (target * 2 - 1) * strength


class TestTargets:
    def test_a_peak_lands_on_the_point(self):
        target = render_target([[100.0, 60.0]], SIZE, stride=STRIDE)
        index = int(torch.argmax(target))
        y, x = divmod(index, CELLS)
        assert (x, y) == (round(100 / STRIDE), round(60 / STRIDE))
        assert float(target.max()) == 1.0

    def test_points_outside_the_image_are_dropped_not_clamped(self):
        # A corner pushed out of frame by a crop is missing information. Clamping
        # it to the border would teach the model that a corner sits wherever the
        # image ends, which is the one thing it must never learn.
        target = render_target([[-30.0, 60.0], [SIZE + 5.0, 10.0]], SIZE, stride=STRIDE)
        assert float(target.max()) == 0.0

    def test_two_close_peaks_stay_two_peaks(self):
        # Combining by sum would give one brighter blob between them, which a peak
        # decoder reads as a single corner in the wrong place.
        points = [[100.0, 100.0], [120.0, 100.0]]
        target = render_target(points, SIZE, stride=STRIDE, sigma=1.0)
        assert float(target.max()) == 1.0
        found = decode(torch.log(target.clamp(1e-6) / (1 - target.clamp(max=1 - 1e-6))))
        best = sorted(found, key=lambda point: -point[2])[:2]
        xs = sorted(round(point[0]) for point in best)
        assert xs == pytest.approx([100, 120], abs=STRIDE)

    def test_the_gaussian_falls_off_with_distance(self):
        heatmap = torch.zeros((CELLS, CELLS))
        gaussian_peak(heatmap, 20.0, 20.0, sigma=2.0)
        assert heatmap[20, 20] == 1.0
        assert heatmap[20, 22] < heatmap[20, 21] < 1.0
        assert heatmap[20, 30] == 0.0  # beyond 3 sigma, untouched

    def test_a_peak_at_the_edge_does_not_wrap_or_raise(self):
        heatmap = torch.zeros((CELLS, CELLS))
        gaussian_peak(heatmap, 0.0, 0.0, sigma=2.0)
        assert heatmap[0, 0] == 1.0
        assert heatmap[-1, -1] == 0.0


class TestFocalLoss:
    def test_a_perfect_prediction_scores_near_zero(self):
        # "Perfect" is a sharp cell, not a reproduction of the Gaussian. The
        # target's shoulder is a *penalty discount* on nearby negatives, not
        # something to regress onto: a model that output the whole blob would be
        # claiming several corners where there is one.
        target = render_target([[100.0, 60.0]], SIZE, stride=STRIDE).unsqueeze(0)
        confident = torch.full_like(target, -12.0)
        confident[target == 1.0] = 12.0
        assert float(focal_loss(confident, target)) < 0.05

    def test_predicting_background_everywhere_is_punished(self):
        target = render_target([[100.0, 60.0]], SIZE, stride=STRIDE).unsqueeze(0)
        empty = torch.full_like(target, -12.0)
        assert float(focal_loss(empty, target)) > 1.0

    def test_a_near_miss_costs_less_than_a_wild_one(self):
        # The whole point of the penalty-reduced form: one cell off is nearly
        # right, and must not be punished like firing on a player's sleeve.
        target = render_target([[100.0, 100.0]], SIZE, stride=STRIDE).unsqueeze(0)
        near = logits_with_peaks([[104.0, 100.0]]).unsqueeze(0)
        far = logits_with_peaks([[200.0, 30.0]]).unsqueeze(0)
        assert float(focal_loss(near, target)) < float(focal_loss(far, target))

    def test_an_image_with_no_corners_in_frame_is_finite(self):
        # Dividing by a zero positive count would return NaN and poison the epoch.
        target = torch.zeros((1, 1, CELLS, CELLS))
        loss = focal_loss(torch.zeros_like(target), target)
        assert math.isfinite(float(loss))


class TestDecode:
    def test_peaks_come_back_where_they_were_put(self):
        points = [[40.0, 40.0], [200.0, 44.0], [210.0, 190.0], [30.0, 180.0]]
        found = decode(logits_with_peaks(points))
        assert len(found) == CORNERS
        for x, y, _ in found:
            assert (
                min(math.hypot(x - px, y - py) for px, py in points) <= STRIDE
            )  # within a cell

    def test_one_blob_gives_one_point_not_nine(self):
        # Without peak suppression a wide blob returns its own neighbourhood as
        # four separate "corners", and four points on one physical corner give a
        # degenerate homography rather than an obviously wrong one.
        found = decode(logits_with_peaks([[128.0, 128.0]], sigma=4.0), count=4)
        strong = [point for point in found if point[2] > 0.5]
        assert len(strong) == 1

    def test_the_answer_is_sub_cell(self):
        # A whole-cell answer quantises to 4 input pixels, which a 3072px
        # photograph then scales up by nearly seven.
        found = decode(logits_with_peaks([[102.0, 100.0]]))
        x, _, _ = max(found, key=lambda point: point[2])
        assert x != pytest.approx(round(x / STRIDE) * STRIDE, abs=1e-6)
        assert x == pytest.approx(102.0, abs=STRIDE)


class TestQuadFromLogits:
    def test_a_quad_is_scaled_back_to_the_original_image(self):
        points = [[40.0, 40.0], [200.0, 44.0], [210.0, 190.0], [30.0, 180.0]]
        logits = logits_with_peaks(points)
        quad = quad_from_logits(logits, size=(1024, 512), input_size=SIZE)
        assert quad is not None
        # The two axes scale independently: the model saw a square resize.
        for x, y in quad:
            assert 0 <= x <= 1024 and 0 <= y <= 512
        assert max(x for x, _ in quad) > 512  # x really was scaled by 4, not 2

    def test_too_few_peaks_give_no_quad_rather_than_a_guess(self):
        logits = logits_with_peaks([[40.0, 40.0], [200.0, 44.0]])
        quad = quad_from_logits(
            logits, size=(SIZE, SIZE), input_size=SIZE, min_score=0.5
        )
        assert quad is None

    def test_the_quad_is_ordered_not_raw(self):
        # Decode returns peaks by score; a homography needs them in order round
        # the board, or the board is read as a bow-tie.
        points = [[210.0, 190.0], [40.0, 40.0], [30.0, 180.0], [200.0, 44.0]]
        quad = quad_from_logits(
            logits_with_peaks(points), size=(SIZE, SIZE), input_size=SIZE
        )
        assert quad is not None
        from chesssight.data.geometry import polygon_signed_area

        # A correctly ordered quad has the area of the board it covers; a
        # bow-tie's self-intersection cancels most of it away.
        assert abs(polygon_signed_area(quad)) > 0.8 * (170 * 145)

    def test_without_a_floor_even_noise_yields_a_quad(self):
        # Not a bug, but the reason `found_rate` must never be quoted without the
        # peak confidences beside it: topk returns four cells from any input, so
        # at min_score=0 the model cannot report that it found no board.
        torch.manual_seed(0)
        noise = torch.randn(1, CELLS, CELLS) * 0.1 - 5.0
        assert quad_from_logits(noise, size=(SIZE, SIZE), input_size=SIZE) is not None
        assert (
            quad_from_logits(noise, size=(SIZE, SIZE), input_size=SIZE, min_score=0.1)
            is None
        )


class TestBackbones:
    """The head is backbone-agnostic; these check that claim rather than assume it."""

    @pytest.mark.parametrize(
        "backbone", ["resnet18", "convnext_tiny", "swin_tiny_patch4_window7_224"]
    )
    def test_a_pyramid_backbone_produces_a_stride_four_heatmap(self, backbone):
        from chesssight.train.heatmap import CornerHeatmapNet

        model = CornerHeatmapNet(backbone, pretrained=False, image_size=224)
        out = model(torch.zeros(1, 3, 224, 224))
        assert out.shape == (1, 1, 224 // STRIDE, 224 // STRIDE)

    def test_a_patch16_vit_is_refused_with_a_reason(self):
        # Not an arbitrary restriction: a ViT's finest map is stride 16, and
        # upsampling it 4x discards the localisation the heatmap exists for.
        from chesssight.train.heatmap import CornerHeatmapNet

        with pytest.raises(ValueError, match="stride"):
            CornerHeatmapNet("vit_base_patch16_224", pretrained=False, image_size=224)

    def test_channels_last_features_are_transposed_not_misread(self):
        # Swin emits NHWC from features_only and ignores output_fmt. Reading it
        # as NCHW would treat the height axis as channels -- which happens to be
        # a legal shape when H equals C, so this cannot be left to chance.
        from chesssight.train.heatmap import CornerHeatmapNet

        nhwc = torch.zeros(1, 32, 32, 96)
        out = CornerHeatmapNet._as_nchw(nhwc, 96)
        assert out.shape == (1, 96, 32, 32)

    def test_a_map_already_in_nchw_is_left_alone(self):
        from chesssight.train.heatmap import CornerHeatmapNet

        nchw = torch.zeros(1, 96, 32, 32)
        assert CornerHeatmapNet._as_nchw(nchw, 96) is nchw

    def test_a_map_with_the_channels_nowhere_is_an_error(self):
        from chesssight.train.heatmap import CornerHeatmapNet

        with pytest.raises(ValueError, match="channels on no axis"):
            CornerHeatmapNet._as_nchw(torch.zeros(1, 8, 32, 32), 96)


def test_square_size_is_an_eighth_of_a_side():
    corners = [[0.0, 0.0], [800.0, 0.0], [800.0, 800.0], [0.0, 800.0]]
    assert square_size(corners) == pytest.approx(100.0)
