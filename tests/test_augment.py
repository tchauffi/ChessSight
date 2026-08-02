"""Train-time augmentation.

The geometric transforms are the risky part: if boxes do not follow the image,
training silently learns from wrong targets and nothing crashes. So the tests here
put a marker at a known position and check the box still bounds it afterwards,
rather than checking that the transform ran.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from chesssight.train.augment import (
    MIN_BOX_SIZE_PX,
    AugmentConfig,
    apply,
    build_transform,
)

MARKER_BOX = [80.0, 100.0, 60.0, 40.0]


def marked_image(size: int = 256) -> Image.Image:
    """White canvas with a red rectangle at ``MARKER_BOX``.

    Red rather than black: rotation fills exposed corners, and a black marker
    would be indistinguishable from that fill.
    """
    image = Image.new("RGB", (size, size), (255, 255, 255))
    x, y, w, h = (int(v) for v in MARKER_BOX)
    image.paste(Image.new("RGB", (w, h), (220, 0, 0)), (x, y))
    return image


def marker_bounds(image: Image.Image) -> tuple[float, float, float, float] | None:
    array = np.asarray(image.convert("RGB")).astype(int)
    red = (array[:, :, 0] > 140) & (array[:, :, 1] < 110) & (array[:, :, 2] < 110)
    if red.sum() < 20:
        return None
    ys, xs = np.where(red)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def iou(box_xywh: list[float], bounds: tuple[float, float, float, float]) -> float:
    x, y, w, h = box_xywh
    px0, py0, px1, py1 = x, y, x + w, y + h
    ax0, ay0, ax1, ay1 = bounds
    ix0, iy0 = max(ax0, px0), max(ay0, py0)
    ix1, iy1 = min(ax1, px1), min(ay1, py1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + w * h - inter
    return inter / union if union > 0 else 0.0


def geometry_only(*, rotation_probability: float = 0.0) -> AugmentConfig:
    """Only the geometric families, so a box test measures geometry alone.

    The crop is not optional -- it is what brings an image to working size -- so
    only rotation is switched here.
    """
    return AugmentConfig(
        image_size=256,
        photometric_probability=0.0,
        noise_probability=0.0,
        blur_probability=0.0,
        jpeg_probability=0.0,
        rotation_probability=rotation_probability,
    )


def run(config: AugmentConfig, draws: int = 30, seed: int = 0) -> list[float]:
    transform = build_transform(config)
    scores = []
    for offset in range(draws):
        torch.manual_seed(seed + offset)
        image, boxes, _ = apply(transform, marked_image(), [list(MARKER_BOX)], [0])
        if not boxes:
            continue
        bounds = marker_bounds(image)
        if bounds is None:
            continue
        scores.append(iou(boxes[0], bounds))
    return scores


class TestGeometryTracksBoxes:
    def test_crop_keeps_the_box_on_the_object(self):
        scores = run(geometry_only())
        assert len(scores) >= 20
        assert min(scores) > 0.85

    def test_rotation_keeps_the_box_on_the_object(self):
        scores = run(geometry_only(rotation_probability=1.0))
        assert len(scores) >= 20
        assert min(scores) > 0.85

    def test_crop_and_rotation_together(self):
        scores = run(geometry_only(rotation_probability=1.0))
        assert len(scores) >= 20
        assert float(np.median(scores)) > 0.85


class TestWorkingResolution:
    """Augmenting at native size is correct but 20x slower, and every one of
    those pixels is discarded by the resize a moment later."""

    def test_output_is_always_the_working_size(self):
        transform = build_transform(AugmentConfig(image_size=256))
        torch.manual_seed(0)
        big = marked_image(1024)
        image, _, _ = apply(transform, big, [[320.0, 400.0, 240.0, 160.0]], [0])
        assert image.size == (256, 256)

    def test_the_crop_cannot_reach_the_rotation_fill(self):
        # The inscribed square of an image rotated by d degrees is
        # 1 / (cos d + sin d) of its width; the crop's upper scale bound must
        # stay under that in area, or fill wedges survive into the output.
        import math

        config = AugmentConfig()
        degrees = math.radians(config.rotation_degrees)
        inscribed_linear = 1.0 / (math.cos(degrees) + math.sin(degrees))
        assert config.crop_scale[1] ** 0.5 < inscribed_linear


class TestNoFlips:
    """Flips are the first thing every detection recipe reaches for, and all of
    them are wrong here: a mirrored board puts the light square on the player's
    left, which is the defect the generator guarantees against."""

    def test_the_pipeline_contains_no_flip_or_quarter_turn(self):
        transform = build_transform(AugmentConfig())
        names = {type(module).__name__ for module in transform.modules()}
        assert "RandomHorizontalFlip" not in names
        assert "RandomVerticalFlip" not in names
        assert "RandomRotation90" not in names

    def test_rotation_stays_within_camera_roll(self):
        # Anything approaching a quarter turn would remap squares.
        assert AugmentConfig().rotation_degrees <= 15


class TestPhotometricLeavesGeometryAlone:
    def test_colour_and_sensor_effects_do_not_move_the_box(self):
        config = AugmentConfig(
            image_size=256,
            rotation_probability=0.0,
            crop_scale=(1.0, 1.0),
            crop_ratio=(1.0, 1.0),
            work_scale=1.0,
            photometric_probability=1.0,
            noise_probability=1.0,
            blur_probability=1.0,
            jpeg_probability=0.0,  # would corrupt the colour test, not the box
        )
        transform = build_transform(config)
        torch.manual_seed(0)
        _, boxes, _ = apply(transform, marked_image(), [list(MARKER_BOX)], [0])
        assert boxes
        # A full-frame crop at working scale 1.0 is the identity, so colour and
        # sensor effects must leave the coordinates untouched.
        assert boxes[0] == pytest.approx(MARKER_BOX, abs=0.5)


class TestSanitisation:
    def test_degenerate_boxes_are_dropped(self):
        transform = build_transform(geometry_only())
        tiny = [10.0, 10.0, 1.0, 1.0]
        _, boxes, labels = apply(transform, marked_image(), [tiny], [0])
        assert boxes == []
        assert labels == []

    def test_labels_stay_aligned_with_surviving_boxes(self):
        transform = build_transform(geometry_only())
        _, boxes, labels = apply(
            transform,
            marked_image(),
            [[10.0, 10.0, 1.0, 1.0], list(MARKER_BOX)],
            [3, 7],
        )
        # The sliver goes, and the label that goes with it must go too.
        assert len(boxes) == len(labels) == 1
        assert labels[0] == 7

    def test_minimum_box_size_is_enforced(self):
        assert MIN_BOX_SIZE_PX >= 2


def test_an_empty_annotation_list_is_handled():
    transform = build_transform(geometry_only())
    image, boxes, labels = apply(transform, marked_image(), [], [])
    assert boxes == []
    assert labels == []
    assert image.size == (256, 256)
