"""The numeric plumbing the ONNX backend had to reimplement.

These need neither onnxruntime nor torch: they pin the pre- and
post-processing against hand-computed expectations, so a drift shows up in CI
rather than only in `chesssight onnx parity`, which needs checkpoints.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from chesssight.inference.onnx import (
    CORNER_MEAN,
    CORNER_STD,
    corner_input,
    decode_detections,
    decode_peaks,
    detector_input,
)
from chesssight.train.labels import BOARD_INDEX, DETECTION_LABELS


def test_detector_input_rescales_but_does_not_normalise() -> None:
    # RT-DETR's saved config has do_normalize false. Normalising here would be
    # the natural mistake and would move every pixel by about two units.
    image = Image.new("RGB", (100, 50), (255, 128, 0))
    array = detector_input(image, 640, 1 / 255)

    assert array.shape == (1, 3, 640, 640)
    assert array.dtype == np.float32
    assert array[0, 0].max() == pytest.approx(1.0, abs=1e-3)
    assert array[0, 1].max() == pytest.approx(128 / 255, abs=1e-2)
    assert array[0, 2].max() == pytest.approx(0.0, abs=1e-3)


def test_corner_input_applies_imagenet_statistics() -> None:
    image = Image.new("RGB", (64, 64), (255, 255, 255))
    array = corner_input(image, 448)

    assert array.shape == (1, 3, 448, 448)
    expected = (1.0 - CORNER_MEAN) / CORNER_STD
    for channel in range(3):
        assert array[0, channel].mean() == pytest.approx(expected[channel], abs=1e-4)


def test_decode_detections_converts_boxes_and_scales_to_the_image() -> None:
    # One query, one confident class: a centred box half the image wide.
    logits = np.full((1, 1, len(DETECTION_LABELS)), -20.0, dtype=np.float32)
    logits[0, 0, BOARD_INDEX] = 20.0
    boxes = np.array([[[0.5, 0.5, 0.5, 0.25]]], dtype=np.float32)

    found = decode_detections(logits, boxes, (400, 200), threshold=0.5)

    assert len(found) == 1
    assert found[0]["name"] == "board"
    assert found[0]["label"] == BOARD_INDEX
    assert found[0]["score"] == pytest.approx(1.0, abs=1e-6)
    # cxcywh relative -> xyxy absolute, each axis scaled by its own dimension.
    assert found[0]["box"] == pytest.approx([100.0, 75.0, 300.0, 125.0])


def test_decode_detections_drops_everything_under_the_threshold() -> None:
    logits = np.full((1, 4, len(DETECTION_LABELS)), -20.0, dtype=np.float32)
    boxes = np.tile(np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32), (1, 4, 1))
    assert decode_detections(logits, boxes, (100, 100), threshold=0.5) == []


def test_decode_detections_lets_one_query_carry_two_classes() -> None:
    # The top-k runs over the flattened score matrix rather than per query, so
    # one query can appear twice under different labels while another
    # contributes nothing. How many are kept is the query count, exactly as
    # transformers does it -- hence two queries here, not one.
    logits = np.full((1, 2, len(DETECTION_LABELS)), -20.0, dtype=np.float32)
    logits[0, 0, 0] = 5.0
    logits[0, 0, 1] = 4.0
    boxes = np.tile(np.array([0.5, 0.5, 0.2, 0.2], dtype=np.float32), (1, 2, 1))

    found = decode_detections(logits, boxes, (100, 100), threshold=0.5)
    assert [d["label"] for d in found] == [0, 1]
    assert found[0]["score"] > found[1]["score"]


def test_decode_peaks_finds_four_separated_maxima() -> None:
    heat = np.full((1, 1, 40, 40), -10.0, dtype=np.float32)
    corners = [(5, 5), (5, 34), (34, 34), (34, 5)]
    for y, x in corners:
        heat[0, 0, y, x] = 10.0

    peaks = decode_peaks(heat, stride=4)

    assert len(peaks) == 4
    found = {(round(y / 4 - 0.5), round(x / 4 - 0.5)) for x, y, _ in peaks}
    assert found == set(corners)
    assert all(score > 0.99 for _, _, score in peaks)


def test_decode_peaks_suppresses_a_single_broad_blob() -> None:
    # Nine adjacent lit cells are one corner, not nine. Without the maximum
    # filter the top-4 would all land inside this blob.
    heat = np.full((1, 1, 40, 40), -10.0, dtype=np.float32)
    heat[0, 0, 9:12, 9:12] = 8.0
    heat[0, 0, 10, 10] = 10.0
    heat[0, 0, 30, 30] = 9.0

    peaks = decode_peaks(heat, stride=4)
    strong = [(x, y) for x, y, score in peaks if score > 0.5]
    assert len(strong) == 2


def test_decode_peaks_places_a_point_between_cells() -> None:
    # Two equally lit neighbours put the refined peak on the boundary rather
    # than snapping to a cell centre -- the whole reason for the soft-argmax.
    heat = np.full((1, 1, 20, 20), -10.0, dtype=np.float32)
    heat[0, 0, 10, 10] = 6.0
    heat[0, 0, 10, 11] = 6.0

    (x, _, _) = decode_peaks(heat, stride=4, count=1)[0]
    assert x == pytest.approx((10.5 + 0.5) * 4, abs=0.3)
