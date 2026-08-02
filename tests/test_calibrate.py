"""The calibration maths, tested without torch.

The point of Platt scaling here is that it is monotone -- it may not reorder
detections, or it would silently change mAP. That property, and the matching
discipline the fit is trained against, are what these tests pin down.
"""

from __future__ import annotations

import numpy as np
import pytest

from chesssight.train.calibrate import (
    Calibration,
    choose_threshold,
    fit_platt,
    iou_matrix,
    match_detections,
)


def make_calibration(scale: float = 1.0, bias: float = 0.0) -> Calibration:
    return Calibration(
        scale=scale,
        bias=bias,
        threshold=0.5,
        fit_split="val",
        precision=0.0,
        recall=0.0,
        f1=0.0,
        detections_used=0,
    )


class TestIoU:
    def test_identical_boxes_have_iou_one(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        assert iou_matrix(boxes, boxes)[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[20.0, 20.0, 30.0, 30.0]])
        assert iou_matrix(a, b)[0, 0] == 0.0

    def test_half_overlap(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[5.0, 0.0, 15.0, 10.0]])
        # Intersection 50, union 150.
        assert iou_matrix(a, b)[0, 0] == pytest.approx(1 / 3)

    def test_empty_inputs_give_empty_matrix(self):
        assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


class TestMatching:
    def test_correct_detection_matches(self):
        matched = match_detections(
            det_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
            det_labels=np.array([2]),
            det_scores=np.array([0.9]),
            truth_boxes=np.array([[1.0, 1.0, 10.0, 10.0]]),
            truth_labels=np.array([2]),
        )
        assert matched.tolist() == [True]

    def test_wrong_class_does_not_match(self):
        matched = match_detections(
            det_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
            det_labels=np.array([2]),
            det_scores=np.array([0.9]),
            truth_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
            truth_labels=np.array([3]),
        )
        assert matched.tolist() == [False]

    def test_duplicate_detection_is_a_false_positive(self):
        # Two boxes on one object: the higher-scoring one claims it, the other
        # must count as unmatched -- that is what teaches the calibrator to push
        # duplicates down.
        matched = match_detections(
            det_boxes=np.array([[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.0, 10.0]]),
            det_labels=np.array([1, 1]),
            det_scores=np.array([0.4, 0.9]),
            truth_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
            truth_labels=np.array([1]),
        )
        # Index 1 has the higher score and claims the object.
        assert matched.tolist() == [False, True]

    def test_low_iou_does_not_match(self):
        matched = match_detections(
            det_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
            det_labels=np.array([1]),
            det_scores=np.array([0.9]),
            truth_boxes=np.array([[8.0, 8.0, 18.0, 18.0]]),
            truth_labels=np.array([1]),
        )
        assert matched.tolist() == [False]


class TestPlattFit:
    def test_recovers_a_known_squashing(self):
        # Simulate the observed pathology: true probabilities squashed into
        # [0, 0.1]. The fit must learn to expand them back.
        rng = np.random.default_rng(0)
        true_probability = rng.uniform(0.01, 0.99, size=4000)
        matched = rng.uniform(size=4000) < true_probability
        squashed = true_probability * 0.1  # scores pinned under 0.1

        scale, bias = fit_platt(squashed, matched)
        calibration = make_calibration(scale, bias)

        recovered = calibration.apply(squashed)
        # Calibrated scores should track the true probabilities closely.
        assert np.mean(np.abs(recovered - true_probability)) < 0.08

    def test_is_monotone(self):
        rng = np.random.default_rng(1)
        scores = rng.uniform(0.001, 0.1, size=2000)
        matched = rng.uniform(size=2000) < scores * 8
        scale, bias = fit_platt(scores, matched)
        assert scale > 0

        calibration = make_calibration(scale, bias)
        ordered = np.sort(scores)
        calibrated = calibration.apply(ordered)
        assert np.all(np.diff(calibrated) >= 0), "calibration reordered detections"

    def test_refuses_single_class_input(self):
        with pytest.raises(ValueError, match="both matched and unmatched"):
            fit_platt(np.array([0.1, 0.2]), np.array([True, True]))

    def test_refuses_anti_correlated_scores(self):
        # If low scores are matches and high scores are junk, a monotone map
        # cannot help and pretending otherwise would corrupt mAP.
        scores = np.concatenate([np.full(500, 0.01), np.full(500, 0.9)])
        matched = np.concatenate([np.ones(500, bool), np.zeros(500, bool)])
        with pytest.raises(ValueError, match="non-positive scale"):
            fit_platt(scores, matched)


class TestThreshold:
    def test_perfect_separation_yields_perfect_f1(self):
        calibrated = np.array([0.9, 0.8, 0.2, 0.1])
        matched = np.array([True, True, False, False])
        threshold, precision, recall, f1 = choose_threshold(calibrated, matched, 2)
        assert 0.2 < threshold < 0.8
        assert precision == 1.0 and recall == 1.0 and f1 == pytest.approx(1.0)

    def test_undetected_truth_counts_against_recall(self):
        calibrated = np.array([0.9])
        matched = np.array([True])
        # Two truth objects, one never detected at any score.
        _, precision, recall, _ = choose_threshold(calibrated, matched, 2)
        assert precision == 1.0
        assert recall == pytest.approx(0.5)


class TestRoundTrip:
    def test_save_and_load(self, tmp_path):
        calibration = Calibration(
            scale=2.5,
            bias=1.1,
            threshold=0.42,
            fit_split="val",
            precision=0.9,
            recall=0.8,
            f1=0.85,
            detections_used=1234,
        )
        calibration.save(tmp_path)
        loaded = Calibration.load(tmp_path)
        assert loaded == calibration

    def test_missing_file_loads_as_none(self, tmp_path):
        assert Calibration.load(tmp_path) is None

    def test_apply_maps_low_scores_up(self):
        # A fit against squashed scores has scale > 1; spot-check the direction.
        calibration = make_calibration(scale=3.0, bias=10.0)
        assert calibration.apply_one(0.05) > 0.5
        assert calibration.apply_one(0.001) < calibration.apply_one(0.05)
