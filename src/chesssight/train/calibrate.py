"""Post-hoc confidence calibration for a trained detector.

Why this exists: a short RT-DETR fine-tune produces scores that *rank* well but sit
absurdly low -- this project's first checkpoint reached mAP 0.85 on real photographs
with no detection scoring above 0.07. That is a property of the training recipe, not
of the model's knowledge: varifocal loss over 300 mostly-background queries grows
positive logits slowly (the reference model trains 72 epochs; the reinitialised head
here got a handful), and the classification term's weight of 1.0 is dwarfed by the
5.0 + 2.0 box terms. Ranking converges long before magnitude does.

The fix here is Platt scaling: fit ``sigmoid(a * logit(s) + b)`` on a held-out
split, where each detection's label is whether it actually matches a ground-truth
object. Two properties make it the right tool:

* It is **monotone** (for ``a > 0``), so it cannot change the ranking -- mAP before
  and after is identical by construction. It only makes the numbers mean something.
* It is fit on ~10k detections with two parameters, so it cannot overfit the split
  it is fit on in any way that matters.

An operating threshold is chosen on the same split by maximising F1, because a
calibrated score is only useful once somebody has to pick a cutoff.

The fitted parameters are saved as ``calibration.json`` inside the checkpoint
directory, and applied automatically by prediction when present.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CALIBRATION_FILENAME = "calibration.json"

#: Detections kept per image when collecting calibration data. Generous on purpose:
#: the fit needs to see the low-score junk, not just the winners.
TOP_K = 100

#: IoU above which a detection counts as having found a ground-truth object.
MATCH_IOU = 0.5

_EPS = 1e-7


@dataclass(frozen=True)
class Calibration:
    """Fitted Platt parameters plus the chosen operating point."""

    scale: float
    bias: float
    threshold: float
    #: Metrics at the threshold, on the split the fit used. Recorded so a
    #: checkpoint states what its numbers were measured on.
    fit_split: str
    precision: float
    recall: float
    f1: float
    detections_used: int

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities. Monotone for scale > 0."""
        clipped = np.clip(np.asarray(scores, dtype=np.float64), _EPS, 1.0 - _EPS)
        logits = np.log(clipped / (1.0 - clipped))
        z = np.clip(self.scale * logits + self.bias, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def apply_one(self, score: float) -> float:
        return float(self.apply(np.asarray([score]))[0])

    def save(self, checkpoint_dir: Path) -> Path:
        path = Path(checkpoint_dir) / CALIBRATION_FILENAME
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, checkpoint_dir: Path) -> Calibration | None:
        path = Path(checkpoint_dir) / CALIBRATION_FILENAME
        if not path.is_file():
            return None
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``(N, 4)`` xyxy arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    left = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    top = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    right = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    bottom = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.clip(union, _EPS, None)


def match_detections(
    det_boxes: np.ndarray,
    det_labels: np.ndarray,
    det_scores: np.ndarray,
    truth_boxes: np.ndarray,
    truth_labels: np.ndarray,
    *,
    iou_threshold: float = MATCH_IOU,
) -> np.ndarray:
    """Return a boolean per detection: does it match an unclaimed truth object?

    Greedy by descending score with one-to-one claiming, the same discipline the
    mAP metric uses -- a duplicate box on an already-claimed object is a false
    positive, and the calibrator must learn to push duplicates down, not up.
    """
    order = np.argsort(-det_scores)
    matched = np.zeros(len(det_scores), dtype=bool)
    claimed = np.zeros(len(truth_boxes), dtype=bool)
    ious = iou_matrix(det_boxes, truth_boxes)

    for index in order:
        candidates = np.flatnonzero((truth_labels == det_labels[index]) & ~claimed)
        if len(candidates) == 0:
            continue
        best = candidates[np.argmax(ious[index, candidates])]
        if ious[index, best] >= iou_threshold:
            matched[index] = True
            claimed[best] = True
    return matched


def fit_platt(scores: np.ndarray, matched: np.ndarray) -> tuple[float, float]:
    """Fit ``sigmoid(a * logit(score) + b)`` to the match labels by Newton descent.

    Two parameters against thousands of points; scipy would also do, but a
    hand-rolled Newton solve keeps this importable without it and converges in a
    handful of iterations on a problem this convex.
    """
    scores = np.clip(np.asarray(scores, dtype=np.float64), _EPS, 1.0 - _EPS)
    labels = np.asarray(matched, dtype=np.float64)
    if labels.min() == labels.max():
        raise ValueError(
            "calibration needs both matched and unmatched detections; "
            "got only one kind"
        )

    features = np.log(scores / (1.0 - scores))

    def loss(scale: float, bias: float) -> float:
        z = np.clip(scale * features + bias, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, _EPS, 1.0 - _EPS)
        return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))

    scale, bias = 1.0, 0.0
    current = loss(scale, bias)

    for _ in range(100):
        z = np.clip(scale * features + bias, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        gradient = np.array([np.sum((p - labels) * features), np.sum(p - labels)])
        w = p * (1.0 - p)
        hessian = np.array(
            [
                [np.sum(w * features * features), np.sum(w * features)],
                [np.sum(w * features), np.sum(w)],
            ]
        )
        hessian += np.eye(2) * 1e-6
        step = np.linalg.solve(hessian, gradient)

        # Backtracking: a raw Newton step overshoots badly when the start is far
        # from the optimum -- exactly the squashed-score case this solver exists
        # for. Without it the fit diverged to scale ~1e9 on the first test run.
        damping = 1.0
        for _ in range(30):
            candidate = loss(scale - damping * step[0], bias - damping * step[1])
            if candidate < current:
                break
            damping /= 2.0
        else:
            break  # no descent direction left; converged as far as it goes

        scale, bias = scale - damping * step[0], bias - damping * step[1]
        if current - candidate < 1e-12:
            current = candidate
            break
        current = candidate

    if scale <= 0:
        # A non-monotone fit would reorder detections and change mAP; refuse.
        raise ValueError(
            f"Platt fit produced non-positive scale {scale:.4f}; the scores do "
            f"not rank matches above non-matches, so calibration cannot help"
        )
    return float(scale), float(bias)


def choose_threshold(
    calibrated: np.ndarray, matched: np.ndarray, total_truth: int
) -> tuple[float, float, float, float]:
    """Threshold maximising F1, returning ``(threshold, precision, recall, f1)``.

    Recall's denominator is *all* ground-truth objects, including ones no detection
    found at any score -- otherwise missed pieces would silently not count.
    """
    order = np.argsort(-calibrated)
    scores = calibrated[order]
    labels = matched[order].astype(np.float64)

    true_positives = np.cumsum(labels)
    kept = np.arange(1, len(scores) + 1)
    precision = true_positives / kept
    recall = true_positives / max(1, total_truth)
    f1 = 2 * precision * recall / np.clip(precision + recall, _EPS, None)

    best = int(np.argmax(f1))
    # Halfway to the next score keeps the operating point off a sample boundary.
    lower = scores[best + 1] if best + 1 < len(scores) else 0.0
    threshold = float((scores[best] + lower) / 2.0)
    return threshold, float(precision[best]), float(recall[best]), float(f1[best])


def collect_detections(
    model,
    processor,
    dataset,
    device,
    *,
    limit: int | None = None,
    top_k: int = TOP_K,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run the detector over a dataset, returning scores, match labels and the
    ground-truth count."""
    from chesssight.train.dataset import _annotations
    from chesssight.train.evaluate import predict_sample
    from chesssight.train.labels import LABEL2ID

    all_scores: list[float] = []
    all_matched: list[bool] = []
    total_truth = 0

    count = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(count):
        sample = dataset.sample(index)
        annotations = _annotations(sample, include_board=dataset.include_board)
        truth_boxes = np.asarray(
            [
                [
                    a["bbox"][0],
                    a["bbox"][1],
                    a["bbox"][0] + a["bbox"][2],
                    a["bbox"][1] + a["bbox"][3],
                ]
                for a in annotations
            ],
            dtype=np.float64,
        ).reshape(-1, 4)
        truth_labels = np.asarray(
            [a["category_id"] for a in annotations], dtype=np.int64
        )
        total_truth += len(annotations)

        detections = predict_sample(
            model, processor, dataset, index, device, threshold=0.0
        )[:top_k]
        if not detections:
            continue
        det_boxes = np.asarray([d["box"] for d in detections], dtype=np.float64)
        det_labels = np.asarray(
            [LABEL2ID[d["label"]] for d in detections], dtype=np.int64
        )
        det_scores = np.asarray([d["score"] for d in detections], dtype=np.float64)

        matched = match_detections(
            det_boxes, det_labels, det_scores, truth_boxes, truth_labels
        )
        all_scores.extend(det_scores.tolist())
        all_matched.extend(matched.tolist())

    return (
        np.asarray(all_scores, dtype=np.float64),
        np.asarray(all_matched, dtype=bool),
        total_truth,
    )


def calibrate(
    model,
    processor,
    dataset,
    device,
    *,
    fit_split: str,
    limit: int | None = None,
) -> Calibration:
    """Fit calibration on one dataset split and package the result."""
    scores, matched, total_truth = collect_detections(
        model, processor, dataset, device, limit=limit
    )
    scale, bias = fit_platt(scores, matched)

    interim = Calibration(
        scale=scale,
        bias=bias,
        threshold=0.5,
        fit_split=fit_split,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        detections_used=len(scores),
    )
    calibrated = interim.apply(scores)
    threshold, precision, recall, f1 = choose_threshold(
        calibrated, matched, total_truth
    )
    return Calibration(
        scale=scale,
        bias=bias,
        threshold=threshold,
        fit_split=fit_split,
        precision=precision,
        recall=recall,
        f1=f1,
        detections_used=len(scores),
    )
