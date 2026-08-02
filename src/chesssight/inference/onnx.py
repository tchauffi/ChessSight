"""Run the pipeline from ONNX graphs instead of the torch checkpoints.

Why this exists
---------------
The two checkpoints need torch, transformers and timm to load -- about three
gigabytes of wheels for a demo whose job is one forward pass. Exported to ONNX
the same two models run under ``onnxruntime`` alone, on CPU, with no training
stack present.

What is shared and what is not
------------------------------
Everything that decides *meaning* is imported, not reimplemented: which label
is which class, where a detection stands on the board, which corner is a8, how
a grid becomes a FEN. Those modules are already torch-free, so this backend
calls exactly the code the measured pipeline calls.

What had to be written twice is the numeric plumbing on either side of the
model: turning a PIL image into a tensor, and turning raw head outputs into
boxes and peaks. That is a second implementation of a rule, which in this
repository has a history of drifting silently, so it is not trusted --
``chesssight onnx parity`` re-reads a real split through both backends and
fails on the first FEN that differs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from chesssight.train.labels import DETECTION_LABELS
from chesssight.train.position import POSITION_THRESHOLD

#: ImageNet statistics, the corner model's normalisation.
CORNER_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
CORNER_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Filenames inside an exported bundle.
DETECTOR_FILE = "detector.onnx"
CORNERS_FILE = "corners.onnx"
META_FILE = "meta.json"


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def export(detector: Path, corners: Path, out: Path) -> Path:
    """Write both graphs and the metadata needed to run them, into ``out``.

    Needs torch; running the result does not.
    """
    import torch

    from chesssight.train.calibrate import Calibration
    from chesssight.train.heatmap import load as load_corners
    from chesssight.train.run import load_trained

    out.mkdir(parents=True, exist_ok=True)

    model, processor, _ = load_trained(detector, "cpu")
    model.eval()
    size = int(processor.size["height"])

    class TwoHeads(torch.nn.Module):
        """RT-DETR returns a dataclass of a dozen tensors, most of them
        auxiliary decoder layers. Exporting it whole puts all of them in the
        graph; only the final logits and boxes are ever read."""

        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values: Any) -> tuple[Any, Any]:
            outputs = self.wrapped(pixel_values=pixel_values)
            return outputs.logits, outputs.pred_boxes

    torch.onnx.export(
        TwoHeads(model),
        (torch.zeros(1, 3, size, size),),
        str(out / DETECTOR_FILE),
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=17,
        dynamo=False,
    )

    corner_model, corner_config = load_corners(corners, "cpu")
    corner_model.eval()
    corner_size = int(corner_config.image_size)
    torch.onnx.export(
        corner_model,
        (torch.zeros(1, 3, corner_size, corner_size),),
        str(out / CORNERS_FILE),
        input_names=["image"],
        output_names=["heatmap"],
        opset_version=17,
        dynamo=False,
    )

    calibration = Calibration.load(detector)
    meta = {
        "detector_size": size,
        "rescale": float(processor.rescale_factor),
        "corner_size": corner_size,
        "corner_stride": int(corner_config.stride),
        "labels": list(DETECTION_LABELS),
        "calibration": (
            None
            if calibration is None
            else {
                "scale": calibration.scale,
                "bias": calibration.bias,
                "threshold": calibration.threshold,
            }
        ),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# the numeric plumbing that had to be written twice
# --------------------------------------------------------------------------
def detector_input(image: Image.Image, size: int, rescale: float) -> np.ndarray:
    """PIL image -> ``1x3xSxS`` float32, matching RT-DETR's processor.

    That processor resizes to a square and rescales; it does *not* normalise
    (``do_normalize`` is false in the saved config), which is easy to get wrong
    by analogy with every other detector.
    """
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) * np.float32(rescale)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def corner_input(image: Image.Image, size: int) -> np.ndarray:
    """PIL image -> ``1x3xSxS`` float32, ImageNet-normalised."""
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    array = (array - CORNER_MEAN) / CORNER_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None].astype(np.float32))


def decode_detections(
    logits: np.ndarray,
    boxes: np.ndarray,
    size: tuple[int, int],
    threshold: float,
) -> list[dict[str, Any]]:
    """Raw heads -> detections in the original image's pixels.

    Mirrors ``RTDetrImageProcessor.post_process_object_detection`` with
    ``use_focal_loss``: scores are sigmoid, the top queries are taken over the
    flattened score matrix, so one query can contribute more than one class.
    """
    width, height = size
    scores = 1.0 / (1.0 + np.exp(-logits[0].astype(np.float64)))
    queries, classes = scores.shape

    flat = scores.reshape(-1)
    take = queries
    top = np.argpartition(-flat, take - 1)[:take]
    top = top[np.argsort(-flat[top], kind="stable")]

    labels = top % classes
    query_index = top // classes
    chosen = boxes[0][query_index].astype(np.float64)

    # cxcywh (relative) -> xyxy (absolute)
    cx, cy, w, h = chosen[:, 0], chosen[:, 1], chosen[:, 2], chosen[:, 3]
    xyxy = np.stack(
        [
            (cx - w / 2) * width,
            (cy - h / 2) * height,
            (cx + w / 2) * width,
            (cy + h / 2) * height,
        ],
        axis=1,
    )

    keep = flat[top] > threshold
    return [
        {
            "label": int(label),
            "name": DETECTION_LABELS[int(label)],
            "score": float(score),
            "box": [float(v) for v in box],
        }
        for label, score, box in zip(
            labels[keep], flat[top][keep], xyxy[keep], strict=True
        )
    ]


def decode_peaks(
    heatmap: np.ndarray,
    *,
    stride: int,
    count: int = 4,
    radius: int = 2,
) -> list[tuple[float, float, float]]:
    """Heatmap -> ``(x, y, score)`` peaks in input-image pixels.

    The torch original in :mod:`chesssight.train.heatmap`: sigmoid, a 3x3
    maximum filter so one blob yields one point, top-k, then an
    intensity-weighted mean over a small window to place the point between
    cells.
    """
    scores = 1.0 / (
        1.0 + np.exp(-heatmap.astype(np.float64).reshape(heatmap.shape[-2:]))
    )
    padded = np.pad(scores, 1, mode="constant", constant_values=-np.inf)
    pooled = np.max(
        np.stack(
            [
                padded[dy : dy + scores.shape[0], dx : dx + scores.shape[1]]
                for dy in range(3)
                for dx in range(3)
            ]
        ),
        axis=0,
    )
    peaks = scores * (scores >= pooled)

    height, width = peaks.shape
    flat = peaks.reshape(-1)
    take = min(count, flat.size)
    top = np.argpartition(-flat, take - 1)[:take]
    top = top[np.argsort(-flat[top], kind="stable")]

    found: list[tuple[float, float, float]] = []
    for index in top:
        cy, cx = divmod(int(index), width)
        left, right = max(0, cx - radius), min(width, cx + radius + 1)
        upper, lower = max(0, cy - radius), min(height, cy + radius + 1)
        window = scores[upper:lower, left:right]
        weight = window.sum()
        if weight > 0:
            xs = np.arange(left, right, dtype=np.float64)
            ys = np.arange(upper, lower, dtype=np.float64)
            x = float((window.sum(axis=0) * xs).sum() / weight)
            y = float((window.sum(axis=1) * ys).sum() / weight)
        else:
            x, y = float(cx), float(cy)
        found.append(((x + 0.5) * stride, (y + 0.5) * stride, float(flat[index])))
    return found


# --------------------------------------------------------------------------
# the reader
# --------------------------------------------------------------------------
@dataclass
class OnnxReader:
    """The same contract as ``PositionReader``: one image in, one dict out."""

    detector: Any
    corners: Any
    meta: dict[str, Any]
    #: Kept in step with the torch reader deliberately: two backends reading
    #: the same photograph at different operating points would disagree by
    #: construction, and `onnx parity` would be measuring the thresholds
    #: rather than the graphs.
    threshold: float | None = POSITION_THRESHOLD

    def read(self, image: Image.Image) -> dict[str, Any]:
        from chesssight.data.fen import grid_to_fen
        from chesssight.data.geometry import board_to_image_homography
        from chesssight.train.corners import order_clockwise
        from chesssight.train.orientation import orient_position
        from chesssight.train.position import grid_from

        corner_size = self.meta["corner_size"]
        heatmap = self.corners.run(
            ["heatmap"], {"image": corner_input(image, corner_size)}
        )[0]
        peaks = decode_peaks(heatmap, stride=self.meta["corner_stride"])
        if len(peaks) < 4:
            return {"corners": None, "grid": None, "fen": None, "detections": []}

        width, height = image.size
        quad = order_clockwise(
            [
                (x * width / corner_size, y * height / corner_size)
                for x, y, _ in peaks[:4]
            ]
        )

        calibration = self.meta.get("calibration")
        threshold = 0.0 if calibration else 0.3
        logits, boxes = self.detector.run(
            ["logits", "pred_boxes"],
            {
                "pixel_values": detector_input(
                    image, self.meta["detector_size"], self.meta["rescale"]
                )
            },
        )
        detections = decode_detections(logits, boxes, image.size, threshold)

        if calibration:
            from chesssight.train.calibrate import Calibration

            platt = Calibration(
                scale=calibration["scale"],
                bias=calibration["bias"],
                threshold=calibration["threshold"],
                fit_split="",
                precision=0.0,
                recall=0.0,
                f1=0.0,
                detections_used=0,
            )
            for detection in detections:
                detection["score"] = platt.apply_one(detection["score"])
            floor = (
                self.threshold
                if self.threshold is not None
                else calibration["threshold"]
            )
            detections = [d for d in detections if d["score"] >= floor]

        homography = board_to_image_homography(np.asarray(quad, dtype=np.float64))
        grid = grid_from(detections, homography)
        grid, quad, evidence = orient_position(grid, quad, image)
        return {
            "corners": quad,
            "grid": grid,
            "fen": grid_to_fen(grid),
            "detections": detections,
            "orientation_evidence": evidence,
        }


def load_reader(
    bundle: Path,
    providers: list[str] | None = None,
    threshold: float | None = POSITION_THRESHOLD,
) -> OnnxReader:
    """Load an exported bundle. Needs onnxruntime, not torch."""
    import onnxruntime as ort

    bundle = Path(bundle)
    meta = json.loads((bundle / META_FILE).read_text(encoding="utf-8"))
    chosen = providers or ["CPUExecutionProvider"]
    return OnnxReader(
        detector=ort.InferenceSession(str(bundle / DETECTOR_FILE), providers=chosen),
        corners=ort.InferenceSession(str(bundle / CORNERS_FILE), providers=chosen),
        meta=meta,
        threshold=threshold,
    )
