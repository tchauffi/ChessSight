"""Detection metrics.

mAP is reported overall and per class. The overall number hides the thing that
actually matters here: a board is one enormous, easy box and the pieces are dozens
of small ones, so a single averaged figure can look healthy while every pawn is
being missed.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from chesssight.train.dataset import ChessDetectionDataset
from chesssight.train.labels import DETECTION_LABELS, class_id_to_index


def _targets_to_xyxy(labels: list[dict], sizes: torch.Tensor) -> list[dict]:
    """Convert normalised cxcywh targets back to absolute xyxy.

    The processor hands the model boxes normalised to the padded image; the metric
    wants pixels. Doing this conversion in one place, against the same size tensor
    the post-processor uses, is what keeps predictions and targets in the same
    coordinate system -- a mismatch here silently reports mAP near zero.
    """
    converted = []
    for label, (height, width) in zip(labels, sizes, strict=True):
        boxes = label["boxes"]
        if boxes.numel():
            cx, cy, bw, bh = boxes.unbind(-1)
            boxes = torch.stack(
                [
                    (cx - bw / 2) * width,
                    (cy - bh / 2) * height,
                    (cx + bw / 2) * width,
                    (cy + bh / 2) * height,
                ],
                dim=-1,
            )
        converted.append({"boxes": boxes, "labels": label["class_labels"]})
    return converted


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    processor,
    device: torch.device,
    *,
    threshold: float = 0.0,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run the detector over a loader and return mAP, overall and per class."""
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"]
        height, width = pixel_values.shape[-2:]
        sizes = torch.tensor(
            [[height, width]] * len(labels), device=device, dtype=torch.float32
        )

        outputs = model(pixel_values=pixel_values)
        predictions = processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=threshold
        )

        metric.update(
            [
                {
                    "boxes": prediction["boxes"].cpu(),
                    "scores": prediction["scores"].cpu(),
                    "labels": prediction["labels"].cpu(),
                }
                for prediction in predictions
            ],
            [
                {"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()}
                for target in _targets_to_xyxy(labels, sizes.cpu())
            ],
        )

    computed = metric.compute()
    report = {
        "map": float(computed["map"]),
        "map_50": float(computed["map_50"]),
        "map_75": float(computed["map_75"]),
        "map_small": float(computed["map_small"]),
        "map_medium": float(computed["map_medium"]),
        "map_large": float(computed["map_large"]),
    }

    per_class = computed.get("map_per_class")
    classes = computed.get("classes")
    if per_class is not None and classes is not None:
        for value, class_index in zip(
            per_class.tolist(), classes.tolist(), strict=True
        ):
            name = DETECTION_LABELS[int(class_index)]
            report[f"map/{name}"] = float(value)
    return report


def format_report(report: dict[str, float]) -> str:
    """Human-readable metrics, overall first then per class."""
    lines = [
        f"  mAP           {report['map']:.4f}",
        f"  mAP@50        {report['map_50']:.4f}",
        f"  mAP@75        {report['map_75']:.4f}",
        f"  mAP small     {report['map_small']:.4f}",
        f"  mAP medium    {report['map_medium']:.4f}",
        f"  mAP large     {report['map_large']:.4f}",
    ]
    per_class = sorted(
        (key[4:], value) for key, value in report.items() if key.startswith("map/")
    )
    if per_class:
        lines.append("  per class:")
        lines.extend(f"    {name:<14} {value:.4f}" for name, value in per_class)
    return "\n".join(lines)


@torch.no_grad()
def predict_sample(
    model,
    processor,
    dataset: ChessDetectionDataset,
    index: int,
    device: torch.device,
    *,
    threshold: float = 0.5,
) -> list[dict]:
    """Detections for one dataset entry, in original image pixels."""
    from PIL import Image

    sample = dataset.sample(index)
    image = Image.open(dataset.root / sample.image).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    sizes = torch.tensor([[image.height, image.width]], device=device)
    result = processor.post_process_object_detection(
        outputs, target_sizes=sizes, threshold=threshold
    )[0]

    return [
        {
            "label": DETECTION_LABELS[int(label)],
            "score": float(score),
            "box": [float(value) for value in box],
        }
        for score, label, box in zip(
            result["scores"].cpu(),
            result["labels"].cpu(),
            result["boxes"].cpu(),
            strict=True,
        )
    ]


def _board_box(sample) -> list[float] | None:
    from chesssight.data.export import board_bbox

    box = board_bbox(sample)
    return list(box.xyxy) if box is not None else None


@torch.no_grad()
def evaluate_samples(
    model,
    processor,
    reader,
    device: torch.device,
    *,
    split: str = "test",
    on_board: bool = False,
    limit: int | None = None,
) -> dict[str, float]:
    """Evaluate sample-by-sample, so each image's board polygon is available.

    The DataLoader path cannot filter by board: it yields tensors, not samples.
    Filtering matters because the synthetic data deliberately teaches the detector
    to find captured pieces beside the board, while ChessReD annotates on-board
    pieces only -- so correct detections of captured pieces score as false
    positives against it.
    """
    from PIL import Image

    from chesssight.data.geometry import polygon_contains
    from chesssight.train.labels import BOARD_INDEX

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    entries = [e for e in reader.entries() if split == "all" or e.split == split]
    if limit:
        entries = entries[:limit]

    for entry in entries:
        sample = reader.load(entry.id)
        image = Image.open(reader.root / sample.image).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        sizes = torch.tensor([[image.height, image.width]], device=device)
        result = processor.post_process_object_detection(
            outputs, target_sizes=sizes, threshold=0.0
        )[0]

        boxes = result["boxes"].cpu()
        scores = result["scores"].cpu()
        labels = result["labels"].cpu()

        if on_board:
            keep = [
                index
                for index in range(len(boxes))
                if int(labels[index]) == BOARD_INDEX
                or polygon_contains(
                    sample.board.corners_px,
                    (
                        (boxes[index][0].item() + boxes[index][2].item()) / 2.0,
                        boxes[index][3].item(),
                    ),
                )
            ]
            selection = torch.tensor(keep, dtype=torch.long)
            boxes = boxes[selection]
            scores = scores[selection]
            labels = labels[selection]

        truth_boxes: list[list[float]] = []
        truth_labels: list[int] = []
        for piece in sample.pieces:
            if piece.bbox is None:
                continue
            truth_boxes.append(list(piece.bbox.xyxy))
            truth_labels.append(class_id_to_index(piece.class_id))
        board = _board_box(sample)
        if board is not None:
            truth_boxes.append(board)
            truth_labels.append(BOARD_INDEX)

        metric.update(
            [{"boxes": boxes, "scores": scores, "labels": labels}],
            [
                {
                    "boxes": torch.tensor(truth_boxes, dtype=torch.float32).reshape(
                        -1, 4
                    ),
                    "labels": torch.tensor(truth_labels, dtype=torch.long),
                }
            ],
        )

    computed = metric.compute()
    report = {
        key: float(computed[key])
        for key in ("map", "map_50", "map_75", "map_small", "map_medium", "map_large")
    }
    per_class = computed.get("map_per_class")
    classes = computed.get("classes")
    if per_class is not None and classes is not None:
        for value, index in zip(per_class.tolist(), classes.tolist(), strict=True):
            report[f"map/{DETECTION_LABELS[int(index)]}"] = float(value)
    return report
