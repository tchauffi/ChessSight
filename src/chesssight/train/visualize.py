"""Draw a detector's predictions on an image, optionally beside the ground truth.

Metrics say how well a detector does on average. They do not say *how* it fails,
and the failure modes here are specific and worth seeing: a bishop called a pawn, a
piece behind another missed entirely, a board box that is a little loose. All of
those are one number in mAP and obvious in a picture.
"""

from __future__ import annotations

import torch
from PIL import Image, ImageDraw

from chesssight.data.schema import Sample
from chesssight.train.labels import BOARD_INDEX, DETECTION_LABELS, class_id_to_index

PREDICTION_COLOR = (255, 80, 80)
TRUTH_COLOR = (80, 220, 120)
BOARD_COLOR = (255, 205, 40)
HEADER_HEIGHT = 34

#: Predictions below this score are not drawn. RT-DETR emits 300 queries per image,
#: most of them near-empty, so an unfiltered overlay is unreadable.
DEFAULT_THRESHOLD = 0.5


@torch.no_grad()
def predict(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int | None = None,
    calibration=None,
) -> list[dict]:
    """Detections for one PIL image, in that image's own pixel coordinates.

    ``top_k`` takes the highest-scoring detections regardless of threshold. This is
    not a convenience: a partly-trained DETR ranks boxes well long before its
    classification head produces confident scores, so a fixed threshold shows an
    empty image and hides a model that is actually working.

    ``calibration`` is the finished version of the same idea: a Platt fit saved
    next to the checkpoint remaps the scores so a normal threshold works. It is
    monotone, so top-k selection is unaffected by applying it.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    sizes = torch.tensor([[image.height, image.width]], device=device)
    result = processor.post_process_object_detection(
        outputs,
        target_sizes=sizes,
        threshold=0.0 if (top_k or calibration) else threshold,
    )[0]

    detections = [
        {
            "label": int(label),
            "name": DETECTION_LABELS[int(label)],
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

    if calibration is not None:
        for detection in detections:
            raw = float(detection["score"])  # type: ignore[arg-type]
            detection["raw_score"] = raw
            detection["score"] = calibration.apply_one(raw)
        if not top_k:
            detections = [
                d for d in detections if float(d["score"]) >= threshold  # type: ignore[arg-type]
            ]

    if top_k:
        detections.sort(key=lambda d: float(d["score"]), reverse=True)  # type: ignore[arg-type]
        detections = detections[:top_k]
    return detections


def on_board_only(predictions: list[dict], sample: Sample) -> list[dict]:
    """Keep detections whose foot lies inside the board outline.

    This is an annotation-convention adjustment, not a trick to flatter the model.
    The synthetic data deliberately contains captured pieces beside the board and
    teaches the detector to find them; ChessReD annotates on-board pieces only. So
    a correct detection of a captured piece scores as a false positive, and worse,
    it consumes budget that should have gone to the board.

    Filtering here measures board-reading against ChessReD's convention. Drop the
    filter to measure detection, which is what matters if a downstream consumer
    cares about captured material.
    """
    from chesssight.data.geometry import polygon_contains

    kept = []
    for prediction in predictions:
        if prediction["label"] == BOARD_INDEX:
            kept.append(prediction)
            continue
        x0, _, x1, y1 = prediction["box"]
        if polygon_contains(sample.board.corners_px, ((x0 + x1) / 2.0, y1)):
            kept.append(prediction)
    return kept


def take_top(predictions: list[dict], count: int) -> list[dict]:
    """The ``count`` highest-scoring detections, board box excluded from the count."""
    pieces = sorted(
        (p for p in predictions if p["label"] != BOARD_INDEX),
        key=lambda d: float(d["score"]),  # type: ignore[arg-type]
        reverse=True,
    )[:count]
    board = [p for p in predictions if p["label"] == BOARD_INDEX][:1]
    return pieces + board


def _scaled(image: Image.Image, width: int) -> tuple[Image.Image, float]:
    if image.width <= width:
        return image.copy(), 1.0
    factor = width / image.width
    size = (width, max(1, int(image.height * factor)))
    return image.resize(size, Image.Resampling.LANCZOS), factor


def draw_predictions(
    image: Image.Image,
    predictions: list[dict],
    *,
    sample: Sample | None = None,
    show_truth: bool = True,
    width: int = 900,
    title: str = "",
) -> Image.Image:
    """Render predictions (red) over an image, with ground truth in green.

    Both are drawn at once because the interesting information is in the
    difference -- a box with no match under it, or a green box with nothing on it.
    """
    scaled, factor = _scaled(image, width)
    canvas = Image.new(
        "RGB", (scaled.width, scaled.height + HEADER_HEIGHT), (16, 16, 20)
    )
    canvas.paste(scaled, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)

    def place(box: list[float]) -> list[float]:
        x0, y0, x1, y1 = (value * factor for value in box)
        return [x0, y0 + HEADER_HEIGHT, x1, y1 + HEADER_HEIGHT]

    if show_truth and sample is not None:
        for piece in sample.pieces:
            if piece.bbox is None:
                continue
            draw.rectangle(place(list(piece.bbox.xyxy)), outline=TRUTH_COLOR)
        corners = [
            (x * factor, y * factor + HEADER_HEIGHT) for x, y in sample.board.corners_px
        ]
        draw.line([*corners, corners[0]], fill=TRUTH_COLOR, width=2)

    for prediction in predictions:
        colour = BOARD_COLOR if prediction["label"] == BOARD_INDEX else PREDICTION_COLOR
        box = place(prediction["box"])
        draw.rectangle(box, outline=colour, width=2)
        draw.text(
            (box[0] + 2, box[1] - 10),
            f"{prediction['name'].replace('_', ' ')} {prediction['score']:.2f}",
            fill=colour,
        )

    pieces = sum(1 for p in predictions if p["label"] != BOARD_INDEX)
    truth_pieces = (
        sum(1 for p in sample.pieces if p.bbox is not None) if sample else None
    )
    summary = f"{pieces} pieces predicted"
    if truth_pieces is not None:
        summary += f" / {truth_pieces} annotated"
    draw.text((6, 4), title or summary, fill=(235, 235, 235))
    draw.text(
        (6, 18),
        f"{summary}   red: predicted piece   yellow: predicted board   "
        f"green: ground truth",
        fill=(150, 150, 160),
    )
    return canvas


def square_accuracy(sample: Sample, predictions: list[dict]) -> dict[str, float]:
    """How well the predictions reproduce the position, square by square.

    A detector's mAP is about boxes; what this project ultimately wants is the
    position. Assigning each prediction to the square under its box's foot -- a
    piece stands on its square -- turns one into the other, and shows whether good
    boxes are actually producing a readable board.
    """
    import numpy as np

    from chesssight.data.fen import BOARD_SIZE

    centers = np.asarray([square.center_px for square in sample.squares])
    # square index -> (label, score), keeping the most confident claim when two
    # boxes land on the same square.
    claimed: dict[int, tuple[int, float]] = {}

    for prediction in predictions:
        if prediction["label"] == BOARD_INDEX:
            continue
        x0, _, x1, y1 = prediction["box"]
        foot = np.array([(x0 + x1) / 2.0, y1])
        index = int(np.argmin(np.linalg.norm(centers - foot, axis=1)))
        existing = claimed.get(index)
        if existing is None or prediction["score"] > existing[1]:
            claimed[index] = (prediction["label"], prediction["score"])

    correct = 0
    occupied_correct = 0
    occupied_total = 0
    for index, square in enumerate(sample.squares):
        truth = class_id_to_index(square.occupant) if square.occupant else None
        guess = claimed[index][0] if index in claimed else None
        if truth is not None:
            occupied_total += 1
            occupied_correct += guess == truth
        correct += guess == truth

    return {
        "squares_correct": correct / (BOARD_SIZE * BOARD_SIZE),
        "occupied_correct": occupied_correct / max(1, occupied_total),
        "board_exact": float(correct == BOARD_SIZE * BOARD_SIZE),
    }
