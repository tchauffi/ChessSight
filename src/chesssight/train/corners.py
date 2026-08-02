"""Read board corners back out of a detector's predictions.

The detector carries corners as a class of small boxes (see
:data:`chesssight.train.labels.CORNER_INDEX`). This module turns those boxes into an
ordered quadrilateral and, from it, the board-plane homography -- which is what turns
a pile of piece boxes into a *position*.

Ordering is geometric, not learned. The model predicts four interchangeable corners
because a chessboard's appearance cannot say which one is a8; what the geometry *can*
say is which is which in image space, consistently. Naming the a8 corner needs the
pieces, and belongs downstream.
"""

from __future__ import annotations

import numpy as np

from chesssight.train.labels import CORNER_INDEX


def centres(detections: list[dict]) -> list[tuple[float, float, float]]:
    """Centre and score of every corner detection, best first."""
    found = [
        (
            (d["box"][0] + d["box"][2]) / 2.0,
            (d["box"][1] + d["box"][3]) / 2.0,
            float(d["score"]),
        )
        for d in detections
        if d["label"] == CORNER_INDEX
    ]
    return sorted(found, key=lambda point: point[2], reverse=True)


def select_quad(
    detections: list[dict], *, min_separation: float = 0.05
) -> list[list[float]] | None:
    """Pick the four corner detections that describe one board, or None.

    Taking the top four scores alone is not enough: the model happily puts two boxes
    on the same physical corner, and four points that include a duplicate produce a
    degenerate homography rather than an obviously wrong one. Candidates are therefore
    accepted greedily by score and rejected when they land on top of an
    already-accepted point.

    ``min_separation`` is a fraction of the spread of the candidates, so it scales
    with how large the board appears rather than assuming a pixel distance.
    """
    found = centres(detections)
    if len(found) < 4:
        return None

    xs = [point[0] for point in found]
    ys = [point[1] for point in found]
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    if spread <= 0:
        return None
    floor = spread * min_separation

    kept: list[tuple[float, float]] = []
    for x, y, _ in found:
        if all(np.hypot(x - kx, y - ky) >= floor for kx, ky in kept):
            kept.append((x, y))
        if len(kept) == 4:
            break
    if len(kept) < 4:
        return None
    return order_clockwise(kept)


def order_clockwise(points: list[tuple[float, float]]) -> list[list[float]]:
    """Order four points clockwise from the top-left in image space.

    By angle about the centroid rather than by sorting coordinates: a board seen from
    a low, rotated viewpoint has corners whose x and y orderings do not correspond to
    its sides at all, and coordinate sorting silently returns a bow-tie.
    """
    array = np.asarray(points, dtype=np.float64)
    centre = array.mean(axis=0)
    angles = np.arctan2(array[:, 1] - centre[1], array[:, 0] - centre[0])
    ordered = array[np.argsort(angles)]

    # Rotate so the point closest to the top-left of the bounding box comes first,
    # giving a stable starting corner for the same board across frames.
    corner = np.array([array[:, 0].min(), array[:, 1].min()])
    start = int(np.argmin(np.linalg.norm(ordered - corner, axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    return [[float(x), float(y)] for x, y in ordered]


def homography_from(detections: list[dict]) -> np.ndarray | None:
    """Board-plane -> image homography from a frame's corner detections."""
    quad = select_quad(detections)
    if quad is None:
        return None
    from chesssight.data.geometry import board_to_image_homography

    try:
        return board_to_image_homography(np.asarray(quad, dtype=np.float64))
    except Exception:
        # A degenerate quad is a detection failure, not a crash: the caller gets
        # "no geometry this frame" and can fall back to the previous one.
        return None


def corner_error(
    predicted: list[list[float]], truth: list[list[float]]
) -> float | None:
    """Mean distance from each true corner to the nearest predicted one, in pixels.

    Nearest-match rather than index-match because the prediction's ordering starts at
    whichever corner is top-left in image space, which need not be the one the ground
    truth happens to list first. Measuring the ordering and the localisation together
    would report a large error for a perfectly-placed quad that merely starts
    elsewhere.
    """
    if not predicted or not truth:
        return None
    p = np.asarray(predicted, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    distances = np.linalg.norm(t[:, None, :] - p[None, :, :], axis=2)
    return float(distances.min(axis=1).mean())
