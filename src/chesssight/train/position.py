"""Turn detections plus a homography into an 8x8 position, and score it.

This is what the corner class is for. mAP says how well boxes are placed; it cannot
say whether the board was *read*, and those are different questions -- a detector can
score well while putting a bishop one square left of where it stands, which is a
perfect box and a wrong position.

Orientation is not resolved *here*. Four interchangeable corners give the board's
geometry up to a rotation, and deciding which corner is a8 needs a separate cue.
:mod:`chesssight.train.orientation` supplies one -- square colour narrows the four
candidates to two, and where White's men are settles the rest -- and measures 99% on
ChessReD, 92% on legal synthetic positions.

:func:`best_rotation` below therefore remains an *upper* bound and should be labelled
as one, but it is no longer the only number available: a pipeline that calls
:func:`chesssight.train.orientation.orient_position` gets a single committed answer
and can be scored honestly against it.
"""

from __future__ import annotations

import numpy as np

from chesssight.data.fen import BOARD_SIZE
from chesssight.data.geometry import board_to_image_homography
from chesssight.train.labels import index_to_class_id, is_piece

#: Where a piece touches the board, as a fraction of its box. Pieces are tall and
#: photographed from above the table, so the contact point is near the bottom of the
#: box; the box centre floats up the piece and lands a square or more behind.
FOOT_X = 0.5

#: Just *above* the bottom edge, not on it. The bottom of a 2D box is the
#: nearest-to-camera point of the piece's base, which sits forward of where its axis
#: meets the board, so a piece standing near a boundary tips over it. Measured over
#: 400 synthetic boards, where the ground truth is exact: board-exact 70.5% at 1.00
#: against 76.0% at 0.96, with 0.94-0.98 all within half a point of the peak. The
#: value was chosen on synthetic data and *then* checked against ChessReD, where it
#: takes the ground-truth ceiling from 98.7% to 100% -- so it is not a fit to the
#: real test set.
FOOT_Y = 0.96


def foot(box: list[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return x0 + (x1 - x0) * FOOT_X, y0 + (y1 - y0) * FOOT_Y


#: Operating point for reading a board, as opposed to detecting objects.
#:
#: A detector checkpoint ships a threshold fitted for detection F1, which
#: balances precision against recall. Reading a position does not want that
#: balance: :func:`grid_from` keeps only the best-scoring detection per square
#: and discards anything outside the board, so a spare low-scoring candidate is
#: usually harmless while a missing one always costs a square. Recall is worth
#: far more than precision here.
#:
#: Chosen by sweeping ChessReD **val** and measured on **test**. Swept per
#: detector: the value belongs to the shipped checkpoint's calibrated score
#: distribution, and rtdetr_v4's prior-bias head init decompressed the scores
#: enough to halve the optimum (the previous detector wanted 0.10). It lives in
#: this module rather than beside the reader so that naming it costs no torch
#: import.
POSITION_THRESHOLD = 0.05


def grid_from(detections: list[dict], homography: np.ndarray) -> list[list[int]]:
    """Assign each detected piece to a square, best score winning ties.

    One piece per square: two detections landing on the same square is a
    contradiction, and keeping the higher-scoring one is both the obvious
    resolution and a free accuracy gain over keeping the last one seen.
    """
    inverse = np.linalg.inv(homography)
    grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    best = [[0.0] * BOARD_SIZE for _ in range(BOARD_SIZE)]

    for detection in detections:
        if not is_piece(int(detection["label"])):
            continue
        x, y = foot(detection["box"])
        point = inverse @ np.array([x, y, 1.0])
        if abs(point[2]) < 1e-9:
            continue
        u, v = point[0] / point[2], point[1] / point[2]
        file_index, rank_index = int(np.floor(u)), int(np.floor(v))
        if not (0 <= file_index < BOARD_SIZE and 0 <= rank_index < BOARD_SIZE):
            continue  # a captured piece beside the board, or a false positive
        score = float(detection["score"])
        if score > best[rank_index][file_index]:
            best[rank_index][file_index] = score
            grid[rank_index][file_index] = index_to_class_id(int(detection["label"]))
    return grid


def rotations(grid: list[list[int]]) -> list[list[list[int]]]:
    """The same board read from each of the four sides."""
    array = np.asarray(grid)
    return [np.rot90(array, k).tolist() for k in range(4)]


def score_grid(
    predicted: list[list[int]], truth: list[list[int]]
) -> tuple[float, bool]:
    """Per-square accuracy and whether the whole board is exactly right."""
    correct = sum(
        1
        for rank in range(BOARD_SIZE)
        for file in range(BOARD_SIZE)
        if predicted[rank][file] == truth[rank][file]
    )
    total = BOARD_SIZE * BOARD_SIZE
    return correct / total, correct == total


def best_rotation(
    predicted: list[list[int]], truth: list[list[int]]
) -> tuple[float, bool]:
    """Score under whichever of the four orientations fits best.

    See the module docstring: this is an upper bound, because a deployed pipeline
    would have to choose the orientation without seeing the answer.
    """
    scored = [score_grid(candidate, truth) for candidate in rotations(predicted)]
    return max(scored, key=lambda result: result[0])


def read_position(
    detections: list[dict], corners: list[list[float]] | None
) -> list[list[int]] | None:
    """Detections plus corners -> a grid, or None when there is no geometry."""
    if corners is None or len(corners) != 4:
        return None
    try:
        homography = board_to_image_homography(np.asarray(corners, dtype=np.float64))
    except Exception:
        return None
    return grid_from(detections, homography)
