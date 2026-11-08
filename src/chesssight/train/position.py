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

from chesssight.data.fen import (
    BOARD_SIZE,
    CLASS_TO_LETTER,
    grid_to_fen,
)
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
#: balance: a square keeps only its best-scoring class and anything outside the
#: board is discarded, so a spare low-scoring candidate is usually harmless
#: while a missing one always costs a square. Recall is worth far more than
#: precision here.
#:
#: It applies to the *fused* score of :func:`read_board` -- the weighted mean
#: over three detector passes -- not to a single pass. A class seen by only one
#: pass is diluted by the others, so the fused optimum sits above the
#: single-pass one (0.05 for this checkpoint read once).
#:
#: Chosen by sweeping ChessReD **val** and measured on **test**. Swept per
#: detector: the value belongs to the shipped checkpoint's calibrated score
#: distribution. It lives in this module rather than beside the reader so that
#: naming it costs no torch import.
POSITION_THRESHOLD = 0.08

#: Below this a fused score is not even an *alternative* for its square.
#: :func:`enforce_material` relabels a square only to a class the detector did
#: propose there, and this bounds how faint that proposal may be. The measured
#: result is flat from 0.003 to 0.02.
CANDIDATE_FLOOR = 0.01

#: How far the board crop extends past the corners' bounding box, as a fraction
#: of its width and height. The top margin is larger because pieces stand *up*
#: from the board and the far rank's tops overhang the far edge. Set by
#: reasoning about piece height, not swept.
CROP_SIDE_MARGIN = 0.10
CROP_TOP_MARGIN = 0.30

#: Starting material per colour: more of a type than this needs a promotion,
#: which real games produce in under 0.1% of positions (Lichess 2013-01). The
#: king is exact in both directions.
STARTING_MATERIAL = {"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1}

#: Cost, in log-score units, of each piece over :data:`STARTING_MATERIAL`.
#: Large enough that any proposed alternative (score >= CANDIDATE_FLOOR, so a
#: relabel costs at most ~log(1/0.01) = 4.6) is preferred to an extra queen.
#: ChessReD reads identically for any value from 10 to 1000 (3 loses one test
#: board), so this is a hard rule with a tie-break, not a fitted weight.
OVER_MATERIAL_COST = 10.0


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


# --------------------------------------------------------------------------
# reading a board from several detector passes
# --------------------------------------------------------------------------
#: ``(rank index, file index, class id) -> score``: every class the detector
#: proposed for every square, not only the winner. RT-DETR's post-processing
#: takes its top queries over the flattened query x class matrix, so one box
#: routinely appears under two labels -- a queen at 0.6 and a king at 0.3 --
#: and the runner-up is exactly what :func:`enforce_material` needs.
SquareScores = dict[tuple[int, int, int], float]


def square_scores(detections: list[dict], homography: np.ndarray) -> SquareScores:
    """The best score each class reached on each square, from one pass."""
    inverse = np.linalg.inv(homography)
    scores: SquareScores = {}
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
            continue
        key = (rank_index, file_index, index_to_class_id(int(detection["label"])))
        scores[key] = max(scores.get(key, 0.0), float(detection["score"]))
    return scores


def fuse(passes: list[tuple[SquareScores, float]]) -> SquareScores:
    """Weighted mean of several passes' square scores.

    A class one pass did not propose counts as zero there, so agreement is
    rewarded and a lone pass's guess is diluted. That is the point: on ChessReD
    the full frame and the board crop each fix ~30 boards the other gets wrong,
    and keeping the per-square *maximum* over passes instead measured no better
    than either alone -- it keeps both passes' mistakes.
    """
    total = sum(weight for _, weight in passes)
    fused: SquareScores = {}
    for scores, weight in passes:
        for key, score in scores.items():
            fused[key] = fused.get(key, 0.0) + weight * score
    return {key: value / total for key, value in fused.items()}


def grid_from_scores(scores: SquareScores, threshold: float) -> list[list[int]]:
    """Each square's best class, if it clears ``threshold``."""
    grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    best = [[0.0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for (rank, file, class_id), score in scores.items():
        if score >= threshold and score > best[rank][file]:
            best[rank][file] = score
            grid[rank][file] = class_id
    return grid


def _material_cost(class_id: int, count: int) -> float:
    cap = STARTING_MATERIAL[CLASS_TO_LETTER[class_id].upper()]
    return OVER_MATERIAL_COST * max(0, count - cap)


def enforce_material(
    grid: list[list[int]], scores: SquareScores, floor: float = CANDIDATE_FLOOR
) -> list[list[int]]:
    """Relabel squares until no colour holds more than its starting material.

    A board with two white kings is not a position, and the detector's
    commonest confusion -- a queen read as her king -- produces exactly that.
    Every relabel is to a class the detector itself proposed on that square
    (fused score >= ``floor``), so the rule chooses between the model's own
    readings and never invents a piece.

    Greedily, the single relabel that most improves ``sum(log score) -
    material cost`` is applied until none does -- best move first, so of two
    queens it is the *weaker* reading that gives way. Measured inside
    :func:`read_board` on ChessReD: 19 more test boards exact and 9 more on
    val (17 and 11 with the board zoomed out to ~37% of the frame), and not
    one board made worse in any of the four.

    The rule only removes excess; it never fills a shortfall. A colour read
    with *no* king stays that way, visibly incomplete. Promoting the square
    where a king was most strongly proposed is worth one val board and no test
    board, and on an out-of-domain photograph it turned a misread board into a
    plausible-looking wrong one -- a missing king is the more useful answer.

    Promoted pieces are real, but rare (<0.1% of game positions). One that the
    detector reads without any alternative label is kept, since there is
    nothing to relabel it to.
    """
    import math
    from collections import Counter

    g = [row[:] for row in grid]
    candidates: dict[tuple[int, int], dict[int, float]] = {}
    for (rank, file, class_id), score in scores.items():
        if score >= floor:
            candidates.setdefault((rank, file), {})[class_id] = score

    counts = Counter(value for row in g for value in row if value)
    while True:
        best: tuple[float, int, int, int] | None = None
        for (rank, file), cell in candidates.items():
            current = g[rank][file]
            if not current:
                continue
            here = math.log(max(cell.get(current, floor), 1e-4))
            for class_id, score in cell.items():
                if class_id == current:
                    continue
                gain = (
                    math.log(score)
                    - here
                    + _material_cost(current, counts[current])
                    - _material_cost(current, counts[current] - 1)
                    + _material_cost(class_id, counts[class_id])
                    - _material_cost(class_id, counts[class_id] + 1)
                )
                if gain > 1e-9 and (best is None or gain > best[0]):
                    best = (gain, rank, file, class_id)
        if best is None:
            return g
        _, rank, file, class_id = best
        counts[g[rank][file]] -= 1
        counts[class_id] += 1
        g[rank][file] = class_id


def board_crop(
    corners: list[list[float]], width: int, height: int
) -> tuple[int, int, int]:
    """``(left, top, side)`` of a square crop around the board, inside the image.

    Square because the detector resizes its input to a square: a square crop
    reaches it undistorted. Clamped to the image rather than padded, so every
    pixel the detector sees is a real one.
    """
    points = np.asarray(corners, dtype=np.float64)
    x0, x1 = points[:, 0].min(), points[:, 0].max()
    y0, y1 = points[:, 1].min(), points[:, 1].max()
    w, h = x1 - x0, y1 - y0
    x0, x1 = x0 - CROP_SIDE_MARGIN * w, x1 + CROP_SIDE_MARGIN * w
    y0, y1 = y0 - CROP_TOP_MARGIN * h, y1 + CROP_SIDE_MARGIN * h
    side = min(max(x1 - x0, y1 - y0), width, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    return left, top, int(round(side))


def full_frame_weight(side: int, width: int, height: int) -> float:
    """How much the full-frame pass counts against each board-crop pass.

    The full frame sees the board at ``side / min(width, height)`` of the
    crop's resolution, and its vote is weighted by the square of that. A board
    filling the photograph keeps a near-equal second opinion -- which is worth
    having, because the two views make different mistakes -- while a distant
    board's thumbnail-sized view stops dragging the crop's reading down. Chosen
    on ChessReD val (native and zoomed-out) from {0, r^4, r^2, r, 1}.
    """
    return (side / min(width, height)) ** 2


def _shifted(detections: list[dict], dx: float, dy: float, mirror: float | None):
    out = []
    for detection in detections:
        x0, y0, x1, y1 = detection["box"]
        if mirror is not None:
            x0, x1 = mirror - x1, mirror - x0
        out.append({**detection, "box": [x0 + dx, y0 + dy, x1 + dx, y1 + dy]})
    return out


def read_board(image, corners, detect, threshold: float = POSITION_THRESHOLD) -> dict:
    """Three detector passes, fused, checked against material, then oriented.

    ``detect(image) -> detections`` is the only thing a backend supplies: a PIL
    image in, calibrated detections in that image's pixels out, with no score
    floor. Everything that decides what the board says lives here, once, so the
    torch and ONNX readers cannot drift apart.

    The passes are the full frame, a square crop around the board, and that
    crop mirrored. The crop gives small pieces real pixels: with the board at
    ~37% of a 3000-pixel photograph, a single full-frame pass reads 31 of 306
    test boards exactly and the fused read 251. The mirror is a second look at
    the same pixels -- a reflected board is still a board -- and averaging the
    two cancels errors that depend on exactly where a piece falls in the input.

    Three passes cost three detector runs: about 0.1 s per photograph end to end on
    an RTX 5070 Ti, and 0.69 s on CPU under onnxruntime against 0.33 s for one.
    """
    from PIL import ImageOps

    from chesssight.train.orientation import orient_position

    homography = board_to_image_homography(np.asarray(corners, dtype=np.float64))
    width, height = image.size
    left, top, side = board_crop(corners, width, height)
    crop = image.crop((left, top, left + side, top + side))

    full = detect(image)
    zoomed = _shifted(detect(crop), left, top, None)
    mirrored = _shifted(detect(ImageOps.mirror(crop)), left, top, float(side))

    scores = fuse(
        [
            (square_scores(full, homography), full_frame_weight(side, width, height)),
            (square_scores(zoomed, homography), 1.0),
            (square_scores(mirrored, homography), 1.0),
        ]
    )
    grid = enforce_material(grid_from_scores(scores, threshold), scores)
    grid, quad, evidence = orient_position(grid, list(corners), image)

    return {
        "corners": quad,
        "grid": grid,
        "fen": grid_to_fen(grid),
        # The crop's own boxes: the sharpest view, and one box per piece
        # rather than three near-copies.
        "detections": [d for d in zoomed if float(d["score"]) >= threshold],
        "orientation_evidence": evidence,
        "crop": [left, top, side],
    }
