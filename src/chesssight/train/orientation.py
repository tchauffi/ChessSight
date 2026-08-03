"""Which corner is a8.

Four interchangeable corners fix the board's geometry up to a rotation, which is
why every position number in this project has so far been quoted as a best-of-four
upper bound. Resolving that rotation is what turns a read grid into an actual
position, and it needs no fifth model -- the answer is already in the image, in
two independent pieces:

**Square colour resolves four down to two.** a1 is dark, and the light/dark
pattern is fixed relative to the board's own axes: the square at ``grid[0][0]``
(a8) is light, and every square is light exactly when ``(rank + file)`` is even.
Rotating the board by 90 degrees flips that parity, so measuring which parity is
brighter eliminates two of the four candidates using nothing but pixels. This
works on an empty board, in any position, and does not care about the pieces.

**Piece colour resolves two down to one.** The surviving pair differ by 180
degrees, and the only thing that distinguishes them is that White's men start at
the bottom of the board. Counting white pieces in the near half against the far
half settles it.

The split matters because the two halves fail differently. Colour parity is
robust and almost always available. The piece vote needs pieces, and a position
symmetric in material -- an empty board, or the opening seen from the side --
leaves a genuine 180-degree ambiguity that no amount of looking will resolve. The
functions below report that rather than picking one and sounding certain.

**Pawns break the 180-degree tie better than material does.** The material
vote assumes White's men sit nearer the camera, and on ChessReD's val split
that assumption is wrong for whole games at a time -- 45 of 330 boards read
flipped, four fifths of the split's wrong squares. Pawns carry a signal that
does not depend on where the players sat: a pawn stands on its own half of the
board for most of its life, and a 180-degree flip reads every pawn as deep in
enemy territory. Counting own-half pawns (:func:`pawn_home_score`) fixes 25 of
those 45 boards and regresses none; the term abstains on pawnless and
runner-endgame positions rather than guessing.
"""

from __future__ import annotations

import numpy as np

from chesssight.data.fen import BOARD_SIZE, LETTER_TO_CLASS, is_black, is_white

_WHITE_PAWN = LETTER_TO_CLASS["P"]
_BLACK_PAWN = LETTER_TO_CLASS["p"]

#: Fraction of a square sampled when measuring its colour. Well inside the edge:
#: a square's border is a high-contrast line, and including it would measure the
#: grid rather than the square.
SAMPLE_FRACTION = 0.5

#: Below this the colour evidence is too weak to trust -- a board photographed in
#: heavy shadow, or one whose squares are nearly the same tone. Expressed as a
#: fraction of the image's own luminance range so it does not assume 8-bit.
MIN_COLOUR_MARGIN = 0.02

#: How strongly the pawn vote counts against the material vote in the
#: 180-degree tie-break. Both terms are normalised to [-1, 1], so 2.0 lets a
#: unanimous pawn vote overrule a unanimous material vote while a split pawn
#: vote still defers. Swept on ChessReD val, where 218/330 boards read exactly
#: right against 198 without the term; the result is flat from 2.0 to 4.0, so
#: the value is a plateau, not a fit.
PAWN_HOME_WEIGHT = 2.0


def square_luminance(image, homography) -> np.ndarray:
    """Mean luminance of each of the 64 squares, as an 8x8 array.

    Sampled through the homography rather than by cropping an axis-aligned box:
    under perspective a distant square is a small trapezium, and a box around it
    is mostly its neighbours.
    """
    from chesssight.data.geometry import apply_homography

    grey = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    height, width = grey.shape

    offsets = np.linspace(0.5 - SAMPLE_FRACTION / 2, 0.5 + SAMPLE_FRACTION / 2, 3)
    points = []
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            for dv in offsets:
                for du in offsets:
                    points.append([file + du, rank + dv])

    pixels = apply_homography(homography, np.asarray(points, dtype=np.float64))
    xs = np.clip(np.round(pixels[:, 0]).astype(int), 0, width - 1)
    ys = np.clip(np.round(pixels[:, 1]).astype(int), 0, height - 1)
    samples = grey[ys, xs].reshape(BOARD_SIZE, BOARD_SIZE, -1)
    return samples.mean(axis=2)


def colour_score(luminance: np.ndarray) -> float:
    """How much brighter the even-parity squares are than the odd ones.

    Positive means ``(rank + file)`` even is the light colour, which is the
    board's own convention with a8 at ``grid[0][0]``. The magnitude is the
    evidence: near zero means the colours could not be told apart.
    """
    ranks, files = np.indices((BOARD_SIZE, BOARD_SIZE))
    even = (ranks + files) % 2 == 0
    return float(luminance[even].mean() - luminance[~even].mean())


def piece_score(grid: list[list[int]]) -> float:
    """How much more white material sits in the near half than the far half.

    Near is the bottom of the grid -- ranks 1 and 2 are ``grid[6]`` and
    ``grid[7]`` -- because ``grid[0]`` is rank 8. Black is counted with the
    opposite sign so that a board with only black pieces still votes, and the
    result is normalised so it is comparable between a full board and an ending.
    """
    near = far = 0
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            occupant = grid[rank][file]
            if not occupant:
                continue
            bottom = rank >= BOARD_SIZE // 2
            if is_white(occupant):
                near += bottom
                far += not bottom
            elif is_black(occupant):
                near += not bottom
                far += bottom
    total = near + far
    return (near - far) / total if total else 0.0


def pawn_home_score(grid: list[list[int]]) -> float:
    """How many pawns stand on their own half of the board, net, in [-1, 1].

    White pawns belong on ranks 2-4 and black pawns on ranks 5-7 for most of a
    game, and unlike the material split this does not assume anything about
    where the players sat: a 180-degree flip reads every pawn as deep in enemy
    territory, so the correct orientation scores higher whenever the pawn
    structure is at all intact. Returns 0 -- no vote -- unless both colours
    still have a pawn: a lone runner is exactly the pawn whose position lies
    about the orientation, and the endgames where that happens are the ones
    the material vote still reads correctly.
    """
    own = total = 0
    half = BOARD_SIZE // 2
    seen_white = seen_black = False
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            occupant = grid[rank][file]
            if occupant == _WHITE_PAWN:
                seen_white = True
                total += 1
                own += rank >= half
            elif occupant == _BLACK_PAWN:
                seen_black = True
                total += 1
                own += rank < half
    if not (seen_white and seen_black):
        return 0.0
    return (2 * own - total) / total


def rotate(array: list[list[int]] | np.ndarray, turns: int) -> np.ndarray:
    """One quarter-turn convention, shared by the grid and the luminance map.

    Both must be rotated by the *same* call. When they were rotated by separate
    expressions the two drifted by a sign and the colour test silently voted for
    the board's mirror image.
    """
    return np.rot90(np.asarray(array), turns)


def orient(
    grid: list[list[int]], luminance: np.ndarray
) -> tuple[int, dict[str, float]]:
    """The quarter-turns that put a8 at ``grid[0][0]``, and the evidence for it.

    Returns ``(turns, evidence)``. Apply ``turns`` with :func:`rotate` to both the
    grid and the corner list to get the oriented board.

    ``evidence`` carries ``colour``, ``pieces`` and ``pawns``: the raw value of
    each signal behind the decision. A caller that needs to know whether to
    trust the answer should look at those rather than at the fact that a number
    came back -- an empty board still returns a rotation, it just has no piece
    evidence behind it.
    """
    candidates = []
    for turns in range(4):
        rotated = rotate(grid, turns).tolist()
        colour = colour_score(rotate(luminance, turns))
        pieces = piece_score(rotated)
        pawns = pawn_home_score(rotated)
        score = pieces + PAWN_HOME_WEIGHT * pawns
        candidates.append((turns, colour, score, pieces, pawns))

    # Colour first, and as a filter rather than a term in a sum: it is the more
    # reliable of the two, and letting a confident piece vote outweigh it would
    # allow an answer that puts a dark square on a8 -- which is not a board.
    best_colour = max(colour for _, colour, _, _, _ in candidates)
    surviving = [
        candidate
        for candidate in candidates
        if candidate[1] >= best_colour - MIN_COLOUR_MARGIN
    ]
    chosen = max(surviving, key=lambda candidate: candidate[2])
    turns, colour, score, pieces, pawns = chosen

    runner_up = max(
        (candidate[2] for candidate in surviving if candidate[0] != turns),
        default=None,
    )
    return turns, {
        "colour": colour,
        "pieces": pieces,
        "pawns": pawns,
        # How much better the winner's tie-break score was than the best
        # alternative that survived the colour filter. Zero means a real
        # 180-degree tie.
        "margin": 0.0 if runner_up is None else score - runner_up,
        "candidates": float(len(surviving)),
    }


def orient_position(
    grid: list[list[int]], corners: list[list[float]], image
) -> tuple[list[list[int]], list[list[float]], dict[str, float]]:
    """Rotate a read position and its corners into board orientation.

    The corners are rotated alongside so that the returned quad still starts at
    a8 -- a caller that keeps the grid and the old corner order would have a
    position and a homography that disagree about which way up the board is.
    """
    from chesssight.data.geometry import board_to_image_homography

    homography = board_to_image_homography(np.asarray(corners, dtype=np.float64))
    luminance = square_luminance(image, homography)
    turns, evidence = orient(grid, luminance)

    # Rotating the grid by k quarter-turns moves what was corner k to the front.
    ordered = corners[turns:] + corners[:turns]
    return rotate(grid, turns).tolist(), ordered, evidence
