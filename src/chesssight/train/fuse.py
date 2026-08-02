"""Combine the two position readers by what each is measurably good at.

Neither model is better than the other. They fail in different places, and the
difference is large and one-sided. Measured on ChessReD's 306 test boards, over
6294 occupied squares:

===================  =======  ======  ===========  ============
model                correct  missed  wrong piece  wrong colour
===================  =======  ======  ===========  ============
detect-then-assign    92.29%   6.66%        1.05%        0.00%
grid classifier       87.04%   0.06%       12.65%        0.25%
===================  =======  ======  ===========  ============

The detector hardly ever misnames a piece it has found -- and never mistakes its
colour -- but it misses one occupied square in fifteen. The grid classifier
essentially never misses a piece but misnames one in eight. Each one's dominant
failure is the other's strongest suit.

So the rule below is not a blend or a vote. It asks each model only the question
it answers well: the grid classifier decides *whether a square is occupied*, and
the detector decides *what the piece is* wherever it found one. Where the
detector found nothing, the grid's own guess stands rather than the square being
emptied -- that is the 6.66% the detector would otherwise lose.
"""

from __future__ import annotations

from chesssight.data.fen import BOARD_SIZE

#: Where the two disagree about a square being empty, whose answer is taken.
#: The grid classifier's, because empty squares outnumber occupied ones 2:1 in
#: ChessReD and the arithmetic still favours it: its 1.17% false-positive rate on
#: 13290 empty squares is 155 errors, against the detector's 6.66% miss rate on
#: 6294 occupied squares, which is 419.
OCCUPANCY_FROM_GRID = True


def fuse(grid: list[list[int]], detector: list[list[int]]) -> list[list[int]]:
    """One position from the two readers' 8x8 grids.

    Both arguments are class ids per square, ``0`` for empty, in the same
    orientation -- rotate them together beforehand if orientation has been
    resolved, or the two will disagree about which square is which and the fusion
    will be worse than either input.
    """
    if len(grid) != BOARD_SIZE or len(detector) != BOARD_SIZE:
        raise ValueError("both grids must be 8x8")

    fused = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            occupied = grid[rank][file]
            named = detector[rank][file]
            if not occupied:
                continue  # the grid classifier is the occupancy authority
            # The detector names it when it found something there; otherwise the
            # grid's own class is kept, which is the whole point -- emptying the
            # square would reintroduce the detector's misses.
            fused[rank][file] = named or occupied
    return fused


def agreement(grid: list[list[int]], detector: list[list[int]]) -> dict[str, int]:
    """How the two readers relate on one board, for diagnosis rather than output.

    Reported because the fusion's value depends entirely on the two disagreeing
    in the expected direction. If ``detector_only`` ever grows large, the grid
    classifier has stopped being the better occupancy detector and the rule above
    needs revisiting rather than trusting.
    """
    counts = {
        "both": 0,
        "grid_only": 0,
        "detector_only": 0,
        "neither": 0,
        "named_differently": 0,
    }
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            occupied = bool(grid[rank][file])
            named = bool(detector[rank][file])
            if occupied and named:
                counts["both"] += 1
                if grid[rank][file] != detector[rank][file]:
                    counts["named_differently"] += 1
            elif occupied:
                counts["grid_only"] += 1
            elif named:
                counts["detector_only"] += 1
            else:
                counts["neither"] += 1
    return counts
