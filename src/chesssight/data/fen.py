"""FEN <-> 8x8 class-grid conversion.

This module is deliberately dependency-free (no ``chess``, no ``numpy``) because the
grid representation is the one thing that crosses every process boundary in the
project: the Blender-side renderer receives a grid, the model predicts a grid, and
real photo annotations are stored as a grid.

Class ids
---------
``0`` is the empty square. Ids ``1..6`` are the white pieces in the order
``P N B R Q K``; ids ``7..12`` are the black pieces in the same order.

Grid orientation
----------------
A grid is ``grid[rank][file]`` with ``rank = 0`` meaning rank 8 (Black's back rank)
and ``file = 0`` meaning the a-file. That is exactly the reading order of a FEN
placement field, which keeps the conversions below trivial and hard to get subtly
backwards.
"""

from __future__ import annotations

from typing import Final, TypeAlias

Grid: TypeAlias = list[list[int]]

BOARD_SIZE: Final = 8
NUM_SQUARES: Final = BOARD_SIZE * BOARD_SIZE
EMPTY: Final = 0
NUM_CLASSES: Final = 13

#: Class id -> FEN piece letter. Index 0 (empty) has no letter.
PIECE_LETTERS: Final = "PNBRQKpnbrqk"

#: FEN piece letter -> class id (1..12).
LETTER_TO_CLASS: Final[dict[str, int]] = {
    letter: index + 1 for index, letter in enumerate(PIECE_LETTERS)
}

#: Class id -> FEN piece letter, with the empty square mapped to ``"."``.
CLASS_TO_LETTER: Final[dict[int, str]] = {EMPTY: "."} | {
    index + 1: letter for index, letter in enumerate(PIECE_LETTERS)
}

#: Human-readable label per class id, useful for detector class names and QA overlays.
CLASS_NAMES: Final[tuple[str, ...]] = (
    "empty",
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
)

STARTING_FEN: Final = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FenError(ValueError):
    """Raised when a FEN string or a grid is structurally invalid."""


def is_white(class_id: int) -> bool:
    """Return whether ``class_id`` is a white piece. Empty squares are not white."""
    return 1 <= class_id <= 6


def is_black(class_id: int) -> bool:
    """Return whether ``class_id`` is a black piece."""
    return 7 <= class_id <= 12


def empty_grid() -> Grid:
    """Return a fresh 8x8 grid of empty squares."""
    return [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]


def square_name(rank_index: int, file_index: int) -> str:
    """Return the algebraic name (e.g. ``"e4"``) of a grid cell.

    ``rank_index`` is the grid row, so row 0 is rank 8.
    """
    if not (0 <= rank_index < BOARD_SIZE and 0 <= file_index < BOARD_SIZE):
        raise FenError(f"square out of range: rank={rank_index} file={file_index}")
    return f"{'abcdefgh'[file_index]}{BOARD_SIZE - rank_index}"


def square_index(rank_index: int, file_index: int) -> int:
    """Return the flat 0..63 index of a grid cell, in grid reading order."""
    if not (0 <= rank_index < BOARD_SIZE and 0 <= file_index < BOARD_SIZE):
        raise FenError(f"square out of range: rank={rank_index} file={file_index}")
    return rank_index * BOARD_SIZE + file_index


def parse_square_name(name: str) -> tuple[int, int]:
    """Return ``(rank_index, file_index)`` for an algebraic square name."""
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        raise FenError(f"invalid square name: {name!r}")
    return BOARD_SIZE - int(name[1]), "abcdefgh".index(name[0])


def validate_grid(grid: Grid) -> None:
    """Raise :class:`FenError` unless ``grid`` is a well-formed 8x8 class grid."""
    if len(grid) != BOARD_SIZE:
        raise FenError(f"grid must have {BOARD_SIZE} ranks, got {len(grid)}")
    for rank_index, row in enumerate(grid):
        if len(row) != BOARD_SIZE:
            raise FenError(
                f"rank {rank_index} must have {BOARD_SIZE} files, got {len(row)}"
            )
        for class_id in row:
            if not isinstance(class_id, int) or isinstance(class_id, bool):
                raise FenError(f"grid values must be ints, got {class_id!r}")
            if not 0 <= class_id < NUM_CLASSES:
                raise FenError(f"class id out of range 0..12: {class_id}")


def fen_to_grid(fen: str) -> Grid:
    """Convert a FEN (full or placement-only) into an 8x8 class grid."""
    if not fen or not fen.strip():
        raise FenError("empty FEN")
    placement = fen.strip().split()[0]

    ranks = placement.split("/")
    if len(ranks) != BOARD_SIZE:
        raise FenError(f"FEN placement must have {BOARD_SIZE} ranks, got {len(ranks)}")

    grid = empty_grid()
    for rank_index, rank_field in enumerate(ranks):
        file_index = 0
        for char in rank_field:
            if char.isdigit():
                skip = int(char)
                if skip == 0:
                    raise FenError(f"invalid empty-run '0' in rank {rank_index}")
                file_index += skip
            else:
                class_id = LETTER_TO_CLASS.get(char)
                if class_id is None:
                    raise FenError(f"invalid piece letter {char!r} in FEN {fen!r}")
                if file_index >= BOARD_SIZE:
                    raise FenError(f"rank {rank_index} overflows the board in {fen!r}")
                grid[rank_index][file_index] = class_id
                file_index += 1
        if file_index != BOARD_SIZE:
            raise FenError(
                f"rank {rank_index} describes {file_index} files, expected {BOARD_SIZE}"
            )
    return grid


def grid_to_fen(
    grid: Grid,
    *,
    side_to_move: str = "w",
    castling: str = "-",
    en_passant: str = "-",
    halfmove_clock: int = 0,
    fullmove_number: int = 1,
) -> str:
    """Convert an 8x8 class grid into a full FEN string.

    Only the placement field carries information from the grid; the remaining fields
    are supplied by the caller because a rendered image cannot express them.
    """
    validate_grid(grid)
    if side_to_move not in ("w", "b"):
        raise FenError(f"side_to_move must be 'w' or 'b', got {side_to_move!r}")

    ranks: list[str] = []
    for row in grid:
        field = ""
        run = 0
        for class_id in row:
            if class_id == EMPTY:
                run += 1
                continue
            if run:
                field += str(run)
                run = 0
            field += CLASS_TO_LETTER[class_id]
        if run:
            field += str(run)
        ranks.append(field)

    placement = "/".join(ranks)
    return (
        f"{placement} {side_to_move} {castling} {en_passant} "
        f"{halfmove_clock} {fullmove_number}"
    )


def grid_to_placement(grid: Grid) -> str:
    """Return only the placement field of the FEN for ``grid``."""
    return grid_to_fen(grid).split()[0]


def iter_occupied(grid: Grid) -> list[tuple[int, int, int]]:
    """Return ``(rank_index, file_index, class_id)`` for every occupied square."""
    validate_grid(grid)
    return [
        (rank_index, file_index, class_id)
        for rank_index, row in enumerate(grid)
        for file_index, class_id in enumerate(row)
        if class_id != EMPTY
    ]


def piece_counts(grid: Grid) -> dict[int, int]:
    """Return a ``{class_id: count}`` map over the occupied squares of ``grid``."""
    counts: dict[int, int] = {}
    for _, _, class_id in iter_occupied(grid):
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts
