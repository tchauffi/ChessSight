"""Sources of board positions to render.

A sampler's only job is to produce an 8x8 class grid. Everything downstream -- the
job spec, the Blender scene, the label record -- consumes grids, so the Blender side
never needs to know what chess is.

Two samplers are provided and can be mixed by weight:

``PgnPositionSampler``
    Positions lifted from real games. These have realistic piece-count
    distributions and realistic structures, which is what the model will actually
    see in photographs.

``RandomPositionSampler``
    Pieces scattered without regard for legality. Real games are dominated by
    near-starting positions, so a pure-PGN dataset under-samples sparse boards and
    lets the model lean on chess priors instead of pixels. Mixing in a fraction of
    these broadens coverage.
"""

from __future__ import annotations

import io
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Final, TextIO

import chess
import chess.pgn

from chesssight.data.fen import (
    BOARD_SIZE,
    EMPTY,
    LETTER_TO_CLASS,
    STARTING_FEN,
    Grid,
    empty_grid,
    fen_to_grid,
)
from chesssight.synth.seeds import derive_rng

#: python-chess piece symbol -> our class id is handled through FEN, so we only need
#: the maximum plausible count of each piece type for the random sampler.
MAX_PIECES_PER_SIDE: Final[dict[str, int]] = {
    "P": 8,
    "N": 4,
    "B": 4,
    "R": 4,
    "Q": 3,
    "K": 1,
}


class PositionSampler(ABC):
    """Produces board positions as 8x8 class grids."""

    @abstractmethod
    def sample(self, rng: random.Random) -> Grid:
        """Return one position. Must be deterministic given ``rng``."""


class StartingPositionSampler(PositionSampler):
    """Always returns the standard starting position. Useful for smoke tests."""

    def sample(self, rng: random.Random) -> Grid:
        return fen_to_grid(STARTING_FEN)


class RandomPositionSampler(PositionSampler):
    """Scatter a plausible number of pieces over random squares.

    Piece *counts* stay in a legal-ish range and the two kings are always present
    and never adjacent, so boards look like chess even though the position may be
    unreachable. Pawns are kept off the first and eighth ranks, which is the one
    illegality that would look obviously wrong in a photo.
    """

    def __init__(self, min_pieces: int = 2, max_pieces: int = 32) -> None:
        if not 2 <= min_pieces <= max_pieces <= 64:
            raise ValueError("require 2 <= min_pieces <= max_pieces <= 64")
        self.min_pieces = min_pieces
        self.max_pieces = max_pieces

    def sample(self, rng: random.Random) -> Grid:
        grid = empty_grid()
        free = [
            (rank, file) for rank in range(BOARD_SIZE) for file in range(BOARD_SIZE)
        ]
        rng.shuffle(free)

        def place(
            letter: str, allowed: list[tuple[int, int]]
        ) -> tuple[int, int] | None:
            for candidate in allowed:
                if candidate in free:
                    free.remove(candidate)
                    grid[candidate[0]][candidate[1]] = LETTER_TO_CLASS[letter]
                    return candidate
            return None

        # Kings first: they are mandatory and constrain each other.
        white_king = place("K", free[:])
        if white_king is None:
            raise RuntimeError("could not place the white king on an empty board")
        non_adjacent = [
            square
            for square in free
            if max(abs(square[0] - white_king[0]), abs(square[1] - white_king[1])) > 1
        ]
        if place("k", non_adjacent) is None:
            raise RuntimeError("could not place the black king")

        target = rng.randint(self.min_pieces, self.max_pieces)
        remaining = max(0, target - 2)

        letters = [
            letter
            for letter, limit in MAX_PIECES_PER_SIDE.items()
            if letter != "K"
            for _ in range(limit)
        ]
        candidates = [letter.upper() for letter in letters] + [
            letter.lower() for letter in letters
        ]
        rng.shuffle(candidates)

        for letter in candidates[:remaining]:
            if letter in ("P", "p"):
                allowed = [square for square in free if 0 < square[0] < BOARD_SIZE - 1]
            else:
                allowed = free[:]
            place(letter, allowed)

        return grid


class PgnPositionSampler(PositionSampler):
    """Positions drawn from games in one or more PGN files.

    Games are read once at construction time and their positions cached as FEN
    placement strings, so sampling itself is cheap and fully deterministic given the
    RNG. ``.pgn`` and zstandard-compressed ``.pgn.zst`` (the format Lichess
    publishes) are both accepted.

    ``plies_per_game`` positions are taken from each game at random plies, skipping
    the first ``skip_opening_plies`` so the dataset is not dominated by the
    starting position. ``max_plies`` bounds the window from above -- pass a small
    one to draw only openings, where the back ranks are still crowded.
    """

    def __init__(
        self,
        paths: list[Path],
        *,
        max_games: int = 5_000,
        plies_per_game: int = 3,
        skip_opening_plies: int = 6,
        max_plies: int | None = None,
        seed: int = 0,
    ) -> None:
        self.positions: list[str] = []
        loader_rng = random.Random(seed)
        for path in paths:
            self._load(
                path,
                max_games=max_games,
                plies_per_game=plies_per_game,
                skip_opening_plies=skip_opening_plies,
                max_plies=max_plies,
                rng=loader_rng,
            )
        if not self.positions:
            raise ValueError(f"no positions could be read from {paths}")

    @staticmethod
    def _open_text(path: Path) -> TextIO:
        if path.suffix == ".zst":
            import zstandard

            decompressor = zstandard.ZstdDecompressor()
            stream = decompressor.stream_reader(path.open("rb"))
            return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
        return path.open("r", encoding="utf-8", errors="replace")

    def _load(
        self,
        path: Path,
        *,
        max_games: int,
        plies_per_game: int,
        skip_opening_plies: int,
        max_plies: int | None,
        rng: random.Random,
    ) -> None:
        with self._open_text(path) as handle:
            for _ in range(max_games):
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                board = game.board()
                fens = []
                for ply, move in enumerate(game.mainline_moves()):
                    if max_plies is not None and ply >= max_plies:
                        break
                    board.push(move)
                    if ply >= skip_opening_plies:
                        fens.append(board.board_fen())
                if not fens:
                    continue
                take = min(plies_per_game, len(fens))
                self.positions.extend(rng.sample(fens, take))

    def __len__(self) -> int:
        return len(self.positions)

    def sample(self, rng: random.Random) -> Grid:
        return fen_to_grid(rng.choice(self.positions))


class MixtureSampler(PositionSampler):
    """Choose between samplers by weight on each draw."""

    def __init__(self, samplers: list[tuple[PositionSampler, float]]) -> None:
        if not samplers:
            raise ValueError("MixtureSampler needs at least one sampler")
        if any(weight <= 0 for _, weight in samplers):
            raise ValueError("all mixture weights must be positive")
        self.samplers = [sampler for sampler, _ in samplers]
        self.weights = [weight for _, weight in samplers]

    def sample(self, rng: random.Random) -> Grid:
        (chosen,) = rng.choices(self.samplers, weights=self.weights, k=1)
        return chosen.sample(rng)


def iter_positions(
    sampler: PositionSampler, count: int, seed: int
) -> Iterator[tuple[int, Grid]]:
    """Yield ``(index, grid)`` pairs, each drawn from its own derived RNG.

    Deriving a fresh RNG per index -- rather than consuming one stream -- means
    sample *i* is reproducible without generating samples 0..i-1 first, which is
    what makes re-rendering a single bad sample cheap.
    """
    for index in range(count):
        yield index, sampler.sample(derive_rng(seed, "position", index))


def count_occupied(grid: Grid) -> int:
    """Number of occupied squares, a cheap sanity metric for sampled positions."""
    return sum(1 for row in grid for value in row if value != EMPTY)
