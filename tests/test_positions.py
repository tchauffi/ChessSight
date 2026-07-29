from __future__ import annotations

import random
from pathlib import Path

import pytest

from chesssight.data.fen import (
    BOARD_SIZE,
    LETTER_TO_CLASS,
    STARTING_FEN,
    fen_to_grid,
    grid_to_placement,
    piece_counts,
    validate_grid,
)
from chesssight.synth import positions as pos

FIXTURE_PGN = Path(__file__).parent / "fixtures" / "sample.pgn"

WHITE_KING = LETTER_TO_CLASS["K"]
BLACK_KING = LETTER_TO_CLASS["k"]
WHITE_PAWN = LETTER_TO_CLASS["P"]
BLACK_PAWN = LETTER_TO_CLASS["p"]


def test_starting_sampler_returns_the_start_position():
    grid = pos.StartingPositionSampler().sample(random.Random(0))
    assert grid_to_placement(grid) == STARTING_FEN.split()[0]


class TestRandomPositionSampler:
    def test_grids_are_structurally_valid(self):
        sampler = pos.RandomPositionSampler()
        for seed in range(50):
            grid = sampler.sample(random.Random(seed))
            validate_grid(grid)

    def test_exactly_one_king_per_side(self):
        sampler = pos.RandomPositionSampler()
        for seed in range(50):
            counts = piece_counts(sampler.sample(random.Random(seed)))
            assert counts.get(WHITE_KING) == 1
            assert counts.get(BLACK_KING) == 1

    def test_kings_are_never_adjacent(self):
        sampler = pos.RandomPositionSampler()
        for seed in range(50):
            grid = sampler.sample(random.Random(seed))
            squares = {
                value: (rank, file)
                for rank, row in enumerate(grid)
                for file, value in enumerate(row)
                if value in (WHITE_KING, BLACK_KING)
            }
            white, black = squares[WHITE_KING], squares[BLACK_KING]
            chebyshev = max(abs(white[0] - black[0]), abs(white[1] - black[1]))
            assert chebyshev > 1

    def test_pawns_stay_off_the_back_ranks(self):
        sampler = pos.RandomPositionSampler()
        for seed in range(50):
            grid = sampler.sample(random.Random(seed))
            for rank in (0, BOARD_SIZE - 1):
                assert WHITE_PAWN not in grid[rank]
                assert BLACK_PAWN not in grid[rank]

    def test_piece_count_respects_the_configured_bounds(self):
        sampler = pos.RandomPositionSampler(min_pieces=4, max_pieces=10)
        for seed in range(50):
            occupied = pos.count_occupied(sampler.sample(random.Random(seed)))
            assert 2 <= occupied <= 10

    def test_same_seed_gives_the_same_grid(self):
        sampler = pos.RandomPositionSampler()
        assert sampler.sample(random.Random(7)) == sampler.sample(random.Random(7))

    def test_different_seeds_generally_differ(self):
        sampler = pos.RandomPositionSampler()
        grids = {grid_to_placement(sampler.sample(random.Random(s))) for s in range(20)}
        assert len(grids) > 15

    def test_invalid_bounds_raise(self):
        with pytest.raises(ValueError):
            pos.RandomPositionSampler(min_pieces=10, max_pieces=4)
        with pytest.raises(ValueError):
            pos.RandomPositionSampler(min_pieces=1)


class TestPgnPositionSampler:
    def test_reads_positions_from_the_fixture(self):
        sampler = pos.PgnPositionSampler([FIXTURE_PGN], plies_per_game=3)
        assert len(sampler) > 0
        for _ in range(20):
            validate_grid(sampler.sample(random.Random(0)))

    def test_positions_have_both_kings(self):
        sampler = pos.PgnPositionSampler([FIXTURE_PGN], plies_per_game=5)
        for placement in sampler.positions:
            counts = piece_counts(fen_to_grid(placement))
            assert counts.get(WHITE_KING) == 1
            assert counts.get(BLACK_KING) == 1

    def test_skips_opening_plies(self):
        # Game 3 is only 4 plies long, so a skip of 6 must exclude it entirely.
        sampler = pos.PgnPositionSampler([FIXTURE_PGN], skip_opening_plies=6)
        start_placement = STARTING_FEN.split()[0]
        assert start_placement not in sampler.positions

    def test_loading_is_deterministic(self):
        first = pos.PgnPositionSampler([FIXTURE_PGN], seed=3).positions
        second = pos.PgnPositionSampler([FIXTURE_PGN], seed=3).positions
        assert first == second

    def test_empty_source_raises(self, tmp_path: Path):
        empty = tmp_path / "empty.pgn"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            pos.PgnPositionSampler([empty])

    def test_reads_zstandard_compressed_pgn(self, tmp_path: Path):
        import zstandard

        compressed = tmp_path / "games.pgn.zst"
        raw = FIXTURE_PGN.read_bytes()
        compressed.write_bytes(zstandard.ZstdCompressor().compress(raw))

        plain = pos.PgnPositionSampler([FIXTURE_PGN], seed=1).positions
        packed = pos.PgnPositionSampler([compressed], seed=1).positions
        assert packed == plain


class TestMixtureSampler:
    def test_draws_from_both_components(self):
        starting = pos.StartingPositionSampler()
        scattered = pos.RandomPositionSampler()
        mixture = pos.MixtureSampler([(starting, 1.0), (scattered, 1.0)])

        placements = {
            grid_to_placement(mixture.sample(random.Random(seed))) for seed in range(40)
        }
        assert STARTING_FEN.split()[0] in placements
        assert len(placements) > 2

    def test_rejects_empty_or_non_positive_weights(self):
        with pytest.raises(ValueError):
            pos.MixtureSampler([])
        with pytest.raises(ValueError):
            pos.MixtureSampler([(pos.StartingPositionSampler(), 0.0)])


def test_iter_positions_is_reproducible_per_index():
    sampler = pos.RandomPositionSampler()
    first = dict(pos.iter_positions(sampler, 10, seed=99))
    second = dict(pos.iter_positions(sampler, 10, seed=99))
    assert first == second

    # Sample i must not depend on having generated 0..i-1 first.
    only_later = dict(list(pos.iter_positions(sampler, 10, seed=99))[5:])
    assert only_later[7] == first[7]
