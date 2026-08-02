from __future__ import annotations

import pytest

from chesssight.data import fen


def test_starting_position_round_trips():
    grid = fen.fen_to_grid(fen.STARTING_FEN)
    assert fen.grid_to_placement(grid) == fen.STARTING_FEN.split()[0]


def test_starting_position_layout():
    grid = fen.fen_to_grid(fen.STARTING_FEN)
    # Row 0 is rank 8, so it holds Black's back rank.
    assert grid[0] == [10, 8, 9, 11, 12, 9, 8, 10]
    assert grid[1] == [7] * 8
    assert grid[6] == [1] * 8
    assert grid[7] == [4, 2, 3, 5, 6, 3, 2, 4]
    assert all(value == fen.EMPTY for row in grid[2:6] for value in row)


def test_empty_board_round_trips():
    grid = fen.empty_grid()
    assert fen.grid_to_placement(grid) == "8/8/8/8/8/8/8/8"
    assert fen.fen_to_grid("8/8/8/8/8/8/8/8") == grid


@pytest.mark.parametrize("letter", list(fen.PIECE_LETTERS))
def test_every_piece_letter_round_trips(letter: str):
    grid = fen.empty_grid()
    grid[3][4] = fen.LETTER_TO_CLASS[letter]
    placement = fen.grid_to_placement(grid)
    assert placement == f"8/8/8/4{letter}3/8/8/8/8"
    assert fen.fen_to_grid(placement) == grid


def test_placement_only_fen_is_accepted():
    assert fen.fen_to_grid("8/8/8/8/8/8/8/8") == fen.empty_grid()


def test_grid_to_fen_includes_supplied_fields():
    grid = fen.fen_to_grid(fen.STARTING_FEN)
    result = fen.grid_to_fen(grid, side_to_move="b", castling="Kq", en_passant="e3")
    assert result.split()[1:] == ["b", "Kq", "e3", "0", "1"]


def test_square_names_match_grid_orientation():
    assert fen.square_name(0, 0) == "a8"
    assert fen.square_name(7, 7) == "h1"
    assert fen.square_name(4, 4) == "e4"
    assert fen.parse_square_name("e4") == (4, 4)
    assert fen.parse_square_name("a8") == (0, 0)


def test_square_index_is_reading_order():
    assert fen.square_index(0, 0) == 0
    assert fen.square_index(7, 7) == 63
    assert fen.square_index(1, 3) == 11


def test_iter_occupied_and_counts():
    grid = fen.fen_to_grid(fen.STARTING_FEN)
    occupied = fen.iter_occupied(grid)
    assert len(occupied) == 32
    counts = fen.piece_counts(grid)
    assert counts[fen.LETTER_TO_CLASS["P"]] == 8
    assert counts[fen.LETTER_TO_CLASS["k"]] == 1


def test_is_white_and_is_black():
    assert fen.is_white(fen.LETTER_TO_CLASS["Q"])
    assert not fen.is_white(fen.EMPTY)
    assert fen.is_black(fen.LETTER_TO_CLASS["q"])
    assert not fen.is_black(fen.LETTER_TO_CLASS["Q"])


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "8/8/8/8/8/8/8",  # seven ranks
        "8/8/8/8/8/8/8/8/8",  # nine ranks
        "9/8/8/8/8/8/8/8",  # rank too wide
        "7/8/8/8/8/8/8/8",  # rank too narrow
        "0/8/8/8/8/8/8/8",  # zero run
        "xppppppp/8/8/8/8/8/8/8",  # bad letter
        "ppppppppp/8/8/8/8/8/8/8",  # overflow
    ],
)
def test_invalid_fens_raise(bad: str):
    with pytest.raises(fen.FenError):
        fen.fen_to_grid(bad)


def test_validate_grid_rejects_bad_shapes_and_values():
    with pytest.raises(fen.FenError):
        fen.validate_grid([[0] * 8] * 7)
    with pytest.raises(fen.FenError):
        fen.validate_grid([[0] * 7] + [[0] * 8] * 7)
    with pytest.raises(fen.FenError):
        fen.validate_grid([[13] + [0] * 7] + [[0] * 8] * 7)
    with pytest.raises(fen.FenError):
        fen.validate_grid([[-1] + [0] * 7] + [[0] * 8] * 7)


def test_class_names_cover_all_classes():
    assert len(fen.CLASS_NAMES) == fen.NUM_CLASSES
    assert fen.CLASS_NAMES[0] == "empty"
    assert fen.CLASS_NAMES[fen.LETTER_TO_CLASS["K"]] == "white_king"
    assert fen.CLASS_NAMES[fen.LETTER_TO_CLASS["k"]] == "black_king"
