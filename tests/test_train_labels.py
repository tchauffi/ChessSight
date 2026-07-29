"""The class-id mapping is the one place an off-by-one is invisible.

If it is wrong the model still trains and the loss still falls -- every prediction
is simply one class out. These tests are cheap insurance against that, and they
import no torch, so they run in CI without the training extra installed.
"""

from __future__ import annotations

import pytest

from chesssight.data.fen import CLASS_NAMES, LETTER_TO_CLASS, NUM_CLASSES
from chesssight.data.masks import BOARD_LABEL
from chesssight.train.labels import (
    BOARD_INDEX,
    DETECTION_LABELS,
    ID2LABEL,
    LABEL2ID,
    NUM_DETECTION_LABELS,
    class_id_to_index,
    index_to_class_id,
    is_piece,
)


def test_there_are_twelve_pieces_plus_the_board():
    assert NUM_DETECTION_LABELS == 13
    assert len(DETECTION_LABELS) == 13


def test_empty_is_not_a_detection_class():
    # An empty square is the absence of a detection, not something to predict.
    assert "empty" not in DETECTION_LABELS
    assert DETECTION_LABELS[0] == "white_pawn"


def test_the_board_is_the_last_class():
    assert DETECTION_LABELS[-1] == "board"
    assert BOARD_INDEX == NUM_DETECTION_LABELS - 1


@pytest.mark.parametrize("class_id", range(1, NUM_CLASSES))
def test_piece_class_ids_round_trip(class_id: int):
    index = class_id_to_index(class_id)
    assert index_to_class_id(index) == class_id
    # And the name must survive the trip, which is what actually matters.
    assert DETECTION_LABELS[index] == CLASS_NAMES[class_id]


def test_the_board_label_round_trips():
    assert class_id_to_index(BOARD_LABEL) == BOARD_INDEX
    assert index_to_class_id(BOARD_INDEX) == BOARD_LABEL


def test_specific_pieces_land_on_the_right_index():
    # Spelled out, because a shift of one would still satisfy a round-trip test.
    assert DETECTION_LABELS[class_id_to_index(LETTER_TO_CLASS["P"])] == "white_pawn"
    assert DETECTION_LABELS[class_id_to_index(LETTER_TO_CLASS["K"])] == "white_king"
    assert DETECTION_LABELS[class_id_to_index(LETTER_TO_CLASS["q"])] == "black_queen"
    assert DETECTION_LABELS[class_id_to_index(LETTER_TO_CLASS["k"])] == "black_king"


def test_id2label_and_label2id_agree():
    assert len(ID2LABEL) == NUM_DETECTION_LABELS
    assert all(LABEL2ID[name] == index for index, name in ID2LABEL.items())
    assert len(set(ID2LABEL.values())) == NUM_DETECTION_LABELS


def test_is_piece_separates_the_board():
    assert all(is_piece(index) for index in range(NUM_DETECTION_LABELS - 1))
    assert not is_piece(BOARD_INDEX)


@pytest.mark.parametrize("bad", [0, -1, BOARD_LABEL + 1, 99])
def test_out_of_range_class_ids_raise(bad: int):
    with pytest.raises(ValueError):
        class_id_to_index(bad)


@pytest.mark.parametrize("bad", [-1, NUM_DETECTION_LABELS, 99])
def test_out_of_range_indices_raise(bad: int):
    with pytest.raises(ValueError):
        index_to_class_id(bad)
