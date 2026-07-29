"""The mapping between ChessSight class ids and detector label indices.

Kept in one small module with no torch import, because an off-by-one here is
invisible: the model trains, the loss falls, and every prediction is one class out.

ChessSight numbers classes ``1..12`` for pieces with ``0`` meaning an empty square,
plus ``13`` for the board. Detectors want contiguous indices from zero. The offset
between the two is therefore exactly one, and it lives here rather than being
open-coded at each call site.
"""

from __future__ import annotations

from chesssight.data.fen import CLASS_NAMES, NUM_CLASSES
from chesssight.data.masks import BOARD_LABEL

#: Detector label index -> human name. Index 0 is the white pawn, not "empty":
#: an empty square is the absence of a detection, not a class to predict.
DETECTION_LABELS: tuple[str, ...] = (*CLASS_NAMES[1:NUM_CLASSES], "board")

#: Number of classes the detection head predicts.
NUM_DETECTION_LABELS = len(DETECTION_LABELS)

ID2LABEL: dict[int, str] = dict(enumerate(DETECTION_LABELS))
LABEL2ID: dict[str, int] = {name: index for index, name in ID2LABEL.items()}

#: Detector index of the board class.
BOARD_INDEX = LABEL2ID["board"]


def class_id_to_index(class_id: int) -> int:
    """ChessSight class id (1..13) -> detector label index (0..12)."""
    if not 1 <= class_id <= BOARD_LABEL:
        raise ValueError(f"class id {class_id} outside 1..{BOARD_LABEL}")
    return class_id - 1


def index_to_class_id(index: int) -> int:
    """Detector label index (0..12) -> ChessSight class id (1..13)."""
    if not 0 <= index < NUM_DETECTION_LABELS:
        raise ValueError(f"label index {index} outside 0..{NUM_DETECTION_LABELS - 1}")
    return index + 1


def is_piece(index: int) -> bool:
    """Whether a detector index names a piece rather than the board."""
    return index != BOARD_INDEX
