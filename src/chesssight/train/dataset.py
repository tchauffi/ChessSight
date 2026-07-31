"""A torch Dataset over a generated ChessSight run.

Reads the sample records directly rather than going through a COCO file. The
records already hold everything a detector needs, and skipping the intermediate
export keeps one fewer copy of the labels on disk and removes a chance for the two
to drift apart.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from chesssight.data.dataset import DatasetReader
from chesssight.data.export import board_bbox
from chesssight.data.schema import Sample
from chesssight.train.labels import BOARD_INDEX, CORNER_INDEX, class_id_to_index

#: Boxes smaller than this are dropped. A two-pixel sliver of a piece behind a
#: queen is a real observation but an unlearnable target, and RT-DETR's matcher
#: will happily waste a query on it.
MIN_BOX_SIDE_PX = 2.0


@dataclass(frozen=True)
class SplitSpec:
    """How a run is divided into train, validation and test.

    Three ways rather than two because ``val`` does double duty during a run: it
    selects the checkpoint and it calibrates the scores. A number read off the same
    samples that chose the weights is not an estimate of anything, so ``test`` is
    held out and never looked at until the run is over.
    """

    val_fraction: float = 0.1
    test_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.val_fraction < 0 or self.test_fraction < 0:
            raise ValueError("split fractions must not be negative")
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError(
                f"val {self.val_fraction} + test {self.test_fraction} leaves no "
                "training data"
            )

    def position(self, sample_id: str) -> float:
        """Where an id falls in [0, 1).

        Hashing the id rather than slicing the index means the split survives the
        dataset being regenerated at a different size, and cannot accidentally put
        a whole correlated block of seeds on one side.
        """
        digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big") / 2**32

    def assign(self, sample_id: str) -> str:
        """Which of train/val/test an id belongs to.

        Validation is carved off the bottom of the range and test immediately above
        it, so adding a test split leaves the *existing* validation set exactly as
        it was -- numbers measured before this change stay comparable, and the test
        set comes out of what used to be training data.
        """
        where = self.position(sample_id)
        if where < self.val_fraction:
            return "val"
        if where < self.val_fraction + self.test_fraction:
            return "test"
        return "train"

    def is_val(self, sample_id: str) -> bool:
        return self.assign(sample_id) == "val"


def select_entries(
    all_entries: list,
    *,
    split: str,
    spec: SplitSpec | None = None,
    split_source: str = "auto",
) -> tuple[list, str]:
    """Pick the entries belonging to ``split``, and say how the split was decided.

    A dataset that carries its own division is respected. ChessReD splits by *game*
    -- images from one game share a board, a room and a camera -- so re-splitting it
    by hash would leak all three across the boundary and flatter every number
    measured against it. A synthetic run has no such structure and stores one split
    for everything, so it gets hashed into three.

    Shared by the loaders and the mAP evaluator on purpose. When only the loaders
    knew this rule, evaluating a single-split dataset on ``val`` matched no stored
    entry at all and quietly reported ``map=-1`` every epoch, which in turn made
    ``best`` whichever epoch happened to run first.
    """
    spec = spec or SplitSpec()
    stored = {entry.split for entry in all_entries}
    if split_source == "auto":
        split_source = "stored" if len(stored) > 1 else "hash"

    if split_source == "stored":
        entries = [
            entry for entry in all_entries if split == "all" or entry.split == split
        ]
    else:
        if split == "test" and spec.test_fraction == 0.0:
            raise ValueError(
                "this dataset stores a single split and test_fraction is 0, so "
                "there is no test set to hash out; use 'val', raise "
                "test_fraction, or pass split_source='stored'"
            )
        entries = [
            entry
            for entry in all_entries
            if split == "all" or spec.assign(entry.id) == split
        ]
    return entries, split_source


#: Side of a corner's box, as a fraction of the board's bounding box. One eighth is
#: about one square, which puts corner boxes at the same scale as the pieces the
#: detector already localises well -- a keypoint-sized box would land straight back in
#: the small-object regime that is this model's weakest.
CORNER_BOX_FRACTION = 1.0 / 8.0


def corner_annotations(sample: Sample) -> list[dict]:
    """Boxes centred on the four board corners.

    A corner is a *point*, but the detector speaks boxes, so each one is carried as a
    small square centred on it and read back from the box centre. This is what lets
    corner prediction ride along in the same model and the same forward pass as
    detection, with no architecture change and no custom loading code.

    Corners outside the frame are dropped rather than clipped: a clipped box has its
    centre somewhere other than the corner, which would train the model to place the
    point wrongly on exactly the crops where geometry is hardest.
    """
    board = sample.board
    if board is None or not board.corners_px:
        return []

    box = board_bbox(sample)
    if box is None:
        return []
    side = max(box.width, box.height) * CORNER_BOX_FRACTION
    if side < MIN_BOX_SIDE_PX:
        return []

    records = []
    for point in board.corners_px:
        x, y = float(point[0]), float(point[1])
        # The sample's own dimensions, not the camera's: a real photograph has
        # corners and a size but no camera block at all.
        if not (0 <= x <= sample.width and 0 <= y <= sample.height):
            continue
        records.append(
            {
                "bbox": [x - side / 2.0, y - side / 2.0, side, side],
                "category_id": CORNER_INDEX,
                "area": float(side * side),
                "iscrowd": 0,
            }
        )
    return records


def _annotations(
    sample: Sample,
    *,
    include_board: bool,
    include_off_board: bool = True,
    include_corners: bool = False,
) -> list[dict]:
    """COCO-style annotations for one sample, in detector label indices."""
    records: list[dict] = []

    for piece in sample.pieces:
        if not piece.visible or piece.bbox is None:
            continue
        if not include_off_board and not piece.on_board:
            continue
        if piece.bbox.width < MIN_BOX_SIDE_PX or piece.bbox.height < MIN_BOX_SIDE_PX:
            continue
        # The modal box, measured from the mask. Amodal boxes would teach the
        # detector to predict pieces it cannot see.
        records.append(
            {
                "bbox": [
                    piece.bbox.x,
                    piece.bbox.y,
                    piece.bbox.width,
                    piece.bbox.height,
                ],
                "category_id": class_id_to_index(piece.class_id),
                "area": float(
                    piece.visible_pixels or piece.bbox.width * piece.bbox.height
                ),
                "iscrowd": 0,
            }
        )

    if include_board:
        box = board_bbox(sample)
        if box is not None:
            records.append(
                {
                    "bbox": [box.x, box.y, box.width, box.height],
                    "category_id": BOARD_INDEX,
                    "area": float(box.width * box.height),
                    "iscrowd": 0,
                }
            )
    if include_corners:
        records.extend(corner_annotations(sample))
    return records


class ChessDetectionDataset(Dataset):
    """Images plus detection targets from a ChessSight run."""

    def __init__(
        self,
        root: Path,
        processor,
        *,
        split: str = "train",
        split_spec: SplitSpec | None = None,
        include_board: bool = True,
        include_off_board: bool = True,
        include_corners: bool = False,
        limit: int | None = None,
        split_source: str = "auto",
        transform=None,
    ) -> None:
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"split must be train, val, test or all; got {split!r}")
        if split_source not in ("auto", "stored", "hash"):
            raise ValueError("split_source must be auto, stored or hash")

        self.root = Path(root)
        self.processor = processor
        self.include_board = include_board
        self.include_off_board = include_off_board
        self.include_corners = include_corners
        self.split = split
        # Augmentation applies to training only; a validation set that changes
        # every epoch measures nothing.
        self.transform = transform

        reader = DatasetReader(self.root)
        spec = split_spec or SplitSpec()
        all_entries = reader.entries()

        entries, self.split_source = select_entries(
            all_entries, split=split, spec=spec, split_source=split_source
        )
        if limit is not None:
            entries = entries[:limit]
        if not entries:
            raise ValueError(f"no samples in split {split!r} under {self.root}")

        self.ids = [entry.id for entry in entries]
        self._reader = reader

    def __len__(self) -> int:
        return len(self.ids)

    def sample(self, index: int) -> Sample:
        return self._reader.load(self.ids[index])

    def __getitem__(self, index: int):
        sample = self.sample(index)
        image = Image.open(self.root / sample.image).convert("RGB")
        records = _annotations(
            sample,
            include_board=self.include_board,
            include_off_board=self.include_off_board,
            include_corners=self.include_corners,
        )

        if self.transform is not None and records:
            from chesssight.train.augment import apply as apply_augmentation

            image, boxes, labels = apply_augmentation(
                self.transform,
                image,
                [record["bbox"] for record in records],
                [record["category_id"] for record in records],
            )
            records = [
                {
                    "bbox": box,
                    "category_id": label,
                    "area": float(box[2] * box[3]),
                    "iscrowd": 0,
                }
                for box, label in zip(boxes, labels, strict=True)
            ]

        target = {"image_id": index, "annotations": records}
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }


def collate(batch: Sequence[dict]) -> dict:
    """Stack a batch, keeping per-image label dicts as a list.

    Detection targets vary in length per image, so they cannot be stacked; the
    model expects a list of dicts alongside a batched image tensor.
    """
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
    }


def describe_split(root: Path, spec: SplitSpec | None = None) -> dict[str, int]:
    """Counts per split, for reporting before a run starts.

    Reports the *hash* split regardless of what the index stores, since that is the
    division a synthetic run will actually be trained under. A dataset that carries
    its own splits is read with ``select_entries`` instead.
    """
    reader = DatasetReader(root)
    spec = spec or SplitSpec()
    counts = {"total": 0, "train": 0, "val": 0, "test": 0}
    for entry in reader.entries():
        counts["total"] += 1
        counts[spec.assign(entry.id)] += 1
    return counts


def annotates_off_board(root: Path, probe: int = 200) -> bool:
    """Whether a dataset labels pieces standing beside the board."""
    reader = DatasetReader(root)
    for entry in reader.entries()[:probe]:
        if any(not piece.on_board for piece in reader.load(entry.id).pieces):
            return True
    return False


def build_mixed(
    roots: Sequence[Path],
    processor,
    *,
    split: str,
    split_spec: SplitSpec | None = None,
    repeats: Sequence[int] | None = None,
    include_board: bool = True,
    include_corners: bool = False,
    transform=None,
) -> ConcatDataset:
    """Concatenate several runs into one training set.

    Off-board annotations are dropped from *every* dataset as soon as one of them
    lacks the convention. Mixing them otherwise is actively harmful: a captured
    piece is a labelled positive in a synthetic render and unlabelled background in
    a ChessReD photograph, so the same visual pattern would be trained in both
    directions at once.

    ``repeats`` oversamples the smaller sets. Real photographs are the target
    domain and there are far fewer of them, so seeing each one once per epoch
    against seven thousand renders wastes most of their value.
    """
    conventions = [annotates_off_board(root) for root in roots]
    include_off_board = all(conventions)

    counts = repeats or [1] * len(roots)
    if len(counts) != len(roots):
        raise ValueError(f"got {len(counts)} repeat values for {len(roots)} datasets")

    parts: list[Dataset] = []
    for root, repeat in zip(roots, counts, strict=True):
        dataset = ChessDetectionDataset(
            root,
            processor,
            split=split,
            split_spec=split_spec,
            include_board=include_board,
            include_off_board=include_off_board,
            include_corners=include_corners,
            transform=transform,
        )
        parts.extend([dataset] * repeat)
    return ConcatDataset(parts)
