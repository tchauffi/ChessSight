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
from torch.utils.data import Dataset

from chesssight.data.dataset import DatasetReader
from chesssight.data.export import board_bbox
from chesssight.data.schema import Sample
from chesssight.train.labels import BOARD_INDEX, class_id_to_index

#: Boxes smaller than this are dropped. A two-pixel sliver of a piece behind a
#: queen is a real observation but an unlearnable target, and RT-DETR's matcher
#: will happily waste a query on it.
MIN_BOX_SIDE_PX = 2.0


@dataclass(frozen=True)
class SplitSpec:
    """How a run is divided into train and validation."""

    val_fraction: float = 0.1

    def is_val(self, sample_id: str) -> bool:
        """Deterministic per-id split.

        Hashing the id rather than slicing the index means the split survives the
        dataset being regenerated at a different size, and cannot accidentally put
        a whole correlated block of seeds on one side.
        """
        digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big") / 2**32 < self.val_fraction


def _annotations(sample: Sample, *, include_board: bool) -> list[dict]:
    """COCO-style annotations for one sample, in detector label indices."""
    records: list[dict] = []

    for piece in sample.pieces:
        if not piece.visible or piece.bbox is None:
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
        limit: int | None = None,
        split_source: str = "auto",
    ) -> None:
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"split must be train, val, test or all; got {split!r}")
        if split_source not in ("auto", "stored", "hash"):
            raise ValueError("split_source must be auto, stored or hash")

        self.root = Path(root)
        self.processor = processor
        self.include_board = include_board
        self.split = split

        reader = DatasetReader(self.root)
        spec = split_spec or SplitSpec()
        all_entries = reader.entries()

        # A dataset that carries its own division is respected. ChessReD splits by
        # *game* -- images from one game share a board, a room and a camera -- so
        # re-splitting it by hash would leak all three across the boundary and
        # flatter every number measured against it. A synthetic run has no such
        # structure and stores one split for everything, so it gets hashed.
        stored = {entry.split for entry in all_entries}
        if split_source == "auto":
            split_source = "stored" if len(stored) > 1 else "hash"
        self.split_source = split_source

        if split_source == "stored":
            entries = [
                entry for entry in all_entries if split == "all" or entry.split == split
            ]
        else:
            if split == "test":
                raise ValueError(
                    "this dataset stores a single split, so there is no separate "
                    "test set to hash out; use 'val' or pass split_source='stored'"
                )
            entries = [
                entry
                for entry in all_entries
                if split == "all" or spec.is_val(entry.id) == (split == "val")
            ]
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
        target = {
            "image_id": index,
            "annotations": _annotations(sample, include_board=self.include_board),
        }
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
    """Counts per split, for reporting before a run starts."""
    reader = DatasetReader(root)
    spec = spec or SplitSpec()
    val = sum(1 for entry in reader.entries() if spec.is_val(entry.id))
    total = len(reader.entries())
    return {"total": total, "train": total - val, "val": val}
