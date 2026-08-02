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


class CornerHeatmapDataset(Dataset):
    """Images plus a corner heatmap target from a ChessSight run.

    Shares :class:`SplitSpec` and :func:`select_entries` with the detection
    dataset, so a corner model trained here has never seen the detector's
    validation images -- which it would silently have done under a second,
    separately-written split rule.
    """

    def __init__(
        self,
        root: Path,
        *,
        image_size: int = 448,
        stride: int = 4,
        sigma: float = 2.0,
        split: str = "train",
        split_spec: SplitSpec | None = None,
        limit: int | None = None,
        split_source: str = "auto",
        transform=None,
    ) -> None:
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"split must be train, val, test or all; got {split!r}")

        self.root = Path(root)
        self.image_size = image_size
        self.stride = stride
        self.sigma = sigma
        self.transform = transform

        reader = DatasetReader(self.root)
        entries, self.split_source = select_entries(
            reader.entries(),
            split=split,
            spec=split_spec or SplitSpec(),
            split_source=split_source,
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

    def __getitem__(self, index: int) -> dict:
        from chesssight.train.heatmap import preprocess, render_target

        sample = self.sample(index)
        image = Image.open(self.root / sample.image).convert("RGB")
        points = [[float(x), float(y)] for x, y in sample.board.corners_px]

        if self.transform is not None:
            from chesssight.train.augment import apply_corners

            image, points = apply_corners(self.transform, image, points)
        else:
            # No augmentation still means a resize to the working square, and the
            # points must follow it: the two axes scale independently when the
            # source is not square.
            width, height = image.size
            scale_x = self.image_size / width
            scale_y = self.image_size / height
            image = image.resize((self.image_size, self.image_size))
            points = [[x * scale_x, y * scale_y] for x, y in points]

        target = render_target(
            points, self.image_size, stride=self.stride, sigma=self.sigma
        )
        return {
            "pixel_values": preprocess(image, self.image_size).squeeze(0),
            "target": target,
            # Kept so validation can measure pixel error against the points the
            # model was actually shown, augmentation included.
            "points": torch.tensor(points, dtype=torch.float32),
            "visible": torch.tensor(
                [
                    0 <= x < self.image_size and 0 <= y < self.image_size
                    for x, y in points
                ],
                dtype=torch.bool,
            ),
        }


class BoxCornerDataset(Dataset):
    """A crop around the board box, and the corners in that crop's coordinates.

    Shares :func:`select_entries` with every other loader here, so a model
    trained on this has not seen the others' validation images.
    """

    def __init__(
        self,
        root: Path,
        *,
        image_size: int = 224,
        margin: float = 0.25,
        jitter_scale: float = 0.0,
        jitter_shift: float = 0.0,
        split: str = "train",
        split_spec: SplitSpec | None = None,
        limit: int | None = None,
        split_source: str = "auto",
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.margin = margin
        self.jitter_scale = jitter_scale
        self.jitter_shift = jitter_shift
        self.seed = seed

        reader = DatasetReader(self.root)
        entries, self.split_source = select_entries(
            reader.entries(),
            split=split,
            spec=split_spec or SplitSpec(),
            split_source=split_source,
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

    def __getitem__(self, index: int) -> dict:
        import numpy as np

        from chesssight.train.boxcorners import crop_box, jitter, prepare, to_box_space
        from chesssight.train.corners import order_clockwise

        sample = self.sample(index)
        box = board_bbox(sample)
        if box is None:
            raise ValueError(f"sample {sample.id} has no usable board box")

        raw = (box.x, box.y, box.x + box.width, box.y + box.height)
        if self.jitter_scale or self.jitter_shift:
            # Seeded per sample and per epoch-independent index so a validation
            # set built with jitter is still the same set every time it is used.
            rng = np.random.default_rng(self.seed + index)
            raw = jitter(raw, rng, scale=self.jitter_scale, shift=self.jitter_shift)
        crop = crop_box(raw, self.margin)

        # Ordered in crop space, which is where the model sees them: the target
        # has to be a deterministic function of the picture, or two images that
        # look identical carry different labels and the regression cannot fit.
        corners = order_clockwise(
            [(float(x), float(y)) for x, y in sample.board.corners_px]
        )
        target = to_box_space(corners, crop)

        image = Image.open(self.root / sample.image)
        return {
            "pixel_values": prepare(image, crop, self.image_size).squeeze(0),
            "target": torch.tensor(target, dtype=torch.float32),
        }


class RectifiedBoardDataset(Dataset):
    """A board warped to a canonical square, and its 8x8 grid of class ids."""

    def __init__(
        self,
        root: Path,
        *,
        image_size: int = 448,
        side_margin: float = 1.65,
        far_margin: float = 2.5,
        corner_jitter: float = 0.0,
        split: str = "train",
        split_spec: SplitSpec | None = None,
        limit: int | None = None,
        split_source: str = "auto",
        transform=None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.side_margin = side_margin
        self.far_margin = far_margin
        self.corner_jitter = corner_jitter
        self.transform = transform
        self.seed = seed

        reader = DatasetReader(self.root)
        entries, self.split_source = select_entries(
            reader.entries(),
            split=split,
            spec=split_spec or SplitSpec(),
            split_source=split_source,
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

    def __getitem__(self, index: int) -> dict:
        import numpy as np

        from chesssight.train.gridnet import prepare, to_tensor
        from chesssight.train.rectify import rectify

        sample = self.sample(index)
        corners = [[float(x), float(y)] for x, y in sample.board.corners_px]

        if self.corner_jitter:
            # Perturb by a fraction of a *square*, not of the image: the same
            # pixel error means very different things on a close-up and a
            # distant board, and the model is deployed behind a corner detector
            # whose error is naturally measured in squares.
            rng = np.random.default_rng(self.seed + index)
            side = (
                float(
                    np.mean(
                        [
                            np.linalg.norm(
                                np.asarray(corners[i])
                                - np.asarray(corners[(i + 1) % len(corners)])
                            )
                            for i in range(len(corners))
                        ]
                    )
                )
                / 8.0
            )
            corners = [
                [
                    x + rng.normal(0, self.corner_jitter) * side,
                    y + rng.normal(0, self.corner_jitter) * side,
                ]
                for x, y in corners
            ]

        image = Image.open(self.root / sample.image)
        warped = rectify(
            image,
            corners,
            size=self.image_size,
            side=self.side_margin,
            far=self.far_margin,
        )
        # Augment on the tensor, not the PIL image: the sensor-realism steps
        # are tensor-only and raise on a PIL input.
        pixels = to_tensor(warped)
        if self.transform is not None:
            pixels = self.transform(pixels)

        grid = [[0] * 8 for _ in range(8)]
        for square_index, square in enumerate(sample.squares):
            grid[square_index // 8][square_index % 8] = square.occupant or 0

        return {
            "pixel_values": prepare(pixels, self.image_size).squeeze(0),
            "target": torch.tensor(grid, dtype=torch.long),
        }


def collate_grid(batch: Sequence[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }


def collate_box_corners(batch: Sequence[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }


def collate_corners(batch: Sequence[dict]) -> dict:
    """Stack a corner batch. Point counts are fixed at four, so everything stacks."""
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "points": torch.stack([item["points"] for item in batch]),
        "visible": torch.stack([item["visible"] for item in batch]),
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
