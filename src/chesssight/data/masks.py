"""Decode the Workbench instance-id pass.

The renderer writes a PNG whose red channel is the instance id and whose green
channel is a role code, with anti-aliasing and dithering disabled so every pixel
holds an exact integer. Decoding happens here, project-side, where Pillow reads rows
top-down and matches the pixel convention used by the projected labels -- Blender's
own ``image.pixels`` is bottom-up and would need a flip that is easy to forget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from chesssight.data.fen import NUM_CLASSES
from chesssight.data.schema import BoundingBox, MaskRLE, Sample

#: Role codes written into the green channel; mirrors ``blender.bl_utils.ROLE_CODES``.
ROLE_CODES = {"piece": 1, "board": 2, "table": 3, "backdrop": 4, "distractor": 5}


@dataclass(frozen=True)
class InstanceStats:
    """What one instance id occupies in the rendered image."""

    instance_id: int
    present: bool
    area: int
    bbox: BoundingBox | None

    @property
    def visible_pixels(self) -> int:
        return self.area


def load_id_image(path: Path | str) -> NDArray[np.uint8]:
    """Load an id pass as an ``(H, W, 3)`` uint8 array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def instance_ids(id_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """The instance-id channel."""
    return id_image[:, :, 0]


def role_codes(id_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """The role channel."""
    return id_image[:, :, 1]


def piece_mask(id_image: NDArray[np.uint8], instance_id: int) -> NDArray[np.bool_]:
    """Boolean mask of one piece.

    The role channel is part of the test because instance id 0 is also the id given
    to the board and table; requiring the piece role keeps them out.
    """
    return (instance_ids(id_image) == instance_id) & (
        role_codes(id_image) == ROLE_CODES["piece"]
    )


def bbox_from_mask(mask: NDArray[np.bool_]) -> BoundingBox | None:
    """Tight box around the set pixels, or ``None`` if the mask is empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0 or len(columns) == 0:
        return None
    return BoundingBox(
        x=float(columns[0]),
        y=float(rows[0]),
        width=float(columns[-1] - columns[0] + 1),
        height=float(rows[-1] - rows[0] + 1),
    )


def instance_stats(id_image: NDArray[np.uint8], instance_id: int) -> InstanceStats:
    """Area and modal bounding box of one instance."""
    mask = piece_mask(id_image, instance_id)
    area = int(mask.sum())
    return InstanceStats(
        instance_id=instance_id,
        present=area > 0,
        area=area,
        bbox=bbox_from_mask(mask) if area else None,
    )


def present_instance_ids(id_image: NDArray[np.uint8]) -> set[int]:
    """Every instance id that actually has pixels, excluding the background."""
    ids = instance_ids(id_image)
    roles = role_codes(id_image)
    values = np.unique(ids[roles == ROLE_CODES["piece"]])
    return {int(value) for value in values if value != 0}


def rle_encode(mask: NDArray[np.bool_]) -> MaskRLE:
    """COCO-style uncompressed RLE, row-major, starting with a run of zeros.

    Stored inside the sample JSON rather than as a second PNG: for a board-sized
    scene this is a few kilobytes against roughly fifty for a PNG, which decides
    whether a 50k-image dataset fits on the disk.
    """
    height, width = mask.shape
    flat = mask.reshape(-1).astype(np.uint8)

    # Run boundaries, with a leading zero-run so the parity is always
    # zeros, ones, zeros, ...
    changes = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate([[0], changes, [flat.size]])
    runs = np.diff(boundaries).tolist()

    if flat.size and flat[0] == 1:
        runs = [0, *runs]

    return MaskRLE(height=height, width=width, counts=[int(run) for run in runs])


def rle_decode(rle: MaskRLE) -> NDArray[np.bool_]:
    """Inverse of :func:`rle_encode`."""
    flat = np.zeros(rle.height * rle.width, dtype=bool)
    position = 0
    value = False
    for run in rle.counts:
        flat[position : position + run] = value
        position += run
        value = not value
    return flat.reshape(rle.height, rle.width)


def box_area(box: BoundingBox | None) -> float:
    return 0.0 if box is None else box.width * box.height


def visibility_ratio(
    modal: BoundingBox | None, amodal: BoundingBox | None, area: int
) -> float | None:
    """Fraction of a piece that is actually visible.

    Compares the mask's pixel count against the area of the amodal box, so a piece
    hidden behind another scores near zero even though its amodal box is unchanged.
    Approximate -- a box is not a silhouette -- but good enough to rank and threshold.
    """
    amodal_area = box_area(amodal)
    if amodal_area <= 0:
        return None
    del modal
    return float(min(1.0, area / amodal_area))


#: Semantic label reserved for the board surface, above the 12 piece classes.
BOARD_LABEL = NUM_CLASSES  # 13


def decode_sample_masks(sample: Sample) -> dict[int, NDArray[np.bool_]]:
    """Per-instance boolean masks decoded from the sample's stored RLE.

    Works from the sample record alone, so a dataset stays fully usable after the
    ``id_pass`` PNGs are deleted -- which is worth doing, since the RLE is an order
    of magnitude smaller.
    """
    decoded = {}
    for piece in sample.pieces:
        if piece.mask is not None and piece.instance_id is not None:
            decoded[piece.instance_id] = rle_decode(piece.mask)
    return decoded


def _pieces_with_masks(sample: Sample) -> list:
    """Pieces that carry a mask, largest first.

    Ordering matters where two decoded masks touch: painting the largest first
    leaves the smaller, more occluded piece on top instead of being swallowed.
    """
    return sorted(
        (piece for piece in sample.pieces if piece.mask is not None),
        key=lambda piece: piece.visible_pixels or 0,
        reverse=True,
    )


def instance_mask(sample: Sample) -> NDArray[np.uint8]:
    """``(H, W)`` of instance ids: 0 is background, 1..N identify pieces."""
    canvas = np.zeros((sample.height, sample.width), dtype=np.uint8)
    for piece in _pieces_with_masks(sample):
        assert piece.mask is not None
        canvas[rle_decode(piece.mask)] = piece.instance_id or 0
    return canvas


def semantic_mask(sample: Sample, *, include_board: bool = True) -> NDArray[np.uint8]:
    """``(H, W)`` of class ids: 0 background, 1..12 pieces, 13 board.

    This is the target for a segmentation model. The board gets its own label
    because segmenting the playing surface is a useful auxiliary task -- it is what
    a corner detector is really looking for.
    """
    canvas = np.zeros((sample.height, sample.width), dtype=np.uint8)

    if include_board:
        # The board polygon comes from the labels, not the mask pass, so this works
        # for annotated real photographs too.
        canvas[_polygon_fill(sample.board.corners_px, sample.height, sample.width)] = (
            BOARD_LABEL
        )

    for piece in _pieces_with_masks(sample):
        assert piece.mask is not None
        canvas[rle_decode(piece.mask)] = piece.class_id
    return canvas


def _polygon_fill(polygon, height: int, width: int) -> NDArray[np.bool_]:
    """Rasterise a convex polygon to a boolean mask."""
    image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in polygon], fill=1)
    return np.asarray(image, dtype=bool)


#: Distinct, reasonably colour-blind-tolerant hues for previewing instances.
_PALETTE_SEED = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
)


def palette(count: int) -> NDArray[np.uint8]:
    """A lookup table of ``count`` RGB colours, index 0 transparent-black."""
    colours = np.zeros((count, 3), dtype=np.uint8)
    for index in range(1, count):
        colours[index] = _PALETTE_SEED[(index - 1) % len(_PALETTE_SEED)]
    return colours


def colorize(labels: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Turn a label image into an RGB preview."""
    return palette(int(labels.max()) + 2)[labels]
