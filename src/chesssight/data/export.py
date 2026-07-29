"""Export a dataset into formats other tools can consume.

Masks live inside each sample record as RLE, which is compact and exact but not
something a training script can open. These writers turn them into the two things a
segmentation pipeline actually wants -- a class-per-pixel image and an
instance-per-pixel image -- plus a COCO file for detection frameworks.

Everything here works from the sample records alone, so the ``id_pass`` PNGs can be
deleted once a run is collected.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from chesssight.data.dataset import DatasetReader
from chesssight.data.fen import CLASS_NAMES, NUM_CLASSES
from chesssight.data.masks import (
    BOARD_LABEL,
    colorize,
    instance_mask,
    rle_decode,
    semantic_mask,
)
from chesssight.data.schema import Sample

SEMANTIC_DIRNAME = "semantic"
INSTANCE_DIRNAME = "instance"
PREVIEW_DIRNAME = "mask_preview"

#: Semantic label names by index, so an exported dataset is self-describing.
SEMANTIC_LABELS: tuple[str, ...] = (*CLASS_NAMES, "board")


def write_masks(
    reader: DatasetReader,
    out_dir: Path,
    *,
    include_board: bool = True,
    previews: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """Write semantic and instance mask PNGs for every sample.

    Both are single-channel 8-bit images whose *pixel values are label indices*,
    not colours -- opening one in an image viewer shows a nearly black picture, which
    is correct. Pass ``previews=True`` for colourised versions to look at.
    """
    out_dir = Path(out_dir)
    semantic_dir = out_dir / SEMANTIC_DIRNAME
    instance_dir = out_dir / INSTANCE_DIRNAME
    semantic_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)
    if previews:
        (out_dir / PREVIEW_DIRNAME).mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for sample in _limited(reader, limit):
        if not any(piece.mask is not None for piece in sample.pieces):
            skipped += 1
            continue

        semantic = semantic_mask(sample, include_board=include_board)
        instance = instance_mask(sample)
        Image.fromarray(semantic, mode="L").save(semantic_dir / f"{sample.id}.png")
        Image.fromarray(instance, mode="L").save(instance_dir / f"{sample.id}.png")

        if previews:
            Image.fromarray(colorize(semantic)).save(
                out_dir / PREVIEW_DIRNAME / f"{sample.id}.png"
            )
        written += 1

    (out_dir / "labels.json").write_text(
        json.dumps(
            {
                "semantic": dict(enumerate(SEMANTIC_LABELS)),
                "board_label": BOARD_LABEL if include_board else None,
                "note": (
                    "semantic/*.png hold class indices per pixel; "
                    "instance/*.png hold per-piece instance ids matching "
                    "sample.pieces[].instance_id"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"written": written, "skipped": skipped}


def _limited(reader: DatasetReader, limit: int | None) -> Iterable[Sample]:
    for index, sample in enumerate(reader):
        if limit is not None and index >= limit:
            return
        yield sample


def _rle_to_coco(counts: list[int], height: int, width: int) -> dict:
    """COCO stores RLE column-major; ours is row-major, so it is transposed here."""
    flat = np.zeros(height * width, dtype=np.uint8)
    position = 0
    value = 0
    for run in counts:
        flat[position : position + run] = value
        position += run
        value = 1 - value
    column_major = flat.reshape(height, width).T.reshape(-1)

    changes = np.flatnonzero(np.diff(column_major)) + 1
    boundaries = np.concatenate([[0], changes, [column_major.size]])
    runs = np.diff(boundaries).tolist()
    if column_major.size and column_major[0] == 1:
        runs = [0, *runs]
    return {"size": [height, width], "counts": [int(run) for run in runs]}


def write_coco(
    reader: DatasetReader,
    out_path: Path,
    *,
    with_masks: bool = True,
    limit: int | None = None,
) -> dict[str, int]:
    """Write a COCO instance-segmentation JSON.

    Only ``visible`` pieces become annotations, and the box used is the *modal* one
    measured from the mask. A detector trained on amodal boxes learns to predict
    pieces it cannot see.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1

    for image_id, sample in enumerate(_limited(reader, limit), start=1):
        images.append(
            {
                "id": image_id,
                "file_name": sample.image,
                "width": sample.width,
                "height": sample.height,
            }
        )
        for piece in sample.pieces:
            if not piece.visible or piece.bbox is None:
                continue
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": piece.class_id,
                "bbox": [
                    piece.bbox.x,
                    piece.bbox.y,
                    piece.bbox.width,
                    piece.bbox.height,
                ],
                "area": float(piece.visible_pixels or 0),
                "iscrowd": 0,
                "square": piece.square,
            }
            if with_masks and piece.mask is not None:
                annotation["segmentation"] = _rle_to_coco(
                    piece.mask.counts, piece.mask.height, piece.mask.width
                )
            annotations.append(annotation)
            annotation_id += 1

    document = {
        "info": {"description": "ChessSight synthetic chess pieces"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id, "name": CLASS_NAMES[class_id], "supercategory": "piece"}
            for class_id in range(1, NUM_CLASSES)
        ],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document), encoding="utf-8")
    return {"images": len(images), "annotations": len(annotations)}


def mask_area_check(sample: Sample) -> list[str]:
    """Confirm decoded masks agree with the counts recorded alongside them."""
    problems = []
    for piece in sample.pieces:
        if piece.mask is None:
            continue
        area = int(rle_decode(piece.mask).sum())
        if piece.visible_pixels is not None and area != piece.visible_pixels:
            problems.append(
                f"{sample.id} instance {piece.instance_id}: mask decodes to {area} "
                f"pixels but visible_pixels says {piece.visible_pixels}"
            )
    return problems
