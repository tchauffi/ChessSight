"""Ingest ChessReD -- real photographs of real boards -- into the ChessSight schema.

ChessReD (Masouris & van Gemert, VISAPP 2024) is 10,800 smartphone photographs of
100 real games, annotated with each piece's square. A 2,078-image subset
("chessred2k") additionally carries per-piece bounding boxes and the four board
corners, which is exactly what this schema needs to describe a sample fully.

Once ingested, a real photograph and a synthetic render are the same object: the 64
square quads come from the same homography code either way, so every downstream
consumer treats them identically. That is the whole point of having built the schema
around real photographs from the start.

Licence
-------
ChessReD is CC BY-NC-SA 4.0. Attribution is required, commercial use is not
permitted, and -- unlike the piece models -- **ShareAlike** applies, so a derivative
dataset must carry the same licence. The terms are written into the ingested
dataset's ``meta.json`` so they travel with the images.

    Masouris, A. and van Gemert, J. (2024). End-to-End Chess Recognition.
    VISAPP 2024. Dataset: https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chesssight.data.fen import (
    BOARD_SIZE,
    CLASS_NAMES,
    EMPTY,
    Grid,
    empty_grid,
    grid_to_fen,
    parse_square_name,
    square_name,
)
from chesssight.data.geometry import (
    BOARD_CORNERS,
    board_to_image_homography,
    is_mirrored,
    matrix_to_list,
    project_square_centers,
    project_square_quads,
)
from chesssight.data.schema import (
    BoardAnnotation,
    BoundingBox,
    PieceAnnotation,
    Sample,
    SquareAnnotation,
)

LICENSE = "CC BY-NC-SA 4.0"
ATTRIBUTION = (
    "Chess Recognition Dataset (ChessReD), Masouris & van Gemert, "
    "4TU.ResearchData, https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f"
)
SOURCE_URL = "https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f"

#: ChessReD's corner keys, in *their* order. Which physical square each one denotes
#: is not stated, so it is determined from the data rather than assumed -- see
#: :func:`resolve_corner_order`.
CORNER_KEYS = ("top_left", "top_right", "bottom_right", "bottom_left")


class ChessReDError(ValueError):
    """Raised when ChessReD annotations cannot be mapped into this schema."""


def _normalise(name: str) -> str:
    return name.replace("-", "_").strip().lower()


def build_category_map(categories: list[dict]) -> dict[int, int]:
    """ChessReD category id -> ChessSight class id, matched by *name*.

    Matching by name rather than by index is not fussiness. ChessReD orders its
    pieces pawn, rook, knight, bishop, queen, king; this project orders them pawn,
    knight, bishop, rook, queen, king. Zipping the two by position would silently
    turn every rook into a knight and every bishop into a rook, producing a dataset
    that is wrong in a way no shape check would ever catch.
    """
    by_name = {_normalise(name): index for index, name in enumerate(CLASS_NAMES)}
    mapping: dict[int, int] = {}
    for category in categories:
        name = _normalise(category["name"])
        if name == "empty":
            mapping[category["id"]] = EMPTY
            continue
        if name not in by_name:
            raise ChessReDError(f"unknown ChessReD category {category['name']!r}")
        mapping[category["id"]] = by_name[name]
    return mapping


@dataclass
class ChessReDAnnotations:
    """The parsed annotation file, indexed for lookup."""

    images: dict[int, dict]
    pieces: dict[int, list[dict]]
    corners: dict[int, dict]
    category_map: dict[int, int]
    splits: dict[str, list[int]]

    @classmethod
    def load(cls, path: Path) -> ChessReDAnnotations:
        document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))

        pieces: dict[int, list[dict]] = {}
        for record in document["annotations"]["pieces"]:
            pieces.setdefault(record["image_id"], []).append(record)

        splits: dict[str, list[int]] = {}
        for name, payload in document["splits"].items():
            if "image_ids" in payload:
                splits[name] = list(payload["image_ids"])
            else:
                # chessred2k nests its own train/val/test.
                for sub, inner in payload.items():
                    splits[f"{name}_{sub}"] = list(inner["image_ids"])

        return cls(
            images={image["id"]: image for image in document["images"]},
            pieces=pieces,
            corners={
                record["image_id"]: record["corners"]
                for record in document["annotations"]["corners"]
            },
            category_map=build_category_map(document["categories"]),
            splits=splits,
        )

    def annotated_image_ids(self) -> list[int]:
        """Images carrying both corners and at least one bounding box."""
        return sorted(
            image_id
            for image_id in self.corners
            if any(piece.get("bbox") for piece in self.pieces.get(image_id, []))
        )


def grid_from_pieces(
    records: list[dict], category_map: dict[int, int]
) -> tuple[Grid, list[dict]]:
    """Build the 8x8 grid, returning it with the records that occupy a square."""
    grid = empty_grid()
    occupied = []
    for record in records:
        class_id = category_map[record["category_id"]]
        if class_id == EMPTY:
            continue
        rank_index, file_index = parse_square_name(record["chessboard_position"])
        grid[rank_index][file_index] = class_id
        occupied.append(record)
    return grid, occupied


def _corner_array(corners: dict, order: tuple[str, ...]) -> np.ndarray:
    return np.asarray([corners[key] for key in order], dtype=np.float64)


def _placement_error(corners_px: np.ndarray, occupied: list[dict], grid: Grid) -> float:
    """Mean distance from each piece's box to the square the labels give it.

    A piece stands *on* its square, so the bottom-centre of its box is the part
    that should coincide with the projected square centre. Using the box centre
    instead biases every measurement upward by half a piece height.
    """
    homography = board_to_image_homography(corners_px)
    centers = project_square_centers(homography)

    distances = []
    for record in occupied:
        box = record.get("bbox")
        if not box:
            continue
        rank_index, file_index = parse_square_name(record["chessboard_position"])
        target = centers[rank_index * BOARD_SIZE + file_index]
        foot = np.array([box[0] + box[2] / 2.0, box[1] + box[3]])
        distances.append(float(np.linalg.norm(foot - target)))

    if not distances:
        return float("inf")
    return float(np.mean(distances))


def resolve_corner_order(
    annotations: ChessReDAnnotations, *, sample_size: int = 40
) -> tuple[str, ...]:
    """Work out which ChessReD corner is a8, from the data.

    ChessReD names its corners ``top_left``/``top_right``/... in *image* space, which
    says nothing about which is the a8 corner of the board -- and boards in the
    photographs are rotated arbitrarily. Rather than guess, every rotation and
    reflection of their four corners is scored by how well the resulting homography
    puts each piece on the square its own annotation claims, and the best is taken.

    Getting this wrong yields a homography that is perfectly self-consistent and
    describes a board rotated 90 degrees, so it has to be measured, not assumed.
    """
    candidates: list[tuple[str, ...]] = []
    for start in range(4):
        rotated = tuple(CORNER_KEYS[(start + offset) % 4] for offset in range(4))
        candidates.append(rotated)
        candidates.append(tuple(reversed(rotated)))

    image_ids = annotations.annotated_image_ids()[:sample_size]
    if not image_ids:
        raise ChessReDError("no images carry both corners and bounding boxes")

    scores: list[tuple[float, tuple[str, ...]]] = []
    for order in candidates:
        errors = []
        for image_id in image_ids:
            grid, occupied = grid_from_pieces(
                annotations.pieces[image_id], annotations.category_map
            )
            corners = _corner_array(annotations.corners[image_id], order)
            if is_mirrored(corners):
                errors.append(float("inf"))
                continue
            errors.append(_placement_error(corners, occupied, grid))
        finite = [value for value in errors if np.isfinite(value)]
        scores.append((float(np.mean(finite)) if finite else float("inf"), order))

    scores.sort(key=lambda item: item[0])
    best_error, best_order = scores[0]
    if not np.isfinite(best_error):
        raise ChessReDError("no corner ordering produced a usable homography")
    return best_order


def build_sample(
    annotations: ChessReDAnnotations,
    image_id: int,
    corner_order: tuple[str, ...],
    *,
    split: str = "test",
    image_root: str = "images",
) -> Sample:
    """Turn one ChessReD image into a validated :class:`Sample`."""
    image = annotations.images[image_id]
    records = annotations.pieces.get(image_id, [])
    grid, occupied = grid_from_pieces(records, annotations.category_map)

    corners_px = _corner_array(annotations.corners[image_id], corner_order)
    homography = board_to_image_homography(corners_px)
    centers = project_square_centers(homography)
    quads = project_square_quads(homography)

    width, height = image["width"], image["height"]
    squares = [
        SquareAnnotation(
            index=index,
            name=square_name(*divmod(index, BOARD_SIZE)),
            center_px=[float(centers[index][0]), float(centers[index][1])],
            quad_px=[[float(x), float(y)] for x, y in quads[index]],
            occupant=grid[index // BOARD_SIZE][index % BOARD_SIZE],
            in_frame=bool(
                (quads[index][:, 0] >= 0).all()
                and (quads[index][:, 0] <= width).all()
                and (quads[index][:, 1] >= 0).all()
                and (quads[index][:, 1] <= height).all()
            ),
        )
        for index in range(BOARD_SIZE * BOARD_SIZE)
    ]

    pieces = []
    for instance_id, record in enumerate(occupied, start=1):
        box = record.get("bbox")
        rank_index, file_index = parse_square_name(record["chessboard_position"])
        pieces.append(
            PieceAnnotation(
                class_id=annotations.category_map[record["category_id"]],
                square=square_name(rank_index, file_index),
                rank_index=rank_index,
                file_index=file_index,
                instance_id=instance_id,
                bbox=(
                    BoundingBox(x=box[0], y=box[1], width=box[2], height=box[3])
                    if box and box[2] > 0 and box[3] > 0
                    else None
                ),
                # No masks and no amodal boxes: a hand-annotated photograph has
                # neither, and inventing them would misrepresent the annotation.
                visible=True,
            )
        )

    corners_in_frame = bool(
        (corners_px[:, 0] >= 0).all()
        and (corners_px[:, 0] <= width).all()
        and (corners_px[:, 1] >= 0).all()
        and (corners_px[:, 1] <= height).all()
    )

    return Sample(
        id=f"chessred_{image_id:06d}",
        image=f"{image_root}/{image['path'].split('images/', 1)[-1]}",
        width=width,
        height=height,
        source="real",
        split=split,  # type: ignore[arg-type]
        annotation_method="manual_corners_fen",
        fen=grid_to_fen(grid),
        grid=grid,
        board=BoardAnnotation(
            corners_px=[[float(x), float(y)] for x, y in corners_px],
            homography=matrix_to_list(homography),
            homography_inv=matrix_to_list(np.linalg.inv(homography)),
            reprojection_error_px=None,
            all_corners_in_frame=corners_in_frame,
        ),
        squares=squares,
        pieces=pieces,
    )


def canonical_corner_reference() -> np.ndarray:
    """The board-plane corners this project treats as canonical, for reference."""
    return np.asarray(BOARD_CORNERS, dtype=np.float64)


def ingest(
    annotations_path: Path,
    images_dir: Path,
    out_dir: Path,
    *,
    subset: str = "chessred2k",
    force_split: str | None = None,
    link_images: bool = True,
) -> dict[str, int]:
    """Convert ChessReD into a ChessSight run directory.

    Images are symlinked rather than copied by default: the annotated subset is
    4.7 GB, and a second copy buys nothing.

    ChessReD's own train/val/test division is respected, because it splits by
    *game* -- images from one game share a board, a room and a camera, so a random
    split would leak those across the boundary and flatter every number.
    """
    from chesssight.data.dataset import DatasetWriter
    from chesssight.data.schema import DatasetMeta

    annotations = ChessReDAnnotations.load(annotations_path)
    corner_order = resolve_corner_order(annotations)

    split_of = _split_lookup(annotations, subset)

    wanted = [
        image_id
        for image_id in annotations.annotated_image_ids()
        if image_id in split_of or force_split is not None
    ]

    writer = DatasetWriter(out_dir)
    writer.initialise(
        DatasetMeta(
            name=out_dir.name,
            created_at=_now(),
            source="real",
            master_seed=0,
            notes=f"ChessReD subset {subset!r}, corner order {corner_order}",
            asset_attribution={
                "set": "ChessReD",
                "source": SOURCE_URL,
                "license": LICENSE,
                "attribution": ATTRIBUTION,
                "warnings": [
                    f"{LICENSE} is non-commercial *and* ShareAlike: a derivative "
                    f"dataset must carry the same licence, which is stricter than "
                    f"attribution alone."
                ],
            },
        )
    )

    if link_images:
        _link_images(writer.root / "images", Path(images_dir))

    counts = {"written": 0, "skipped": 0}
    for image_id in wanted:
        try:
            sample = build_sample(
                annotations,
                image_id,
                corner_order,
                split=force_split or split_of.get(image_id, "test"),
            )
        except (ChessReDError, ValueError):
            counts["skipped"] += 1
            continue
        if not (writer.root / sample.image).exists():
            counts["skipped"] += 1
            continue
        writer.add(sample)
        counts["written"] += 1

    return counts


def _split_lookup(annotations: ChessReDAnnotations, subset: str) -> dict[int, str]:
    """Image id -> split name, using ChessReD's own division.

    Theirs splits by *game*: images from one game share a board, a room and a
    camera, so a random split would leak all three across the boundary and flatter
    every number reported against it.
    """
    lookup: dict[int, str] = {}
    prefix = f"{subset}_"
    for name, image_ids in annotations.splits.items():
        if not name.startswith(prefix):
            continue
        split = name[len(prefix) :]
        for image_id in image_ids:
            lookup[image_id] = split
    return lookup


def _link_images(link: Path, images_dir: Path) -> None:
    """Point the run's images/ at the extracted archive rather than copying it."""
    if link.is_symlink():
        link.unlink()
    elif link.is_dir() and not any(link.iterdir()):
        link.rmdir()
    if not link.exists():
        link.symlink_to(images_dir.expanduser().resolve())


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
