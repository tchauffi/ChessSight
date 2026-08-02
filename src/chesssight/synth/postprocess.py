"""Turn raw Blender output into a validated :class:`Sample`.

The Blender side measures; this side reasons. Keeping the homography solve, the mask
decoding and the schema validation here means they are type-checked, unit-tested, and
-- most importantly -- shared with the path that ingests real photographs, so a real
sample and a synthetic one are produced by the same code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chesssight.data.fen import BOARD_SIZE, fen_to_grid, square_name
from chesssight.data.geometry import (
    BOARD_CORNERS,
    all_square_centers_board,
    board_to_image_homography,
    is_mirrored,
    matrix_to_list,
    project_square_centers,
    project_square_quads,
    reprojection_error,
)
from chesssight.data.masks import (
    instance_stats,
    load_id_image,
    piece_mask,
    rle_encode,
    visibility_ratio,
)
from chesssight.data.schema import (
    BoardAnnotation,
    BoundingBox,
    CameraInfo,
    PieceAnnotation,
    RenderInfo,
    Sample,
    SquareAnnotation,
)

#: A piece with fewer visible pixels than this is treated as hidden. Small enough to
#: keep genuinely peeking pieces, large enough to reject a stray anti-aliased sliver.
MIN_VISIBLE_PIXELS = 12

#: The two independent projections of the square centres must agree to well under a
#: pixel; anything larger means a coordinate convention is wrong somewhere.
MAX_PROJECTION_DISAGREEMENT_PX = 0.5


class PostprocessError(RuntimeError):
    """Raised when raw output cannot be turned into a trustworthy sample."""


def _bbox_from_xyxy(values: list[float] | None) -> BoundingBox | None:
    if values is None:
        return None
    x_min, y_min, x_max, y_max = values
    width, height = x_max - x_min, y_max - y_min
    if width <= 0 or height <= 0:
        return None
    return BoundingBox(x=x_min, y=y_min, width=width, height=height)


def _quad_in_frame(quad: np.ndarray, width: int, height: int) -> bool:
    return bool(
        (quad[:, 0] >= 0).all()
        and (quad[:, 0] <= width).all()
        and (quad[:, 1] >= 0).all()
        and (quad[:, 1] <= height).all()
    )


def build_sample(
    raw: dict,
    *,
    image_rel_path: str,
    mask_rel_path: str | None = None,
    split: str = "train",
    store_masks: bool = True,
) -> Sample:
    """Assemble and validate one sample from a raw label record."""
    grid = fen_to_grid(raw["fen"])
    if grid != raw["grid"]:
        raise PostprocessError(f"{raw['id']}: fen and grid disagree in the raw record")

    width, height = raw["width"], raw["height"]
    corners_px = np.asarray(raw["board"]["corners_px"], dtype=np.float64)

    homography = board_to_image_homography(corners_px)
    centers = project_square_centers(homography)
    quads = project_square_quads(homography)

    # Cross-check against Blender's own projection of the same points. These come
    # from completely different maths, so agreement is strong evidence the board
    # orientation, the y-flip and the corner ordering are all right.
    direct = np.asarray(raw["square_centers_px"], dtype=np.float64)
    disagreement = float(np.max(np.linalg.norm(centers - direct, axis=1)))
    if disagreement > MAX_PROJECTION_DISAGREEMENT_PX:
        raise PostprocessError(
            f"{raw['id']}: homography and direct projection disagree by "
            f"{disagreement:.3f} px (limit {MAX_PROJECTION_DISAGREEMENT_PX})"
        )

    # A mirrored board keeps every label self-consistent, so the projection
    # cross-check above cannot see it -- only the winding order can.
    if is_mirrored(corners_px):
        raise PostprocessError(
            f"{raw['id']}: projected board corners wind the wrong way, so the board "
            f"is mirrored. Every label is still self-consistent, but the scene is a "
            f"mirror image of a real board and must not enter the dataset."
        )

    id_image = None
    if mask_rel_path and raw.get("id_pass_path"):
        id_image = load_id_image(raw["id_pass_path"])

    squares = [
        SquareAnnotation(
            index=index,
            name=square_name(*divmod(index, BOARD_SIZE)),
            center_px=[float(centers[index][0]), float(centers[index][1])],
            quad_px=[[float(x), float(y)] for x, y in quads[index]],
            occupant=grid[index // BOARD_SIZE][index % BOARD_SIZE],
            in_frame=_quad_in_frame(quads[index], width, height),
        )
        for index in range(BOARD_SIZE * BOARD_SIZE)
    ]

    pieces = []
    for entry in raw["pieces"]:
        amodal = _bbox_from_xyxy(entry.get("bbox_amodal"))
        modal: BoundingBox | None = None
        visible_pixels: int | None = None
        visibility: float | None = None
        mask_rle = None

        if id_image is not None:
            stats = instance_stats(id_image, entry["instance_id"])
            modal = stats.bbox
            visible_pixels = stats.area
            visibility = visibility_ratio(modal, amodal, stats.area)
            if store_masks and stats.present:
                mask_rle = rle_encode(piece_mask(id_image, entry["instance_id"]))

        on_board = entry.get("on_board", True)
        pieces.append(
            PieceAnnotation(
                class_id=entry["class_id"],
                # A captured piece has no square, which is the whole point of
                # keeping it in the record: it is visible but not in play.
                square=(
                    square_name(entry["rank_index"], entry["file_index"])
                    if on_board
                    else None
                ),
                rank_index=entry["rank_index"] if on_board else None,
                file_index=entry["file_index"] if on_board else None,
                on_board=on_board,
                instance_id=entry["instance_id"],
                bbox=modal,
                bbox_amodal=amodal,
                mask=mask_rle,
                visible_pixels=visible_pixels,
                visibility=visibility,
                visible=visible_pixels is None or visible_pixels >= MIN_VISIBLE_PIXELS,
                base_center_px=[float(v) for v in entry["base_center_px"]],
            )
        )

    render_raw = raw.get("render")
    camera_raw = raw.get("camera")

    return Sample(
        id=raw["id"],
        image=image_rel_path,
        mask_image=mask_rel_path,
        width=width,
        height=height,
        source="synthetic",
        split=split,  # type: ignore[arg-type]
        annotation_method="synthetic_gt",
        seed=render_raw["seed"] if render_raw else None,
        fen=raw["fen"],
        grid=grid,
        board=BoardAnnotation(
            corners_px=[[float(x), float(y)] for x, y in corners_px],
            homography=matrix_to_list(homography),
            homography_inv=matrix_to_list(np.linalg.inv(homography)),
            reprojection_error_px=reprojection_error(
                homography, all_square_centers_board(), direct
            ),
            all_corners_in_frame=bool(raw["board"]["all_corners_in_frame"]),
        ),
        squares=squares,
        pieces=pieces,
        camera=(
            CameraInfo(
                intrinsics=camera_raw["intrinsics"],
                extrinsics=camera_raw["extrinsics"],
                focal_mm=camera_raw["focal_mm"],
                sensor_width_mm=camera_raw["sensor_width_mm"],
                resolution=camera_raw["resolution"],
            )
            if camera_raw
            else None
        ),
        render=(
            RenderInfo(
                engine=render_raw["engine"],
                samples=render_raw["samples"],
                seed=render_raw["seed"],
                blender_version=render_raw["blender_version"],
                render_seconds=render_raw.get("render_seconds"),
            )
            if render_raw
            else None
        ),
    )


def corner_reprojection_error(corners_px: np.ndarray) -> float:
    """Residual of the 4-point fit itself. Zero by construction for exactly 4 points,
    but non-zero if a caller ever passes more."""
    homography = board_to_image_homography(corners_px)
    return reprojection_error(homography, np.asarray(BOARD_CORNERS), corners_px)


def load_raw(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
