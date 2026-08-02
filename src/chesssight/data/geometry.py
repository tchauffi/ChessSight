"""Planar geometry: board <-> image homographies and square projection.

Board coordinates
-----------------
The board plane uses "square units": a point ``(u, v)`` has ``u`` running along the
files (a-file at ``u = 0``, h-file edge at ``u = 8``) and ``v`` running along the
ranks in *grid order*, so ``v = 0`` is the rank-8 edge and ``v = 8`` is the rank-1
edge. The centre of ``grid[rank][file]`` is therefore ``(file + 0.5, rank + 0.5)``,
matching :mod:`chesssight.data.fen` with no axis flips anywhere.

The four board corners, in the canonical order used throughout the project, are::

    0: (0, 0)  a8 corner        1: (8, 0)  h8 corner
    3: (0, 8)  a1 corner        2: (8, 8)  h1 corner

i.e. clockwise starting from the a8 corner.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from chesssight.data.fen import BOARD_SIZE

Point: TypeAlias = tuple[float, float]
Matrix3: TypeAlias = list[list[float]]

#: Board-plane coordinates of the four board corners, in canonical order.
BOARD_CORNERS: tuple[Point, Point, Point, Point] = (
    (0.0, 0.0),
    (float(BOARD_SIZE), 0.0),
    (float(BOARD_SIZE), float(BOARD_SIZE)),
    (0.0, float(BOARD_SIZE)),
)


class GeometryError(ValueError):
    """Raised when a homography cannot be determined from the given points."""


def _as_points(points: object, name: str) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise GeometryError(f"{name} must have shape (N, 2), got {array.shape}")
    if not np.isfinite(array).all():
        raise GeometryError(f"{name} contains non-finite values")
    return array


def _normalize(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Hartley normalisation: centre at the origin, mean distance ``sqrt(2)``."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_distance = float(np.sqrt((centred**2).sum(axis=1)).mean())
    if mean_distance < 1e-12:
        raise GeometryError("degenerate point set: all points are coincident")
    scale = np.sqrt(2.0) / mean_distance
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return centred * scale, transform


def solve_homography(src: object, dst: object) -> NDArray[np.float64]:
    """Solve the 3x3 homography ``H`` with ``dst ~ H @ src`` by normalised DLT.

    At least four point correspondences are required; with more than four the
    solution is the total-least-squares fit.
    """
    source = _as_points(src, "src")
    target = _as_points(dst, "dst")
    if source.shape != target.shape:
        raise GeometryError(
            f"src and dst must have the same shape, "
            f"got {source.shape} and {target.shape}"
        )
    if len(source) < 4:
        raise GeometryError(f"need at least 4 correspondences, got {len(source)}")

    source_n, source_t = _normalize(source)
    target_n, target_t = _normalize(target)

    rows = []
    for (x, y), (u, v) in zip(source_n, target_n, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    design = np.asarray(rows, dtype=np.float64)

    _, singular_values, vt = np.linalg.svd(design)
    # A well-conditioned 4-point problem leaves exactly one near-zero singular
    # value; a second one means the points are collinear or otherwise degenerate.
    if singular_values[-2] < 1e-9 * singular_values[0]:
        raise GeometryError("degenerate correspondences: no unique homography")

    homography_n = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(target_t) @ homography_n @ source_t

    if abs(homography[2, 2]) < 1e-12:
        raise GeometryError("degenerate homography: H[2, 2] is zero")
    return homography / homography[2, 2]


def apply_homography(homography: object, points: object) -> NDArray[np.float64]:
    """Map ``points`` (N, 2) through ``homography``, returning shape (N, 2)."""
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise GeometryError(f"homography must be 3x3, got {matrix.shape}")
    source = _as_points(points, "points")

    homogeneous = np.hstack([source, np.ones((len(source), 1))])
    projected = homogeneous @ matrix.T
    w = projected[:, 2:3]
    if np.any(np.abs(w) < 1e-12):
        raise GeometryError("point maps to the line at infinity")
    return np.asarray(projected[:, :2] / w, dtype=np.float64)


def board_to_image_homography(corners_px: object) -> NDArray[np.float64]:
    """Return the homography mapping board coordinates to pixels.

    ``corners_px`` are the four board corners in canonical order (see module
    docstring), as pixel coordinates.
    """
    corners = _as_points(corners_px, "corners_px")
    if len(corners) != 4:
        raise GeometryError(f"expected exactly 4 corners, got {len(corners)}")
    return solve_homography(np.asarray(BOARD_CORNERS), corners)


def square_center_board(rank_index: int, file_index: int) -> Point:
    """Board-plane centre of ``grid[rank_index][file_index]``."""
    return (file_index + 0.5, rank_index + 0.5)


def square_quad_board(rank_index: int, file_index: int) -> list[Point]:
    """Board-plane corners of a square, clockwise from its top-left."""
    u, v = float(file_index), float(rank_index)
    return [(u, v), (u + 1.0, v), (u + 1.0, v + 1.0), (u, v + 1.0)]


def all_square_centers_board() -> NDArray[np.float64]:
    """Board-plane centres of all 64 squares, in grid reading order."""
    return np.asarray(
        [
            square_center_board(rank_index, file_index)
            for rank_index in range(BOARD_SIZE)
            for file_index in range(BOARD_SIZE)
        ],
        dtype=np.float64,
    )


def project_square_centers(homography: object) -> NDArray[np.float64]:
    """Pixel centres of all 64 squares, in grid reading order. Shape (64, 2)."""
    return apply_homography(homography, all_square_centers_board())


def project_square_quads(homography: object) -> NDArray[np.float64]:
    """Pixel corners of all 64 squares, in grid reading order. Shape (64, 4, 2)."""
    quads = np.asarray(
        [
            square_quad_board(rank_index, file_index)
            for rank_index in range(BOARD_SIZE)
            for file_index in range(BOARD_SIZE)
        ],
        dtype=np.float64,
    )
    flat = apply_homography(homography, quads.reshape(-1, 2))
    return flat.reshape(BOARD_SIZE * BOARD_SIZE, 4, 2)


def reprojection_error(
    homography: object, board_points: object, pixel_points: object
) -> float:
    """Return the maximum reprojection error, in pixels.

    Used as a per-sample self-check: the renderer projects square centres with
    Blender's camera model, and this recomputes them through the fitted homography.
    A large value means the board is not planar in the labels, the corner order is
    wrong, or the camera model was misread.
    """
    predicted = apply_homography(homography, board_points)
    observed = _as_points(pixel_points, "pixel_points")
    if predicted.shape != observed.shape:
        raise GeometryError(f"shape mismatch: {predicted.shape} vs {observed.shape}")
    return float(np.max(np.linalg.norm(predicted - observed, axis=1)))


def matrix_to_list(matrix: object) -> Matrix3:
    """Convert a 3x3 array into nested Python lists for JSON serialisation."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        raise GeometryError(f"expected a 3x3 matrix, got {array.shape}")
    return [[float(value) for value in row] for row in array]


def polygon_signed_area(polygon: object) -> float:
    """Signed area by the shoelace formula; the sign encodes the winding order.

    This is how a *mirrored* board is detected. Flipping an axis somewhere in the
    pipeline leaves every label perfectly self-consistent -- corners, square centres
    and piece placements all move together -- so no amount of cross-checking
    projections will notice. What does change is the handedness: the projected
    corners wind the other way. Since a real chessboard is always set up with the
    light square on the player's right, a mirrored render is a genuine domain defect
    even though its labels are internally consistent.
    """
    vertices = _as_points(polygon, "polygon")
    if len(vertices) < 3:
        raise GeometryError("polygon needs at least 3 vertices")
    x, y = vertices[:, 0], vertices[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_mirrored(corners_px: object) -> bool:
    """Whether projected board corners wind the wrong way.

    The canonical corner order a8 -> h8 -> h1 -> a1, seen by any camera above the
    board, yields a positive signed area. A negative one means an axis is flipped.
    """
    return polygon_signed_area(corners_px) <= 0.0


def polygon_contains(polygon: object, point: Point) -> bool:
    """Return whether ``point`` lies inside a convex polygon given clockwise or
    counter-clockwise.

    Used by the QA checks to assert that projected square centres fall inside the
    projected board outline.
    """
    vertices = _as_points(polygon, "polygon")
    if len(vertices) < 3:
        raise GeometryError("polygon needs at least 3 vertices")

    px, py = point
    signs = []
    for index in range(len(vertices)):
        ax, ay = vertices[index]
        bx, by = vertices[(index + 1) % len(vertices)]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if abs(cross) > 1e-9:
            signs.append(cross > 0)
    return all(signs) or not any(signs)
