"""Extract raw pixel-space measurements from a built scene.

This module emits *measurements*, not the final annotation. It projects points with
Blender's own camera model and writes plain JSON; the homography solve, mask
decoding and schema validation all happen project-side in
:mod:`chesssight.synth.postprocess`, where the code is type-checked and unit-tested.

Projection convention
---------------------
``world_to_camera_view`` returns normalised device coordinates with the origin at the
**bottom left**, so the vertical axis is flipped here to the top-left origin that
every image library uses. Its ``z`` is the distance in front of the camera; ``z <= 0``
means the point is behind the camera, which really happens at low elevations and must
not be silently projected to a plausible-looking pixel.
"""

from __future__ import annotations

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

from chesssight.blender import bl_utils, board
from chesssight.data.fen import BOARD_SIZE

#: Cap on how many vertices are projected per piece for the amodal box. Pieces have
#: a few thousand after the lathe; a few hundred bound the silhouette just as well.
MAX_VERTICES_PER_PIECE = 400


def render_resolution() -> tuple[float, float]:
    """Render size in pixels, honouring ``resolution_percentage``."""
    render = bpy.context.scene.render
    scale = render.resolution_percentage / 100.0
    return render.resolution_x * scale, render.resolution_y * scale


def project(camera: bpy.types.Object, world_co) -> tuple[float, float, float]:
    """World point to ``(x_px, y_px, depth)`` with a top-left pixel origin."""
    scene = bpy.context.scene
    ndc = world_to_camera_view(scene, camera, Vector(world_co))
    width, height = render_resolution()
    return (ndc.x * width, (1.0 - ndc.y) * height, ndc.z)


def project_many(camera: bpy.types.Object, points: np.ndarray) -> np.ndarray:
    """Project an ``(N, 3)`` array, returning ``(N, 3)`` of ``(x, y, depth)``."""
    return np.asarray(
        [project(camera, point) for point in points], dtype=np.float64
    ).reshape(-1, 3)


def evaluated_vertices(obj: bpy.types.Object) -> np.ndarray:
    """World-space vertices of an object after its modifiers, as ``(N, 3)``.

    ``foreach_get`` into a numpy buffer rather than a Python loop: Blender bundles
    numpy 2.3.4, and a lathed piece has thousands of vertices.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        count = len(mesh.vertices)
        if count == 0:
            return np.zeros((0, 3), dtype=np.float64)
        flat = np.empty(count * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", flat)
        local = flat.reshape(count, 3).astype(np.float64)
    finally:
        evaluated.to_mesh_clear()

    if count > MAX_VERTICES_PER_PIECE:
        step = count // MAX_VERTICES_PER_PIECE
        local = local[::step]

    matrix = np.array(obj.matrix_world.to_4x4())
    homogeneous = np.hstack([local, np.ones((len(local), 1))])
    return (homogeneous @ matrix.T)[:, :3]


def intrinsics_matrix(camera: bpy.types.Object) -> list[list[float]]:
    """Pinhole intrinsics from the camera's lens, sensor and render size.

    With ``sensor_fit='AUTO'`` Blender fits the sensor to the **larger** image
    dimension, which is the detail everyone gets wrong the first time.
    """
    data = camera.data
    width, height = render_resolution()

    if data.sensor_fit == "AUTO":
        fit_px = max(width, height)
        sensor = data.sensor_width
    elif data.sensor_fit == "HORIZONTAL":
        fit_px = width
        sensor = data.sensor_width
    else:
        fit_px = height
        sensor = data.sensor_height

    focal_px = data.lens / sensor * fit_px
    center_x = width / 2.0 + data.shift_x * fit_px
    center_y = height / 2.0 - data.shift_y * fit_px

    return [
        [focal_px, 0.0, center_x],
        [0.0, focal_px, center_y],
        [0.0, 0.0, 1.0],
    ]


def extrinsics_matrix(camera: bpy.types.Object) -> list[list[float]]:
    """World-to-camera 4x4 in the OpenCV convention.

    Blender's camera looks down its local ``-Z`` with ``+Y`` up; OpenCV looks down
    ``+Z`` with ``+Y`` down. Converting once, here, keeps the flip out of every
    downstream consumer.
    """
    world_to_camera = camera.matrix_world.inverted()
    flip = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))
    return [[float(value) for value in row] for row in (flip @ world_to_camera)]


def _corner_points(corners: list[bpy.types.Object]) -> np.ndarray:
    """Corner marker positions, ordered by their tagged corner id."""
    ordered = sorted(corners, key=lambda obj: obj[bl_utils.CORNER_KEY])
    return np.asarray(
        [tuple(obj.matrix_world.translation) for obj in ordered], dtype=np.float64
    )


def _square_center_points(square_size: float) -> np.ndarray:
    """World positions of all 64 square centres, in grid reading order."""
    return np.asarray(
        [
            board.board_to_world(file_index + 0.5, rank_index + 0.5, square_size)
            for rank_index in range(BOARD_SIZE)
            for file_index in range(BOARD_SIZE)
        ],
        dtype=np.float64,
    )


def extract(job: dict, objects: dict) -> dict:
    """Collect every raw measurement for one rendered scene."""
    camera = objects["camera"]
    square_size = objects["square_size"]
    width, height = render_resolution()

    corners = project_many(camera, _corner_points(objects["corners"]))
    centers = project_many(camera, _square_center_points(square_size))

    pieces = []
    for obj in objects["pieces"]:
        vertices = evaluated_vertices(obj)
        projected = project_many(camera, vertices)
        in_front = projected[:, 2] > 0

        base = project(camera, obj.matrix_world.translation)
        apex_world = np.asarray(vertices[:, 2].argmax()) if len(vertices) else None
        apex = (
            project(camera, vertices[int(apex_world)])
            if apex_world is not None
            else base
        )

        if in_front.any():
            visible = projected[in_front]
            bbox_amodal = [
                float(visible[:, 0].min()),
                float(visible[:, 1].min()),
                float(visible[:, 0].max()),
                float(visible[:, 1].max()),
            ]
        else:
            bbox_amodal = None

        on_board = not obj.get(bl_utils.CAPTURED_KEY, False)
        pieces.append(
            {
                "instance_id": obj[bl_utils.INSTANCE_KEY],
                "class_id": obj[bl_utils.CLASS_KEY],
                "on_board": on_board,
                "rank_index": obj[bl_utils.RANK_KEY] if on_board else None,
                "file_index": obj[bl_utils.FILE_KEY] if on_board else None,
                "bbox_amodal": bbox_amodal,
                "base_center_px": [base[0], base[1]],
                "apex_px": [apex[0], apex[1]],
                "depth": base[2],
                "behind_camera": bool(base[2] <= 0),
            }
        )

    return {
        "id": job["id"],
        "fen": job["fen"],
        "grid": job["grid"],
        "width": int(round(width)),
        "height": int(round(height)),
        "image_path": job["image_path"],
        "id_pass_path": job.get("id_pass_path"),
        "board": {
            "corners_px": [[point[0], point[1]] for point in corners],
            "corner_depths": [point[2] for point in corners],
            "all_corners_in_front": bool((corners[:, 2] > 0).all()),
            "all_corners_in_frame": bool(
                (corners[:, 2] > 0).all()
                and (corners[:, 0] >= 0).all()
                and (corners[:, 0] <= width).all()
                and (corners[:, 1] >= 0).all()
                and (corners[:, 1] <= height).all()
            ),
        },
        # Projected directly through Blender's camera model. Postprocess recomputes
        # these from the homography and compares: for a flat board the two must
        # agree to well under a pixel, so a disagreement is a bug in the coordinate
        # convention rather than something to paper over.
        "square_centers_px": [[point[0], point[1]] for point in centers],
        "square_center_depths": [point[2] for point in centers],
        "pieces": pieces,
        "camera": {
            "intrinsics": intrinsics_matrix(camera),
            "extrinsics": extrinsics_matrix(camera),
            "focal_mm": camera.data.lens,
            "sensor_width_mm": camera.data.sensor_width,
            "resolution": [int(round(width)), int(round(height))],
        },
    }
