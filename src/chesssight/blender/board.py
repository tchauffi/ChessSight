"""The board: 64 squares, a base slab, an optional border, and corner markers.

Coordinate convention
---------------------
The board is centred on the world origin with its playing surface at ``z = 0``.
:func:`board_to_world` is the *only* place board-plane coordinates become world
coordinates -- inlining it anywhere else is how an axis flip gets into the labels
without any test noticing.
"""

from __future__ import annotations

import bpy

from chesssight.blender import bl_utils, materials
from chesssight.data.fen import BOARD_SIZE

#: Board-plane corners in the canonical order used by ``chesssight.data.geometry``:
#: clockwise from the a8 corner.
BOARD_CORNERS_UV = ((0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0))

HALF = BOARD_SIZE / 2.0

#: Squares sit exactly at z=0; the slab top is nudged below to avoid z-fighting.
SQUARE_Z = 0.0
SLAB_GAP = 0.002


def board_to_world(u: float, v: float, square_size: float = 1.0, z: float = 0.0):
    """Board-plane ``(u, v)`` to world ``(x, y, z)``.

    ``v`` is negated because it grows towards rank 1 while world ``+y`` grows
    towards rank 8.
    """
    return ((u - HALF) * square_size, (HALF - v) * square_size, z)


def is_light_square(rank_index: int, file_index: int) -> bool:
    """Whether ``grid[rank_index][file_index]`` is a light square.

    a8 (``grid[0][0]``) and h1 (``grid[7][7]``) are light, a1 is dark -- the
    standard orientation.
    """
    return (rank_index + file_index) % 2 == 0


def build_squares(
    square_size: float,
    light_material: bpy.types.Material,
    dark_material: bpy.types.Material,
) -> bpy.types.Object:
    """One mesh holding all 64 squares, with two material slots by colour.

    A single mesh rather than 64 objects: it matches the physical reality of a
    printed board, and it keeps the object count (and so the id pass) clean.

    The materials are arguments rather than assigned afterwards because Blender
    clamps ``polygon.material_index`` to the number of existing slots -- setting the
    indices on a slotless mesh silently makes every square light.
    """
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    material_indices: list[int] = []

    for rank_index in range(BOARD_SIZE):
        for file_index in range(BOARD_SIZE):
            u, v = float(file_index), float(rank_index)
            base = len(vertices)
            vertices.extend(
                [
                    board_to_world(u, v, square_size, SQUARE_Z),
                    board_to_world(u + 1.0, v, square_size, SQUARE_Z),
                    board_to_world(u + 1.0, v + 1.0, square_size, SQUARE_Z),
                    board_to_world(u, v + 1.0, square_size, SQUARE_Z),
                ]
            )
            faces.append((base, base + 1, base + 2, base + 3))
            material_indices.append(0 if is_light_square(rank_index, file_index) else 1)

    obj = bl_utils.new_mesh_object("Squares", vertices, [], faces)
    materials.assign(obj, light_material, dark_material)
    for polygon, index in zip(obj.data.polygons, material_indices, strict=True):
        polygon.material_index = index
    return obj


def build_slab(
    square_size: float, thickness: float, border_width: float
) -> bpy.types.Object:
    """The solid body under the squares, extended by the border width."""
    extent = (HALF + border_width) * square_size
    top = SQUARE_Z - SLAB_GAP
    bottom = top - thickness

    corners_xy = [
        (-extent, -extent),
        (extent, -extent),
        (extent, extent),
        (-extent, extent),
    ]
    vertices = [(x, y, bottom) for x, y in corners_xy] + [
        (x, y, top) for x, y in corners_xy
    ]
    faces = [
        (0, 3, 2, 1),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return bl_utils.new_mesh_object("BoardSlab", vertices, [], faces)


def build_corner_markers(square_size: float) -> list[bpy.types.Object]:
    """Four empties at the playing-surface corners, in canonical order.

    The label pass projects these rather than deriving corners from a bounding box,
    which would silently include the border once one is configured.
    """
    markers = []
    for corner_id, (u, v) in enumerate(BOARD_CORNERS_UV):
        empty = bpy.data.objects.new(f"Corner{corner_id}", None)
        empty.empty_display_type = "PLAIN_AXES"
        empty.empty_display_size = 0.2 * square_size
        empty.location = board_to_world(u, v, square_size, SQUARE_Z)
        empty[bl_utils.CORNER_KEY] = corner_id
        bl_utils.link(empty)
        markers.append(empty)
    return markers


def build_board(spec: dict) -> dict:
    """Build the whole board from a resolved job spec's ``board`` section.

    Returns the created objects so the caller can pose them as a unit.
    """
    square_size = spec["square_size"]
    light = tuple(spec["light_color"])
    dark = tuple(spec["dark_color"])
    roughness = spec["roughness"]

    # Both colours get the same style so the two square sets read as one board
    # rather than as two materials butted together.
    style = spec.get("material")
    maps = spec.get("maps")
    squares = build_squares(
        square_size,
        # Squares vary a touch square to square, as an inlaid board does, but far
        # less than the pieces: a chequerboard whose squares differ wildly reads as
        # damaged rather than as handmade.
        materials.organic(
            materials.styled(
                "SquareLight", light, style, roughness=roughness, coat=0.15, flat=True,
                maps=maps
            ),
            bevel_radius=0.002,
            instance_value=0.04,
            instance_saturation=0.03,
        ),
        materials.organic(
            materials.styled(
                "SquareDark", dark, style, roughness=roughness, coat=0.15, flat=True,
                maps=maps
            ),
            bevel_radius=0.002,
            instance_value=0.04,
            instance_saturation=0.03,
        ),
    )

    slab = build_slab(square_size, spec["thickness"], spec["border_width"])
    materials.assign(
        slab,
        # The frame carries the largest bevel in the scene: it is the one edge a
        # hand actually rests on, and a sharp arris there is the clearest CG tell.
        materials.organic(
            materials.styled(
                "BoardFrame",
                tuple(channel * 0.75 for channel in dark),
                style,
                roughness=min(1.0, roughness + 0.1),
                coat=0.1,
                flat=True,
                maps=maps,
            ),
            bevel_radius=0.010,
            instance_value=0.03,
            instance_saturation=0.02,
        ),
    )

    bl_utils.join(squares, [slab])
    squares.name = "Board"
    bl_utils.tag(squares, "board", instance_id=0)

    return {
        "board": squares,
        "corners": build_corner_markers(square_size),
        "square_size": square_size,
    }
