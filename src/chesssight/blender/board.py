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


#: Height above the slab at which coordinate letters sit. Small enough to read as
#: printed rather than as a raised object, large enough not to z-fight.
LABEL_Z = 0.0015


def build_coordinates(
    square_size: float, border_width: float, material: bpy.types.Material
) -> list[bpy.types.Object]:
    """Rank and file letters printed on the border, as tournament boards carry.

    Not decoration. The border is the cue that separates the playing surface from
    the board's outer edge, and on a real board it is *labelled* -- the letters
    are the most distinctive thing about it. A generator whose borders are blank
    teaches a corner model that the edge of the wood is the edge of the game,
    which is exactly the confusion measured against ChessReD: predicted quads came
    back 4.4% too large there and unbiased on renders.
    """
    if border_width < 0.35:
        return []  # no room to print anything legible

    size = min(border_width * 0.55, 0.42) * square_size
    inset = border_width * 0.5 * square_size
    letters: list[bpy.types.Object] = []

    def place(text: str, x: float, y: float, rotation: float) -> None:
        curve = bpy.data.curves.new(type="FONT", name=f"Coord{text}{x:.2f}{y:.2f}")
        curve.body = text
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.size = size
        text_object = bpy.data.objects.new(curve.name, curve)
        text_object.location = (x, y, SQUARE_Z + LABEL_Z)
        text_object.rotation_euler = (0.0, 0.0, rotation)
        bl_utils.link(text_object)

        # Convert to a mesh straight away. A font object's data is a Curve, and
        # everything downstream -- joining, modifier application, the index pass --
        # assumes meshes; leaving it a curve fails inside `join` with an error
        # that names neither the text nor the board.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mesh = bpy.data.meshes.new_from_object(text_object.evaluated_get(depsgraph))
        obj = bpy.data.objects.new(curve.name, mesh)
        obj.matrix_world = text_object.matrix_world.copy()
        bpy.data.objects.remove(text_object, do_unlink=True)
        bl_utils.link(obj)

        obj.data.materials.append(material)
        # Tagged as board so the index pass does not treat a letter as a piece.
        bl_utils.tag(obj, "board", instance_id=0)
        letters.append(obj)

    edge = HALF * square_size
    for index in range(BOARD_SIZE):
        centre = (index + 0.5 - HALF) * square_size
        # Files along the bottom and top, ranks down both sides -- the layout a
        # tournament board uses, so the model sees letters on all four borders.
        place("abcdefgh"[index], centre, -edge - inset, 0.0)
        place("abcdefgh"[index], centre, edge + inset, 0.0)
        place(str(BOARD_SIZE - index), -edge - inset, -centre, 0.0)
        place(str(BOARD_SIZE - index), edge + inset, -centre, 0.0)
    return letters


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
                "SquareLight",
                light,
                style,
                roughness=roughness,
                coat=0.15,
                flat=True,
                maps=maps,
            ),
            bevel_radius=0.002,
            instance_value=0.04,
            instance_saturation=0.03,
        ),
        materials.organic(
            materials.styled(
                "SquareDark",
                dark,
                style,
                roughness=roughness,
                coat=0.15,
                flat=True,
                maps=maps,
            ),
            bevel_radius=0.002,
            instance_value=0.04,
            instance_saturation=0.03,
        ),
    )

    # The border's *tone* is sampled, not fixed. It used to be `dark * 0.75`, so
    # every synthetic board got a frame darker than its darkest square, and a
    # corner model could learn "the playing area ends where it gets darker". Real
    # tournament boards very often do the opposite -- a white frame carrying
    # printed coordinates -- and against those the learned cue points at the
    # slab's outer edge instead. Measured: quads 4.4% too large on ChessReD,
    # unbiased on renders. Interpolating between the dark and light square
    # colours covers both conventions and everything between.
    slab = build_slab(square_size, spec["thickness"], spec["border_width"])
    tone = float(spec.get("border_tone", 0.0))
    frame_color = tuple(
        d + (light_channel - d) * tone
        for d, light_channel in zip(dark, light, strict=True)
    )
    frame = materials.organic(
        materials.styled(
            "BoardFrame",
            tuple(channel * 0.75 for channel in frame_color),
            style,
            roughness=min(1.0, roughness + 0.1),
            coat=0.1,
            flat=True,
            maps=maps,
        ),
        bevel_radius=0.010,
        instance_value=0.03,
        instance_saturation=0.02,
    )
    materials.assign(
        slab,
        # The frame carries the largest bevel in the scene: it is the one edge a
        # hand actually rests on, and a sharp arris there is the clearest CG tell.
        frame,
    )

    # Coordinates contrast against the frame rather than against the squares --
    # printed on a light border they are dark, on a dark border they are light.
    ink = light if tone < 0.5 else tuple(channel * 0.25 for channel in dark)
    coordinates = build_coordinates(
        square_size,
        spec["border_width"],
        materials.styled(
            "BoardCoordinates",
            ink,
            style,
            roughness=min(1.0, roughness + 0.2),
            coat=0.0,
            flat=True,
        ),
    )

    bl_utils.join(squares, [slab, *coordinates])
    squares.name = "Board"
    bl_utils.tag(squares, "board", instance_id=0)

    return {
        "board": squares,
        "corners": build_corner_markers(square_size),
        "square_size": square_size,
    }
