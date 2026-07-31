"""Assemble a full scene from a resolved job spec."""

from __future__ import annotations

import math
import random
from pathlib import Path

import bpy

from chesssight.blender import bl_utils, board, materials, pieces
from chesssight.data.fen import CLASS_TO_LETTER, is_white
from chesssight.synth import profiles


def build_world(spec: dict) -> None:
    """Background lighting: an HDRI when one was resolved, else a flat colour."""
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bpy.context.scene.world = world

    tree = world.node_tree
    background = next(node for node in tree.nodes if node.type == "BACKGROUND")
    background.inputs["Strength"].default_value = spec["world_strength"]

    if spec["hdri_path"]:
        environment = tree.nodes.new("ShaderNodeTexEnvironment")
        environment.image = bpy.data.images.load(spec["hdri_path"])

        coords = tree.nodes.new("ShaderNodeTexCoord")
        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value = (
            0.0,
            0.0,
            math.radians(spec["hdri_rotation_deg"]),
        )
        tree.links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
        tree.links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
        tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    else:
        background.inputs["Color"].default_value = (*spec["world_color"], 1.0)


def build_lighting(spec: dict) -> list[bpy.types.Object]:
    """A sun as key light plus optional area fills."""
    created = []

    # Zero energy means an HDRI is doing the lighting; creating the lamp anyway
    # would leave a dead object in the scene for every frame of a long run.
    if spec["sun_energy"] <= 0.0:
        return created

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = spec["sun_energy"]
    sun_data.angle = math.radians(spec["sun_angle_deg"])
    sun_data.color = tuple(spec["sun_color"])
    sun = bl_utils.link(bpy.data.objects.new("Sun", sun_data))
    sun.location = tuple(spec["sun_location"])
    bl_utils.look_at(sun, (0.0, 0.0, 0.0))
    created.append(sun)

    for index, lamp_spec in enumerate(spec["lamps"]):
        lamp_data = bpy.data.lights.new(f"Fill{index}", type="AREA")
        lamp_data.energy = lamp_spec["energy"]
        lamp_data.size = lamp_spec["size"]
        lamp_data.color = tuple(lamp_spec["color"])
        lamp = bl_utils.link(bpy.data.objects.new(f"Fill{index}", lamp_data))
        lamp.location = tuple(lamp_spec["location"])
        bl_utils.look_at(lamp, (0.0, 0.0, 0.0))
        created.append(lamp)

    return created


def _box(
    name: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> bpy.types.Object:
    """A closed axis-aligned box."""
    vertices = [
        (x, y, z) for z in (z0, z1) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    ]
    faces = [
        (3, 2, 1, 0),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return bl_utils.new_mesh_object(name, vertices, [], faces)


def build_table(
    spec: dict, square_size: float, board_thickness: float
) -> bpy.types.Object:
    """The slab the board rests on.

    Placed exactly at the board's underside. A fixed offset leaves the board
    hovering by up to half a square once its thickness is randomised -- invisible
    from directly above, obvious at the low camera angles this dataset is full of,
    and it would leave captured pieces floating too.

    A finite box rather than a bare quad: an edgeless table running to the horizon
    is one of the clearest giveaways that an image was rendered, and the visible
    edge is what separates the table from the floor beyond it.
    """
    from chesssight.blender.board import SLAB_GAP

    extent = spec["table_size"] / 2.0
    z = -(SLAB_GAP + board_thickness)
    table = _box(
        "Table", -extent, extent, -extent, extent, z - spec["table_thickness"], z
    )
    texture = spec.get("table_texture")
    if texture:
        # A photographed surface when one was resolved. The tabletop is the largest
        # area in frame after the board, and procedural noise never made it read as
        # a material rather than a tinted plane.
        material = materials.textured(
            "TableTop",
            texture["maps"],
            scale=texture["scale"],
            rotation=math.radians(texture["rotation_deg"]),
            tint=tuple(texture["tint"]),
            roughness_shift=texture["roughness_shift"],
            hue_shift=texture.get("hue_shift", 0.0),
            saturation=texture.get("saturation", 1.0),
            brightness=texture.get("brightness", 1.0),
        )
    else:
        material = materials.wood(
            "TableTop",
            tuple(spec["table_color"]),
            roughness=spec["table_roughness"],
            grain_scale=3.0,
        )
    materials.assign(table, material)
    bl_utils.tag(table, "table")
    return table


def _disc_facing_y(
    name: str,
    radius: float,
    thickness: float,
    location: tuple[float, float, float],
    segments: int = 24,
) -> bpy.types.Object:
    """A short cylinder whose axis runs along Y, i.e. a dial facing the player."""
    vertices = []
    for ring, offset in ((0, 0.0), (1, thickness)):
        for index in range(segments):
            angle = math.tau * index / segments
            vertices.append(
                (
                    location[0] + radius * math.cos(angle),
                    location[1] + offset,
                    location[2] + radius * math.sin(angle),
                )
            )
        del ring
    faces = [
        (
            index,
            (index + 1) % segments,
            segments + (index + 1) % segments,
            segments + index,
        )
        for index in range(segments)
    ]
    faces.append(tuple(range(segments - 1, -1, -1)))
    faces.append(tuple(range(segments, 2 * segments)))
    return bl_utils.new_mesh_object(name, vertices, [], faces)


def _wedge(
    name: str,
    width: float,
    depth: float,
    front_height: float,
    back_height: float,
) -> bpy.types.Object:
    """A box with a sloped top -- the shape of most digital clocks.

    Built at the origin facing -Y; the caller rotates and places it. The slope is
    what makes the display readable from a seated player's angle, and it is the
    feature that distinguishes a digital clock's silhouette from a plain box.
    """
    half_w, half_d = width / 2.0, depth / 2.0
    vertices = [
        (-half_w, -half_d, 0.0),
        (half_w, -half_d, 0.0),
        (half_w, half_d, 0.0),
        (-half_w, half_d, 0.0),
        (-half_w, -half_d, front_height),
        (half_w, -half_d, front_height),
        (half_w, half_d, back_height),
        (-half_w, half_d, back_height),
    ]
    faces = [
        (3, 2, 1, 0),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return bl_utils.new_mesh_object(name, vertices, [], faces)


def build_clock(spec: dict) -> list[bpy.types.Object]:
    """A chess clock standing beside the board.

    Two models, because the two are common and look nothing alike: an analogue case
    with a pair of round dials and plungers on top, and a wedge-shaped digital one
    with a display and buttons. Proportions come from real clocks -- roughly
    200x125x58 mm analogue and 166x114x65 mm digital -- so against a 50 mm square a
    clock is about four squares wide.

    Everything here is tagged as scenery. A clock is emphatically not a piece, and
    the detector has already been observed calling one's plungers a bishop; the
    point of putting it in the training set is to teach it that this object exists
    and is not part of the position.
    """
    if not spec:
        return []

    width = spec["width"]
    kind = spec["kind"]
    body_material = materials.solid(
        "ClockBody", tuple(spec["body_color"]), roughness=0.45, coat=0.2
    )
    face_material = materials.solid(
        "ClockFace", tuple(spec["face_color"]), roughness=0.25, coat=0.4
    )
    button_material = materials.solid(
        "ClockButton", tuple(spec["button_color"]), roughness=0.35
    )

    # Every proportion arrives resolved, so the geometry here is a pure function of
    # the spec and two clocks in different scenes are genuinely different objects
    # rather than the same model twice.
    depth = width * spec["depth_ratio"]
    height = width * spec["height_ratio"]
    face_ratio = spec["face_ratio"]
    face_offset = spec["face_offset"]
    knob = width * spec["knob_ratio"]
    knob_count = spec["knob_count"]

    parts: list[bpy.types.Object] = []
    if kind == "digital":
        front = height * spec["slope"]
        body = _wedge("ClockBody", width, depth, front, height)
        materials.assign(body, body_material)
        parts.append(body)

        # The display sits on the sloped top, inset from the edges. Its height is
        # taken from the front so a shallow wedge does not push the panel through
        # the case.
        panel = pieces._box(
            "ClockDisplay",
            (width * face_ratio, depth * 0.34, width * 0.012),
            (0.0, -depth * face_offset, front + (height - front) * 0.30),
        )
        materials.assign(panel, face_material)
        parts.append(panel)

        # Two levers, sometimes a third in the middle -- both layouts are common.
        offsets = (
            (-width * 0.30, 0.0, width * 0.30)
            if knob_count == 3
            else (-width * 0.30, width * 0.30)
        )
        for offset in offsets:
            button = _cylinder(
                "ClockButton",
                knob,
                width * 0.05,
                (offset, depth * 0.26, height),
            )
            materials.assign(button, button_material)
            parts.append(button)
    else:
        body = pieces._box("ClockBody", (width, depth, height), (0.0, 0.0, height / 2))
        materials.assign(body, body_material)
        parts.append(body)

        for sign in (-1.0, 1.0):
            centre = (
                sign * width * face_offset,
                -depth / 2.0 - width * 0.010,
                height * 0.58,
            )
            if spec.get("bezel"):
                # A raised rim around the dial, as most cases have. Built as a
                # slightly larger disc set a touch further out, so it reads as a
                # surround rather than as a second dial.
                rim = _disc_facing_y(
                    "ClockBezel",
                    width * face_ratio * 1.16,
                    width * 0.010,
                    (centre[0], centre[1] - width * 0.006, centre[2]),
                )
                materials.assign(rim, body_material)
                parts.append(rim)

            dial = _disc_facing_y(
                "ClockDial", width * face_ratio, width * 0.012, centre
            )
            materials.assign(dial, face_material)
            parts.append(dial)

            plunger = _cylinder(
                "ClockPlunger",
                knob,
                width * 0.05,
                (sign * width * (face_offset + 0.10), 0.0, height),
            )
            materials.assign(plunger, button_material)
            parts.append(plunger)

    clock = parts[0]
    bl_utils.join(clock, parts[1:])
    clock.name = "Clock"
    clock.location = (spec["x"], spec["y"], 0.0)
    clock.rotation_euler = (0.0, 0.0, math.radians(spec["rotation_deg"]))
    # Tagged as a distractor, not as a role of its own. As far as the label pass is
    # concerned a clock *is* clutter: visible scenery that occupies no square and
    # enters no annotation. A new role would need its own code in the id pass and
    # handling everywhere downstream, to express a distinction nothing acts on.
    bl_utils.tag(clock, "distractor", instance_id=0)
    return [clock]


def build_distractors(spec: dict) -> list[bpy.types.Object]:
    """Clutter around the board, so the model cannot assume a clean table."""
    created = []
    for index, distractor in enumerate(spec["distractors"]):
        size = distractor["size"]
        x, y, _ = distractor["location"]
        kind = distractor["kind"]

        spin = math.radians(distractor["rotation_deg"])

        # Things people put down while playing, each at its own proportions. The
        # earlier set was a cup, a cube and a box sharing one size range, which made
        # half of them the wrong scale and all of them read as toy blocks.
        if kind == "cup":
            mesh_obj = _cylinder(f"Cup{index}", size * 0.45, size * 1.2, (x, y, 0.0))
        elif kind == "glass":
            # Narrower and taller than a mug, and slightly tapered by being built
            # at a smaller radius -- enough to read as a different object.
            mesh_obj = _cylinder(f"Glass{index}", size * 0.34, size * 1.7, (x, y, 0.0))
        elif kind == "notepad":
            # Flat on the table: a pad is mostly outline, and its long low
            # silhouette is nothing like a piece.
            mesh_obj = pieces._box(
                f"Notepad{index}",
                (size, size * 0.72, size * 0.06),
                (x, y, size * 0.03),
                rotation_z=spin,
            )
        elif kind == "phone":
            mesh_obj = pieces._box(
                f"Phone{index}",
                (size * 0.50, size, size * 0.035),
                (x, y, size * 0.018),
                rotation_z=spin,
            )
        elif kind == "pen":
            # Lying down, so the cylinder is rotated onto its side. A pen is the
            # one piece of clutter thin enough to be mistaken for nothing at all,
            # which is exactly why it is worth having in frame.
            mesh_obj = _cylinder(
                f"Pen{index}", size * 0.035, size, (x, y, size * 0.035)
            )
            mesh_obj.rotation_euler = (math.pi / 2.0, 0.0, spin)
        else:
            mesh_obj = pieces._box(
                f"Block{index}",
                (size, size, size),
                (x, y, size / 2.0),
                rotation_z=spin,
            )

        materials.assign(
            mesh_obj,
            materials.solid(
                f"Distractor{index}", tuple(distractor["color"]), roughness=0.5
            ),
        )
        bl_utils.tag(mesh_obj, "distractor", instance_id=0)
        created.append(mesh_obj)
    return created


def _cylinder(
    name: str,
    radius: float,
    height: float,
    location: tuple[float, float, float],
    segments: int = 20,
) -> bpy.types.Object:
    """A closed cylinder standing on ``location``."""
    vertices = []
    for ring, z in ((0, 0.0), (1, height)):
        for index in range(segments):
            angle = math.tau * index / segments
            vertices.append(
                (
                    location[0] + radius * math.cos(angle),
                    location[1] + radius * math.sin(angle),
                    location[2] + z,
                )
            )
        del ring
    faces = [
        (
            index,
            (index + 1) % segments,
            segments + (index + 1) % segments,
            segments + index,
        )
        for index in range(segments)
    ]
    faces.append(tuple(range(segments - 1, -1, -1)))
    faces.append(tuple(range(segments, 2 * segments)))
    return bl_utils.new_mesh_object(name, vertices, [], faces)


def place_pieces(
    spec: dict, square_size: float, rng: random.Random
) -> tuple[list[bpy.types.Object], pieces.PieceSet]:
    """Instantiate and position every piece named in the spec."""
    # An external set is registered on demand, so the renderer never needs to know
    # whether the geometry it places is procedural or imported.
    manifest_path = spec.get("asset_manifest")
    if manifest_path:
        from chesssight.blender import assets

        assets.register_manifest(Path(manifest_path))

    piece_set = pieces.PieceSet(spec, square_size, rng)
    placed = []

    for placement in spec["placements"]:
        class_id = placement["class_id"]
        letter = CLASS_TO_LETTER[class_id].upper()
        white = is_white(class_id)

        obj = piece_set.instantiate(class_id, f"Piece_{placement['instance_id']:02d}")

        u = placement["file_index"] + 0.5 + placement["offset_u"]
        v = placement["rank_index"] + 0.5 + placement["offset_v"]
        x, y, _ = board.board_to_world(u, v, square_size)

        if letter == "N":
            # Knights are not radially symmetric: they face the opponent, give or
            # take a nudge. A uniform spin would leave half the set facing
            # backwards, which never happens on a real board.
            facing = 0.0 if white else 180.0
            yaw = facing + placement.get("knight_yaw_deg", 0.0)
        else:
            yaw = placement["rotation_deg"]

        if placement["tipped"]:
            # A toppled piece lies on its side, resting on its widest radius.
            radius = piece_set.style.radius(letter)
            obj.location = (x, y, radius)
            obj.rotation_euler = (
                math.radians(90.0),
                0.0,
                math.radians(yaw),
            )
        else:
            obj.location = (x, y, 0.0)
            obj.rotation_euler = (
                math.radians(placement["tilt_deg"]),
                math.radians(placement["tilt_deg"] * 0.5),
                math.radians(yaw),
            )

        bl_utils.tag(
            obj,
            "piece",
            instance_id=placement["instance_id"],
            class_id=class_id,
            rank_index=placement["rank_index"],
            file_index=placement["file_index"],
        )
        placed.append(obj)

    # The set is returned so captured pieces can reuse the same prototypes: six
    # lathes, or six imports, is not something to do twice per scene.
    return placed, piece_set


def place_captured(
    spec: dict, piece_set: pieces.PieceSet, square_size: float, table_z: float
) -> list[bpy.types.Object]:
    """Stand the captured pieces on the table beside the board.

    They are tagged as pieces so they get masks and boxes like anything else, but
    carry no rank or file -- the label pass emits them with ``on_board`` false, and
    they never enter the grid.
    """
    placed = []
    for capture in spec.get("captured", []):
        class_id = capture["class_id"]
        letter = CLASS_TO_LETTER[class_id].upper()
        obj = piece_set.instantiate(class_id, f"Captured_{capture['instance_id']:02d}")

        if capture["lying"]:
            obj.location = (
                capture["x"],
                capture["y"],
                table_z + piece_set.style.radius(letter),
            )
            obj.rotation_euler = (
                math.radians(90.0),
                0.0,
                math.radians(capture["rotation_deg"]),
            )
        else:
            obj.location = (capture["x"], capture["y"], table_z)
            obj.rotation_euler = (0.0, 0.0, math.radians(capture["rotation_deg"]))

        bl_utils.tag(
            obj,
            "piece",
            instance_id=capture["instance_id"],
            class_id=class_id,
        )
        obj[bl_utils.CAPTURED_KEY] = True
        placed.append(obj)
    return placed


def build_camera(spec: dict) -> bpy.types.Object:
    """The render camera, aimed and rolled per the spec."""
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.lens = spec["focal_mm"]
    camera_data.sensor_width = spec["sensor_width_mm"]
    camera_data.sensor_fit = "AUTO"

    depth_of_field = spec["depth_of_field"]
    if depth_of_field["enabled"]:
        camera_data.dof.use_dof = True
        camera_data.dof.aperture_fstop = depth_of_field["f_stop"]
        camera_data.dof.focus_distance = depth_of_field["focus_distance"]

    camera = bl_utils.link(bpy.data.objects.new("Camera", camera_data))
    camera.location = tuple(spec["location"])
    bl_utils.look_at(camera, tuple(spec["look_at"]), spec["roll_deg"])

    bpy.context.scene.camera = camera
    return camera


def build_scene(job: dict) -> dict:
    """Build the complete scene for one job and return its key objects."""
    bl_utils.reset_scene()

    square_size = job["board"]["square_size"]
    rng = random.Random(job["seed"])

    build_world(job["lighting"])
    lights = build_lighting(job["lighting"])
    table = build_table(job["scene"], square_size, job["board"]["thickness"])
    board_parts = board.build_board(job["board"])
    placed, piece_set = place_pieces(job["pieces"], square_size, rng)
    table_z = -(board.SLAB_GAP + job["board"]["thickness"])
    placed += place_captured(job["pieces"], piece_set, square_size, table_z)
    distractors = build_distractors(job["scene"])
    distractors += build_clock(job["scene"].get("clock"))
    camera = build_camera(job["camera"])

    bpy.context.view_layer.update()
    bl_utils.assert_all_tagged()

    return {
        "camera": camera,
        "board": board_parts["board"],
        "corners": board_parts["corners"],
        "pieces": placed,
        "table": table,
        "lights": lights,
        "distractors": distractors,
        "square_size": square_size,
    }


def piece_letter(class_id: int) -> str:
    """Uppercase letter for a class id, e.g. 8 -> ``"N"``."""
    letter = CLASS_TO_LETTER[class_id].upper()
    if letter not in profiles.PIECE_LETTERS:
        raise ValueError(f"class id {class_id} is not a piece")
    return letter
