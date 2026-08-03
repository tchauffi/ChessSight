"""Procedural chess pieces, plus the provider registry.

Each piece is a surface of revolution swept from a profile in
:mod:`chesssight.synth.profiles`, finished with additive geometry where a lathe
cannot do the job: crenellations on the rook, a cross on the king, a head on the
knight. Nothing here uses boolean modifiers -- booleans on generated meshes are the
usual source of nondeterministic garbage in long batch runs, and every feature these
pieces need can be added instead of subtracted.

Prototypes and instances
------------------------
A prototype is built once per scene per piece letter, then every placement is an
``obj.copy()`` sharing that mesh datablock. Colour lives in an object-level material
slot, so one mesh serves both sets. A 32-piece board therefore performs 6 lathes
rather than 32.
"""

from __future__ import annotations

import math
import random
from typing import Protocol

import bpy

from chesssight.blender import bl_utils, materials
from chesssight.data.fen import CLASS_TO_LETTER, is_white
from chesssight.synth import profiles


class PieceStyle:
    """Per-scene style, applied to every piece so a set looks coherent."""

    def __init__(
        self,
        *,
        square_size: float = 1.0,
        height_scale: float = 1.0,
        radius_scale: float = 1.0,
        taper: float = 0.0,
        bevel_width: float = 0.006,
        lathe_segments: int = 32,
        queen_coronet: bool = False,
        rook_merlon_range: tuple[int, int] | None = None,
        letter_height_scales: dict[str, float] | None = None,
    ) -> None:
        self.square_size = square_size
        self.height_scale = height_scale
        self.radius_scale = radius_scale
        self.taper = taper
        self.bevel_width = bevel_width
        self.lathe_segments = lathe_segments
        self.queen_coronet = queen_coronet
        self.rook_merlon_range = rook_merlon_range or profiles.ROOK_CRENELLATIONS
        self.letter_height_scales = letter_height_scales or {}

    @classmethod
    def from_spec(cls, spec: dict, square_size: float) -> PieceStyle:
        merlons = spec.get("rook_merlon_range")
        return cls(
            square_size=square_size,
            height_scale=spec["height_scale"],
            radius_scale=spec["radius_scale"],
            taper=spec.get("taper", 0.0),
            bevel_width=spec["bevel_width"],
            lathe_segments=spec["lathe_segments"],
            queen_coronet=spec.get("queen_coronet", False),
            rook_merlon_range=tuple(merlons) if merlons else None,
            letter_height_scales=spec.get("letter_height_scales") or {},
        )

    def height_scale_for(self, letter: str) -> float:
        """The scene's height scale times this letter's own jitter.

        The per-letter factor is what keeps the king/queen height gap from
        being a constant the detector can memorise in place of the silhouette.
        """
        return self.height_scale * self.letter_height_scales.get(letter, 1.0)

    def height(self, letter: str) -> float:
        return profiles.piece_height(
            letter,
            square_size=self.square_size,
            height_scale=self.height_scale_for(letter),
        )

    def top_radius_scale(self, letter: str) -> float:
        """Radius scale at the top of the turned part.

        The additive geometry -- the rook's merlons, the knight's head -- stands on
        the last lathe ring, so it has to shrink and grow with that ring rather than
        with the piece's overall size. At ``z = PROFILE_TOP`` the taper factor is
        exactly ``1 - taper``; ignoring it leaves merlons jutting into space on a
        base-heavy set.
        """
        return self.radius_scale * profiles.taper_factor(
            letter, profiles.PROFILE_TOP[letter], self.taper
        )

    def radius(self, letter: str) -> float:
        return profiles.piece_radius(
            letter,
            square_size=self.square_size,
            radius_scale=self.radius_scale,
            taper=self.taper,
        )


class PieceProvider(Protocol):
    """Source of piece geometry.

    Contract every provider must satisfy, because it is all the label pass knows:
    the returned object has its origin at the base centre, stands on ``z = 0``,
    is already scaled to board units, and faces ``+Y``.
    """

    name: str

    def build(
        self, letter: str, style: PieceStyle, rng: random.Random
    ) -> bpy.types.Object:
        """Return a prototype object for ``letter`` (an uppercase piece letter)."""


REGISTRY: dict[str, PieceProvider] = {}


def register_provider(provider: PieceProvider) -> PieceProvider:
    """Register a provider under its ``name``."""
    REGISTRY[provider.name] = provider
    return provider


def get_provider(name: str) -> PieceProvider:
    if name not in REGISTRY:
        raise KeyError(f"unknown piece provider {name!r}; have {sorted(REGISTRY)}")
    return REGISTRY[name]


def _lathe(
    name: str, profile: list[tuple[float, float]], style: PieceStyle
) -> bpy.types.Object:
    """Sweep a profile around the Z axis and bake the result into a mesh."""
    vertices = [(radius, 0.0, z) for radius, z in profile]
    edges = [(index, index + 1) for index in range(len(vertices) - 1)]
    obj = bl_utils.new_mesh_object(name, vertices, edges, [])

    screw = obj.modifiers.new("Lathe", "SCREW")
    screw.axis = "Z"
    screw.angle = math.tau
    screw.steps = style.lathe_segments
    screw.render_steps = style.lathe_segments
    screw.use_merge_vertices = True
    screw.use_normal_calculate = True
    screw.use_smooth_shade = True

    if style.bevel_width > 0:
        bevel = obj.modifiers.new("Bevel", "BEVEL")
        bevel.limit_method = "ANGLE"
        bevel.angle_limit = math.radians(35)
        bevel.width = style.bevel_width
        bevel.segments = 2

    bl_utils.apply_modifiers(obj)
    return obj


def _box(
    name: str,
    size: tuple[float, float, float],
    location: tuple[float, float, float],
    rotation_z: float = 0.0,
) -> bpy.types.Object:
    """An axis-aligned box, optionally spun about Z, centred on ``location``."""
    half_x, half_y, half_z = (value / 2.0 for value in size)
    corners = [
        (-half_x, -half_y, -half_z),
        (half_x, -half_y, -half_z),
        (half_x, half_y, -half_z),
        (-half_x, half_y, -half_z),
        (-half_x, -half_y, half_z),
        (half_x, -half_y, half_z),
        (half_x, half_y, half_z),
        (-half_x, half_y, half_z),
    ]
    cosine, sine = math.cos(rotation_z), math.sin(rotation_z)
    vertices = [
        (
            x * cosine - y * sine + location[0],
            x * sine + y * cosine + location[1],
            z + location[2],
        )
        for x, y, z in corners
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return bl_utils.new_mesh_object(name, vertices, [], faces)


def _add_rook_crenellations(
    obj: bpy.types.Object, style: PieceStyle, rng: random.Random
) -> None:
    """Merlons standing on the rook's rim wall.

    They are added rather than cut, and sized to sit *within* the rim: radially they
    span exactly the wall thickness, so their outer face is flush with the rim and
    they read as part of the tower. Making them thicker than the wall -- or centring
    them on the outer radius -- leaves them jutting into space.
    """
    scale = style.square_size * style.top_radius_scale("R")
    height = style.height("R")
    rim_z = profiles.PROFILE_TOP["R"] * height
    merlon_height = height - rim_z

    outer = profiles.ROOK_RIM_OUTER * scale
    inner = profiles.ROOK_RIM_INNER * scale
    mid_radius = (outer + inner) / 2.0
    thickness = outer - inner

    count = rng.randint(*style.rook_merlon_range)
    # Merlon and gap alternate evenly around the rim.
    width = math.pi * mid_radius / count

    blocks = []
    for index in range(count):
        angle = math.tau * index / count
        blocks.append(
            _box(
                f"Merlon{index}",
                (thickness, width, merlon_height),
                (
                    mid_radius * math.cos(angle),
                    mid_radius * math.sin(angle),
                    rim_z + merlon_height / 2.0,
                ),
                rotation_z=angle,
            )
        )
    bl_utils.join(obj, blocks)


def _add_queen_coronet(
    obj: bpy.types.Object, style: PieceStyle, rng: random.Random
) -> None:
    """A ring of points encircling the queen's crown.

    Sized like the rook's merlons -- off the top-of-lathe radius scale -- and
    standing at a ring radius that clears the crown's own silhouette across the
    taper range, so the points read as part of the crown rather than floating
    beside it. Without this the procedural queen is a pure lathe whose outline
    is the king's minus the cross.
    """
    scale = style.square_size * style.top_radius_scale("Q")
    height = style.height("Q")
    z_low, z_high = (fraction * height for fraction in profiles.QUEEN_CORONET_BAND)
    ring = profiles.QUEEN_CORONET_RADIUS * scale
    thickness = profiles.QUEEN_CORONET_THICKNESS * scale

    count = rng.randint(*profiles.QUEEN_CORONET_POINTS)
    # Narrower than the merlons' 50% duty: points, not battlements.
    width = math.pi * ring / count * 0.6

    blocks = []
    for index in range(count):
        angle = math.tau * index / count
        blocks.append(
            _box(
                f"Coronet{index}",
                (thickness, width, z_high - z_low),
                (
                    ring * math.cos(angle),
                    ring * math.sin(angle),
                    (z_low + z_high) / 2.0,
                ),
                rotation_z=angle,
            )
        )
    bl_utils.join(obj, blocks)


def _add_king_cross(obj: bpy.types.Object, style: PieceStyle) -> None:
    """The king's finial: two crossed bars above the turned body."""
    height = style.height("K")
    base_z = profiles.PROFILE_TOP["K"] * height
    span = height - base_z
    bar = span * 0.28

    upright = _box("CrossUp", (bar, bar, span), (0.0, 0.0, base_z + span / 2.0))
    arm = _box(
        "CrossArm",
        (span * 0.62, bar, bar),
        (0.0, 0.0, base_z + span * 0.66),
    )
    bl_utils.join(obj, [upright, arm])


KNIGHT_PATH: tuple[tuple[float, float, float, float], ...] = (
    (-0.03, -0.26, 0.115, 0.13),
    (-0.02, -0.06, 0.145, 0.17),
    (-0.01, 0.16, 0.155, 0.21),
    (0.01, 0.38, 0.155, 0.22),
    (0.04, 0.58, 0.150, 0.21),
    (0.11, 0.72, 0.140, 0.17),
    (0.24, 0.79, 0.115, 0.13),
    (0.38, 0.75, 0.090, 0.100),
    (0.49, 0.71, 0.078, 0.086),
    (0.56, 0.665, 0.066, 0.072),
)
"""The knight's head as a lofted tube: ``(y, z, half_width_x, half_thickness)`` in
units of the head span, tracing neck -> throat -> jaw -> cheek -> muzzle -> nose.
``y`` is forward (towards the opponent) and ``z`` is up.

Three details are deliberate. The first point sits *below* zero and is narrower than
the pedestal it starts inside, so the two merge rather than the neck bulging out and
leaving a notch. The neck's ``half_thickness`` runs wider than the head's, which puts
the mane's bulk into the swept surface itself -- a separate mane slab reads as a
detached plank from every angle except dead-on. And the muzzle narrows gradually but
stops *blunt* rather than running to a point -- a real horse's muzzle is squared off,
and a tapered tip reads as a beak in profile, which is the view a camera sitting at a
player's seat sees most often."""

#: Path index the ears are anchored to -- the top of the skull, just behind the eye.
KNIGHT_EAR_ANCHOR = 6


def _tapered_box(
    name: str,
    base: tuple[float, float, float],
    *,
    base_half: tuple[float, float],
    tip_half: tuple[float, float],
    height: float,
    lean_y: float = 0.0,
) -> bpy.types.Object:
    """A box that narrows towards its top, optionally leaning along +Y.

    Used for the knight's ears. A plain box reads as a bar stuck on the head; a
    taper reads as an ear.
    """
    base_x, base_y = base_half
    tip_x, tip_y = tip_half
    x0, y0, z0 = base
    vertices = [
        (x0 - base_x, y0 - base_y, z0),
        (x0 + base_x, y0 - base_y, z0),
        (x0 + base_x, y0 + base_y, z0),
        (x0 - base_x, y0 + base_y, z0),
        (x0 - tip_x, y0 - tip_y + lean_y, z0 + height),
        (x0 + tip_x, y0 - tip_y + lean_y, z0 + height),
        (x0 + tip_x, y0 + tip_y + lean_y, z0 + height),
        (x0 - tip_x, y0 + tip_y + lean_y, z0 + height),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    return bl_utils.new_mesh_object(name, vertices, [], faces)


def _path_surface_top(
    path: tuple[tuple[float, float, float, float], ...],
    index: int,
    *,
    origin_z: float,
    span: float,
    offset_x: float,
    scale: float,
) -> tuple[float, float]:
    """``(y, z)`` of the tube's upper surface at ``path[index]``, offset sideways.

    Anchoring the ears to the geometry the loft actually produces -- rather than to
    hand-tuned constants -- is what stops them floating clear of the skull whenever
    the path or the style scaling changes.
    """
    y, z, half_width, half_thickness = path[index]
    previous = path[max(0, index - 1)]
    following = path[min(len(path) - 1, index + 1)]
    tangent_y = following[0] - previous[0]
    tangent_z = following[1] - previous[1]
    length = math.hypot(tangent_y, tangent_z) or 1.0
    normal_y, normal_z = -tangent_z / length, tangent_y / length

    # How far up the ellipse we still are once we step sideways by offset_x.
    across = min(1.0, abs(offset_x) / (half_width * scale)) if half_width else 1.0
    reach = math.sqrt(max(0.0, 1.0 - across * across)) * half_thickness * span

    return (y * span + normal_y * reach, origin_z + z * span + normal_z * reach)


def _loft(
    name: str,
    path: tuple[tuple[float, float, float, float], ...],
    *,
    origin_z: float,
    span: float,
    scale: float,
    segments: int = 12,
) -> bpy.types.Object:
    """Sweep an elliptical cross-section along a path in the YZ plane.

    Each ring lies in the plane normal to the local path direction, so the tube
    keeps its thickness around the bend at the jaw instead of pinching.
    """
    rings: list[list[int]] = []
    vertices: list[tuple[float, float, float]] = []

    for index, (y, z, half_width, half_thickness) in enumerate(path):
        # Local tangent from the neighbouring path points, then its normal in YZ.
        previous = path[max(0, index - 1)]
        following = path[min(len(path) - 1, index + 1)]
        tangent_y = following[0] - previous[0]
        tangent_z = following[1] - previous[1]
        length = math.hypot(tangent_y, tangent_z) or 1.0
        normal_y, normal_z = -tangent_z / length, tangent_y / length

        ring = []
        for step in range(segments):
            angle = math.tau * step / segments
            across = half_width * math.cos(angle) * scale
            along = half_thickness * math.sin(angle) * span
            ring.append(len(vertices))
            vertices.append(
                (
                    across,
                    y * span + normal_y * along,
                    origin_z + z * span + normal_z * along,
                )
            )
        rings.append(ring)

    faces: list[tuple[int, ...]] = []
    for lower, upper in zip(rings, rings[1:], strict=False):
        for step in range(segments):
            following_step = (step + 1) % segments
            faces.append(
                (lower[step], lower[following_step], upper[following_step], upper[step])
            )
    faces.append(tuple(reversed(rings[0])))
    faces.append(tuple(rings[-1]))

    obj = bl_utils.new_mesh_object(name, vertices, [], faces)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _add_knight_head(
    obj: bpy.types.Object, style: PieceStyle, rng: random.Random
) -> None:
    """A lofted horse head on the knight's pedestal.

    The silhouette is what matters: at dataset resolution a per-square classifier
    tells a knight from a bishop almost entirely by its outline, so the muzzle has
    to project forward and the ears have to break the top edge. Stacked boxes read
    as a bundle of bars from most azimuths; a swept tube keeps a continuous,
    horse-like profile from every direction the camera might look from.
    """
    height = style.height("N")
    scale = style.square_size * style.top_radius_scale("N")
    neck_z = profiles.PROFILE_TOP["N"] * height
    span = height - neck_z

    parts = [
        _loft(
            "KnightHead",
            KNIGHT_PATH,
            origin_z=neck_z,
            span=span,
            scale=scale,
            segments=12,
        )
    ]
    # Ears break the top edge of the silhouette, which is much of what makes a
    # knight legible at a glance. Their base is placed on the skull surface the loft
    # actually produced and then sunk into it, so they can never end up floating.
    ear_offset = 0.055 * scale
    ear_height = span * 0.19
    for side, name in ((-1.0, "EarL"), (1.0, "EarR")):
        anchor_y, anchor_z = _path_surface_top(
            KNIGHT_PATH,
            KNIGHT_EAR_ANCHOR,
            origin_z=neck_z,
            span=span,
            offset_x=ear_offset,
            scale=scale,
        )
        parts.append(
            _tapered_box(
                name,
                (side * ear_offset, anchor_y, anchor_z - ear_height * 0.45),
                base_half=(0.032 * scale, 0.030 * scale),
                tip_half=(0.008 * scale, 0.008 * scale),
                height=ear_height,
                lean_y=-0.02 * scale,
            )
        )

    bl_utils.join(obj, parts)

    # Real knights are not perfectly upright and no two are carved identically.
    obj.rotation_euler = (math.radians(rng.uniform(-4.0, 4.0)), 0.0, 0.0)


class ProceduralProvider:
    """Builds every piece from the lathe profiles, with no external assets."""

    name = "procedural"

    def build(
        self, letter: str, style: PieceStyle, rng: random.Random
    ) -> bpy.types.Object:
        profile = profiles.scaled_profile(
            letter,
            square_size=style.square_size,
            height_scale=style.height_scale_for(letter),
            radius_scale=style.radius_scale,
            taper=style.taper,
        )
        obj = _lathe(f"Piece_{letter}", profile, style)

        if letter == "R":
            _add_rook_crenellations(obj, style, rng)
        elif letter == "Q" and style.queen_coronet:
            _add_queen_coronet(obj, style, rng)
        elif letter == "K":
            _add_king_cross(obj, style)
        elif letter == "N":
            _add_knight_head(obj, style, rng)

        # One empty slot so instances can override the material per object.
        obj.data.materials.append(None)
        return obj


register_provider(ProceduralProvider())


class PieceSet:
    """Prototypes for one scene, plus the two colour materials."""

    def __init__(self, spec: dict, square_size: float, rng: random.Random) -> None:
        self.style = PieceStyle.from_spec(spec, square_size)
        provider = get_provider(spec["provider"])

        self.prototypes: dict[str, bpy.types.Object] = {}
        for letter in profiles.PIECE_LETTERS:
            prototype = provider.build(letter, self.style, rng)
            # Keep prototypes out of the scene entirely; only their copies are
            # placed. Unlinking rather than hiding means a stray prototype cannot
            # be rendered at the origin by some later change.
            bl_utils.unlink(prototype)
            self.prototypes[letter] = prototype

        # Both colours share the scene's style: a set is turned from one timber or
        # cut from one stone, and only the finish differs between the two sides.
        style = spec.get("material")
        # Pieces are the most repeated object in frame -- eight identical pawns a
        # side -- so they get the largest per-instance variation, and the smallest
        # bevel, since a turned piece has softened rather than chamfered edges.
        self.white_material = materials.organic(
            materials.styled(
                "PieceWhite",
                tuple(spec["white_color"]),
                style,
                roughness=spec["roughness"],
                maps=spec.get("light_maps"),
            ),
            bevel_radius=0.003,
        )
        self.black_material = materials.organic(
            materials.styled(
                "PieceBlack",
                tuple(spec["black_color"]),
                style,
                roughness=spec["roughness"],
                maps=spec.get("dark_maps"),
            ),
            bevel_radius=0.003,
        )

    def instantiate(self, class_id: int, name: str) -> bpy.types.Object:
        """A linked duplicate of the prototype for ``class_id``, ready to place."""
        letter = CLASS_TO_LETTER[class_id].upper()
        prototype = self.prototypes[letter]

        obj = prototype.copy()  # shares the mesh datablock
        obj.name = name
        bl_utils.link(obj)

        materials.assign_object_level(
            obj, self.white_material if is_white(class_id) else self.black_material
        )
        return obj
