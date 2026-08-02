"""Turn a config plus a seed into a fully resolved :class:`JobSpec`.

This is where domain randomisation actually happens. It lives on the project side --
not inside Blender -- so it can be unit-tested, and so that a job spec is a complete,
inspectable record of the scene that was rendered.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Literal, cast

from chesssight.data.fen import (
    BOARD_SIZE,
    Grid,
    grid_to_fen,
    is_white,
    iter_occupied,
)
from chesssight.synth import profiles
from chesssight.synth.config import GeneratorConfig, PiecesConfig, SceneConfig
from chesssight.synth.jobspec import (
    CapturedPlacement,
    Distractor,
    JobSpec,
    MaterialStyle,
    PiecePlacement,
    ResolvedBoard,
    ResolvedCamera,
    ResolvedClock,
    ResolvedDepthOfField,
    ResolvedLamp,
    ResolvedLighting,
    ResolvedPieces,
    ResolvedRender,
    ResolvedScene,
    TableTexture,
    board_to_world,
)
from chesssight.synth.seeds import derive_rng
from chesssight.synth.textures import texture_sets

HDRI_SUFFIXES = (".hdr", ".exr")

#: Clutter a table beside a chess game actually carries. The originals were a cup,
#: a block and a clock in colours drawn uniformly per channel, which produced
#: saturated candy cubes -- next to photographed tables and varied clocks they were
#: the most obviously synthetic objects left in frame. These are things people put
#: down while playing, in the colours such things come in.
DistractorKind = Literal["cup", "block", "notepad", "pen", "phone", "glass"]
DISTRACTOR_KINDS: tuple[DistractorKind, ...] = (
    "cup",
    "block",
    "notepad",
    "pen",
    "phone",
    "glass",
)

#: Size in squares. A pen is not a mug and neither is a notepad, so the one shared
#: range that used to cover all of them made half of them the wrong scale.
DISTRACTOR_SIZES: dict[DistractorKind, tuple[float, float]] = {
    "cup": (0.9, 1.5),
    "block": (0.6, 1.4),
    "notepad": (1.8, 3.2),
    "pen": (1.6, 2.6),
    "phone": (1.5, 2.6),
    "glass": (0.7, 1.2),
}

#: Muted and plausible rather than uniform-random. Mugs are white, dark or glazed;
#: phones are black or grey; paper is off-white or squared blue.
DISTRACTOR_COLORS: dict[DistractorKind, tuple[list[float], ...]] = {
    "cup": (
        [0.82, 0.80, 0.76],
        [0.12, 0.13, 0.15],
        [0.20, 0.30, 0.42],
        [0.55, 0.14, 0.12],
    ),
    "block": ([0.45, 0.32, 0.20], [0.30, 0.31, 0.33], [0.70, 0.66, 0.58]),
    "notepad": ([0.86, 0.84, 0.78], [0.90, 0.90, 0.86], [0.25, 0.32, 0.48]),
    "pen": ([0.06, 0.06, 0.07], [0.14, 0.22, 0.45], [0.55, 0.12, 0.10]),
    "phone": ([0.05, 0.05, 0.06], [0.24, 0.25, 0.27], [0.80, 0.79, 0.76]),
    "glass": ([0.80, 0.84, 0.82], [0.72, 0.78, 0.80]),
}

#: The material styles the Blender side knows how to build. Kept here so a typo in
#: a config's weights fails at resolve time rather than rendering a plain board.
MaterialKind = Literal["plastic", "wood", "marble", "plain", "textured"]
MATERIAL_KINDS: frozenset[str] = frozenset(
    ("plastic", "wood", "marble", "plain", "textured")
)

#: Height of the tallest piece in square units, before per-scene style jitter. The
#: camera has to leave room for a king standing at the far edge, which at grazing
#: elevations is a bigger vertical extent than the whole foreshortened board.
TALLEST_PIECE = max(profiles.PIECE_HEIGHTS.values())


def _jitter_color(color: list[float], rng: random.Random, amount: float) -> list[float]:
    """Vary a colour by a *relative* amount, preserving its hue.

    ``amount`` is a fraction, not an absolute offset. That distinction matters more
    than it looks: an absolute +-0.06 is an 8% nudge on a light square at 0.76, but a
    120% swing on a dark square at 0.05 -- enough to push green above red and turn a
    brown board olive. Scaling instead keeps dark and light squares equally varied
    and keeps both recognisably the colour that was asked for.

    A shared brightness factor is applied first so a set of squares varies together
    (as a differently-finished board would), then a smaller per-channel factor adds
    a little tint.
    """
    brightness = 1.0 + rng.uniform(-amount, amount)
    return [
        min(
            1.0,
            max(0.0, channel * brightness * (1.0 + rng.uniform(-amount, amount) / 2)),
        )
        for channel in color
    ]


def _color_temperature_rgb(kelvin: float) -> list[float]:
    """A cheap blackbody approximation, good enough for lighting variation.

    Blender has a proper blackbody node, but resolving the tint here keeps the
    Blender side free of node-graph branching.
    """
    temperature = max(1000.0, min(12000.0, kelvin)) / 100.0
    if temperature <= 66:
        red = 1.0
        green = 0.39008157 * math.log(temperature) - 0.63184144
    else:
        red = 1.29293618 * (temperature - 60) ** -0.1332047592
        green = 1.12989086 * (temperature - 60) ** -0.0755148492
    if temperature >= 66:
        blue = 1.0
    elif temperature <= 19:
        blue = 0.0
    else:
        blue = 0.54320679 * math.log(temperature - 10) - 1.19625408
    return [min(1.0, max(0.0, channel)) for channel in (red, green, blue)]


Vector = tuple[float, float, float]


def camera_basis(
    azimuth_rad: float, elevation_rad: float
) -> tuple[Vector, Vector, Vector]:
    """``(toward_camera, right, up)`` for a camera aimed at the origin.

    ``toward_camera`` is the unit vector from the origin out to the camera, so a
    point's dot product with it is how much *nearer* the camera that point is than
    the board centre. That term is what makes grazing framing work.
    """
    toward_camera = (
        math.cos(elevation_rad) * math.cos(azimuth_rad),
        math.cos(elevation_rad) * math.sin(azimuth_rad),
        math.sin(elevation_rad),
    )
    right = (-math.sin(azimuth_rad), math.cos(azimuth_rad), 0.0)
    up = (
        toward_camera[1] * right[2] - toward_camera[2] * right[1],
        toward_camera[2] * right[0] - toward_camera[0] * right[2],
        toward_camera[0] * right[1] - toward_camera[1] * right[0],
    )
    return toward_camera, right, up


def framing_distance(
    focal_mm: float,
    sensor_width_mm: float,
    *,
    square_size: float,
    margin: float,
    azimuth_rad: float,
    elevation_rad: float,
    aspect: float = 1.0,
    piece_height: float = TALLEST_PIECE,
) -> float:
    """Distance at which the board and its tallest piece fill ``1 / margin`` of frame.

    Derived rather than sampled: focal length and distance are not independent, and
    drawing them separately produces a long lens at close range, which puts most of
    the board outside the image.

    Solved under true perspective, for the eight corners of the box enclosing the
    board *and a standing king*. For a camera at distance ``d`` aimed at the origin, a
    corner ``p`` sits at depth ``d - p.toward_camera`` with image offsets ``p.right``
    and ``p.up`` that do not depend on ``d`` at all, so "this corner is inside the
    frame" rearranges to a plain lower bound on ``d``::

        d >= p.toward_camera + |p.right| / tan(fov_right)

    and the answer is the largest bound over the corners and both axes. No iteration.

    The ``p.toward_camera`` term is the one that matters and the one a board-diagonal
    formula silently drops. It is the corner's depth relative to the board centre, and
    at grazing elevations the near corner is *much* closer to the camera than the
    centre is -- so it subtends a far larger angle than its size at the centre's depth
    suggests. Measured with that term missing, at 8 degrees elevation square-on to the
    board, the board spanned 1.3x to 2.9x the frame width with no corner in shot. It
    also drops piece height, which cropped the tops of kings standing near the far
    corner.
    """
    half = BOARD_SIZE / 2.0 * square_size
    top = piece_height * square_size
    toward_camera, right, up = camera_basis(azimuth_rad, elevation_rad)

    # Blender's AUTO sensor fit maps ``sensor_width`` to the longer image dimension.
    long_tan = sensor_width_mm / (2.0 * focal_mm)
    short_tan = long_tan * min(aspect, 1.0 / aspect)
    tan_right, tan_up = (
        (long_tan, short_tan) if aspect >= 1.0 else (short_tan, long_tan)
    )

    def bound(point: Vector) -> float:
        depth_offset = sum(a * b for a, b in zip(point, toward_camera, strict=True))
        lateral = abs(sum(a * b for a, b in zip(point, right, strict=True)))
        vertical = abs(sum(a * b for a, b in zip(point, up, strict=True)))
        return depth_offset + max(lateral / tan_right, vertical / tan_up)

    return margin * max(
        bound((x, y, z))
        for x in (-half, half)
        for y in (-half, half)
        for z in (0.0, top)
    )


def resolve_camera(config: GeneratorConfig, rng: random.Random) -> ResolvedCamera:
    """Place the camera on a sphere around the board centre."""
    camera_config = config.camera
    azimuth = math.radians(camera_config.azimuth_deg.sample(rng))
    elevation = math.radians(camera_config.elevation_deg.sample(rng))

    focal_mm = camera_config.focal_mm.sample(rng)
    distance = framing_distance(
        focal_mm,
        camera_config.sensor_width_mm,
        square_size=config.board.square_size,
        margin=camera_config.framing_margin.sample(rng),
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        aspect=config.render.resolution[0] / config.render.resolution[1],
        piece_height=TALLEST_PIECE * config.pieces.height_scale.max,
    )
    distance = min(
        max(distance, camera_config.distance.min * config.board.square_size),
        camera_config.distance.max * config.board.square_size,
    )

    horizontal = distance * math.cos(elevation)
    location = [
        horizontal * math.cos(azimuth),
        horizontal * math.sin(azimuth),
        distance * math.sin(elevation),
    ]
    look_at = [
        camera_config.target_jitter.sample(rng) * config.board.square_size,
        camera_config.target_jitter.sample(rng) * config.board.square_size,
        0.0,
    ]

    focus_distance = math.dist(location, look_at)
    depth_of_field = ResolvedDepthOfField(
        enabled=rng.random() < camera_config.dof_probability,
        f_stop=camera_config.f_stop.sample(rng),
        focus_distance=max(0.1, focus_distance),
    )
    return ResolvedCamera(
        location=location,
        look_at=look_at,
        roll_deg=camera_config.roll_deg.sample(rng),
        focal_mm=focal_mm,
        sensor_width_mm=camera_config.sensor_width_mm,
        depth_of_field=depth_of_field,
    )


def _find_hdris(directory: Path | None) -> list[Path]:
    if directory is None:
        return []
    expanded = Path(directory).expanduser()
    if not expanded.is_dir():
        return []
    return sorted(
        path for path in expanded.iterdir() if path.suffix.lower() in HDRI_SUFFIXES
    )


def resolve_lighting(config: GeneratorConfig, rng: random.Random) -> ResolvedLighting:
    lighting = config.lighting
    hdris = _find_hdris(lighting.hdri_dir)
    use_hdri = bool(hdris) and rng.random() < lighting.hdri_probability

    # An environment map already contains the room's windows, fixtures and bounce.
    # Adding the procedural sun and fills on top triple-lights the scene: it washes
    # the image out, casts a second set of shadows in a contradictory direction, and
    # erases the very lighting character the map was fetched for. Measured before
    # this guard: mean luminance 148-202 of 255 with almost no contrast.
    lamps = []
    lamp_count = 0 if use_hdri else lighting.lamp_count.sample(rng)
    for _ in range(lamp_count):
        azimuth = math.radians(rng.uniform(0.0, 360.0))
        elevation = math.radians(lighting.lamp_elevation_deg.sample(rng))
        distance = lighting.lamp_distance.sample(rng)
        horizontal = distance * math.cos(elevation)
        lamps.append(
            ResolvedLamp(
                location=[
                    horizontal * math.cos(azimuth),
                    horizontal * math.sin(azimuth),
                    distance * math.sin(elevation),
                ],
                energy=lighting.lamp_energy.sample(rng),
                size=lighting.lamp_size.sample(rng),
                color=_color_temperature_rgb(
                    lighting.world_color_temperature.sample(rng)
                ),
            )
        )

    sun_azimuth = math.radians(rng.uniform(0.0, 360.0))
    sun_elevation = math.radians(lighting.sun_elevation_deg.sample(rng))
    sun_horizontal = math.cos(sun_elevation)

    strength = (
        lighting.hdri_strength.sample(rng)
        if use_hdri
        else lighting.world_strength.sample(rng)
    )

    return ResolvedLighting(
        hdri_path=str(rng.choice(hdris)) if use_hdri else None,
        hdri_rotation_deg=lighting.hdri_rotation_deg.sample(rng),
        world_strength=strength,
        world_color=_color_temperature_rgb(
            lighting.world_color_temperature.sample(rng)
        ),
        sun_location=[
            sun_horizontal * math.cos(sun_azimuth) * 20.0,
            sun_horizontal * math.sin(sun_azimuth) * 20.0,
            math.sin(sun_elevation) * 20.0,
        ],
        sun_energy=0.0 if use_hdri else lighting.sun_energy.sample(rng),
        sun_angle_deg=lighting.sun_angle_deg.sample(rng),
        sun_color=_color_temperature_rgb(lighting.world_color_temperature.sample(rng)),
        lamps=lamps,
    )


def luminance(color: list[float]) -> float:
    """Perceived brightness, Rec. 709 weights."""
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def separate_from(
    color: list[float], others: list[list[float]], *, minimum: float
) -> list[float]:
    """Push a colour's brightness away from others until it is distinguishable.

    Board squares and piece colours are drawn from independent palettes, so nothing
    stopped a near-black square being paired with near-black pieces: measured on the
    defaults, the closest gap was 0.030 in luminance, which renders an ebony piece
    almost invisible against the square it stands on. The label still says a piece
    is there. That is the same failure as a board with no chequer -- an image that
    does not show what its annotation describes -- and no amount of *challenge* is
    served by a target the detector cannot see.

    Only brightness is moved, and only far enough to clear the threshold, so a hard
    pairing stays hard: this sets a floor, not a comfortable margin.
    """
    result = list(color)
    for other in others:
        gap = luminance(result) - luminance(other)
        if abs(gap) >= minimum:
            continue
        # Move away from the neighbour, or downward if it would clip past white.
        direction = 1.0 if gap >= 0.0 else -1.0
        target = luminance(other) + direction * minimum
        if target > 1.0:
            target = luminance(other) - minimum
        current = max(luminance(result), 1e-4)
        if target <= 0.0:
            result = [minimum * 0.5] * 3
        else:
            scale = target / current
            result = [min(1.0, max(0.0, channel * scale)) for channel in result]
    return result


def choose_material_style(
    weights: dict[str, float], base_color: list[float], rng: random.Random
) -> MaterialStyle:
    """Draw what a surface is made of, and the parameters its procedural needs.

    The accent colour is derived from the base rather than drawn independently: a
    growth ring is the same timber a shade darker, and a marble vein is the same
    stone with a mineral streak. Sampling the two colours separately produces
    two-tone plastic, which is the look this is meant to avoid.
    """
    kinds = list(weights)
    unknown = [k for k in kinds if k not in MATERIAL_KINDS]
    if unknown:
        # A typo in `material_styles` would otherwise fall through to the plain
        # solid and quietly produce a dataset with none of the style asked for.
        raise ValueError(
            f"unknown material style(s) {unknown}; have {sorted(MATERIAL_KINDS)}"
        )
    (drawn,) = rng.choices(kinds, weights=[weights[k] for k in kinds], k=1)
    kind = cast(MaterialKind, drawn)
    if kind == "marble":
        # Veins run darker than the body on light stone and lighter on dark stone,
        # so the contrast survives whichever way the palette went. Kept moderate:
        # a near-black vein on near-white stone reads as printed pattern, not rock.
        return MaterialStyle(
            kind=kind,
            # Low frequency on purpose. Measured on a parameter sweep: above ~1.0
            # there are enough bands across a piece that the ramp catches every
            # peak and the result is regular pinstripes rather than veining.
            scale=rng.uniform(0.3, 0.8),
            # Enough to bend the bands well out of line. Below ~3 they stay
            # straight and read as stripes; above ~8 they curl into closed blobs.
            distortion=rng.uniform(3.5, 6.0),
            contrast=rng.uniform(0.28, 0.45),
            vein_width=rng.uniform(0.12, 0.18),
            rotation_deg=[rng.uniform(0.0, 360.0) for _ in range(3)],
        )
    if kind == "wood":
        return MaterialStyle(
            kind=kind,
            # Also measured on a sweep: past ~2.5 the rings tighten into regular
            # banding that reads as turned laminate rather than as timber.
            scale=rng.uniform(0.8, 2.5),
            distortion=rng.uniform(1.5, 4.0),
            # Grain is a modulation of one timber, not two tones: a fifth of the
            # body colour reads as figure, a half reads as paint.
            contrast=rng.uniform(0.12, 0.28),
            rotation_deg=[rng.uniform(0.0, 360.0) for _ in range(3)],
        )
    if kind == "textured":
        # A photographed surface carries its own figure, so the only thing to draw
        # is how it is coloured and how big it is laid down. The ranges are wide on
        # purpose: the point of this dataset is images that are *hard*, and a board
        # in a colour nobody sells forces the detector onto shape rather than hue.
        return MaterialStyle(
            kind=kind,
            scale=rng.uniform(0.15, 0.6),
            distortion=0.0,
            contrast=0.0,
            hue_shift=rng.uniform(-0.5, 0.5),
            saturation=rng.uniform(0.4, 1.5),
            brightness=rng.uniform(0.6, 1.4),
        )
    return MaterialStyle(kind=kind, scale=1.0, distortion=0.0, contrast=0.0)


def resolve_board(config: GeneratorConfig, rng: random.Random) -> ResolvedBoard:
    board = config.board
    jitter = board.color_jitter.max
    light = _jitter_color(rng.choice(board.light_square_color), rng, jitter)
    dark = _jitter_color(rng.choice(board.dark_square_color), rng, jitter)

    # Keep both squares clear of both piece colours. Done here rather than when the
    # pieces are resolved because the board has the freer palette: a set is boxwood
    # and ebony, but a board can be any of a dozen stains.
    pieces_colors = [config.pieces.white_color, config.pieces.black_color]
    light = separate_from(light, pieces_colors, minimum=board.min_piece_contrast)
    dark = separate_from(dark, pieces_colors, minimum=board.min_piece_contrast)
    # Keyed off the dark squares: they carry the figure a real veneered or inlaid
    # board shows, and the light ones follow the same style so the two read as one
    # board rather than two materials butted together.
    style = choose_material_style(board.material_styles, dark, rng)
    # The board is eight squares across, so a scale stated per *square* runs at
    # roughly twenty times the frequency the same style uses on a piece -- which
    # turned marble veining into dense scribble. Divide into board units so the two
    # surfaces carry figure at a comparable size.
    maps = None
    if style.kind == "textured":
        maps = _pick_texture(config.scene.texture_dir, rng)
        if maps is None:
            # No textures downloaded: fall back to the procedural rather than
            # rendering an untextured surface that silently ignores its style.
            style = style.model_copy(update={"kind": "wood"})
    if style.kind != "textured":
        style = style.model_copy(
            update={"scale": board.grain_scale.sample(rng) / BOARD_SIZE}
        )
    return ResolvedBoard(
        square_size=board.square_size,
        thickness=board.thickness.sample(rng),
        border_width=board.border_width.sample(rng),
        border_tone=board.border_tone.sample(rng),
        light_color=light,
        dark_color=dark,
        roughness=board.roughness.sample(rng),
        material=style,
        maps=maps,
    )


def resolve_clock(
    scene: SceneConfig, rng: random.Random, *, square_size: float
) -> ResolvedClock | None:
    """Place a chess clock beside the board, or None.

    Positioned against one of the four edges rather than anywhere on the table: a
    clock stands where a player can reach it without leaning over the board, which
    in practice means square to an edge and clear of the playing surface. Randomly
    chosen edge, so the model does not learn that a clock is always on the right.
    """
    if rng.random() >= scene.clock_probability:
        return None

    digital = rng.random() < scene.clock_digital_probability
    width = scene.clock_width.sample(rng) * square_size

    # Proportions are drawn, not fixed. Two rigid models would put the same two
    # objects into every scene that has a clock, and a detector would learn those
    # two silhouettes rather than the idea of a clock. The ranges straddle the
    # reference dimensions rather than sitting on them: an analogue case is an
    # *upright* box, roughly 220 wide, 60 deep and 125 tall, because its face has
    # to carry two 75 mm dials side by side -- modelled as a 60 mm-tall slab, the
    # dials came out wider than the case and rendered as a pair of headlights.
    if digital:
        depth_ratio = rng.uniform(0.58, 0.82)
        height_ratio = rng.uniform(0.30, 0.48)
        face_ratio = rng.uniform(0.45, 0.68)
        face_offset = rng.uniform(0.08, 0.18)
        knob_ratio = rng.uniform(0.055, 0.095)
        slope = rng.uniform(0.42, 0.95)
    else:
        depth_ratio = rng.uniform(0.22, 0.36)
        height_ratio = rng.uniform(0.46, 0.70)
        face_ratio = rng.uniform(0.145, 0.195)
        face_offset = rng.uniform(0.21, 0.27)
        knob_ratio = rng.uniform(0.032, 0.055)
        slope = 1.0
    depth = width * depth_ratio

    half_board = BOARD_SIZE / 2.0 * square_size
    # Beside the board, never on it. Players sit at the two ends, so the clock goes
    # to the left or the right -- the sides along the files -- and its long axis runs
    # *parallel* to that edge, which is how it sits on a real table. Clearance is
    # measured against the clock's depth, since that is the extent facing the board:
    # using its width instead let a four-square-wide clock centred just outside the
    # edge reach back across two ranks of squares.
    # Left or right of the board, drawn independently of where the camera stands.
    # Coupling the two was tried and reverted: a real clock's position has nothing
    # to do with where the photographer is, and letting the camera decide would bake
    # a correlation into the dataset that does not exist in the world. Seeing the
    # back of a clock half the time is not a defect -- it is what half the seats in
    # the room see, and a detector should recognise the object from either side.
    side = rng.choice((0.0, 180.0))
    angle = math.radians(side)
    clearance = half_board + depth / 2.0 + scene.clock_offset.sample(rng) * square_size
    along = rng.uniform(-half_board * 0.45, half_board * 0.45)
    x = clearance * math.cos(angle) - along * math.sin(angle)
    y = clearance * math.sin(angle) + along * math.cos(angle)
    # A wider palette than the two obvious ones. Real clocks come in stained wood,
    # cream, red, green and black plastic, brushed grey -- and the point of this
    # dataset is range, so the body colour is drawn from a spread rather than from
    # one plausible default per kind.
    palette = (
        [0.42, 0.28, 0.16],  # stained wood
        [0.78, 0.74, 0.66],  # cream
        [0.05, 0.05, 0.06],  # black
        [0.12, 0.20, 0.14],  # dark green
        [0.35, 0.09, 0.08],  # oxblood
        [0.30, 0.32, 0.35],  # grey
    )
    body = _jitter_color(list(rng.choice(palette)), rng, 0.28)
    return ResolvedClock(
        kind="digital" if digital else "analogue",
        x=x,
        y=y,
        # A quarter turn from the edge normal, so the long axis runs along the edge
        # and the face looks away from the board rather than into it.
        rotation_deg=side + 90.0 + rng.uniform(-10.0, 10.0),
        width=width,
        depth_ratio=depth_ratio,
        height_ratio=height_ratio,
        face_ratio=face_ratio,
        face_offset=face_offset,
        knob_ratio=knob_ratio,
        # Most digital clocks have a pair of levers; some add a centre button.
        knob_count=3 if digital and rng.random() < 0.35 else 2,
        slope=slope,
        bezel=(not digital) and rng.random() < 0.6,
        body_color=body,
        # A dial face is off-white paper behind glass, and a digital display is a
        # pale LCD. Jittered narrowly: at 0.08 relative the analogue dials came out
        # orange, which reads as a warning light rather than as a clock.
        face_color=_jitter_color([0.90, 0.89, 0.86], rng, 0.03),
        button_color=[rng.uniform(0.05, 0.25) for _ in range(3)],
    )


def _pick_texture(
    texture_dir: Path | None, rng: random.Random
) -> dict[str, str] | None:
    """Any complete texture set from the library, or None when there are none."""
    if texture_dir is None:
        return None
    available = texture_sets(texture_dir)
    return rng.choice(available)["maps"] if available else None


def choose_veneers(
    config: GeneratorConfig, material: MaterialStyle, pieces: PiecesConfig
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """Photographed veneer maps for each side, when the style is wood.

    Looked up by slug rather than drawn at random: the two sides want a *pair* --
    a pale timber against a dark one, as a real set is made -- and picking two
    veneers independently would sometimes give two pale ones and no contrast.
    """
    texture_dir = config.scene.texture_dir
    if material.kind != "wood" or not pieces.veneers or texture_dir is None:
        return None, None
    available = {entry["slug"]: entry["maps"] for entry in texture_sets(texture_dir)}
    return (
        available.get(pieces.veneers.get("light", "")),
        available.get(pieces.veneers.get("dark", "")),
    )


def choose_piece_set(
    pieces: PiecesConfig, rng: random.Random
) -> tuple[str, Path | None]:
    """Pick the chess set for one scene: ``(provider, asset_manifest)``.

    Drawn per scene rather than per run, so a single dataset mixes sets. The
    single-``provider`` form stays valid and is what a config without ``sets`` means.
    """
    if not pieces.sets:
        return pieces.provider, pieces.asset_manifest
    (chosen,) = rng.choices(
        pieces.sets, weights=[choice.weight for choice in pieces.sets], k=1
    )
    return chosen.provider, chosen.asset_manifest


def resolve_pieces(
    config: GeneratorConfig, grid: Grid, rng: random.Random
) -> ResolvedPieces:
    pieces = config.pieces
    placements = []
    for instance_id, (rank_index, file_index, class_id) in enumerate(
        iter_occupied(grid), start=1
    ):
        tipped = rng.random() < pieces.tipped_probability
        placements.append(
            PiecePlacement(
                instance_id=instance_id,
                class_id=class_id,
                rank_index=rank_index,
                file_index=file_index,
                offset_u=pieces.position_jitter.sample(rng),
                offset_v=pieces.position_jitter.sample(rng),
                rotation_deg=pieces.rotation_jitter_deg.sample(rng),
                knight_yaw_deg=pieces.knight_facing_jitter_deg.sample(rng),
                tilt_deg=pieces.tilt_jitter_deg.sample(rng),
                tipped=tipped,
            )
        )

    captured = resolve_captured(
        config, grid, rng, first_instance_id=len(placements) + 1
    )

    provider, manifest = choose_piece_set(pieces, rng)
    white_color = _jitter_color(pieces.white_color, rng, 0.05)
    black_color = _jitter_color(pieces.black_color, rng, 0.03)
    material = choose_material_style(pieces.material_styles, white_color, rng)
    material = material.model_copy(
        update={
            "hue_shift": pieces.veneer_hue_shift.sample(rng),
            "saturation": pieces.veneer_saturation.sample(rng),
            "brightness": pieces.veneer_brightness.sample(rng),
        }
    )
    light_maps, dark_maps = choose_veneers(config, material, pieces)

    return ResolvedPieces(
        provider=provider,
        asset_manifest=str(Path(manifest).expanduser()) if manifest else None,
        height_scale=pieces.height_scale.sample(rng),
        radius_scale=pieces.radius_scale.sample(rng),
        taper=pieces.taper.sample(rng),
        bevel_width=pieces.bevel_width.sample(rng),
        lathe_segments=pieces.lathe_segments.sample(rng),
        white_color=white_color,
        black_color=black_color,
        roughness=pieces.roughness.sample(rng),
        material=material,
        light_maps=light_maps,
        dark_maps=dark_maps,
        placements=placements,
        captured=captured,
    )


#: How many of each class a complete set holds, keyed by class id.
FULL_SET: dict[int, int] = {
    1: 8,
    2: 2,
    3: 2,
    4: 2,
    5: 1,
    6: 1,  # white P N B R Q K
    7: 8,
    8: 2,
    9: 2,
    10: 2,
    11: 1,
    12: 1,  # black
}


def missing_pieces(grid: Grid) -> list[int]:
    """Class ids absent from ``grid`` relative to a complete set.

    This is what makes the captured pile *consistent with the position* rather than
    decorative: the pieces beside the board are exactly the ones not on it. A random
    pile would teach the model that the two are unrelated, which is the opposite of
    what a real game shows.
    """
    on_board: dict[int, int] = {}
    for _, _, class_id in iter_occupied(grid):
        on_board[class_id] = on_board.get(class_id, 0) + 1

    absent = []
    for class_id, total in FULL_SET.items():
        absent.extend([class_id] * max(0, total - on_board.get(class_id, 0)))
    return absent


def resolve_captured(
    config: GeneratorConfig,
    grid: Grid,
    rng: random.Random,
    *,
    first_instance_id: int,
) -> list[CapturedPlacement]:
    """Lay the captured pieces out beside the board.

    Grouped by colour on opposite sides, in the loose double row that captured
    pieces actually end up in -- not a tidy line, and frequently on their side.
    """
    pieces = config.pieces
    if rng.random() >= pieces.captured_probability:
        return []

    absent = missing_pieces(grid)
    if not absent:
        return []

    rng.shuffle(absent)
    chosen = absent[: pieces.captured_count.sample(rng)]

    square_size = config.board.square_size
    half_board = BOARD_SIZE / 2.0 * square_size
    offset = pieces.captured_offset.sample(rng) * square_size
    spacing = pieces.captured_spacing.sample(rng) * square_size

    placements = []
    per_side: dict[bool, int] = {True: 0, False: 0}
    for index, class_id in enumerate(chosen):
        white = is_white(class_id)
        slot = per_side[white]
        per_side[white] = slot + 1

        # Two staggered rows either side of the board, as a real pile forms.
        row, column = divmod(slot, 6)
        side = -1.0 if white else 1.0
        x = side * (half_board + offset + row * spacing)
        y = (column - 2.5) * spacing

        placements.append(
            CapturedPlacement(
                instance_id=first_instance_id + index,
                class_id=class_id,
                x=x + rng.uniform(-0.12, 0.12) * square_size,
                y=y + rng.uniform(-0.12, 0.12) * square_size,
                rotation_deg=rng.uniform(0.0, 360.0),
                lying=rng.random() < pieces.captured_lying_probability,
            )
        )
    return placements


def resolve_scene(config: GeneratorConfig, rng: random.Random) -> ResolvedScene:
    scene = config.scene
    square_size = config.board.square_size
    half_board = BOARD_SIZE / 2.0 * square_size

    distractors: list[Distractor] = []
    if rng.random() < scene.distractor_probability:
        for _ in range(scene.distractor_count.sample(rng)):
            # Keep clutter off the board itself so it never hides a square by accident
            # more often than the occlusion we deliberately want from pieces.
            kind = rng.choice(DISTRACTOR_KINDS)
            angle = rng.uniform(0.0, 2 * math.pi)
            radius = rng.uniform(half_board * 1.25, half_board * 2.4)
            distractors.append(
                Distractor(
                    kind=kind,
                    location=[
                        radius * math.cos(angle),
                        radius * math.sin(angle),
                        0.0,
                    ],
                    size=rng.uniform(*DISTRACTOR_SIZES[kind]) * square_size,
                    rotation_deg=rng.uniform(0.0, 360.0),
                    color=_jitter_color(
                        list(rng.choice(DISTRACTOR_COLORS[kind])), rng, 0.25
                    ),
                )
            )

    clock = resolve_clock(scene, rng, square_size=square_size)
    table_size = scene.table_size.sample(rng) * square_size
    return ResolvedScene(
        table_size=table_size,
        table_thickness=scene.table_thickness.sample(rng) * square_size,
        table_color=_jitter_color(rng.choice(scene.table_color), rng, 0.05),
        table_roughness=scene.table_roughness.sample(rng),
        table_texture=choose_table_texture(scene, rng, table_size=table_size),
        clock=clock,
        distractors=distractors,
    )


def choose_table_texture(
    scene: SceneConfig, rng: random.Random, *, table_size: float = 1.0
) -> TableTexture | None:
    """Pick a photographed tabletop for one scene, or None for the flat colour.

    Resolved here rather than on the Blender side so the choice lands in the job
    spec: a scene is then reproducible from its spec alone, even if the texture
    directory later gains or loses files.

    ``texture_scale`` is stated as repeats across the whole table and converted to
    Blender's per-unit mapping scale here, because the tabletop is 20-45 squares
    wide: left per-unit, a scale of 1 tiles the map thirty times and reads as fine
    fabric rather than as a wooden tabletop.
    """
    if scene.texture_dir is None or rng.random() >= scene.texture_probability:
        return None
    available = texture_sets(scene.texture_dir)
    if not available:
        # A missing or empty directory is not an error: it is the no-assets path,
        # and the flat colour still renders a valid scene.
        return None

    chosen = rng.choice(available)
    tint = scene.texture_tint.sample(rng)
    repeats = scene.texture_scale.sample(rng)
    return TableTexture(
        slug=chosen["slug"],
        maps=chosen["maps"],
        scale=repeats / max(table_size, 1e-6),
        rotation_deg=scene.texture_rotation_deg.sample(rng),
        tint=[tint, tint, tint],
        roughness_shift=scene.texture_roughness_shift.sample(rng),
        hue_shift=scene.texture_hue_shift.sample(rng),
        saturation=scene.texture_saturation.sample(rng),
        brightness=scene.texture_brightness.sample(rng),
    )


def resolve_render(config: GeneratorConfig) -> ResolvedRender:
    render = config.render
    return ResolvedRender(
        engine=render.engine,
        samples=render.samples,
        resolution=list(render.resolution),
        use_gpu=render.use_gpu,
        compute_device_type=render.compute_device_type,
        denoise=render.denoise,
        image_format=render.image_format,
        jpeg_quality=render.jpeg_quality,
        render_id_pass=render.render_id_pass,
    )


def build_job(
    config: GeneratorConfig,
    *,
    sample_id: str,
    grid: Grid,
    seed: int,
    image_path: Path,
    labels_path: Path,
    id_pass_path: Path | None,
) -> JobSpec:
    """Resolve one complete job spec.

    Each aspect draws from its own derived RNG so that, for example, changing the
    lighting config does not shift the camera poses of an otherwise identical run.
    """
    return JobSpec(
        id=sample_id,
        seed=seed,
        fen=grid_to_fen(grid),
        grid=grid,
        image_path=str(image_path),
        labels_path=str(labels_path),
        id_pass_path=str(id_pass_path) if id_pass_path is not None else None,
        render=resolve_render(config),
        camera=resolve_camera(config, derive_rng(seed, "camera")),
        lighting=resolve_lighting(config, derive_rng(seed, "lighting")),
        board=resolve_board(config, derive_rng(seed, "board")),
        pieces=resolve_pieces(config, grid, derive_rng(seed, "pieces")),
        scene=resolve_scene(config, derive_rng(seed, "scene")),
    )


def placement_world_xy(
    placement: PiecePlacement, square_size: float
) -> tuple[float, float]:
    """World x/y of a placed piece, including its jitter.

    Shared with the Blender side's expectations, and used by tests to assert that
    jitter never pushes a piece outside its own square.
    """
    u = placement.file_index + 0.5 + placement.offset_u
    v = placement.rank_index + 0.5 + placement.offset_v
    return board_to_world(u, v, square_size)
