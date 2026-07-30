"""Turn a config plus a seed into a fully resolved :class:`JobSpec`.

This is where domain randomisation actually happens. It lives on the project side --
not inside Blender -- so it can be unit-tested, and so that a job spec is a complete,
inspectable record of the scene that was rendered.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from chesssight.data.fen import (
    BOARD_SIZE,
    Grid,
    grid_to_fen,
    is_white,
    iter_occupied,
)
from chesssight.synth import profiles
from chesssight.synth.config import GeneratorConfig
from chesssight.synth.jobspec import (
    CapturedPlacement,
    Distractor,
    JobSpec,
    PiecePlacement,
    ResolvedBoard,
    ResolvedCamera,
    ResolvedDepthOfField,
    ResolvedLamp,
    ResolvedLighting,
    ResolvedPieces,
    ResolvedRender,
    ResolvedScene,
    board_to_world,
)
from chesssight.synth.seeds import derive_rng

HDRI_SUFFIXES = (".hdr", ".exr")

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


def resolve_board(config: GeneratorConfig, rng: random.Random) -> ResolvedBoard:
    board = config.board
    jitter = board.color_jitter.max
    return ResolvedBoard(
        square_size=board.square_size,
        thickness=board.thickness.sample(rng),
        border_width=board.border_width.sample(rng),
        light_color=_jitter_color(rng.choice(board.light_square_color), rng, jitter),
        dark_color=_jitter_color(rng.choice(board.dark_square_color), rng, jitter),
        roughness=board.roughness.sample(rng),
    )


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

    return ResolvedPieces(
        provider=pieces.provider,
        asset_manifest=(
            str(Path(pieces.asset_manifest).expanduser())
            if pieces.asset_manifest
            else None
        ),
        height_scale=pieces.height_scale.sample(rng),
        radius_scale=pieces.radius_scale.sample(rng),
        bevel_width=pieces.bevel_width.sample(rng),
        lathe_segments=pieces.lathe_segments.sample(rng),
        white_color=_jitter_color(pieces.white_color, rng, 0.05),
        black_color=_jitter_color(pieces.black_color, rng, 0.03),
        roughness=pieces.roughness.sample(rng),
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
            angle = rng.uniform(0.0, 2 * math.pi)
            radius = rng.uniform(half_board * 1.25, half_board * 2.4)
            distractors.append(
                Distractor(
                    kind=rng.choice(["cup", "block", "clock"]),
                    location=[
                        radius * math.cos(angle),
                        radius * math.sin(angle),
                        0.0,
                    ],
                    size=rng.uniform(0.5, 1.6) * square_size,
                    rotation_deg=rng.uniform(0.0, 360.0),
                    color=[rng.uniform(0.05, 0.9) for _ in range(3)],
                )
            )

    return ResolvedScene(
        table_size=scene.table_size.sample(rng) * square_size,
        table_color=_jitter_color(rng.choice(scene.table_color), rng, 0.05),
        table_roughness=scene.table_roughness.sample(rng),
        distractors=distractors,
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
