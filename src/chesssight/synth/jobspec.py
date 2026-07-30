"""The resolved job specification handed to Blender.

A :class:`JobSpec` contains **no ranges and no probabilities** -- every random choice
has already been made on the project side by :mod:`chesssight.synth.randomize`. The
Blender script is therefore a pure function of its job spec, which has two payoffs:
the randomisation logic is unit-testable without Blender, and re-rendering a sample
is a matter of replaying one JSON object.

World convention
----------------
The board is centred on the world origin with its playing surface at ``z = 0``.
Board-plane coordinates ``(u, v)`` from :mod:`chesssight.data.geometry` map to world
``x = (u - 4) * square_size`` and ``y = (4 - v) * square_size``, so rank 8 is at
``+y`` and the a-file is at ``-x``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from chesssight.data.fen import BOARD_SIZE, Grid

Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
RGB = Annotated[list[float], Field(min_length=3, max_length=3)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def board_to_world(u: float, v: float, square_size: float) -> tuple[float, float]:
    """Convert board-plane coordinates to world x/y."""
    half = BOARD_SIZE / 2.0
    return (u - half) * square_size, (half - v) * square_size


class ResolvedRender(StrictModel):
    engine: Literal["CYCLES", "BLENDER_EEVEE"]
    samples: int = Field(gt=0)
    resolution: Annotated[list[int], Field(min_length=2, max_length=2)]
    use_gpu: bool
    compute_device_type: Literal["OPTIX", "CUDA", "NONE"]
    denoise: bool
    image_format: Literal["JPEG", "PNG"]
    jpeg_quality: int = Field(ge=1, le=100)
    render_id_pass: bool


class ResolvedDepthOfField(StrictModel):
    enabled: bool
    f_stop: float = Field(gt=0)
    focus_distance: float = Field(gt=0)


class ResolvedCamera(StrictModel):
    location: Vec3
    look_at: Vec3
    roll_deg: float
    focal_mm: float = Field(gt=0)
    sensor_width_mm: float = Field(gt=0)
    depth_of_field: ResolvedDepthOfField


class ResolvedLamp(StrictModel):
    location: Vec3
    energy: float = Field(ge=0)
    size: float = Field(gt=0)
    color: RGB


class ResolvedLighting(StrictModel):
    hdri_path: str | None
    hdri_rotation_deg: float
    world_strength: float = Field(ge=0)
    world_color: RGB
    #: Position the sun is placed at; it is aimed at the origin. Only the direction
    #: matters -- a sun lamp's irradiance does not fall off with distance.
    sun_location: Vec3
    sun_energy: float = Field(ge=0)
    sun_angle_deg: float = Field(ge=0)
    sun_color: RGB
    lamps: list[ResolvedLamp]


class ResolvedBoard(StrictModel):
    square_size: float = Field(gt=0)
    thickness: float = Field(gt=0)
    border_width: float = Field(ge=0)
    light_color: RGB
    dark_color: RGB
    roughness: float = Field(ge=0, le=1)


class PiecePlacement(StrictModel):
    """One piece, already assigned its square and its jitter."""

    #: Unique within a job; also the integer encoded in the id pass.
    instance_id: int = Field(ge=1)
    class_id: int = Field(ge=1, le=12)
    rank_index: int = Field(ge=0, lt=BOARD_SIZE)
    file_index: int = Field(ge=0, lt=BOARD_SIZE)
    #: Offset from the square centre, as a fraction of a square.
    offset_u: float
    offset_v: float
    rotation_deg: float
    #: Yaw for a knight: measured from "facing the opponent", not from world axes.
    knight_yaw_deg: float = 0.0
    tilt_deg: float
    tipped: bool


class CapturedPlacement(StrictModel):
    """A piece standing beside the board, taken out of play.

    Positioned in world coordinates rather than by square, because that is exactly
    what it no longer has.
    """

    instance_id: int = Field(ge=1)
    class_id: int = Field(ge=1, le=12)
    #: World x/y on the table surface.
    x: float
    y: float
    rotation_deg: float
    #: Captured pieces get knocked over far more often than pieces in play.
    lying: bool = False


class ResolvedPieces(StrictModel):
    provider: str
    #: Manifest for an external chess set. When set, the Blender side registers an
    #: ``AssetLibraryProvider`` under the manifest's name before building pieces.
    asset_manifest: str | None = None
    height_scale: float = Field(gt=0)
    radius_scale: float = Field(gt=0)
    #: Silhouette warp for procedural sets: positive widens the base and narrows the
    #: top. Bounded so ``1 + |taper|`` cannot push a piece outside its square.
    taper: float = Field(default=0.0, gt=-1.0, lt=1.0)
    bevel_width: float = Field(ge=0)
    lathe_segments: int = Field(ge=3)
    white_color: RGB
    black_color: RGB
    roughness: float = Field(ge=0, le=1)
    placements: list[PiecePlacement]
    #: Pieces beside the board. They appear in the image and in ``pieces``, but
    #: never in the grid.
    captured: list[CapturedPlacement] = Field(default_factory=list)


class Distractor(StrictModel):
    kind: Literal["cup", "block", "clock"]
    location: Vec3
    size: float = Field(gt=0)
    rotation_deg: float
    color: RGB


class ResolvedScene(StrictModel):
    table_size: float = Field(gt=0)
    table_thickness: float = Field(gt=0)
    table_color: RGB
    table_roughness: float = Field(ge=0, le=1)
    distractors: list[Distractor]


class JobSpec(StrictModel):
    """Everything Blender needs to render one sample and emit its raw labels."""

    id: str
    seed: int
    fen: str
    grid: Grid
    #: Absolute paths. Blender is snap-confined here, so these must be under $HOME.
    image_path: str
    labels_path: str
    id_pass_path: str | None

    render: ResolvedRender
    camera: ResolvedCamera
    lighting: ResolvedLighting
    board: ResolvedBoard
    pieces: ResolvedPieces
    scene: ResolvedScene
