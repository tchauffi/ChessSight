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


class MaterialStyle(StrictModel):
    """What a surface is made of, and the parameters its procedural needs.

    Resolved project-side like everything else, so the Blender layer only ever
    selects a node graph -- it never draws a random number.
    """

    kind: Literal["plastic", "wood", "marble", "plain", "textured"]
    scale: float = Field(default=3.0, gt=0)
    distortion: float = Field(default=4.0, ge=0)
    #: How far the figure departs from the body colour: the growth ring in wood, the
    #: vein in marble. A *relative* strength, not an absolute second colour, because
    #: one style is applied to both a near-white and a near-black set. An absolute
    #: accent tuned on the white pieces is ten times lighter than a black piece's
    #: base and turns it into a zebra; a relative one darkens or lightens each side
    #: by the same proportion of its own colour.
    contrast: float = Field(default=0.2, ge=0.0, le=0.9)
    #: Marble only: what fraction of the surface the vein occupies. Real stone is
    #: roughly nine parts body to one part vein; above about 0.25 it stops reading
    #: as rock and starts reading as camouflage.
    vein_width: float = Field(default=0.14, gt=0.0, le=0.4)
    #: Which way the figure runs, in degrees. Randomised so a dataset does not have
    #: every vein and every grain line at the same angle.
    rotation_deg: Vec3 = [0.0, 0.0, 0.0]
    #: Colour adjustment applied to a *photographed* texture, so that the two usable
    #: veneers cover a range of timbers rather than appearing identically in every
    #: wooden set. Hue is an offset from no-change, not an absolute.
    hue_shift: float = Field(default=0.0, ge=-0.5, le=0.5)
    saturation: float = Field(default=1.0, ge=0.0, le=2.0)
    brightness: float = Field(default=1.0, ge=0.0, le=2.0)


class ResolvedBoard(StrictModel):
    square_size: float = Field(gt=0)
    thickness: float = Field(gt=0)
    border_width: float = Field(ge=0)
    light_color: RGB
    dark_color: RGB
    roughness: float = Field(ge=0, le=1)
    #: None keeps the plain flat-colour squares, which is also the default.
    material: MaterialStyle | None = None
    #: Photographed surface for the squares, when the drawn style is `textured`.
    #: Flat-projected: a board is a flat quad, so unlike the pieces it needs no
    #: triplanar trick.
    maps: dict[str, str] | None = None


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
    material: MaterialStyle = MaterialStyle(kind="plastic")
    #: Photographed veneer for each side, when the drawn style is wood and suitable
    #: textures exist. Box-projected, because a lathed piece carries no UV map.
    #: None falls back to the procedural grain.
    light_maps: dict[str, str] | None = None
    dark_maps: dict[str, str] | None = None
    placements: list[PiecePlacement]
    #: Pieces beside the board. They appear in the image and in ``pieces``, but
    #: never in the grid.
    captured: list[CapturedPlacement] = Field(default_factory=list)


class Distractor(StrictModel):
    kind: Literal["cup", "block", "notepad", "pen", "phone", "glass"]
    location: Vec3
    size: float = Field(gt=0)
    rotation_deg: float
    color: RGB


class TableTexture(StrictModel):
    """A photographed surface for the tabletop, resolved to absolute paths."""

    slug: str
    #: Map name -> absolute path. Every map the material needs is present, or the
    #: whole texture is dropped: a diffuse map wired up without its roughness map
    #: renders shinier than intended rather than failing visibly.
    maps: dict[str, str]
    scale: float = Field(gt=0)
    rotation_deg: float
    tint: RGB
    roughness_shift: float
    hue_shift: float = Field(default=0.0, ge=-0.5, le=0.5)
    saturation: float = Field(default=1.0, ge=0.0, le=2.0)
    brightness: float = Field(default=1.0, ge=0.0, le=2.0)


class ResolvedClock(StrictModel):
    """A chess clock standing beside the board.

    Sized from real ones: an analogue case is about 200x125x58 mm and a digital one
    about 166x114x65 mm, so against a 50 mm square a clock is roughly four squares
    wide. Getting that wrong matters more than it sounds -- a clock rendered at
    piece scale is a different object as far as a detector is concerned.
    """

    kind: Literal["analogue", "digital"]
    #: World x/y on the table surface, beside the board.
    x: float
    y: float
    #: Facing, in degrees. A clock sits square to the board edge it stands beside.
    rotation_deg: float
    #: Overall width in squares. Everything else is a ratio of it, so one number
    #: sets the scale and the rest set the shape.
    width: float = Field(gt=0)
    depth_ratio: float = Field(gt=0)
    height_ratio: float = Field(gt=0)
    #: Analogue: dial radius and how far apart the pair sits, both as a fraction of
    #: width. Digital: display panel size and inset.
    face_ratio: float = Field(gt=0)
    face_offset: float = Field(gt=0)
    #: Plungers on an analogue case, buttons on a digital one.
    knob_ratio: float = Field(gt=0)
    knob_count: int = Field(ge=2, le=3)
    #: Digital only: front height as a fraction of back height. 1.0 is a plain box,
    #: lower is a more steeply raked wedge.
    slope: float = Field(gt=0, le=1.0)
    #: A raised rim around an analogue dial, as most cases have.
    bezel: bool = False
    body_color: RGB
    face_color: RGB
    button_color: RGB


class ResolvedScene(StrictModel):
    table_size: float = Field(gt=0)
    table_thickness: float = Field(gt=0)
    table_color: RGB
    table_roughness: float = Field(ge=0, le=1)
    #: None means the flat colour above is used, which is also the no-assets path.
    table_texture: TableTexture | None = None
    #: Players use a clock in most serious games, so it is its own element rather
    #: than one more piece of random clutter -- it has a characteristic size and a
    #: characteristic place, beside the board and square to it.
    clock: ResolvedClock | None = None
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
