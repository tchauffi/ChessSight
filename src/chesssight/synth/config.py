"""Generator configuration.

Every randomised quantity is expressed here as an explicit range, so the strength of
domain randomisation is a config decision rather than something buried in the
renderer. :mod:`chesssight.synth.randomize` turns a config plus a seed into a fully
resolved scene description; the Blender side only ever sees resolved values.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Engine = Literal["CYCLES", "BLENDER_EEVEE"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FloatRange(StrictModel):
    """Inclusive float range, sampled uniformly."""

    min: float
    max: float

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.min > self.max:
            raise ValueError(f"range min {self.min} exceeds max {self.max}")
        return self

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.min, self.max)


class IntRange(StrictModel):
    """Inclusive integer range."""

    min: int
    max: int

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.min > self.max:
            raise ValueError(f"range min {self.min} exceeds max {self.max}")
        return self

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.min, self.max)


Probability = Annotated[float, Field(ge=0.0, le=1.0)]
RGB = Annotated[list[float], Field(min_length=3, max_length=3)]


class RenderConfig(StrictModel):
    """Engine and output settings."""

    engine: Engine = "BLENDER_EEVEE"
    samples: int = Field(default=64, gt=0)
    resolution: Annotated[list[int], Field(min_length=2, max_length=2)] = [512, 512]
    use_gpu: bool = True
    #: Cycles only. OPTIX is fastest on RTX hardware; CUDA is the fallback.
    compute_device_type: Literal["OPTIX", "CUDA", "NONE"] = "OPTIX"
    denoise: bool = True
    image_format: Literal["JPEG", "PNG"] = "JPEG"
    jpeg_quality: int = Field(default=92, ge=1, le=100)
    #: Rendered separately in EEVEE at 1 sample to recover instance masks.
    render_id_pass: bool = True


class CameraConfig(StrictModel):
    """Camera pose and lens randomisation, in board units (1 unit = one square)."""

    azimuth_deg: FloatRange = FloatRange(min=0.0, max=360.0)
    #: Low elevations mimic hand-held phone shots and broadcast cameras set up across
    #: the table -- the grazing views the detector was worst on. The floor is a
    #: *usefulness* limit, not a correctness one: labels stay exact all the way down
    #: to 4 degrees (measured reprojection error 0.0002 px, no rejected samples, piece
    #: boxes barely shrink because pieces lose depth but keep height). It sits at 8
    #: because below that a rank is under 9 px deep and the near rank hides the far
    #: one outright, so the extra samples add occlusion rather than new appearance.
    elevation_deg: FloatRange = FloatRange(min=8.0, max=75.0)
    #: How much room to leave around the board, as a multiple of the distance at
    #: which it exactly fills the frame. Distance is *derived* from this, the focal
    #: length and the camera's own pose rather than drawn independently -- sampling
    #: distance separately means a long lens at a short distance, and most of the
    #: board off-screen. See :func:`~chesssight.synth.randomize.framing_distance`.
    #: Values below 1.0 deliberately crop the board, which real photographs do
    #: constantly and a model must cope with.
    framing_margin: FloatRange = FloatRange(min=0.95, max=1.9)
    #: Hard clamp on the derived distance, so an extreme lens cannot put the camera
    #: inside the table or out at infinity.
    distance: FloatRange = FloatRange(min=6.0, max=40.0)
    roll_deg: FloatRange = FloatRange(min=-6.0, max=6.0)
    focal_mm: FloatRange = FloatRange(min=24.0, max=85.0)
    sensor_width_mm: float = Field(default=36.0, gt=0)
    #: Jitter of the point the camera aims at, so the board is not always centred.
    target_jitter: FloatRange = FloatRange(min=-1.2, max=1.2)
    dof_probability: Probability = 0.35
    f_stop: FloatRange = FloatRange(min=1.8, max=11.0)


class LightingConfig(StrictModel):
    """World and lamp randomisation."""

    hdri_dir: Path | None = None
    #: With a directory configured, every scene defaults to an HDRI environment: a
    #: real room behind the table is the single biggest realism win over a flat
    #: colour, and the flat-colour rig remains only as the no-assets fallback. Dial
    #: this down to mix the procedural world back in.
    hdri_probability: Probability = 1.0
    hdri_rotation_deg: FloatRange = FloatRange(min=0.0, max=360.0)
    #: Multiplier on a flat-colour world, which carries no absolute scale.
    world_strength: FloatRange = FloatRange(min=0.3, max=2.5)
    #: Multiplier on an *HDRI*, kept much narrower. An environment map already
    #: encodes absolute radiance, so 1.0 is roughly correct exposure and the wide
    #: flat-colour range simply blows the image out.
    hdri_strength: FloatRange = FloatRange(min=0.5, max=1.6)
    world_color_temperature: FloatRange = FloatRange(min=3200.0, max=7500.0)
    #: Sun energy is an irradiance in W/m^2, so unlike point/area lamps it does not
    #: depend on how large the scene is in Blender units. That makes it the reliable
    #: key light here, where one unit is one chess square rather than one metre.
    sun_energy: FloatRange = FloatRange(min=1.0, max=6.0)
    #: Angular diameter of the sun disc; larger means softer shadows.
    sun_angle_deg: FloatRange = FloatRange(min=0.5, max=8.0)
    sun_elevation_deg: FloatRange = FloatRange(min=25.0, max=85.0)
    lamp_count: IntRange = IntRange(min=0, max=2)
    #: Area-lamp power falls off with distance squared, and this scene is ~8 units
    #: across, so these are far larger than the values a metre-scale scene needs.
    lamp_energy: FloatRange = FloatRange(min=1500.0, max=12000.0)
    lamp_size: FloatRange = FloatRange(min=1.0, max=8.0)
    lamp_distance: FloatRange = FloatRange(min=8.0, max=18.0)
    lamp_elevation_deg: FloatRange = FloatRange(min=25.0, max=85.0)


class BoardConfig(StrictModel):
    """Board geometry and material randomisation."""

    square_size: float = Field(default=1.0, gt=0)
    thickness: FloatRange = FloatRange(min=0.15, max=0.5)
    border_width: FloatRange = FloatRange(min=0.0, max=1.0)
    #: How light the border is, 0 being the dark squares' colour and 1 the light
    #: squares'. Sampled across the whole range because both conventions are
    #: common: club boards often have a frame darker than the dark squares, and
    #: tournament boards a white one carrying printed coordinates. Rendering only
    #: the dark convention -- which this generator did until now -- teaches a
    #: corner model that the playing area ends wherever the surface darkens.
    border_tone: FloatRange = FloatRange(min=0.0, max=1.0)
    light_square_color: list[RGB] = [
        [0.72, 0.62, 0.46],
        [0.85, 0.82, 0.75],
        [0.66, 0.57, 0.40],
    ]
    #: Linear-space values, so they look considerably darker than the sRGB numbers
    #: you would pick in a colour picker. Real dark squares sit around 0.10-0.20.
    dark_square_color: list[RGB] = [
        [0.16, 0.09, 0.05],
        [0.11, 0.16, 0.11],
        [0.09, 0.09, 0.12],
    ]
    roughness: FloatRange = FloatRange(min=0.15, max=0.75)
    #: Palette variation as a *relative* fraction, so dark and light squares vary by
    #: the same proportion and keep their hue. An absolute offset would leave light
    #: squares almost unchanged while randomising dark ones into a different colour.
    color_jitter: FloatRange = FloatRange(min=-0.18, max=0.18)
    #: Least brightness difference allowed between a square and the pieces standing
    #: on it. Board and piece palettes are drawn independently, so without a floor
    #: they can land almost on top of each other -- measured on the defaults, black
    #: pieces came within 0.030 luminance of the darkest squares. Low contrast is
    #: legitimately hard and worth training on; *no* contrast is a label describing
    #: something the image does not show.
    min_piece_contrast: float = Field(default=0.14, ge=0.0, le=0.5)
    #: What the board is made of, drawn per scene by weight. Boards are veneered
    #: wood or inlaid stone far more often than they are flat colour, and against a
    #: photographed tabletop a flat-colour board became the most obviously rendered
    #: thing in the frame.
    #: `textured` draws a photograph from the texture library; `plastic` is the
    #: moulded vinyl of a roll-up tournament board. Keeping both, rather than
    #: converging on the most realistic one, is the point: this dataset is meant to
    #: be hard, and a detector that has only seen veneer will not read a plastic
    #: board across a hall.
    material_styles: dict[str, float] = {
        "textured": 0.30,
        "wood": 0.25,
        "marble": 0.15,
        "plastic": 0.30,
    }
    #: Grain scale for the squares, in repeats per square. Randomised because a
    #: fixed figure size is itself a constant a detector can key on.
    grain_scale: FloatRange = FloatRange(min=1.5, max=6.0)


class PieceSetChoice(StrictModel):
    """One chess set the generator may draw from, and how often to draw it."""

    provider: str
    #: Path to an external chess-set manifest (see `chesssight assets --help`).
    #: When set, `provider` should be the manifest's own name.
    asset_manifest: Path | None = None
    #: Relative weight, normalised against the other entries.
    weight: float = Field(default=1.0, gt=0.0)


class PiecesConfig(StrictModel):
    """Procedural piece style and placement randomisation."""

    provider: str = "procedural"
    #: Path to an external chess-set manifest (see `chesssight assets --help`).
    #: When set, `provider` should be the manifest's own name.
    asset_manifest: Path | None = None
    #: Several sets, one drawn per scene, instead of the single `provider` above.
    #: Real chess sets differ in silhouette far more than in size, and a dataset
    #: rendered from one set lets a detector key on that set's outline -- which is
    #: exactly the cue that does not transfer to a photograph of someone else's
    #: board. When non-empty this takes precedence over `provider`.
    sets: list[PieceSetChoice] = Field(default_factory=list)
    #: Multiplies every profile height, applied once per scene so a set is coherent.
    height_scale: FloatRange = FloatRange(min=0.85, max=1.15)
    radius_scale: FloatRange = FloatRange(min=0.85, max=1.15)
    #: Redistributes radius along each piece -- positive widens the base and narrows
    #: the top, negative the reverse -- so the procedural set spans squat through
    #: slender variants rather than a single fixed outline. Procedural sets only;
    #: imported geometry is used as authored.
    taper: FloatRange = FloatRange(min=-0.12, max=0.12)
    bevel_width: FloatRange = FloatRange(min=0.002, max=0.012)
    lathe_segments: IntRange = IntRange(min=16, max=48)
    white_color: RGB = [0.90, 0.87, 0.80]
    black_color: RGB = [0.07, 0.06, 0.06]
    roughness: FloatRange = FloatRange(min=0.1, max=0.7)
    #: What the pieces are made of, drawn per scene by weight. Real sets are turned
    #: boxwood, moulded plastic or cut stone, and the three look nothing alike under
    #: the same light -- plastic reads as a smooth coloured solid, wood shows rings
    #: around the axis it was turned on, marble shows veins running through it.
    #: Weights, not a single choice, so one dataset carries all three.
    #: Both a moulded-plastic look and a photographed one are kept deliberately.
    #: The target footage is full of cheap plastic tournament sets, and the mix of
    #: a flat moulded surface against a photographed timber is exactly the range
    #: the detector has to cope with.
    material_styles: dict[str, float] = {
        "plastic": 0.40,
        "wood": 0.35,
        "marble": 0.25,
    }
    #: Veneer textures used when the wood style is drawn, box-projected onto the
    #: pieces. Curated separately from the table's: a floor texture's plank joins
    #: wrap a piece as hoops and make it look coopered, so only seamless raw-timber
    #: veneers belong here. Empty, or a missing texture, falls back to procedural
    #: grain -- so the pipeline still runs with no downloaded assets.
    veneers: dict[str, str] = {
        "light": "oak_veneer_01",
        "dark": "rosewood_veneer1",
    }
    #: Per-scene colour shift on the veneers. Two textures cannot cover the range of
    #: timbers real sets are made from, but a hue rotation and a brightness change
    #: turn oak and rosewood into something nearer walnut, cherry or mahogany. Kept
    #: narrow: a large hue rotation makes wood green, which is domain *noise*.
    #: Applied to both sides together, since a set is finished as a pair.
    veneer_hue_shift: FloatRange = FloatRange(min=-0.06, max=0.06)
    veneer_saturation: FloatRange = FloatRange(min=0.75, max=1.25)
    veneer_brightness: FloatRange = FloatRange(min=0.8, max=1.25)
    #: Per-piece placement noise, as a fraction of a square.
    position_jitter: FloatRange = FloatRange(min=-0.16, max=0.16)
    #: Full spin for the radially symmetric pieces -- nobody aligns a pawn.
    rotation_jitter_deg: FloatRange = FloatRange(min=-180.0, max=180.0)
    #: Knights are the one piece whose facing is visible, and on a real board they
    #: point at the opponent give or take a nudge. Spinning them freely would put
    #: half the set facing backwards, which is not a thing that happens.
    knight_facing_jitter_deg: FloatRange = FloatRange(min=-30.0, max=30.0)
    tilt_jitter_deg: FloatRange = FloatRange(min=-3.0, max=3.0)
    #: Occasionally lay a piece on its side, as happens in real photos.
    tipped_probability: Probability = 0.02

    #: Chance that a scene shows the captured pieces beside the board, as a real
    #: game in progress does. The captured set is derived from what is *missing*
    #: from the position, so it is always consistent with the board.
    captured_probability: Probability = 0.4
    #: How many of the missing pieces to show. Real games leave them in a loose
    #: cluster rather than a tidy row, so this is a count, not a layout.
    captured_count: IntRange = IntRange(min=1, max=12)
    #: Distance from the board edge to the captured cluster, in squares.
    captured_offset: FloatRange = FloatRange(min=0.9, max=2.4)
    captured_spacing: FloatRange = FloatRange(min=0.55, max=0.85)
    #: Pieces off the board are handled casually and often end up on their side.
    captured_lying_probability: Probability = 0.25

    @model_validator(mode="after")
    def _check_pieces_fit_their_square(self) -> Self:
        """Reject scaling that would let the widest piece overflow its square.

        Enlarging and tapering both push radius outward, and they multiply, so
        neither range is safe to judge on its own. The consequence of getting it
        wrong is not a crash but touching pieces -- a silently wrong dataset -- so
        it is worth failing at config-load time.
        """
        from chesssight.synth.profiles import MAX_RADII

        widest = max(MAX_RADII.values())
        widest_taper = max(abs(self.taper.min), abs(self.taper.max))
        worst = widest * self.radius_scale.max * (1.0 + widest_taper)
        if worst >= 0.5:
            raise ValueError(
                f"radius_scale.max {self.radius_scale.max} with taper "
                f"+-{widest_taper} takes the widest piece to {worst:.3f} squares, "
                f"which overflows its square (limit 0.5)"
            )
        return self


class SceneConfig(StrictModel):
    """Surroundings: table, backdrop and clutter."""

    table_size: FloatRange = FloatRange(min=20.0, max=45.0)
    #: Slab thickness of the tabletop, in squares. The table used to be an infinite
    #: plane, and an edgeless table filling the frame to the horizon is one of the
    #: things that most loudly says "render" -- every real photo shows the table
    #: *ending* somewhere.
    table_thickness: FloatRange = FloatRange(min=0.4, max=1.0)
    table_color: list[RGB] = [
        [0.25, 0.16, 0.10],
        [0.55, 0.52, 0.48],
        [0.10, 0.10, 0.12],
    ]
    table_roughness: FloatRange = FloatRange(min=0.2, max=0.9)
    #: A directory of Poly Haven PBR maps (fetch with `chesssight assets textures`).
    #: With one configured the table is a photographed surface rather than a flat
    #: colour, which is the single largest untextured area in the frame. Missing
    #: directory falls back to the flat colour, so the pipeline still runs with no
    #: external assets.
    texture_dir: Path | None = None
    texture_probability: Probability = 0.85
    #: How many times the map repeats across the *whole table*, not per unit. Stated
    #: this way because the table is 20-45 squares across, so a per-unit scale of 1
    #: tiles the map thirty times and turns oak into fine fabric. Randomised per
    #: scene: one fixed grain size is itself a constant to memorise.
    texture_scale: FloatRange = FloatRange(min=1.5, max=5.0)
    texture_rotation_deg: FloatRange = FloatRange(min=0.0, max=360.0)
    #: Multiplied into the diffuse map, so twelve downloads cover more than twelve
    #: tables. Kept near white -- a strong tint reads as coloured light, not as a
    #: different wood.
    texture_tint: FloatRange = FloatRange(min=0.75, max=1.0)
    texture_roughness_shift: FloatRange = FloatRange(min=-0.15, max=0.15)
    #: Same idea as the veneers: fourteen table textures go further when each can
    #: appear as a lighter or darker version of itself. Deliberately *tighter* than
    #: the veneers' range, and mostly brightness rather than hue -- the table is a
    #: large flat area, and a rotation the pieces tolerate turns a whole tabletop
    #: mustard-yellow. A table nobody owns is domain noise, not randomisation: it
    #: spends a sample teaching the detector a colour it will never meet.
    texture_hue_shift: FloatRange = FloatRange(min=-0.02, max=0.02)
    texture_saturation: FloatRange = FloatRange(min=0.85, max=1.15)
    texture_brightness: FloatRange = FloatRange(min=0.8, max=1.2)
    #: Most serious games are played with a clock, and the detector has already been
    #: seen to mistake one for a piece -- in test footage it boxed a clock's
    #: plungers as a bishop. Training on scenes that contain one attacks that
    #: directly, and a clock is labelled as scenery, never as a piece.
    clock_probability: Probability = 0.45
    #: Analogue against digital. Both are common; the two look nothing alike, so a
    #: model that has only seen one will not recognise the other as furniture.
    clock_digital_probability: Probability = 0.5
    #: Overall width in squares. Real clocks run 165-220 mm against a ~50 mm square.
    clock_width: FloatRange = FloatRange(min=3.0, max=4.6)
    #: How far the clock stands from the board edge, in squares.
    clock_offset: FloatRange = FloatRange(min=0.8, max=2.6)
    distractor_count: IntRange = IntRange(min=0, max=3)
    distractor_probability: Probability = 0.3


class PositionsConfig(StrictModel):
    """Where board positions come from."""

    pgn_paths: list[Path] = Field(default_factory=list)
    max_games: int = Field(default=5_000, gt=0)
    plies_per_game: int = Field(default=3, gt=0)
    skip_opening_plies: int = Field(default=6, ge=0)
    #: Relative weights; ``pgn`` is ignored when no PGN paths are configured.
    weight_pgn: float = Field(default=0.8, ge=0)
    weight_random: float = Field(default=0.2, ge=0)
    random_min_pieces: int = Field(default=2, ge=2, le=64)
    random_max_pieces: int = Field(default=32, ge=2, le=64)

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        if self.weight_pgn <= 0 and self.weight_random <= 0:
            raise ValueError("at least one position-source weight must be positive")
        if self.random_min_pieces > self.random_max_pieces:
            raise ValueError("random_min_pieces exceeds random_max_pieces")
        if not self.pgn_paths and self.weight_random <= 0:
            raise ValueError(
                "no pgn_paths configured, so weight_random must be positive"
            )
        return self


class OutputConfig(StrictModel):
    """Where the dataset lands."""

    #: Defaults outside the repo: datasets are large and must never be committed.
    root: Path = Path("~/datasets/chesssight")
    run_name: str = "run"
    split: Literal["train", "val", "test"] = "train"

    def run_dir(self) -> Path:
        return Path(self.root).expanduser() / self.run_name


class GeneratorConfig(StrictModel):
    """The whole generator configuration."""

    count: int = Field(default=1000, gt=0)
    master_seed: int = 20260729
    render: RenderConfig = RenderConfig()
    camera: CameraConfig = CameraConfig()
    lighting: LightingConfig = LightingConfig()
    board: BoardConfig = BoardConfig()
    pieces: PiecesConfig = PiecesConfig()
    scene: SceneConfig = SceneConfig()
    positions: PositionsConfig = PositionsConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> GeneratorConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
