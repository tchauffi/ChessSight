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
    #: Low elevations mimic hand-held phone shots taken from across the table.
    elevation_deg: FloatRange = FloatRange(min=18.0, max=75.0)
    #: How much room to leave around the board, as a multiple of the distance at
    #: which it exactly fills the frame. Distance is *derived* from this and the
    #: focal length rather than drawn independently -- sampling the two separately
    #: means a long lens at a short distance, and most of the board off-screen.
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
    hdri_probability: Probability = 0.5
    hdri_rotation_deg: FloatRange = FloatRange(min=0.0, max=360.0)
    world_strength: FloatRange = FloatRange(min=0.3, max=2.5)
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


class PiecesConfig(StrictModel):
    """Procedural piece style and placement randomisation."""

    provider: str = "procedural"
    #: Path to an external chess-set manifest (see `chesssight assets --help`).
    #: When set, `provider` should be the manifest's own name.
    asset_manifest: Path | None = None
    #: Multiplies every profile height, applied once per scene so a set is coherent.
    height_scale: FloatRange = FloatRange(min=0.85, max=1.15)
    radius_scale: FloatRange = FloatRange(min=0.85, max=1.15)
    bevel_width: FloatRange = FloatRange(min=0.002, max=0.012)
    lathe_segments: IntRange = IntRange(min=16, max=48)
    white_color: RGB = [0.90, 0.87, 0.80]
    black_color: RGB = [0.07, 0.06, 0.06]
    roughness: FloatRange = FloatRange(min=0.1, max=0.7)
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


class SceneConfig(StrictModel):
    """Surroundings: table, backdrop and clutter."""

    table_size: FloatRange = FloatRange(min=20.0, max=45.0)
    table_color: list[RGB] = [
        [0.25, 0.16, 0.10],
        [0.55, 0.52, 0.48],
        [0.10, 0.10, 0.12],
    ]
    table_roughness: FloatRange = FloatRange(min=0.2, max=0.9)
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
