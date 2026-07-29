"""Pydantic models for dataset samples and dataset metadata.

One schema serves both synthetic renders and annotated real photographs. Fields that
only a renderer can know (instance masks, exact camera parameters, render settings)
are optional; everything a model needs to train or be evaluated against is required.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chesssight.data.fen import (
    BOARD_SIZE,
    CLASS_NAMES,
    NUM_CLASSES,
    Grid,
    fen_to_grid,
    grid_to_placement,
    square_name,
    validate_grid,
)

Pixel = Annotated[list[float], Field(min_length=2, max_length=2)]
Matrix3 = Annotated[list[list[float]], Field(min_length=3, max_length=3)]
Matrix4 = Annotated[list[list[float]], Field(min_length=4, max_length=4)]

SampleSource = Literal["synthetic", "real"]
Split = Literal["train", "val", "test"]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields, so schema drift fails loudly."""

    model_config = ConfigDict(extra="forbid")


class MaskRLE(StrictModel):
    """A binary mask in COCO-style uncompressed RLE, row-major.

    ``counts`` alternates run lengths starting with a run of zeros. The runs must sum
    to ``height * width``.
    """

    height: int = Field(gt=0)
    width: int = Field(gt=0)
    counts: list[int]

    @model_validator(mode="after")
    def _check_counts(self) -> Self:
        if any(count < 0 for count in self.counts):
            raise ValueError("RLE counts must be non-negative")
        total = sum(self.counts)
        expected = self.height * self.width
        if total != expected:
            raise ValueError(f"RLE counts sum to {total}, expected {expected}")
        return self


class BoundingBox(StrictModel):
    """Axis-aligned box in pixels, ``xywh`` with the origin at the top-left."""

    x: float
    y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class PieceAnnotation(StrictModel):
    """One physical piece on the board."""

    class_id: int = Field(ge=1, lt=NUM_CLASSES)
    #: Null for a captured piece sitting beside the board. Keeping those in
    #: ``pieces`` teaches a detector that "a piece is visible" does not imply "a
    #: square is occupied" -- without them it learns to assign every piece it sees
    #: to the nearest square, which is wrong the moment a real game has captures.
    square: str | None = Field(default=None, min_length=2, max_length=2)
    rank_index: int | None = Field(default=None, ge=0, lt=BOARD_SIZE)
    file_index: int | None = Field(default=None, ge=0, lt=BOARD_SIZE)
    on_board: bool = True
    #: Value this piece has in the id pass. Renderer-only.
    instance_id: int | None = Field(default=None, ge=1)
    #: Occlusion-correct box, measured from the mask. This is the correct target
    #: for a detector; training on the amodal box teaches it to hallucinate pieces
    #: it cannot see.
    bbox: BoundingBox | None = None
    #: The whole piece including hidden parts, from projected mesh vertices. May
    #: extend outside the image.
    bbox_amodal: BoundingBox | None = None
    mask: MaskRLE | None = None
    visible_pixels: int | None = Field(default=None, ge=0)
    visibility: float | None = Field(default=None, ge=0, le=1)
    #: False when the piece is entirely hidden. It still keeps its grid entry: the
    #: square really is ambiguous from this viewpoint and the dataset should say so.
    visible: bool = True
    upright: bool = True
    base_center_px: Pixel | None = None

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]

    @model_validator(mode="after")
    def _check_square_matches_indices(self) -> Self:
        if not self.on_board:
            if self.square is not None or self.rank_index is not None:
                raise ValueError("a captured piece must not claim a square")
            return self

        if self.square is None or self.rank_index is None or self.file_index is None:
            raise ValueError("a piece on the board needs a square and indices")

        expected = square_name(self.rank_index, self.file_index)
        if self.square != expected:
            raise ValueError(
                f"square {self.square!r} disagrees with indices "
                f"(rank={self.rank_index}, file={self.file_index}) -> {expected!r}"
            )
        return self


class SquareAnnotation(StrictModel):
    """Projected geometry and occupancy of one of the 64 squares."""

    index: int = Field(ge=0, lt=BOARD_SIZE * BOARD_SIZE)
    name: str = Field(min_length=2, max_length=2)
    center_px: Pixel
    quad_px: Annotated[list[Pixel], Field(min_length=4, max_length=4)]
    occupant: int = Field(ge=0, lt=NUM_CLASSES)
    #: Whether the whole square lies within the image bounds.
    in_frame: bool = True


class BoardAnnotation(StrictModel):
    """Board outline and the board-plane -> image homography."""

    corners_px: Annotated[list[Pixel], Field(min_length=4, max_length=4)]
    homography: Matrix3
    #: Image -> board plane, so a consumer can rectify without inverting it again.
    homography_inv: Matrix3 | None = None
    #: Max reprojection error over the 64 square centres, in pixels. Renderer-only.
    reprojection_error_px: float | None = Field(default=None, ge=0)
    #: Real photographs crop the board constantly, so a partially visible board is
    #: a legitimate sample rather than a failure.
    all_corners_in_frame: bool = True


class CameraInfo(StrictModel):
    """Camera parameters, known exactly for renders and unknown for real photos."""

    intrinsics: Matrix3
    #: World-to-camera 4x4 transform.
    extrinsics: Matrix4
    focal_mm: float = Field(gt=0)
    sensor_width_mm: float = Field(gt=0)
    resolution: Annotated[list[int], Field(min_length=2, max_length=2)]


class RenderInfo(StrictModel):
    """How a synthetic sample was produced, enough to reproduce it exactly."""

    engine: Literal["CYCLES", "BLENDER_EEVEE"]
    samples: int = Field(gt=0)
    seed: int
    blender_version: str
    render_seconds: float | None = Field(default=None, ge=0)


class Sample(StrictModel):
    """One image and its complete annotation."""

    id: str
    image: str
    #: The id pass, kept so masks can be re-decoded without re-rendering.
    mask_image: str | None = None
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source: SampleSource
    split: Split = "train"
    #: How the annotation was produced. Synthetic samples are ground truth; real
    #: photos are only as good as the corners someone clicked.
    annotation_method: Literal["synthetic_gt", "manual_corners_fen", "manual_full"] = (
        "synthetic_gt"
    )
    seed: int | None = None
    fen: str
    grid: Grid
    board: BoardAnnotation
    squares: Annotated[
        list[SquareAnnotation],
        Field(min_length=BOARD_SIZE * BOARD_SIZE, max_length=BOARD_SIZE * BOARD_SIZE),
    ]
    pieces: list[PieceAnnotation] = Field(default_factory=list)
    camera: CameraInfo | None = None
    render: RenderInfo | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        validate_grid(self.grid)

        if fen_to_grid(self.fen) != self.grid:
            raise ValueError(
                f"fen and grid disagree: fen placement is {self.fen.split()[0]!r}, "
                f"grid encodes {grid_to_placement(self.grid)!r}"
            )

        for position, square in enumerate(self.squares):
            if square.index != position:
                raise ValueError(
                    f"squares must be in grid reading order; "
                    f"position {position} has index {square.index}"
                )
            rank_index, file_index = divmod(position, BOARD_SIZE)
            if square.occupant != self.grid[rank_index][file_index]:
                raise ValueError(
                    f"square {square.name} occupant {square.occupant} disagrees "
                    f"with grid value {self.grid[rank_index][file_index]}"
                )

        for piece in self.pieces:
            # Captured pieces are deliberately absent from the grid: they are in
            # the image but not on the board, which is exactly the distinction
            # this field exists to record.
            if not piece.on_board:
                continue
            assert piece.rank_index is not None and piece.file_index is not None
            expected = self.grid[piece.rank_index][piece.file_index]
            if piece.class_id != expected:
                raise ValueError(
                    f"piece on {piece.square} has class {piece.class_id} but the "
                    f"grid says {expected}"
                )
        return self


class IndexEntry(StrictModel):
    """One line of ``index.jsonl``: enough to filter a dataset without opening
    every sample record."""

    id: str
    image: str
    sample: str
    source: SampleSource
    split: Split
    fen: str


class DatasetMeta(StrictModel):
    """Top-level provenance for a generated run."""

    name: str
    created_at: str
    source: SampleSource
    master_seed: int
    git_commit: str | None = None
    blender_version: str | None = None
    generator_config: dict[str, object] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    #: Provenance of an external piece set, when one was used. Recorded so the
    #: licence travels with the renders rather than living only in someone's memory.
    asset_attribution: dict[str, object] | None = None
    notes: str | None = None
