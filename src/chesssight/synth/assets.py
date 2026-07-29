"""Manifests describing an external chess set.

A downloaded chess set is never in the coordinate system, scale or orientation this
pipeline needs. The manifest records what the set actually is, and
:mod:`chesssight.blender.assets` normalises it on import, so the renderer itself
stays unaware of where geometry came from.

This module is project-side (typed and unit-tested); the Blender side reads the same
JSON without importing pydantic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chesssight.synth.asset_spec import (
    AssetError,
    forward_yaw,
    kind_for_file,
    up_pitch,
)
from chesssight.synth.profiles import PIECE_LETTERS

__all__ = [
    "AssetError",
    "AssetManifest",
    "PieceAsset",
    "blank_manifest",
    "forward_yaw",
    "height_for",
    "up_pitch",
]

AssetKind = Literal["obj", "gltf", "fbx", "collada", "stl", "blend", "usd"]
Axis = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


class PieceAsset(BaseModel):
    """Where one piece's geometry lives."""

    model_config = ConfigDict(extra="forbid")

    #: File containing the piece, relative to the manifest.
    file: str
    #: Object name inside the file. Required for multi-object files such as a
    #: ``.blend`` or a single ``.glb`` holding the whole set; ignored when the file
    #: contains exactly one mesh.
    object: str | None = None
    #: Per-piece height override in board squares. Falls back to the set's
    #: ``king_height`` scaled by the standard Staunton ratios.
    height: float | None = Field(default=None, gt=0)


class AssetManifest(BaseModel):
    """A complete external chess set."""

    model_config = ConfigDict(extra="forbid")

    name: str
    #: Free text; recording where a set came from matters because most downloadable
    #: chess models carry attribution or share-alike terms.
    source: str | None = None
    license: str | None = None
    #: Credit line to reproduce wherever renders of this set are published.
    attribution: str | None = None
    #: Set by ``assets bake``: the manifest this one was pre-normalised from.
    baked_from: str | None = None

    #: Overrides the extension-based guess when a file is unusually named.
    kind: AssetKind | None = None

    #: Which way the pieces face in their own file. Knights are the only pieces
    #: where this is visible, and a knight facing the wrong way is the single most
    #: obvious defect an imported set can have.
    forward_axis: Axis = "+Y"
    up_axis: Axis = "+Z"

    #: Height of the king in board squares. Everything is scaled to match, so a set
    #: modelled in millimetres, inches or arbitrary units all end up correct.
    king_height: float = Field(default=1.55, gt=0)

    #: ``uniform`` measures the king and applies that one factor to the whole set,
    #: preserving the proportions it was designed with -- the right choice for a real
    #: set. ``per_piece`` forces each piece to a standard height instead, which only
    #: helps when the set was assembled from mismatched sources.
    scale_mode: Literal["uniform", "per_piece"] = "uniform"

    #: Optional mesh reduction, as the fraction of faces to keep. Print-ready STLs
    #: routinely carry hundreds of thousands of triangles per piece -- detail that is
    #: invisible at dataset resolutions but real in memory and BVH build time.
    decimate: float | None = Field(default=None, gt=0, le=1.0)

    #: ``{"P": PieceAsset, ...}``; every piece letter must be present.
    pieces: dict[str, PieceAsset]

    @model_validator(mode="after")
    def _check_pieces(self) -> Self:
        missing = set(PIECE_LETTERS) - set(self.pieces)
        if missing:
            raise ValueError(
                f"manifest is missing pieces {sorted(missing)}; "
                f"all of {sorted(PIECE_LETTERS)} are required"
            )
        unknown = set(self.pieces) - set(PIECE_LETTERS)
        if unknown:
            raise ValueError(f"manifest has unknown piece letters {sorted(unknown)}")
        return self

    @classmethod
    def load(cls, path: Path) -> AssetManifest:
        path = Path(path).expanduser()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise AssetError(f"cannot read manifest {path}: {error}") from error
        except json.JSONDecodeError as error:
            raise AssetError(f"manifest {path} is not valid JSON: {error}") from error
        return cls.model_validate(data)

    def save(self, path: Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def kind_for(self, asset: PieceAsset) -> str:
        """Import kind for one piece, from the manifest or the file extension."""
        return kind_for_file(asset.file, self.kind)

    def resolve(self, asset: PieceAsset, root: Path) -> Path:
        """Absolute path to a piece's file, relative to the manifest directory."""
        return (Path(root).expanduser() / asset.file).resolve()

    def check(self, root: Path) -> list[str]:
        """Return a list of problems, empty when the set is ready to render."""
        problems = []
        for letter in sorted(PIECE_LETTERS):
            asset = self.pieces[letter]
            path = self.resolve(asset, root)
            if not path.is_file():
                problems.append(f"{letter}: missing file {path}")
                continue
            try:
                self.kind_for(asset)
            except AssetError as error:
                problems.append(f"{letter}: {error}")
        return problems


def height_for(manifest: AssetManifest, letter: str) -> float:
    """Target height of a piece in board squares."""
    from chesssight.synth.profiles import PIECE_HEIGHTS

    override = manifest.pieces[letter].height
    if override is not None:
        return override
    # Scale the built-in ratios so the set keeps sane relative proportions even when
    # the source models are inconsistently sized.
    ratio = PIECE_HEIGHTS[letter] / PIECE_HEIGHTS["K"]
    return manifest.king_height * ratio


def blank_manifest(name: str, extension: str = ".obj") -> AssetManifest:
    """A manifest skeleton naming one file per piece, for a set to be filled in."""
    names = {
        "P": "pawn",
        "N": "knight",
        "B": "bishop",
        "R": "rook",
        "Q": "queen",
        "K": "king",
    }
    return AssetManifest(
        name=name,
        pieces={
            letter: PieceAsset(file=f"{names[letter]}{extension}")
            for letter in PIECE_LETTERS
        },
    )
