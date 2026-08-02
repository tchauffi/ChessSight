"""Asset-set conventions shared by both sides of the Blender boundary.

Deliberately **stdlib only**. :mod:`chesssight.synth.assets` layers pydantic
validation on top for project-side use; :mod:`chesssight.blender.assets` imports this
module directly, because Blender's bundled Python has no pydantic. Keeping the shared
vocabulary here rather than in the validated module is what lets both sides agree on
what a manifest means without the Blender side pulling in dependencies it cannot have.

``tests/test_boundary.py`` enforces the rule.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Piece letters, duplicated from ``chesssight.synth.profiles`` to keep this module
#: import-free.
PIECE_LETTERS = "PNBRQK"

#: Heights in board squares, keyed by piece letter. Mirrors ``profiles.PIECE_HEIGHTS``.
DEFAULT_HEIGHTS: dict[str, float] = {
    "P": 0.85,
    "N": 1.10,
    "B": 1.20,
    "R": 0.95,
    "Q": 1.40,
    "K": 1.55,
}

#: File extension -> import kind.
KIND_BY_SUFFIX: dict[str, str] = {
    ".obj": "obj",
    ".glb": "gltf",
    ".gltf": "gltf",
    ".fbx": "fbx",
    ".dae": "collada",
    ".stl": "stl",
    ".blend": "blend",
    ".usd": "usd",
    ".usdc": "usd",
    ".usda": "usd",
}

#: Rotation about Z, in degrees, taking a set's forward axis onto ``+Y``.
FORWARD_YAW: dict[str, float] = {"+Y": 0.0, "+X": 90.0, "-Y": 180.0, "-X": -90.0}

#: Rotation about X, in degrees, taking a set's up axis onto ``+Z``.
UP_PITCH: dict[str, float] = {"+Z": 0.0, "+Y": 90.0, "-Z": 180.0, "-Y": -90.0}


class AssetError(ValueError):
    """Raised when a manifest does not describe a usable chess set."""


def kind_for_file(filename: str, override: str | None = None) -> str:
    """Import kind for a file, from an explicit override or its extension."""
    if override:
        return override
    suffix = Path(filename).suffix.lower()
    if suffix not in KIND_BY_SUFFIX:
        raise AssetError(
            f"cannot tell how to import {filename!r}; set `kind` explicitly "
            f"or use one of {sorted(KIND_BY_SUFFIX)}"
        )
    return KIND_BY_SUFFIX[suffix]


def forward_yaw(forward_axis: str) -> float:
    """Yaw needed to point a set's forward axis towards the opponent."""
    if forward_axis not in FORWARD_YAW:
        raise AssetError(
            f"forward_axis {forward_axis!r} must be one of {sorted(FORWARD_YAW)}; "
            f"a vertical forward axis is not meaningful for a chess piece"
        )
    return FORWARD_YAW[forward_axis]


def up_pitch(up_axis: str) -> float:
    """Pitch needed to stand a set upright.

    Sets exported from Y-up tools -- glTF's native convention, and most game
    engines -- arrive lying on their side without this.
    """
    if up_axis not in UP_PITCH:
        raise AssetError(f"up_axis {up_axis!r} must be one of {sorted(UP_PITCH)}")
    return UP_PITCH[up_axis]


#: How a set's pieces are scaled into board units.
SCALE_MODES = ("uniform", "per_piece")
DEFAULT_SCALE_MODE = "uniform"


def scale_mode(manifest: dict) -> str:
    """Which scaling rule a set uses.

    ``uniform`` (the default) measures the king and applies that *one* factor to
    every piece, preserving the set's own proportions. This is what you want for a
    real set, where the pieces were designed together -- a Staunton king is nearly
    2.5x its own base while a pawn is barely 1.6x, and forcing each piece to a
    prescribed height would quietly redesign the set.

    ``per_piece`` scales each piece to its own target height instead. Use it for a
    set assembled from mismatched sources, where the relative sizes are wrong and
    imposing standard proportions is an improvement rather than a distortion.
    """
    mode = manifest.get("scale_mode", DEFAULT_SCALE_MODE)
    if mode not in SCALE_MODES:
        raise AssetError(f"scale_mode {mode!r} must be one of {list(SCALE_MODES)}")
    return mode


def target_height(manifest: dict, letter: str) -> float:
    """Height a piece should end up, in board squares, under ``per_piece`` scaling.

    A per-piece override wins; otherwise the set's king height is scaled by the
    standard proportions.
    """
    entry = manifest["pieces"][letter]
    override = entry.get("height")
    if override is not None:
        return float(override)
    king_height = float(manifest.get("king_height", DEFAULT_HEIGHTS["K"]))
    return king_height * DEFAULT_HEIGHTS[letter] / DEFAULT_HEIGHTS["K"]


def king_height(manifest: dict) -> float:
    """Target king height in board squares."""
    entry = manifest["pieces"].get("K", {})
    override = entry.get("height")
    if override is not None:
        return float(override)
    return float(manifest.get("king_height", DEFAULT_HEIGHTS["K"]))


def decimate_ratio(manifest: dict) -> float | None:
    """Optional mesh reduction, as a surviving-face fraction.

    Print-ready STLs routinely carry hundreds of thousands of triangles per piece --
    detail that is invisible at dataset resolutions but real in BVH build time and
    memory across a long batch.
    """
    ratio = manifest.get("decimate")
    if ratio is None:
        return None
    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise AssetError(f"decimate must be in (0, 1], got {ratio}")
    return None if ratio == 1.0 else ratio


def load_manifest(path: Path) -> dict:
    """Read a manifest as a plain dict, with only structural checks.

    Full validation lives in :mod:`chesssight.synth.assets`; this is what the
    Blender side uses, and it only needs enough checking to fail with a useful
    message rather than a ``KeyError`` halfway through a render.
    """
    path = Path(path).expanduser()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AssetError(f"cannot read manifest {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise AssetError(f"manifest {path} is not valid JSON: {error}") from error

    if not isinstance(manifest, dict) or "pieces" not in manifest:
        raise AssetError(f"manifest {path} has no `pieces` mapping")
    missing = [letter for letter in PIECE_LETTERS if letter not in manifest["pieces"]]
    if missing:
        raise AssetError(f"manifest {path} is missing pieces {missing}")
    return manifest


#: Licence fragments that restrict how renders of a set may be used. Matching is
#: deliberately loose -- the point is to prompt a human decision, not to adjudicate.
RESTRICTIVE_LICENCE_MARKERS = ("NC", "NON-COMMERCIAL", "NONCOMMERCIAL", "ND", "NODERIV")


def licence_warnings(manifest: dict) -> list[str]:
    """Licence caveats worth surfacing before a set is used to build a dataset.

    Renders are derivative works of the models they depict, so the set's terms
    follow the dataset, and anything trained on it. This is not legal advice; it
    exists so the terms are visible at the moment the set is chosen rather than
    discovered later.
    """
    licence = (manifest.get("license") or "").strip()
    if not licence:
        return [
            "no `license` recorded -- unlicensed models default to all rights "
            "reserved, which does not permit redistributing renders of them"
        ]

    warnings = []
    upper = licence.upper().replace("-", " ")
    if any(f" {marker} " in f" {upper} " for marker in RESTRICTIVE_LICENCE_MARKERS):
        warnings.append(
            f"{licence} restricts commercial use and/or derivatives. Renders are "
            f"derivative works, so this follows the dataset and anything trained "
            f"on it. Fine for research and personal projects; not for a commercial "
            f"product without different assets."
        )
    if "BY" in upper.split() and not manifest.get("attribution"):
        warnings.append(
            "the licence requires attribution but no `attribution` field is set"
        )
    return warnings


def attribution(manifest: dict) -> dict:
    """Provenance recorded into the dataset so the terms travel with the renders."""
    return {
        "set": manifest.get("name", "unknown"),
        "source": manifest.get("source"),
        "license": manifest.get("license"),
        "attribution": manifest.get("attribution"),
    }
