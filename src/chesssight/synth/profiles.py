"""Lathe profiles for the procedurally generated chess pieces.

Pure data and pure arithmetic -- no ``bpy``. This module lives on the project side so
the profiles can be unit-tested in CI, but it is deliberately import-light so the
Blender side can import it too (see :mod:`chesssight.blender.pieces`).

Profile space
-------------
A profile is a list of ``(radius, z)`` control points describing the silhouette that
is swept around the Z axis:

* ``radius`` is in **square units** -- a radius of ``0.30`` means the piece is 0.6
  squares wide, so it fits comfortably inside its square.
* ``z`` is **normalised to the piece's own height**, running from ``0.0`` at the base
  to ``1.0`` at the apex. Multiply by ``PIECE_HEIGHTS[letter] * square_size`` to get
  world units.

Every profile must start and end on the axis (``radius == 0``) so the swept surface
closes into a solid; :func:`validate_profile` enforces that.

Pieces that are not surfaces of revolution -- the rook's crenellations, the king's
cross, the knight's head -- have those features added as separate geometry by the
provider. The profile here describes only the turned part.
"""

from __future__ import annotations

from typing import Final, TypeAlias

Profile: TypeAlias = tuple[tuple[float, float], ...]

#: Piece letters in the canonical order used by :mod:`chesssight.data.fen`.
PIECE_LETTERS: Final = "PNBRQK"

#: Total height of each piece, in square units. A real Staunton king is roughly
#: 1.7x the square width; the rest are scaled down from there in the usual ratios.
PIECE_HEIGHTS: Final[dict[str, float]] = {
    "P": 0.85,
    "N": 1.10,
    "B": 1.20,
    "R": 0.95,
    "Q": 1.40,
    "K": 1.55,
}

#: Fraction of the total height reached by the *turned* part of each piece. It is
#: below 1.0 exactly where non-lathe geometry finishes the piece off: the knight's
#: head sits on a short pedestal, and the king's cross tops out the last 6%.
PROFILE_TOP: Final[dict[str, float]] = {
    "P": 1.00,
    "N": 0.45,
    "B": 1.00,
    "R": 0.88,
    "Q": 1.00,
    "K": 0.94,
}

#: Rook merlons occupy the band between ``PROFILE_TOP["R"]`` and the apex, and span
#: the rim wall radially so their outer face is flush with the rim rather than
#: floating outside it.
ROOK_CRENELLATIONS: Final = (4, 6)
ROOK_RIM_OUTER: Final = 0.29
ROOK_RIM_INNER: Final = 0.20

#: The queen's coronet: a ring of points encircling the crown, added as separate
#: geometry when ``pieces.queen_coronet`` is enabled. Without it the procedural
#: queen is a pure surface of revolution whose profile differs from the king's by
#: a few hundredths -- height is then the only cue separating the two letters,
#: and queen<->king is the detector's single worst confusion on real photographs.
#: The ring radius clears the crown's own radius (<= 0.12 over the band) across
#: the full taper range, so the points always break the silhouette.
QUEEN_CORONET_POINTS: Final = (5, 9)
QUEEN_CORONET_RADIUS: Final = 0.135
QUEEN_CORONET_THICKNESS: Final = 0.045
QUEEN_CORONET_BAND: Final = (0.86, 0.97)

PAWN: Final[Profile] = (
    (0.00, 0.00),
    (0.30, 0.00),
    (0.30, 0.05),
    (0.26, 0.09),
    (0.15, 0.14),
    (0.11, 0.30),
    (0.13, 0.42),
    (0.17, 0.50),
    (0.16, 0.55),
    (0.10, 0.60),
    (0.15, 0.70),
    (0.17, 0.80),
    (0.13, 0.92),
    (0.00, 1.00),
)

#: The rook's turned part traces a *hollow* rim: up the outside, in across the top
#: annulus, down the inside wall, then across the recessed floor. The merlons are
#: added on top of that ring, which is what makes them read as part of the rim
#: rather than as blocks resting on a flat disc.
ROOK: Final[Profile] = (
    (0.00, 0.00),
    (0.33, 0.00),
    (0.33, 0.06),
    (0.28, 0.12),
    (0.19, 0.18),
    (0.17, 0.56),
    (0.21, 0.64),
    (0.29, 0.72),
    (0.29, 0.88),  # outer rim, top
    (0.20, 0.88),  # across the rim
    (0.20, 0.79),  # down the inner wall
    (0.00, 0.79),  # recessed floor
)

KNIGHT: Final[Profile] = (
    (0.00, 0.00),
    (0.32, 0.00),
    (0.32, 0.06),
    (0.28, 0.11),
    (0.19, 0.16),
    (0.16, 0.34),
    (0.18, 0.42),
    (0.17, 0.45),
    (0.00, 0.45),
)

BISHOP: Final[Profile] = (
    (0.00, 0.00),
    (0.31, 0.00),
    (0.31, 0.06),
    (0.27, 0.11),
    (0.16, 0.17),
    (0.12, 0.38),
    (0.15, 0.46),
    (0.20, 0.52),
    (0.19, 0.57),
    (0.12, 0.60),
    (0.17, 0.70),
    (0.16, 0.82),
    (0.10, 0.88),
    (0.06, 0.90),
    (0.08, 0.94),
    (0.05, 0.99),
    (0.00, 1.00),
)

QUEEN: Final[Profile] = (
    (0.00, 0.00),
    (0.35, 0.00),
    (0.35, 0.06),
    (0.30, 0.12),
    (0.18, 0.18),
    (0.13, 0.40),
    (0.16, 0.50),
    (0.22, 0.58),
    (0.26, 0.66),
    (0.24, 0.72),
    (0.16, 0.75),
    (0.20, 0.80),
    (0.19, 0.86),
    (0.12, 0.89),
    (0.07, 0.92),
    (0.09, 0.96),
    (0.05, 0.99),
    (0.00, 1.00),
)

KING: Final[Profile] = (
    (0.00, 0.00),
    (0.36, 0.00),
    (0.36, 0.06),
    (0.31, 0.12),
    (0.19, 0.18),
    (0.13, 0.42),
    (0.16, 0.52),
    (0.23, 0.60),
    (0.26, 0.68),
    (0.24, 0.74),
    (0.17, 0.78),
    (0.21, 0.83),
    (0.20, 0.88),
    (0.13, 0.91),
    (0.09, 0.93),
    (0.00, 0.94),
)

PROFILES: Final[dict[str, Profile]] = {
    "P": PAWN,
    "N": KNIGHT,
    "B": BISHOP,
    "R": ROOK,
    "Q": QUEEN,
    "K": KING,
}

#: Largest radius of each profile, used to check pieces fit inside a square and to
#: rest a tipped-over piece at the right height.
MAX_RADII: Final[dict[str, float]] = {
    letter: max(radius for radius, _ in profile) for letter, profile in PROFILES.items()
}


class ProfileError(ValueError):
    """Raised when a profile could not produce a closed, well-formed solid."""


def validate_profile(letter: str, profile: Profile) -> None:
    """Raise :class:`ProfileError` unless ``profile`` is sweepable into a solid."""
    if len(profile) < 3:
        raise ProfileError(f"{letter}: profile needs at least 3 points")

    first_radius, first_z = profile[0]
    if first_radius != 0.0:
        raise ProfileError(f"{letter}: profile must start on the axis (radius 0)")
    if first_z != 0.0:
        raise ProfileError(f"{letter}: profile must start at z=0")

    last_radius, _ = profile[-1]
    if last_radius != 0.0:
        raise ProfileError(f"{letter}: profile must end on the axis (radius 0)")

    expected_top = PROFILE_TOP[letter]
    actual_top = max(z for _, z in profile)
    if abs(actual_top - expected_top) > 1e-9:
        raise ProfileError(
            f"{letter}: turned part reaches z={actual_top}, expected {expected_top}"
        )

    for radius, z in profile:
        if radius < 0.0:
            raise ProfileError(f"{letter}: negative radius {radius}")
        if not 0.0 <= z <= 1.0:
            raise ProfileError(f"{letter}: z={z} outside [0, 1]")
        if radius >= 0.5:
            raise ProfileError(
                f"{letter}: radius {radius} would overflow its square (limit 0.5)"
            )


def taper_factor(letter: str, z: float, taper: float) -> float:
    """Radius multiplier at profile height ``z``, for a silhouette ``taper``.

    ``height_scale`` and ``radius_scale`` change how big a set is, not what shape it
    is: scaled uniformly, every procedural set has the same outline, and a detector
    can learn that one outline instead of learning what a bishop is. ``taper``
    redistributes radius along the piece instead -- ``+t`` widens the base and narrows
    the top by the same fraction, ``-t`` does the reverse, giving squat and slender
    variants of the same set. Normalising by :data:`PROFILE_TOP` makes ``1 + t`` the
    factor at the base and ``1 - t`` at the top of the turned part for every letter,
    whether or not non-lathe geometry finishes it off.

    The factor never exceeds ``1 + |taper|``, which is what bounds the widened base
    inside its square -- see :func:`chesssight.synth.config.PiecesConfig`.
    """
    return 1.0 + taper * (1.0 - 2.0 * z / PROFILE_TOP[letter])


def scaled_profile(
    letter: str,
    *,
    square_size: float = 1.0,
    height_scale: float = 1.0,
    radius_scale: float = 1.0,
    taper: float = 0.0,
) -> list[tuple[float, float]]:
    """Return the profile in world units, ready to be swept.

    ``height_scale``, ``radius_scale`` and ``taper`` are the per-scene style jitter:
    applying them once per scene keeps every piece in an image looking like one set.
    """
    if letter not in PROFILES:
        raise ProfileError(f"unknown piece letter {letter!r}")
    height = PIECE_HEIGHTS[letter] * square_size * height_scale
    return [
        (
            radius * square_size * radius_scale * taper_factor(letter, z, taper),
            z * height,
        )
        for radius, z in PROFILES[letter]
    ]


def piece_height(
    letter: str, *, square_size: float = 1.0, height_scale: float = 1.0
) -> float:
    """Total height of a piece in world units."""
    if letter not in PIECE_HEIGHTS:
        raise ProfileError(f"unknown piece letter {letter!r}")
    return PIECE_HEIGHTS[letter] * square_size * height_scale


def piece_radius(
    letter: str,
    *,
    square_size: float = 1.0,
    radius_scale: float = 1.0,
    taper: float = 0.0,
) -> float:
    """Largest radius of a piece in world units.

    Measured over the warped profile rather than from :data:`MAX_RADII`, because a
    taper moves *which* point is widest: the additive geometry sized against this
    (the rook rim, the knight's pedestal) has to match the lathe it sits on.
    """
    if letter not in MAX_RADII:
        raise ProfileError(f"unknown piece letter {letter!r}")
    if taper == 0.0:
        return MAX_RADII[letter] * square_size * radius_scale
    widest = max(
        radius * taper_factor(letter, z, taper) for radius, z in PROFILES[letter]
    )
    return widest * square_size * radius_scale


def validate_all() -> None:
    """Validate every built-in profile. Called by the test suite."""
    for name, mapping in (
        ("PROFILES", PROFILES),
        ("PIECE_HEIGHTS", PIECE_HEIGHTS),
        ("PROFILE_TOP", PROFILE_TOP),
    ):
        if set(mapping) != set(PIECE_LETTERS):
            raise ProfileError(
                f"{name} keys {sorted(mapping)} != {sorted(PIECE_LETTERS)}"
            )
    for letter, profile in PROFILES.items():
        validate_profile(letter, profile)
