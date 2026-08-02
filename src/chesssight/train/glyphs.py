"""Vector chess pieces for diagrams.

Drawn from primitives rather than typed from a font. The Unicode chess symbols
exist and DejaVu carries them, but a font lookup that fails renders a tofu box --
and it fails on exactly the machines where nobody is watching, in a CI image or a
container without desktop fonts. A missing glyph in a position diagram is worse
than a plain letter, because it still looks like a piece is there.

Shapes are defined in a unit square with y running down, matching image
coordinates, and scaled at draw time. They are silhouettes, not illustrations:
at diagram size -- a square is often under 40 pixels -- what has to survive is
which of the six a piece is, and the distinguishing feature of each is its
outline.
"""

from __future__ import annotations

from chesssight.data.fen import CLASS_TO_LETTER

#: Every piece stands on the same base, so it is defined once.
BASE = [
    (0.22, 0.79),
    (0.78, 0.79),
    (0.86, 0.93),
    (0.14, 0.93),
]

#: Polygons per piece, in a unit square. Drawn in order, each filled with the
#: piece's own colour and outlined in the contrasting one.
SHAPES: dict[str, list[list[tuple[float, float]]]] = {
    "p": [
        [(0.38, 0.44), (0.62, 0.44), (0.68, 0.79), (0.32, 0.79)],
        BASE,
    ],
    "r": [
        [
            (0.24, 0.16),
            (0.36, 0.16),
            (0.36, 0.25),
            (0.44, 0.25),
            (0.44, 0.16),
            (0.56, 0.16),
            (0.56, 0.25),
            (0.64, 0.25),
            (0.64, 0.16),
            (0.76, 0.16),
            (0.76, 0.34),
            (0.24, 0.34),
        ],
        [(0.32, 0.34), (0.68, 0.34), (0.72, 0.79), (0.28, 0.79)],
        BASE,
    ],
    # A horse's head in profile, facing left. The muzzle is what makes it read as
    # a knight rather than a spiky blob, so it projects well past the body and the
    # ears stay short -- the first attempt had tall ears and a blunt nose and read
    # as a wolf.
    "n": [
        [
            (0.30, 0.79),
            (0.28, 0.64),
            (0.22, 0.58),
            (0.14, 0.56),
            (0.11, 0.48),
            (0.18, 0.42),
            (0.28, 0.40),
            (0.34, 0.34),
            (0.38, 0.23),
            (0.44, 0.16),
            (0.48, 0.26),
            (0.56, 0.19),
            (0.61, 0.33),
            (0.67, 0.40),
            (0.63, 0.46),
            (0.70, 0.53),
            (0.73, 0.65),
            (0.75, 0.79),
        ],
        BASE,
    ],
    "b": [
        [(0.36, 0.52), (0.64, 0.52), (0.70, 0.79), (0.30, 0.79)],
        BASE,
    ],
    "q": [
        [
            (0.16, 0.46),
            (0.24, 0.22),
            (0.33, 0.42),
            (0.42, 0.18),
            (0.50, 0.40),
            (0.58, 0.18),
            (0.67, 0.42),
            (0.76, 0.22),
            (0.84, 0.46),
            (0.76, 0.60),
            (0.24, 0.60),
        ],
        [(0.26, 0.60), (0.74, 0.60), (0.76, 0.79), (0.24, 0.79)],
        BASE,
    ],
    "k": [
        [
            (0.45, 0.05),
            (0.55, 0.05),
            (0.55, 0.14),
            (0.64, 0.14),
            (0.64, 0.23),
            (0.55, 0.23),
            (0.55, 0.32),
            (0.45, 0.32),
            (0.45, 0.23),
            (0.36, 0.23),
            (0.36, 0.14),
            (0.45, 0.14),
        ],
        [(0.26, 0.36), (0.74, 0.36), (0.70, 0.58), (0.30, 0.58)],
        [(0.30, 0.58), (0.70, 0.58), (0.74, 0.79), (0.26, 0.79)],
        BASE,
    ],
}

#: Round parts, as ``(centre_x, centre_y, radius_x, radius_y)``. Kept apart from
#: the polygons because a circle approximated by a polygon at this size reads as
#: a lumpy heptagon.
ROUND: dict[str, list[tuple[float, float, float, float]]] = {
    "p": [(0.50, 0.31, 0.14, 0.14)],
    "b": [(0.50, 0.33, 0.16, 0.21), (0.50, 0.12, 0.055, 0.055)],
    "q": [
        (0.24, 0.20, 0.05, 0.05),
        (0.42, 0.16, 0.05, 0.05),
        (0.58, 0.16, 0.05, 0.05),
        (0.76, 0.20, 0.05, 0.05),
    ],
}

#: Lines drawn in the contrasting colour on top -- the bishop's mitre slit and
#: the knight's eye, the two details that separate them from a pawn at a glance.
DETAIL: dict[str, list[tuple[float, float, float, float]]] = {
    "b": [(0.50, 0.18, 0.58, 0.34)],
    "n": [(0.32, 0.42, 0.37, 0.46)],
}


def draw_piece(
    draw,
    class_id: int,
    *,
    origin: tuple[float, float],
    size: float,
    body: tuple[int, int, int],
    ink: tuple[int, int, int],
) -> None:
    """Draw one piece filling a ``size``-square box at ``origin``.

    ``body`` is the piece's own colour and ``ink`` the contrasting one, so the
    same shapes serve both sides -- which is the point: a white knight and a
    black knight must be the same silhouette, or the diagram is telling the
    reader something untrue about the position.
    """
    letter = CLASS_TO_LETTER[class_id].lower()
    shapes = SHAPES.get(letter)
    if shapes is None:
        return
    width = max(1, int(size / 22))

    def at(x: float, y: float) -> tuple[float, float]:
        return (origin[0] + x * size, origin[1] + y * size)

    for polygon in shapes:
        draw.polygon([at(x, y) for x, y in polygon], fill=body, outline=ink)
    for cx, cy, rx, ry in ROUND.get(letter, []):
        draw.ellipse(
            [*at(cx - rx, cy - ry), *at(cx + rx, cy + ry)],
            fill=body,
            outline=ink,
            width=width,
        )
    for x0, y0, x1, y1 in DETAIL.get(letter, []):
        draw.line([at(x0, y0), at(x1, y1)], fill=ink, width=width)
