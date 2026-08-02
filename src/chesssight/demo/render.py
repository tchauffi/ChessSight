"""Pictures of a reading: the overlay drawn on the photograph, and the board.

Kept apart from the server so it imports without torch -- the drawing needs
only PIL and python-chess, and the tests exercise it that way.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw

#: The board outline, and the two piece colours. These are the demo page's own
#: accent and the two hues it labels in its legend; changing one means changing
#: the other.
QUAD_COLOR = (224, 175, 69)
WHITE_PIECE_COLOR = (125, 211, 252)
BLACK_PIECE_COLOR = (244, 114, 182)

#: Tint for a square the pipeline read wrongly. One value, not a theme pair:
#: the SVG is written once and has to stay legible on either background.
WRONG_SQUARE = "#d0464f"

#: python-chess draws an orange board by default. These are the demo's neutrals,
#: chosen to read on a light or a dark page, since one SVG serves both.
BOARD_COLORS = {
    "square light": "#e6e8df",
    "square dark": "#9aa392",
    "margin": "none",
    "coord": "#7d857e",
    "inner border": "#8b938a",
    "outer border": "none",
}


def board_svg(fen: str, *, size: int = 360, wrong: Iterable[str] | None = None) -> str:
    """An SVG diagram of a position, in the demo's colours.

    Shared with ``chesssight predict --diagram`` and the published site, so a
    diagram looks the same whichever way it was produced.

    ``wrong`` names squares to tint -- given ground truth, that is what turns a
    diagram from an assertion into a comparison.
    """
    import chess
    import chess.svg

    fill = {chess.parse_square(name): WRONG_SQUARE for name in wrong or ()}
    return chess.svg.board(
        chess.Board(fen),
        size=size,
        coordinates=True,
        colors=BOARD_COLORS,
        fill=fill,
    )


def fit(image: Image.Image, width: int) -> tuple[Image.Image, float]:
    """Downscale to ``width``, returning the scale factor applied.

    Detections are in the original image's pixels, so anything drawn on the
    resized copy has to be scaled by the same factor.
    """
    if image.width <= width:
        return image.copy(), 1.0
    factor = width / image.width
    return (
        image.resize(
            (width, max(1, round(image.height * factor))),
            Image.Resampling.LANCZOS,
        ),
        factor,
    )


def overlay(
    image: Image.Image,
    detections: list[dict[str, Any]],
    corners: list[list[float]] | None,
    *,
    width: int = 900,
) -> Image.Image:
    """The photograph with what the two models found drawn over it.

    The board outline comes from the corner model and the boxes from the
    detector, so a reading that went wrong shows *which* of the two produced
    the wrongness -- an outline off the board, or a piece with no box.
    """
    canvas, factor = fit(image.convert("RGB"), width)
    draw = ImageDraw.Draw(canvas)

    for detection in detections:
        name = str(detection["name"])
        if name == "board":
            continue
        x0, y0, x1, y1 = (float(value) * factor for value in detection["box"])
        colour = WHITE_PIECE_COLOR if name.startswith("white") else BLACK_PIECE_COLOR
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=2)

    if corners:
        points = [(x * factor, y * factor) for x, y in corners]
        draw.line([*points, points[0]], fill=QUAD_COLOR, width=3)
        for x, y in points:
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=QUAD_COLOR)
    return canvas


def jpeg_uri(image: Image.Image, *, quality: int = 82) -> str:
    """A ``data:`` URI, so a result needs no second request to display."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
