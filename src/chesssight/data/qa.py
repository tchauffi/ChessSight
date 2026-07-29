"""Draw a sample's labels onto its image.

Numeric checks catch a homography that is wrong by a lot. This catches the ones that
are wrong by a little, and the ones that are self-consistently wrong -- a transposed
board or a mirrored rank order satisfies every algebraic assertion but is instantly
obvious when the square names are drawn where the labels claim they are.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from chesssight.data.fen import CLASS_NAMES
from chesssight.data.schema import Sample

BOARD_OUTLINE = (255, 210, 40)
SQUARE_LINE = (90, 200, 255)
CENTER_DOT = (60, 255, 120)
PIECE_BOX = (255, 90, 90)
AMODAL_BOX = (255, 160, 60)
CORNER_LABELS = ("a8", "h8", "h1", "a1")
HEADER_HEIGHT = 46


def tint_masks(
    image: Image.Image, sample: Sample, *, alpha: float = 0.45
) -> Image.Image:
    """Blend each instance mask over the image in its own colour.

    Drawing the masks -- rather than trusting that they exist because a pixel count
    is non-zero -- is the only way to see that a mask actually covers the piece it
    claims to, and that it is not offset or attached to the wrong instance.
    """
    if not any(piece.mask is not None for piece in sample.pieces):
        return image

    import numpy as np

    from chesssight.data.masks import colorize, instance_mask

    labels = instance_mask(sample)
    tint = colorize(labels)
    array = np.asarray(image, dtype=np.float32)
    painted = labels > 0
    array[painted] = (1.0 - alpha) * array[painted] + alpha * tint[painted]
    return Image.fromarray(array.astype("uint8"))


def _draw_polygon(draw: ImageDraw.ImageDraw, points, color, width: int = 1) -> None:
    flat = [(float(x), float(y)) for x, y in points]
    draw.line([*flat, flat[0]], fill=color, width=width)


def _draw_pieces(draw: ImageDraw.ImageDraw, sample: Sample, shift) -> None:
    """Both boxes per piece: dashed-looking amodal outline, solid modal box.

    Seeing them together is the point -- where they diverge is exactly where a piece
    is occluded, and a detector trained on the wrong one will hallucinate.
    """
    for piece in sample.pieces:
        if piece.bbox_amodal is not None:
            x0, y0, x1, y1 = piece.bbox_amodal.xyxy
            draw.rectangle([*shift((x0, y0)), *shift((x1, y1))], outline=AMODAL_BOX)
        if piece.bbox is not None and piece.visible:
            x0, y0, x1, y1 = piece.bbox.xyxy
            draw.rectangle(
                [*shift((x0, y0)), *shift((x1, y1))], outline=PIECE_BOX, width=2
            )
            draw.text(
                shift((x0, y0 - 9)),
                CLASS_NAMES[piece.class_id].replace("_", " "),
                fill=PIECE_BOX,
            )


def render_overlay(
    sample: Sample,
    image_path: Path,
    *,
    show_squares: bool = True,
    show_pieces: bool = True,
    show_names: bool = False,
    show_masks: bool = True,
    mask_alpha: float = 0.45,
) -> Image.Image:
    """Return the sample's image with its labels drawn on top."""
    base = Image.open(image_path).convert("RGB")
    if show_masks:
        base = tint_masks(base, sample, alpha=mask_alpha)

    canvas = Image.new("RGB", (base.width, base.height + HEADER_HEIGHT), (18, 18, 22))
    canvas.paste(base, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)

    def shift(point) -> tuple[float, float]:
        return (float(point[0]), float(point[1]) + HEADER_HEIGHT)

    if show_squares:
        for square in sample.squares:
            _draw_polygon(draw, [shift(p) for p in square.quad_px], SQUARE_LINE)
            x, y = shift(square.center_px)
            draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=CENTER_DOT)
            if show_names:
                draw.text((x + 3, y - 5), square.name, fill=CENTER_DOT)

    _draw_polygon(draw, [shift(p) for p in sample.board.corners_px], BOARD_OUTLINE, 2)
    for index, corner in enumerate(sample.board.corners_px):
        x, y = shift(corner)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=BOARD_OUTLINE, width=2)
        draw.text((x + 6, y - 6), CORNER_LABELS[index], fill=BOARD_OUTLINE)

    if show_pieces:
        _draw_pieces(draw, sample, shift)

    visible = sum(1 for piece in sample.pieces if piece.visible)
    error = sample.board.reprojection_error_px
    draw.text((6, 4), f"{sample.id}  {sample.fen.split()[0]}", fill=(235, 235, 235))
    draw.text(
        (6, 18),
        f"{len(sample.pieces)} pieces ({visible} visible)   "
        f"reproj {error:.4f} px   "
        f"corners in frame: {sample.board.all_corners_in_frame}   "
        f"{sample.render.engine if sample.render else 'n/a'}",
        fill=(160, 160, 170),
    )
    draw.text(
        (6, 32),
        "yellow: board outline   blue: squares   green: centres   "
        "red: mask box   orange: amodal box",
        fill=(120, 120, 130),
    )
    return canvas


def save_overlay(sample: Sample, image_path: Path, out_path: Path, **kwargs) -> Path:
    """Render an overlay and write it to ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_overlay(sample, image_path, **kwargs).save(out_path)
    return out_path


def contact_sheet(
    images: list[Image.Image], columns: int = 4, cell_width: int = 420
) -> Image.Image:
    """Tile images into one sheet for eyeballing randomisation coverage."""
    if not images:
        raise ValueError("contact_sheet needs at least one image")

    scaled = []
    for image in images:
        ratio = cell_width / image.width
        scaled.append(
            image.resize(
                (cell_width, max(1, int(image.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        )

    cell_height = max(image.height for image in scaled)
    rows = (len(scaled) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (12, 12, 15))
    for index, image in enumerate(scaled):
        row, column = divmod(index, columns)
        sheet.paste(image, (column * cell_width, row * cell_height))
    return sheet
