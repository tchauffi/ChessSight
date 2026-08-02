"""Warp a photographed board to a canonical square.

Once the corners are known the board's geometry is fully determined, and there is
no reason to make a classifier rediscover it. Rectifying puts every square in the
same place at the same size in every image, which turns "which square is this
piece on" from a perspective problem into an indexing one -- and lets the model
that reads pieces spend its capacity on what a piece *is*.

The margin is the part that is easy to get wrong
------------------------------------------------
A homography maps the board *plane*. Pieces stand up out of that plane, so under
the warp each one smears away from the camera, and the top of a tall piece on the
far rank projects well outside the 8x8 playing area. Warping exactly the playing
area therefore cuts the heads off precisely the pieces that are hardest to
identify -- and the result still looks like a clean board, so nothing complains.

The rectified region is extended beyond the board on every side, and further on
the far side, because that is where the smear goes.
"""

from __future__ import annotations

import numpy as np

from chesssight.data.fen import BOARD_SIZE

#: Extra board-units kept beyond the playing area, measured rather than guessed.
#: Projecting the top-centre of every annotated piece box through the inverse
#: homography over 150 ChessReD boards, piece tops overhang the board by a mean
#: of 0.75 squares sideways and 1.28 along the smear, with p90 at 1.65 and 2.50.
#: These are those p90s.
#:
#: Deliberately not the maxima, which are 5.31 and 4.33. Covering those would put
#: the board in under half the output's width and spend the resolution that
#: actually distinguishes a bishop from a pawn on empty tablecloth. The tail is
#: clipped knowingly: one board in ten loses the very top of its most extreme
#: piece, against every board losing a third of its detail.
SIDE_MARGIN = 1.65

#: Larger than the side margin because the perspective smear runs along v. A
#: board shot from overhead simply gets empty bands there, which costs nothing.
FAR_MARGIN = 2.5


def rectified_bounds(
    corners: list[list[float]],
    *,
    side: float = SIDE_MARGIN,
    far: float = FAR_MARGIN,
) -> tuple[float, float, float, float]:
    """The board-plane region to warp, as ``(u0, v0, u1, v1)``.

    ``v`` runs from rank 8 down to rank 1, so the *far* side of the board in
    board coordinates is whichever end is further from the camera -- which the
    corners alone cannot say. The region is therefore extended by ``far`` at both
    ends rather than guessing, and the classifier sees an empty band on whichever
    side happened to be near.
    """
    del corners  # kept in the signature: a camera-aware version would need them
    return (-side, -far, BOARD_SIZE + side, BOARD_SIZE + far)


def rectify(
    image,
    corners: list[list[float]],
    *,
    size: int = 512,
    side: float = SIDE_MARGIN,
    far: float = FAR_MARGIN,
):
    """Warp ``image`` so the board's playing area is axis-aligned and centred.

    Returns the warped PIL image. Its geometry is fixed and known: the playing
    area occupies the sub-rectangle given by :func:`playing_area_px`, so a model
    reading an 8x8 grid off it needs no further calibration.
    """
    from PIL import Image

    from chesssight.data.geometry import board_to_image_homography

    u0, v0, u1, v1 = rectified_bounds(corners, side=side, far=far)
    board = board_to_image_homography(np.asarray(corners, dtype=np.float64))

    # PIL's PERSPECTIVE transform wants the map from *output* pixels back to
    # input pixels, so this composes output -> board-plane -> image and inverts
    # nothing by hand. Doing it the other way round is the classic silent flip.
    scale_u = (u1 - u0) / size
    scale_v = (v1 - v0) / size
    output_to_board = np.array(
        [[scale_u, 0.0, u0], [0.0, scale_v, v0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    output_to_image = board @ output_to_board
    output_to_image = output_to_image / output_to_image[2, 2]

    return image.convert("RGB").transform(
        (size, size),
        Image.Transform.PERSPECTIVE,
        output_to_image.flatten()[:8].tolist(),
        resample=Image.Resampling.BILINEAR,
    )


def playing_area_px(
    size: int = 512, *, side: float = SIDE_MARGIN, far: float = FAR_MARGIN
) -> tuple[float, float, float, float]:
    """Where the 8x8 playing area sits inside a rectified image, in pixels."""
    u0, v0, u1, v1 = -side, -far, BOARD_SIZE + side, BOARD_SIZE + far
    return (
        (0.0 - u0) / (u1 - u0) * size,
        (0.0 - v0) / (v1 - v0) * size,
        (BOARD_SIZE - u0) / (u1 - u0) * size,
        (BOARD_SIZE - v0) / (v1 - v0) * size,
    )


def square_centre_px(
    rank: int,
    file: int,
    size: int = 512,
    *,
    side: float = SIDE_MARGIN,
    far: float = FAR_MARGIN,
) -> tuple[float, float]:
    """Centre of ``grid[rank][file]`` in a rectified image, in pixels."""
    x0, y0, x1, y1 = playing_area_px(size, side=side, far=far)
    cell_w = (x1 - x0) / BOARD_SIZE
    cell_h = (y1 - y0) / BOARD_SIZE
    return (x0 + (file + 0.5) * cell_w, y0 + (rank + 0.5) * cell_h)
