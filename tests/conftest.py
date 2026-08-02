"""Shared test helpers.

The ``make_sample`` factory builds a *self-consistent* :class:`Sample` -- squares in
reading order, occupants matching the grid, pieces matching their squares -- so tests
that care about one field can mutate just that field and assert the validator
complains.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from chesssight.data.fen import (
    BOARD_SIZE,
    STARTING_FEN,
    Grid,
    fen_to_grid,
    iter_occupied,
    square_name,
)
from chesssight.data.geometry import (
    BOARD_CORNERS,
    all_square_centers_board,
    apply_homography,
    board_to_image_homography,
    matrix_to_list,
    project_square_centers,
    project_square_quads,
    reprojection_error,
)
from chesssight.data.schema import (
    BoardAnnotation,
    BoundingBox,
    PieceAnnotation,
    Sample,
    SquareAnnotation,
)

#: A homography with real perspective, used to make plausible pixel coordinates.
TEST_HOMOGRAPHY = np.array(
    [
        [46.0, 4.0, 70.0],
        [-5.0, 51.0, 60.0],
        [0.003, 0.010, 1.0],
    ]
)


def blender_available() -> bool:
    return shutil.which("blender") is not None


requires_blender = pytest.mark.skipif(
    not blender_available(), reason="blender not installed"
)


def make_sample(
    fen: str = STARTING_FEN,
    *,
    sample_id: str = "000000",
    with_pieces: bool = True,
) -> Sample:
    grid: Grid = fen_to_grid(fen)
    corners_px = apply_homography(TEST_HOMOGRAPHY, np.asarray(BOARD_CORNERS))
    homography = board_to_image_homography(corners_px)
    centers = project_square_centers(homography)
    quads = project_square_quads(homography)

    squares = []
    for index in range(BOARD_SIZE * BOARD_SIZE):
        rank_index, file_index = divmod(index, BOARD_SIZE)
        squares.append(
            SquareAnnotation(
                index=index,
                name=square_name(rank_index, file_index),
                center_px=[float(centers[index][0]), float(centers[index][1])],
                quad_px=[[float(x), float(y)] for x, y in quads[index]],
                occupant=grid[rank_index][file_index],
            )
        )

    pieces = []
    if with_pieces:
        for rank_index, file_index, class_id in iter_occupied(grid):
            center = centers[rank_index * BOARD_SIZE + file_index]
            pieces.append(
                PieceAnnotation(
                    class_id=class_id,
                    square=square_name(rank_index, file_index),
                    rank_index=rank_index,
                    file_index=file_index,
                    bbox=BoundingBox(
                        x=float(center[0]) - 10.0,
                        y=float(center[1]) - 20.0,
                        width=20.0,
                        height=30.0,
                    ),
                    visible_pixels=350,
                )
            )

    return Sample(
        id=sample_id,
        image=f"images/{sample_id}.jpg",
        width=512,
        height=512,
        source="synthetic",
        split="train",
        fen=fen,
        grid=grid,
        board=BoardAnnotation(
            corners_px=[[float(x), float(y)] for x, y in corners_px],
            homography=matrix_to_list(homography),
            reprojection_error_px=reprojection_error(
                homography, all_square_centers_board(), centers
            ),
        ),
        squares=squares,
        pieces=pieces,
    )


@pytest.fixture
def sample() -> Sample:
    return make_sample()
