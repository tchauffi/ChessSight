from __future__ import annotations

import numpy as np
import pytest

from chesssight.data import geometry as geo


def _known_homography() -> np.ndarray:
    """A homography with a genuine perspective component."""
    return np.array(
        [
            [42.0, 3.0, 61.0],
            [-4.0, 47.0, 55.0],
            [0.004, 0.011, 1.0],
        ]
    )


def test_solve_recovers_a_known_homography():
    truth = _known_homography()
    source = np.asarray(geo.BOARD_CORNERS)
    target = geo.apply_homography(truth, source)

    recovered = geo.solve_homography(source, target)
    np.testing.assert_allclose(recovered, truth / truth[2, 2], atol=1e-8)


def test_round_trip_through_board_to_image_homography():
    truth = _known_homography()
    corners_px = geo.apply_homography(truth, np.asarray(geo.BOARD_CORNERS))

    homography = geo.board_to_image_homography(corners_px)
    centers = geo.project_square_centers(homography)

    expected = geo.apply_homography(truth, geo.all_square_centers_board())
    np.testing.assert_allclose(centers, expected, atol=1e-6)
    assert (
        geo.reprojection_error(homography, geo.all_square_centers_board(), expected)
        < 1e-6
    )


def test_projected_centers_are_in_grid_reading_order():
    # Identity-scaled homography: board units become pixels directly.
    homography = np.eye(3)
    centers = geo.project_square_centers(homography)
    assert centers.shape == (64, 2)
    # index 0 is grid[0][0] -> centre (0.5, 0.5)
    np.testing.assert_allclose(centers[0], [0.5, 0.5])
    # index 63 is grid[7][7] -> centre (7.5, 7.5)
    np.testing.assert_allclose(centers[63], [7.5, 7.5])
    # index 8 is grid[1][0]
    np.testing.assert_allclose(centers[8], [0.5, 1.5])


def test_square_quads_have_four_corners_each():
    quads = geo.project_square_quads(np.eye(3))
    assert quads.shape == (64, 4, 2)
    np.testing.assert_allclose(quads[0], [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_every_square_center_lies_inside_the_board_outline():
    truth = _known_homography()
    corners_px = geo.apply_homography(truth, np.asarray(geo.BOARD_CORNERS))
    homography = geo.board_to_image_homography(corners_px)

    for center in geo.project_square_centers(homography):
        assert geo.polygon_contains(corners_px, (float(center[0]), float(center[1])))


def test_reprojection_error_is_reported_in_pixels():
    homography = np.eye(3)
    board_points = np.array([[0.0, 0.0], [1.0, 0.0]])
    pixel_points = np.array([[0.0, 0.0], [1.0, 3.0]])
    assert geo.reprojection_error(
        homography, board_points, pixel_points
    ) == pytest.approx(3.0)


def test_collinear_correspondences_raise():
    source = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    target = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    with pytest.raises(geo.GeometryError):
        geo.solve_homography(source, target)


def test_coincident_points_raise():
    source = np.zeros((4, 2))
    target = np.asarray(geo.BOARD_CORNERS)
    with pytest.raises(geo.GeometryError):
        geo.solve_homography(source, target)


def test_too_few_correspondences_raise():
    with pytest.raises(geo.GeometryError):
        geo.solve_homography(np.zeros((3, 2)), np.zeros((3, 2)))


def test_mismatched_shapes_raise():
    with pytest.raises(geo.GeometryError):
        geo.solve_homography(np.zeros((4, 2)), np.zeros((5, 2)))


def test_wrong_corner_count_raises():
    with pytest.raises(geo.GeometryError):
        geo.board_to_image_homography(np.zeros((3, 2)))


def test_non_finite_input_raises():
    corners = np.asarray(geo.BOARD_CORNERS, dtype=float).copy()
    corners[0, 0] = np.nan
    with pytest.raises(geo.GeometryError):
        geo.board_to_image_homography(corners)


def test_matrix_to_list_is_json_friendly():
    listed = geo.matrix_to_list(np.eye(3))
    assert listed == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert all(isinstance(value, float) for row in listed for value in row)


class TestChirality:
    """A mirrored board is self-consistent, so only the winding order reveals it."""

    def test_canonical_corners_project_with_positive_winding(self):
        truth = _known_homography()
        corners = geo.apply_homography(truth, np.asarray(geo.BOARD_CORNERS))
        assert geo.polygon_signed_area(corners) > 0
        assert not geo.is_mirrored(corners)

    def test_a_mirrored_board_is_detected(self):
        # Flipping one world axis -- the exact bug class this guards against --
        # negates the winding while leaving every other label untouched.
        truth = _known_homography()
        corners = geo.apply_homography(truth, np.asarray(geo.BOARD_CORNERS))
        mirrored = corners.copy()
        mirrored[:, 1] = -mirrored[:, 1]

        assert geo.polygon_signed_area(mirrored) == pytest.approx(
            -geo.polygon_signed_area(corners)
        )
        assert geo.is_mirrored(mirrored)

    def test_mirroring_leaves_the_homography_solvable_and_consistent(self):
        # Demonstrates why this check is needed: the mirrored board still solves
        # cleanly and still reprojects perfectly, so no residual-based test sees it.
        truth = _known_homography()
        corners = geo.apply_homography(truth, np.asarray(geo.BOARD_CORNERS))
        mirrored = corners.copy()
        mirrored[:, 1] = -mirrored[:, 1]

        homography = geo.board_to_image_homography(mirrored)
        error = geo.reprojection_error(
            homography, np.asarray(geo.BOARD_CORNERS), mirrored
        )
        assert error < 1e-9
        assert geo.is_mirrored(mirrored)

    def test_signed_area_matches_the_known_board_area(self):
        # Identity homography: board units are pixels, so the area is 8x8.
        assert geo.polygon_signed_area(np.asarray(geo.BOARD_CORNERS)) == pytest.approx(
            64.0
        )

    def test_degenerate_polygon_raises(self):
        with pytest.raises(geo.GeometryError):
            geo.polygon_signed_area(np.zeros((2, 2)))


def test_polygon_contains_rejects_outside_points():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert geo.polygon_contains(square, (0.5, 0.5))
    assert not geo.polygon_contains(square, (1.5, 0.5))
    assert not geo.polygon_contains(square, (-0.1, 0.5))
