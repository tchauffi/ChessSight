"""Reading an 8x8 position out of detections plus board geometry.

mAP cannot answer the question these tests ask. A detector that puts a perfect box
one square to the left scores well and reads the board wrongly, so square assignment
needs its own tests -- built on a known homography, so an error here is an error in
the assignment and not in the geometry it rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from chesssight.data.fen import BOARD_SIZE, LETTER_TO_CLASS
from chesssight.data.geometry import apply_homography, board_to_image_homography
from chesssight.train.labels import BOARD_INDEX, CORNER_INDEX, class_id_to_index
from chesssight.train.position import (
    POSITION_THRESHOLD,
    best_rotation,
    board_crop,
    enforce_material,
    foot,
    full_frame_weight,
    fuse,
    grid_from,
    grid_from_scores,
    read_board,
    read_position,
    score_grid,
    square_scores,
)

#: A board filling a 800x800 image, seen square-on. Deliberately not a perspective
#: view: these tests are about assignment, and a projective board would make a
#: failure ambiguous between the two.
CORNERS = [[100.0, 100.0], [700.0, 100.0], [700.0, 700.0], [100.0, 700.0]]
HOMOGRAPHY = board_to_image_homography(np.asarray(CORNERS))
SQUARE_PX = 600.0 / BOARD_SIZE


def detection_on(rank: int, file: int, class_id: int, score: float = 0.9) -> dict:
    """A piece box whose foot sits at the centre of one square."""
    centre = apply_homography(HOMOGRAPHY, [[file + 0.5, rank + 0.5]])[0]
    x, y = float(centre[0]), float(centre[1])
    # Tall box standing on the square, as a real piece's box is.
    return {
        "label": class_id_to_index(class_id),
        "score": score,
        "box": [x - SQUARE_PX / 3, y - SQUARE_PX * 2, x + SQUARE_PX / 3, y],
    }


def test_the_foot_is_near_the_bottom_centre_not_the_middle():
    # Pieces are tall and shot from above the table; the box centre floats up the
    # piece and lands a square or more behind where it actually stands.
    x, y = foot([10.0, 20.0, 30.0, 60.0])
    assert x == 20.0  # horizontally centred
    middle, bottom = 40.0, 60.0
    assert bottom - 0.1 * (bottom - middle) <= y <= bottom
    assert y > middle


def test_the_foot_sits_just_inside_the_bottom_edge():
    # Not *on* the edge: the bottom of a box is the nearest-to-camera point of the
    # base, forward of where the piece's axis meets the board, so a piece standing
    # near a square boundary is otherwise read one square over. Worth 5 points of
    # board-exact accuracy on synthetic boards.
    assert foot([0.0, 0.0, 10.0, 100.0])[1] < 100.0


class TestGridFrom:
    def test_a_piece_lands_on_its_own_square(self):
        grid = grid_from([detection_on(0, 0, 5)], HOMOGRAPHY)
        assert grid[0][0] == 5
        assert sum(sum(row) for row in grid) == 5  # nothing anywhere else

    @pytest.mark.parametrize("rank,file", [(0, 0), (0, 7), (7, 0), (7, 7), (3, 4)])
    def test_every_corner_and_the_middle_assign_correctly(self, rank, file):
        # An off-by-one in the u/v -> index conversion shows up at the edges first.
        grid = grid_from([detection_on(rank, file, 11)], HOMOGRAPHY)
        assert grid[rank][file] == 11

    def test_two_pieces_on_one_square_keep_the_better_one(self):
        grid = grid_from(
            [
                detection_on(2, 3, 1, score=0.4),
                detection_on(2, 3, 9, score=0.8),
            ],
            HOMOGRAPHY,
        )
        assert grid[2][3] == 9

    def test_a_piece_beside_the_board_is_not_placed(self):
        # Captured pieces stand off the board; the synthetic set labels them, and
        # they must not be forced onto a square.
        outside = detection_on(0, 0, 1)
        outside["box"] = [10.0, 10.0, 40.0, 60.0]
        assert grid_from([outside], HOMOGRAPHY) == [
            [0] * BOARD_SIZE for _ in range(BOARD_SIZE)
        ]

    def test_the_board_and_corner_classes_are_not_pieces(self):
        board = detection_on(4, 4, 1)
        board["label"] = BOARD_INDEX
        corner = detection_on(2, 2, 1)
        corner["label"] = CORNER_INDEX
        grid = grid_from([board, corner], HOMOGRAPHY)
        assert all(value == 0 for row in grid for value in row)


class TestScoring:
    def test_a_perfect_read_is_exact(self):
        truth = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        truth[0][0], truth[7][7] = 1, 12
        accuracy, exact = score_grid(truth, truth)
        assert accuracy == 1.0 and exact

    def test_one_wrong_square_costs_exactness(self):
        truth = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        predicted = [row[:] for row in truth]
        predicted[3][3] = 7
        accuracy, exact = score_grid(predicted, truth)
        assert not exact
        assert accuracy == pytest.approx(63 / 64)

    def test_a_rotated_read_scores_under_best_rotation(self):
        # Four interchangeable corners fix the geometry only up to a rotation, so a
        # correct read of a rotated board must not be reported as a failure.
        truth = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        truth[0][1] = 4
        rotated = np.rot90(np.asarray(truth), -1).tolist()
        assert best_rotation(rotated, truth) == (1.0, True)
        # ...while the unrotated comparison does see it as wrong.
        assert not score_grid(rotated, truth)[1]


class TestReadPosition:
    def test_missing_corners_give_no_position_rather_than_a_guess(self):
        assert read_position([detection_on(0, 0, 1)], None) is None
        assert read_position([detection_on(0, 0, 1)], CORNERS[:3]) is None

    def test_degenerate_corners_are_a_failure_not_a_crash(self):
        collapsed = [[100.0, 100.0]] * 4
        assert read_position([detection_on(0, 0, 1)], collapsed) is None

    def test_a_full_read_round_trips(self):
        truth = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        detections = []
        for rank, file, class_id in [(0, 0, 4), (1, 5, 1), (6, 2, 7), (7, 7, 10)]:
            truth[rank][file] = class_id
            detections.append(detection_on(rank, file, class_id))
        grid = read_position(detections, CORNERS)
        assert grid == truth


def test_position_threshold_is_below_a_detection_operating_point() -> None:
    """Reading a board wants recall; detecting objects wants balance.

    Pinned because the two thresholds are easy to conflate, and using the
    detection one for board reading costs 37 points of board accuracy.
    """
    from chesssight.train.predict_position import POSITION_THRESHOLD

    assert 0.0 < POSITION_THRESHOLD < 0.33


# --------------------------------------------------------------------------
# reading from several passes
# --------------------------------------------------------------------------
K, Q, R, N, P = (LETTER_TO_CLASS[x] for x in "KQRNP")
k, q, p = (LETTER_TO_CLASS[x] for x in "kqp")


def empty() -> list[list[int]]:
    return [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]


class TestSquareScores:
    def test_every_class_proposed_on_a_square_is_kept(self):
        # The runner-up label is the whole point: it is what a relabel uses.
        scores = square_scores(
            [detection_on(7, 4, Q, 0.6), detection_on(7, 4, K, 0.3)], HOMOGRAPHY
        )
        assert scores == {(7, 4, Q): 0.6, (7, 4, K): 0.3}

    def test_a_class_keeps_its_best_score(self):
        scores = square_scores(
            [detection_on(2, 2, N, 0.2), detection_on(2, 2, N, 0.7)], HOMOGRAPHY
        )
        assert scores == {(2, 2, N): 0.7}

    def test_board_corner_and_off_board_detections_are_ignored(self):
        board = detection_on(4, 4, 1)
        board["label"] = BOARD_INDEX
        outside = detection_on(0, 0, 1)
        outside["box"] = [10.0, 10.0, 40.0, 60.0]
        assert square_scores([board, outside], HOMOGRAPHY) == {}


class TestFuse:
    def test_a_weighted_mean_with_absence_counting_as_zero(self):
        a = {(0, 0, P): 0.9, (1, 1, N): 0.6}
        b = {(0, 0, P): 0.3}
        fused = fuse([(a, 1.0), (b, 2.0)])
        assert fused[(0, 0, P)] == pytest.approx((0.9 + 2 * 0.3) / 3)
        # Seen by one pass only: diluted, which is what makes agreement count.
        assert fused[(1, 1, N)] == pytest.approx(0.6 / 3)

    def test_grid_from_scores_takes_the_best_class_over_the_threshold(self):
        grid = grid_from_scores({(0, 0, Q): 0.5, (0, 0, K): 0.4, (1, 1, P): 0.01}, 0.05)
        assert grid[0][0] == Q
        assert grid[1][1] == 0


class TestEnforceMaterial:
    def test_a_legal_board_is_left_alone(self):
        grid = empty()
        grid[7][4], grid[0][4], grid[7][3] = K, k, Q
        scores = {(7, 4, K): 0.9, (7, 4, Q): 0.5, (0, 4, k): 0.9, (7, 3, Q): 0.8}
        assert enforce_material(grid, scores) == grid

    def test_of_two_kings_the_weaker_reading_gives_way(self):
        # e1 is a king beyond doubt; d1's king reading barely beats its queen
        # reading. The cheaper relabel is d1's.
        grid = empty()
        grid[7][4], grid[7][3], grid[0][4] = K, K, k
        scores = {
            (7, 4, K): 0.9,
            (7, 4, Q): 0.05,
            (7, 3, K): 0.5,
            (7, 3, Q): 0.45,
            (0, 4, k): 0.9,
        }
        fixed = enforce_material(grid, scores)
        assert fixed[7][4] == K and fixed[7][3] == Q

    def test_the_best_move_wins_not_the_first_square_scanned(self):
        # Scanning a8..h1 meets the strong queen first; relabelling it would
        # also remove the excess, but costs far more than relabelling the weak one.
        grid = empty()
        grid[0][0], grid[7][7], grid[7][4], grid[0][4] = Q, Q, K, k
        scores = {
            (0, 0, Q): 0.9,
            (0, 0, R): 0.5,
            (7, 7, Q): 0.3,
            (7, 7, R): 0.28,
            (7, 4, K): 0.9,
            (0, 4, k): 0.9,
        }
        fixed = enforce_material(grid, scores)
        assert fixed[0][0] == Q and fixed[7][7] == R

    def test_a_missing_king_is_not_invented(self):
        # No black king read, though one was faintly proposed on a queen's
        # square. Filling the gap would hide a broken read behind a plausible one.
        grid = empty()
        grid[7][4], grid[3][3] = K, q
        scores = {(7, 4, K): 0.9, (3, 3, q): 0.6, (3, 3, k): 0.2}
        assert enforce_material(grid, scores) == grid

    def test_an_excess_piece_with_no_alternative_is_kept(self):
        # A genuine promotion the detector saw cleanly: nothing to relabel to.
        grid = empty()
        grid[7][4], grid[0][4], grid[7][3], grid[2][2] = K, k, Q, Q
        scores = {(7, 4, K): 0.9, (0, 4, k): 0.9, (7, 3, Q): 0.9, (2, 2, Q): 0.8}
        assert enforce_material(grid, scores) == grid

    def test_relabels_only_to_proposals_above_the_floor(self):
        grid = empty()
        grid[7][4], grid[7][3], grid[0][4] = K, K, k
        scores = {(7, 4, K): 0.9, (7, 3, K): 0.5, (7, 3, Q): 0.001, (0, 4, k): 0.9}
        assert enforce_material(grid, scores) == grid


class TestCrop:
    def test_the_crop_is_square_inside_the_image_and_holds_the_board(self):
        left, top, side = board_crop(CORNERS, 1600, 1200)
        assert 0 <= left and left + side <= 1600
        assert 0 <= top and top + side <= 1200
        for x, y in CORNERS:
            assert left <= x <= left + side and top <= y <= top + side

    def test_more_room_above_the_board_than_below(self):
        # Pieces stand up from the board, so the far rank overhangs its edge.
        corners = [[500.0, 500.0], [700.0, 500.0], [700.0, 700.0], [500.0, 700.0]]
        left, top, side = board_crop(corners, 2000, 2000)
        assert 500 - top > (top + side) - 700

    def test_the_full_frame_counts_by_its_relative_resolution_squared(self):
        assert full_frame_weight(1000, 1000, 1500) == pytest.approx(1.0)
        assert full_frame_weight(500, 1000, 1000) == pytest.approx(0.25)


class TestReadBoard:
    """Through a fake detector that sees the same pieces in whatever view it gets.

    What this pins is the plumbing a real model cannot test: the crop's boxes
    must be shifted back into the photograph and the mirrored crop's
    reflected back, or a pass votes for the wrong square.
    """

    #: A small asymmetric position, so that a reflection bug lands pieces on
    #: empty squares instead of on look-alikes.
    PIECES = [(7, 6, K), (6, 0, P), (6, 1, P), (0, 4, k), (1, 7, p), (1, 6, p)]
    SIZE = (1600, 1200)
    MARK = (255, 0, 0)

    def photograph(self):
        from PIL import Image

        image = Image.new("RGB", self.SIZE, (120, 120, 120))
        for rank in range(BOARD_SIZE):
            for file in range(BOARD_SIZE):
                light = (rank + file) % 2 == 0
                x, y = 100 + file * SQUARE_PX, 100 + rank * SQUARE_PX
                box = (int(x), int(y), int(x + SQUARE_PX), int(y + SQUARE_PX))
                image.paste((210, 210, 210) if light else (45, 45, 45), box)
        # A marker at the crop's own top-left: where it lands in a view says
        # whether that view was mirrored.
        left, top, _ = board_crop(CORNERS, *self.SIZE)
        image.paste(self.MARK, (left, top, left + 6, top + 6))
        return image

    def detector(self, only: str | None = None):
        left, top, side = board_crop(CORNERS, *self.SIZE)
        world = [detection_on(r, f, c) for r, f, c in self.PIECES]

        def detect(view):
            if view.size == self.SIZE:
                kind = "full"
            elif view.getpixel((2, 2)) == self.MARK:
                kind = "crop"
            else:
                assert view.getpixel((side - 3, 2)) == self.MARK
                kind = "mirrored"
            if only is not None and kind != only:
                return []
            out = []
            for d in world:
                x0, y0, x1, y1 = d["box"]
                if kind != "full":
                    x0, y0, x1, y1 = x0 - left, y0 - top, x1 - left, y1 - top
                if kind == "mirrored":
                    x0, x1 = side - x1, side - x0
                out.append({**d, "box": [x0, y0, x1, y1]})
            return out

        return detect

    def truth(self):
        grid = empty()
        for rank, file, class_id in self.PIECES:
            grid[rank][file] = class_id
        return grid

    def test_three_agreeing_passes_read_the_position(self):
        result = read_board(self.photograph(), CORNERS, self.detector())
        assert result["grid"] == self.truth()
        assert result["fen"].startswith("4k3/6pp/8/8/8/8/PP6/6K1")

    @pytest.mark.parametrize("only", ["full", "crop", "mirrored"])
    def test_each_pass_alone_votes_for_the_right_squares(self, only):
        result = read_board(self.photograph(), CORNERS, self.detector(only))
        assert result["grid"] == self.truth()


def test_the_fused_threshold_still_reads_a_lone_confident_pass() -> None:
    """A piece only one pass sees, at a confident 0.9, must survive fusion.

    With the full frame at its smallest weight that pass is a third of the
    mean, and 0.3 clears the threshold by a wide margin.
    """
    assert 0.9 / 3 > POSITION_THRESHOLD
