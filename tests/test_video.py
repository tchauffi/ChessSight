"""The ffmpeg plumbing for video annotation, tested with a generated clip.

Skipped where ffmpeg is absent (CI), like the Blender suite.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)


def corner_detections(points, score: float = 0.9) -> list[dict]:
    """Four corner-class detections, as small boxes centred on ``points``."""
    from chesssight.train.labels import CORNER_INDEX

    return [
        {
            "label": CORNER_INDEX,
            "name": "corner",
            "score": score,
            "box": [x - 4, y - 4, x + 4, y + 4],
        }
        for x, y in points
    ]


@pytest.fixture(scope="module")
def tiny_clip(tmp_path_factory):
    """One second of test pattern, 64x48 at 10 fps."""
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def test_probe_reads_geometry(tiny_clip):
    from chesssight.train.video import probe

    info = probe(tiny_clip)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(10.0)
    assert info.duration == pytest.approx(1.0, abs=0.2)


def test_frames_round_trip_through_the_pipes(tiny_clip, tmp_path):

    from chesssight.train.video import FrameWriter, probe, read_frames

    info = probe(tiny_clip)
    frames = list(read_frames(tiny_clip, info))
    assert len(frames) == 10
    assert frames[0].size == (64, 48)

    out = tmp_path / "out.mp4"
    writer = FrameWriter(out, info.width, info.height, info.fps)
    for frame in frames:
        writer.write(frame)
    assert writer.close() == 0

    written = probe(out)
    assert (written.width, written.height) == (64, 48)
    assert list(read_frames(out, written))


def test_odd_dimensions_are_cropped_even(tmp_path):
    from PIL import Image

    from chesssight.train.video import FrameWriter, probe

    out = tmp_path / "odd.mp4"
    # 65x47 would make yuv420p encoders refuse; the writer must crop to even.
    writer = FrameWriter(out, 65, 47, 10.0)
    for _ in range(5):
        writer.write(Image.new("RGB", (65, 47), (120, 40, 40)))
    assert writer.close() == 0
    info = probe(out)
    assert (info.width, info.height) == (64, 46)


def test_draw_frame_scales_with_resolution():
    from PIL import Image

    from chesssight.train.video import draw_frame

    frame = draw_frame(
        Image.new("RGB", (640, 360), (30, 30, 30)),
        [
            {
                "label": 0,
                "name": "white_pawn",
                "score": 0.91,
                "box": [100, 100, 160, 200],
            }
        ],
        status="test",
    )
    assert frame.size == (640, 360)


class TestBoardGate:
    """The gate is pure geometry, so it tests without torch or ffmpeg."""

    @staticmethod
    def piece(x0, y0, x1, y1, score=0.9, label=0):
        return {
            "label": label,
            "name": "white_pawn",
            "score": score,
            "box": [x0, y0, x1, y1],
        }

    @staticmethod
    def board(x0, y0, x1, y1, score=0.9):
        from chesssight.train.labels import BOARD_INDEX

        return {
            "label": BOARD_INDEX,
            "name": "board",
            "score": score,
            "box": [x0, y0, x1, y1],
        }

    def test_pieces_off_the_board_are_dropped(self):
        from chesssight.train.video import gate_to_board

        detections = [
            self.board(100, 100, 500, 400),
            self.piece(200, 200, 240, 300),  # foot inside
            self.piece(700, 50, 760, 150),  # a "piece" on someone's shirt
        ]
        kept, remembered, age = gate_to_board(detections, None)
        assert len(kept) == 2  # board + the real piece
        assert remembered == [100, 100, 500, 400]
        assert age == 0

    def test_board_is_chosen_by_piece_votes_not_score(self):
        # A confident sliver of a board must not beat the box the pieces stand
        # in -- observed dropping 19 of 23 detections on one real frame.
        from chesssight.train.video import gate_to_board

        detections = [
            self.board(0, 0, 40, 100, score=0.95),  # high-scoring garbage
            self.board(100, 100, 500, 400, score=0.60),
            self.piece(200, 200, 240, 300),
            self.piece(300, 200, 340, 310),
        ]
        kept, remembered, _ = gate_to_board(detections, None)
        assert remembered == [100, 100, 500, 400]
        assert sum(1 for d in kept if d["name"] == "white_pawn") == 2

    def test_remembered_board_expires(self):
        from chesssight.train.video import BOARD_MEMORY, gate_to_board

        stale = [0.0, 0.0, 10.0, 10.0]  # a box from a different shot
        detections = [self.piece(200, 200, 240, 300)]
        # Within memory: the stale box still gates (and drops the piece).
        kept, remembered, age = gate_to_board(detections, stale, age=0)
        assert kept == [] and remembered == stale and age == 1
        # Past memory: the gate opens rather than silently killing everything.
        kept, remembered, age = gate_to_board(detections, stale, age=BOARD_MEMORY)
        assert len(kept) == 1
        assert remembered is None

    def test_no_board_anywhere_passes_everything(self):
        from chesssight.train.video import gate_to_board

        detections = [self.piece(0, 0, 10, 10)]
        kept, remembered, _ = gate_to_board(detections, None)
        assert kept == detections
        assert remembered is None


class TestSuppressionAndCap:
    """Bounding the out-of-domain failure with geometry and physics."""

    @staticmethod
    def piece(x0, y0, x1, y1, score, name="white_pawn", label=0):
        return {"label": label, "name": name, "score": score, "box": [x0, y0, x1, y1]}

    def test_duplicate_boxes_collapse_to_the_best(self):
        from chesssight.train.video import suppress_overlaps

        detections = [
            self.piece(100, 100, 140, 200, 0.9),
            self.piece(102, 103, 141, 201, 0.7),  # same object, lower score
            self.piece(400, 100, 440, 200, 0.8),  # a different object
        ]
        kept = suppress_overlaps(detections)
        assert len(kept) == 2
        assert {d["score"] for d in kept} == {0.9, 0.8}

    def test_suppression_is_class_agnostic(self):
        # Out of domain one physical piece draws boxes that disagree about class;
        # per-class NMS would keep both, so suppression ignores the label.
        from chesssight.train.video import suppress_overlaps

        detections = [
            self.piece(100, 100, 140, 200, 0.9, "white_knight", 1),
            self.piece(101, 101, 141, 201, 0.85, "white_queen", 4),
        ]
        assert len(suppress_overlaps(detections)) == 1

    def test_board_is_never_suppressed(self):
        from chesssight.train.labels import BOARD_INDEX
        from chesssight.train.video import suppress_overlaps

        board = {
            "label": BOARD_INDEX,
            "name": "board",
            "score": 0.99,
            "box": [0, 0, 500, 500],
        }
        # The board overlaps every piece; it must survive regardless.
        kept = suppress_overlaps([board, self.piece(100, 100, 140, 200, 0.9)])
        assert any(d["label"] == BOARD_INDEX for d in kept)
        assert len(kept) == 2

    def test_cap_enforces_a_full_chess_set(self):
        from chesssight.train.video import MAX_PIECES, cap_pieces

        detections = [
            self.piece(i * 10, 0, i * 10 + 5, 20, score=i / 100.0) for i in range(90)
        ]
        kept = cap_pieces(detections)
        assert len(kept) == MAX_PIECES == 32
        # Highest-scoring survive, so ranking decides which 32.
        assert min(d["score"] for d in kept) > 0.5

    def test_cap_keeps_the_board_outside_the_budget(self):
        from chesssight.train.labels import BOARD_INDEX
        from chesssight.train.video import cap_pieces

        board = {
            "label": BOARD_INDEX,
            "name": "board",
            "score": 0.5,
            "box": [0, 0, 10, 10],
        }
        detections = [board] + [
            self.piece(i * 10, 0, i * 10 + 5, 20, 0.9) for i in range(40)
        ]
        kept = cap_pieces(detections)
        assert sum(1 for d in kept if d["label"] != BOARD_INDEX) == 32
        assert sum(1 for d in kept if d["label"] == BOARD_INDEX) == 1

    def test_only_one_board_survives_the_gate(self):
        from chesssight.train.labels import BOARD_INDEX
        from chesssight.train.video import gate_to_board

        def board(x0, y0, x1, y1, score):
            return {
                "label": BOARD_INDEX,
                "name": "board",
                "score": score,
                "box": [x0, y0, x1, y1],
            }

        detections = [
            board(0, 0, 40, 100, 0.99),
            board(100, 100, 500, 400, 0.60),
            self.piece(200, 200, 240, 300, 0.9),
        ]
        kept, _, _ = gate_to_board(detections, None)
        assert sum(1 for d in kept if d["label"] == BOARD_INDEX) == 1


class TestBoardTrackerSmoothing:
    def test_no_smoothing_takes_the_current_frame_alone(self):
        # With smoothing off a quad must not inherit anything from the frame
        # before it: fed unrelated views -- a slideshow, a cut -- an averaged
        # quad matches neither and lays its grid across the squares.
        from chesssight.train.video import BoardTracker

        tracker = BoardTracker(smoothing=0.0, memory=0)
        first = corner_detections([[10, 10], [90, 10], [90, 90], [10, 90]])
        second = corner_detections([[50, 50], [150, 50], [150, 150], [50, 150]])
        tracker.update(first)
        quad = tracker.update(second)
        assert quad is not None
        assert min(x for x, _ in quad) >= 49  # nothing left of the new view

    def test_no_memory_forgets_immediately(self):
        from chesssight.train.video import BoardTracker

        tracker = BoardTracker(smoothing=0.0, memory=0)
        tracker.update(corner_detections([[10, 10], [90, 10], [90, 90], [10, 90]]))
        assert tracker.update([]) is None

    def test_smoothing_on_still_damps(self):
        from chesssight.train.video import BoardTracker

        tracker = BoardTracker(smoothing=0.8, memory=12)
        tracker.update(corner_detections([[10, 10], [90, 10], [90, 90], [10, 90]]))
        quad = tracker.update(
            corner_detections([[50, 50], [150, 50], [150, 150], [50, 150]])
        )
        assert quad is not None
        assert min(x for x, _ in quad) < 49  # held back towards the old view


class TestPieceTracker:
    """Steadying detections across frames.

    Every test here is about a *sequence*; a tracker that behaves correctly on any
    single frame can still flicker, which is the whole reason it exists.
    """

    @staticmethod
    def piece(x0, y0, x1, y1, score, name="white_pawn", label=0):
        return {"label": label, "name": name, "score": score, "box": [x0, y0, x1, y1]}

    def tracker(self, **kwargs):
        from chesssight.train.video import PieceTracker

        return PieceTracker(enter_score=0.4, **kwargs)

    def drawn(self, tracker, frames):
        """How many pieces are drawn on each frame of a sequence."""
        return [
            len([d for d in tracker.update(frame) if d["label"] != 12])
            for frame in frames
        ]

    def test_a_score_oscillating_around_the_threshold_stops_blinking(self):
        # The headline case. Untracked, a piece scoring 0.45/0.35 alternately is
        # drawn on every other frame; the low score still clears the keep
        # threshold, so the track survives.
        tracker = self.tracker()
        scores = [0.45, 0.35, 0.45, 0.35, 0.45, 0.35]
        counts = self.drawn(
            tracker, [[self.piece(10, 10, 30, 60, score)] for score in scores]
        )
        # One frame to reach min_hits, then continuously present.
        assert counts == [0, 1, 1, 1, 1, 1]

    def test_a_piece_missing_for_a_frame_is_held(self):
        # A hand passes over the board, or one frame blurs. Dropping the box and
        # bringing it back is more distracting than holding it.
        tracker = self.tracker()
        present = [self.piece(10, 10, 30, 60, 0.9)]
        counts = self.drawn(tracker, [present, present, [], [], present])
        assert counts == [0, 1, 1, 1, 1]

    def test_a_piece_gone_for_good_is_eventually_dropped(self):
        from chesssight.train.video import MAX_MISSES

        tracker = self.tracker()
        present = [self.piece(10, 10, 30, 60, 0.9)]
        counts = self.drawn(tracker, [present, present] + [[]] * (MAX_MISSES + 2))
        assert counts[-1] == 0

    def test_a_one_frame_false_positive_is_never_drawn(self):
        # min_hits: a spurious high-scoring box on a single frame must not flash up.
        tracker = self.tracker()
        counts = self.drawn(
            tracker, [[], [self.piece(300, 300, 320, 360, 0.95)], [], []]
        )
        assert counts == [0, 0, 0, 0]

    def test_a_flapping_class_settles_on_one_label(self):
        # Same box, alternating labels. Whichever wins, the caption must not change
        # every frame -- and the winner should be the better-supported class.
        tracker = self.tracker()
        frames = [
            [self.piece(10, 10, 30, 60, 0.9, "white_knight", 1)],
            [self.piece(10, 10, 30, 60, 0.5, "white_bishop", 2)],
            [self.piece(10, 10, 30, 60, 0.9, "white_knight", 1)],
            [self.piece(10, 10, 30, 60, 0.5, "white_bishop", 2)],
        ]
        labels = [
            [d["name"] for d in tracker.update(frame) if d["label"] != 12]
            for frame in frames
        ]
        assert labels[1:] == [["white_knight"]] * 3

    def test_box_jitter_is_damped(self):
        tracker = self.tracker()
        for _ in range(3):
            tracker.update([self.piece(10, 10, 30, 60, 0.9)])
        # A single frame's 8px wobble must move the drawn box by much less. Any
        # jump large enough to break overlap is a different matter -- that is a
        # piece being moved, and it correctly starts a new track rather than
        # sliding the old box across the board.
        drawn = tracker.update([self.piece(18, 10, 38, 60, 0.9)])
        moved = drawn[0]["box"][0] - 10
        assert 0 < moved < 4

    def test_a_score_below_the_keep_threshold_cannot_start_a_track(self):
        tracker = self.tracker()
        weak = [self.piece(10, 10, 30, 60, 0.1)]
        assert self.drawn(tracker, [weak] * 5) == [0] * 5

    def test_the_board_passes_through_untracked(self):
        from chesssight.train.labels import BOARD_INDEX

        tracker = self.tracker()
        board = {
            "label": BOARD_INDEX,
            "name": "board",
            "score": 0.99,
            "box": [0, 0, 500, 500],
        }
        # Drawn on the very first frame -- no min_hits delay, no smoothing lag.
        drawn = tracker.update([board])
        assert drawn == [board]
