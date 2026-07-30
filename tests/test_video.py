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
