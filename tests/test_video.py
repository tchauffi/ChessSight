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
