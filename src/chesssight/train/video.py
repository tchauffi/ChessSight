"""Run the detector over a video and write an annotated copy.

Frames stream through ffmpeg as raw RGB in both directions, so no video library
joins the dependencies -- ffmpeg is already the one tool guaranteed to read whatever
a phone actually produces.

Two honest limitations, stated rather than hidden:

* The detector emits *boxes*, including the board's -- but not the board's corners,
  so there is no homography and no per-square position readout on video. What this
  shows is detection: which pieces, where, how confidently.
* The calibration was fit on ChessReD's photographs. On a different piece set,
  board or lighting, the ranking usually survives while the absolute scores drift --
  measured on stock footage the drift went *upward*, flooding the threshold with
  false positives. The board gate removes most of them (the board detection
  transfers far better than piece scores), and ``--top-k`` falls back to ranking;
  for real use on a new domain, calibrate on a few annotated frames of it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chesssight.train.labels import BOARD_INDEX

PREDICTION_COLOR = (255, 80, 80)
BOARD_COLOR = (255, 205, 40)


class VideoError(RuntimeError):
    """Raised when a video cannot be read or written."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float

    @property
    def frames(self) -> int:
        return int(self.duration * self.fps)


def probe(path: Path) -> VideoInfo:
    """Read a video's geometry with ffprobe."""
    if shutil.which("ffprobe") is None:
        raise VideoError("ffprobe not found; install ffmpeg")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise VideoError(f"cannot read {path}: {result.stderr.strip()[-300:]}")

    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, _, denominator = stream["avg_frame_rate"].partition("/")
    fps = float(numerator) / float(denominator or 1)
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        duration=float(payload["format"].get("duration", 0.0)),
    )


def read_frames(
    path: Path, info: VideoInfo, *, max_seconds: float | None = None
) -> Iterator[Image.Image]:
    """Decode a video to RGB frames through an ffmpeg pipe."""
    command = ["ffmpeg", "-v", "error", "-i", str(path)]
    if max_seconds:
        command += ["-t", str(max_seconds)]
    command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    stdout = process.stdout
    assert stdout is not None  # PIPE above guarantees it
    frame_bytes = info.width * info.height * 3
    try:
        while True:
            chunk = stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            array = np.frombuffer(chunk, dtype=np.uint8).reshape(
                info.height, info.width, 3
            )
            yield Image.fromarray(array)
    finally:
        stdout.close()
        process.wait()


class FrameWriter:
    """Encode RGB frames back to H.264 through an ffmpeg pipe."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        # yuv420p needs even dimensions; H.264 players insist on it too.
        self.size = (width - width % 2, height - height % 2)
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.size[0]}x{self.size[1]}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        stdin = self.process.stdin
        assert stdin is not None  # PIPE above guarantees it
        self.stdin = stdin

    def write(self, frame: Image.Image) -> None:
        if frame.size != self.size:
            frame = frame.crop((0, 0, self.size[0], self.size[1]))
        self.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> int:
        self.stdin.close()
        return self.process.wait()


def draw_frame(
    frame: Image.Image, predictions: list[dict], *, status: str = ""
) -> Image.Image:
    """Draw predictions at video scale.

    Line weights and type scale with frame height: a 2px box that reads fine on a
    512px render disappears on a 2160p phone video.
    """
    draw = ImageDraw.Draw(frame)
    stroke = max(2, frame.height // 360)
    try:
        font = ImageFont.load_default(size=max(12, frame.height // 54))
        small = ImageFont.load_default(size=max(10, frame.height // 68))
    except TypeError:  # very old Pillow: fixed-size bitmap font
        font = small = ImageFont.load_default()

    for prediction in predictions:
        colour = BOARD_COLOR if prediction["label"] == BOARD_INDEX else PREDICTION_COLOR
        x0, y0, x1, y1 = prediction["box"]
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=stroke)
        text = f"{prediction['name'].replace('_', ' ')} {prediction['score']:.2f}"
        top = max(0, y0 - (font.size + 6 if hasattr(font, "size") else 14))
        box = draw.textbbox((x0, top), text, font=font)
        draw.rectangle(
            [box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill=(10, 10, 12)
        )
        draw.text((x0, top), text, fill=colour, font=font)

    if status:
        draw.rectangle(
            [0, 0, frame.width, (small.size if hasattr(small, "size") else 12) + 10],
            fill=(10, 10, 12),
        )
        draw.text((8, 4), status, fill=(230, 230, 230), font=small)
    return frame


#: How many consecutive detection passes a remembered board box survives without
#: being re-confirmed. A one-frame flicker must not open the gate, but a *stale*
#: box is worse than none: after a scene cut it belongs to a different shot and
#: silently gates out every legitimate piece.
BOARD_MEMORY = 12


def gate_to_board(
    detections: list[dict],
    last_board: list[float] | None,
    *,
    margin: float = 0.15,
    age: int = 0,
) -> tuple[list[dict], list[float] | None, int]:
    """Drop piece detections that do not stand on the detected board.

    Out of domain, piece scores drift and boxes land on people and furniture --
    but the *board* detection stays reliable well past the point piece scores
    break down. Requiring each piece's foot to fall inside the board box (grown
    by ``margin``) uses the model's own most-transferable output to police its
    least-transferable one. No ground truth involved.

    Returns the surviving detections, the board box to remember, and its age.
    """
    boards = [d for d in detections if d["label"] == BOARD_INDEX]
    if boards:
        # Choose the board that explains the pieces, not the one with the best
        # score. The model sometimes emits a confident sliver of a board box, and
        # score-based selection then gates every legitimate piece out -- observed
        # dropping 19 of 23 detections on one frame. A piece-vote is
        # self-consistent: feet standing inside a candidate are evidence for it.
        feet = [
            ((d["box"][0] + d["box"][2]) / 2.0, d["box"][3])
            for d in detections
            if d["label"] != BOARD_INDEX
        ]

        def votes(candidate: dict) -> int:
            cx0, cy0, cx1, cy1 = candidate["box"]
            return sum(1 for fx, fy in feet if cx0 <= fx <= cx1 and cy0 <= fy <= cy1)

        best = max(boards, key=lambda d: (votes(d), d["score"]))
        last_board = list(best["box"])
        age = 0
    else:
        age += 1
        if age > BOARD_MEMORY:
            last_board = None
    if last_board is None:
        return detections, None, age

    x0, y0, x1, y1 = last_board
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0, x1, y1 = x0 - dx, y0 - dy, x1 + dx, y1 + dy

    kept = []
    for detection in detections:
        if detection["label"] == BOARD_INDEX:
            kept.append(detection)
            continue
        bx0, _, bx1, by1 = detection["box"]
        foot_x = (bx0 + bx1) / 2.0
        if x0 <= foot_x <= x1 and y0 <= by1 <= y1:
            kept.append(detection)
    return kept, last_board, age


def annotate_video(
    checkpoint: Path,
    source: Path,
    destination: Path,
    *,
    threshold: float | None = None,
    top_k: int | None = None,
    stride: int = 1,
    board_gate: bool = True,
    max_seconds: float | None = None,
    device: str | None = None,
    progress=print,
) -> dict:
    """Detect on every ``stride``-th frame and write the annotated video.

    Between detection frames the previous predictions are redrawn -- at typical
    hand-held panning speeds the boxes lag imperceptibly at stride 2-3, and
    inference cost falls by the same factor.
    """
    import time

    from chesssight.train.calibrate import Calibration
    from chesssight.train.run import load_trained
    from chesssight.train.visualize import predict

    info = probe(source)
    model, processor, resolved = load_trained(checkpoint, device)
    calibration = Calibration.load(checkpoint)
    if threshold is None:
        threshold = calibration.threshold if calibration else 0.5
    if calibration is None:
        progress(
            "[chesssight] no calibration.json in this checkpoint; raw scores are "
            "squashed, so consider --top-k or `chesssight train calibrate` first"
        )

    writer = FrameWriter(destination, info.width, info.height, info.fps)
    detections: list[dict] = []
    counts: list[int] = []
    last_board: list[float] | None = None
    board_age = 0
    started = time.monotonic()
    frame_index = 0

    try:
        for frame in read_frames(source, info, max_seconds=max_seconds):
            if frame_index % stride == 0:
                detections = predict(
                    model,
                    processor,
                    frame,
                    resolved,
                    threshold=threshold,
                    top_k=top_k,
                    calibration=calibration,
                )
                if board_gate:
                    detections, last_board, board_age = gate_to_board(
                        detections, last_board, age=board_age
                    )
            pieces = sum(1 for d in detections if d["label"] != BOARD_INDEX)
            counts.append(pieces)
            has_board = any(d["label"] == BOARD_INDEX for d in detections)
            status = (
                f"ChessSight | {pieces} pieces{' + board' if has_board else ''}"
                f" | thr {threshold:.2f} | frame {frame_index}"
            )
            writer.write(draw_frame(frame, detections, status=status))
            frame_index += 1
            if frame_index % 50 == 0:
                rate = frame_index / (time.monotonic() - started)
                progress(f"[chesssight] {frame_index} frames, {rate:.1f} fps")
    finally:
        code = writer.close()

    if code != 0:
        raise VideoError(f"ffmpeg encoder exited {code}")
    elapsed = time.monotonic() - started
    return {
        "frames": frame_index,
        "fps": frame_index / elapsed if elapsed else 0.0,
        "mean_pieces": float(np.mean(counts)) if counts else 0.0,
        "output": str(destination),
    }
