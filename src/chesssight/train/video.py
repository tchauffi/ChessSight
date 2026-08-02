"""Run the detector over a video and write an annotated copy.

Frames stream through ffmpeg as raw RGB in both directions, so no video library
joins the dependencies -- ffmpeg is already the one tool guaranteed to read whatever
a phone actually produces.

A corner-trained checkpoint (one carrying :data:`chesssight.train.labels.CORNER_INDEX`)
also yields the board quad per frame, and with it the homography -- so the overlay can
show the projected 8x8 grid and the position it reads, not only the boxes. An older
13-class checkpoint never emits that class, so the board overlay is simply absent and
nothing else changes.

Orientation is resolved by :mod:`chesssight.train.orientation` and steadied across
frames by :class:`OrientationTracker`, so the diagram is drawn the right way up and
the board's corners can be named. That decision needs pieces, so a frame with a quad
but no readable position gets the grid drawn and the corners left unlabelled.

One honest limitation, stated rather than hidden:

* The calibration was fit on ChessReD's photographs, and the further footage moves
  from those, the less the numbers mean. Measured on real tournament video:

    - Close board, vinyl set, table view (nearest the training domain): the board
      is found at 0.99 and the pieces it reports are right, but it *under*-detects
      -- roughly 10 of 15 -- because the calibrated threshold is tuned elsewhere.
      Lower ``--threshold`` or use ``--top-k``.
    - Small, blurred, near-edge-on board: it fails. Piece scores saturate near
      1.00 on people and background, and the board box itself comes back ten times
      too large, so even the board gate cannot police it. Three guards below bound
      the damage to something readable; none of them make it correct.

  The honest fix for a new domain is a handful of annotated frames from it and a
  fresh ``chesssight train calibrate`` -- or fine-tuning on it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from chesssight.data.fen import BOARD_SIZE, CLASS_TO_LETTER
from chesssight.train.labels import BOARD_INDEX, is_piece

PREDICTION_COLOR = (255, 80, 80)
BOARD_COLOR = (255, 205, 40)
GRID_COLOR = (60, 220, 255)
LIGHT_SQUARE = (226, 218, 198)
DARK_SQUARE = (126, 100, 78)


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


def draw_grid(draw: ImageDraw.ImageDraw, quad: list[list[float]], stroke: int) -> None:
    """Draw the board outline and its 8x8 grid, projected through the homography.

    The outline alone only shows that four corners were found. Projecting the
    *interior* lines shows whether the homography they imply actually agrees with
    the board in the image, which is the thing a per-square readout depends on --
    and a wrong one is obvious at a glance rather than only in a number.
    """
    from chesssight.data.geometry import apply_homography, board_to_image_homography

    try:
        homography = board_to_image_homography(np.asarray(quad, dtype=np.float64))
    except Exception:
        return  # a degenerate quad draws nothing rather than raising mid-video

    def line(start: list[float], end: list[float], width: int, colour) -> None:
        points = apply_homography(homography, [start, end])
        draw.line(
            [tuple(float(v) for v in point) for point in points],
            fill=colour,
            width=width,
        )

    for step in range(1, BOARD_SIZE):
        line([step, 0], [step, BOARD_SIZE], max(1, stroke // 2), GRID_COLOR)
        line([0, step], [BOARD_SIZE, step], max(1, stroke // 2), GRID_COLOR)

    draw.line(
        [(float(x), float(y)) for x, y in [*quad, quad[0]]],
        fill=GRID_COLOR,
        width=stroke + 1,
    )
    radius = stroke * 2
    for x, y in quad:
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=GRID_COLOR,
            outline=(10, 10, 12),
        )


#: Names of the quad's corners once orientation has put a8 first.
CORNER_NAMES = ("a8", "h8", "h1", "a1")


def label_corners(
    draw: ImageDraw.ImageDraw, quad: list[list[float]], font, stroke: int
) -> None:
    """Name the four corners. Only meaningful after orientation has run."""
    for name, (x, y) in zip(CORNER_NAMES, quad, strict=True):
        draw.text(
            (x + stroke * 3, y - stroke * 4),
            name,
            fill=GRID_COLOR,
            font=font,
            stroke_width=2,
            stroke_fill=(10, 10, 12),
        )


def draw_position(
    draw: ImageDraw.ImageDraw,
    grid: list[list[int]],
    *,
    origin: tuple[float, float],
    size: float,
    font,
) -> None:
    """Draw the read position as a small board diagram.

    Pieces are drawn as shapes, not letters. A letter has to be read; a
    silhouette is recognised, which is what a diagram glanced at beside a video
    frame needs. ``font`` is unused now and kept only so existing callers do not
    break.
    """
    from chesssight.train.glyphs import draw_piece

    cell = size / BOARD_SIZE
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            x0 = origin[0] + file * cell
            y0 = origin[1] + rank * cell
            light = (rank + file) % 2 == 0
            draw.rectangle(
                [x0, y0, x0 + cell, y0 + cell],
                fill=LIGHT_SQUARE if light else DARK_SQUARE,
            )
            occupant = grid[rank][file]
            if not occupant:
                continue
            white = CLASS_TO_LETTER[occupant].isupper()
            draw_piece(
                draw,
                occupant,
                origin=(x0, y0),
                size=cell,
                body=(248, 246, 240) if white else (26, 26, 30),
                ink=(26, 26, 30) if white else (248, 246, 240),
            )
    draw.rectangle(
        [origin[0], origin[1], origin[0] + size, origin[1] + size],
        outline=GRID_COLOR,
        width=2,
    )


def draw_frame(
    frame: Image.Image,
    predictions: list[dict],
    *,
    quad: list[list[float]] | None = None,
    grid: list[list[int]] | None = None,
    status: str = "",
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

    if quad is not None:
        draw_grid(draw, quad, stroke)
        # Only labelled when a position was read: the names come from orientation,
        # and orientation needs the pieces. A quad with no grid is still just four
        # interchangeable corners and naming them would be a guess.
        if grid is not None:
            label_corners(draw, quad, small, stroke)

    for prediction in predictions:
        if not is_piece(int(prediction["label"])):
            if prediction["label"] != BOARD_INDEX:
                continue  # corners are shown by the quad, not as boxes
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

    if grid is not None:
        size = frame.height * 0.32
        margin = frame.height * 0.03
        origin = (frame.width - size - margin, frame.height - size - margin)
        try:
            glyphs = ImageFont.load_default(size=max(9, int(size / BOARD_SIZE * 0.72)))
        except TypeError:
            glyphs = ImageFont.load_default()
        draw_position(draw, grid, origin=origin, size=size, font=glyphs)
        caption = "read position   a8 top-left"
        draw.text(
            (origin[0], origin[1] - (small.size if hasattr(small, "size") else 12) - 4),
            caption,
            fill=GRID_COLOR,
            font=small,
            stroke_width=2,
            stroke_fill=(10, 10, 12),
        )

    if status:
        draw.rectangle(
            [0, 0, frame.width, (small.size if hasattr(small, "size") else 12) + 10],
            fill=(10, 10, 12),
        )
        draw.text((8, 4), status, fill=(230, 230, 230), font=small)
    return frame


#: A complete chess set has 32 pieces, so no frame can legitimately contain more.
#: This is a physical constraint, not a tuned threshold.
MAX_PIECES = 32

#: IoU above which two piece boxes are treated as the same detection. RT-DETR's
#: one-to-one matching is *supposed* to make NMS unnecessary, and in-domain it does
#: -- but out of domain it emits dozens of near-duplicate boxes per object, so the
#: guarantee does not survive the domain shift.
NMS_IOU = 0.55


def suppress_overlaps(
    detections: list[dict], *, iou_threshold: float = NMS_IOU
) -> list[dict]:
    """Class-agnostic greedy NMS over piece detections; the board passes through.

    Class-agnostic rather than per-class on purpose: the duplicates seen out of
    domain disagree about *class* as well as position -- one physical piece drawing
    a "white knight" and a "white queen" box on the same pixels -- so per-class
    suppression would keep both.

    The board and the corners pass through untouched: they are not pieces, and
    corner boxes legitimately overlap the piece standing on that square.
    """
    others = [d for d in detections if not is_piece(int(d["label"]))]
    pieces = sorted(
        (d for d in detections if is_piece(int(d["label"]))),
        key=lambda d: float(d["score"]),
        reverse=True,
    )

    kept: list[dict] = []
    for candidate in pieces:
        cx0, cy0, cx1, cy1 = candidate["box"]
        area = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0)
        duplicate = False
        for accepted in kept:
            ax0, ay0, ax1, ay1 = accepted["box"]
            ix = max(0.0, min(cx1, ax1) - max(cx0, ax0))
            iy = max(0.0, min(cy1, ay1) - max(cy0, ay0))
            intersection = ix * iy
            if intersection <= 0:
                continue
            other = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
            union = area + other - intersection
            if union > 0 and intersection / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept + others


def cap_pieces(detections: list[dict], limit: int = MAX_PIECES) -> list[dict]:
    """Keep at most ``limit`` piece detections, highest-scoring first.

    A chess set has 32 pieces. Emitting 288 is not a borderline error to be tuned
    away, it is impossible, so the count is bounded by the physics rather than by
    a confidence threshold that has already been shown to drift out of domain.

    The cap is on *pieces*: capping corners too would starve the quad selector,
    which needs its four candidates to survive a frame full of piece detections.
    """
    others = [d for d in detections if not is_piece(int(d["label"]))]
    pieces = sorted(
        (d for d in detections if is_piece(int(d["label"]))),
        key=lambda d: float(d["score"]),
        reverse=True,
    )[:limit]
    return pieces + others


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
            if is_piece(int(d["label"]))
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

    # Only the selected board survives: drawing every candidate implies the model
    # found several boards, when in fact one was chosen and the rest discarded.
    chosen = dict(best) if boards else None
    if chosen is not None:
        chosen["box"] = list(last_board)
    kept = [chosen] if chosen is not None else []
    for detection in detections:
        if detection["label"] == BOARD_INDEX:
            continue
        bx0, by0, bx1, by1 = detection["box"]
        # A piece is gated on its foot -- where it stands. A corner has no foot:
        # it is a point, marked by a small box around it, and it sits *on* the
        # board's boundary, so gating it on the bottom edge would reject the two
        # near corners of every board.
        point = (
            ((bx0 + bx1) / 2.0, by1)
            if is_piece(int(detection["label"]))
            else ((bx0 + bx1) / 2.0, (by0 + by1) / 2.0)
        )
        if x0 <= point[0] <= x1 and y0 <= point[1] <= y1:
            kept.append(detection)
    return kept, last_board, age


#: IoU above which a detection is taken to be the same physical piece as an
#: existing track. Deliberately loose: pieces barely move between frames, so a
#: near-miss is a jittering box on one piece far more often than it is two pieces.
MATCH_IOU = 0.35

#: A track must clear the entry threshold to be *created*, but only this fraction
#: of it to *survive*. Two thresholds rather than one is the whole point: a single
#: threshold makes a piece scoring around it blink on and off every other frame,
#: which is most of the flicker.
KEEP_RATIO = 0.5

#: Detections a track needs before it is drawn, and detection frames it may coast
#: through unseen before it is dropped. The first suppresses one-frame false
#: positives, the second bridges the momentary misses of a real piece -- a hand
#: passing over the board, a blurred frame.
MIN_HITS = 2
MAX_MISSES = 4

#: Weight on the remembered box when a track is updated. Pieces are static for
#: most of a game, so a heavily-damped box removes the few-pixel per-frame wobble
#: without lagging visibly when one is actually moved.
BOX_SMOOTHING = 0.7


def iou(first: list[float], second: list[float]) -> float:
    """Intersection over union of two boxes."""
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = ix * iy
    if intersection <= 0:
        return 0.0
    union = (
        max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
        + max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    """One physical piece, followed across frames."""

    box: list[float]
    score: float
    hits: int = 1
    misses: int = 0
    #: Score-weighted votes per class. A piece whose class flickers between two
    #: near-tied labels gets one steady answer instead of alternating captions.
    votes: dict[int, float] = field(default_factory=dict)

    @property
    def label(self) -> int:
        return max(self.votes, key=lambda key: self.votes[key])


class PieceTracker:
    """Turn per-frame detections into steady tracks.

    Three independent causes of flicker, three mechanisms:

    * a score oscillating around the threshold -> enter high, survive low;
    * a class label alternating between near-tied guesses -> vote over the track's
      whole history rather than trusting the current frame;
    * box coordinates wobbling a few pixels -> exponentially smooth them.

    Association is greedy by IoU with no motion model. A Kalman filter earns its
    keep when targets move predictably between frames; chess pieces are stationary
    for seconds at a time and then teleport when a hand moves them, which is the
    one case a constant-velocity prior gets *wrong*.
    """

    def __init__(
        self,
        *,
        enter_score: float,
        keep_ratio: float = KEEP_RATIO,
        match_iou: float = MATCH_IOU,
        min_hits: int = MIN_HITS,
        max_misses: int = MAX_MISSES,
        smoothing: float = BOX_SMOOTHING,
    ) -> None:
        self.enter_score = enter_score
        self.keep_score = enter_score * keep_ratio
        self.match_iou = match_iou
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.smoothing = smoothing
        self.tracks: list[Track] = []
        #: Label -> display name, accumulated across frames. A coasting track may
        #: hold a class that no detection in the current frame carries.
        self.names: dict[int, str] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        """Fold one frame's detections in and return what should be drawn."""
        pieces = sorted(
            (d for d in detections if is_piece(int(d["label"]))),
            key=lambda d: float(d["score"]),
            reverse=True,
        )
        self.names.update({d["label"]: d["name"] for d in pieces})

        unmatched = list(self.tracks)
        for detection in pieces:
            if detection["score"] < self.keep_score:
                continue
            best, best_iou = None, self.match_iou
            for track in unmatched:
                overlap = iou(detection["box"], track.box)
                if overlap >= best_iou:
                    best, best_iou = track, overlap
            if best is None:
                if detection["score"] >= self.enter_score:
                    self.tracks.append(
                        Track(
                            box=list(detection["box"]),
                            score=float(detection["score"]),
                            votes={detection["label"]: float(detection["score"])},
                        )
                    )
                continue

            unmatched.remove(best)
            keep = self.smoothing
            best.box = [
                keep * old + (1.0 - keep) * new
                for old, new in zip(best.box, detection["box"], strict=True)
            ]
            best.score = keep * best.score + (1.0 - keep) * float(detection["score"])
            best.votes[detection["label"]] = best.votes.get(
                detection["label"], 0.0
            ) + float(detection["score"])
            best.hits += 1
            best.misses = 0

        for track in unmatched:
            track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        drawn = [
            {
                "box": list(track.box),
                "label": track.label,
                "name": self.names.get(track.label, str(track.label)),
                "score": track.score,
            }
            for track in self.tracks
            if track.hits >= self.min_hits
        ]
        # The board is not tracked here: it is one object, already stabilised by
        # gate_to_board's memory, and smoothing it would lag a camera cut. Corners
        # pass through for the same reason -- BoardTracker steadies them instead,
        # as a quad rather than as four independent boxes.
        return drawn + [d for d in detections if not is_piece(int(d["label"]))]


#: Weight on the remembered quad when new corners arrive. Higher than the piece
#: smoothing: the whole readout hangs off this one quadrilateral, and a corner
#: jumping half a square repaints every square on the diagram at once.
CORNER_SMOOTHING = 0.8

#: Detection frames the last good quad survives without being re-found. A hand
#: crossing the board hides a corner for a few frames and the geometry is still
#: valid; a scene cut hides it forever and the stale quad must expire.
CORNER_MEMORY = 12


def align_quad(
    quad: list[list[float]], previous: list[list[float]]
) -> list[list[float]]:
    """Roll ``quad`` to the cyclic order that best matches ``previous``.

    Both are already clockwise, so only the starting corner can differ -- and it
    does: :func:`chesssight.train.corners.order_clockwise` starts at whichever
    corner is nearest the bounding box's top-left, which swaps between two
    near-tied corners as the camera moves. Smoothing without this re-alignment
    averages one physical corner against another and collapses the quad.
    """
    best, best_cost = quad, None
    for shift in range(len(quad)):
        rolled = quad[shift:] + quad[:shift]
        cost = sum(
            float(np.hypot(a[0] - b[0], a[1] - b[1]))
            for a, b in zip(rolled, previous, strict=True)
        )
        if best_cost is None or cost < best_cost:
            best, best_cost = rolled, cost
    return best


class BoardTracker:
    """Hold one steady board quad across frames.

    The per-frame quad is usable but jittery, and unlike a piece box a jittery
    quad is not a cosmetic problem: it feeds the homography, so a few pixels of
    corner noise moves squares on the readout.
    """

    def __init__(
        self, *, smoothing: float = CORNER_SMOOTHING, memory: int = CORNER_MEMORY
    ) -> None:
        self.smoothing = smoothing
        self.memory = memory
        self.quad: list[list[float]] | None = None
        self.age = 0

    def update(self, detections: list[dict]) -> list[list[float]] | None:
        """Fold one frame's corner detections in; return the quad to use, if any."""
        from chesssight.train.corners import select_quad

        found = select_quad(detections)
        if found is None:
            self.age += 1
            if self.age > self.memory:
                self.quad = None
            return self.quad

        self.age = 0
        if self.quad is not None:
            found = align_quad(found, self.quad)
            keep = self.smoothing
            found = [
                [
                    keep * old[0] + (1.0 - keep) * new[0],
                    keep * old[1] + (1.0 - keep) * new[1],
                ]
                for old, new in zip(self.quad, found, strict=True)
            ]
        self.quad = found
        return self.quad


#: How fast a frame's orientation vote decays. Orientation is a property of the
#: board, not of the frame, so evidence should accumulate over seconds; but it
#: must still be able to change after a cut to a camera on the other side.
ORIENTATION_DECAY = 0.9


class OrientationTracker:
    """Hold one steady answer to "which corner is a8" across frames.

    Per-frame orientation is decided on evidence that varies with exposure,
    occlusion and which pieces happen to be detected, so deciding independently
    every frame makes the diagram spin -- and a diagram that rotates between
    frames is worse than no diagram, because each individual frame still looks
    authoritative.

    Votes are weighted by the frame's own margin, so confident frames count for
    more than ambiguous ones, and decayed so a scene cut is eventually forgotten.
    This is only coherent because :class:`BoardTracker` keeps the quad's starting
    corner stable; against a quad that reorders itself, ``turns`` would mean
    something different every frame.
    """

    def __init__(self, decay: float = ORIENTATION_DECAY) -> None:
        self.decay = decay
        self.votes = [0.0, 0.0, 0.0, 0.0]

    def update(self, turns: int, evidence: dict[str, float]) -> int:
        self.votes = [vote * self.decay for vote in self.votes]
        # A tie carries no information about which way round the board is, but it
        # is still evidence that *this* pair of candidates is the right one, so a
        # floor keeps the vote from vanishing on an empty board.
        self.votes[turns] += max(0.05, float(evidence.get("margin", 0.0)))
        return max(range(4), key=lambda index: self.votes[index])


def drop_weak_pieces(detections: list[dict], threshold: float) -> list[dict]:
    """Remove pieces below ``threshold``; board and corners pass through.

    Used when detection ran below the display threshold to feed the corner
    ranking, so the pieces that floor admitted do not end up drawn.
    """
    return [
        detection
        for detection in detections
        if not is_piece(int(detection["label"]))
        or float(detection["score"]) >= threshold
    ]


def read_geometry(
    frame: Image.Image,
    detections: list[dict],
    board: BoardTracker | None,
    orientation: OrientationTracker,
) -> tuple[list[list[float]] | None, list[list[int]] | None, bool]:
    """Board quad, oriented position, and whether geometry was found this frame.

    The flag is returned separately from ``grid`` because it is what the run's
    geometry rate counts, and it must mean "this frame yielded a board" -- not
    "something is available to draw", which a remembered quad would also satisfy.
    """
    from chesssight.train.position import read_position

    if board is None:
        return None, None, False
    quad = board.update(detections)
    grid = read_position(detections, quad)
    found = grid is not None
    grid, quad = settle_orientation(frame, grid, quad, orientation)
    return quad, grid, found


def settle_orientation(
    frame: Image.Image,
    grid: list[list[int]] | None,
    quad: list[list[float]] | None,
    tracker: OrientationTracker,
) -> tuple[list[list[int]] | None, list[list[float]] | None]:
    """Decide this frame's orientation, then defer to what the run agreed on.

    The grid and the quad turn together, so the diagram and the labels drawn on
    the board never disagree about which corner is a8.
    """
    if grid is None or quad is None:
        return grid, quad

    import numpy as np

    from chesssight.data.geometry import board_to_image_homography
    from chesssight.train.orientation import orient, rotate, square_luminance

    try:
        homography = board_to_image_homography(np.asarray(quad, dtype=np.float64))
    except Exception:
        return grid, quad
    turns, evidence = orient(grid, square_luminance(frame, homography))
    settled = tracker.update(turns, evidence)
    return rotate(grid, settled).tolist(), quad[settled:] + quad[:settled]


def status_line(
    detections: list[dict],
    grid: list[list[int]] | None,
    threshold: float,
    frame_index: int,
) -> str:
    """The banner drawn across the top of each frame."""
    pieces = sum(1 for d in detections if is_piece(int(d["label"])))
    has_board = any(d["label"] == BOARD_INDEX for d in detections)
    status = (
        f"ChessSight | {pieces} pieces{' + board' if has_board else ''}"
        f" | thr {threshold:.2f} | frame {frame_index}"
    )
    if grid is not None:
        occupied = sum(1 for row in grid for value in row if value)
        status += f" | {occupied} squares read"
    return status


def annotate_video(
    checkpoint: Path,
    source: Path,
    destination: Path,
    *,
    threshold: float | None = None,
    top_k: int | None = None,
    stride: int = 1,
    board_gate: bool = True,
    smooth: bool = True,
    corners: bool = True,
    max_pieces: int = MAX_PIECES,
    max_seconds: float | None = None,
    device: str | None = None,
    progress=print,
) -> dict:
    """Detect on every ``stride``-th frame and write the annotated video.

    Between detection frames the previous predictions are redrawn -- at typical
    hand-held panning speeds the boxes lag imperceptibly at stride 2-3, and
    inference cost falls by the same factor.

    With ``smooth``, detections are associated across frames by a
    :class:`PieceTracker` rather than drawn independently. Detection then runs at a
    *lower* threshold than the one displayed, so the tracker can see the marginal
    scores it needs to hold a piece through a bad frame; nothing below the display
    threshold is ever drawn on its own.

    With ``corners``, corner detections are turned into a board quad and the
    position it implies is drawn alongside. A checkpoint without the corner class
    never emits one, so this costs nothing and simply shows nothing.
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

    tracker = PieceTracker(enter_score=threshold) if smooth else None
    # Detect below what is displayed, for two independent reasons. The tracker
    # needs the marginal scores to hold a piece through a blurred frame. And the
    # corner class must not be judged by a threshold calibrated on *pieces*: a
    # board has exactly four corners, so they are chosen by rank among themselves
    # in `select_quad`, and thresholding them first simply throws away the
    # candidates it would have ranked. Sub-threshold *pieces* are dropped again
    # below when there is no tracker to justify keeping them.
    detect_threshold = threshold * KEEP_RATIO if (smooth or corners) else threshold

    # `smooth` governs every temporal assumption, not only the piece tracks. A
    # quad carried over from the previous frame is only valid if the previous
    # frame shows the same board from nearly the same place; feed in a sequence
    # of unrelated views and the EMA blends viewpoints into a quad that matches
    # none of them, which draws a grid lying across the squares instead of along
    # them. Orientation still accumulates: it is a property of the board, and
    # survives a change of camera.
    board = (
        BoardTracker(
            smoothing=CORNER_SMOOTHING if smooth else 0.0,
            memory=CORNER_MEMORY if smooth else 0,
        )
        if corners
        else None
    )
    orientation = OrientationTracker()
    writer = FrameWriter(destination, info.width, info.height, info.fps)
    detections: list[dict] = []
    counts: list[int] = []
    quad: list[list[float]] | None = None
    grid: list[list[int]] | None = None
    read_frames_count = 0
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
                    threshold=detect_threshold,
                    top_k=top_k,
                    calibration=calibration,
                )
                detections = suppress_overlaps(detections)
                if board_gate:
                    detections, last_board, board_age = gate_to_board(
                        detections, last_board, age=board_age
                    )
                # Track before capping: the cap keeps the 32 highest *instantaneous*
                # scores, which is exactly the ranking the tracker exists to steady.
                if tracker is not None:
                    detections = tracker.update(detections)
                detections = cap_pieces(detections, max_pieces)
                quad, grid, found = read_geometry(frame, detections, board, orientation)
                read_frames_count += found
                if tracker is None and detect_threshold < threshold:
                    detections = drop_weak_pieces(detections, threshold)
            counts.append(sum(1 for d in detections if is_piece(int(d["label"]))))
            status = status_line(detections, grid, threshold, frame_index)
            writer.write(
                draw_frame(frame, detections, quad=quad, grid=grid, status=status)
            )
            frame_index += 1
            if frame_index % 50 == 0:
                rate = frame_index / (time.monotonic() - started)
                progress(f"[chesssight] {frame_index} frames, {rate:.1f} fps")
    finally:
        code = writer.close()

    if code != 0:
        raise VideoError(f"ffmpeg encoder exited {code}")
    elapsed = time.monotonic() - started
    detected_on = (frame_index + stride - 1) // stride
    return {
        "frames": frame_index,
        "fps": frame_index / elapsed if elapsed else 0.0,
        "mean_pieces": float(np.mean(counts)) if counts else 0.0,
        # What fraction of the frames we actually ran the detector on yielded a
        # board quad. This is the honest denominator: frames that were only
        # redrawn never had a chance to find one.
        "geometry_frames": read_frames_count,
        "geometry_rate": read_frames_count / detected_on if detected_on else 0.0,
        "output": str(destination),
    }
