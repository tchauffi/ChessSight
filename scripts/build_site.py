"""Build the GitHub Pages site in ``docs/``.

Two phases, deliberately separable:

``--refresh`` runs the pipeline over a real test split and writes every figure
and every number into ``docs/data.json`` plus ``docs/assets/``. It needs the two
checkpoints and the ingested dataset, so it is not something CI or a
contributor can do.

Without it the script only re-renders ``docs/index.html`` from the committed
``data.json``, which needs nothing but the standard library. That is what keeps
the site editable by anyone who has cloned the repository.

    uv run python scripts/build_site.py --refresh \\
        --detector ~/runs/rtdetr_corners/best \\
        --corners  ~/runs/corner_swin_v2/best \\
        --data     ~/datasets/chesssight/chessred
    uv run python scripts/build_site.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
DATA = DOCS / "data.json"
TEMPLATE = Path(__file__).resolve().parent / "site_template.html"

#: Width the photographs are published at. A page that ships four worked
#: examples cannot afford full-resolution ones, and the boxes stay legible here.
PHOTO_WIDTH = 680
PHOTO_QUALITY = 76

#: Renders shown as training data, from the generated dataset.
SYNTH_IDS = ("000000", "000004", "000011", "000023", "000042", "000057")

#: The examples, chosen to cover a clean read and the three ways it goes wrong.
#: Ids are ChessReD sample ids; the prose is checked against the measured
#: result in :func:`refresh`, which fails rather than publish a stale claim.
EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "id": "chessred_000010",
        "verdict": "near",
        "title": "Two squares, one royal swap",
        "text": (
            "Thirty-two pieces on a glossy vinyl board, seen at an angle. "
            "Sixty-two squares are right; the two that are not are d8 and e8, "
            "where the black queen and king &mdash; standing shoulder to "
            "shoulder &mdash; are read as each other. Naming the two tallest "
            "pieces apart on a crowded back rank is what remains of this "
            "problem."
        ),
        "expect_wrong": 2,
    },
    {
        "id": "chessred_000002",
        "verdict": "good",
        "title": "A clean read",
        "text": (
            "The opening position after 1.b4 d5. All 64 squares correct, "
            "including the knight on b1 half hidden behind its rook &mdash; the "
            "square the previous model misnamed from this angle. Every piece "
            "found, every piece named."
        ),
        "expect_wrong": 0,
    },
    {
        "id": "chessred_000001",
        "verdict": "bad",
        "title": "The back rank, where the tall pieces stand together",
        "text": (
            "Three wrong squares, all on the first rank: the queen on d1 is "
            "called a king, the king on e1 shrinks to a pawn behind its "
            "neighbours, and the bishop on f1 becomes a rook. A crowded back "
            "rank seen from low down is the hardest thing on the board &mdash; "
            "queen-for-king is still the most common mistake in the whole "
            "split, though one square fewer goes wrong here than before."
        ),
        "expect_wrong": 3,
    },
    {
        "id": "chessred_000101",
        "verdict": "bad",
        "title": "Read from the wrong end",
        "text": (
            "The worst board in the split, and the pieces are not the problem: "
            "turn this answer around and it scores 64 of 64. Eight pieces on an "
            "almost empty board leave very little evidence for the last step, "
            "deciding which corner is a8, and it went the wrong way. Nothing "
            "else in 306 photographs is wrong by more than five squares."
        ),
        "expect_wrong": 16,
    },
)


# --------------------------------------------------------------------------
# refresh: needs the checkpoints
# --------------------------------------------------------------------------
def refresh(detector: Path, corners: Path, data_root: Path, split: str) -> None:
    """Measure the pipeline and write every figure and number the page shows."""
    from PIL import Image

    from chesssight.data.dataset import DatasetReader
    from chesssight.data.fen import square_name
    from chesssight.demo.render import board_svg, fit, overlay
    from chesssight.train.predict_position import load_reader

    reader = DatasetReader(data_root)
    entries = reader.entries(split)
    pipeline = load_reader(detector, corners)
    ASSETS.mkdir(parents=True, exist_ok=True)

    wanted = {example["id"]: example for example in EXAMPLES}
    figures: dict[str, dict[str, Any]] = {}
    correct_total = 0
    occupied_total = 0
    occupied_correct = 0
    wrong_counts: Counter[int] = Counter()
    kinds = Counter({"missed": 0, "spurious": 0, "misnamed": 0})
    located = 0

    for index, entry in enumerate(entries, 1):
        sample = reader.load(entry.id)
        image = Image.open(reader.image_path(sample)).convert("RGB")
        result = pipeline.read(image)
        truth = sample.grid

        if result["grid"] is None:
            wrong_counts[64] += 1
            occupied_total += sum(1 for row in truth for v in row if v)
            continue

        located += 1
        score = _score(result["grid"], truth, square_name)
        correct_total += score["correct"]
        occupied_total += score["occupied"]
        occupied_correct += score["occupied_correct"]
        kinds.update(score["kinds"])
        wrong = score["wrong"]
        wrong_counts[len(wrong)] += 1

        if entry.id in wanted:
            figures[entry.id] = _write_example(
                entry.id, image, result, truth, wrong, board_svg, fit, overlay
            )
        if index % 50 == 0:
            print(f"  {index}/{len(entries)}", flush=True)

    _copy_synth(data_root, fit)

    total = len(entries)
    payload: dict[str, Any] = {
        "photographs": total,
        "split": split,
        "located": located,
        "squares": correct_total / (64 * total),
        "exact": wrong_counts[0] / total,
        "within_one": sum(wrong_counts[n] for n in (0, 1)) / total,
        "within_three": sum(wrong_counts[n] for n in range(4)) / total,
        "occupied_named": occupied_correct / occupied_total,
        "wrong_squares": sum(kinds.values()),
        "kinds": dict(kinds),
        "histogram": [
            (
                {"wrong": n, "count": sum(v for k, v in wrong_counts.items() if k >= 6)}
                if n == 6
                else {"wrong": n, "count": wrong_counts[n]}
            )
            for n in range(7)
        ],
        "worst": max(wrong_counts),
        "examples": [
            {**example, **figures[example["id"]]}
            for example in EXAMPLES
            if example["id"] in figures
        ],
        "synth": [f"assets/synth-{n}.jpg" for n in range(1, len(SYNTH_IDS) + 1)],
    }

    mismatches = [
        f"{example['id']}: prose claims {example['expect_wrong']} wrong "
        f"squares, measured {example['wrong_count']}."
        for example in payload["examples"]
        if example["wrong_count"] != example["expect_wrong"]
    ]
    if mismatches:
        # All of them at once: fixing the prose one 5-minute re-measurement at
        # a time is how a checkpoint swap turns into an afternoon.
        raise SystemExit("Update the text.\n" + "\n".join(mismatches))

    DATA.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {DATA.relative_to(ROOT)}")


def _score(grid: Any, truth: Any, square_name: Any) -> dict[str, Any]:
    """Compare one predicted grid against the truth, square by square.

    The three ways a square can be wrong are counted separately because they
    say different things: a miss is the detector not seeing a piece, a misname
    is it seeing one and calling it wrong, and a spurious square is a piece
    invented on an empty one.
    """
    correct = occupied = occupied_correct = 0
    wrong: list[str] = []
    kinds: Counter[str] = Counter()

    for rank in range(8):
        for file in range(8):
            got, want = grid[rank][file], truth[rank][file]
            if want:
                occupied += 1
                if got == want:
                    occupied_correct += 1
            if got == want:
                correct += 1
                continue
            wrong.append(square_name(rank, file))
            if got == 0:
                kinds["missed"] += 1
            elif want == 0:
                kinds["spurious"] += 1
            else:
                kinds["misnamed"] += 1

    return {
        "correct": correct,
        "occupied": occupied,
        "occupied_correct": occupied_correct,
        "wrong": wrong,
        "kinds": kinds,
    }


def _write_example(
    sample_id: str,
    image: Any,
    result: dict[str, Any],
    truth: Any,
    wrong: list[str],
    board_svg: Any,
    fit: Any,
    overlay: Any,
) -> dict[str, Any]:
    """Write one example's four files and return how to reference them."""
    stem = sample_id.replace("chessred_", "ex")
    photo, _ = fit(image, PHOTO_WIDTH)
    photo.save(
        ASSETS / f"{stem}-photo.jpg", "JPEG", quality=PHOTO_QUALITY, optimize=True
    )
    marked = overlay(image, result["detections"], result["corners"], width=PHOTO_WIDTH)
    marked.save(
        ASSETS / f"{stem}-found.jpg", "JPEG", quality=PHOTO_QUALITY, optimize=True
    )

    wrong_set = set(wrong)
    (ASSETS / f"{stem}-read.svg").write_text(
        board_svg(result["fen"], wrong=wrong_set), encoding="utf-8"
    )
    files = {
        "photo": f"assets/{stem}-photo.jpg",
        "found": f"assets/{stem}-found.jpg",
        "read": f"assets/{stem}-read.svg",
        "truth": None,
    }
    if wrong:
        from chesssight.data.fen import grid_to_fen

        (ASSETS / f"{stem}-truth.svg").write_text(
            board_svg(grid_to_fen(truth)), encoding="utf-8"
        )
        files["truth"] = f"assets/{stem}-truth.svg"

    pieces = [d for d in result["detections"] if d["name"] != "board"]
    return {
        **files,
        "fen": str(result["fen"]).split()[0],
        "wrong": wrong,
        "wrong_count": len(wrong),
        "boxes": len(pieces),
    }


def _copy_synth(data_root: Path, fit: Any) -> None:
    """Publish a strip of training renders, resized for the web."""
    from PIL import Image

    source = data_root.parent / "train6_border" / "images"
    if not source.is_dir():
        print(f"  no renders at {source}, keeping the committed strip")
        return
    for n, name in enumerate(SYNTH_IDS, 1):
        image = Image.open(source / f"{name}.jpg").convert("RGB")
        resized, _ = fit(image, 330)
        resized.save(ASSETS / f"synth-{n}.jpg", "JPEG", quality=78, optimize=True)


# --------------------------------------------------------------------------
# render: needs only the standard library
# --------------------------------------------------------------------------
def write_constants(detector: Path | None = None) -> None:
    """Emit the browser demo's constants from the Python values.

    Typing these into the JavaScript by hand is exactly how a second copy of a
    rule drifts: the label order or the foot point would change here and the
    browser would keep reading boards with the old one, wrongly but plausibly.
    """
    from chesssight.data.geometry import BOARD_CORNERS
    from chesssight.inference.onnx import CORNER_MEAN, CORNER_STD
    from chesssight.train.labels import BOARD_INDEX, CORNER_INDEX, DETECTION_LABELS
    from chesssight.train.orientation import (
        MIN_COLOUR_MARGIN,
        PAWN_HOME_WEIGHT,
        SAMPLE_FRACTION,
    )
    from chesssight.train.position import FOOT_X, FOOT_Y, POSITION_THRESHOLD

    calibration = None
    if detector is not None:
        from chesssight.train.calibrate import Calibration

        fitted = Calibration.load(detector)
        if fitted is not None:
            calibration = {"scale": fitted.scale, "bias": fitted.bias}

    # Input geometry: read from an exported bundle when there is one, so the
    # browser never disagrees with the graph it is actually running.
    sizes = {"detectorSize": 640, "cornerSize": 448, "cornerStride": 4}
    meta_file = DOCS / "models" / "meta.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        sizes = {
            "detectorSize": meta["detector_size"],
            "cornerSize": meta["corner_size"],
            "cornerStride": meta["corner_stride"],
        }

    payload = {
        **sizes,
        "labels": list(DETECTION_LABELS),
        "boardIndex": BOARD_INDEX,
        "cornerIndex": CORNER_INDEX,
        "threshold": POSITION_THRESHOLD,
        "footX": FOOT_X,
        "footY": FOOT_Y,
        "sampleFraction": SAMPLE_FRACTION,
        "minColourMargin": MIN_COLOUR_MARGIN,
        "pawnHomeWeight": PAWN_HOME_WEIGHT,
        "boardCorners": [list(point) for point in BOARD_CORNERS],
        "cornerMean": CORNER_MEAN.tolist(),
        "cornerStd": CORNER_STD.tolist(),
        "calibration": calibration,
    }
    target = DOCS / "demo" / "constants.js"
    target.parent.mkdir(parents=True, exist_ok=True)
    if calibration is None and target.exists():
        # A render without checkpoints must not blank out a fitted calibration.
        existing = json.loads(
            target.read_text(encoding="utf-8")
            .split("=", 1)[1]
            .rsplit(";", 1)[0]
            .strip()
        )
        payload["calibration"] = existing.get("calibration")
    target.write_text(
        "// Generated by scripts/build_site.py. Do not edit.\n"
        f"export const C = {json.dumps(payload, indent=2)};\n",
        encoding="utf-8",
    )
    print(f"wrote {target.relative_to(ROOT)}")


def piece_defs() -> str:
    """The twelve piece glyphs, as reusable SVG defs.

    Taken from python-chess so the board the browser demo draws is the same
    drawing as the diagrams baked into the page, rather than a second set of
    shapes that happen to look similar.
    """
    import re

    import chess
    import chess.svg

    # Every piece type present exactly once, so every glyph gets defined.
    board = chess.Board("qrbnkbnr/pppppppp/8/8/8/8/PPPPPPPP/QRBNKBNR")
    svg = chess.svg.board(board, size=8, coordinates=False)
    match = re.search(r"<defs>(.*?)</defs>", svg, re.S)
    if match is None:
        raise SystemExit("python-chess produced no <defs> to reuse")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0" '
        'style="position:absolute" aria-hidden="true"><defs>'
        + match.group(1)  # python-chess already names them white-pawn etc.
        + "</defs></svg>"
    )


def render() -> None:
    """Turn ``data.json`` and the template into ``docs/index.html``."""
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    html = TEMPLATE.read_text(encoding="utf-8")

    kinds = payload["kinds"]
    wrong_total = payload["wrong_squares"]

    for key, value in {
        "PIECE_DEFS": piece_defs(),
        "EXAMPLES": _examples_html(payload),
        "HISTOGRAM": _histogram_html(payload),
        "SYNTH": "\n".join(
            f'        <img src="{src}" width="330" height="330" loading="lazy"'
            f' alt="A synthetic Blender render of a chessboard" />'
            for src in payload["synth"]
        ),
        "N": str(payload["photographs"]),
        "N_SQUARES": f"{payload['photographs'] * 64:,}".replace(",", "&thinsp;"),
        "LOCATED": f"{payload['located']}/{payload['photographs']}",
        "PCT_SQUARES": f"{payload['squares']:.2%}",
        "PCT_EXACT": f"{payload['exact']:.1%}",
        "PCT_WITHIN_ONE": f"{payload['within_one']:.0%}",
        "PCT_WITHIN_THREE": f"{payload['within_three']:.0%}",
        "PCT_OCCUPIED": f"{payload['occupied_named']:.1%}",
        "WRONG_TOTAL": str(wrong_total),
        "PCT_MISSED": f"{kinds['missed'] / wrong_total:.0%}",
        "PCT_MISNAMED": f"{kinds['misnamed'] / wrong_total:.0%}",
        "N_SPURIOUS": str(kinds["spurious"]),
        "WORST": str(payload["worst"]),
        "HERO_PHOTO": payload["examples"][0]["photo"],
        "HERO_BOARD": (DOCS / payload["examples"][0]["read"]).read_text(
            encoding="utf-8"
        ),
        "HERO_FEN": payload["examples"][0]["fen"],
    }.items():
        html = html.replace("{{" + key + "}}", value)

    if "{{" in html:
        raise SystemExit(f"unsubstituted token: {html[html.index('{{'):][:60]!r}")

    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    size = (DOCS / "index.html").stat().st_size / 1024
    print(f"wrote docs/index.html ({size:.0f} KB)")
    _write_inlined(html)


def _write_inlined(html: str) -> None:
    """One self-contained file, for hosts that serve a page and no assets.

    Generated from the same HTML as the site rather than from a second
    template: two copies of this page would drift, and the difference would be
    a number quietly disagreeing with itself.

    The document wrapper is dropped along with the canonical and Open Graph
    tags, which name the Pages URL and would be wrong anywhere else. Browsers
    supply html/body for a fragment, so the file still opens on its own.
    """
    import base64
    import re

    def embed(match: re.Match[str]) -> str:
        src = match.group(1)
        if src.startswith(("data:", "http")):
            return match.group(0)
        encoded = base64.b64encode((DOCS / src).read_bytes()).decode("ascii")
        return f'src="data:image/jpeg;base64,{encoded}"'

    inlined = re.sub(r'src="([^"]+\.jpg)"', embed, html)
    inlined = re.sub(
        r"<!doctype html>|</?html[^>]*>|</?head>|</?body>|"
        r'<meta (?:name="viewport"|charset)[^>]*>|'
        r'<link rel="canonical"[^>]*>|<meta (?:property="og:|name="twitter:)[^>]*>',
        "",
        inlined,
    )
    target = DOCS / "standalone.html"
    target.write_text(inlined.strip() + "\n", encoding="utf-8")
    print(f"wrote docs/standalone.html ({target.stat().st_size / 1024:.0f} KB)")


def _examples_html(payload: dict[str, Any]) -> str:
    blocks = []
    for example in payload["examples"]:
        wrong = example["wrong"]
        verdict = "exact" if not wrong else f"{len(wrong)} wrong"
        panels = [
            (example["photo"], "1", "the photograph", True),
            (example["found"], "2", "what the two models found", True),
            (example["read"], "3", "the position that came out", False),
        ]
        if example["truth"]:
            panels.append(
                (example["truth"], "4", "the position that was actually there", False)
            )

        figures = []
        for src, number, caption, is_photo in panels:
            if is_photo:
                media = (
                    f'<img src="{src}" loading="lazy" alt="{caption}" '
                    f'width="{PHOTO_WIDTH}" />'
                )
            else:
                svg = (DOCS / src).read_text(encoding="utf-8")
                media = f'<div class="board">{svg}</div>'
            figures.append(
                f'          <figure class="panel">{media}'
                f'<figcaption><span class="n">{number}</span>{caption}'
                f"</figcaption></figure>"
            )

        squares = (
            ""
            if not wrong
            else '\n        <p class="wrong-list"><span class="k">wrong at</span> '
            + " ".join(f"<code>{s}</code>" for s in wrong)
            + "</p>"
        )
        four = " panels--four" if example["truth"] else ""
        blocks.append(
            f"""      <article class="example">
        <header class="example-head">
          <h3>{example["title"]}</h3>
          <span class="verdict verdict--{example["verdict"]}">
            {64 - len(wrong)}/64 &middot; {verdict}</span>
        </header>
        <p class="example-blurb">{example["text"]}</p>
        <div class="panels{four}">
{chr(10).join(figures)}
        </div>{squares}
        <p class="fen"><span class="k">FEN</span> <code>{example["fen"]}</code></p>
      </article>"""
        )
    return "\n".join(blocks)


def _histogram_html(payload: dict[str, Any]) -> str:
    rows = payload["histogram"]
    total = payload["photographs"]
    top = max(row["count"] for row in rows)
    labels = {
        0: ("exact", "every square right"),
        6: ("6 or more", f"worst was {payload['worst']}"),
    }
    out = []
    for row in rows:
        n, count = row["wrong"], row["count"]
        label, note = labels.get(n, (f"{n} wrong", ""))
        note_html = f'<span class="bar-note">{note}</span>' if note else ""
        out.append(
            f'          <tr class="bar-row">'
            f'<th scope="row">{label}{note_html}</th>'
            f'<td class="bar-cell">'
            f'<span class="bar" style="--w:{count / top:.4f}"></span></td>'
            f'<td class="bar-value"><span class="bar-count">{count}</span>'
            f'<span class="bar-pct">{count / total:.0%}</span></td></tr>'
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-measure and redraw")
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--corners", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    if args.refresh:
        missing = [
            name
            for name in ("detector", "corners", "data")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(f"--refresh needs {', '.join('--' + m for m in missing)}")
        refresh(args.detector, args.corners, args.data, args.split)
    write_constants(args.detector)
    render()


if __name__ == "__main__":
    main()
