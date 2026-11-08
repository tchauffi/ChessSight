"""Dump what the browser pipeline must reproduce, from the Python one.

The JavaScript in ``docs/demo/pipeline.js`` is a third implementation of rules
that already exist twice in Python. This writes the model outputs for a few
real photographs together with the position Python derives from them, so
``scripts/check_browser_pipeline.mjs`` can run the JavaScript over identical
inputs and compare.

Each photograph carries all three detector passes -- full frame, board crop,
mirrored crop -- in the coordinates of the view each pass saw, so the browser's
own shifting and un-mirroring is checked along with the fusion and material
rules, and the expected values at every stage: crop box, fused grid, grid after
the material rule, orientation, FEN.

Photographs are downscaled first: the fixture has to carry a greyscale copy of
the image for the orientation step, and a 3072-pixel one would be forty
megabytes of JSON. Agreement is what is being tested, not accuracy, so any
consistent input serves.

    uv run python scripts/dump_browser_fixture.py \\
        --detector ~/runs/rtdetr_v4/best \\
        --corners  ~/runs/corner_swin_v2/best \\
        --data     ~/datasets/chesssight/chessred --out fixtures.json \\
        --ids chessred_000004 chessred_000011 chessred_003914
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Photographs are resized to this before anything runs.
SOURCE = 640

#: Detections below this are left out of the fixture to keep it small. The
#: expected values are computed from the *same* floored lists, so the two
#: sides still see identical inputs.
FIXTURE_FLOOR = 1e-3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--corners", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--ids",
        nargs="*",
        default=[],
        help="Extra test-split ids to include -- pick ones where the material "
        "rule relabels, or its browser copy is checked only on a no-op.",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from chesssight.data.dataset import DatasetReader
    from chesssight.data.geometry import board_to_image_homography
    from chesssight.train.corners import order_clockwise
    from chesssight.train.heatmap import preprocess as corner_preprocess
    from chesssight.train.heatmap import quad_from_logits
    from chesssight.train.orientation import orient, square_luminance
    from chesssight.train.position import (
        POSITION_THRESHOLD,
        _shifted,
        board_crop,
        enforce_material,
        full_frame_weight,
        fuse,
        grid_from_scores,
        read_board,
        square_scores,
    )
    from chesssight.train.predict_position import load_reader
    from chesssight.train.visualize import predict

    reader = DatasetReader(args.data)
    pipeline = load_reader(args.detector, args.corners, "cpu")
    cases = []

    entries = reader.entries("test")
    chosen = entries[: args.count] + [
        e for e in entries if e.id in set(args.ids) and e not in entries[: args.count]
    ]
    for entry in chosen:
        sample = reader.load(entry.id)
        image = Image.open(reader.image_path(sample)).convert("RGB")
        image = image.resize((SOURCE, SOURCE), Image.Resampling.BILINEAR)
        width, height = image.size

        with torch.no_grad():
            heat = pipeline.corner_model(
                corner_preprocess(image, pipeline.corner_config.image_size)
            )
        heatmap = heat.float().cpu().numpy()

        # The same decode the reader does, so the expected values are the
        # pipeline's own rather than a second opinion.
        quad = quad_from_logits(
            heat.float().cpu(),
            size=image.size,
            input_size=pipeline.corner_config.image_size,
            stride=pipeline.corner_config.stride,
        )
        assert quad is not None
        quad = order_clockwise([(x, y) for x, y in quad])
        quad = [[float(x), float(y)] for x, y in quad]

        views: list[list[dict]] = []

        def detect(view: Image.Image, views: list[list[dict]] = views) -> list[dict]:
            found = predict(
                pipeline.detector,
                pipeline.processor,
                view,
                pipeline.device,
                threshold=0.0,
                calibration=pipeline.calibration,
            )
            kept = [
                {"label": d["label"], "score": d["score"], "box": d["box"]}
                for d in found
                if float(d["score"]) >= FIXTURE_FLOOR
            ]
            views.append(kept)
            return kept

        result = read_board(image, quad, detect, POSITION_THRESHOLD)
        full, zoomed, mirrored = views

        # The stages read_board passes through, recomputed from the recorded
        # passes so a failure in the browser can be localised.
        left, top, side = board_crop(quad, width, height)
        homography = board_to_image_homography(np.asarray(quad, dtype=np.float64))
        weight = full_frame_weight(side, width, height)
        crop_scores = square_scores(_shifted(zoomed, left, top, None), homography)
        flip_scores = square_scores(
            _shifted(mirrored, left, top, float(side)), homography
        )
        scores = fuse(
            [
                (square_scores(full, homography), weight),
                (crop_scores, 1.0),
                (flip_scores, 1.0),
            ]
        )
        fused = grid_from_scores(scores, POSITION_THRESHOLD)
        grid = enforce_material(fused, scores)
        luminance = square_luminance(image, homography)
        turns, evidence = orient(grid, luminance)
        oriented = np.rot90(np.asarray(grid), turns).tolist()
        assert oriented == result["grid"], "recomputed stages disagree with read_board"

        grey = (np.asarray(image.convert("L"), dtype=np.float32) / 255.0).reshape(-1)
        cases.append(
            {
                "id": entry.id,
                "width": width,
                "height": height,
                "heatSize": heatmap.shape[-1],
                "heatmap": [round(float(v), 6) for v in heatmap.reshape(-1)],
                "grey": [round(float(v), 6) for v in grey],
                "passes": {"full": full, "crop": zoomed, "mirrored": mirrored},
                "expected": {
                    "quad": quad,
                    "crop": [left, top, side],
                    "luminance": luminance.tolist(),
                    "fusedGrid": fused,
                    "grid": grid,
                    "turns": int(turns),
                    "colour": evidence["colour"],
                    "pieces": evidence["pieces"],
                    "pawns": evidence["pawns"],
                    "fen": result["fen"],
                    "relabelled": sum(
                        a != b
                        for row_a, row_b in zip(fused, grid, strict=True)
                        for a, b in zip(row_a, row_b, strict=True)
                    ),
                },
            }
        )
        print(
            f"  {entry.id}: {sum(len(v) for v in views)} detections over 3 passes, "
            f"turns {turns}, {cases[-1]['expected']['relabelled']} relabelled"
        )

    args.out.write_text(json.dumps(cases), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
