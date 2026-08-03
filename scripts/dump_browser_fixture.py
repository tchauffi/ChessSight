"""Dump what the browser pipeline must reproduce, from the Python one.

The JavaScript in ``docs/demo/pipeline.js`` is a third implementation of rules
that already exist twice in Python. This writes the model outputs for a few
real photographs together with the position Python derives from them, so
``scripts/check_browser_pipeline.mjs`` can run the JavaScript over identical
inputs and compare.

Photographs are downscaled first: the fixture has to carry a greyscale copy of
the image for the orientation step, and a 3072-pixel one would be forty
megabytes of JSON. Agreement is what is being tested, not accuracy, so any
consistent input serves.

    uv run python scripts/dump_browser_fixture.py \\
        --detector ~/runs/rtdetr_corners/best \\
        --corners  ~/runs/corner_swin_v2/best \\
        --data     ~/datasets/chesssight/chessred --out fixtures.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Photographs are resized to this before anything runs.
SOURCE = 640


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--corners", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    from chesssight.data.dataset import DatasetReader
    from chesssight.data.geometry import board_to_image_homography
    from chesssight.train.corners import order_clockwise
    from chesssight.train.heatmap import preprocess as corner_preprocess
    from chesssight.train.orientation import orient, square_luminance
    from chesssight.train.position import POSITION_THRESHOLD, grid_from
    from chesssight.train.predict_position import load_reader
    from chesssight.train.visualize import predict

    reader = DatasetReader(args.data)
    pipeline = load_reader(args.detector, args.corners, "cpu")
    cases = []

    for entry in reader.entries("test")[: args.count]:
        sample = reader.load(entry.id)
        image = Image.open(reader.image_path(sample)).convert("RGB")
        image = image.resize((SOURCE, SOURCE), Image.Resampling.BILINEAR)
        width, height = image.size

        with torch.no_grad():
            heat = pipeline.corner_model(
                corner_preprocess(image, pipeline.corner_config.image_size)
            )
        heatmap = heat.float().cpu().numpy()

        detections = predict(
            pipeline.detector,
            pipeline.processor,
            image,
            pipeline.device,
            threshold=POSITION_THRESHOLD,
            calibration=pipeline.calibration,
        )

        # The same decode the reader does, so the expected values are the
        # pipeline's own rather than a second opinion.
        from chesssight.train.heatmap import quad_from_logits

        quad = quad_from_logits(
            heat.float().cpu(),
            size=image.size,
            input_size=pipeline.corner_config.image_size,
            stride=pipeline.corner_config.stride,
        )
        assert quad is not None
        quad = order_clockwise([(x, y) for x, y in quad])
        homography = board_to_image_homography(np.asarray(quad, dtype=np.float64))
        grid = grid_from(detections, homography)
        luminance = square_luminance(image, homography)
        turns, evidence = orient(grid, luminance)

        grey = (np.asarray(image.convert("L"), dtype=np.float32) / 255.0).reshape(-1)
        cases.append(
            {
                "id": entry.id,
                "width": width,
                "height": height,
                "heatSize": heatmap.shape[-1],
                "heatmap": [round(float(v), 6) for v in heatmap.reshape(-1)],
                "grey": [round(float(v), 6) for v in grey],
                "detections": [
                    {"label": d["label"], "score": d["score"], "box": d["box"]}
                    for d in detections
                ],
                "expected": {
                    "quad": [[float(x), float(y)] for x, y in quad],
                    "grid": grid,
                    "luminance": luminance.tolist(),
                    "turns": int(turns),
                    "colour": evidence["colour"],
                    "pieces": evidence["pieces"],
                    "pawns": evidence["pawns"],
                },
            }
        )
        print(f"  {entry.id}: {len(detections)} detections, turns {turns}")

    args.out.write_text(json.dumps(cases), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
