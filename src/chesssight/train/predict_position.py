"""One photograph in, one position out.

This is what the project is for, assembled from the parts that measured best:
the RT-DETR detector finds the pieces, the corner heatmap finds the board, the
homography assigns pieces to squares, and colour parity plus piece placement
decide which corner is a8. Everything else in the repository exists to train or
evaluate the two models this function loads.

The pipeline's measured accuracy on ChessReD's held-out test split, with both
models doing their own work end to end: 99.32% of squares, 71.24% of boards
exactly right, geometry found on 306 of 306 photographs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from chesssight.data.fen import grid_to_fen
from chesssight.train.position import POSITION_THRESHOLD

__all__ = ["POSITION_THRESHOLD", "PositionReader", "load_reader"]


@dataclass
class PositionReader:
    """The loaded pipeline. Build one with :func:`load_reader` and reuse it."""

    # Typed loosely: the members come from transformers and the heatmap loader,
    # and pinning their classes here would couple this module to both.
    detector: Any
    processor: Any
    calibration: Any
    corner_model: Any
    corner_config: Any
    device: torch.device
    #: Score a detection must reach to be considered for a square. Defaults to
    #: :data:`POSITION_THRESHOLD`; pass ``None`` to fall back to the
    #: calibration's own operating point, which is fitted for detection F1 --
    #: a different objective, and a much stricter one.
    threshold: float | None = POSITION_THRESHOLD

    @torch.no_grad()
    def read(self, image) -> dict:
        """Read one PIL image.

        Returns a dict rather than raising on failure, because "no board found"
        is an answer, not an error: ``corners`` is None and there is no grid.
        """
        import numpy as np

        from chesssight.data.geometry import board_to_image_homography
        from chesssight.train.heatmap import predict_quad
        from chesssight.train.orientation import orient_position
        from chesssight.train.position import grid_from
        from chesssight.train.visualize import predict

        quad = predict_quad(
            self.corner_model,
            image,
            self.device,
            input_size=self.corner_config.image_size,
            stride=self.corner_config.stride,
        )
        if quad is None:
            return {"corners": None, "grid": None, "fen": None, "detections": []}

        if self.threshold is not None:
            threshold = self.threshold
        elif self.calibration:
            threshold = self.calibration.threshold
        else:
            threshold = 0.3
        detections = predict(
            self.detector,
            self.processor,
            image,
            self.device,
            threshold=threshold,
            calibration=self.calibration,
        )
        homography = board_to_image_homography(np.asarray(quad, dtype=np.float64))
        grid = grid_from(detections, homography)
        grid, quad, evidence = orient_position(grid, quad, image)

        return {
            "corners": quad,
            "grid": grid,
            "fen": grid_to_fen(grid),
            "detections": detections,
            "orientation_evidence": evidence,
        }


def load_reader(
    detector: Path,
    corners: Path,
    device: str | None = None,
    threshold: float | None = POSITION_THRESHOLD,
) -> PositionReader:
    """Load both checkpoints once; the reader is then cheap per image.

    ``threshold`` defaults to :data:`POSITION_THRESHOLD`. Passing ``None``
    restores the calibration's detection operating point, which reads boards
    considerably worse.
    """
    from chesssight.train.calibrate import Calibration
    from chesssight.train.heatmap import load as load_corners
    from chesssight.train.run import load_trained

    model, processor, resolved = load_trained(Path(detector), device)
    corner_model, corner_config = load_corners(Path(corners), resolved)
    return PositionReader(
        detector=model,
        processor=processor,
        calibration=Calibration.load(Path(detector)),
        corner_model=corner_model,
        corner_config=corner_config,
        device=resolved,
        threshold=threshold,
    )
