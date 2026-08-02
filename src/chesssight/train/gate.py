"""When to refuse to report a board's geometry.

The corner model always returns four peaks. ``topk`` does not know what a corner
is, so a photograph with no board in it yields a quad exactly as readily as one
with a board -- and everything downstream then reads a position off a homography
fitted to noise. Nothing in the pipeline objects, because a wrong homography and
a right one are the same shape.

What makes refusal possible is that the confidence of the *weakest* of the four
peaks separates the two cases. Measured on ChessReD's validation split, and on
out-of-domain photographs where the failures actually live, the boards the model
gets right carry a weakest peak around 0.5-0.7 and the ones it gets wrong sit
below 0.3. So a single threshold on that value converts a model that is
confidently wrong into one that says nothing -- which is the difference between a
pipeline that can be trusted and one that cannot.

The threshold is fitted on held-out data rather than chosen, saved next to the
checkpoint as ``gate.json``, and applied automatically when present. This mirrors
:mod:`chesssight.train.calibrate`, which does the same job for detection scores.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

GATE_FILENAME = "gate.json"

#: Corner error, in squares, above which a board counts as *not* found. A fifth
#: of a square is comfortably inside the tolerance the square-assignment step
#: needs; beyond half a square pieces start landing on the wrong square, which is
#: a wrong position rather than an imprecise one.
USABLE_SQUARES = 0.5

#: Below this a fitted threshold is not really refusing anything. Not a tuning
#: knob -- it exists so a gate fitted on a set with no failures in it can be
#: recognised as such instead of being shipped as protection it does not provide.
MEANINGFUL_MIN_SCORE = 0.05


@dataclass(frozen=True)
class Gate:
    """A minimum peak confidence, and what it bought on the split it was fit on."""

    min_score: float
    fit_split: str
    #: Of the boards accepted, the fraction that were actually usable.
    precision: float
    #: Of the usable boards, the fraction accepted. What refusing costs.
    recall: float
    f1: float
    boards: int

    @classmethod
    def load(cls, checkpoint: Path) -> Gate | None:
        path = Path(checkpoint) / GATE_FILENAME
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, checkpoint: Path) -> Path:
        path = Path(checkpoint) / GATE_FILENAME
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    def accepts(self, weakest_peak: float) -> bool:
        return weakest_peak >= self.min_score

    @property
    def is_degenerate(self) -> bool:
        """Whether this gate refuses essentially nothing.

        A threshold at or near zero is what F1 chooses when the fitting set has
        almost no failures in it -- and a gate fitted on clean data is a gate
        that will not fire on dirty data either. Measured here: fitting on
        ChessReD's validation split, where 99.4% of boards are usable, returns
        0.000. That is the correct answer to the question asked and the wrong
        answer to the question meant, so callers are given a way to notice.
        """
        return self.min_score < MEANINGFUL_MIN_SCORE


def fit(scores: list[float], errors: list[float | None], *, split: str = "val") -> Gate:
    """Choose the peak-confidence threshold that best separates usable from not.

    ``errors`` is the corner error in *squares* per board, or None where no quad
    came back at all. Maximising F1 rather than accuracy on purpose: the classes
    are lopsided -- most boards in a clean set are usable -- and accuracy is
    maximised by accepting everything, which is the behaviour this exists to stop.
    """
    if len(scores) != len(errors):
        raise ValueError("scores and errors must describe the same boards")

    usable = np.array(
        [error is not None and error <= USABLE_SQUARES for error in errors], dtype=bool
    )
    confidence = np.asarray(scores, dtype=float)
    if not usable.any():
        raise ValueError("no usable boards in the fitting set; nothing to separate")

    best = Gate(0.0, split, 0.0, 1.0, 0.0, len(scores))
    best_f1 = -1.0
    # Every observed confidence is a candidate cut, plus one below the minimum so
    # "accept everything" stays on the table and has to win on merit.
    for candidate in [0.0, *sorted(set(confidence.tolist()))]:
        accepted = confidence >= candidate
        if not accepted.any():
            continue
        precision = float(usable[accepted].mean())
        recall = float(accepted[usable].mean())
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best = Gate(
                min_score=float(candidate),
                fit_split=split,
                precision=precision,
                recall=recall,
                f1=f1,
                boards=len(scores),
            )
    return best
