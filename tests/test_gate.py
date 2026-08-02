"""Refusing to report a board.

The decoder cannot fail: it returns four peaks from noise as readily as from a
board. So the property under test is that a fitted threshold turns "confidently
wrong" into "says nothing" -- and, just as importantly, that it does not achieve
that by refusing everything, which would score well on accuracy and be useless.
"""

from __future__ import annotations

import pytest

from chesssight.train.gate import USABLE_SQUARES, Gate, fit


class TestFit:
    def test_a_clean_separation_is_found(self):
        scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
        errors = [0.1, 0.15, 0.2, 2.0, 3.0, 5.0]
        gate = fit(scores, errors)
        assert 0.2 < gate.min_score <= 0.7
        assert gate.precision == 1.0 and gate.recall == 1.0

    def test_accepting_everything_is_a_candidate_but_must_earn_it(self):
        # When every board is usable, refusing anything only costs recall.
        gate = fit([0.9, 0.5, 0.3], [0.1, 0.1, 0.1])
        assert gate.min_score == 0.0
        assert gate.recall == 1.0

    def test_a_board_with_no_quad_counts_as_unusable(self):
        # `None` means the decoder returned nothing at that confidence; that is a
        # failure to report, not a missing measurement to be skipped.
        gate = fit([0.9, 0.1], [0.1, None])
        assert gate.accepts(0.9) and not gate.accepts(0.1)

    def test_the_usable_boundary_is_where_squares_start_moving(self):
        # Just inside the boundary is usable, just outside is not.
        gate = fit([0.9, 0.8], [USABLE_SQUARES - 0.01, USABLE_SQUARES + 0.01])
        assert gate.precision == 1.0
        assert gate.min_score > 0.8

    def test_mismatched_inputs_are_rejected(self):
        with pytest.raises(ValueError, match="same boards"):
            fit([0.5], [0.1, 0.2])

    def test_nothing_usable_is_an_error_not_a_silent_gate(self):
        # A gate fitted where no board was ever usable would encode "refuse
        # everything" and look like a working threshold.
        with pytest.raises(ValueError, match="no usable boards"):
            fit([0.9, 0.5], [3.0, 4.0])

    def test_it_never_refuses_every_board(self):
        # F1 is zero when recall is zero, so an all-refusing cut can never win.
        gate = fit([0.9, 0.6, 0.4, 0.2], [0.1, 2.0, 0.2, 3.0])
        assert gate.recall > 0


class TestDegenerate:
    def test_a_gate_fitted_on_clean_data_is_flagged(self):
        # Measured for real: fitting on ChessReD's val split, where 99.4% of
        # boards are usable, returns 0.000. Shipping that as protection is the
        # failure this flag exists to make visible.
        assert fit([0.9, 0.7, 0.5], [0.1, 0.1, 0.1]).is_degenerate

    def test_a_gate_that_actually_separates_is_not(self):
        assert not fit([0.9, 0.8, 0.2, 0.1], [0.1, 0.1, 2.0, 3.0]).is_degenerate


class TestRoundTrip:
    def test_a_gate_survives_saving_and_loading(self, tmp_path):
        gate = Gate(0.42, "val", 0.95, 0.9, 0.92, 300)
        gate.save(tmp_path)
        assert Gate.load(tmp_path) == gate

    def test_a_checkpoint_without_one_loads_as_none(self, tmp_path):
        # Absence must be explicit: a missing gate means "not fitted", and the
        # caller decides, rather than silently defaulting to accept-everything.
        assert Gate.load(tmp_path) is None

    def test_accepts_is_inclusive_at_the_threshold(self):
        gate = Gate(0.4, "val", 1.0, 1.0, 1.0, 10)
        assert gate.accepts(0.4)
        assert not gate.accepts(0.399)
