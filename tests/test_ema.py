"""The exponential moving average kept alongside training.

The averaged copy is what gets evaluated and saved, so a subtle error here does not
crash anything -- it silently ships slightly wrong weights. Hence tests on the exact
mixing arithmetic rather than on "it runs".
"""

from __future__ import annotations

import math

import pytest
import torch

from chesssight.train.engine import ModelEma


def tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    # A buffer, to check copy-not-average behaviour alongside parameters.
    model.register_buffer("step_count", torch.tensor(0))
    return model


class TestUpdateArithmetic:
    def test_one_update_mixes_by_the_current_decay(self):
        model = tiny_model()
        ema = ModelEma(model, decay=0.5, warmup_steps=0)
        before = {k: v.clone() for k, v in ema.module.state_dict().items()}

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        ema.update(model)

        for key, value in model.state_dict().items():
            if not value.dtype.is_floating_point:
                continue
            expected = 0.5 * before[key] + 0.5 * value
            torch.testing.assert_close(ema.module.state_dict()[key], expected)

    def test_repeated_updates_converge_to_the_live_weights(self):
        model = tiny_model()
        ema = ModelEma(model, decay=0.9, warmup_steps=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(2.0)
        for _ in range(200):
            ema.update(model)

        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                torch.testing.assert_close(
                    ema.module.state_dict()[key], value, atol=1e-4, rtol=1e-4
                )

    def test_integer_buffers_are_copied_not_averaged(self):
        model = tiny_model()
        ema = ModelEma(model, decay=0.9999, warmup_steps=0)
        model.step_count.fill_(7)
        ema.update(model)
        # An "average" of the counters 0 and 7 would be neither; the copy must win.
        assert ema.module.step_count.item() == 7


class TestWarmup:
    def test_decay_ramps_from_zero_to_the_configured_value(self):
        ema = ModelEma(tiny_model(), decay=0.9999, warmup_steps=2000)
        assert ema.current_decay() == 0.0  # nothing seen yet
        ema.updates = 2000
        assert ema.current_decay() == pytest.approx(0.9999 * (1 - math.exp(-1)))
        ema.updates = 100_000
        assert ema.current_decay() == pytest.approx(0.9999, abs=1e-6)

    def test_early_updates_track_the_model_almost_exactly(self):
        # With the ramp, step one has decay ~0: the average must jump to the live
        # weights rather than preserving the deep-copied initialisation.
        model = tiny_model()
        ema = ModelEma(model, decay=0.9999, warmup_steps=2000)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(3.0)
        ema.update(model)
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                torch.testing.assert_close(
                    ema.module.state_dict()[key], value, atol=5e-3, rtol=5e-3
                )


class TestIsolation:
    def test_the_average_never_requires_grad(self):
        ema = ModelEma(tiny_model())
        assert all(not p.requires_grad for p in ema.module.parameters())

    def test_updating_the_average_leaves_the_model_untouched(self):
        model = tiny_model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        ema = ModelEma(model, decay=0.5, warmup_steps=0)
        ema.update(model)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_training_the_model_does_not_drag_the_average_with_it(self):
        model = tiny_model()
        ema = ModelEma(model, decay=0.9999, warmup_steps=0)
        frozen = {k: v.clone() for k, v in ema.module.state_dict().items()}
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.mul_(5.0)
        # No update() call: the copy must be a copy, not a view.
        for key, value in ema.module.state_dict().items():
            torch.testing.assert_close(value, frozen[key])
