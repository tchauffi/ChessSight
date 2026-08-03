"""The classification head's prior-probability bias init.

A wrong match here fails silently in both directions: missing a head leaves its
scores compressed exactly as before, and hitting a box head would corrupt the
regression branch. So the tests pin down which modules are touched, by name and
shape, on a stand-in with RT-DETR's naming.
"""

from __future__ import annotations

import math

import pytest
import torch

from chesssight.train.engine import init_head_prior
from chesssight.train.labels import NUM_DETECTION_LABELS


class StandIn(torch.nn.Module):
    """The shapes and names that matter: per-layer class heads, the encoder
    score head, a box head, and an unrelated layer that happens to share the
    class count."""

    def __init__(self):
        super().__init__()
        self.class_embed = torch.nn.ModuleList(
            [torch.nn.Linear(8, NUM_DETECTION_LABELS) for _ in range(3)]
        )
        self.enc_score_head = torch.nn.Linear(8, NUM_DETECTION_LABELS)
        self.bbox_embed = torch.nn.Linear(8, 4)
        self.unrelated = torch.nn.Linear(8, NUM_DETECTION_LABELS)


def test_every_class_head_gets_the_prior_bias():
    model = StandIn()
    init_head_prior(model, 0.01)
    expected = -math.log(0.99 / 0.01)
    for head in list(model.class_embed) + [model.enc_score_head]:
        assert torch.allclose(head.bias, torch.full_like(head.bias, expected))
        assert torch.sigmoid(head.bias).mean().item() == pytest.approx(0.01)


def test_box_and_unrelated_heads_are_left_alone():
    model = StandIn()
    before_box = model.bbox_embed.bias.clone()
    before_unrelated = model.unrelated.bias.clone()
    init_head_prior(model, 0.01)
    assert torch.equal(model.bbox_embed.bias, before_box)
    assert torch.equal(model.unrelated.bias, before_unrelated)
