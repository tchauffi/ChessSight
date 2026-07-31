"""Material choices resolved on the project side, before Blender sees anything.

The Blender half of the material layer cannot be imported without ``bpy``, so it is
covered by the render smoke test. Everything *decided* here can be tested directly,
and these are the decisions that silently produce a wrong-looking dataset rather
than an error: which style a surface gets, how far its figure departs from the body
colour, and whether the veneer pair is a pair at all.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from chesssight.data.fen import BOARD_SIZE
from chesssight.synth.config import GeneratorConfig
from chesssight.synth.randomize import (
    MATERIAL_KINDS,
    choose_material_style,
    choose_table_texture,
    choose_veneers,
    resolve_board,
    resolve_clock,
    resolve_pieces,
)
from tests.test_textures import write_set

LIGHT = [0.88, 0.84, 0.76]
DARK = [0.07, 0.06, 0.06]


def config(**overrides) -> GeneratorConfig:
    return GeneratorConfig.model_validate(overrides)


class TestStyleSelection:
    def test_only_known_kinds_are_drawn(self):
        weights = {"plastic": 1.0, "wood": 1.0, "marble": 1.0, "plain": 1.0}
        kinds = {
            choose_material_style(weights, LIGHT, random.Random(seed)).kind
            for seed in range(60)
        }
        assert kinds <= MATERIAL_KINDS

    def test_weights_are_honoured(self):
        weights = {"wood": 9.0, "marble": 1.0}
        drawn = [
            choose_material_style(weights, LIGHT, random.Random(seed)).kind
            for seed in range(400)
        ]
        assert 0.75 < drawn.count("wood") / len(drawn) < 0.98

    def test_an_unknown_kind_is_rejected_rather_than_ignored(self):
        # Falling through to the plain solid would render a whole dataset with none
        # of the style asked for, and nothing would say so.
        with pytest.raises(ValueError, match="unknown material style"):
            choose_material_style({"wodo": 1.0}, LIGHT, random.Random(0))

    @pytest.mark.parametrize("kind", ["marble", "wood"])
    def test_figure_parameters_stay_in_their_measured_ranges(self, kind: str):
        # Both ranges were fixed by rendering a parameter sweep: past them, marble
        # becomes pinstripes and wood becomes a barcode.
        for seed in range(40):
            style = choose_material_style({kind: 1.0}, LIGHT, random.Random(seed))
            assert 0.0 < style.scale <= 6.0
            assert 0.0 <= style.contrast <= 0.5


class TestContrastIsRelative:
    """The bug that striped the black pieces while the white ones looked right."""

    def test_one_style_serves_both_a_light_and_a_dark_set(self):
        style = choose_material_style({"wood": 1.0}, LIGHT, random.Random(3))
        # The style carries a *strength*, not a second colour, so the same value can
        # be resolved against either side. An absolute accent tuned on near-white
        # boxwood is ten times lighter than near-black ebony.
        assert hasattr(style, "contrast")
        assert not hasattr(style, "accent")

    def test_contrast_is_bounded_below_a_half(self):
        for seed in range(60):
            style = choose_material_style({"wood": 1.0}, LIGHT, random.Random(seed))
            assert style.contrast < 0.5


class TestVeneers:
    def veneer_config(self, root: Path) -> GeneratorConfig:
        return config(
            scene={"texture_dir": str(root)},
            pieces={"material_styles": {"wood": 1.0}},
        )

    def test_both_sides_get_maps_when_the_pair_is_present(self, tmp_path: Path):
        write_set(tmp_path, "oak_veneer_01")
        write_set(tmp_path, "rosewood_veneer1")
        cfg = self.veneer_config(tmp_path)
        style = choose_material_style({"wood": 1.0}, LIGHT, random.Random(0))
        light, dark = choose_veneers(cfg, style, cfg.pieces)
        assert light and dark
        assert light != dark

    def test_a_non_wood_style_gets_no_veneer(self, tmp_path: Path):
        write_set(tmp_path, "oak_veneer_01")
        write_set(tmp_path, "rosewood_veneer1")
        cfg = self.veneer_config(tmp_path)
        style = choose_material_style({"marble": 1.0}, LIGHT, random.Random(0))
        assert choose_veneers(cfg, style, cfg.pieces) == (None, None)

    def test_missing_textures_fall_back_rather_than_raising(self, tmp_path: Path):
        # The no-assets path again: procedural grain still renders a valid scene.
        cfg = self.veneer_config(tmp_path)
        style = choose_material_style({"wood": 1.0}, LIGHT, random.Random(0))
        assert choose_veneers(cfg, style, cfg.pieces) == (None, None)

    def test_no_texture_dir_configured_is_safe(self):
        cfg = config(pieces={"material_styles": {"wood": 1.0}})
        style = choose_material_style({"wood": 1.0}, LIGHT, random.Random(0))
        assert choose_veneers(cfg, style, cfg.pieces) == (None, None)


class TestTableTexture:
    def test_a_texture_is_chosen_when_the_directory_has_one(self, tmp_path: Path):
        write_set(tmp_path, "oak_veneer_01")
        scene = config(scene={"texture_dir": str(tmp_path)}).scene
        chosen = choose_table_texture(scene, random.Random(0), table_size=30.0)
        assert chosen is not None
        assert chosen.slug == "oak_veneer_01"

    def test_scale_is_repeats_across_the_table_not_per_unit(self, tmp_path: Path):
        # Stated per table and converted here, because the table is 20-45 squares
        # across: left per-unit, a scale of 1 tiles the map thirty times and oak
        # comes out looking like fine fabric.
        write_set(tmp_path, "oak_veneer_01")
        scene = config(
            scene={
                "texture_dir": str(tmp_path),
                "texture_scale": {"min": 3.0, "max": 3.0},
            }
        ).scene
        chosen = choose_table_texture(scene, random.Random(0), table_size=30.0)
        assert chosen is not None
        assert chosen.scale == pytest.approx(3.0 / 30.0)

    def test_probability_zero_never_draws_one(self, tmp_path: Path):
        write_set(tmp_path, "oak_veneer_01")
        scene = config(
            scene={"texture_dir": str(tmp_path), "texture_probability": 0.0}
        ).scene
        for seed in range(20):
            assert (
                choose_table_texture(scene, random.Random(seed), table_size=30.0)
                is None
            )

    def test_an_empty_directory_falls_back_to_flat_colour(self, tmp_path: Path):
        scene = config(scene={"texture_dir": str(tmp_path)}).scene
        assert choose_table_texture(scene, random.Random(0), table_size=30.0) is None


class TestClock:
    """A clock stands beside the board, on the left or the right, never on it."""

    def clocks(self, count: int = 300, **scene):
        cfg = config(scene={"clock_probability": 1.0, **scene})
        return [
            resolve_clock(cfg.scene, random.Random(seed), square_size=1.0)
            for seed in range(count)
        ]

    @staticmethod
    def footprint(clock) -> list[tuple[float, float]]:
        """The four corners of the clock's base, in world coordinates."""
        depth = clock.width * (0.69 if clock.kind == "digital" else 0.62)
        theta = math.radians(clock.rotation_deg)
        corners = []
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                lx, ly = sx * clock.width, sy * depth
                corners.append(
                    (
                        clock.x + lx * math.cos(theta) - ly * math.sin(theta),
                        clock.y + lx * math.sin(theta) + ly * math.cos(theta),
                    )
                )
        return corners

    def test_no_clock_ever_overlaps_the_board(self):
        # The bug this pins: the clock's *width* axis pointed at the board rather
        # than along the edge, so a four-square-wide clock centred just outside the
        # edge reached back across two ranks and sat on the squares.
        half = BOARD_SIZE / 2.0
        for clock in self.clocks():
            assert clock is not None
            for x, y in self.footprint(clock):
                assert max(abs(x), abs(y)) > half, (
                    f"{clock.kind} clock corner ({x:.2f}, {y:.2f}) is on the board"
                )

    def test_it_stands_to_the_left_or_the_right(self):
        # Players sit at the two ends; the clock goes beside the board, not in
        # front of a player.
        for clock in self.clocks():
            assert clock is not None
            assert abs(clock.x) > abs(clock.y)

    def test_both_sides_are_used(self):
        sides = {clock.x > 0 for clock in self.clocks() if clock}
        assert sides == {True, False}

    def test_both_models_are_built(self):
        kinds = {clock.kind for clock in self.clocks() if clock}
        assert kinds == {"analogue", "digital"}

    def test_the_long_axis_runs_along_the_edge(self):
        # A quarter turn from the edge normal: the face looks away from the board,
        # which is how a clock sits where both players can read it.
        for clock in self.clocks(count=60):
            assert clock is not None
            offset = min(
                abs((clock.rotation_deg - reference) % 180.0) for reference in (90.0,)
            )
            assert offset < 15.0 or offset > 165.0

    def test_probability_zero_never_places_one(self):
        scene = config(scene={"clock_probability": 0.0}).scene
        for seed in range(40):
            assert resolve_clock(scene, random.Random(seed), square_size=1.0) is None

    def test_width_matches_a_real_clock(self):
        # Real clocks are 165-220 mm against a ~50 mm square, so three to four and a
        # half squares. A clock at piece scale is a different object entirely.
        for clock in self.clocks(count=60):
            assert clock is not None
            assert 2.5 < clock.width < 5.0


class TestResolvedIntoTheSpec:
    def test_the_board_carries_a_style(self):
        board = resolve_board(config(), random.Random(0))
        assert board.material is not None
        assert board.material.kind in MATERIAL_KINDS

    def test_board_grain_is_scaled_into_board_units(self):
        # The board is eight squares across, so a per-square scale runs at roughly
        # twenty times the frequency the same style uses on a piece.
        cfg = config(board={"grain_scale": {"min": 4.0, "max": 4.0}})
        board = resolve_board(cfg, random.Random(0))
        assert board.material is not None
        assert board.material.scale == pytest.approx(4.0 / 8)

    def test_pieces_carry_a_style_and_a_veneer_slot(self):
        from chesssight.data.fen import STARTING_FEN, fen_to_grid

        pieces = resolve_pieces(config(), fen_to_grid(STARTING_FEN), random.Random(0))
        assert pieces.material.kind in MATERIAL_KINDS
        # Absent without a texture directory, which is the default.
        assert pieces.light_maps is None
        assert pieces.dark_maps is None
