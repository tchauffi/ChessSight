from __future__ import annotations

import pytest

from chesssight.synth import profiles
from chesssight.synth.config import PiecesConfig


def test_all_builtin_profiles_are_valid():
    profiles.validate_all()


def test_every_piece_letter_is_covered():
    assert set(profiles.PROFILES) == set("PNBRQK")
    assert set(profiles.PIECE_HEIGHTS) == set("PNBRQK")
    assert set(profiles.PROFILE_TOP) == set("PNBRQK")


@pytest.mark.parametrize("letter", list(profiles.PIECE_LETTERS))
def test_profiles_start_and_end_on_the_axis(letter: str):
    profile = profiles.PROFILES[letter]
    assert profile[0] == (0.0, 0.0)
    assert profile[-1][0] == 0.0


@pytest.mark.parametrize("letter", list(profiles.PIECE_LETTERS))
def test_pieces_fit_inside_their_square(letter: str):
    # A radius of 0.5 square units would exactly touch the square edge; anything at
    # or beyond that would overlap a neighbour once placement jitter is applied.
    assert profiles.MAX_RADII[letter] < 0.5
    assert profiles.piece_radius(letter, square_size=1.0) < 0.5


@pytest.mark.parametrize("letter", list(profiles.PIECE_LETTERS))
def test_turned_part_reaches_its_declared_top(letter: str):
    top = max(z for _, z in profiles.PROFILES[letter])
    assert top == pytest.approx(profiles.PROFILE_TOP[letter])


def test_the_king_is_the_tallest_and_the_pawn_the_shortest():
    heights = profiles.PIECE_HEIGHTS
    assert heights["K"] == max(heights.values())
    assert heights["P"] == min(heights.values())
    assert heights["K"] > heights["Q"] > heights["B"] > heights["R"]


def test_knight_and_king_leave_room_for_added_geometry():
    # These two are finished with non-lathe parts, so their turned section must
    # stop short of the apex or the head/cross would be buried inside it.
    assert profiles.PROFILE_TOP["N"] < 0.6
    assert profiles.PROFILE_TOP["K"] < 1.0
    assert profiles.PROFILE_TOP["R"] < 1.0


class TestScaling:
    def test_scaled_profile_matches_the_requested_height(self):
        scaled = profiles.scaled_profile("Q", square_size=2.0)
        top = max(z for _, z in scaled)
        assert top == pytest.approx(profiles.PIECE_HEIGHTS["Q"] * 2.0)

    def test_height_and_radius_scale_independently(self):
        scaled = profiles.scaled_profile("P", height_scale=2.0, radius_scale=0.5)
        assert max(z for _, z in scaled) == pytest.approx(
            profiles.PIECE_HEIGHTS["P"] * 2.0
        )
        assert max(r for r, _ in scaled) == pytest.approx(profiles.MAX_RADII["P"] * 0.5)

    def test_square_size_scales_both_axes(self):
        base = profiles.scaled_profile("B", square_size=1.0)
        doubled = profiles.scaled_profile("B", square_size=2.0)
        for (r1, z1), (r2, z2) in zip(base, doubled, strict=True):
            assert r2 == pytest.approx(r1 * 2.0)
            assert z2 == pytest.approx(z1 * 2.0)

    def test_piece_height_helper(self):
        assert profiles.piece_height("K", square_size=3.0) == pytest.approx(
            profiles.PIECE_HEIGHTS["K"] * 3.0
        )


class TestTaper:
    """The silhouette warp that gives the procedural set more than one outline."""

    @pytest.mark.parametrize("letter", list(profiles.PIECE_LETTERS))
    def test_no_taper_leaves_the_profile_untouched(self, letter: str):
        plain = profiles.scaled_profile(letter)
        untapered = profiles.scaled_profile(letter, taper=0.0)
        assert plain == untapered

    @pytest.mark.parametrize("letter", list(profiles.PIECE_LETTERS))
    def test_the_factor_is_symmetric_about_the_turned_part(self, letter: str):
        # 1+t at the base, 1-t at the top of the turned part, for every letter --
        # that is what makes one taper value mean the same shape change on all six.
        assert profiles.taper_factor(letter, 0.0, 0.1) == pytest.approx(1.1)
        assert profiles.taper_factor(
            letter, profiles.PROFILE_TOP[letter], 0.1
        ) == pytest.approx(0.9)

    @pytest.mark.parametrize("letter", list(profiles.PIECE_LETTERS))
    def test_positive_taper_widens_the_base_and_narrows_the_top(self, letter: str):
        base = profiles.scaled_profile(letter)
        squat = profiles.scaled_profile(letter, taper=0.12)
        top = profiles.PROFILE_TOP[letter]
        # Compare at matching profile points, since z is unchanged by a taper.
        lower = [
            (b[0], s[0])
            for b, s, (_, z) in zip(base, squat, profiles.PROFILES[letter], strict=True)
            if z < top * 0.5 and b[0] > 0
        ]
        upper = [
            (b[0], s[0])
            for b, s, (_, z) in zip(base, squat, profiles.PROFILES[letter], strict=True)
            if z > top * 0.5 and b[0] > 0
        ]
        assert lower and upper
        assert all(tapered > plain for plain, tapered in lower)
        assert all(tapered < plain for plain, tapered in upper)

    @pytest.mark.parametrize("letter", list(profiles.PIECE_LETTERS))
    @pytest.mark.parametrize("taper", [-0.12, -0.05, 0.05, 0.12])
    def test_a_tapered_profile_is_still_sweepable(self, letter: str, taper: float):
        # A warp that broke the axis endpoints or turned a radius negative would
        # sweep into an open, self-intersecting shell rather than a solid.
        scaled = profiles.scaled_profile(letter, taper=taper)
        assert scaled[0][0] == pytest.approx(0.0)
        assert scaled[-1][0] == pytest.approx(0.0)
        assert all(radius >= 0.0 for radius, _ in scaled)
        turned_top = profiles.piece_height(letter) * profiles.PROFILE_TOP[letter]
        assert max(z for _, z in scaled) == pytest.approx(turned_top)

    @pytest.mark.parametrize("letter", list(profiles.PIECE_LETTERS))
    @pytest.mark.parametrize("taper", [-0.12, 0.0, 0.12])
    def test_piece_radius_reports_the_widest_point_of_the_warped_profile(
        self, letter: str, taper: float
    ):
        scaled = profiles.scaled_profile(letter, taper=taper)
        assert profiles.piece_radius(letter, taper=taper) == pytest.approx(
            max(radius for radius, _ in scaled)
        )

    @pytest.mark.parametrize("letter", list(profiles.PIECE_LETTERS))
    def test_the_widest_configured_piece_still_fits_its_square(self, letter: str):
        # The real invariant behind the taper bound: enlarging and tapering multiply,
        # and two pieces that touch make the dataset quietly wrong rather than loud.
        pieces = PiecesConfig()
        worst = max(
            profiles.piece_radius(
                letter, radius_scale=pieces.radius_scale.max, taper=taper
            )
            for taper in (pieces.taper.min, pieces.taper.max)
        )
        assert worst < 0.5


class TestValidation:
    def test_unknown_letter_raises(self):
        with pytest.raises(profiles.ProfileError):
            profiles.scaled_profile("X")
        with pytest.raises(profiles.ProfileError):
            profiles.piece_height("X")
        with pytest.raises(profiles.ProfileError):
            profiles.piece_radius("X")

    def test_profile_not_starting_at_the_origin_is_rejected(self):
        with pytest.raises(profiles.ProfileError, match="start on the axis"):
            profiles.validate_profile("P", ((0.2, 0.0), (0.2, 0.5), (0.0, 1.0)))

    def test_profile_not_ending_on_the_axis_is_rejected(self):
        with pytest.raises(profiles.ProfileError, match="end on the axis"):
            profiles.validate_profile("P", ((0.0, 0.0), (0.2, 0.5), (0.2, 1.0)))

    def test_overwide_profile_is_rejected(self):
        with pytest.raises(profiles.ProfileError, match="overflow its square"):
            profiles.validate_profile("P", ((0.0, 0.0), (0.7, 0.5), (0.0, 1.0)))

    def test_wrong_top_is_rejected(self):
        with pytest.raises(profiles.ProfileError, match="expected"):
            profiles.validate_profile("P", ((0.0, 0.0), (0.2, 0.4), (0.0, 0.5)))

    def test_too_few_points_is_rejected(self):
        with pytest.raises(profiles.ProfileError, match="at least 3"):
            profiles.validate_profile("P", ((0.0, 0.0), (0.0, 1.0)))
