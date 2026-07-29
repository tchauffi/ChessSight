from __future__ import annotations

import json
from pathlib import Path

import pytest

from chesssight.synth import asset_spec
from chesssight.synth.assets import AssetError, AssetManifest, blank_manifest
from chesssight.synth.profiles import PIECE_HEIGHTS, PIECE_LETTERS


def write_set(tmp_path: Path, extension: str = ".obj", **overrides) -> Path:
    """A manifest plus stand-in model files, enough for the loader to accept."""
    manifest = blank_manifest("test-set", extension).model_dump()
    manifest.update(overrides)
    for entry in manifest["pieces"].values():
        (tmp_path / entry["file"]).write_text("# stand-in", encoding="utf-8")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class TestManifest:
    def test_blank_manifest_covers_every_piece(self):
        manifest = blank_manifest("demo")
        assert set(manifest.pieces) == set(PIECE_LETTERS)
        assert manifest.pieces["N"].file == "knight.obj"

    def test_round_trips_through_disk(self, tmp_path: Path):
        path = write_set(tmp_path)
        manifest = AssetManifest.load(path)
        assert manifest.name == "test-set"
        assert manifest.check(tmp_path) == []

    def test_missing_piece_is_rejected(self, tmp_path: Path):
        payload = blank_manifest("partial").model_dump()
        del payload["pieces"]["N"]
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="missing pieces"):
            AssetManifest.load(path)

    def test_unknown_piece_letter_is_rejected(self, tmp_path: Path):
        payload = blank_manifest("odd").model_dump()
        payload["pieces"]["Z"] = {"file": "zebra.obj"}
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown piece letters"):
            AssetManifest.load(path)

    def test_missing_files_are_reported_not_raised(self, tmp_path: Path):
        payload = blank_manifest("ghost").model_dump()
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        problems = AssetManifest.load(path).check(tmp_path)
        assert len(problems) == 6
        assert all("missing file" in problem for problem in problems)

    def test_unreadable_manifest_raises_asset_error(self, tmp_path: Path):
        with pytest.raises(AssetError, match="cannot read manifest"):
            AssetManifest.load(tmp_path / "nope.json")

    def test_malformed_json_raises_asset_error(self, tmp_path: Path):
        path = tmp_path / "manifest.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(AssetError, match="not valid JSON"):
            AssetManifest.load(path)

    def test_unknown_extension_is_rejected(self, tmp_path: Path):
        path = write_set(tmp_path, ".xyz")
        manifest = AssetManifest.load(path)
        problems = manifest.check(tmp_path)
        assert any("cannot tell how to import" in problem for problem in problems)

    def test_explicit_kind_overrides_the_extension(self, tmp_path: Path):
        path = write_set(tmp_path, ".xyz", kind="obj")
        assert AssetManifest.load(path).check(tmp_path) == []


class TestOrientation:
    @pytest.mark.parametrize(
        ("axis", "expected"),
        [("+Y", 0.0), ("+X", 90.0), ("-Y", 180.0), ("-X", -90.0)],
    )
    def test_forward_yaw(self, axis: str, expected: float):
        assert asset_spec.forward_yaw(axis) == expected

    @pytest.mark.parametrize(
        ("axis", "expected"),
        [("+Z", 0.0), ("+Y", 90.0), ("-Z", 180.0), ("-Y", -90.0)],
    )
    def test_up_pitch(self, axis: str, expected: float):
        # glTF and most game engines are Y-up; without this a set renders lying down.
        assert asset_spec.up_pitch(axis) == expected

    def test_a_vertical_forward_axis_is_rejected(self):
        with pytest.raises(AssetError, match="not meaningful"):
            asset_spec.forward_yaw("+Z")

    def test_a_horizontal_up_axis_that_is_unsupported_is_rejected(self):
        with pytest.raises(AssetError, match="must be one of"):
            asset_spec.up_pitch("+X")


class TestTargetHeight:
    def test_defaults_scale_from_the_king(self):
        manifest = {"king_height": 1.55, "pieces": {"P": {"file": "p.obj"}}}
        expected = 1.55 * PIECE_HEIGHTS["P"] / PIECE_HEIGHTS["K"]
        assert asset_spec.target_height(manifest, "P") == pytest.approx(expected)

    def test_a_taller_king_scales_the_whole_set(self):
        manifest = {"king_height": 3.10, "pieces": {"Q": {"file": "q.obj"}}}
        base = {"king_height": 1.55, "pieces": {"Q": {"file": "q.obj"}}}
        assert asset_spec.target_height(manifest, "Q") == pytest.approx(
            2.0 * asset_spec.target_height(base, "Q")
        )

    def test_per_piece_override_wins(self):
        manifest = {
            "king_height": 1.55,
            "pieces": {"R": {"file": "r.obj", "height": 2.5}},
        }
        assert asset_spec.target_height(manifest, "R") == 2.5

    def test_relative_proportions_are_preserved(self):
        manifest = {
            "king_height": 2.0,
            "pieces": {letter: {"file": f"{letter}.obj"} for letter in PIECE_LETTERS},
        }
        heights = {
            letter: asset_spec.target_height(manifest, letter)
            for letter in PIECE_LETTERS
        }
        assert heights["K"] > heights["Q"] > heights["B"] > heights["R"] > heights["P"]


class TestStdlibLoader:
    """The Blender side reads manifests through asset_spec, without pydantic."""

    def test_loads_a_valid_manifest(self, tmp_path: Path):
        path = write_set(tmp_path)
        manifest = asset_spec.load_manifest(path)
        assert manifest["name"] == "test-set"
        assert set(manifest["pieces"]) == set(PIECE_LETTERS)

    def test_reports_missing_pieces(self, tmp_path: Path):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"name": "x", "pieces": {}}), encoding="utf-8")
        with pytest.raises(AssetError, match="missing pieces"):
            asset_spec.load_manifest(path)

    def test_rejects_a_manifest_without_pieces(self, tmp_path: Path):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"name": "x"}), encoding="utf-8")
        with pytest.raises(AssetError, match="no `pieces` mapping"):
            asset_spec.load_manifest(path)

    def test_agrees_with_the_validated_loader(self, tmp_path: Path):
        # The two loaders must not drift: the Blender side and the CLI have to
        # understand the same file the same way.
        path = write_set(tmp_path)
        plain = asset_spec.load_manifest(path)
        validated = AssetManifest.load(path)
        assert plain["name"] == validated.name
        assert set(plain["pieces"]) == set(validated.pieces)
        for letter in PIECE_LETTERS:
            assert asset_spec.target_height(plain, letter) == pytest.approx(
                validated.king_height * PIECE_HEIGHTS[letter] / PIECE_HEIGHTS["K"]
            )


def test_every_supported_extension_maps_to_a_kind():
    for suffix, kind in asset_spec.KIND_BY_SUFFIX.items():
        assert asset_spec.kind_for_file(f"piece{suffix}") == kind


class TestLicenceWarnings:
    """Renders are derivative works, so a set's terms follow the dataset."""

    def test_noncommercial_is_flagged(self):
        warnings = asset_spec.licence_warnings(
            {"license": "CC BY-NC 4.0", "attribution": "someone"}
        )
        assert any("commercial" in warning for warning in warnings)

    def test_no_derivatives_is_flagged(self):
        warnings = asset_spec.licence_warnings({"license": "CC BY-ND 4.0"})
        assert warnings

    def test_permissive_licences_pass_clean(self):
        assert asset_spec.licence_warnings({"license": "CC0 1.0"}) == []
        assert asset_spec.licence_warnings({"license": "MIT"}) == []

    def test_attribution_licence_without_a_credit_line_is_flagged(self):
        warnings = asset_spec.licence_warnings({"license": "CC BY 4.0"})
        assert any("attribution" in warning for warning in warnings)

    def test_a_credited_attribution_licence_passes(self):
        assert (
            asset_spec.licence_warnings(
                {"license": "CC BY 4.0", "attribution": "Someone, somewhere"}
            )
            == []
        )

    def test_missing_licence_is_flagged_as_all_rights_reserved(self):
        warnings = asset_spec.licence_warnings({})
        assert any("all rights reserved" in warning for warning in warnings)

    def test_attribution_record_carries_the_terms(self):
        record = asset_spec.attribution(
            {
                "name": "demo",
                "source": "https://example.invalid",
                "license": "CC BY-NC 4.0",
                "attribution": "Someone",
            }
        )
        assert record["set"] == "demo"
        assert record["license"] == "CC BY-NC 4.0"
        assert record["attribution"] == "Someone"


class TestScaleMode:
    def test_uniform_is_the_default(self):
        assert asset_spec.scale_mode({}) == "uniform"

    def test_per_piece_is_accepted(self):
        assert asset_spec.scale_mode({"scale_mode": "per_piece"}) == "per_piece"

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(AssetError, match="must be one of"):
            asset_spec.scale_mode({"scale_mode": "nonsense"})


class TestDecimate:
    def test_absent_means_no_decimation(self):
        assert asset_spec.decimate_ratio({}) is None

    def test_a_ratio_of_one_is_a_no_op(self):
        assert asset_spec.decimate_ratio({"decimate": 1.0}) is None

    def test_a_valid_ratio_passes_through(self):
        assert asset_spec.decimate_ratio({"decimate": 0.06}) == pytest.approx(0.06)

    @pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
    def test_out_of_range_is_rejected(self, bad: float):
        with pytest.raises(AssetError, match="decimate must be in"):
            asset_spec.decimate_ratio({"decimate": bad})
