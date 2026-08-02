"""Discovery of downloaded PBR texture sets.

The set discovery is the part worth pinning: it turns a directory of files into
the maps a material wires up, and every failure mode is silent. A half-matched set
renders shinier than intended rather than raising, and a slug parsed wrongly makes
the whole directory look empty -- which shows up as a dataset that quietly went
back to flat-colour tables.
"""

from __future__ import annotations

from pathlib import Path

from chesssight.synth.textures import (
    CURATED,
    MAPS,
    all_slugs,
    texture_sets,
)


def write_set(root: Path, slug: str, *, maps=MAPS, resolution: str = "2k") -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in maps:
        (root / f"{slug}_{name}_{resolution}.jpg").write_bytes(b"stub")


class TestDiscovery:
    def test_a_complete_set_is_found(self, tmp_path: Path):
        write_set(tmp_path, "oak_veneer_01")
        found = texture_sets(tmp_path)
        assert [entry["slug"] for entry in found] == ["oak_veneer_01"]
        assert sorted(found[0]["maps"]) == sorted(MAPS)

    def test_slugs_containing_underscores_and_digits_survive(self, tmp_path: Path):
        # The slug is recovered by stripping a known suffix rather than by splitting
        # on separators, because the slugs themselves are full of them. Splitting
        # produced paths like `oak_veneer_01_Diffuse_Diffuse_2k.jpg` and found none.
        for slug in ("plank_flooring_02", "brushed_concrete", "rosewood_veneer1"):
            write_set(tmp_path, slug)
        assert {entry["slug"] for entry in texture_sets(tmp_path)} == {
            "plank_flooring_02",
            "brushed_concrete",
            "rosewood_veneer1",
        }

    def test_paths_are_absolute_and_point_at_real_files(self, tmp_path: Path):
        write_set(tmp_path, "oak_veneer_01")
        maps = texture_sets(tmp_path)[0]["maps"]
        for path in maps.values():
            assert Path(path).is_absolute()
            assert Path(path).is_file()

    def test_an_incomplete_set_is_skipped_entirely(self, tmp_path: Path):
        # Half a set is worse than none: a diffuse map wired up without its
        # roughness map renders glossier than intended and nothing reports it.
        write_set(tmp_path, "partial", maps=("Diffuse",))
        assert texture_sets(tmp_path) == []

    def test_a_missing_directory_is_not_an_error(self, tmp_path: Path):
        # This is the no-assets path: the generator falls back to flat colour and
        # still produces a valid dataset, so it must not raise.
        assert texture_sets(tmp_path / "nothing-here") == []

    def test_other_resolutions_are_ignored(self, tmp_path: Path):
        write_set(tmp_path, "oak_veneer_01", resolution="1k")
        assert texture_sets(tmp_path, resolution="2k") == []
        assert len(texture_sets(tmp_path, resolution="1k")) == 1


class TestCuration:
    def test_every_curated_slug_is_unique(self):
        slugs = all_slugs()
        assert len(slugs) == len(set(slugs))

    def test_groups_are_non_empty(self):
        assert CURATED
        assert all(group for group in CURATED.values())
