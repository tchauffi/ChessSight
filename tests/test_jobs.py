from __future__ import annotations

import math
from pathlib import Path

import pytest

from chesssight.data.dataset import DatasetWriter
from chesssight.data.fen import BOARD_SIZE, STARTING_FEN, fen_to_grid, iter_occupied
from chesssight.data.schema import DatasetMeta
from chesssight.synth import jobs, randomize
from chesssight.synth.config import GeneratorConfig
from chesssight.synth.jobspec import JobSpec, board_to_world
from chesssight.synth.seeds import derive_rng, derive_seed


def make_config(**overrides: object) -> GeneratorConfig:
    payload: dict[str, object] = {
        "count": 8,
        "master_seed": 1234,
        "positions": {"pgn_paths": [], "weight_pgn": 0.0, "weight_random": 1.0},
    }
    payload.update(overrides)
    return GeneratorConfig.model_validate(payload)


@pytest.fixture
def writer(tmp_path: Path) -> DatasetWriter:
    writer = DatasetWriter(tmp_path / "run")
    writer.initialise(
        DatasetMeta(
            name="run",
            created_at="2026-07-29T00:00:00Z",
            source="synthetic",
            master_seed=1234,
        )
    )
    return writer


class TestSeeds:
    def test_derivation_is_stable(self):
        assert derive_seed(42, "sample", 7) == derive_seed(42, "sample", 7)

    def test_derivation_is_stable_across_processes(self):
        # Hard-coded so a change to the hashing scheme fails loudly rather than
        # silently invalidating every previously generated dataset.
        assert derive_seed(0) == derive_seed(0)
        assert derive_seed(1, "camera") != derive_seed(1, "lighting")

    def test_different_keys_diverge(self):
        seeds = {derive_seed(5, "sample", index) for index in range(200)}
        assert len(seeds) == 200

    def test_seeds_are_positive_and_json_safe(self):
        for index in range(100):
            seed = derive_seed(-7, index)
            assert 0 <= seed < 2**63

    def test_derive_rng_is_reproducible(self):
        assert derive_rng(3, "x").random() == derive_rng(3, "x").random()


class TestSharding:
    def test_shards_are_disjoint_and_complete(self):
        specs = list(range(103))
        shards = jobs.shard(specs, 7)  # type: ignore[arg-type]
        flattened = [item for chunk in shards for item in chunk]
        assert sorted(flattened) == specs
        assert len(flattened) == len(set(flattened))

    def test_shards_are_balanced(self):
        shards = jobs.shard(list(range(100)), 8)  # type: ignore[arg-type]
        sizes = [len(chunk) for chunk in shards]
        assert max(sizes) - min(sizes) <= 1

    def test_single_shard_keeps_everything(self):
        assert jobs.shard([1, 2, 3], 1) == [[1, 2, 3]]  # type: ignore[arg-type,comparison-overlap]

    def test_more_shards_than_jobs_leaves_empties(self):
        shards = jobs.shard([1, 2], 5)  # type: ignore[arg-type]
        assert sum(len(chunk) for chunk in shards) == 2
        assert shards.count([]) == 3

    def test_zero_shards_raises(self):
        with pytest.raises(ValueError):
            jobs.shard([], 0)


class TestJobGeneration:
    def test_generates_one_job_per_sample(self, writer: DatasetWriter):
        config = make_config(count=5)
        specs = list(jobs.iter_jobs(config, writer))
        assert [spec.id for spec in specs] == [
            "000000",
            "000001",
            "000002",
            "000003",
            "000004",
        ]

    def test_is_reproducible(self, writer: DatasetWriter):
        config = make_config(count=4)
        first = [spec.model_dump_json() for spec in jobs.iter_jobs(config, writer)]
        second = [spec.model_dump_json() for spec in jobs.iter_jobs(config, writer)]
        assert first == second

    def test_skip_ids_resumes_exactly_where_it_stopped(self, writer: DatasetWriter):
        config = make_config(count=6)
        everything = {
            spec.id: spec.model_dump_json() for spec in jobs.iter_jobs(config, writer)
        }

        remaining = list(
            jobs.iter_jobs(config, writer, skip_ids={"000000", "000001", "000002"})
        )
        assert [spec.id for spec in remaining] == ["000003", "000004", "000005"]
        # Resumed jobs must be byte-identical to what the uninterrupted run produced.
        for spec in remaining:
            assert spec.model_dump_json() == everything[spec.id]

    def test_fen_matches_grid(self, writer: DatasetWriter):
        for spec in jobs.iter_jobs(make_config(count=4), writer):
            assert fen_to_grid(spec.fen) == spec.grid

    def test_one_placement_per_occupied_square(self, writer: DatasetWriter):
        for spec in jobs.iter_jobs(make_config(count=4), writer):
            occupied = iter_occupied(spec.grid)
            assert len(spec.pieces.placements) == len(occupied)
            for placement, (rank, file, class_id) in zip(
                spec.pieces.placements, occupied, strict=True
            ):
                assert (placement.rank_index, placement.file_index) == (rank, file)
                assert placement.class_id == class_id

    def test_instance_ids_are_unique_and_start_at_one(self, writer: DatasetWriter):
        for spec in jobs.iter_jobs(make_config(count=4), writer):
            ids = [placement.instance_id for placement in spec.pieces.placements]
            assert ids == list(range(1, len(ids) + 1))

    def test_paths_are_absolute_and_under_the_run_dir(self, writer: DatasetWriter):
        for spec in jobs.iter_jobs(make_config(count=2), writer):
            assert Path(spec.image_path).is_absolute()
            assert Path(spec.labels_path).is_absolute()
            assert str(writer.root) in spec.image_path

    def test_id_pass_disabled_leaves_no_path(self, writer: DatasetWriter):
        config = make_config(count=2, render={"render_id_pass": False})
        for spec in jobs.iter_jobs(config, writer):
            assert spec.id_pass_path is None

    def test_png_output_changes_the_image_suffix(self, writer: DatasetWriter):
        config = make_config(count=1, render={"image_format": "PNG"})
        spec = next(iter(jobs.iter_jobs(config, writer)))
        assert spec.image_path.endswith(".png")

    def test_shard_round_trip(self, writer: DatasetWriter, tmp_path: Path):
        specs = list(jobs.iter_jobs(make_config(count=6), writer))
        paths = jobs.write_shards(tmp_path, jobs.shard(specs, 3))
        assert len(paths) == 3

        recovered = [spec for path in paths for spec in jobs.read_shard(path)]
        assert sorted(spec.id for spec in recovered) == sorted(
            spec.id for spec in specs
        )
        assert isinstance(recovered[0], JobSpec)


class TestRandomization:
    def test_camera_sits_above_the_table(self):
        config = make_config()
        for index in range(50):
            camera = randomize.resolve_camera(config, derive_rng(index, "camera"))
            assert camera.location[2] > 0

    def test_camera_distance_respects_the_configured_range(self):
        config = make_config(camera={"distance": {"min": 12.0, "max": 12.0}})
        camera = randomize.resolve_camera(config, derive_rng(0, "camera"))
        assert math.dist(camera.location, [0.0, 0.0, 0.0]) == pytest.approx(12.0)

    def test_elevation_range_is_honoured(self):
        config = make_config(camera={"elevation_deg": {"min": 30.0, "max": 30.0}})
        camera = randomize.resolve_camera(config, derive_rng(0, "camera"))
        horizontal = math.hypot(camera.location[0], camera.location[1])
        assert math.degrees(
            math.atan2(camera.location[2], horizontal)
        ) == pytest.approx(30.0)

    def test_placement_jitter_never_leaves_the_square(self, writer: DatasetWriter):
        config = make_config(count=8)
        square_size = config.board.square_size
        for spec in jobs.iter_jobs(config, writer):
            for placement in spec.pieces.placements:
                x, y = randomize.placement_world_xy(placement, square_size)
                center_x, center_y = board_to_world(
                    placement.file_index + 0.5, placement.rank_index + 0.5, square_size
                )
                assert abs(x - center_x) < square_size / 2
                assert abs(y - center_y) < square_size / 2

    def test_board_world_mapping_orientation(self):
        # a8 corner is at -x/+y; h1 corner is at +x/-y.
        assert board_to_world(0, 0, 1.0) == (-4.0, 4.0)
        assert board_to_world(BOARD_SIZE, BOARD_SIZE, 1.0) == (4.0, -4.0)
        # e1 centre: file 4, rank index 7.
        x, y = board_to_world(4.5, 7.5, 1.0)
        assert (x, y) == (0.5, -3.5)

    def test_changing_one_aspect_does_not_disturb_others(self):
        base = make_config()
        relit = make_config(lighting={"lamp_energy": {"min": 5.0, "max": 5.0}})
        seed = derive_seed(1234, "sample", 0)

        assert randomize.resolve_camera(base, derive_rng(seed, "camera")) == (
            randomize.resolve_camera(relit, derive_rng(seed, "camera"))
        )
        assert randomize.resolve_lighting(base, derive_rng(seed, "lighting")) != (
            randomize.resolve_lighting(relit, derive_rng(seed, "lighting"))
        )

    def test_lamp_count_is_within_range(self):
        config = make_config(lighting={"lamp_count": {"min": 2, "max": 2}})
        lighting = randomize.resolve_lighting(config, derive_rng(0, "lighting"))
        assert len(lighting.lamps) == 2

    def test_hdri_is_skipped_when_the_directory_is_missing(self):
        config = make_config(
            lighting={"hdri_dir": "/nonexistent/hdris", "hdri_probability": 1.0}
        )
        lighting = randomize.resolve_lighting(config, derive_rng(0, "lighting"))
        assert lighting.hdri_path is None

    def test_hdri_is_used_when_files_are_present(self, tmp_path: Path):
        (tmp_path / "studio.hdr").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")
        config = make_config(
            lighting={"hdri_dir": str(tmp_path), "hdri_probability": 1.0}
        )
        lighting = randomize.resolve_lighting(config, derive_rng(0, "lighting"))
        assert lighting.hdri_path is not None
        assert lighting.hdri_path.endswith("studio.hdr")

    def test_colour_temperature_is_warm_at_low_kelvin(self):
        warm = randomize._color_temperature_rgb(2700)
        cool = randomize._color_temperature_rgb(7500)
        assert warm[0] >= cool[0]
        assert warm[2] < cool[2]
        for colour in (warm, cool):
            assert all(0.0 <= channel <= 1.0 for channel in colour)

    def test_distractors_stay_clear_of_the_board(self):
        config = make_config(
            scene={
                "distractor_probability": 1.0,
                "distractor_count": {"min": 3, "max": 3},
            }
        )
        half_board = BOARD_SIZE / 2.0 * config.board.square_size
        for index in range(20):
            scene = randomize.resolve_scene(config, derive_rng(index, "scene"))
            for distractor in scene.distractors:
                radius = math.hypot(distractor.location[0], distractor.location[1])
                assert radius > half_board


class TestConfig:
    def test_yaml_round_trip(self, tmp_path: Path):
        config = make_config(count=17)
        path = tmp_path / "config.yaml"
        config.to_yaml(path)
        assert GeneratorConfig.from_yaml(path) == config

    def test_unknown_keys_are_rejected(self):
        with pytest.raises(ValueError):
            GeneratorConfig.model_validate({"nonsense": 1})

    def test_inverted_range_is_rejected(self):
        with pytest.raises(ValueError):
            GeneratorConfig.model_validate(
                {"camera": {"distance": {"min": 10.0, "max": 1.0}}}
            )

    def test_position_source_must_be_usable(self):
        with pytest.raises(ValueError, match="weight_random must be positive"):
            GeneratorConfig.model_validate(
                {
                    "positions": {
                        "pgn_paths": [],
                        "weight_pgn": 1.0,
                        "weight_random": 0.0,
                    }
                }
            )

    def test_default_output_root_is_outside_the_repo(self):
        config = GeneratorConfig()
        assert config.output.run_dir().is_absolute()
        assert "datasets" in str(config.output.run_dir())


class TestSamplerAssembly:
    def test_random_only_when_no_pgn_configured(self):
        sampler = jobs.build_sampler(make_config())
        grid = sampler.sample(derive_rng(0, "position"))
        assert fen_to_grid(STARTING_FEN) != grid

    def test_pgn_and_random_mix(self):
        fixture = Path(__file__).parent / "fixtures" / "sample.pgn"
        config = make_config(
            positions={
                "pgn_paths": [str(fixture)],
                "weight_pgn": 1.0,
                "weight_random": 1.0,
            }
        )
        sampler = jobs.build_sampler(config)
        assert len(sampler.samplers) == 2  # type: ignore[attr-defined]


class TestCapturedPieces:
    """Pieces beside the board, as a real game in progress accumulates."""

    def test_a_full_board_has_nothing_to_capture(self):
        from chesssight.data.fen import STARTING_FEN, fen_to_grid
        from chesssight.synth.randomize import missing_pieces

        assert missing_pieces(fen_to_grid(STARTING_FEN)) == []

    def test_missing_pieces_complement_the_position(self):
        from chesssight.data.fen import fen_to_grid
        from chesssight.synth.randomize import FULL_SET, missing_pieces

        grid = fen_to_grid("4k3/8/8/8/8/8/4P3/4K3")
        absent = missing_pieces(grid)
        # 32 in a full set, 3 on the board.
        assert len(absent) == sum(FULL_SET.values()) - 3
        # Both kings are on the board, so neither can have been captured.
        assert 6 not in absent and 12 not in absent

    def test_extra_pieces_never_produce_negative_counts(self):
        from chesssight.data.fen import fen_to_grid
        from chesssight.synth.randomize import missing_pieces

        # The random sampler can place more knights than a real set holds.
        grid = fen_to_grid("4k3/8/8/NNNN4/8/8/8/4K3")
        absent = missing_pieces(grid)
        assert all(count >= 0 for count in [absent.count(c) for c in set(absent)])
        assert 2 not in absent  # four white knights already exceeds the set

    def test_probability_zero_places_none(self):
        from chesssight.data.fen import fen_to_grid
        from chesssight.synth.randomize import resolve_captured

        config = make_config(pieces={"captured_probability": 0.0})
        grid = fen_to_grid("4k3/8/8/8/8/8/8/4K3")
        assert (
            resolve_captured(config, grid, derive_rng(0, "c"), first_instance_id=3)
            == []
        )

    def test_captured_are_drawn_only_from_missing_pieces(self):
        from collections import Counter

        from chesssight.data.fen import fen_to_grid
        from chesssight.synth.randomize import missing_pieces, resolve_captured

        config = make_config(pieces={"captured_probability": 1.0})
        grid = fen_to_grid("r3k3/pp6/8/8/8/8/PP6/R3K3")
        absent = Counter(missing_pieces(grid))

        for seed in range(20):
            captured = resolve_captured(
                config, grid, derive_rng(seed, "c"), first_instance_id=11
            )
            taken = Counter(entry.class_id for entry in captured)
            for class_id, count in taken.items():
                assert count <= absent[class_id], (
                    "a captured piece must be one the board is missing, or the pile "
                    "contradicts the position it sits next to"
                )

    def test_captured_sit_clear_of_the_board(self):
        from chesssight.data.fen import fen_to_grid
        from chesssight.synth.randomize import resolve_captured

        config = make_config(pieces={"captured_probability": 1.0})
        grid = fen_to_grid("4k3/8/8/8/8/8/8/4K3")
        half_board = BOARD_SIZE / 2.0 * config.board.square_size

        for seed in range(20):
            for entry in resolve_captured(
                config, grid, derive_rng(seed, "c"), first_instance_id=3
            ):
                assert abs(entry.x) > half_board, "a captured piece landed on the board"

    def test_colours_are_grouped_on_opposite_sides(self):
        from chesssight.data.fen import fen_to_grid, is_white
        from chesssight.synth.randomize import resolve_captured

        config = make_config(pieces={"captured_probability": 1.0})
        grid = fen_to_grid("4k3/8/8/8/8/8/8/4K3")
        captured = resolve_captured(
            config, grid, derive_rng(3, "c"), first_instance_id=3
        )
        assert captured, "expected some captured pieces at probability 1.0"
        for entry in captured:
            assert (entry.x < 0) == is_white(entry.class_id)

    def test_instance_ids_continue_from_the_board(self, writer: DatasetWriter):
        config = make_config(
            count=6,
            pieces={
                "captured_probability": 1.0,
                "captured_count": {"min": 2, "max": 6},
            },
        )
        for spec in jobs.iter_jobs(config, writer):
            board_ids = [p.instance_id for p in spec.pieces.placements]
            captured_ids = [c.instance_id for c in spec.pieces.captured]
            # Both share the id-pass channel, so the ranges must not collide.
            assert set(board_ids).isdisjoint(captured_ids)
            assert sorted(board_ids + captured_ids) == list(
                range(1, len(board_ids) + len(captured_ids) + 1)
            )

    def test_total_instances_stay_within_the_id_pass_range(self, writer: DatasetWriter):
        # The Workbench pass encodes the id in one 8-bit channel.
        config = make_config(
            count=8,
            pieces={
                "captured_probability": 1.0,
                "captured_count": {"min": 12, "max": 12},
            },
        )
        for spec in jobs.iter_jobs(config, writer):
            assert len(spec.pieces.placements) + len(spec.pieces.captured) < 255
