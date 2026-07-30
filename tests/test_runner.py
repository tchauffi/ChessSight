from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from chesssight.data.dataset import DatasetReader, read_index
from chesssight.synth import jobs, runner
from chesssight.synth.config import GeneratorConfig
from tests.test_postprocess import make_raw


def make_config(tmp_path: Path, **overrides: object) -> GeneratorConfig:
    payload: dict[str, object] = {
        "count": 6,
        "master_seed": 77,
        "positions": {"pgn_paths": [], "weight_pgn": 0.0, "weight_random": 1.0},
        "output": {"root": str(tmp_path), "run_name": "run"},
        "render": {"image_format": "PNG"},
    }
    payload.update(overrides)
    return GeneratorConfig.model_validate(payload)


class TestPlan:
    def test_writes_meta_and_shards(self, tmp_path: Path):
        config = make_config(tmp_path)
        writer, shards, planned = runner.plan(config, workers=2)

        assert planned == 6
        assert len(shards) == 2
        assert writer.meta_path.is_file()
        assert sum(len(jobs.read_shard(shard)) for shard in shards) == 6

    def test_meta_records_provenance(self, tmp_path: Path):
        config = make_config(tmp_path)
        writer, _, _ = runner.plan(config)
        meta = json.loads(writer.meta_path.read_text())

        assert meta["master_seed"] == 77
        assert meta["source"] == "synthetic"
        # The full config is stored so a dataset can be regenerated from itself.
        assert meta["generator_config"]["count"] == 6

    def test_shards_partition_the_work_exactly(self, tmp_path: Path):
        config = make_config(tmp_path, count=17)
        _, shards, _ = runner.plan(config, workers=4)
        ids = [job.id for shard in shards for job in jobs.read_shard(shard)]
        assert sorted(ids) == sorted({f"{index:06d}" for index in range(17)})

    def test_resume_skips_already_indexed_ids(self, tmp_path: Path):
        config = make_config(tmp_path)
        writer, _, _ = runner.plan(config)

        # Pretend two samples already made it into the index.
        for index in range(2):
            raw = make_raw()
            raw["id"] = f"{index:06d}"
            from chesssight.synth.postprocess import build_sample

            writer.add(build_sample(raw, image_rel_path=f"images/{index:06d}.png"))

        _, shards, planned = runner.plan(config, workers=1, resume=True)
        assert planned == 4
        remaining = [job.id for job in jobs.read_shard(shards[0])]
        assert remaining == ["000002", "000003", "000004", "000005"]

    def test_planning_renders_nothing(self, tmp_path: Path):
        config = make_config(tmp_path)
        writer, _, _ = runner.plan(config)
        assert list(writer.images_dir.iterdir()) == []


class TestWorkerLimits:
    def test_eevee_keeps_the_requested_workers(self, tmp_path: Path):
        config = make_config(tmp_path, render={"engine": "BLENDER_EEVEE"})
        workers, warning = runner.effective_workers(config, 8)
        assert workers == 8
        assert warning is None

    def test_cycles_on_gpu_is_capped(self, tmp_path: Path):
        # Parallel Cycles workers share one GPU and each keeps its own copy of the
        # scene in VRAM, so oversubscribing is slower and can run out of memory.
        config = make_config(tmp_path, render={"engine": "CYCLES", "use_gpu": True})
        workers, warning = runner.effective_workers(config, 8)
        assert workers == runner.MAX_GPU_WORKERS
        assert warning is not None and "share one GPU" in warning

    def test_cycles_on_cpu_is_not_capped(self, tmp_path: Path):
        config = make_config(tmp_path, render={"engine": "CYCLES", "use_gpu": False})
        workers, warning = runner.effective_workers(config, 8)
        assert workers == 8
        assert warning is None

    def test_zero_workers_is_rejected(self, tmp_path: Path):
        with pytest.raises(runner.RunnerError):
            runner.effective_workers(make_config(tmp_path), 0)


class TestCollect:
    def _write_raw(self, writer, sample_id: str, **mutate) -> None:
        raw = make_raw()
        raw["id"] = sample_id
        raw.update(mutate)
        path = writer.root / jobs.RAW_LABELS_DIRNAME / f"{sample_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw), encoding="utf-8")

    def test_turns_raw_records_into_indexed_samples(self, tmp_path: Path):
        config = make_config(tmp_path)
        writer, _, _ = runner.plan(config)
        for index in range(3):
            self._write_raw(writer, f"{index:06d}")

        result = runner.collect(writer, config, quiet=True)
        assert result.rendered == 3
        assert result.failed == 0
        assert len(list(read_index(writer.root))) == 3
        assert len(DatasetReader(writer.root)) == 3

    def test_is_idempotent(self, tmp_path: Path):
        config = make_config(tmp_path)
        writer, _, _ = runner.plan(config)
        self._write_raw(writer, "000000")

        runner.collect(writer, config, quiet=True)
        second = runner.collect(writer, config, quiet=True)
        # Re-collecting must not duplicate an already-indexed sample.
        assert second.rendered == 0
        assert len(list(read_index(writer.root))) == 1

    def test_a_bad_record_is_recorded_and_the_rest_still_land(self, tmp_path: Path):
        config = make_config(tmp_path)
        writer, _, _ = runner.plan(config)
        self._write_raw(writer, "000000")
        # A record whose two projections disagree must be rejected, not indexed.
        broken = make_raw()
        broken["id"] = "000001"
        broken["square_centers_px"][10][0] += 25.0
        path = writer.root / jobs.RAW_LABELS_DIRNAME / "000001.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        self._write_raw(writer, "000002")

        result = runner.collect(writer, config, quiet=True)
        assert result.rendered == 2
        assert result.failed == 1
        assert [entry.id for entry in read_index(writer.root)] == ["000000", "000002"]
        assert writer.failures_path.is_file()
        assert "000001" in writer.failures_path.read_text()

    def test_missing_raw_directory_is_not_an_error(self, tmp_path: Path):
        config = make_config(tmp_path)
        writer, _, _ = runner.plan(config)
        assert runner.collect(writer, config, quiet=True).rendered == 0

    def test_split_comes_from_the_config(self, tmp_path: Path):
        config = make_config(tmp_path, output={"root": str(tmp_path), "split": "val"})
        writer, _, _ = runner.plan(config)
        self._write_raw(writer, "000000")

        runner.collect(writer, config, quiet=True)
        assert DatasetReader(writer.root).load("000000").split == "val"

    def test_masks_can_be_omitted(self, tmp_path: Path):
        config = make_config(tmp_path)
        writer, _, _ = runner.plan(config)
        self._write_raw(writer, "000000")

        runner.collect(writer, config, store_masks=False, quiet=True)
        sample = DatasetReader(writer.root).load("000000")
        assert all(piece.mask is None for piece in sample.pieces)


class TestFramingDistance:
    """Distance is derived from the lens *and the pose*, under true perspective."""

    TOP_DOWN_45 = {
        "azimuth_rad": math.radians(45.0),
        "elevation_rad": math.radians(90.0),
    }

    @staticmethod
    def framing(focal=50.0, sensor=36.0, **kwargs):
        from chesssight.synth.randomize import framing_distance

        kwargs.setdefault("azimuth_rad", math.radians(45.0))
        kwargs.setdefault("elevation_rad", math.radians(45.0))
        kwargs.setdefault("margin", 1.0)
        return framing_distance(focal, sensor, square_size=1.0, **kwargs)

    @staticmethod
    def project(point, *, distance, azimuth_rad, elevation_rad):
        """Image-plane offsets of ``point``, in units of the distance-1 frustum."""
        from chesssight.synth.randomize import camera_basis

        toward_camera, right, up = camera_basis(azimuth_rad, elevation_rad)
        depth = distance - sum(a * b for a, b in zip(point, toward_camera, strict=True))
        assert depth > 0, "point is behind the camera"
        lateral = sum(a * b for a, b in zip(point, right, strict=True))
        vertical = sum(a * b for a, b in zip(point, up, strict=True))
        return lateral / depth, vertical / depth

    def test_distance_is_proportional_to_focal_length(self):
        # Exact only where the depth-offset term vanishes -- straight down at a flat
        # board, every corner sits at the centre's depth -- so this pins the lens
        # half of the formula without the perspective half confounding it.
        pose = {**self.TOP_DOWN_45, "piece_height": 0.0}
        assert self.framing(85.0, **pose) / self.framing(24.0, **pose) == pytest.approx(
            85.0 / 24.0
        )

    def test_longer_lenses_stand_further_back_at_every_pose(self):
        for elevation_deg in (8.0, 20.0, 45.0, 75.0):
            pose = {
                "azimuth_rad": math.radians(20.0),
                "elevation_rad": math.radians(elevation_deg),
            }
            assert self.framing(85.0, **pose) > self.framing(24.0, **pose)

    def test_margin_scales_the_distance_linearly(self):
        assert self.framing(margin=2.0) == pytest.approx(2.0 * self.framing())

    def test_top_down_at_45_degrees_is_exactly_the_board_diagonal(self):
        # Straight down, every board corner sits at the same depth as the centre, so
        # perspective and the board diagonal coincide. The one pose whose answer is
        # derivable by hand, which pins the projection against a known value.
        focal, sensor = 50.0, 36.0
        distance = self.framing(focal, sensor, piece_height=0.0, **self.TOP_DOWN_45)
        half_fov = math.atan(sensor / (2.0 * focal))
        assert distance * math.tan(half_fov) == pytest.approx(math.hypot(8, 8) / 2)

    def test_an_axis_aligned_board_needs_less_room_than_a_diagonal_one(self):
        # Viewed down a file from above, the board's lateral extent is 8 squares, not
        # 8*sqrt(2). The old diagonal formula stood sqrt(2) too far back for such poses.
        square_on = self.framing(
            azimuth_rad=0.0, elevation_rad=math.radians(90.0), piece_height=0.0
        )
        diagonal = self.framing(piece_height=0.0, **self.TOP_DOWN_45)
        assert square_on == pytest.approx(diagonal / math.sqrt(2.0))

    def test_grazing_views_stand_back_for_the_near_corner(self):
        # The bug this formula exists to fix. A weak-perspective estimate measures the
        # board at the centre's depth and concludes a grazing view can come closer,
        # because foreshortening has collapsed the board's depth. It cannot: the near
        # corner is much closer to the camera than the centre and subtends a far
        # larger angle. Rendered without this correction, the board spanned up to 2.9x
        # the frame width at 8 degrees with no corner in shot.
        grazing = self.framing(azimuth_rad=0.0, elevation_rad=math.radians(8.0))
        top_down = self.framing(azimuth_rad=0.0, elevation_rad=math.radians(90.0))
        assert grazing > top_down

    def test_piece_height_is_accounted_for(self):
        # A board diagonal ignores the king standing on it, and so cropped its top.
        pose = {"azimuth_rad": math.radians(45.0), "elevation_rad": math.radians(75.0)}
        assert self.framing(piece_height=1.55, **pose) > self.framing(
            piece_height=0.0, **pose
        )

    def test_a_portrait_frame_stands_further_back_than_a_landscape_one(self):
        # The board is width-limited here, and a portrait frame is narrower.
        pose = {"azimuth_rad": 0.0, "elevation_rad": math.radians(20.0)}
        assert self.framing(aspect=9 / 16, **pose) > self.framing(aspect=16 / 9, **pose)

    def test_aspect_is_symmetric_about_square(self):
        # Straight down at a square-on board the lateral and vertical extents are both
        # 8 squares, so rotating the frame cannot change how far back the camera goes.
        # If the sensor-fit branch mixed up the long and short axes, one of these two
        # would use the wide field of view for the narrow image dimension.
        pose = {
            "azimuth_rad": 0.0,
            "elevation_rad": math.radians(90.0),
            "piece_height": 0.0,
        }
        wide = self.framing(aspect=16 / 9, **pose)
        tall = self.framing(aspect=9 / 16, **pose)
        assert tall == pytest.approx(wide)
        # ...and both stand further back than a square frame, whose short axis
        # is the longer of the two.
        assert wide > self.framing(aspect=1.0, **pose)

    @pytest.mark.parametrize("aspect", [1.0, 16 / 9, 4 / 3, 9 / 16])
    def test_every_pose_in_range_keeps_the_whole_box_in_frame(self, aspect: float):
        """The property that matters, checked by projecting rather than by formula.

        At margin 1.0 no corner of the board-plus-king box may fall outside the frame
        for any pose the config can sample, and at least one corner must touch an edge
        -- otherwise the camera is further back than it needs to be and the board is
        smaller in frame than it should be.
        """
        from chesssight.synth import randomize
        from chesssight.synth.config import CameraConfig

        camera = CameraConfig()
        focal, sensor = 50.0, 36.0
        long_tan = sensor / (2.0 * focal)
        short_tan = long_tan * min(aspect, 1.0 / aspect)
        tan_right, tan_up = (
            (long_tan, short_tan) if aspect >= 1.0 else (short_tan, long_tan)
        )

        for azimuth_deg in range(0, 360, 11):
            for elevation_deg in (
                camera.elevation_deg.min,
                20.0,
                45.0,
                camera.elevation_deg.max,
            ):
                pose = {
                    "azimuth_rad": math.radians(azimuth_deg),
                    "elevation_rad": math.radians(elevation_deg),
                }
                distance = randomize.framing_distance(
                    focal, sensor, square_size=1.0, margin=1.0, aspect=aspect, **pose
                )
                touches = False
                for x in (-4.0, 4.0):
                    for y in (-4.0, 4.0):
                        for z in (0.0, randomize.TALLEST_PIECE):
                            lateral, vertical = self.project(
                                (x, y, z), distance=distance, **pose
                            )
                            assert abs(lateral) <= tan_right * (1 + 1e-9), (
                                f"azimuth {azimuth_deg} elevation {elevation_deg}: "
                                f"corner {(x, y, z)} is off the side of the frame"
                            )
                            assert abs(vertical) <= tan_up * (1 + 1e-9), (
                                f"azimuth {azimuth_deg} elevation {elevation_deg}: "
                                f"corner {(x, y, z)} is off the top of the frame"
                            )
                            touches |= abs(lateral) > tan_right * (1 - 1e-6)
                            touches |= abs(vertical) > tan_up * (1 - 1e-6)
                assert touches, (
                    f"azimuth {azimuth_deg} elevation {elevation_deg}: nothing reaches "
                    "a frame edge, so the camera is needlessly far back"
                )
