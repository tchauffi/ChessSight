from __future__ import annotations

import json
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
    def test_longer_lenses_stand_further_back(self):
        from chesssight.synth.randomize import framing_distance

        short = framing_distance(24.0, 36.0, square_size=1.0, margin=1.0)
        long_lens = framing_distance(85.0, 36.0, square_size=1.0, margin=1.0)
        assert long_lens > short * 3

    def test_margin_scales_the_distance_linearly(self):
        from chesssight.synth.randomize import framing_distance

        base = framing_distance(50.0, 36.0, square_size=1.0, margin=1.0)
        wide = framing_distance(50.0, 36.0, square_size=1.0, margin=2.0)
        assert wide == pytest.approx(2.0 * base)

    def test_the_board_fits_the_frame_at_margin_one(self):
        # At margin 1.0 the board's diagonal should exactly span the horizontal FOV.
        import math

        from chesssight.synth.randomize import framing_distance

        focal, sensor = 50.0, 36.0
        distance = framing_distance(focal, sensor, square_size=1.0, margin=1.0)
        half_fov = math.atan(sensor / (2.0 * focal))
        assert distance * math.tan(half_fov) == pytest.approx(math.hypot(8, 8) / 2)


def test_find_blender_raises_when_absent(monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    with pytest.raises(runner.RunnerError, match="not found on PATH"):
        runner.find_blender()


def test_find_blender_accepts_an_explicit_path():
    assert runner.find_blender("/opt/blender/blender") == "/opt/blender/blender"


def test_entry_script_exists():
    # The runner launches this by path; a rename would only surface at render time.
    assert runner.ENTRY_SCRIPT.is_file()
