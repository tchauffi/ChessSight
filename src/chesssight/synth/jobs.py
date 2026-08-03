"""Building, sharding and persisting the job list for a run."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from chesssight.data.dataset import DatasetWriter
from chesssight.synth.config import GeneratorConfig
from chesssight.synth.jobspec import JobSpec
from chesssight.synth.positions import (
    MixtureSampler,
    PgnPositionSampler,
    PositionSampler,
    RandomPositionSampler,
)
from chesssight.synth.randomize import build_job
from chesssight.synth.seeds import derive_rng, derive_seed

JOBS_DIRNAME = "jobs"
ID_PASS_DIRNAME = "id_pass"
RAW_LABELS_DIRNAME = "raw_labels"


def sample_id_for(index: int) -> str:
    """Zero-padded sample id. Six digits covers a million-image run."""
    return f"{index:06d}"


def build_sampler(config: GeneratorConfig) -> PositionSampler:
    """Assemble the position sampler described by ``config.positions``."""
    positions = config.positions
    components: list[tuple[PositionSampler, float]] = []

    if positions.pgn_paths and positions.weight_pgn > 0:
        components.append(
            (
                PgnPositionSampler(
                    [Path(path).expanduser() for path in positions.pgn_paths],
                    max_games=positions.max_games,
                    plies_per_game=positions.plies_per_game,
                    skip_opening_plies=positions.skip_opening_plies,
                    seed=config.master_seed,
                ),
                positions.weight_pgn,
            )
        )
    if positions.pgn_paths and positions.weight_pgn_opening > 0:
        components.append(
            (
                PgnPositionSampler(
                    [Path(path).expanduser() for path in positions.pgn_paths],
                    max_games=positions.max_games,
                    plies_per_game=positions.plies_per_game,
                    skip_opening_plies=positions.opening_skip_plies,
                    max_plies=positions.opening_max_plies,
                    seed=config.master_seed,
                ),
                positions.weight_pgn_opening,
            )
        )
    if positions.weight_random > 0:
        components.append(
            (
                RandomPositionSampler(
                    min_pieces=positions.random_min_pieces,
                    max_pieces=positions.random_max_pieces,
                ),
                positions.weight_random,
            )
        )
    if not components:
        raise ValueError("no usable position source; check config.positions")
    if len(components) == 1:
        return components[0][0]
    return MixtureSampler(components)


def iter_jobs(
    config: GeneratorConfig,
    writer: DatasetWriter,
    *,
    sampler: PositionSampler | None = None,
    skip_ids: set[str] | None = None,
) -> Iterator[JobSpec]:
    """Yield one job per sample that still needs rendering.

    ``skip_ids`` is what makes a run resumable: pass the ids already in the index and
    the same seeds regenerate exactly the same remaining work.
    """
    sampler = sampler or build_sampler(config)
    skip = skip_ids or set()

    suffix = ".jpg" if config.render.image_format == "JPEG" else ".png"
    id_pass_dir = writer.root / ID_PASS_DIRNAME
    raw_labels_dir = writer.root / RAW_LABELS_DIRNAME

    for index in range(config.count):
        sample_id = sample_id_for(index)
        if sample_id in skip:
            continue
        seed = derive_seed(config.master_seed, "sample", index)
        grid = sampler.sample(derive_rng(seed, "position"))
        yield build_job(
            config,
            sample_id=sample_id,
            grid=grid,
            seed=seed,
            image_path=writer.image_path(sample_id, suffix),
            labels_path=raw_labels_dir / f"{sample_id}.json",
            id_pass_path=(
                id_pass_dir / f"{sample_id}.png"
                if config.render.render_id_pass
                else None
            ),
        )


def shard(jobs: list[JobSpec], shard_count: int) -> list[list[JobSpec]]:
    """Split jobs round-robin across ``shard_count`` shards.

    Round-robin rather than contiguous blocks: render cost correlates with piece
    count, and positions arrive in a correlated order, so contiguous blocks would
    finish at noticeably different times.
    """
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    shards: list[list[JobSpec]] = [[] for _ in range(shard_count)]
    for index, job in enumerate(jobs):
        shards[index % shard_count].append(job)
    return shards


def write_shard(path: Path, jobs: list[JobSpec]) -> Path:
    """Write one shard as JSON Lines and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(job.model_dump_json() + "\n")
    return path


def write_shards(root: Path, shards: list[list[JobSpec]]) -> list[Path]:
    """Write every shard under ``<root>/jobs/`` and return the paths."""
    jobs_dir = Path(root) / JOBS_DIRNAME
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return [
        write_shard(jobs_dir / f"shard_{index:03d}.jsonl", jobs)
        for index, jobs in enumerate(shards)
    ]


def read_shard(path: Path) -> list[JobSpec]:
    """Read a shard file back into job specs."""
    jobs = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                jobs.append(JobSpec.model_validate(json.loads(line)))
    return jobs
