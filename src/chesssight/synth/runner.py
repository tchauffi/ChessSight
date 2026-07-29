"""Plan a run, drive Blender subprocesses, and turn their output into samples."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from chesssight.data.dataset import IMAGES_DIRNAME, DatasetWriter
from chesssight.data.schema import DatasetMeta
from chesssight.synth import asset_spec, jobs
from chesssight.synth.config import GeneratorConfig
from chesssight.synth.jobspec import JobSpec
from chesssight.synth.postprocess import PostprocessError, build_sample, load_raw

ENTRY_SCRIPT = Path(__file__).resolve().parents[1] / "blender" / "entry.py"

#: Cycles workers all contend for the same GPU, and each holds its own copy of the
#: scene in VRAM, so more than a couple is slower and risks running out of memory.
MAX_GPU_WORKERS = 2


class RunnerError(RuntimeError):
    """Raised when a run cannot be started or completed."""


@dataclass
class RunResult:
    planned: int = 0
    rendered: int = 0
    failed: int = 0
    elapsed: float = 0.0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.rendered / self.elapsed if self.elapsed > 0 else 0.0


def find_blender(explicit: str | None = None) -> str:
    """Locate the Blender executable."""
    candidate = explicit or shutil.which("blender")
    if not candidate:
        raise RunnerError(
            "blender not found on PATH; install it or pass --blender /path/to/blender"
        )
    return candidate


def git_commit() -> str | None:
    """Current commit, recorded so a dataset can be traced to the code that made it."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def effective_workers(
    config: GeneratorConfig, requested: int
) -> tuple[int, str | None]:
    """Clamp the worker count, returning it with a warning when it was reduced."""
    if requested < 1:
        raise RunnerError(f"workers must be >= 1, got {requested}")
    if config.render.engine == "CYCLES" and config.render.use_gpu:
        if requested > MAX_GPU_WORKERS:
            return MAX_GPU_WORKERS, (
                f"reduced workers {requested} -> {MAX_GPU_WORKERS}: Cycles workers "
                f"share one GPU, and more of them is slower, not faster"
            )
    return requested, None


def plan(
    config: GeneratorConfig,
    root: Path | None = None,
    *,
    workers: int = 1,
    resume: bool = False,
) -> tuple[DatasetWriter, list[Path], int]:
    """Write ``meta.json`` and the job shards. Renders nothing.

    Separating planning from rendering means the job specs -- which fully describe
    every scene -- can be inspected before spending GPU time on them.
    """
    run_dir = Path(root) if root else config.output.run_dir()
    writer = DatasetWriter(run_dir)
    writer.initialise(
        DatasetMeta(
            name=config.output.run_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            source="synthetic",
            master_seed=config.master_seed,
            git_commit=git_commit(),
            generator_config=config.model_dump(mode="json"),
            asset_attribution=asset_attribution(config),
        )
    )

    skip_ids = writer.existing_ids() if resume else set()
    specs: list[JobSpec] = list(jobs.iter_jobs(config, writer, skip_ids=skip_ids))
    shard_paths = jobs.write_shards(run_dir, jobs.shard(specs, workers))
    return writer, shard_paths, len(specs)


def asset_attribution(config: GeneratorConfig) -> dict[str, object] | None:
    """Provenance of the piece set, recorded into the dataset's ``meta.json``.

    Renders are derivative works of the models in them, so a set's licence follows
    the dataset and anything trained on it. Writing it into the dataset means the
    terms are discoverable from the data itself rather than from whoever ran the
    generator.
    """
    manifest_path = config.pieces.asset_manifest
    if not manifest_path:
        return None
    try:
        manifest = asset_spec.load_manifest(Path(manifest_path))
    except asset_spec.AssetError:
        return {"set": str(manifest_path), "license": "UNREADABLE MANIFEST"}
    record = asset_spec.attribution(manifest)
    record["warnings"] = asset_spec.licence_warnings(manifest)
    return record


def _launch(blender: str, shard: Path, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            blender,
            "--background",
            "--factory-startup",
            "--python",
            str(ENTRY_SCRIPT),
            "--",
            "--shard",
            str(shard),
        ],
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    # Keep the handle alive for the process's lifetime.
    process._chesssight_log = handle  # type: ignore[attr-defined]
    return process


def render_shards(
    shard_paths: list[Path],
    root: Path,
    *,
    blender: str,
    timeout: float | None = None,
    quiet: bool = False,
) -> None:
    """Run every shard in parallel and wait for them all."""
    processes = []
    for index, shard in enumerate(shard_paths):
        log_path = Path(root) / "logs" / f"shard_{index:03d}.log"
        processes.append((index, shard, log_path, _launch(blender, shard, log_path)))

    deadline = time.monotonic() + timeout if timeout else None
    for index, _shard, log_path, process in processes:
        remaining = max(1.0, deadline - time.monotonic()) if deadline else None
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            if not quiet:
                print(
                    f"[chesssight] shard {index} timed out; see {log_path}",
                    file=sys.stderr,
                )
        finally:
            process._chesssight_log.close()  # type: ignore[attr-defined]

        # A non-zero exit means some jobs in the shard failed, which is expected and
        # already recorded per-job; the shard as a whole is not a failure.
        if process.returncode not in (0, 1) and not quiet:
            print(
                f"[chesssight] shard {index} exited {process.returncode}; "
                f"see {log_path}",
                file=sys.stderr,
            )


def collect(
    writer: DatasetWriter,
    config: GeneratorConfig,
    *,
    store_masks: bool = True,
    quiet: bool = False,
) -> RunResult:
    """Turn every raw label file into a validated sample and index it."""
    result = RunResult()
    raw_dir = writer.root / jobs.RAW_LABELS_DIRNAME
    if not raw_dir.is_dir():
        return result

    already = writer.existing_ids()
    suffix = ".jpg" if config.render.image_format == "JPEG" else ".png"

    for raw_path in sorted(raw_dir.glob("*.json")):
        sample_id = raw_path.stem
        result.planned += 1
        if sample_id in already:
            continue
        try:
            raw = load_raw(raw_path)
            sample = build_sample(
                raw,
                image_rel_path=f"{IMAGES_DIRNAME}/{sample_id}{suffix}",
                mask_rel_path=(
                    f"{jobs.ID_PASS_DIRNAME}/{sample_id}.png"
                    if raw.get("id_pass_path")
                    else None
                ),
                split=config.output.split,
                store_masks=store_masks,
            )
        except (PostprocessError, ValueError, KeyError, OSError) as error:
            result.failed += 1
            result.failures.append((sample_id, str(error)))
            writer.record_failure(sample_id, str(error))
            if not quiet:
                print(f"[chesssight] {sample_id} rejected: {error}", file=sys.stderr)
            continue

        writer.add(sample)
        result.rendered += 1

    return result


def run(
    config: GeneratorConfig,
    root: Path | None = None,
    *,
    workers: int = 1,
    resume: bool = False,
    blender: str | None = None,
    timeout: float | None = None,
    store_masks: bool = True,
    quiet: bool = False,
) -> RunResult:
    """Plan, render and collect a complete run."""
    executable = find_blender(blender)
    workers, warning = effective_workers(config, workers)
    if warning and not quiet:
        print(f"[chesssight] {warning}", file=sys.stderr)

    started = time.monotonic()
    writer, shard_paths, planned = plan(config, root, workers=workers, resume=resume)
    if not quiet:
        print(
            f"[chesssight] {planned} samples to render across {len(shard_paths)} "
            f"shard(s) into {writer.root}",
            file=sys.stderr,
        )

    if planned:
        render_shards(
            shard_paths, writer.root, blender=executable, timeout=timeout, quiet=quiet
        )

    result = collect(writer, config, store_masks=store_masks, quiet=quiet)
    result.planned = planned
    result.elapsed = time.monotonic() - started

    if not quiet:
        print(
            f"[chesssight] {result.rendered} ok, {result.failed} failed in "
            f"{result.elapsed:.1f}s ({result.rate:.2f} img/s)",
            file=sys.stderr,
        )
    return result
