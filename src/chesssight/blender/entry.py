"""Blender-side entry point.

Run as::

    blender --background --factory-startup \
            --python src/chesssight/blender/entry.py -- --shard jobs/shard_000.jsonl

``--factory-startup`` matters for reproducibility: without it a user's preferences
and add-ons leak into the render. The cost is that the Cycles compute device has to
be selected on every run, which :mod:`chesssight.blender.render` does anyway.

Jobs are processed in a batch within one Blender process because a cold start costs
1.5-3 seconds -- more than an EEVEE frame. Each job is isolated in a try/except so
one bad position cannot lose the rest of the shard.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback

# Blender runs this file by path, so the project's `src` directory has to be put on
# sys.path before any `chesssight` import. This works only because
# `chesssight/__init__.py` and `chesssight/blender/__init__.py` are import-light --
# pydantic is not available in Blender's Python.
_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chesssight.blender import bl_utils, labels, render, scene  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a shard of chess scenes.")
    parser.add_argument("--shard", required=True, help="Path to a shard .jsonl file")
    parser.add_argument(
        "--limit", type=int, default=0, help="Render at most this many jobs (0 = all)"
    )
    return parser.parse_args(argv)


def script_args() -> list[str]:
    """Everything after the ``--`` separator Blender uses to pass through argv."""
    if "--" not in sys.argv:
        raise SystemExit(
            "expected arguments after `--`, e.g. "
            "blender -b -P entry.py -- --shard jobs/shard_000.jsonl"
        )
    return sys.argv[sys.argv.index("--") + 1 :]


def read_jobs(path: pathlib.Path) -> list[dict]:
    jobs = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def render_job(job: dict) -> dict:
    """Build, render and label one scene. Returns the raw label record."""
    started = time.perf_counter()

    objects = scene.build_scene(job)
    render.configure_render(job["render"])
    device = render.configure_devices(job["render"])

    image_path = pathlib.Path(job["image_path"])
    image_path.parent.mkdir(parents=True, exist_ok=True)
    render.render_to(str(image_path))

    id_pass_path = job.get("id_pass_path")
    if id_pass_path:
        pathlib.Path(id_pass_path).parent.mkdir(parents=True, exist_ok=True)
        render.render_id_pass(str(id_pass_path))

    raw = labels.extract(job, objects)
    raw["render"] = {
        "engine": job["render"]["engine"],
        "samples": job["render"]["samples"],
        "seed": job["seed"],
        "blender_version": render.blender_version(),
        "render_seconds": round(time.perf_counter() - started, 3),
        "device": device,
    }

    labels_path = pathlib.Path(job["labels_path"])
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(raw), encoding="utf-8")
    return raw


def main() -> int:
    args = parse_args(script_args())
    jobs = read_jobs(pathlib.Path(args.shard))
    if args.limit:
        jobs = jobs[: args.limit]

    failures = 0
    for index, job in enumerate(jobs, start=1):
        try:
            raw = render_job(job)
            print(
                f"[chesssight] {index}/{len(jobs)} {job['id']} "
                f"ok in {raw['render']['render_seconds']}s",
                flush=True,
            )
        except BaseException:  # noqa: BLE001 - one bad job must not kill the shard
            failures += 1
            error_path = pathlib.Path(job["labels_path"]).with_suffix(".error.txt")
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(
                f"[chesssight] {index}/{len(jobs)} {job['id']} FAILED "
                f"(see {error_path})",
                file=sys.stderr,
                flush=True,
            )
        finally:
            bl_utils.purge_orphans()

    print(
        f"[chesssight] shard complete: {len(jobs) - failures} ok, {failures} failed",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
